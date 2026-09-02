//! Batch P (ARCHITECTURE.md §12 P item 2): state-dependent `SystemOperator` residual, JVP, and
//! VJP with the GX-A3 chain-rule tangents, so transient/nonlinear multi-equation systems execute
//! through the generic system realization instead of refusing `RUN_SYSTEM_UNSUPPORTED`.
//!
//! Fixture: a hermetic two-field transient nonlinear system whose property tangents cross the
//! fields (`ka = ka(b)` inside the `a` equation, `kb = kb(a)` and a product term `a * b` inside
//! the `b` equation, a state-dependent capacity `ca(a)` on `dt(a)`).
//!
//! Evidence:
//! 1. the residual's centered differences match the JVP at a nonzero (state, rate) point;
//! 2. the JVP and VJP are exact transposes (`1e-12`) at that point, for rate shifts 0 and 2.5;
//! 3. the zero-point linear view and load vector are exactly the stateful actions at zero (the
//!    E6 Stokes/Darcy path is byte-for-byte unchanged);
//! 4. the essential-constraint-eliminated operator is a consistent, transposable Methodus
//!    `DaeOperator`, and Methodus BDF steps advance the transient system through it;
//! 5. `system_constitutive_from_sources` with Scientia property kernels reproduces the
//!    hand-written closures' residual and JVP to roundoff (the exact kernel tangent flows into
//!    the off-diagonal blocks), and a kernel without a tangent is refused typed.

use finitum::{
    BlockLayout, FieldSource, FinitumError, MeshProfile, PointEvaluation, ReducedSystemOperator,
    RegionMap, RegionTagId, SystemConstitutiveInput, SystemEssentialConstraintRequirement,
    SystemOperator, SystemRealizationPlan, TaggedMesh, essential_constraints_from_system, realize,
    system_constitutive_from_sources,
};
use methodus::{
    BdfConfig, BdfOrder, BdfState, EvaluationContext, LinearOperator, NewtonConfig, StepOutcome,
    TransposableOperator, bdf_step, verify_dae_jvp,
};
use quantitas::{Dimension, QuantityKindId, UnitRegistry};
use scientia::scientific::{
    FrameSemantics, OutOfValidityPolicy, PropertyDomain, PropertyEvidence, PropertyInput,
    PropertyLocality, PropertyModel, PropertyOutput, PropertySignature, TensorSymmetry, ValueShape,
};
use scientia::{
    DerivativeContract, DerivativeEvaluation, InputSourceRequirement, OperatorSystem,
    PropertyDefinition, SemanticModel, SymbolId, TensorInputRole, compile_operator_system,
    compile_semantics, lower_property_kernel, parse_expression,
};
use std::collections::BTreeMap;

const COUPLED: &str = r#"
module w7_p.coupled;
model Coupled {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field a: state scalar H1(order=1) on Omega { time_role = differential; };
  field b: state scalar H1(order=1) on Omega { time_role = differential; };
  property ka = diffusivity_a(b);
  property kb = diffusivity_b(a);
  property ca = capacity_a(a);
  source fa: VolumetricSource;
  source fb: VolumetricSource;
  equation ea on Omega { ca * dt(a) - div(ka * grad(a)) = fa; }
  equation eb on Omega { dt(b) - div(kb * grad(b)) + a * b = fb; }
  boundary walls_a on boundary("walls") { dirichlet a = 0; }
  boundary walls_b on boundary("walls") { dirichlet b = 0; }
}
"#;

const IDENTITY_TOLERANCE: f64 = 1.0e-12;
const FA: f64 = 1.0;
const FB: f64 = 0.5;

fn ka(b: f64) -> f64 {
    1.0 + 0.3 * b * b
}
fn d_ka(b: f64, db: f64) -> f64 {
    0.6 * b * db
}
fn kb(a: f64) -> f64 {
    1.0 + 0.5 * a
}
fn d_kb(_a: f64, da: f64) -> f64 {
    0.5 * da
}
fn ca(a: f64) -> f64 {
    1.0 + 0.2 * a * a
}
fn d_ca(a: f64, da: f64) -> f64 {
    0.4 * a * da
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn probe_vector(dimension: usize, seed: f64, scale: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| scale * ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

fn assert_identity(left: f64, right: f64, what: &str) {
    let scale = left.abs().max(right.abs()).max(1.0e-300);
    assert!(
        (left - right).abs() <= IDENTITY_TOLERANCE * scale,
        "{what}: <Au,v>={left}, <u,A^Tv>={right}, relative discrepancy {}",
        (left - right).abs() / scale
    );
}

fn assert_close(actual: &[f64], expected: &[f64], relative: f64, what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= relative * scale,
            "{what} at {index}: {actual} != {expected} (relative {relative})"
        );
    }
}

fn unit_square(subdivisions: usize) -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

struct Compiled {
    model: SemanticModel,
    system: OperatorSystem,
}

fn compile() -> Compiled {
    let compilation = compile_semantics(COUPLED, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(&compilation.semantic, "Coupled", &["ea", "eb"]).unwrap();
    Compiled {
        model: compilation.semantic.models[0].clone(),
        system,
    }
}

fn symbol(model: &SemanticModel, name: &str) -> SymbolId {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name)
        .map(|symbol| symbol.id)
        .unwrap_or_else(|| panic!("model has no symbol {name}"))
}

/// The hand-written closure resolution of every non-basis input, each reading the other
/// field's value by its own active-input identity.
fn closure_constitutive(compiled: &Compiled) -> Vec<SystemConstitutiveInput> {
    let model = &compiled.model;
    let mut constitutive = Vec::new();
    for block in &compiled.system.blocks {
        for integral in &block.factorization.integrals {
            let value_input = |field: &str| {
                let field = symbol(model, field);
                integral
                    .primal
                    .inputs
                    .iter()
                    .find(|input| {
                        input.source == InputSourceRequirement::Basis
                            && input.role == TensorInputRole::Active
                            && input.binding.symbol == field
                            && input.binding.evaluation.derivative == DerivativeEvaluation::Value
                    })
                    .map(|input| input.id)
                    .expect("the property's field has a Value-kind active input here")
            };
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = model.symbols[input.binding.symbol.index()].name.as_str();
                let equation = block.equation.clone();
                let index = integral.integral_index;
                let built = match name {
                    "ka" => {
                        let b = value_input("b");
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "w7_p/ka=1+0.3b^2",
                            move |point: &PointEvaluation| {
                                vec![ka(point.input_values(b).unwrap()[0])]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                vec![d_ka(
                                    point.input_values(b).unwrap()[0],
                                    direction.input_values(b).unwrap()[0],
                                )]
                            },
                        )
                    }
                    "kb" => {
                        let a = value_input("a");
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "w7_p/kb=1+0.5a",
                            move |point: &PointEvaluation| {
                                vec![kb(point.input_values(a).unwrap()[0])]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                vec![d_kb(
                                    point.input_values(a).unwrap()[0],
                                    direction.input_values(a).unwrap()[0],
                                )]
                            },
                        )
                    }
                    "ca" => {
                        let a = value_input("a");
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "w7_p/ca=1+0.2a^2",
                            move |point: &PointEvaluation| {
                                vec![ca(point.input_values(a).unwrap()[0])]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                vec![d_ca(
                                    point.input_values(a).unwrap()[0],
                                    direction.input_values(a).unwrap()[0],
                                )]
                            },
                        )
                    }
                    "fa" | "fb" => {
                        let value = if name == "fa" { FA } else { FB };
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            format!("w7_p/{name}={value}"),
                            move |_: &PointEvaluation| vec![value],
                            |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                        )
                    }
                    other => panic!("unexpected non-basis input {other}"),
                };
                constitutive.push(built.unwrap());
            }
        }
    }
    constitutive
}

fn scalar_input(name: &str) -> PropertyInput {
    PropertyInput {
        name: name.into(),
        quantity_kind: QuantityKindId::new("Dimensionless"),
        dimension: Dimension::DIMENSIONLESS,
        shape: ValueShape::Scalar,
        physical_min: None,
        physical_max: None,
        nominal: None,
    }
}

fn property_definition(
    id: &str,
    input: &str,
    expression: &str,
    differentiability: DerivativeContract,
) -> PropertyDefinition {
    PropertyDefinition {
        signature: PropertySignature {
            id: id.into(),
            inputs: vec![scalar_input(input)],
            output: PropertyOutput {
                quantity_kind: QuantityKindId::new("Dimensionless"),
                dimension: Dimension::DIMENSIONLESS,
                shape: ValueShape::Scalar,
                symmetry: TensorSymmetry::None,
                frame: FrameSemantics::Scalar,
            },
            locality: PropertyLocality::Pointwise,
            differentiability,
        },
        model: PropertyModel::Expression(parse_expression(expression).unwrap()),
        domain: PropertyDomain {
            physical_bounds: vec![],
            validity_bounds: vec![],
            phase_constraints: vec![],
            composition_constraints: vec![],
            assumptions: vec![],
            out_of_validity: OutOfValidityPolicy::Warn,
        },
        evidence: PropertyEvidence {
            sources: vec![],
            dataset_digest: None,
            fit_digest: None,
            uncertainty: None,
            notes: Default::default(),
        },
    }
}

fn kernel_source(id: &str, input: &str, expression: &str) -> FieldSource {
    let definition = property_definition(id, input, expression, DerivativeContract::Symbolic);
    let kernel = lower_property_kernel(&definition, &UnitRegistry::si_bootstrap()).unwrap();
    FieldSource::kernel(kernel).unwrap()
}

/// The GX-A3 resolution: property kernels (with symbolic tangents) for the three properties,
/// constants for the two sources.
fn kernel_sources(compiled: &Compiled) -> Vec<(SymbolId, FieldSource)> {
    let model = &compiled.model;
    vec![
        (
            symbol(model, "ka"),
            kernel_source("diffusivity_a", "b", "1.0 + 0.3 * b * b"),
        ),
        (
            symbol(model, "kb"),
            kernel_source("diffusivity_b", "a", "1.0 + 0.5 * a"),
        ),
        (
            symbol(model, "ca"),
            kernel_source("capacity_a", "a", "1.0 + 0.2 * a * a"),
        ),
        (symbol(model, "fa"), FieldSource::constant([FA])),
        (symbol(model, "fb"), FieldSource::constant([FB])),
    ]
}

fn build(
    compiled: &Compiled,
    tagged: &TaggedMesh,
    constitutive: Vec<SystemConstitutiveInput>,
) -> (SystemOperator, ReducedSystemOperator) {
    let model = &compiled.model;
    let a = symbol(model, "a");
    let b = symbol(model, "b");
    let vertex_count = tagged.mesh.vertices().len();
    let layout = BlockLayout::new([(a, vertex_count, 1), (b, vertex_count, 1)]).unwrap();
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), tagged.mesh.clone(), layout).unwrap();
    let operator = plan.bind_kernels(constitutive, BTreeMap::new()).unwrap();
    let mut requirements = Vec::new();
    let mut region_map = RegionMap::new();
    for block in &compiled.system.blocks {
        for requirement in &block.factorization.essential_constraints {
            region_map.insert(
                requirement.region,
                ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
            );
            requirements.push(SystemEssentialConstraintRequirement {
                field: block.row,
                requirement: requirement.clone(),
                value: FieldSource::constant([0.0]),
            });
        }
    }
    let constraints =
        essential_constraints_from_system(&operator, tagged, &region_map, &requirements).unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    (operator, reduced)
}

#[test]
fn residual_centered_differences_match_the_jvp_at_a_nonzero_state() {
    let compiled = compile();
    let tagged = unit_square(3);
    let (operator, _) = build(&compiled, &tagged, closure_constitutive(&compiled));
    let dimension = operator.dimension();
    let time = 0.3;
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let state_direction = probe_vector(dimension, 1.1, 1.0);
    let rate_direction = probe_vector(dimension, 3.3, 0.8);
    let mut analytic = vec![0.0; dimension];
    operator
        .jacobian_vector_product(
            time,
            &state,
            &rate,
            &state_direction,
            &rate_direction,
            &mut analytic,
        )
        .unwrap();
    assert!(analytic.iter().any(|value| value.abs() > 1.0e-6));

    let step = 1.0e-5;
    let shifted = |sign: f64| {
        let state = state
            .iter()
            .zip(&state_direction)
            .map(|(s, d)| s + sign * step * d)
            .collect::<Vec<_>>();
        let rate = rate
            .iter()
            .zip(&rate_direction)
            .map(|(r, d)| r + sign * step * d)
            .collect::<Vec<_>>();
        let mut residual = vec![0.0; dimension];
        operator
            .residual(time, &state, &rate, &mut residual)
            .unwrap();
        residual
    };
    let plus = shifted(1.0);
    let minus = shifted(-1.0);
    let centered = plus
        .iter()
        .zip(&minus)
        .map(|(p, m)| (p - m) / (2.0 * step))
        .collect::<Vec<_>>();
    assert_close(
        &centered,
        &analytic,
        1.0e-7,
        "system JVP vs centered residual differences",
    );
}

#[test]
fn jvp_and_vjp_are_exact_transposes_at_a_nonzero_state() {
    let compiled = compile();
    let tagged = unit_square(3);
    let (operator, _) = build(&compiled, &tagged, closure_constitutive(&compiled));
    let dimension = operator.dimension();
    let context = EvaluationContext::reproducible();
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let u = probe_vector(dimension, 1.7, 1.0);
    let v = probe_vector(dimension, 5.3, 0.9);
    for rate_shift in [0.0, 2.5] {
        let jacobian = operator.linearize(0.3, &state, &rate, rate_shift).unwrap();
        let mut forward = vec![0.0; dimension];
        jacobian.apply(&context, &u, &mut forward).unwrap();
        let mut backward = vec![0.0; dimension];
        jacobian
            .apply_transpose(&context, &v, &mut backward)
            .unwrap();
        assert_identity(
            dot(&forward, &v),
            dot(&u, &backward),
            &format!("system Jacobian with rate shift {rate_shift}"),
        );
        // The physical (unconstrained) transpose is also reachable directly.
        let mut direct = vec![0.0; dimension];
        operator
            .vector_jacobian_product_shifted(0.3, &state, &rate, &v, rate_shift, &mut direct)
            .unwrap();
        assert_eq!(direct, backward);
    }
}

#[test]
fn zero_point_linear_view_and_load_vector_are_the_stateful_actions_at_zero() {
    let compiled = compile();
    let tagged = unit_square(2);
    let (operator, _) = build(&compiled, &tagged, closure_constitutive(&compiled));
    let dimension = operator.dimension();
    let zero = vec![0.0; dimension];
    let x = probe_vector(dimension, 0.9, 1.0);
    let mut action = vec![0.0; dimension];
    operator.apply_action(&x, &mut action).unwrap();
    let mut jvp = vec![0.0; dimension];
    operator
        .jacobian_vector_product(0.0, &zero, &zero, &x, &zero, &mut jvp)
        .unwrap();
    assert_eq!(action, jvp);
    let load = operator.load_vector().unwrap();
    let mut residual = vec![0.0; dimension];
    operator.residual(0.0, &zero, &zero, &mut residual).unwrap();
    for (load, residual) in load.iter().zip(&residual) {
        assert_eq!(*load, -residual);
    }
    assert!(
        load.iter().any(|value| value.abs() > 1.0e-6),
        "the sources load the system"
    );
    // The zero-point transpose view of the physical operator.
    let context = EvaluationContext::reproducible();
    let v = probe_vector(dimension, 4.4, 1.0);
    let mut transposed = vec![0.0; dimension];
    operator
        .apply_transpose(&context, &v, &mut transposed)
        .unwrap();
    assert_identity(
        dot(&action, &v),
        dot(&x, &transposed),
        "zero-point system transpose",
    );
}

#[test]
fn reduced_operator_is_a_consistent_transposable_dae_operator() {
    let compiled = compile();
    let tagged = unit_square(3);
    let (_, reduced) = build(&compiled, &tagged, closure_constitutive(&compiled));
    let dimension = LinearOperator::rows(&reduced);
    let context = EvaluationContext::reproducible();
    let time = 0.2;
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let state_direction = probe_vector(dimension, 1.1, 1.0);
    let rate_direction = probe_vector(dimension, 3.3, 0.8);

    // Methodus's own DAE consistency probe over the `DaeOperator` implementation.
    let discrepancy = verify_dae_jvp(
        &reduced,
        &context,
        time,
        &state,
        &rate,
        &state_direction,
        &rate_direction,
        1.0e-5,
    )
    .unwrap();
    assert!(discrepancy <= 1.0e-6, "DAE JVP discrepancy {discrepancy}");

    // Constrained rows are their own constraint residual.
    let mut residual = vec![0.0; dimension];
    reduced
        .residual(time, &state, &rate, &mut residual)
        .unwrap();
    let mut constrained = 0;
    for constraint in reduced.constraints().constraints() {
        assert_eq!(residual[constraint.target.0], state[constraint.target.0]);
        constrained += 1;
    }
    assert!(constrained > 0);

    // The eliminated Jacobian is an exact transpose pair, with and without a rate shift.
    let u = probe_vector(dimension, 1.7, 1.0);
    let v = probe_vector(dimension, 5.3, 0.9);
    for rate_shift in [0.0, 3.1] {
        let jacobian = reduced.linearize(time, &state, &rate, rate_shift).unwrap();
        assert!(jacobian.constraints().is_some());
        let mut forward = vec![0.0; dimension];
        jacobian.apply(&context, &u, &mut forward).unwrap();
        let mut backward = vec![0.0; dimension];
        jacobian
            .apply_transpose(&context, &v, &mut backward)
            .unwrap();
        assert_identity(
            dot(&forward, &v),
            dot(&u, &backward),
            &format!("reduced system Jacobian with rate shift {rate_shift}"),
        );
    }
}

#[test]
fn bdf_steps_advance_the_reduced_transient_system() {
    let compiled = compile();
    let tagged = unit_square(3);
    let (operator, reduced) = build(&compiled, &tagged, closure_constitutive(&compiled));
    let dimension = operator.dimension();
    let context = EvaluationContext::reproducible();
    // A Dirichlet-consistent bump in both fields.
    let a_block = operator
        .layout()
        .block(symbol(&compiled.model, "a"))
        .unwrap()
        .offset;
    let b_block = operator
        .layout()
        .block(symbol(&compiled.model, "b"))
        .unwrap()
        .offset;
    let mut initial = vec![0.0; dimension];
    for (vertex, point) in tagged.mesh.vertices().iter().enumerate() {
        let bump =
            (std::f64::consts::PI * point[0]).sin() * (std::f64::consts::PI * point[1]).sin();
        initial[a_block + vertex] = bump;
        initial[b_block + vertex] = 0.5 * bump;
    }
    let mut state = BdfState {
        time: 0.0,
        values: initial.clone(),
        previous_values: None,
        previous_step: None,
        accepted_steps: 0,
    };
    let config = BdfConfig {
        order: BdfOrder::One,
        absolute_tolerance: 1.0e-3,
        relative_tolerance: 1.0e-3,
        minimum_step: 1.0e-8,
        maximum_step: 1.0,
        newton: NewtonConfig::default(),
    };
    for _ in 0..3 {
        match bdf_step(&reduced, &context, &state, 0.05, &config).unwrap() {
            StepOutcome::Accepted(accepted) => state = accepted.state,
            StepOutcome::Rejected(rejected) => {
                panic!(
                    "BDF step rejected with error estimate {}",
                    rejected.error_estimate
                )
            }
        }
    }
    assert_eq!(state.accepted_steps, 3);
    assert!((state.time - 0.15).abs() <= 1.0e-12);
    assert!(state.values.iter().all(|value| value.is_finite()));
    let moved = state
        .values
        .iter()
        .zip(&initial)
        .map(|(after, before)| (after - before).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        moved > 1.0e-3,
        "the transient system evolved (max change {moved})"
    );
    for constraint in reduced.constraints().constraints() {
        assert!(state.values[constraint.target.0].abs() <= 1.0e-12);
    }
}

#[test]
fn kernel_field_sources_carry_exact_property_tangents_into_the_system_path() {
    let compiled = compile();
    let tagged = unit_square(3);
    let (by_closure, _) = build(&compiled, &tagged, closure_constitutive(&compiled));
    let constitutive = system_constitutive_from_sources(
        &compiled.system,
        &compiled.model,
        &kernel_sources(&compiled),
    )
    .unwrap();
    let (by_kernel, _) = build(&compiled, &tagged, constitutive);
    let dimension = by_closure.dimension();
    let time = 0.3;
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let state_direction = probe_vector(dimension, 1.1, 1.0);
    let rate_direction = probe_vector(dimension, 3.3, 0.8);

    let mut residual_closure = vec![0.0; dimension];
    let mut residual_kernel = vec![0.0; dimension];
    by_closure
        .residual(time, &state, &rate, &mut residual_closure)
        .unwrap();
    by_kernel
        .residual(time, &state, &rate, &mut residual_kernel)
        .unwrap();
    assert_close(
        &residual_kernel,
        &residual_closure,
        1.0e-12,
        "kernel-sourced residual",
    );

    let mut jvp_closure = vec![0.0; dimension];
    let mut jvp_kernel = vec![0.0; dimension];
    by_closure
        .jacobian_vector_product(
            time,
            &state,
            &rate,
            &state_direction,
            &rate_direction,
            &mut jvp_closure,
        )
        .unwrap();
    by_kernel
        .jacobian_vector_product(
            time,
            &state,
            &rate,
            &state_direction,
            &rate_direction,
            &mut jvp_kernel,
        )
        .unwrap();
    assert_close(&jvp_kernel, &jvp_closure, 1.0e-12, "kernel-sourced JVP");

    let v = probe_vector(dimension, 5.3, 0.9);
    let mut vjp_closure = vec![0.0; dimension];
    let mut vjp_kernel = vec![0.0; dimension];
    by_closure
        .vector_jacobian_product(time, &state, &rate, &v, &mut vjp_closure)
        .unwrap();
    by_kernel
        .vector_jacobian_product(time, &state, &rate, &v, &mut vjp_kernel)
        .unwrap();
    assert_close(&vjp_kernel, &vjp_closure, 1.0e-12, "kernel-sourced VJP");
}

#[test]
fn a_kernel_without_a_tangent_for_its_state_input_is_refused_typed() {
    let compiled = compile();
    let model = &compiled.model;
    let definition = property_definition(
        "diffusivity_a",
        "b",
        "1.0 + 0.3 * b * b",
        DerivativeContract::None,
    );
    let kernel = lower_property_kernel(&definition, &UnitRegistry::si_bootstrap()).unwrap();
    let mut sources = kernel_sources(&compiled);
    sources[0] = (symbol(model, "ka"), FieldSource::kernel(kernel).unwrap());
    let error = system_constitutive_from_sources(&compiled.system, model, &sources).unwrap_err();
    assert!(
        matches!(error, FinitumError::RealizationTangentUnavailable(_)),
        "expected a typed tangent refusal, got {error:?}"
    );
}
