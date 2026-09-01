//! SV2-B4 continuation (E6): executable Scientia-form/Malleus-kernel-driven mixed realization,
//! decisive acceptance.
//!
//! Drives the real `25-stokes.res` corpus (Taylor-Hood-labeled P2 velocity / L2(order=1)
//! pressure, realized here through a continuous P1 Lagrange pressure basis -- see
//! `finitum::system`'s `build_field_elements` doc comment for why an L2-typed field is
//! admitted) through `compile_semantics` -> `compile_operator_system` ->
//! `derive_operator_structure_for_system` -> `SystemRealizationPlan::bind_kernels` (real bound
//! Malleus kernels) -> `methodus::solve_minres`, and cross-checks the realized action against an
//! independently hand-composed `finitum::mixed::MixedOperator`.
//!
//! A real, load-bearing finding this file documents and works around explicitly: the corpus's
//! own equation authorship (`momentum` uses `+grad(pressure)`, which after integration by parts
//! becomes `-integral(p * div(v))`; `incompressibility` uses the un-negated `div(velocity) = 0`,
//! which weakly is `+integral(q * div(u))`) gives a saddle-point coupling that is genuinely NOT
//! symmetric relative to a single shared sign -- confirmed empirically below (the momentum block
//! and the incompressibility block are exact negatives of one another relative to the same
//! underlying divergence integral) and consistent with Scientia's own structural
//! `OperatorStructure::form_symmetry` correctly reporting `Unknown` (never `Symmetric`) for the
//! unsigned system. `SystemRealizationPlan::bind_kernels`'s `equation_sign` parameter exists
//! for exactly this: multiplying the `incompressibility` equation row by `-1` is solution-
//! preserving (its right-hand side is zero) and restores genuine symmetry, verified here by
//! assembly (`SystemOperator::prove_symmetry`), not assumed.

use finitum::{
    BlockCoupling, BlockLayout, Cell, ConstraintSet, CouplingKind, FieldSource, FieldSpec,
    FinitumError, Mesh, MeshProfile, MixedOperator, MixedSpace, PointEvaluation, RegionMap,
    RegionTagId, SystemConstitutiveInput, SystemEssentialConstraintRequirement,
    SystemRealizationPlan, VertexId, essential_constraints_from_system, quadratic_simplex_dof_map,
    realize, vector_nodal_dof_map,
};
use methodus::{
    EvaluationContext, LinearOperator, MinresConfig, NullspaceProjector, OperatorSymmetry,
    solve_minres,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, RegionId, SymbolId,
    compile_operator_system, compile_semantics,
};
use std::collections::BTreeMap;
use std::fs;

const STOKES_CORPUS: &str = "/projects/sinbad/sinbad/physics/corpus/25-stokes.res";
const DARCY_CORPUS: &str = "/projects/sinbad/sinbad/physics/corpus/13-mixed-darcy.res";

/// Viscosity constant this test supplies through a [`SystemConstitutiveInput`] closure -- the
/// corpus declares `mu` as a provider-resolved property (`dynamic_viscosity(0)`), and Scientia
/// never resolves provider *values* itself (they are a realization-side concern, matching how
/// `tests/sv2_elasticity.rs` supplies its own Lame constants through the same
/// `DynamicExternalInput` mechanism `SystemConstitutiveInput` mirrors).
const MU: f64 = 1.7;

fn stress(strain: &[f64]) -> Vec<f64> {
    strain.iter().map(|value| 2.0 * MU * value).collect()
}

struct CompiledStokes {
    system: OperatorSystem,
    velocity: SymbolId,
    pressure: SymbolId,
}

fn compile_stokes() -> CompiledStokes {
    let source = fs::read_to_string(STOKES_CORPUS).expect("25-stokes.res corpus is readable");
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "StokesFlow",
        &["momentum", "incompressibility"],
    )
    .unwrap();
    let velocity = system
        .blocks
        .iter()
        .find(|block| block.equation == "momentum")
        .expect("momentum block")
        .row;
    let pressure = system
        .blocks
        .iter()
        .find(|block| block.equation == "incompressibility")
        .expect("incompressibility block")
        .row;
    CompiledStokes {
        system,
        velocity,
        pressure,
    }
}

fn unit_square(subdivisions: usize) -> finitum::TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

fn taylor_hood_layout(mesh: &Mesh, velocity: SymbolId, pressure: SymbolId) -> BlockLayout {
    let velocity_nodes = quadratic_simplex_dof_map(mesh, 2).unwrap().dof_count() / 2;
    let pressure_nodes = mesh.vertices().len();
    BlockLayout::new([(velocity, velocity_nodes, 2), (pressure, pressure_nodes, 1)]).unwrap()
}

/// Every non-`Basis` primal input across the compiled system, resolved generically by shape (no
/// physics-name dispatch): a 4-component (`[2,2]`) tensor input is the momentum block's
/// `ModelDefinedConstitutive` viscous stress (`2 * mu * sym_grad(velocity)`, linear, so its
/// value/direction share the same formula); everything else (the `ExternalValue` body-force
/// input) is bound to an always-zero closure, since this file builds the pure bilinear operator
/// action (no forcing term) for a MINRES demonstration against a synthetically constructed,
/// self-consistent right-hand side -- mirroring `tests/sv2b_mixed.rs`'s own MINRES fixture,
/// which likewise builds its right-hand side from a known solution rather than a physical load.
fn stokes_constitutive(system: &OperatorSystem) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let components = input.shape.iter().product::<usize>().max(1);
                let equation = block.equation.clone();
                let integral_index = integral.integral_index;
                let input_id = input.id;
                let binding = if components == 4 {
                    SystemConstitutiveInput::new(
                        equation,
                        integral_index,
                        input_id,
                        components,
                        "sv2b4-stokes/viscosity",
                        |evaluation: &PointEvaluation| {
                            stress(
                                evaluation
                                    .values(DerivativeEvaluation::SymmetricGradient)
                                    .expect("active symmetric-gradient input"),
                            )
                        },
                        |_evaluation: &PointEvaluation, direction: &PointEvaluation| {
                            stress(
                                direction
                                    .values(DerivativeEvaluation::SymmetricGradient)
                                    .expect("active symmetric-gradient direction"),
                            )
                        },
                    )
                } else {
                    SystemConstitutiveInput::new(
                        equation,
                        integral_index,
                        input_id,
                        components,
                        "sv2b4-stokes/no-forcing",
                        move |_evaluation: &PointEvaluation| vec![0.0; components],
                        move |_evaluation: &PointEvaluation, _direction: &PointEvaluation| {
                            vec![0.0; components]
                        },
                    )
                };
                constitutive.push(binding.unwrap());
            }
        }
    }
    constitutive
}

fn walls_region_map(region: RegionId) -> RegionMap {
    let mut map = RegionMap::new();
    map.insert(
        region,
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("x_max"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
        ],
    );
    map
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: {actual} != {expected} within {tolerance}"
        );
    }
}

fn pseudo_random_vector(dimension: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..dimension)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * bits - 1.0
        })
        .collect()
}

/// The corpus's own equation authorship gives a genuinely non-symmetric saddle-point coupling
/// (see this file's module doc comment): Scientia's structural `OperatorStructure` correctly
/// reports `form_symmetry: Unknown` for it, `SystemOperator::symmetry()` threads that claim
/// through unmodified (mission item 8), and `methodus::solve_minres` -- which requires a
/// declared `Symmetric` operator -- correctly refuses it, rather than either crate silently
/// accepting a system that has not been shown to satisfy MINRES's precondition.
#[test]
fn unsigned_stokes_operator_reports_unknown_symmetry_and_minres_refuses_it() {
    let compiled = compile_stokes();
    let mesh = unit_square(2);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let constitutive = stokes_constitutive(&compiled.system);
    let operator = plan.bind_kernels(constitutive, BTreeMap::new()).unwrap();

    let structure = operator.structure();
    assert!(structure.saddle_point);
    assert_eq!(structure.nullspace_candidates.len(), 1);
    assert_eq!(structure.nullspace_candidates[0].field, compiled.pressure);
    assert_eq!(operator.nullspace_candidates().len(), 1);
    assert_eq!(operator.symmetry(), OperatorSymmetry::Unknown);

    let dimension = operator.dimension();
    let constraints = ConstraintSet::new(dimension, Vec::new()).unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    assert_eq!(reduced.symmetry(), OperatorSymmetry::Unknown);

    let right_hand_side = vec![0.0; dimension];
    let config = MinresConfig::default();
    let error = solve_minres(
        &reduced,
        None,
        None,
        &EvaluationContext::reproducible(),
        &right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("Symmetric") || message.contains("symmetric"),
        "expected a symmetry-related refusal, got: {message}"
    );
}

/// Decisive acceptance: the `equation_sign`-corrected system operator is genuinely symmetric by
/// assembly, agrees entrywise with an independently hand-composed `MixedOperator`
/// (`SymmetricGradientGradient` + `DivergenceValue`) on both the unconstrained and the
/// Dirichlet-reduced action, and drives `methodus::solve_minres` with the auto-derived pressure
/// nullspace projector to convergence on a consistent right-hand side.
#[test]
fn signed_stokes_system_matches_mixed_operator_and_minres_converges() {
    let compiled = compile_stokes();
    let mesh = unit_square(2);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let constitutive = stokes_constitutive(&compiled.system);
    let equation_sign = BTreeMap::from([("incompressibility".to_string(), -1.0)]);
    let operator = plan.bind_kernels(constitutive, equation_sign).unwrap();
    let dimension = operator.dimension();

    // Symmetry is not assumed: it is proven by assembly, and the proof is cached.
    assert_eq!(operator.symmetry(), OperatorSymmetry::Unknown);
    assert_eq!(
        operator.prove_symmetry(1.0e-9).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert_eq!(operator.symmetry(), OperatorSymmetry::Symmetric);

    // Cross-check: an independently hand-composed MixedOperator over the SAME mesh/fields,
    // built from generic structural couplings (SymmetricGradientGradient for the viscosity
    // block, DivergenceValue at scale -1.0 to match the pressure coupling this file's module
    // doc comment derives empirically), agreeing entrywise on several probe vectors.
    let mixed_space = MixedSpace::new(
        mesh.mesh.clone(),
        vec![
            FieldSpec {
                symbol: compiled.velocity,
                order: 2,
                components: 2,
            },
            FieldSpec {
                symbol: compiled.pressure,
                order: 1,
                components: 1,
            },
        ],
    )
    .unwrap();
    let mixed_operator = MixedOperator::new(
        mixed_space,
        vec![
            BlockCoupling {
                test: compiled.velocity,
                trial: compiled.velocity,
                kind: CouplingKind::SymmetricGradientGradient,
                scale: 2.0 * MU,
            },
            BlockCoupling {
                test: compiled.velocity,
                trial: compiled.pressure,
                kind: CouplingKind::DivergenceValue,
                scale: -1.0,
            },
        ],
    )
    .unwrap();
    assert_eq!(mixed_operator.dimension(), dimension);
    for seed in 0..6u64 {
        let probe = pseudo_random_vector(dimension, seed);
        let mut system_output = vec![0.0; dimension];
        operator.apply_action(&probe, &mut system_output).unwrap();
        let mut mixed_output = vec![0.0; dimension];
        mixed_operator
            .apply_action(&probe, &mut mixed_output)
            .unwrap();
        assert_close(&system_output, &mixed_output, 5.0e-11);
    }

    // Region-tag-driven multi-field essential constraints (mission item 3): zero velocity on
    // every unit-square wall.
    let momentum_requirement = compiled
        .system
        .blocks
        .iter()
        .find(|block| block.equation == "momentum")
        .unwrap()
        .factorization
        .essential_constraints
        .first()
        .expect("momentum declares one essential-constraint requirement (the walls boundary)")
        .clone();
    let region_map = walls_region_map(momentum_requirement.region);
    let constraints = essential_constraints_from_system(
        &operator,
        &mesh,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: compiled.velocity,
            requirement: momentum_requirement,
            value: FieldSource::constant(vec![0.0, 0.0]),
        }],
    )
    .unwrap();
    assert!(constraints.constraints().next().is_some());

    let mixed_reduced = mixed_operator.reduced(constraints.clone()).unwrap();
    let system_reduced = operator.reduced(constraints.clone()).unwrap();
    assert_eq!(system_reduced.symmetry(), OperatorSymmetry::Symmetric);
    for seed in 10..16u64 {
        let probe = pseudo_random_vector(dimension, seed);
        let mut system_output = vec![0.0; dimension];
        system_reduced
            .apply(&EvaluationContext::default(), &probe, &mut system_output)
            .unwrap();
        let mut mixed_output = vec![0.0; dimension];
        mixed_reduced
            .apply(&EvaluationContext::default(), &probe, &mut mixed_output)
            .unwrap();
        assert_close(&system_output, &mixed_output, 5.0e-11);
    }

    // Auto-derived nullspace candidate (mission item 9), resolved and verified in the kernel of
    // the reduced operator.
    let candidates = operator.nullspace_candidates();
    assert_eq!(candidates.len(), 1);
    let mode = candidates[0].resolve(operator.layout()).unwrap();
    assert!(
        mode.verify_in_kernel(&system_reduced, 1.0e-8).unwrap(),
        "the auto-derived constant pressure mode should verify against the reduced operator"
    );

    // Known solution respecting the homogeneous Dirichlet data and orthogonal to the declared
    // constant-pressure nullspace (zero-mean pressure), so the generated right-hand side is
    // exactly consistent (mirrors tests/sv2b_mixed.rs's own MINRES fixture).
    let velocity_block = operator.layout().block(compiled.velocity).unwrap();
    let pressure_block = operator.layout().block(compiled.pressure).unwrap();
    let mut x_true = pseudo_random_vector(dimension, 4242);
    for constraint in constraints.constraints() {
        x_true[constraint.target.0] = 0.0;
    }
    let pressure_mean = x_true
        [pressure_block.offset..pressure_block.offset + pressure_block.extent]
        .iter()
        .sum::<f64>()
        / pressure_block.extent as f64;
    for value in &mut x_true[pressure_block.offset..pressure_block.offset + pressure_block.extent] {
        *value -= pressure_mean;
    }
    let _ = velocity_block;

    let mut right_hand_side = vec![0.0; dimension];
    system_reduced
        .apply(&EvaluationContext::default(), &x_true, &mut right_hand_side)
        .unwrap();

    let config = MinresConfig {
        max_iterations: 4 * dimension,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-10,
    };
    let report = solve_minres(
        &system_reduced,
        None,
        Some(mode.projector() as &dyn NullspaceProjector),
        &EvaluationContext::reproducible(),
        &right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap();
    assert!(
        report.converged,
        "minres did not converge on the reduced Stokes system"
    );
    assert_close(&report.solution, &x_true, 1.0e-6);
    let mut recovered = vec![0.0; dimension];
    system_reduced
        .apply(
            &EvaluationContext::default(),
            &report.solution,
            &mut recovered,
        )
        .unwrap();
    assert_close(&recovered, &right_hand_side, 1.0e-6);
}

/// `13-mixed-darcy.res` pairs an `HDiv(order=0)` flux field with an `L2(order=0)` pressure field
/// (`@inf_sup(pair = "RT0-P0")`) -- a genuine compatible-element (Hdiv) discretization, item 1's
/// out-of-scope DOF map, and an order (`0`) this crate's Lagrange machinery does not admit
/// either way. `SystemRealizationPlan::bind_kernels` refuses it typed rather than silently
/// treating it as Lagrange.
#[test]
fn mixed_darcy_hdiv_pairing_is_refused_typed_not_faked_as_lagrange() {
    let source = fs::read_to_string(DARCY_CORPUS).expect("13-mixed-darcy.res corpus is readable");
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "MixedDarcy",
        &["darcy_law", "mass_balance"],
    )
    .unwrap();
    let flux = system
        .blocks
        .iter()
        .find(|block| block.equation == "darcy_law")
        .unwrap()
        .row;
    let pressure = system
        .blocks
        .iter()
        .find(|block| block.equation == "mass_balance")
        .unwrap()
        .row;

    let mesh = Mesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        }],
    )
    .unwrap();
    // Entity/component counts only need to satisfy `validate_components`'s shape check (this
    // realization never reaches `build_field_elements`'s DOF-map construction); the vertex-major
    // node map used here is not claimed to be a real RT0/P0 DOF map.
    let layout = BlockLayout::new([
        (
            flux,
            vector_nodal_dof_map(&mesh, 3).unwrap().dof_count() / 3,
            3,
        ),
        (pressure, mesh.vertices().len(), 1),
    ])
    .unwrap();
    let plan = SystemRealizationPlan::new(system, mesh.clone(), layout).unwrap();

    // `darcy_law` also carries a Neumann (`ExteriorFacet`) boundary integral, so the full
    // two-equation system is refused by the facet-measure check (item 5) before ever reaching
    // the element-family check; either refusal is a genuine, typed "not this lane's scope"
    // rather than a fake Lagrange realization, so only the error *kind* is asserted here.
    let error = plan.bind_kernels(Vec::new(), BTreeMap::new()).unwrap_err();
    assert!(
        matches!(error, FinitumError::UnsupportedRealization(_)),
        "expected a typed UnsupportedRealization refusal, got: {error:?}"
    );

    // Isolate the Hdiv/L2(order=0) element-family refusal specifically: `mass_balance` alone has
    // no facet integral (`impermeable` is attached only to `darcy_law`), so this reaches
    // `build_field_elements`'s admitted-family check.
    let source = fs::read_to_string(DARCY_CORPUS).unwrap();
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let mass_balance_system =
        compile_operator_system(&compilation.semantic, "MixedDarcy", &["mass_balance"]).unwrap();
    let mass_balance_layout = BlockLayout::new([
        (
            flux,
            vector_nodal_dof_map(&mesh, 3).unwrap().dof_count() / 3,
            3,
        ),
        (pressure, mesh.vertices().len(), 1),
    ])
    .unwrap();
    let mass_balance_plan =
        SystemRealizationPlan::new(mass_balance_system, mesh, mass_balance_layout).unwrap();
    let element_error = mass_balance_plan
        .bind_kernels(Vec::new(), BTreeMap::new())
        .unwrap_err();
    assert!(
        matches!(element_error, FinitumError::UnsupportedRealization(_)),
        "expected a typed UnsupportedRealization refusal, got: {element_error:?}"
    );
    let element_message = element_error.to_string();
    assert!(
        element_message.contains("H1") || element_message.contains("L2"),
        "expected the refusal to name the admitted Lagrange families, got: {element_message}"
    );
}
