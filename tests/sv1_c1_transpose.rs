//! SV1-C1/C3 (E7): the global transpose as a Methodus operator pair, and distributed-coefficient
//! derivative products, over the executed Malleus VJP kernels (GX-F7).
//!
//! Acceptance evidence (hermetic, no corpus):
//! 1. `RealizationPlan::linearize` yields a `LinearOperator + TransposableOperator` whose
//!    adjoint identity `<J u, v> == <u, J^T v>` holds to `1e-12` relative on a Poisson fixture
//!    (nodal conductivity) and on a transient nonlinear fixture at a nonzero linearization
//!    state, including the rate-shifted Jacobian `dR/du + alpha dR/du_t` an implicit step
//!    solves with;
//! 2. `MatrixFreeOperator` and `AssembledOperator` implement `TransposableOperator` and agree
//!    with each other and with the materialized CSR transpose;
//! 3. the distributed-coefficient JVP and VJP are exact transposes of one another for the
//!    per-vertex, per-cell, and per-quadrature-point layouts, and the JVP matches centered
//!    differences of rebuilt realizations;
//! 4. an adjoint objective gradient `dJ/dk = -lambda^T dR/dk` (with `J^T lambda = dJ/du` solved
//!    through the explicit transpose) matches centered differences of the fully rebuilt solve
//!    with tightening error across two step sizes.

use finitum::{
    CoefficientLayout, DerivativeProduct, DistributedCoefficient, DynamicExternalInput,
    ExternalInput, FieldSource, MeshProfile, PreparedElement, RealizationPlan, RegionMap,
    RegionTagId, TaggedMesh, essential_constraints_from, realize, vector_nodal_dof_map,
};
use methodus::{
    ConjugateGradientConfig, EvaluationContext, GmresConfig, LinearOperator, TransposableOperator,
    TransposeOperator, solve_conjugate_gradient, solve_gmres, verify_adjoint_identity,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};

const POISSON: &str = r#"
module sv1_c1.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

const TRANSIENT_NONLINEAR: &str = r#"
module sv1_c1.transient_nonlinear;
model TransientNonlinear {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  property capacity = storage_capacity(u);
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation evolution on Omega { capacity * dt(u) - div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(t); }
}
"#;

/// The adjoint identity tolerance this file asserts, relative to the larger inner product.
const IDENTITY_TOLERANCE: f64 = 1.0e-12;
const SOURCE: f64 = 0.4;

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

fn unit_square(subdivisions: usize) -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

fn walls_region_map(region: scientia::RegionId) -> RegionMap {
    let mut map = RegionMap::new();
    map.insert(
        region,
        ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
    );
    map
}

/// A smooth, positive nodal conductivity design field: `k_i = 1 + 0.5 x_i + 0.25 y_i^2`.
fn nodal_design(mesh: &TaggedMesh) -> Vec<f64> {
    mesh.mesh
        .vertices()
        .iter()
        .map(|vertex| 1.0 + 0.5 * vertex[0] + 0.25 * vertex[1] * vertex[1])
        .collect()
}

/// The Poisson realization with `k` bound as a distributed coefficient of the given layout
/// (design vector `design`) and `f` stored constant.
fn poisson_plan(
    tagged: &TaggedMesh,
    layout: CoefficientLayout,
    design: &[f64],
) -> (RealizationPlan, DistributedCoefficient) {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let mesh = tagged.mesh.clone();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let dofs = vector_nodal_dof_map(&mesh, 1).unwrap();
    let region_map = walls_region_map(factorization.essential_constraints[0].region);
    let constraints = essential_constraints_from(
        tagged,
        &dofs,
        &factorization.essential_constraints,
        &region_map,
        &[FieldSource::constant([0.0])],
    )
    .unwrap();
    let model = &compilation.semantic.models[0];
    let mut external = Vec::new();
    let mut coefficient = None;
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.as_str();
            match name {
                "k" => {
                    coefficient = Some(DistributedCoefficient {
                        integral_index: integral.integral_index,
                        input: input.id,
                        layout,
                    });
                    external.push(
                        ExternalInput::from_coefficient(
                            integral.integral_index,
                            input.id,
                            1,
                            &mesh,
                            &element,
                            layout,
                            design,
                        )
                        .unwrap(),
                    );
                }
                "f" => external.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        |_, _| vec![SOURCE],
                    )
                    .unwrap(),
                ),
                other => panic!("unexpected external input {other}"),
            }
        }
    }
    let plan = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        external,
    )
    .unwrap();
    (plan, coefficient.expect("the Poisson form binds k"))
}

/// The transient nonlinear realization: `capacity = 1 + 0.3 u^2`, `k = 1 + 0.2 u`, both
/// dynamic external inputs with exact directional derivatives, `f` stored zero.
fn nonlinear_plan(tagged: &TaggedMesh) -> RealizationPlan {
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let mesh = tagged.mesh.clone();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let dofs = vector_nodal_dof_map(&mesh, 1).unwrap();
    let region_map = walls_region_map(factorization.essential_constraints[0].region);
    let constraints = essential_constraints_from(
        tagged,
        &dofs,
        &factorization.essential_constraints,
        &region_map,
        &[FieldSource::constant([0.0])],
    )
    .unwrap();
    let model = &compilation.semantic.models[0];
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.as_str();
            match name {
                "capacity" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        "capacity=1+0.3u^2;direction=0.6u*du/v1",
                        |evaluation| {
                            let u = evaluation.values(DerivativeEvaluation::Value).unwrap()[0];
                            vec![1.0 + 0.3 * u * u]
                        },
                        |evaluation, direction| {
                            let u = evaluation.values(DerivativeEvaluation::Value).unwrap()[0];
                            let du = direction.values(DerivativeEvaluation::Value).unwrap()[0];
                            vec![0.6 * u * du]
                        },
                    )
                    .unwrap(),
                ),
                "k" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        "k=1+0.2u;direction=0.2du/v1",
                        |evaluation| {
                            vec![
                                1.0 + 0.2
                                    * evaluation.values(DerivativeEvaluation::Value).unwrap()[0],
                            ]
                        },
                        |_, direction| {
                            vec![0.2 * direction.values(DerivativeEvaluation::Value).unwrap()[0]]
                        },
                    )
                    .unwrap(),
                ),
                "f" => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        |_, _| vec![0.0],
                    )
                    .unwrap(),
                ),
                other => panic!("unexpected external input {other}"),
            }
        }
    }
    RealizationPlan::new_stateful(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        stored,
        dynamic,
    )
    .unwrap()
}

#[test]
fn capability_reports_coefficient_products_for_a_stored_cell_coefficient() {
    let tagged = unit_square(3);
    let design = nodal_design(&tagged);
    let (plan, _) = poisson_plan(&tagged, CoefficientLayout::Vertex, &design);
    let products = plan.capability().derivative_products;
    assert!(products.contains(&DerivativeProduct::Vjp));
    assert!(products.contains(&DerivativeProduct::CoefficientJvp));
    assert!(products.contains(&DerivativeProduct::CoefficientVjp));
}

#[test]
fn poisson_linearized_jacobian_satisfies_the_adjoint_identity_to_1e12() {
    let tagged = unit_square(4);
    let design = nodal_design(&tagged);
    let (plan, _) = poisson_plan(&tagged, CoefficientLayout::Vertex, &design);
    let dimension = plan.dimension();
    let state = probe_vector(dimension, 0.3, 0.7);
    let rate = vec![0.0; dimension];
    let jacobian = plan.linearize(0.0, &state, &rate, 0.0).unwrap();
    let context = EvaluationContext::reproducible();

    let u = probe_vector(dimension, 1.0, 1.0);
    let v = probe_vector(dimension, 2.5, 0.8);
    let mut forward = vec![0.0; dimension];
    jacobian.apply(&context, &u, &mut forward).unwrap();
    let mut backward = vec![0.0; dimension];
    jacobian
        .apply_transpose(&context, &v, &mut backward)
        .unwrap();
    assert_identity(
        dot(&forward, &v),
        dot(&u, &backward),
        "Poisson linearized Jacobian",
    );

    // The same identity through Methodus's explicit transpose view, the shape an adjoint
    // solve consumes (SV1-C5/D1).
    let transpose = TransposeOperator::explicit(&jacobian);
    let scale = dot(&forward, &v).abs().max(1.0);
    let discrepancy = verify_adjoint_identity(
        &jacobian,
        &transpose,
        &context,
        &u,
        &v,
        IDENTITY_TOLERANCE * scale,
    )
    .unwrap();
    assert!(discrepancy <= IDENTITY_TOLERANCE * scale);
}

#[test]
fn matrix_free_and_assembled_transposes_agree_with_the_materialized_csr_transpose() {
    let tagged = unit_square(4);
    let design = nodal_design(&tagged);
    let (plan, _) = poisson_plan(&tagged, CoefficientLayout::Vertex, &design);
    let dimension = plan.dimension();
    let context = EvaluationContext::reproducible();
    let matrix_free = plan.matrix_free();
    let assembled = plan.assemble().unwrap();
    let materialized = assembled.transpose().unwrap();
    let v = probe_vector(dimension, 4.2, 1.3);

    let mut from_matrix_free = vec![0.0; dimension];
    matrix_free
        .apply_transpose(&context, &v, &mut from_matrix_free)
        .unwrap();
    let mut from_assembled = vec![0.0; dimension];
    assembled
        .apply_transpose(&context, &v, &mut from_assembled)
        .unwrap();
    let mut from_materialized = vec![0.0; dimension];
    materialized
        .apply(&context, &v, &mut from_materialized)
        .unwrap();
    for index in 0..dimension {
        let scale = from_materialized[index].abs().max(1.0);
        assert!(
            (from_matrix_free[index] - from_materialized[index]).abs() <= 1.0e-12 * scale,
            "matrix-free transpose differs at {index}"
        );
        assert!(
            (from_assembled[index] - from_materialized[index]).abs() <= 1.0e-12 * scale,
            "assembled transpose differs at {index}"
        );
    }

    // And the matrix-free transpose is the exact transpose of the matrix-free forward action.
    let u = probe_vector(dimension, 0.9, 0.5);
    let mut forward = vec![0.0; dimension];
    matrix_free.apply(&context, &u, &mut forward).unwrap();
    assert_identity(
        dot(&forward, &v),
        dot(&u, &from_matrix_free),
        "matrix-free transpose",
    );
}

#[test]
fn nonlinear_shifted_jacobian_satisfies_the_adjoint_identity_at_a_nonzero_state() {
    let tagged = unit_square(3);
    let plan = nonlinear_plan(&tagged);
    let dimension = plan.dimension();
    let context = EvaluationContext::reproducible();
    let time = 0.7;
    let state = probe_vector(dimension, 0.4, 1.5);
    let rate = probe_vector(dimension, 3.1, 0.6);
    let u = probe_vector(dimension, 1.7, 1.0);
    let v = probe_vector(dimension, 5.3, 0.9);

    for rate_shift in [0.0, 3.7] {
        let jacobian = plan.linearize(time, &state, &rate, rate_shift).unwrap();
        let mut forward = vec![0.0; dimension];
        jacobian.apply(&context, &u, &mut forward).unwrap();
        let mut backward = vec![0.0; dimension];
        jacobian
            .apply_transpose(&context, &v, &mut backward)
            .unwrap();
        assert_identity(
            dot(&forward, &v),
            dot(&u, &backward),
            &format!("nonlinear Jacobian with rate shift {rate_shift}"),
        );

        // The forward action means what the shift says: `R(u + e x, u_t + e*shift*x)`
        // differentiated at `e = 0`, checked against centered differences of the residual.
        let step = 1.0e-5;
        let mut plus = vec![0.0; dimension];
        let mut minus = vec![0.0; dimension];
        let shifted = |sign: f64| {
            let state = state
                .iter()
                .zip(&u)
                .map(|(s, d)| s + sign * step * d)
                .collect::<Vec<_>>();
            let rate = rate
                .iter()
                .zip(&u)
                .map(|(r, d)| r + sign * step * rate_shift * d)
                .collect::<Vec<_>>();
            (state, rate)
        };
        let (state_plus, rate_plus) = shifted(1.0);
        let (state_minus, rate_minus) = shifted(-1.0);
        plan.residual(time, &state_plus, &rate_plus, &mut plus)
            .unwrap();
        plan.residual(time, &state_minus, &rate_minus, &mut minus)
            .unwrap();
        for index in 0..dimension {
            let centered = (plus[index] - minus[index]) / (2.0 * step);
            let scale = forward[index].abs().max(1.0);
            assert!(
                (centered - forward[index]).abs() <= 1.0e-7 * scale,
                "shifted Jacobian row {index}: centered {centered}, action {}",
                forward[index]
            );
        }
    }
}

#[test]
fn coefficient_jvp_and_vjp_are_exact_transposes_for_every_layout() {
    let tagged = unit_square(3);
    let element = PreparedElement::linear_simplex(2).unwrap();
    for layout in [
        CoefficientLayout::Vertex,
        CoefficientLayout::Cell,
        CoefficientLayout::QuadraturePoint,
    ] {
        let design_dimension = layout.dimension(&tagged.mesh, &element, 1).unwrap();
        let design = probe_vector(design_dimension, 0.2, 0.3)
            .iter()
            .map(|value| 1.5 + value)
            .collect::<Vec<_>>();
        let (plan, coefficient) = poisson_plan(&tagged, layout, &design);
        assert_eq!(
            plan.coefficient_dimension(&coefficient).unwrap(),
            design_dimension
        );
        let dimension = plan.dimension();
        let state = probe_vector(dimension, 0.6, 1.1);
        let rate = vec![0.0; dimension];
        let d = probe_vector(design_dimension, 2.2, 1.0);
        let v = probe_vector(dimension, 7.1, 0.9);

        let mut forward = vec![0.0; dimension];
        plan.coefficient_jacobian_vector_product(
            0.0,
            &state,
            &rate,
            &coefficient,
            &d,
            &mut forward,
        )
        .unwrap();
        let mut backward = vec![0.0; design_dimension];
        plan.coefficient_vector_jacobian_product(
            0.0,
            &state,
            &rate,
            &coefficient,
            &v,
            &mut backward,
        )
        .unwrap();
        assert_identity(
            dot(&forward, &v),
            dot(&d, &backward),
            &format!("coefficient products under {layout:?}"),
        );
        assert!(
            forward.iter().any(|value| value.abs() > 1.0e-6),
            "the coefficient JVP is not identically zero under {layout:?}"
        );
    }
}

#[test]
fn coefficient_jvp_matches_centered_differences_of_rebuilt_realizations() {
    let tagged = unit_square(3);
    let design = nodal_design(&tagged);
    let (plan, coefficient) = poisson_plan(&tagged, CoefficientLayout::Vertex, &design);
    let dimension = plan.dimension();
    let state = probe_vector(dimension, 0.6, 1.1);
    let rate = vec![0.0; dimension];
    let d = probe_vector(design.len(), 2.2, 1.0);
    let mut forward = vec![0.0; dimension];
    plan.coefficient_jacobian_vector_product(0.0, &state, &rate, &coefficient, &d, &mut forward)
        .unwrap();

    let step = 1.0e-3;
    let residual_at = |sign: f64| {
        let perturbed = design
            .iter()
            .zip(&d)
            .map(|(k, dk)| k + sign * step * dk)
            .collect::<Vec<_>>();
        let (plan, _) = poisson_plan(&tagged, CoefficientLayout::Vertex, &perturbed);
        let mut residual = vec![0.0; dimension];
        plan.residual(0.0, &state, &rate, &mut residual).unwrap();
        residual
    };
    let plus = residual_at(1.0);
    let minus = residual_at(-1.0);
    for index in 0..dimension {
        let centered = (plus[index] - minus[index]) / (2.0 * step);
        let scale = forward[index].abs().max(1.0);
        assert!(
            (centered - forward[index]).abs() <= 1.0e-9 * scale,
            "coefficient JVP row {index}: centered {centered}, action {}",
            forward[index]
        );
    }
}

/// `J(k) = c . u(k)` where `u(k)` solves the realized Poisson system with nodal conductivity
/// `k`; the linear solve is driven to `1e-14` so finite-difference quotients see truncation
/// error only.
fn objective(tagged: &TaggedMesh, design: &[f64], c: &[f64]) -> f64 {
    let (plan, _) = poisson_plan(tagged, CoefficientLayout::Vertex, design);
    let solution = solve_poisson(&plan);
    dot(c, &solution)
}

fn solve_poisson(plan: &RealizationPlan) -> Vec<f64> {
    plan.prove_symmetry(1.0e-12).unwrap();
    let operator = plan.matrix_free();
    let rhs = plan.load_vector().unwrap();
    let report = solve_conjugate_gradient(
        &operator,
        None,
        &EvaluationContext::reproducible(),
        &rhs,
        &vec![0.0; plan.dimension()],
        &ConjugateGradientConfig {
            max_iterations: 10_000,
            absolute_tolerance: 1.0e-14,
            relative_tolerance: 1.0e-14,
            ..ConjugateGradientConfig::default()
        },
    )
    .unwrap();
    assert!(report.converged);
    report.solution
}

#[test]
fn adjoint_gradient_of_the_objective_matches_centered_differences_with_tightening_error() {
    let tagged = unit_square(4);
    let design = nodal_design(&tagged);
    let (plan, coefficient) = poisson_plan(&tagged, CoefficientLayout::Vertex, &design);
    let dimension = plan.dimension();
    let context = EvaluationContext::reproducible();
    let c = probe_vector(dimension, 9.4, 1.0);

    // Primal solve, then the adjoint solve `J^T lambda = c` through the explicit transpose of
    // the linearized operator (GMRES, since the linearized operator claims no symmetry).
    let solution = solve_poisson(&plan);
    let zero_rate = vec![0.0; dimension];
    let jacobian = plan.linearize(0.0, &solution, &zero_rate, 0.0).unwrap();
    let transpose = TransposeOperator::explicit(&jacobian);
    let adjoint = solve_gmres(
        &transpose,
        None,
        &context,
        &c,
        &vec![0.0; dimension],
        &GmresConfig {
            max_iterations: 10_000,
            restart: 50,
            absolute_tolerance: 1.0e-14,
            relative_tolerance: 1.0e-14,
        },
    )
    .unwrap();
    assert!(adjoint.converged);

    // dJ/dk = -lambda^T dR/dk at the converged state.
    let mut gradient = vec![0.0; design.len()];
    plan.coefficient_vector_jacobian_product(
        0.0,
        &solution,
        &zero_rate,
        &coefficient,
        &adjoint.solution,
        &mut gradient,
    )
    .unwrap();
    for value in &mut gradient {
        *value = -*value;
    }

    let direction = probe_vector(design.len(), 11.0, 1.0);
    let predicted = dot(&gradient, &direction);
    assert!(predicted.abs() > 1.0e-6, "the objective is sensitive to k");
    let centered = |step: f64| {
        let perturbed = |sign: f64| {
            design
                .iter()
                .zip(&direction)
                .map(|(k, dk)| k + sign * step * dk)
                .collect::<Vec<_>>()
        };
        (objective(&tagged, &perturbed(1.0), &c) - objective(&tagged, &perturbed(-1.0), &c))
            / (2.0 * step)
    };
    let coarse_error = (centered(2.0e-2) - predicted).abs();
    let fine_error = (centered(1.0e-2) - predicted).abs();
    assert!(
        fine_error <= coarse_error / 3.0,
        "centered-difference error does not tighten: coarse {coarse_error}, fine {fine_error}"
    );
    assert!(
        fine_error <= 1.0e-5 * predicted.abs(),
        "adjoint gradient {predicted} disagrees with finite differences by {fine_error}"
    );
}
