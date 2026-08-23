//! R3D/SV1-G0B acceptance: exact CAD design velocities, annulus realization,
//! and geometry-sensitivity actions verified against rebuilt-system centered
//! differences.

use cadabra_provider::{
    AnalyticFamily, AnalyticProvider, DifferentiabilityDisposition, FamilyRequest, ProviderFamily,
    ProviderFrame, RectangleProvider, RectangleRequest,
};
use finitum::{
    CadBoundaryCondition, CadGeometryRealization, ExternalInput, ExternalSensitivityInput,
    FinitumError, GeometryParameterSensitivity, PreparedElement, RealizationPlan,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, TensorInputId, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};

const POISSON: &str = r#"
module r3d.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

fn rectangle(width: f64, height: f64, revision: u64) -> RectangleProvider {
    let request = RectangleRequest::try_new(
        "fixture/cad/r3d",
        revision,
        ProviderFrame::world(),
        width,
        height,
    )
    .expect("valid rectangle request");
    match RectangleProvider::admit(request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth rectangle, got {other:?}"),
    }
}

fn annulus(inner: f64, outer: f64, revision: u64) -> AnalyticProvider {
    let request = FamilyRequest::try_new(
        "fixture/cad/ring",
        revision,
        ProviderFrame::world(),
        AnalyticFamily::Annulus {
            inner_radius: inner,
            outer_radius: outer,
        },
    )
    .expect("valid annulus request");
    match AnalyticProvider::admit(request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth annulus, got {other:?}"),
    }
}

#[test]
fn annulus_realization_is_deterministic_oriented_and_boundary_stable() {
    let provider = annulus(0.5, 2.0, 7);
    let first = CadGeometryRealization::from_family(&provider, 7, [2, 8]).unwrap();
    let cold = CadGeometryRealization::from_family(&provider, 7, [2, 8]).unwrap();
    assert_eq!(first, cold);
    assert_eq!(first.digest(), cold.digest());
    assert_eq!(first.nodes().len(), 3 * 8);
    assert_eq!(first.cells().len(), 2 * 8 * 2);
    // The angular seam wraps onto column zero instead of duplicating nodes.
    assert!(
        first
            .nodes()
            .iter()
            .all(|node| node.reference_coordinate[1] < 1.0)
    );
    // Every triangle is positively oriented in the polar chart.
    let vertices = first.mesh().vertices();
    for cell in first.mesh().cells() {
        let a = &vertices[cell.vertices[0].0];
        let b = &vertices[cell.vertices[1].0];
        let c = &vertices[cell.vertices[2].0];
        let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        assert!(
            twice_area > 0.0,
            "cell is not counter-clockwise: {twice_area}"
        );
    }
    let snapshot = provider.snapshot();
    let inner = first.boundary(snapshot.boundaries[0].id.as_str()).unwrap();
    let outer = first.boundary(snapshot.boundaries[1].id.as_str()).unwrap();
    assert_eq!(
        inner.vertices,
        (0..8).map(finitum::VertexId).collect::<Vec<_>>()
    );
    assert_eq!(
        outer.vertices,
        (16..24).map(finitum::VertexId).collect::<Vec<_>>()
    );
    for vertex in &inner.vertices {
        let position = &vertices[vertex.0];
        let radius = position[0].hypot(position[1]);
        assert!(
            (radius - 0.5).abs() < 1.0e-15,
            "inner ring radius {radius} is not the authored 0.5"
        );
    }
}

#[test]
fn non_annulus_families_and_stale_revisions_are_refused() {
    let box_request = FamilyRequest::try_new(
        "fixture/cad/solid",
        3,
        ProviderFrame::world(),
        AnalyticFamily::AffineBox {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        },
    )
    .unwrap();
    let solid = match AnalyticProvider::admit(box_request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth box, got {other:?}"),
    };
    assert_eq!(
        CadGeometryRealization::from_family(&solid, 3, [2, 8]),
        Err(FinitumError::UnsupportedCadFamily {
            family: format!("{:?}", ProviderFamily::AffineBox),
            reason: "the first R3D slice realizes admitted planar annuli only",
        })
    );
    let provider = annulus(0.5, 2.0, 11);
    assert_eq!(
        CadGeometryRealization::from_family(&provider, 10, [2, 8]),
        Err(FinitumError::StaleGeometryRevision {
            expected: 11,
            actual: 10,
        })
    );
    assert!(matches!(
        CadGeometryRealization::from_family(&provider, 11, [2, 2]),
        Err(FinitumError::InvalidCadGeometry(message))
            if message.contains("three angular columns")
    ));
}

#[test]
fn rectangle_design_velocities_match_centered_positions() {
    let provider = rectangle(2.0, 1.0, 13);
    let geometry = CadGeometryRealization::from_rectangle(&provider, 13, [3, 2]).unwrap();
    let velocities = geometry
        .rectangle_parameter_velocity(&provider, 13, 0)
        .unwrap();
    let step = 1.0e-6;
    let plus = rectangle(2.0 + step, 1.0, 13);
    let minus = rectangle(2.0 - step, 1.0, 13);
    for (node, velocity) in geometry.nodes().iter().zip(&velocities) {
        let xi = node.reference_coordinate;
        let forward = plus.evaluate(xi).unwrap().position;
        let backward = minus.evaluate(xi).unwrap().position;
        for axis in 0..2 {
            let estimate = (forward[axis] - backward[axis]) / (2.0 * step);
            assert!(
                (estimate - velocity[axis]).abs() <= 1.0e-6 * (1.0 + estimate.abs()),
                "velocity {velocity:?} differs from centered estimate {estimate} at {xi:?}"
            );
        }
    }
    assert!(matches!(
        geometry.rectangle_parameter_velocity(&provider, 12, 0),
        Err(FinitumError::StaleGeometryRevision { .. })
    ));
    assert!(matches!(
        geometry.rectangle_parameter_velocity(&provider, 13, 2),
        Err(FinitumError::InvalidCadGeometry(message))
            if message.contains("outside the two declared parameters")
    ));
}

#[test]
fn annulus_design_velocities_match_centered_positions() {
    let provider = annulus(0.5, 2.0, 17);
    let geometry = CadGeometryRealization::from_family(&provider, 17, [2, 8]).unwrap();
    let velocities = geometry
        .family_parameter_velocity(&provider, 17, 0)
        .unwrap();
    let step = 1.0e-6;
    let plus = annulus(0.5 + step, 2.0, 17);
    let minus = annulus(0.5 - step, 2.0, 17);
    for (node, velocity) in geometry.nodes().iter().zip(&velocities) {
        let xi = node.reference_coordinate;
        let forward = plus.evaluate(&xi).unwrap().position;
        let backward = minus.evaluate(&xi).unwrap().position;
        for axis in 0..2 {
            let estimate = (forward[axis] - backward[axis]) / (2.0 * step);
            assert!(
                (estimate - velocity[axis]).abs() <= 1.0e-6 * (1.0 + estimate.abs()),
                "velocity {velocity:?} differs from centered estimate {estimate} at {xi:?}"
            );
        }
    }
    // A mismatched geometry source cannot pull velocities from this provider.
    let unrelated =
        CadGeometryRealization::from_rectangle(&rectangle(2.0, 1.0, 17), 17, [2, 2]).unwrap();
    assert_eq!(
        unrelated.family_parameter_velocity(&provider, 17, 0),
        Err(FinitumError::CadGeometrySourceMismatch)
    );
}

/// Chart-authored forcing: explicit parameter dependence, no inversion.
fn shape_forcing(width: f64, height: f64, point: &[f64]) -> Vec<f64> {
    let xi = point[0] / width;
    let eta = point[1] / height;
    let shape = (std::f64::consts::PI * xi).sin() * (std::f64::consts::PI * eta).sin();
    vec![std::f64::consts::PI.powi(2) * (width.recip().powi(2) + height.recip().powi(2)) * shape]
}

/// Exact d/dp of [`shape_forcing`] at a frozen chart coordinate.
fn shape_forcing_derivative(
    width: f64,
    height: f64,
    parameter_index: usize,
    point: &[f64],
) -> Vec<f64> {
    let xi = point[0] / width;
    let eta = point[1] / height;
    let shape = (std::f64::consts::PI * xi).sin() * (std::f64::consts::PI * eta).sin();
    let scale = if parameter_index == 0 {
        -2.0 * std::f64::consts::PI.powi(2) * width.recip().powi(3)
    } else {
        -2.0 * std::f64::consts::PI.powi(2) * height.recip().powi(3)
    };
    vec![scale * shape]
}

/// Builds the Poisson realization for one authored rectangle and reports its
/// named stored inputs so direction tables can discriminate diffusivity from
/// forcing.
#[allow(clippy::type_complexity)]
fn chart_forcing_plan(
    width: f64,
    height: f64,
    revision: u64,
    subdivisions: [usize; 2],
) -> (
    CadGeometryRealization,
    RealizationPlan,
    Vec<(String, usize, TensorInputId)>,
) {
    let provider = rectangle(width, height, revision);
    let geometry =
        CadGeometryRealization::from_rectangle(&provider, revision, subdivisions).unwrap();
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let model = &compilation.semantic.models[0];
    let mut stored_inputs = Vec::new();
    let boundary_conditions = provider
        .snapshot()
        .boundaries
        .iter()
        .map(|boundary| CadBoundaryCondition {
            entity_id: boundary.id.as_str().to_owned(),
            value: 2.5,
        })
        .collect::<Vec<_>>();
    let constraints = geometry
        .essential_constraints(&boundary_conditions)
        .unwrap();
    let mesh = geometry.mesh().clone();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let mut external = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.clone();
            let sampled = match name.as_str() {
                "k" => ExternalInput::sampled(
                    integral.integral_index,
                    input.id,
                    1,
                    &mesh,
                    &element,
                    |_, _| vec![1.0],
                ),
                "f" => ExternalInput::sampled(
                    integral.integral_index,
                    input.id,
                    1,
                    &mesh,
                    &element,
                    |_, point| shape_forcing(width, height, point),
                ),
                other => panic!("unexpected external input {other}"),
            }
            .unwrap();
            stored_inputs.push((name, integral.integral_index, input.id));
            external.push(sampled);
        }
    }
    let plan = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        geometry.nodal_dof_map().unwrap(),
        constraints,
        external,
    )
    .unwrap();
    (geometry, plan, stored_inputs)
}

fn chart_forcing_directions(
    geometry: &CadGeometryRealization,
    stored_inputs: &[(String, usize, TensorInputId)],
    width: f64,
    height: f64,
    parameter_index: usize,
) -> GeometryParameterSensitivity {
    let provider = rectangle(width, height, geometry.source().revision);
    let velocities = geometry
        .rectangle_parameter_velocity(&provider, geometry.source().revision, parameter_index)
        .unwrap();
    let mut flattened = Vec::with_capacity(geometry.nodes().len() * 2);
    for velocity in &velocities {
        flattened.extend_from_slice(velocity);
    }
    let element = PreparedElement::linear_simplex(2).unwrap();
    let mut directions = Vec::new();
    for (name, integral_index, input_id) in stored_inputs {
        let values = if name == "k" {
            ExternalSensitivityInput::new(
                *integral_index,
                *input_id,
                1,
                vec![0.0; geometry.cells().len()],
            )
            .unwrap()
        } else {
            ExternalSensitivityInput::sampled(
                *integral_index,
                *input_id,
                1,
                geometry.mesh(),
                &element,
                |_, point| shape_forcing_derivative(width, height, parameter_index, point),
            )
            .unwrap()
        };
        directions.push(values);
    }
    GeometryParameterSensitivity {
        parameter_index,
        node_velocities: flattened,
        external_directions: directions,
    }
}

#[test]
fn residual_geometry_sensitivity_matches_rebuilt_centered_differences() {
    let width = 2.0_f64;
    let height = 1.0_f64;
    let (_, geometry, plan, stored) = build_fixture(width, height, 23);
    let dimension = plan.dimension();
    let state = deterministic_vector(dimension, 37);
    for parameter_index in 0..2 {
        let sensitivity =
            chart_forcing_directions(&geometry, &stored, width, height, parameter_index);
        let mut analytic = vec![0.0; dimension];
        plan.residual_geometry_sensitivity(0.0, &state, &sensitivity, &mut analytic)
            .unwrap();
        let mut previous_error = None;
        for step in [1.0e-4, 1.0e-5] {
            let (plus_w, plus_h, minus_w, minus_h) =
                perturbation(width, height, parameter_index, step);
            let forward = rebuilt_residual(plus_w, plus_h, &state);
            let backward = rebuilt_residual(minus_w, minus_h, &state);
            let error = analytic
                .iter()
                .enumerate()
                .map(|(slot, exact)| {
                    ((forward[slot] - backward[slot]) / (2.0 * step) - exact).abs()
                })
                .fold(0.0, f64::max);
            assert!(
                error < 5.0e-6,
                "parameter {parameter_index} step {step}: max residual sensitivity error {error}"
            );
            if let Some(previous) = previous_error {
                assert!(
                    error < previous,
                    "parameter {parameter_index}: error did not tighten ({previous} -> {error})"
                );
            }
            previous_error = Some(error);
        }
    }
}

#[test]
fn adjoint_identity_gradient_matches_rebuilt_centered_differences() {
    use methodus::{
        ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, EvaluationContext,
        solve_conjugate_gradient,
    };
    let width = 2.0_f64;
    let height = 1.0_f64;
    let context = EvaluationContext::reproducible();
    let solver = ConjugateGradientConfig {
        symmetry_policy: ConjugateGradientSymmetryPolicy::AssumeSymmetric,
        ..ConjugateGradientConfig::default()
    };

    let solve_j = |w: f64, h: f64| -> (Vec<f64>, Vec<f64>, f64, usize) {
        let (_, geometry, plan, stored) = build_fixture(w, h, 37);
        let dimension = plan.dimension();
        let operator = plan.matrix_free();
        let right_hand_side = plan.load_vector().unwrap();
        let initial = vec![0.0; dimension];
        let primal = solve_conjugate_gradient(
            &operator,
            None,
            &context,
            &right_hand_side,
            &initial,
            &solver,
        )
        .unwrap();
        assert!(primal.converged);
        let objective = primal.solution.iter().map(|v| v * v).sum::<f64>() / dimension as f64;
        let j_u = primal
            .solution
            .iter()
            .map(|v| 2.0 * v / dimension as f64)
            .collect::<Vec<_>>();
        let adjoint =
            solve_conjugate_gradient(&operator, None, &context, &j_u, &initial, &solver).unwrap();
        assert!(adjoint.converged);
        // Per-parameter exact residual sensitivity at the converged state.
        let mut gradients = Vec::new();
        for parameter_index in 0..2 {
            let sensitivity = chart_forcing_directions(&geometry, &stored, w, h, parameter_index);
            let mut residual_sensitivity = vec![0.0; dimension];
            plan.residual_geometry_sensitivity(
                0.0,
                &primal.solution,
                &sensitivity,
                &mut residual_sensitivity,
            )
            .unwrap();
            gradients.push(
                -residual_sensitivity
                    .iter()
                    .zip(&adjoint.solution)
                    .map(|(sensitivity, weight)| sensitivity * weight)
                    .sum::<f64>(),
            );
        }
        let _ = geometry.nodes().len();
        (primal.solution, gradients, objective, dimension)
    };
    let (_, analytic, _, _) = solve_j(width, height);
    for (parameter_index, exact) in analytic.iter().enumerate() {
        let mut previous_error = None;
        for step in [2.0e-4, 1.0e-4] {
            let (plus_w, plus_h, minus_w, minus_h) =
                perturbation(width, height, parameter_index, step);
            let (_, _, j_plus, _) = solve_j(plus_w, plus_h);
            let (_, _, j_minus, _) = solve_j(minus_w, minus_h);
            let estimate = (j_plus - j_minus) / (2.0 * step);
            let error = (estimate - exact).abs();
            assert!(
                error < 5.0e-6,
                "parameter {parameter_index} step {step}: |adjoint {exact:+} - fd {estimate:.9}| = {error}"
            );
            if let Some(previous) = previous_error {
                assert!(
                    error <= previous,
                    "error must tighten: {previous} -> {error}"
                );
            }
            previous_error = Some(error);
        }
    }
}

#[test]
fn incomplete_or_extra_sensitivity_inputs_are_refused() {
    let width = 2.0_f64;
    let height = 1.0_f64;
    let (_, geometry, plan, stored) = build_fixture(width, height, 31);
    let dimension = plan.dimension();
    let state = deterministic_vector(dimension, 71);
    let complete = chart_forcing_directions(&geometry, &stored, width, height, 0);

    let mut missing = complete.clone();
    missing.external_directions.pop();
    assert!(matches!(
        plan.residual_geometry_sensitivity(0.0, &state, &missing, &mut vec![0.0; dimension]),
        Err(FinitumError::MissingExternalInput { .. })
    ));

    let mut extra = complete.clone();
    extra
        .external_directions
        .push(extra.external_directions[0].clone());
    assert!(matches!(
        plan.residual_geometry_sensitivity(0.0, &state, &extra, &mut vec![0.0; dimension]),
        Err(FinitumError::InvalidRealization(message)) if message.contains("more than once")
    ));

    let mut truncated = complete.clone();
    truncated.node_velocities.truncate(4);
    assert!(matches!(
        plan.residual_geometry_sensitivity(0.0, &state, &truncated, &mut vec![0.0; dimension]),
        Err(FinitumError::InvalidRealization(message))
            if message.contains("node velocities have length")
    ));

    let mut nonfinite = complete;
    nonfinite.node_velocities[0] = f64::NAN;
    assert!(matches!(
        plan.residual_geometry_sensitivity(0.0, &state, &nonfinite, &mut vec![0.0; dimension]),
        Err(FinitumError::InvalidRealization(message))
            if message.contains("non-finite")
    ));
}

fn build_fixture(
    width: f64,
    height: f64,
    revision: u64,
) -> (
    (),
    CadGeometryRealization,
    RealizationPlan,
    Vec<(String, usize, TensorInputId)>,
) {
    let (geometry, plan, stored) = chart_forcing_plan(width, height, revision, [3, 2]);
    ((), geometry, plan, stored)
}

fn perturbation(
    width: f64,
    height: f64,
    parameter_index: usize,
    step: f64,
) -> (f64, f64, f64, f64) {
    if parameter_index == 0 {
        (width + step, height, width - step, height)
    } else {
        (width, height + step, width, height - step)
    }
}

fn rebuilt_residual(width: f64, height: f64, state: &[f64]) -> Vec<f64> {
    let (_, plan, _) = chart_forcing_plan(width, height, 23, [3, 2]);
    let mut output = vec![0.0; plan.dimension()];
    plan.residual(0.0, state, state, &mut output).unwrap();
    output
}

fn deterministic_vector(dimension: usize, seed: usize) -> Vec<f64> {
    (0..dimension)
        .map(|index| ((index * seed % 101) as f64) / 101.0 - 0.5)
        .collect()
}

#[test]
fn annulus_residual_sensitivity_matches_rebuilt_centered_differences() {
    use cadabra_provider::{AnalyticFamily, AnalyticProvider, FamilyRequest};
    let inner = 0.5_f64;
    let outer = 2.0_f64;
    let revision = 91;
    let admit = |ri: f64, ro: f64| {
        let request = FamilyRequest::try_new(
            "fixture/cad/r3d-ring",
            revision,
            ProviderFrame::world(),
            AnalyticFamily::Annulus {
                inner_radius: ri,
                outer_radius: ro,
            },
        )
        .unwrap();
        match AnalyticProvider::admit(request) {
            DifferentiabilityDisposition::Smooth { value, .. } => value,
            other => panic!("{other:?}"),
        }
    };
    // Forcing is frozen to zero so the rebuilt centered differences isolate
    // the pure geometry chain on curved-cell velocity fields.
    let build = |ri: f64, ro: f64| {
        let provider = admit(ri, ro);
        let geometry = CadGeometryRealization::from_family(&provider, revision, [2, 8]).unwrap();
        let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
        let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
        let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
        let factorization = factor_operator(&form, &requirements).unwrap();
        let kernels = lower_operator_kernels(&factorization).unwrap();
        let model = &compilation.semantic.models[0];
        let bcs: Vec<CadBoundaryCondition> = provider
            .snapshot()
            .boundaries
            .iter()
            .map(|b| CadBoundaryCondition {
                entity_id: b.id.as_str().to_owned(),
                value: 0.25,
            })
            .collect();
        let constraints = geometry.essential_constraints(&bcs).unwrap();
        let mesh = geometry.mesh().clone();
        let element = PreparedElement::linear_simplex(2).unwrap();
        let mut external = Vec::new();
        for integral in &factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = model.symbols[input.binding.symbol.index()].name.clone();
                let sampled = match name.as_str() {
                    "k" => ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        |_, _| vec![1.0],
                    )
                    .unwrap(),
                    "f" => ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        |_, _| vec![0.0],
                    )
                    .unwrap(),
                    other => panic!("{other}"),
                };
                external.push(sampled);
            }
        }
        let plan = RealizationPlan::new(
            requirements,
            factorization,
            kernels,
            mesh,
            element,
            geometry.nodal_dof_map().unwrap(),
            constraints,
            external,
        )
        .unwrap();
        (geometry, plan)
    };

    let (geometry, plan) = build(inner, outer);
    let dimension = plan.dimension();
    let state = (0..dimension)
        .map(|i| ((i * 29 % 83) as f64) / 83.0 - 0.4)
        .collect::<Vec<_>>();
    let provider = admit(inner, outer);
    let mut velocities = Vec::new();
    for v in geometry
        .family_parameter_velocity(&provider, revision, 0)
        .unwrap()
    {
        velocities.extend_from_slice(&v);
    }
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let factorization = factor_operator(
        &form,
        &infer_form_requirements(&compilation.semantic, &form).unwrap(),
    )
    .unwrap();
    let mut dirs = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            dirs.push(
                ExternalSensitivityInput::new(
                    integral.integral_index,
                    input.id,
                    1,
                    vec![0.0; geometry.cells().len()],
                )
                .unwrap(),
            );
        }
    }
    let sensitivity = GeometryParameterSensitivity {
        parameter_index: 0,
        node_velocities: velocities,
        external_directions: dirs,
    };
    let mut analytic = vec![0.0; dimension];
    plan.residual_geometry_sensitivity(0.0, &state, &sensitivity, &mut analytic)
        .unwrap();

    let step = 1.0e-5;
    let (_, forward_plan) = build(inner + step, outer);
    let (_, backward_plan) = build(inner - step, outer);
    let mut forward = vec![0.0; dimension];
    let mut backward = vec![0.0; dimension];
    forward_plan
        .residual(0.0, &state, &state, &mut forward)
        .unwrap();
    backward_plan
        .residual(0.0, &state, &state, &mut backward)
        .unwrap();
    let max_err = analytic
        .iter()
        .enumerate()
        .map(|(i, a)| ((forward[i] - backward[i]) / (2.0 * step) - a).abs())
        .fold(0.0, f64::max);
    assert!(max_err < 5.0e-6, "annulus engine mismatch {max_err}");
}
