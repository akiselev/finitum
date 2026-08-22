use cadabra_provider::{
    DifferentiabilityDisposition, ProviderFrame, RectangleProvider, RectangleRequest,
};
use finitum::{
    CadBoundaryCondition, CadGeometryRealization, CadPrimalPlan, DofMap, ElementRestriction,
    ExternalInput, FinitumError, PreparedElement, RealizationPlan,
};
use methodus::{
    ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, EvaluationContext, LinearOperator,
    solve_conjugate_gradient,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, TensorInputRole, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};

const POISSON: &str = r#"
module r3p.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

fn rectangle(revision: u64) -> RectangleProvider {
    let request = RectangleRequest::try_new(
        "fixture/cad/plate",
        revision,
        ProviderFrame::world(),
        2.0,
        1.0,
    )
    .expect("valid rectangle request");
    match RectangleProvider::admit(request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth rectangle, got {other:?}"),
    }
}

#[test]
fn same_provider_revision_recreates_every_primal_association() {
    let provider = rectangle(17);
    let first = CadGeometryRealization::from_rectangle(&provider, 17, [3, 2]).unwrap();
    let cold = CadGeometryRealization::from_rectangle(&provider, 17, [3, 2]).unwrap();
    let refined = CadGeometryRealization::from_rectangle(&provider, 17, [4, 2]).unwrap();
    assert_eq!(first, cold);
    assert_eq!(first.digest(), cold.digest());
    assert_ne!(first.digest(), refined.digest());
    assert_eq!(
        first.source().semantic_digest,
        provider.snapshot().semantic_digest.bytes()
    );
    assert_eq!(first.parameters().len(), 2);
    assert_eq!(first.nodes().len(), 12);
    assert_eq!(first.cells().len(), 12);
    assert!(
        first
            .cells()
            .iter()
            .all(|cell| cell.region_id == provider.snapshot().region.id.as_str())
    );
    for boundary in &provider.snapshot().boundaries {
        assert!(
            !first
                .boundary(boundary.id.as_str())
                .unwrap()
                .vertices
                .is_empty()
        );
    }
}

#[test]
fn stale_missing_and_ambiguous_cad_associations_are_refused() {
    let provider = rectangle(23);
    assert_eq!(
        CadGeometryRealization::from_rectangle(&provider, 22, [2, 2]),
        Err(FinitumError::StaleGeometryRevision {
            expected: 23,
            actual: 22,
        })
    );
    let geometry = CadGeometryRealization::from_rectangle(&provider, 23, [2, 2]).unwrap();
    assert_eq!(
        geometry.require_rectangle_source(&rectangle(24)),
        Err(FinitumError::StaleGeometryRevision {
            expected: 23,
            actual: 24,
        })
    );
    assert_eq!(
        geometry
            .require_rectangle_source(&rectangle(24))
            .unwrap_err()
            .to_string(),
        "stale CAD geometry revision: expected 23, got 24"
    );
    let other_request = RectangleRequest::try_new(
        "fixture/cad/other-plate",
        23,
        ProviderFrame::world(),
        2.0,
        1.0,
    )
    .unwrap();
    let other_provider = match RectangleProvider::admit(other_request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth rectangle, got {other:?}"),
    };
    assert_eq!(
        geometry.require_rectangle_source(&other_provider),
        Err(FinitumError::CadGeometrySourceMismatch)
    );
    let mutated_request =
        RectangleRequest::try_new("fixture/cad/plate", 23, ProviderFrame::world(), 3.0, 1.0)
            .unwrap();
    let mutated_provider = match RectangleProvider::admit(mutated_request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth rectangle, got {other:?}"),
    };
    assert_eq!(
        geometry.require_rectangle_source(&mutated_provider),
        Err(FinitumError::CadGeometrySourceMismatch)
    );
    assert_eq!(
        geometry.boundary("fixture/cad/plate/boundary/missing"),
        Err(FinitumError::MissingCadBoundary(
            "fixture/cad/plate/boundary/missing".into()
        ))
    );
    assert_eq!(
        geometry.essential_constraints(&[CadBoundaryCondition {
            entity_id: "fixture/cad/plate/boundary/missing".into(),
            value: 0.0,
        }]),
        Err(FinitumError::MissingCadBoundary(
            "fixture/cad/plate/boundary/missing".into()
        ))
    );
    let bottom = provider.snapshot().boundaries[0].id.as_str().to_owned();
    assert_eq!(
        geometry.essential_constraints(&[
            CadBoundaryCondition {
                entity_id: bottom.clone(),
                value: 0.0,
            },
            CadBoundaryCondition {
                entity_id: bottom.clone(),
                value: 0.0,
            },
        ]),
        Err(FinitumError::AmbiguousCadBoundary(bottom))
    );
    let left = provider.snapshot().boundaries[3].id.as_str().to_owned();
    assert!(matches!(
        geometry.essential_constraints(&[
            CadBoundaryCondition {
                entity_id: provider.snapshot().boundaries[0].id.as_str().to_owned(),
                value: 0.0,
            },
            CadBoundaryCondition {
                entity_id: left,
                value: 1.0,
            },
        ]),
        Err(FinitumError::AmbiguousCadBoundary(message)) if message.contains("overlap")
    ));
}

#[test]
fn embedded_rectangle_is_refused_instead_of_dropping_a_coordinate() {
    let frame = ProviderFrame::try_new(
        [0.0; 3],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    )
    .unwrap();
    let request = RectangleRequest::try_new("fixture/cad/embedded", 1, frame, 1.0, 1.0).unwrap();
    let provider = match RectangleProvider::admit(request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth rectangle, got {other:?}"),
    };
    assert!(matches!(
        CadGeometryRealization::from_rectangle(&provider, 1, [2, 2]),
        Err(FinitumError::InvalidCadGeometry(message)) if message.contains("XY carrier")
    ));
}

#[test]
fn scientia_poisson_uses_cad_boundary_ids_and_both_primal_paths_agree() {
    let provider = rectangle(31);
    let geometry = CadGeometryRealization::from_rectangle(&provider, 31, [2, 2]).unwrap();
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
    assert_eq!(constraints.constraints().count(), 8);

    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let mesh = geometry.mesh().clone();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let external: Vec<ExternalInput> = factorization
        .integrals
        .iter()
        .flat_map(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .filter(|input| input.source != InputSourceRequirement::Basis)
                .map(|input| {
                    assert_ne!(input.role, TensorInputRole::Active);
                    let value = match model.symbols[input.binding.symbol.index()].name.as_str() {
                        "k" => 1.0,
                        "f" => 0.0,
                        other => panic!("unexpected external input {other}"),
                    };
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        move |_, _| vec![value],
                    )
                    .unwrap()
                })
        })
        .collect();
    let canonical_dofs = geometry.nodal_dof_map().unwrap();
    let forged_dofs = DofMap::new(
        canonical_dofs.dof_count(),
        canonical_dofs
            .restrictions()
            .iter()
            .map(|restriction| {
                let mut dofs = restriction.dofs.clone();
                dofs.reverse();
                ElementRestriction { dofs }
            })
            .collect(),
    )
    .unwrap();
    let forged_dof_plan = RealizationPlan::new(
        requirements.clone(),
        factorization.clone(),
        kernels.clone(),
        mesh.clone(),
        element.clone(),
        forged_dofs,
        constraints.clone(),
        external.clone(),
    )
    .unwrap();
    assert!(matches!(
        CadPrimalPlan::new(
            geometry.clone(),
            boundary_conditions.clone(),
            forged_dof_plan
        ),
        Err(FinitumError::InvalidCadGeometry(message)) if message.contains("DOF map")
    ));
    let plan = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        canonical_dofs,
        constraints,
        external,
    )
    .unwrap();
    let dimension = plan.dimension();
    let mismatched_geometry =
        CadGeometryRealization::from_rectangle(&provider, 31, [3, 2]).unwrap();
    assert!(matches!(
        CadPrimalPlan::new(
            mismatched_geometry,
            boundary_conditions.clone(),
            plan.clone()
        ),
        Err(FinitumError::InvalidCadGeometry(message)) if message.contains("operator mesh differs")
    ));
    let forged_conditions = boundary_conditions
        .iter()
        .map(|condition| CadBoundaryCondition {
            entity_id: condition.entity_id.clone(),
            value: 1.5,
        })
        .collect();
    assert!(matches!(
        CadPrimalPlan::new(geometry.clone(), forged_conditions, plan.clone()),
        Err(FinitumError::InvalidCadGeometry(message)) if message.contains("operator constraints")
    ));
    let cad_plan = CadPrimalPlan::new(geometry, boundary_conditions, plan).unwrap();
    assert_eq!(cad_plan.geometry().source().revision, 31);
    cad_plan.require_rectangle_source(&provider).unwrap();
    assert_eq!(cad_plan.boundary_conditions().len(), 4);
    assert!(
        cad_plan
            .boundary_conditions()
            .windows(2)
            .all(|pair| pair[0].entity_id < pair[1].entity_id)
    );
    assert!(
        cad_plan
            .boundary_conditions()
            .iter()
            .all(|condition| condition.value == 2.5)
    );
    assert_ne!(cad_plan.digest(), cad_plan.realization().digest());
    let matrix_free = cad_plan.matrix_free();
    let assembled = cad_plan.assemble().unwrap();
    let context = EvaluationContext::reproducible();
    let probe = (0..dimension)
        .map(|index| index as f64 * 0.17 - 0.4)
        .collect::<Vec<_>>();
    let mut matrix_free_action = vec![0.0; dimension];
    let mut assembled_action = vec![0.0; dimension];
    matrix_free
        .apply(&context, &probe, &mut matrix_free_action)
        .unwrap();
    assembled
        .apply(&context, &probe, &mut assembled_action)
        .unwrap();
    assert_close(&matrix_free_action, &assembled_action, 1.0e-14);

    let right_hand_side = cad_plan.load_vector().unwrap();
    let config = ConjugateGradientConfig {
        symmetry_policy: ConjugateGradientSymmetryPolicy::AssumeSymmetric,
        ..ConjugateGradientConfig::default()
    };
    let initial = vec![0.0; dimension];
    let matrix_free_solution = solve_conjugate_gradient(
        &matrix_free,
        None,
        &context,
        &right_hand_side,
        &initial,
        &config,
    )
    .unwrap();
    let assembled_solution = solve_conjugate_gradient(
        &assembled,
        None,
        &context,
        &right_hand_side,
        &initial,
        &config,
    )
    .unwrap();
    assert!(matrix_free_solution.converged);
    assert!(assembled_solution.converged);
    assert_close(
        &matrix_free_solution.solution,
        &assembled_solution.solution,
        1.0e-12,
    );
    for value in &matrix_free_solution.solution {
        assert!(
            (value - 2.5).abs() < 1.0e-12,
            "manufactured constant was {value}"
        );
    }
    for value in &assembled_solution.solution {
        assert!(
            (value - 2.5).abs() < 1.0e-12,
            "manufactured constant was {value}"
        );
    }
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }
}
