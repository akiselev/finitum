//! GX-C5 (partial): provable `MatrixFreeOperator`/`AssembledOperator` symmetry declarations
//! and the `RealizationCapability`/`RealizationReceipt` product-inspection contract (SV2-A2).

use finitum::{
    AffineConstraint, Cell, ConstraintKind, ConstraintSet, DerivativeProduct, DofId, DofMap,
    ElementRestriction, ExternalInput, HangingNodeConstraint, Mesh, PreparedElement,
    RealizationPlan, RepresentationKind, VertexId,
};
use methodus::{
    EvaluationContext, LinearOperator, OperatorSymmetry, transpose_view, verify_adjoint_identity,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, TensorInputRole, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};

const POISSON: &str = r#"
module gx_c5.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

/// Builds the same generated Poisson plan with either only fixed essential rows (provably
/// symmetric) or one additional affine hanging-node dependency constraint (destroys symmetry).
fn poisson_plan(with_hanging_constraint: bool) -> RealizationPlan {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let subdivisions = 3;
    let width = subdivisions + 1;
    let vertices = (0..=subdivisions)
        .flat_map(|row| {
            (0..=subdivisions).map(move |column| {
                vec![
                    column as f64 / subdivisions as f64,
                    row as f64 / subdivisions as f64,
                ]
            })
        })
        .collect::<Vec<_>>();
    let cells = (0..subdivisions)
        .flat_map(|row| {
            (0..subdivisions).flat_map(move |column| {
                let lower_left = row * width + column;
                let lower_right = lower_left + 1;
                let upper_left = lower_left + width;
                let upper_right = upper_left + 1;
                [
                    Cell {
                        vertices: vec![
                            VertexId(lower_left),
                            VertexId(lower_right),
                            VertexId(upper_right),
                        ],
                    },
                    Cell {
                        vertices: vec![
                            VertexId(lower_left),
                            VertexId(upper_right),
                            VertexId(upper_left),
                        ],
                    },
                ]
            })
        })
        .collect::<Vec<_>>();
    let restrictions = cells
        .iter()
        .map(|cell| ElementRestriction {
            dofs: cell.vertices.iter().map(|vertex| DofId(vertex.0)).collect(),
        })
        .collect();
    let mesh = Mesh::new(2, vertices, cells).unwrap();
    let dofs = DofMap::new(width * width, restrictions).unwrap();
    let mut constraints = (0..width * width)
        .filter(|index| {
            let row = index / width;
            let column = index % width;
            row == 0 || column == 0 || row == subdivisions || column == subdivisions
        })
        .map(|target| AffineConstraint {
            target: DofId(target),
            dependencies: Vec::new(),
            offset: 0.0,
        })
        .collect::<Vec<_>>();
    if with_hanging_constraint {
        constraints.push(
            HangingNodeConstraint::linear(DofId(5), DofId(1), DofId(9), 0.5)
                .unwrap()
                .into_affine(),
        );
    }
    let constraints = ConstraintSet::new(width * width, constraints).unwrap();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let external = factorization
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
                    let name = &model.symbols[input.binding.symbol.index()].name;
                    let value = match name.as_str() {
                        "k" => 1.0,
                        "f" => 0.4,
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
    RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        external,
    )
    .unwrap()
}

#[test]
fn matrix_free_and_assembled_declare_symmetric_when_provable() {
    let plan = poisson_plan(false);
    let matrix_free = plan.matrix_free();
    // No proof has been established yet: the matrix-free action makes no claim.
    assert_eq!(matrix_free.symmetry(), OperatorSymmetry::Unknown);
    // An explicit, tolerance-based proof records `Symmetric` on the realization, and every
    // clone (including the operator created before the proof) reports it.
    assert_eq!(
        plan.prove_symmetry(1.0e-12).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert_eq!(matrix_free.symmetry(), OperatorSymmetry::Symmetric);
    assert_eq!(
        plan.prove_symmetry(0.0).unwrap(),
        OperatorSymmetry::Symmetric,
        "a recorded proof is returned without reassembly"
    );
    let assembled = plan.assemble().unwrap();
    assert_eq!(assembled.symmetry(), OperatorSymmetry::Symmetric);

    // `transpose_view`/`verify_adjoint_identity` (SV1-C5/D1) now succeed on the matrix-free
    // realization because it declares `Symmetric`, not just on the assembled matrix.
    let context = EvaluationContext::reproducible();
    let transpose = transpose_view(&matrix_free).expect("symmetric matrix-free transpose view");
    let u = (0..plan.dimension())
        .map(|index| 0.3 - 0.1 * index as f64)
        .collect::<Vec<_>>();
    let v = (0..plan.dimension())
        .map(|index| 0.05 * index as f64 - 0.2)
        .collect::<Vec<_>>();
    let discrepancy =
        verify_adjoint_identity(&matrix_free, &transpose, &context, &u, &v, 1.0e-8).unwrap();
    assert!(
        discrepancy < 1.0e-8,
        "adjoint identity discrepancy {discrepancy}"
    );
}

#[test]
fn matrix_free_stays_nonsymmetric_with_an_affine_dependency_constraint() {
    let plan = poisson_plan(true);
    assert_eq!(
        plan.matrix_free().symmetry(),
        OperatorSymmetry::Nonsymmetric
    );
    assert_eq!(
        plan.prove_symmetry(1.0e-12).unwrap(),
        OperatorSymmetry::Nonsymmetric
    );
    assert_eq!(
        plan.assemble().unwrap().symmetry(),
        OperatorSymmetry::Nonsymmetric
    );
    assert!(transpose_view(&plan.matrix_free()).is_err());
}

#[test]
fn capability_reports_admitted_shape_and_symmetry_with_a_canonical_digest() {
    let plan = poisson_plan(false);
    plan.prove_symmetry(1.0e-12).unwrap();
    let capability = plan.capability();
    assert_eq!(capability.topology_dimension, 2);
    assert_eq!(capability.symmetry, OperatorSymmetry::Symmetric);
    assert!(
        capability
            .elements
            .iter()
            .any(|element| element.polynomial_order == 1)
    );
    assert!(!capability.measures.is_empty());
    assert!(capability.constraint_kinds.contains(&ConstraintKind::Fixed));
    assert!(
        !capability
            .constraint_kinds
            .contains(&ConstraintKind::AffineDependency)
    );
    assert!(
        capability
            .representation_kinds
            .contains(&RepresentationKind::MatrixFree)
    );
    assert!(
        capability
            .representation_kinds
            .contains(&RepresentationKind::Assembled)
    );
    assert!(
        capability
            .derivative_products
            .contains(&DerivativeProduct::Primal)
    );
    assert!(
        capability
            .derivative_products
            .contains(&DerivativeProduct::Jvp)
    );
    assert_eq!(&capability.receipt.realization_digest, plan.digest());

    // Deterministic: recomputing from the same plan reproduces the same digest.
    let repeat = plan.capability();
    assert_eq!(capability.digest, repeat.digest);

    // A structurally different plan (the hanging-node constraint changes both the constraint
    // kinds present and the declared symmetry) gets a different capability digest.
    let hanging_capability = poisson_plan(true).capability();
    assert!(
        hanging_capability
            .constraint_kinds
            .contains(&ConstraintKind::AffineDependency)
    );
    assert_eq!(hanging_capability.symmetry, OperatorSymmetry::Nonsymmetric);
    assert_ne!(hanging_capability.digest, capability.digest);
}
