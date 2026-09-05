//! W7 package 7c (single compile path), deliverable B: `properties().symmetry` on
//! `SystemOperator` / `ReducedSystemOperator` / `LinearizedSystemOperator` (and the capability)
//! reflects a taken `prove_symmetry` -- Scientia's structural claim before, the proof after --
//! so Methodus conjugate gradient admits a proven operator with no caller-side
//! `AssumeSymmetric`. `17-linear-elasticity.res` (3-D, an opaque constitutive stress, so the
//! structural claim is not `Symmetric`) is refused by CG before the proof and accepted after;
//! `01-poisson.res` is structurally `Symmetric` already and stays so through its proof. The
//! failed-proof direction (a proof never upgrades) is `tests/sv2b4_system_stokes.rs`'s
//! unsigned Stokes system.

use finitum::{
    BlockLayout, ExternalInput, FieldSource, MeshProfile, PointEvaluation, ReducedSystemOperator,
    RegionMap, RegionTagId, SysResId, SystemConstitutiveInput,
    SystemEssentialConstraintRequirement, SystemExternalInput, SystemQuadrature,
    SystemRealizationPlan, TaggedMesh, essential_constraints_from_system, realize,
};
use methodus::{
    ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, EvaluationContext, LinearOperator,
    OperatorSymmetry, solve_conjugate_gradient,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, compile_operator_system,
    compile_semantics,
};
use std::collections::BTreeMap;

const ELASTICITY: &str = include_str!("fixtures/corpus/17-linear-elasticity.res");
const POISSON: &str = include_str!("fixtures/corpus/01-poisson.res");
const LAMBDA: f64 = 1.0;
const MU: f64 = 0.5;
const BODY_FORCE: [f64; 3] = [0.0, 0.0, -1.0];

fn compile(source: &str, model: &str, equation: &str) -> OperatorSystem {
    let semantic = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    compile_operator_system(&semantic.semantic, model, &[equation]).unwrap()
}

fn simplex_box(dimension: usize, subdivisions: usize) -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: dimension as u8,
        extent: vec![[0.0, 1.0]; dimension],
        subdivisions: vec![subdivisions; dimension],
    })
    .unwrap()
}

fn boundary_region_map(region: scientia::RegionId, dimension: usize) -> RegionMap {
    let mut map = RegionMap::new();
    let tags = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"][..2 * dimension]
        .iter()
        .map(|tag| RegionTagId::new(*tag))
        .collect::<Vec<_>>();
    map.insert(region, tags);
    map
}

/// The small-strain tensor (row-major `d x d`) of a displacement evaluation: the factorization's
/// own symmetric gradient when it exposes one, else the symmetrized gradient.
fn strain_of(evaluation: &PointEvaluation, dimension: usize) -> Vec<f64> {
    if let Some(strain) = evaluation.values(DerivativeEvaluation::SymmetricGradient) {
        return strain.to_vec();
    }
    let gradient = evaluation
        .values(DerivativeEvaluation::Gradient)
        .expect("a displacement gradient is active");
    (0..dimension * dimension)
        .map(|index| {
            let (row, column) = (index / dimension, index % dimension);
            0.5 * (gradient[row * dimension + column] + gradient[column * dimension + row])
        })
        .collect()
}

/// `sigma = lambda tr(eps) I + 2 mu eps`, linear in `eps`.
fn stress(strain: &[f64], dimension: usize) -> Vec<f64> {
    let trace = (0..dimension)
        .map(|axis| strain[axis * dimension + axis])
        .sum::<f64>();
    (0..dimension * dimension)
        .map(|index| {
            let diagonal = if index / dimension == index % dimension {
                LAMBDA * trace
            } else {
                0.0
            };
            diagonal + 2.0 * MU * strain[index]
        })
        .collect()
}

/// `17-linear-elasticity.res` as a one-instance system on the unit cube: the opaque `stress`
/// constitutive a closure with its exact (symmetric) tangent, every other non-basis input a
/// stored constant table (`body_force = (0, 0, -1)`), the whole boundary clamped to zero.
fn elasticity_system(tagged: &TaggedMesh, rule: SystemQuadrature) -> ReducedSystemOperator {
    let system = compile(ELASTICITY, "LinearElasticity", "momentum");
    let block = &system.blocks[0];
    let mesh = tagged.mesh.clone();
    let dimension = mesh.dimension();
    let unknown = block.row;
    let layout = BlockLayout::new([(unknown, mesh.vertices().len(), dimension)]).unwrap();
    let plan =
        SystemRealizationPlan::with_quadrature(system.clone(), mesh.clone(), layout, rule).unwrap();
    let quadrature = plan.quadrature().unwrap();
    let extent = mesh.cells().len() * quadrature.len();
    let mut constitutive = Vec::new();
    let mut stored = Vec::new();
    for integral in &block.factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let components = input.shape.iter().product::<usize>().max(1);
            if matches!(
                input.source,
                InputSourceRequirement::ModelDefinedConstitutive { .. }
            ) {
                assert_eq!(components, dimension * dimension, "the stress tensor");
                constitutive.push(
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "linear-elasticity/hooke",
                        move |evaluation| stress(&strain_of(evaluation, dimension), dimension),
                        move |_evaluation, direction| {
                            stress(&strain_of(direction, dimension), dimension)
                        },
                    )
                    .unwrap(),
                );
                continue;
            }
            let values = if components == dimension {
                BODY_FORCE[..dimension].repeat(extent)
            } else {
                vec![1.0; extent * components]
            };
            stored.push(SystemExternalInput {
                residual: SysResId(0),
                input: ExternalInput::new(integral.integral_index, input.id, components, values)
                    .unwrap(),
            });
        }
    }
    let operator = plan
        .bind_kernels_with_inputs(constitutive, stored, BTreeMap::new(), BTreeMap::new())
        .unwrap();
    let requirement = block.factorization.essential_constraints[0].clone();
    let region_map = boundary_region_map(requirement.region, dimension);
    let constraints = essential_constraints_from_system(
        &operator,
        tagged,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: unknown,
            requirement,
            value: FieldSource::constant([0.0, 0.0, 0.0]),
        }],
    )
    .unwrap();
    operator.reduced(constraints).unwrap()
}

/// `01-poisson.res` as a one-instance system with `k = 1`, `f = 1` stored tables and
/// homogeneous walls.
fn poisson_system(tagged: &TaggedMesh) -> ReducedSystemOperator {
    let system = compile(POISSON, "Poisson", "balance");
    let block = &system.blocks[0];
    let mesh = tagged.mesh.clone();
    let unknown = block.row;
    let layout = BlockLayout::new([(unknown, mesh.vertices().len(), 1)]).unwrap();
    let plan = SystemRealizationPlan::with_quadrature(
        system.clone(),
        mesh.clone(),
        layout,
        SystemQuadrature::Barycenter,
    )
    .unwrap();
    let extent = mesh.cells().len() * plan.quadrature().unwrap().len();
    let mut stored = Vec::new();
    for integral in &block.factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            stored.push(SystemExternalInput {
                residual: SysResId(0),
                input: ExternalInput::new(integral.integral_index, input.id, 1, vec![1.0; extent])
                    .unwrap(),
            });
        }
    }
    let operator = plan
        .bind_kernels_with_inputs(Vec::new(), stored, BTreeMap::new(), BTreeMap::new())
        .unwrap();
    let requirement = block.factorization.essential_constraints[0].clone();
    let region_map = boundary_region_map(requirement.region, 2);
    let constraints = essential_constraints_from_system(
        &operator,
        tagged,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: unknown,
            requirement,
            value: FieldSource::constant([0.0]),
        }],
    )
    .unwrap();
    operator.reduced(constraints).unwrap()
}

/// Every symmetry surface Sinbad consults, for one reduced system.
fn declared_symmetry(reduced: &ReducedSystemOperator) -> [OperatorSymmetry; 5] {
    let zero = vec![0.0; reduced.rows()];
    [
        reduced.operator().symmetry(),
        reduced.symmetry(),
        reduced.properties().symmetry(),
        reduced.capability().symmetry,
        reduced
            .linearize(0.0, &zero, &zero, 0.0)
            .unwrap()
            .symmetry(),
    ]
}

fn conjugate_gradient(
    reduced: &ReducedSystemOperator,
    right_hand_side: &[f64],
) -> Result<methodus::LinearSolveReport, methodus::SolveError> {
    let config = ConjugateGradientConfig {
        max_iterations: 2_000,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-10,
        symmetry_policy: ConjugateGradientSymmetryPolicy::RequireDeclared,
    };
    solve_conjugate_gradient(
        reduced,
        None,
        &EvaluationContext::reproducible(),
        right_hand_side,
        &vec![0.0; reduced.rows()],
        &config,
    )
}

#[test]
fn linear_elasticity_is_admitted_by_conjugate_gradient_only_after_its_symmetry_proof() {
    let tagged = simplex_box(3, 3);
    let reduced = elasticity_system(&tagged, SystemQuadrature::Barycenter);
    let structural = reduced.operator().symmetry();
    assert_ne!(
        structural,
        OperatorSymmetry::Symmetric,
        "an opaque constitutive stress carries no structural symmetry"
    );
    assert!(
        declared_symmetry(&reduced)
            .iter()
            .all(|symmetry| *symmetry == structural)
    );
    let digest_before = reduced.operator().digest().clone();
    let right_hand_side = reduced.load_vector().unwrap();
    assert!(right_hand_side.iter().any(|value| value.abs() > 0.0));

    let refusal = conjugate_gradient(&reduced, &right_hand_side).unwrap_err();
    let message = refusal.to_string();
    assert!(
        message.to_lowercase().contains("symmetr"),
        "expected a symmetry refusal before the proof, got: {message}"
    );

    assert_eq!(
        reduced.operator().prove_symmetry(1.0e-10).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert!(
        declared_symmetry(&reduced)
            .iter()
            .all(|symmetry| *symmetry == OperatorSymmetry::Symmetric),
        "every symmetry surface reflects the proof: {:?}",
        declared_symmetry(&reduced)
    );
    assert_eq!(
        reduced.operator().digest(),
        &digest_before,
        "a proof is evidence about the operator, not part of its identity"
    );

    let report = conjugate_gradient(&reduced, &right_hand_side).unwrap();
    assert!(report.converged, "CG converged after the proof");
    let mut residual = vec![0.0; reduced.rows()];
    reduced
        .apply(
            &EvaluationContext::reproducible(),
            &report.solution,
            &mut residual,
        )
        .unwrap();
    let residual_norm = residual
        .iter()
        .zip(&right_hand_side)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = right_hand_side
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(
        residual_norm <= 1.0e-8 * rhs_norm,
        "CG solved the clamped elasticity system (relative residual {:e})",
        residual_norm / rhs_norm
    );
    eprintln!(
        "17-linear-elasticity: structural claim {structural:?}, proven Symmetric, CG {} \
         iterations, relative residual {:e}",
        report.trace.len(),
        residual_norm / rhs_norm
    );
}

#[test]
fn poisson_is_structurally_symmetric_and_its_proof_confirms_the_claim() {
    let tagged = simplex_box(2, 3);
    let reduced = poisson_system(&tagged);
    assert!(
        declared_symmetry(&reduced)
            .iter()
            .all(|symmetry| *symmetry == OperatorSymmetry::Symmetric),
        "Scientia's structural claim for Poisson is Symmetric (C5.4/C5.5)"
    );
    let right_hand_side = reduced.load_vector().unwrap();
    let before = conjugate_gradient(&reduced, &right_hand_side).unwrap();
    assert!(before.converged);
    assert_eq!(
        reduced.operator().prove_symmetry(1.0e-12).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert!(
        declared_symmetry(&reduced)
            .iter()
            .all(|symmetry| *symmetry == OperatorSymmetry::Symmetric)
    );
    let after = conjugate_gradient(&reduced, &right_hand_side).unwrap();
    assert_eq!(before.solution, after.solution);
}
