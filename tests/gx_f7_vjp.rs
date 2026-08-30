//! GX-F7: `RealizationPlan::vector_jacobian_product` executes the bound Malleus VJP kernels as
//! the exact transpose of `jacobian_vector_product` at the same linearization point.
//!
//! Acceptance evidence (hermetic, no corpus):
//! 1. the adjoint identity `<A u, v> == <u, A^T v>` between the JVP and the VJP, for a scalar
//!    Poisson-style fixture, a vector H1 elasticity fixture with a dynamic constitutive law, and
//!    a nonlinear fixture with a dynamic external input at a nonzero linearization state;
//! 2. agreement with `AssembledOperator::transpose()` applied to the same vector;
//! 3. `check_global_transpose` accepting the matrix-free forward action against the VJP as its
//!    transpose;
//! 4. a typed refusal for affine dependency constraints;
//! 5. `RealizationCapability` reports `Vjp` only when the plan can execute it.

use finitum::{
    AffineConstraint, Cell, ConstraintSet, DerivativeProduct, DofId, DofMap, DynamicExternalInput,
    ElementRestriction, ExternalInput, HangingNodeConstraint, Mesh, PreparedElement,
    RealizationPlan, VerificationSubject, VertexId, check_global_transpose, vector_nodal_dof_map,
};
use methodus::{ComparisonTolerance, EvaluationContext, LinearOperator, NumericError};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};
use std::collections::BTreeMap;

const POISSON: &str = r#"
module gx_f7.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

const ELASTICITY: &str = r#"
module gx_f7.elasticity;
model Elasticity {
  domain Omega { dimension = 3; coordinates = cartesian; }
  field u: unknown vector(3) H1(order=1) on Omega;
  property lambda = lame_lambda(0);
  property mu = lame_mu(0);
  source body_force: MechanicalBodyForce;

  constitutive strain = sym_grad(u);
  constitutive stress = lambda * trace(strain) * identity(3) + 2 * mu * strain;

  equation momentum on Omega {
    -div(stress) = body_force;
  }
  boundary clamp on boundary("clamp") { dirichlet u = [0, 0, 0]; }
}
"#;

const TRANSIENT_NONLINEAR: &str = r#"
module gx_f7.transient_nonlinear;
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

const LAMBDA: f64 = 1.25;
const MU: f64 = 1.0;

const IDENTITY_TOLERANCE: f64 = 1.0e-9;

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn probe_vector(dimension: usize, seed: f64, scale: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| scale * ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

/// Isotropic stress sigma = lambda tr(eps) I + 2 mu eps, row-major [a][b].
fn stress(lambda: f64, mu: f64, strain: &[f64]) -> Vec<f64> {
    let dimension = 3;
    let mut trace = 0.0;
    for axis in 0..dimension {
        trace += strain[axis * dimension + axis];
    }
    let mut result = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            result[row * dimension + column] = 2.0 * mu * strain[row * dimension + column]
                + if row == column { lambda * trace } else { 0.0 };
        }
    }
    result
}

/// Asserts `<jvp(direction), adjoint> == <direction, vjp(adjoint)>` at the given (state,
/// state_rate) linearization point, with rate direction held at zero (matching the domain
/// `vector_jacobian_product` inverts).
fn assert_adjoint_identity(
    plan: &RealizationPlan,
    time: f64,
    state: &[f64],
    state_rate: &[f64],
    direction: &[f64],
    adjoint: &[f64],
) {
    let dimension = plan.dimension();
    let zero_rate_direction = vec![0.0; dimension];
    let mut forward = vec![0.0; dimension];
    plan.jacobian_vector_product(
        time,
        state,
        state_rate,
        direction,
        &zero_rate_direction,
        &mut forward,
    )
    .unwrap();
    let mut backward = vec![0.0; dimension];
    plan.vector_jacobian_product(time, state, state_rate, adjoint, &mut backward)
        .unwrap();
    let left = dot(&forward, adjoint);
    let right = dot(direction, &backward);
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= IDENTITY_TOLERANCE * scale,
        "adjoint identity mismatch: <Au,v>={left}, <u,A^Tv>={right}"
    );
}

/// Wraps `RealizationPlan::vector_jacobian_product` at zero state/state-rate as a Methodus
/// `LinearOperator`, matching `MatrixFreeOperator`'s own zero-state convention so the two can be
/// compared through `check_global_transpose`.
struct VjpAtZeroState<'a> {
    plan: &'a RealizationPlan,
}

impl LinearOperator for VjpAtZeroState<'_> {
    fn rows(&self) -> usize {
        self.plan.dimension()
    }

    fn columns(&self) -> usize {
        self.plan.dimension()
    }

    fn symmetry(&self) -> methodus::OperatorSymmetry {
        methodus::OperatorSymmetry::Unknown
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; self.plan.dimension()];
        self.plan
            .vector_jacobian_product(0.0, &zero, &zero, input, output)
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

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
                    let name = &model.symbols[input.binding.symbol.index()].name;
                    let value = match name.as_str() {
                        "k" => 1.3,
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

/// Kuhn decomposition of one axis-aligned brick into six tets sharing the main diagonal;
/// face-conforming across neighboring bricks.
fn brick_tets(origin: [usize; 3]) -> Vec<[[usize; 3]; 4]> {
    let o = origin;
    let corner = |dx: usize, dy: usize, dz: usize| [o[0] + dx, o[1] + dy, o[2] + dz];
    let a = corner(0, 0, 0);
    let b = corner(1, 0, 0);
    let c = corner(1, 1, 0);
    let d = corner(0, 1, 0);
    let e = corner(0, 0, 1);
    let f = corner(1, 0, 1);
    let g = corner(1, 1, 1);
    let h = corner(0, 1, 1);
    vec![
        [a, b, c, g],
        [a, b, f, g],
        [a, e, f, g],
        [a, e, h, g],
        [a, d, h, g],
        [a, d, c, g],
    ]
}

fn cube_mesh(n: usize) -> Mesh {
    let mut vertices: Vec<Vec<f64>> = Vec::new();
    let mut index: BTreeMap<[usize; 3], usize> = BTreeMap::new();
    for z in 0..=n {
        for y in 0..=n {
            for x in 0..=n {
                index.insert([x, y, z], vertices.len());
                vertices.push(vec![
                    x as f64 / n as f64,
                    y as f64 / n as f64,
                    z as f64 / n as f64,
                ]);
            }
        }
    }
    let id = |p: [usize; 3]| VertexId(index[&p]);
    let mut cells = Vec::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                for tet in brick_tets([x, y, z]) {
                    cells.push(Cell {
                        vertices: tet.map(id).to_vec(),
                    });
                }
            }
        }
    }
    Mesh::new(3, vertices, cells).unwrap()
}

fn is_boundary_vertex(mesh: &Mesh, vertex: VertexId) -> bool {
    let point = &mesh.vertices()[vertex.0];
    point
        .iter()
        .any(|coordinate| *coordinate == 0.0 || *coordinate == 1.0)
}

fn elasticity_plan() -> RealizationPlan {
    let compilation = compile_semantics(ELASTICITY, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Elasticity", "momentum").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let mesh = cube_mesh(2);
    let element = PreparedElement::linear_simplex(3).unwrap();
    let dofs = vector_nodal_dof_map(&mesh, 3).unwrap();

    let mut stored = Vec::new();
    let mut dynamics = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            match input.source {
                InputSourceRequirement::ModelDefinedConstitutive { .. } => dynamics.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        9,
                        format!("gx_f7/isotropic-lame/{}", integral.integral_index),
                        move |evaluation| {
                            let strain = evaluation
                                .values(DerivativeEvaluation::SymmetricGradient)
                                .expect("active symmetric gradient");
                            stress(LAMBDA, MU, strain)
                        },
                        move |_evaluation, direction_evaluation| {
                            let d_strain = direction_evaluation
                                .values(DerivativeEvaluation::SymmetricGradient)
                                .expect("active symmetric gradient direction");
                            stress(LAMBDA, MU, d_strain)
                        },
                    )
                    .unwrap(),
                ),
                _ => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        3,
                        &mesh,
                        &element,
                        |_, _| vec![0.0, 0.0, 0.0],
                    )
                    .unwrap(),
                ),
            }
        }
    }

    let boundary: Vec<VertexId> = (0..mesh.vertices().len())
        .map(VertexId)
        .filter(|vertex| is_boundary_vertex(&mesh, *vertex))
        .collect();
    let constraints = ConstraintSet::new(
        3 * mesh.vertices().len(),
        boundary.iter().flat_map(|vertex| {
            (0..3).map(move |component| AffineConstraint {
                target: DofId(vertex.0 * 3 + component),
                dependencies: Vec::new(),
                offset: 0.0,
            })
        }),
    )
    .unwrap();

    RealizationPlan::new_stateful(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        stored,
        dynamics,
    )
    .unwrap()
}

fn nonlinear_plan() -> RealizationPlan {
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let subdivisions = 2;
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
    let constraints = (0..width * width)
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
    let constraints = ConstraintSet::new(width * width, constraints).unwrap();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = &model.symbols[input.binding.symbol.index()].name;
            match name.as_str() {
                "capacity" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        "capacity=1;direction=0/v1",
                        |_| vec![1.0],
                        |_, _| vec![0.0],
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
fn vjp_matches_jvp_adjoint_identity_for_scalar_poisson() {
    let plan = poisson_plan(false);
    let dimension = plan.dimension();
    let state = vec![0.0; dimension];
    let rate = vec![0.0; dimension];
    let direction = probe_vector(dimension, 0.7, 0.9);
    let adjoint = probe_vector(dimension, 3.1, 1.3);
    assert_adjoint_identity(&plan, 0.0, &state, &rate, &direction, &adjoint);
}

#[test]
fn vjp_matches_jvp_adjoint_identity_for_vector_elasticity() {
    let plan = elasticity_plan();
    let dimension = plan.dimension();
    let state = vec![0.0; dimension];
    let rate = vec![0.0; dimension];
    let direction = probe_vector(dimension, 0.21, 0.6);
    let adjoint = probe_vector(dimension, 4.4, 0.8);
    assert_adjoint_identity(&plan, 0.0, &state, &rate, &direction, &adjoint);
}

#[test]
fn vjp_matches_jvp_adjoint_identity_for_nonlinear_dynamic_input() {
    let plan = nonlinear_plan();
    let dimension = plan.dimension();
    // A nonzero linearization state so `k = 1 + 0.2u`'s dynamic chain-rule term is nonzero.
    let state = probe_vector(dimension, 1.9, 0.35);
    let rate = probe_vector(dimension, 5.2, -0.17);
    let direction = probe_vector(dimension, 0.4, 0.73);
    let adjoint = probe_vector(dimension, 2.6, -0.41);
    assert_adjoint_identity(&plan, 0.3, &state, &rate, &direction, &adjoint);
}

#[test]
fn vjp_agrees_with_assembled_transpose() {
    let plan = poisson_plan(false);
    let dimension = plan.dimension();
    let assembled = plan.assemble().unwrap();
    let transpose = assembled.transpose().unwrap();
    let context = EvaluationContext::reproducible();
    let adjoint = probe_vector(dimension, 1.1, 1.0);
    let mut expected = vec![0.0; dimension];
    transpose.apply(&context, &adjoint, &mut expected).unwrap();
    let zero = vec![0.0; dimension];
    let mut actual = vec![0.0; dimension];
    plan.vector_jacobian_product(0.0, &zero, &zero, &adjoint, &mut actual)
        .unwrap();
    for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
        let scale = expected.abs().max(actual.abs()).max(1.0);
        assert!(
            (expected - actual).abs() <= 1.0e-12 * scale,
            "row {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn vjp_passes_check_global_transpose() {
    let plan = poisson_plan(false);
    let dimension = plan.dimension();
    let forward = plan.matrix_free();
    let transpose = VjpAtZeroState { plan: &plan };
    let left = probe_vector(dimension, 0.5, 1.0);
    let right = probe_vector(dimension, 2.3, 1.0);
    let subject =
        VerificationSubject::from_serializable("gx-f7-poisson-vjp-transpose", &"gx-f7").unwrap();
    let tolerance = ComparisonTolerance {
        absolute: 1.0e-9,
        relative: 1.0e-9,
    };
    let report = check_global_transpose(
        subject.clone(),
        &forward,
        &transpose,
        &left,
        &right,
        tolerance,
    )
    .unwrap();
    assert!(
        report
            .validate(subject, &forward, &transpose)
            .unwrap()
            .accepted
    );
}

#[test]
fn vjp_refuses_affine_dependency_constraints() {
    let plan = poisson_plan(true);
    let dimension = plan.dimension();
    let zero = vec![0.0; dimension];
    let adjoint = probe_vector(dimension, 0.9, 1.0);
    let mut output = vec![0.0; dimension];
    let error = plan
        .vector_jacobian_product(0.0, &zero, &zero, &adjoint, &mut output)
        .unwrap_err();
    assert!(matches!(
        error,
        finitum::FinitumError::UnsupportedRealization(_)
    ));
}

#[test]
fn capability_reports_vjp_only_when_executable() {
    let executable = poisson_plan(false).capability();
    assert!(
        executable
            .derivative_products
            .contains(&DerivativeProduct::Vjp)
    );

    let with_hanging_constraint = poisson_plan(true).capability();
    assert!(
        !with_hanging_constraint
            .derivative_products
            .contains(&DerivativeProduct::Vjp)
    );
}
