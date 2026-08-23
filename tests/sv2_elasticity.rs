//! SV2-A production slice: vector H1(order=1) elasticity executes end-to-end
//! through the generated kernels with a dynamically-wired constitutive law.
//!
//! Acceptance evidence:
//! 1. single-tet stiffness against an independent hand formula;
//! 2. load vector against the barycenter rule by hand;
//! 3. C^0 patch test on a structured tet grid;
//! 4. rigid-rotation nullspace about the clamped vertex;
//! 5. Jacobian equals centered differences of the residual;
//! 6. trigonometric MMS converges at second order in L2.

use finitum::{
    DynamicExternalInput, ExternalInput, Mesh, PreparedElement, RealizationPlan, VertexId,
    vector_nodal_dof_map,
};
use methodus::LinearOperator as _;
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels,
};
use std::collections::BTreeMap;

const ELASTICITY: &str = r#"
module sv2a.elasticity;
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

const LAMBDA: f64 = 1.25;
const MU: f64 = 1.0;
const PI: f64 = std::f64::consts::PI;

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

/// Kuhn decomposition of one axis-aligned brick into six tets sharing the
/// main diagonal; face-conforming across neighboring bricks.
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

/// Structured grid of bricks on the unit cube split by Kuhn tets.
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
                    cells.push(finitum::Cell {
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

struct Compiled {
    requirements: scientia::FormRequirements,
    factorization: scientia::OperatorFactorization,
    kernels: scientia::StructuredOperatorKernels,
}

fn compile() -> Compiled {
    let compilation = compile_semantics(ELASTICITY, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Elasticity", "momentum").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    Compiled {
        requirements,
        factorization,
        kernels,
    }
}

fn plan_with(
    mesh: &Mesh,
    compiled: &Compiled,
    constraints: finitum::ConstraintSet,
    body_force: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
) -> RealizationPlan {
    let element = PreparedElement::linear_simplex(3).unwrap();
    let dofs = vector_nodal_dof_map(mesh, 3).unwrap();

    let mut stored = Vec::new();
    let mut dynamics = Vec::new();
    for integral in &compiled.factorization.integrals {
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
                        format!("sv2a/isotropic-lame/{}", integral.integral_index),
                        move |evaluation| {
                            let strain = evaluation
                                .values(scientia::DerivativeEvaluation::SymmetricGradient)
                                .expect("active symmetric gradient");
                            stress(LAMBDA, MU, strain)
                        },
                        move |_evaluation, direction_evaluation| {
                            let d_strain = direction_evaluation
                                .values(scientia::DerivativeEvaluation::SymmetricGradient)
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
                        mesh,
                        &element,
                        |_, point| body_force(point),
                    )
                    .unwrap(),
                ),
            }
        }
    }

    RealizationPlan::new_stateful(
        compiled.requirements.clone(),
        compiled.factorization.clone(),
        compiled.kernels.clone(),
        mesh.clone(),
        element,
        dofs,
        constraints,
        stored,
        dynamics,
    )
    .unwrap()
}

fn solve(plan: &RealizationPlan) -> Vec<f64> {
    let dimension = plan.dimension();
    let right_hand_side = plan.load_vector().unwrap();
    let solver = methodus::ConjugateGradientConfig {
        max_iterations: dimension * 8,
        absolute_tolerance: 1.0e-13,
        relative_tolerance: 1.0e-11,
        symmetry_policy: methodus::ConjugateGradientSymmetryPolicy::AssumeSymmetric,
    };
    let solution = methodus::solve_conjugate_gradient(
        &plan.matrix_free(),
        None,
        &methodus::EvaluationContext::reproducible(),
        &right_hand_side,
        &vec![0.0; dimension],
        &solver,
    )
    .unwrap();
    assert!(solution.converged);
    solution.solution
}

#[test]
fn load_vector_matches_hand_for_single_tet() {
    let compiled = compile();
    let mesh = Mesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0],
            vec![0.1, 0.3, 1.0],
        ],
        vec![finitum::Cell {
            vertices: vec![
                finitum::VertexId(0),
                finitum::VertexId(1),
                finitum::VertexId(2),
                finitum::VertexId(3),
            ],
        }],
    )
    .unwrap();
    let constraints = finitum::ConstraintSet::new(
        12,
        std::iter::once(finitum::AffineConstraint {
            target: finitum::DofId(0),
            dependencies: Vec::new(),
            offset: 0.0,
        }),
    )
    .unwrap();
    const FORCE: [f64; 3] = [1.0, -2.0, 3.0];
    fn constant_force(_point: &[f64]) -> Vec<f64> {
        FORCE.to_vec()
    }
    let plan = plan_with(&mesh, &compiled, constraints, Box::new(constant_force));
    let rhs = plan.load_vector().unwrap();
    // Volume = |det|/6 = 1/6 here; barycenter rule gives
    // row(i,c) = V * phi_i(center) * f_c = V/4 * f_c for free nodes.
    let expected = (1.0 / 6.0) * 0.25;
    for node in 1..4 {
        for (component, force_component) in FORCE.iter().enumerate() {
            let slot = node * 3 + component;
            assert!(
                (rhs[slot] - expected * force_component).abs() < 1.0e-13,
                "load row {slot}: {} vs {}",
                rhs[slot],
                expected * force_component
            );
        }
    }
}

#[test]
fn single_tet_stiffness_matches_hand_formula() {
    let compiled = compile();
    let mesh = Mesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.2, 1.0, 0.0],
            vec![0.1, 0.3, 1.0],
        ],
        vec![finitum::Cell {
            vertices: vec![
                finitum::VertexId(0),
                finitum::VertexId(1),
                finitum::VertexId(2),
                finitum::VertexId(3),
            ],
        }],
    )
    .unwrap();
    let constraints = finitum::ConstraintSet::new(
        12,
        std::iter::once(finitum::AffineConstraint {
            target: finitum::DofId(0),
            dependencies: Vec::new(),
            offset: 0.0,
        }),
    )
    .unwrap();
    fn zero_force(_point: &[f64]) -> Vec<f64> {
        vec![0.0; 3]
    }
    let plan = plan_with(&mesh, &compiled, constraints, Box::new(zero_force));
    let assembled = plan.assemble().unwrap();

    // Reference gradients are rows of B^{-1}; g_0 = -sum of the others.
    let p0 = mesh.vertices()[0].clone();
    let cols: Vec<[f64; 3]> = (1..4)
        .map(|j| {
            let point = &mesh.vertices()[j];
            [point[0] - p0[0], point[1] - p0[1], point[2] - p0[2]]
        })
        .collect();
    let m = [
        [cols[0][0], cols[0][1], cols[0][2]],
        [cols[1][0], cols[1][1], cols[1][2]],
        [cols[2][0], cols[2][1], cols[2][2]],
    ];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut binv = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let a = m[(i + 1) % 3][(j + 1) % 3];
            let b = m[(i + 1) % 3][(j + 2) % 3];
            let c = m[(i + 2) % 3][(j + 1) % 3];
            let d = m[(i + 2) % 3][(j + 2) % 3];
            binv[i][j] = (a * d - b * c) / det;
        }
    }
    let volume = det.abs() / 6.0;
    let mut grads: Vec<[f64; 3]> = (0..4)
        .map(|j| {
            if j == 0 {
                [0.0; 3]
            } else {
                [binv[j - 1][0], binv[j - 1][1], binv[j - 1][2]]
            }
        })
        .collect();
    grads[0] = [
        -(grads[1][0] + grads[2][0] + grads[3][0]),
        -(grads[1][1] + grads[2][1] + grads[3][1]),
        -(grads[1][2] + grads[2][2] + grads[3][2]),
    ];

    // K(i,c ; j,d) = V * sigma(phi_j e_d) : grad(phi_i e_c).
    let hand = |i: usize, ci: usize, j: usize, cj: usize| -> f64 {
        let mut eps = vec![0.0; 9];
        for a in 0..3 {
            for b in 0..3 {
                eps[a * 3 + b] = 0.5
                    * (grads[j][a] * (cj == b) as i32 as f64
                        + grads[j][b] * (cj == a) as i32 as f64);
            }
        }
        let sigma = stress(LAMBDA, MU, &eps);
        let mut contraction = 0.0;
        for a in 0..3 {
            contraction += sigma[a * 3 + ci] * grads[i][a];
        }
        contraction * volume
    };

    let matrix = assembled.matrix();
    let get = |row: usize, col: usize| -> Option<f64> {
        let start = matrix.row_offsets()[row];
        let end = matrix.row_offsets()[row + 1];
        (start..end)
            .find(|slot| matrix.column_indices()[*slot] == col)
            .map(|slot| matrix.values()[slot])
    };

    let mut fitted: Option<(f64, f64)> = None;
    'outer: for probe_row in 3..12usize {
        for probe_col in 3..12usize {
            if let Some(value) = get(probe_row, probe_col) {
                if value != 0.0 {
                    let reference =
                        hand(probe_row / 3, probe_row % 3, probe_col / 3, probe_col % 3);
                    if reference.abs() > 1.0e-14 {
                        fitted = Some((value, reference));
                        break 'outer;
                    }
                }
            }
        }
    }
    let (assembled_value, hand_value) = fitted.expect("a free nonzero entry");
    let scale = assembled_value / hand_value;
    let constrained = [0_usize];
    for row in 0..12usize {
        for col in 0..12usize {
            if constrained.contains(&row) || constrained.contains(&col) {
                continue;
            }
            if let Some(value) = get(row, col) {
                let expected = scale * hand(row / 3, row % 3, col / 3, col % 3);
                assert!(
                    (value - expected).abs() <= 1.0e-11 * (1.0 + value.abs()),
                    "({row},{col}): assembled {value} vs scaled-hand {expected}"
                );
            }
        }
    }
}

#[test]
fn elasticity_patch_test_recovers_the_linear_field_exactly() {
    let compiled = compile();
    let mesh = cube_mesh(2);
    let boundary: Vec<VertexId> = (0..mesh.vertices().len())
        .map(VertexId)
        .filter(|vertex| is_boundary_vertex(&mesh, *vertex))
        .collect();

    let displacement = |point: &[f64]| -> [f64; 3] {
        [
            0.7 * point[0] - 0.2 * point[1] + 0.1,
            0.3 * point[1] + 0.5 * point[2] - 0.05,
            -0.4 * point[0] + 0.25 * point[2] + 0.02,
        ]
    };
    let constraints = finitum::ConstraintSet::new(
        3 * mesh.vertices().len(),
        boundary.iter().flat_map(|vertex| {
            let value = displacement(&mesh.vertices()[vertex.0]);
            (0..3).map(move |component| finitum::AffineConstraint {
                target: finitum::DofId(vertex.0 * 3 + component),
                dependencies: Vec::new(),
                offset: value[component],
            })
        }),
    )
    .unwrap();
    fn zero_force(_point: &[f64]) -> Vec<f64> {
        vec![0.0; 3]
    }
    let plan = plan_with(&mesh, &compiled, constraints, Box::new(zero_force));
    let solution = solve(&plan);

    for (index, point) in mesh.vertices().iter().enumerate() {
        let expected = displacement(point);
        for component in 0..3 {
            let actual = solution[index * 3 + component];
            assert!(
                (actual - expected[component]).abs() < 1.0e-11,
                "node {index} component {component}: {actual} != {}",
                expected[component]
            );
        }
    }
}

#[test]
fn rotations_about_the_clamped_vertex_are_exact_nullspace_modes() {
    let compiled = compile();
    let mesh = Mesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![finitum::Cell {
            vertices: vec![
                finitum::VertexId(0),
                finitum::VertexId(1),
                finitum::VertexId(2),
                finitum::VertexId(3),
            ],
        }],
    )
    .unwrap();
    let constraints = finitum::ConstraintSet::new(
        12,
        (0..3).map(|component| finitum::AffineConstraint {
            target: finitum::DofId(component),
            dependencies: Vec::new(),
            offset: 0.0,
        }),
    )
    .unwrap();
    fn zero_force(_point: &[f64]) -> Vec<f64> {
        vec![0.0; 3]
    }
    let plan = plan_with(&mesh, &compiled, constraints, Box::new(zero_force));
    let assembled = plan.assemble().unwrap();
    let context = methodus::EvaluationContext::reproducible();

    for omega in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        let mut mode = vec![0.0; plan.dimension()];
        for (index, point) in mesh.vertices().iter().enumerate() {
            for component in 0..3 {
                mode[index * 3 + component] = omega[(component + 1) % 3]
                    * point[(component + 2) % 3]
                    - omega[(component + 2) % 3] * point[(component + 1) % 3];
            }
        }
        let mut action = vec![0.0; plan.dimension()];
        assembled.apply(&context, &mode, &mut action).unwrap();
        let norm = action
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.abs()));
        assert!(
            norm < 1.0e-12,
            "rotation {omega:?} produced stiffness action {norm}"
        );
    }
}

#[test]
fn jacobian_matches_residual_finite_differences_on_elasticity() {
    use methodus::LinearOperator as _;
    let compiled = compile();
    let mesh = cube_mesh(2);
    let constraints = finitum::ConstraintSet::new(
        3 * mesh.vertices().len(),
        (0..mesh.vertices().len())
            .filter(|index| is_boundary_vertex(&mesh, VertexId(*index)))
            .flat_map(|vertex| {
                (0..3).map(move |component| finitum::AffineConstraint {
                    target: finitum::DofId(vertex * 3 + component),
                    dependencies: Vec::new(),
                    offset: 0.0,
                })
            }),
    )
    .unwrap();
    fn zero_force(_point: &[f64]) -> Vec<f64> {
        vec![0.0; 3]
    }
    let plan = plan_with(&mesh, &compiled, constraints, Box::new(zero_force));
    let dimension = plan.dimension();
    let step = 1.0e-6;
    let mut worst = 0.0_f64;
    let context = methodus::EvaluationContext::reproducible();
    for k in (0..dimension).step_by(7) {
        let mut plus_state = vec![0.0; dimension];
        let mut minus_state = vec![0.0; dimension];
        plus_state[k] = step;
        minus_state[k] = -step;
        let mut forward = vec![0.0; dimension];
        let mut backward = vec![0.0; dimension];
        plan.residual(0.0, &plus_state, &plus_state, &mut forward)
            .unwrap();
        plan.residual(0.0, &minus_state, &minus_state, &mut backward)
            .unwrap();
        let mut analytic = vec![0.0; dimension];
        let mut unit = vec![0.0; dimension];
        unit[k] = 1.0;
        plan.matrix_free()
            .apply(&context, &unit, &mut analytic)
            .unwrap();
        for slot in 0..dimension {
            let finite_difference = (forward[slot] - backward[slot]) / (2.0 * step);
            worst = worst.max((finite_difference - analytic[slot]).abs());
        }
    }
    assert!(worst < 1.0e-8, "JVP vs residual FD max discrepancy {worst}");
}

#[test]
fn manufactured_displacement_converges_second_order() {
    let compiled = compile();
    fn phi(point: &[f64]) -> f64 {
        (PI * point[0]).sin() * (PI * point[1]).sin() * (PI * point[2]).sin()
    }
    let exact = |point: &[f64]| -> [f64; 3] {
        let scale = phi(point);
        [scale, scale, scale]
    };
    // u_i = phi: f_i = -mu lap(phi) - (lambda+mu) d_i(div u).
    // d_i phi = pi cos(pi x_i) prod_{m!=i} sin(pi x_m).
    // lap(phi) = -3 pi^2 phi.
    // d_i(div u) = -pi^2 phi + sum_{k!=i} pi^2 cos_i cos_k prod_{m not i,k} sin_m.
    let body_force = |point: &[f64]| -> Vec<f64> {
        let sines: Vec<f64> = (0..3).map(|axis| (PI * point[axis]).sin()).collect();
        let cosines: Vec<f64> = (0..3).map(|axis| (PI * point[axis]).cos()).collect();
        let lap_phi = -3.0 * PI * PI * phi(point);
        let mut force = [0.0_f64; 3];
        for i in 0..3 {
            let mut coupling = -PI * PI * phi(point);
            for k in 0..3 {
                if k == i {
                    continue;
                }
                let mixed: f64 = (0..3)
                    .filter(|m| m != &i && m != &k)
                    .map(|m| sines[m])
                    .product::<f64>();
                coupling += PI * PI * cosines[i] * cosines[k] * mixed;
            }
            force[i] = -(MU * lap_phi + (LAMBDA + MU) * coupling);
        }
        force.to_vec()
    };

    let mut errors = Vec::new();
    for n in [3_usize, 6] {
        let mesh = cube_mesh(n);
        let boundary: Vec<VertexId> = (0..mesh.vertices().len())
            .map(VertexId)
            .filter(|vertex| is_boundary_vertex(&mesh, *vertex))
            .collect();
        let constraints = finitum::ConstraintSet::new(
            3 * mesh.vertices().len(),
            boundary.iter().flat_map(|vertex| {
                let value = exact(&mesh.vertices()[vertex.0]);
                let vertex = vertex.0;
                (0..3).map(move |component| finitum::AffineConstraint {
                    target: finitum::DofId(vertex * 3 + component),
                    dependencies: Vec::new(),
                    offset: value[component],
                })
            }),
        )
        .unwrap();
        let plan = plan_with(&mesh, &compiled, constraints, Box::new(body_force));
        let solution = solve(&plan);
        let dimension = solution.len();
        let error = (solution
            .chunks(3)
            .enumerate()
            .map(|(node, values)| {
                let expected = exact(&mesh.vertices()[node]);
                values
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / dimension as f64)
            .sqrt();
        errors.push(error);
    }
    let rate = errors[1] / errors[0];
    assert!(
        (0.15..0.55).contains(&rate) && errors[1] < 2.0e-2,
        "second-order L2 convergence expected, observed ratio {rate} ({errors:?})"
    );
}

#[test]
fn analytic_body_force_matches_numeric_divergence() {
    fn phi(point: &[f64]) -> f64 {
        (PI * point[0]).sin() * (PI * point[1]).sin() * (PI * point[2]).sin()
    }
    let displacement = |point: &[f64]| -> [f64; 3] {
        let scale = phi(point);
        [scale, scale, scale]
    };
    let sigma_at = |point: &[f64]| -> [f64; 9] {
        let h = 1.0e-5;
        let mut columns = [[0.0_f64; 3]; 3];
        for axis in 0..3 {
            let mut plus = point.to_vec();
            plus[axis] += h;
            let mut minus = point.to_vec();
            minus[axis] -= h;
            let up = displacement(&plus);
            let um = displacement(&minus);
            for comp in 0..3 {
                columns[comp][axis] = (up[comp] - um[comp]) / (2.0 * h);
            }
        }
        let mut eps = [0.0_f64; 9];
        for a in 0..3 {
            for b in 0..3 {
                eps[a * 3 + b] = 0.5 * (columns[a][b] + columns[b][a]);
            }
        }
        let mut trace = 0.0;
        for a in 0..3 {
            trace += eps[a * 3 + a];
        }
        let mut out = [0.0_f64; 9];
        for a in 0..3 {
            for b in 0..3 {
                out[a * 3 + b] =
                    2.0 * MU * eps[a * 3 + b] + if a == b { LAMBDA * trace } else { 0.0 };
            }
        }
        out
    };
    let point = [0.31, 0.42, 0.53];
    let h = 1.0e-4;
    let _sigma = sigma_at(&point);
    let mut numeric_force = [0.0_f64; 3];
    for i in 0..3 {
        let mut plus_point = point.to_vec();
        plus_point[i] += h;
        let mut minus_point = point.to_vec();
        minus_point[i] -= h;
        let sp = sigma_at(&plus_point);
        let sm = sigma_at(&minus_point);
        for c in 0..3 {
            numeric_force[c] -= (sp[i * 3 + c] - sm[i * 3 + c]) / (2.0 * h);
        }
    }
    // Authored formula, identical to the MMS closure.
    let sines: Vec<f64> = (0..3).map(|axis| (PI * point[axis]).sin()).collect();
    let cosines: Vec<f64> = (0..3).map(|axis| (PI * point[axis]).cos()).collect();
    let lap_phi = -3.0 * PI * PI * phi(&point);
    let mut authored_force = [0.0_f64; 3];
    for i in 0..3 {
        let mut coupling = -PI * PI * phi(&point);
        for k in 0..3 {
            if k == i {
                continue;
            }
            let mixed: f64 = (0..3)
                .filter(|m| m != &i && m != &k)
                .map(|m| sines[m])
                .product::<f64>();
            coupling += PI * PI * cosines[i] * cosines[k] * mixed;
        }
        authored_force[i] = -(MU * lap_phi + (LAMBDA + MU) * coupling);
    }
    for axis in 0..3 {
        assert!(
            (authored_force[axis] - numeric_force[axis]).abs() < 1.0e-5,
            "force axis {axis}: authored {} vs numeric {}",
            authored_force[axis],
            numeric_force[axis]
        );
    }
}
