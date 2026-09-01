//! SV2-B1: P2 simplex elements alongside P1 -- basis/gradient identities, a P2 executable in
//! `RealizationPlan` (Poisson manufactured-solution convergence, the decisive fixture), and
//! typed refusals for unsupported element requests.

use finitum::{
    Cell, ConstraintSet, DofId, ExternalInput, FinitumError, Mesh, PreparedElement,
    RealizationPlan, VertexId, quadratic_simplex_dof_map, quadratic_simplex_node_points,
    simplex_basis,
};
use methodus::{
    ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, EvaluationContext,
    solve_conjugate_gradient,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels,
};
use std::f64::consts::PI;

const POISSON: &str = r#"
module sv2b1.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=2) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

#[test]
fn p2_basis_is_a_partition_of_unity_with_a_consistent_zero_gradient_sum() {
    for dimension in 1..=3 {
        let element = PreparedElement::quadratic_simplex(dimension).unwrap();
        for point in 0..element.quadrature().len() {
            let mut value_sum = 0.0;
            let mut gradient_sum = vec![0.0; dimension];
            for basis in 0..element.basis_count() {
                value_sum += element.basis_value(point, basis).unwrap();
                for (axis, component) in element
                    .basis_gradient(point, basis)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    gradient_sum[axis] += component;
                }
            }
            assert!(
                (value_sum - 1.0).abs() < 1.0e-12,
                "dimension {dimension} point {point}: basis values sum to {value_sum}, not 1"
            );
            for (axis, sum) in gradient_sum.iter().enumerate() {
                assert!(
                    sum.abs() < 1.0e-11,
                    "dimension {dimension} point {point} axis {axis}: gradient sum {sum}, not 0"
                );
            }
        }
    }
}

#[test]
fn p2_basis_is_kronecker_delta_nodal_and_gradients_match_finite_differences() {
    for dimension in 1..=3usize {
        let node_count = (dimension + 1) * (dimension + 2) / 2;
        let nodes = reference_nodes(dimension);
        assert_eq!(nodes.len(), node_count);
        for (node_index, node) in nodes.iter().enumerate() {
            let (values, _) = simplex_basis(dimension, 2, node).unwrap();
            for (basis, value) in values.iter().enumerate() {
                let expected = if basis == node_index { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1.0e-12,
                    "dimension {dimension}: basis {basis} at node {node_index} is {value}, \
                     expected {expected}"
                );
            }
        }

        // Central-difference gradient check at a handful of interior points away from any node.
        let probes = interior_probes(dimension);
        let step = 1.0e-6;
        for probe in probes {
            let (_, gradients) = simplex_basis(dimension, 2, &probe).unwrap();
            for (basis, gradient) in gradients.iter().enumerate() {
                for axis in 0..dimension {
                    let mut forward = probe.clone();
                    forward[axis] += step;
                    let mut backward = probe.clone();
                    backward[axis] -= step;
                    let (forward_values, _) = simplex_basis(dimension, 2, &forward).unwrap();
                    let (backward_values, _) = simplex_basis(dimension, 2, &backward).unwrap();
                    let numeric = (forward_values[basis] - backward_values[basis]) / (2.0 * step);
                    assert!(
                        (numeric - gradient[axis]).abs() < 1.0e-6,
                        "dimension {dimension} basis {basis} axis {axis}: analytic gradient \
                         {}, numeric {numeric}",
                        gradient[axis]
                    );
                }
            }
        }
    }
}

#[test]
fn p2_element_and_dof_map_refuse_malformed_shapes() {
    assert_eq!(
        PreparedElement::quadratic_simplex(0).unwrap_err(),
        FinitumError::InvalidDimension(0)
    );
    assert_eq!(
        PreparedElement::quadratic_simplex(4).unwrap_err(),
        FinitumError::InvalidDimension(4)
    );
    assert!(matches!(
        simplex_basis(2, 3, &[0.25, 0.25]).unwrap_err(),
        FinitumError::InvalidElementShape(_)
    ));
    assert!(matches!(
        simplex_basis(2, 2, &[0.25]).unwrap_err(),
        FinitumError::InvalidElementShape(_)
    ));

    let mesh = Mesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
        }],
    )
    .unwrap();
    assert!(matches!(
        quadratic_simplex_dof_map(&mesh, 0).unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
}

#[test]
fn p2_dof_map_has_no_overlap_or_gap_and_orders_deterministically() {
    let (mesh, _, _, _) = square_mesh_p2(3);
    let dofs = quadratic_simplex_dof_map(&mesh, 1).unwrap();
    let vertex_count = mesh.vertices().len();
    // Every DOF must be referenced by at least one restriction (no gap), and building the map
    // twice must produce byte-identical restrictions (determinism).
    let mut referenced = vec![false; dofs.dof_count()];
    for restriction in dofs.restrictions() {
        for dof in &restriction.dofs {
            referenced[dof.0] = true;
        }
    }
    assert!(
        referenced.iter().all(|seen| *seen),
        "every DOF must be reachable from some cell"
    );
    assert_eq!(dofs.dof_count(), vertex_count + expected_edge_count(&mesh));

    let repeated = quadratic_simplex_dof_map(&mesh, 1).unwrap();
    assert_eq!(dofs.restrictions(), repeated.restrictions());

    // Vector case: no overlap between distinct nodes' component ranges, and every restriction's
    // local length matches the P2 basis count times components.
    let vector_dofs = quadratic_simplex_dof_map(&mesh, 2).unwrap();
    assert_eq!(vector_dofs.dof_count(), 2 * dofs.dof_count());
    for restriction in vector_dofs.restrictions() {
        assert_eq!(restriction.dofs.len(), 6 * 2);
        let mut seen = std::collections::BTreeSet::new();
        for dof in &restriction.dofs {
            assert!(seen.insert(dof.0), "restriction repeats DOF {}", dof.0);
        }
    }
}

/// P2 Poisson manufactured solution `u = sin(pi x) sin(pi y)` on the unit square, executed
/// end-to-end through the single-field `RealizationPlan` (the SV2-B1 unlock: `validate_
/// discretization` now admits `polynomial_order == 2`). `u` vanishes on every boundary DOF
/// (vertex and edge-midpoint alike), so every constraint is homogeneous.
#[test]
fn p2_poisson_converges_at_order_three_in_l2_on_a_manufactured_solution() {
    let mut errors = Vec::new();
    let mut mesh_sizes = Vec::new();
    for subdivisions in [2usize, 4, 8] {
        let (mesh, dofs, constraints, node_points) = square_mesh_p2(subdivisions);
        let element = PreparedElement::quadratic_simplex(2).unwrap();

        let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
        let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
        let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
        let factorization = factor_operator(&form, &requirements).unwrap();
        let kernels = lower_operator_kernels(&factorization).unwrap();
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
                        ExternalInput::sampled(
                            integral.integral_index,
                            input.id,
                            1,
                            &mesh,
                            &element,
                            move |_, physical| match name.as_str() {
                                "k" => vec![1.0],
                                "f" => vec![manufactured_source(physical)],
                                other => panic!("unexpected external input {other}"),
                            },
                        )
                        .unwrap()
                    })
            })
            .collect();
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

        let matrix_free = plan.matrix_free();
        let right_hand_side = plan.load_vector().unwrap();
        let context = EvaluationContext::reproducible();
        let report = solve_conjugate_gradient(
            &matrix_free,
            None,
            &context,
            &right_hand_side,
            &vec![0.0; plan.dimension()],
            &cg_config_assuming_symmetry(),
        )
        .unwrap();
        assert!(report.converged);

        let exact = node_points
            .iter()
            .map(|point| manufactured_exact(point))
            .collect::<Vec<_>>();
        let squared_error = report
            .solution
            .iter()
            .zip(&exact)
            .map(|(computed, exact)| (computed - exact).powi(2))
            .sum::<f64>()
            / report.solution.len() as f64;
        errors.push(squared_error.sqrt());
        mesh_sizes.push(1.0 / subdivisions as f64);
    }

    // Measured convergence order between successive refinements; P2 targets order ~3 in L2.
    for window in errors.windows(2).zip(mesh_sizes.windows(2)) {
        let (error_pair, size_pair) = window;
        let order = (error_pair[0] / error_pair[1]).ln() / (size_pair[0] / size_pair[1]).ln();
        assert!(
            order > 2.5,
            "measured L2 convergence order {order} from errors {error_pair:?} at sizes \
             {size_pair:?}, expected order roughly 3"
        );
    }
}

fn manufactured_exact(point: &[f64]) -> f64 {
    (PI * point[0]).sin() * (PI * point[1]).sin()
}

fn manufactured_source(point: &[f64]) -> f64 {
    2.0 * PI * PI * (PI * point[0]).sin() * (PI * point[1]).sin()
}

fn reference_nodes(dimension: usize) -> Vec<Vec<f64>> {
    let mut nodes = Vec::new();
    let mut vertices = vec![vec![0.0; dimension]; dimension + 1];
    for (index, vertex) in vertices.iter_mut().enumerate().skip(1) {
        vertex[index - 1] = 1.0;
    }
    nodes.extend(vertices.iter().cloned());
    for left in 0..vertices.len() {
        for right in left + 1..vertices.len() {
            nodes.push(
                vertices[left]
                    .iter()
                    .zip(&vertices[right])
                    .map(|(a, b)| 0.5 * (a + b))
                    .collect(),
            );
        }
    }
    nodes
}

fn interior_probes(dimension: usize) -> Vec<Vec<f64>> {
    match dimension {
        1 => vec![vec![0.2], vec![0.55], vec![0.83]],
        2 => vec![vec![0.15, 0.2], vec![0.4, 0.35], vec![0.1, 0.7]],
        3 => vec![
            vec![0.15, 0.2, 0.3],
            vec![0.2, 0.2, 0.2],
            vec![0.1, 0.3, 0.4],
        ],
        _ => unreachable!(),
    }
}

fn expected_edge_count(mesh: &Mesh) -> usize {
    let mut edges = std::collections::BTreeSet::new();
    for cell in mesh.cells() {
        for left in 0..cell.vertices.len() {
            for right in left + 1..cell.vertices.len() {
                let mut pair = [cell.vertices[left].0, cell.vertices[right].0];
                pair.sort_unstable();
                edges.insert(pair);
            }
        }
    }
    edges.len()
}

/// Structured unit-square P2 mesh (two triangles per grid cell): the mesh, its scalar P2 DOF
/// map, homogeneous-Dirichlet boundary constraints on every boundary DOF (vertex and
/// edge-midpoint), and each DOF's physical node point.
fn square_mesh_p2(subdivisions: usize) -> (Mesh, finitum::DofMap, ConstraintSet, Vec<Vec<f64>>) {
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
    let mesh = Mesh::new(2, vertices, cells).unwrap();
    let dofs = quadratic_simplex_dof_map(&mesh, 1).unwrap();
    let node_points = quadratic_simplex_node_points(&mesh);
    let boundary = |point: &[f64]| {
        point[0] <= 1.0e-12
            || point[1] <= 1.0e-12
            || point[0] >= 1.0 - 1.0e-12
            || point[1] >= 1.0 - 1.0e-12
    };
    let constraints = ConstraintSet::new(
        dofs.dof_count(),
        (0..dofs.dof_count())
            .filter(|index| boundary(&node_points[*index]))
            .map(|target| finitum::AffineConstraint {
                target: DofId(target),
                dependencies: Vec::new(),
                offset: 0.0,
            }),
    )
    .unwrap();
    (mesh, dofs, constraints, node_points)
}

fn cg_config_assuming_symmetry() -> ConjugateGradientConfig {
    ConjugateGradientConfig {
        symmetry_policy: ConjugateGradientSymmetryPolicy::AssumeSymmetric,
        ..ConjugateGradientConfig::default()
    }
}
