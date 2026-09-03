//! SC-W1 Finitum side, item 3 / SV2-B2: interface and interior-facet measure realization over
//! shared facets, binding Malleus facet-pair kernels through their side roles.
//!
//! Evidence:
//! 1. a P0 jump-penalty kernel over every interior facet assembles to the hand-built
//!    facet-weighted graph Laplacian, annihilates constants, is symmetric, and its JVP/VJP are
//!    exact transposes; flipping every facet's minus/plus choice leaves the global operator
//!    bit-for-bit unchanged, and the Malleus swap receipt passes;
//! 2. a normal-dependent central-flux kernel (`Facet { Odd }` normal) is likewise orientation
//!    invariant, while a deliberately one-sided variant fails the swap receipt and does change
//!    under the flip -- the receipt detects exactly the kernels whose global action would
//!    depend on Finitum's orientation choice;
//! 3. a two-field P1 interface coupling between the left and right halves of a square
//!    assembles to the segment mass matrices `[M -M; -M M]` on the shared interface nodes and
//!    vanishes for equal constants; its one-sided operands are refused by the swap receipt
//!    typed;
//! 4. a gradient-trace consistency kernel on a discontinuous P1 field reproduces the closed
//!    form `-/+ n_x |F| / 2` for `u = x`, exercising the `Gradient` trace, the cell-owned DOF
//!    map, and the normal;
//! 5. binding refusals are typed (side/parity/access/shape mismatches, unknown fields,
//!    exterior facets, unsupported orders) and `InterfaceSpace::over_layout` checks extents.

use finitum::{
    BlockLayout, CellId, FacetTopology, FinitumError, InterfaceKernel, InterfaceMeasure,
    InterfaceOperand, InterfaceOperator, InterfaceSpace, Mesh, MeshProfile, TraceEvaluation,
    TraceFieldSpec, realize,
};
use malleus::{
    AccessMode, AxisId, BinaryOp, FacetOperandRole, FacetPairKernel, FacetSide, IndexExpr,
    IndexingMap, IterationDomain, IteratorKind, KernelOperand, KernelRegion, NumericPolicy,
    OperandId, ReductionOp, ScalarExpr, Statement, StructuredKernel, SwapParity, UnaryOp,
};
use methodus::{EvaluationContext, LinearOperator, TransposableOperator};
use scientia::SymbolId;
use std::collections::BTreeSet;

const DIM: usize = 2;
const U: SymbolId = SymbolId(0);
const A: SymbolId = SymbolId(3);
const B: SymbolId = SymbolId(4);

fn op(index: usize) -> OperandId {
    OperandId::new(index)
}
fn load(index: usize) -> ScalarExpr {
    ScalarExpr::Load(op(index))
}
fn c(value: f64) -> ScalarExpr {
    ScalarExpr::Constant(value)
}
fn mul(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Mul, lhs, rhs)
}
fn add(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Add, lhs, rhs)
}
fn sub(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Sub, lhs, rhs)
}
fn neg(value: ScalarExpr) -> ScalarExpr {
    ScalarExpr::unary(UnaryOp::Neg, value)
}
fn cell(side: FacetSide, partner: Option<usize>) -> FacetOperandRole {
    FacetOperandRole::Cell {
        side,
        partner: partner.map(op),
    }
}
fn facet(parity: SwapParity) -> FacetOperandRole {
    FacetOperandRole::Facet { parity }
}
fn store(index: usize, value: ScalarExpr) -> Statement {
    Statement::Store {
        operand: op(index),
        value,
    }
}

fn unit_square(subdivisions: usize) -> Mesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
    .mesh
}

fn probe(dimension: usize, seed: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn dense(operator: &InterfaceOperator) -> Vec<Vec<f64>> {
    let n = operator.dimension();
    let mut matrix = vec![vec![0.0; n]; n];
    let mut direction = vec![0.0; n];
    let mut output = vec![0.0; n];
    for column in 0..n {
        direction[column] = 1.0;
        operator.apply_action(&direction, &mut output).unwrap();
        for row in 0..n {
            matrix[row][column] = output[row];
        }
        direction[column] = 0.0;
    }
    matrix
}

fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for (column, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{what} at ({row}, {column}): {actual} != {expected}"
            );
        }
    }
}

fn assert_transpose(operator: &InterfaceOperator, state: &[f64], what: &str) {
    let n = operator.dimension();
    let x = probe(n, 3.1);
    let y = probe(n, 7.9);
    let mut jx = vec![0.0; n];
    operator
        .jacobian_vector_product(state, &x, &mut jx)
        .unwrap();
    let mut jty = vec![0.0; n];
    operator
        .vector_jacobian_product(state, &y, &mut jty)
        .unwrap();
    let left = dot(&y, &jx);
    let right = dot(&jty, &x);
    let scale = left.abs().max(right.abs()).max(1.0e-300);
    assert!(
        (left - right).abs() <= 1.0e-12 * scale,
        "{what}: <y, J x> = {left}, <J^T y, x> = {right}"
    );
}

fn facet_length(mesh: &Mesh, topology: &FacetTopology, facet: usize) -> f64 {
    let vertices = &topology.facets()[facet].vertices;
    let a = &mesh.vertices()[vertices[0].0];
    let b = &mesh.vertices()[vertices[1].0];
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

/// `r_minus = eta [u]`, `r_plus = -eta [u]` with `[u] = u_minus - u_plus`: the P0 jump penalty.
fn jump_penalty_kernel(eta: f64) -> FacetPairKernel {
    let jump = sub(load(0), load(1));
    FacetPairKernel {
        kernel: StructuredKernel {
            name: "p0-jump-penalty".into(),
            iteration_domain: IterationDomain::new(vec![1]),
            iterators: vec![IteratorKind::Reduction],
            operands: vec![
                KernelOperand::scalar("u_minus", AccessMode::Read),
                KernelOperand::scalar("u_plus", AccessMode::Read),
                KernelOperand::scalar("r_minus", AccessMode::Reduce(ReductionOp::Add)),
                KernelOperand::scalar("r_plus", AccessMode::Reduce(ReductionOp::Add)),
            ],
            indexing_maps: (0..4).map(|index| IndexingMap::scalar(op(index))).collect(),
            body: KernelRegion {
                statements: vec![
                    store(2, mul(c(eta), jump.clone())),
                    store(3, neg(mul(c(eta), jump))),
                ],
            },
            numeric_policy: NumericPolicy::default(),
        },
        roles: vec![
            cell(FacetSide::Minus, Some(1)),
            cell(FacetSide::Plus, Some(0)),
            cell(FacetSide::Minus, Some(3)),
            cell(FacetSide::Plus, Some(2)),
        ],
    }
}

fn u_value_bindings() -> Vec<InterfaceOperand> {
    vec![
        InterfaceOperand::Trace {
            field: U,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Trace {
            field: U,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Residual {
            field: U,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Residual {
            field: U,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Value,
        },
    ]
}

/// Central flux `r_minus = (beta . n) {u}`, `r_plus = -(beta . n) {u}` (swap covariant), or
/// the one-sided `r_minus = (beta . n) u_minus`, `r_plus = -(beta . n) u_minus` (not
/// covariant) when `one_sided`.
fn flux_kernel(one_sided: bool) -> FacetPairKernel {
    let axis = AxisId::new(0);
    let vector = |index: usize| IndexingMap::new(op(index), vec![IndexExpr::axis(axis)]);
    let beta_n = mul(load(3), load(2));
    let carried = if one_sided {
        load(0)
    } else {
        mul(c(0.5), add(load(0), load(1)))
    };
    FacetPairKernel {
        kernel: StructuredKernel {
            name: if one_sided {
                "one-sided-flux".into()
            } else {
                "central-flux".into()
            },
            iteration_domain: IterationDomain::new(vec![DIM]),
            iterators: vec![IteratorKind::Reduction],
            operands: vec![
                KernelOperand::scalar("u_minus", AccessMode::Read),
                KernelOperand::scalar("u_plus", AccessMode::Read),
                KernelOperand::tensor("normal", vec![DIM], AccessMode::Read),
                KernelOperand::tensor("beta", vec![DIM], AccessMode::Read),
                KernelOperand::scalar("r_minus", AccessMode::Reduce(ReductionOp::Add)),
                KernelOperand::scalar("r_plus", AccessMode::Reduce(ReductionOp::Add)),
            ],
            indexing_maps: vec![
                IndexingMap::scalar(op(0)),
                IndexingMap::scalar(op(1)),
                vector(2),
                vector(3),
                IndexingMap::scalar(op(4)),
                IndexingMap::scalar(op(5)),
            ],
            body: KernelRegion {
                statements: vec![
                    store(4, mul(beta_n.clone(), carried.clone())),
                    store(5, neg(mul(beta_n, carried))),
                ],
            },
            numeric_policy: NumericPolicy::default(),
        },
        roles: vec![
            cell(FacetSide::Minus, Some(1)),
            cell(FacetSide::Plus, Some(0)),
            facet(SwapParity::Odd),
            facet(SwapParity::Even),
            cell(FacetSide::Minus, Some(5)),
            cell(FacetSide::Plus, Some(4)),
        ],
    }
}

fn flux_bindings(beta: [f64; 2]) -> Vec<InterfaceOperand> {
    vec![
        InterfaceOperand::Trace {
            field: U,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Trace {
            field: U,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Normal,
        InterfaceOperand::Constant {
            values: beta.to_vec(),
        },
        InterfaceOperand::Residual {
            field: U,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Residual {
            field: U,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Value,
        },
    ]
}

/// Two-field interface coupling `r_a = a_minus - b_plus`, `r_b = -(a_minus - b_plus)`.
fn two_field_kernel() -> FacetPairKernel {
    let jump = sub(load(0), load(1));
    FacetPairKernel {
        kernel: StructuredKernel {
            name: "two-field-interface".into(),
            iteration_domain: IterationDomain::new(vec![1]),
            iterators: vec![IteratorKind::Reduction],
            operands: vec![
                KernelOperand::scalar("a_minus", AccessMode::Read),
                KernelOperand::scalar("b_plus", AccessMode::Read),
                KernelOperand::scalar("r_a", AccessMode::Reduce(ReductionOp::Add)),
                KernelOperand::scalar("r_b", AccessMode::Reduce(ReductionOp::Add)),
            ],
            indexing_maps: (0..4).map(|index| IndexingMap::scalar(op(index))).collect(),
            body: KernelRegion {
                statements: vec![store(2, jump.clone()), store(3, neg(jump))],
            },
            numeric_policy: NumericPolicy::default(),
        },
        roles: vec![
            cell(FacetSide::Minus, None),
            cell(FacetSide::Plus, None),
            cell(FacetSide::Minus, None),
            cell(FacetSide::Plus, None),
        ],
    }
}

/// Consistency term `r_minus = -{grad u} . n`, `r_plus = +{grad u} . n`.
fn gradient_average_kernel() -> FacetPairKernel {
    let axis = AxisId::new(0);
    let vector = |index: usize| IndexingMap::new(op(index), vec![IndexExpr::axis(axis)]);
    let average_flux = mul(mul(c(0.5), add(load(0), load(1))), load(2));
    FacetPairKernel {
        kernel: StructuredKernel {
            name: "gradient-average".into(),
            iteration_domain: IterationDomain::new(vec![DIM]),
            iterators: vec![IteratorKind::Reduction],
            operands: vec![
                KernelOperand::tensor("grad_minus", vec![DIM], AccessMode::Read),
                KernelOperand::tensor("grad_plus", vec![DIM], AccessMode::Read),
                KernelOperand::tensor("normal", vec![DIM], AccessMode::Read),
                KernelOperand::scalar("r_minus", AccessMode::Reduce(ReductionOp::Add)),
                KernelOperand::scalar("r_plus", AccessMode::Reduce(ReductionOp::Add)),
            ],
            indexing_maps: vec![
                vector(0),
                vector(1),
                vector(2),
                IndexingMap::scalar(op(3)),
                IndexingMap::scalar(op(4)),
            ],
            body: KernelRegion {
                statements: vec![store(3, neg(average_flux.clone())), store(4, average_flux)],
            },
            numeric_policy: NumericPolicy::default(),
        },
        roles: vec![
            cell(FacetSide::Minus, Some(1)),
            cell(FacetSide::Plus, Some(0)),
            facet(SwapParity::Odd),
            cell(FacetSide::Minus, Some(4)),
            cell(FacetSide::Plus, Some(3)),
        ],
    }
}

fn p0_space(mesh: &Mesh) -> InterfaceSpace {
    InterfaceSpace::new(
        mesh.clone(),
        vec![TraceFieldSpec {
            symbol: U,
            order: 0,
            components: 1,
            continuous: false,
        }],
    )
    .unwrap()
}

#[test]
fn p0_jump_penalty_assembles_the_facet_weighted_graph_laplacian_and_is_orientation_invariant() {
    let eta = 2.5;
    let mesh = unit_square(3);
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    let measure = InterfaceMeasure::interior(&topology).unwrap();
    let cells = mesh.cells().len();
    let kernel = InterfaceKernel::new(jump_penalty_kernel(eta), u_value_bindings()).unwrap();
    let operator = InterfaceOperator::new(p0_space(&mesh), measure.clone(), vec![kernel]).unwrap();
    assert_eq!(operator.dimension(), cells);
    assert_eq!(
        operator.measure().pairs().len(),
        topology.interior().count()
    );

    let mut expected = vec![vec![0.0; cells]; cells];
    for pair in measure.pairs() {
        let weight = eta * facet_length(&mesh, &topology, pair.facet.0);
        let (minus, plus) = (pair.minus.cell.0, pair.plus.cell.0);
        expected[minus][minus] += weight;
        expected[minus][plus] -= weight;
        expected[plus][minus] -= weight;
        expected[plus][plus] += weight;
    }
    let actual = dense(&operator);
    assert_matrix_close(&actual, &expected, 1.0e-13, "jump penalty");

    // Constants are annihilated; residual == linear action for this linear kernel.
    let mut output = vec![0.0; cells];
    operator.residual(&vec![1.7; cells], &mut output).unwrap();
    assert!(output.iter().all(|value| value.abs() <= 1.0e-13));
    let state = probe(cells, 0.4);
    let mut residual = vec![0.0; cells];
    operator.residual(&state, &mut residual).unwrap();
    let mut action = vec![0.0; cells];
    operator.apply_action(&state, &mut action).unwrap();
    for (residual, action) in residual.iter().zip(&action) {
        assert!((residual - action).abs() <= 1.0e-13);
    }
    assert_transpose(&operator, &state, "jump penalty");

    // Methodus views.
    let context = EvaluationContext::default();
    let mut via_trait = vec![0.0; cells];
    operator.apply(&context, &state, &mut via_trait).unwrap();
    assert_eq!(via_trait, action);
    let mut transposed = vec![0.0; cells];
    operator
        .apply_transpose(&context, &state, &mut transposed)
        .unwrap();
    for (transposed, action) in transposed.iter().zip(&action) {
        assert!((transposed - action).abs() <= 1.0e-13, "symmetric kernel");
    }
    let csr = operator.assemble().unwrap();
    assert_eq!(
        csr.values().len(),
        actual.iter().flatten().filter(|v| **v != 0.0).count()
    );

    // Orientation invariance: the flipped measure realizes the identical operator, and the
    // Malleus swap receipt certifies the kernel at the realized traces.
    let flipped = InterfaceOperator::new(
        p0_space(&mesh),
        measure.flipped(),
        vec![InterfaceKernel::new(jump_penalty_kernel(eta), u_value_bindings()).unwrap()],
    )
    .unwrap();
    assert_eq!(dense(&flipped), actual);
    assert_ne!(flipped.digest(), operator.digest());
    let receipt = operator.swap_symmetry_receipt(&state, 1.0e-12).unwrap();
    assert!(receipt.within_tolerance, "{receipt:?}");
    assert_eq!(receipt.facets_checked, measure.pairs().len());
    assert_eq!(receipt.kernels.len(), 1);
    assert_eq!(
        &receipt.kernels[0].kernel,
        operator.kernels().next().unwrap().digest()
    );
}

#[test]
fn normal_dependent_flux_is_orientation_invariant_exactly_when_the_swap_receipt_passes() {
    let beta = [0.8, -0.35];
    let mesh = unit_square(2);
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    let measure = InterfaceMeasure::interior(&topology).unwrap();
    let cells = mesh.cells().len();
    let build = |one_sided: bool, measure: &InterfaceMeasure| {
        InterfaceOperator::new(
            p0_space(&mesh),
            measure.clone(),
            vec![InterfaceKernel::new(flux_kernel(one_sided), flux_bindings(beta)).unwrap()],
        )
        .unwrap()
    };
    let state = probe(cells, 1.3);

    // Covariant central flux: hand-built expectation, flip invariance, receipt passes.
    let central = build(false, &measure);
    let mut expected = vec![vec![0.0; cells]; cells];
    for pair in measure.pairs() {
        let facet = &topology.facets()[pair.facet.0];
        let a = &mesh.vertices()[facet.vertices[0].0];
        let b = &mesh.vertices()[facet.vertices[1].0];
        let tangent = [b[0] - a[0], b[1] - a[1]];
        let length = (tangent[0] * tangent[0] + tangent[1] * tangent[1]).sqrt();
        let mut normal = [tangent[1] / length, -tangent[0] / length];
        let minus_cell = mesh.cell(pair.minus.cell).unwrap();
        let opposite = &mesh.vertices()[minus_cell.vertices[pair.minus.local_facet].0];
        if normal[0] * (opposite[0] - a[0]) + normal[1] * (opposite[1] - a[1]) > 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        let beta_n = beta[0] * normal[0] + beta[1] * normal[1];
        let (minus, plus) = (pair.minus.cell.0, pair.plus.cell.0);
        expected[minus][minus] += 0.5 * beta_n * length;
        expected[minus][plus] += 0.5 * beta_n * length;
        expected[plus][minus] -= 0.5 * beta_n * length;
        expected[plus][plus] -= 0.5 * beta_n * length;
    }
    let actual = dense(&central);
    assert_matrix_close(&actual, &expected, 1.0e-13, "central flux");
    assert_eq!(dense(&build(false, &measure.flipped())), actual);
    assert_transpose(&central, &state, "central flux");
    let receipt = central.swap_symmetry_receipt(&state, 1.0e-12).unwrap();
    assert!(receipt.within_tolerance, "{receipt:?}");

    // One-sided flux: the receipt fails, and the flipped realization differs.
    let one_sided = build(true, &measure);
    let receipt = one_sided.swap_symmetry_receipt(&state, 1.0e-12).unwrap();
    assert!(!receipt.within_tolerance, "{receipt:?}");
    assert!(receipt.max_absolute > 1.0e-3);
    let flipped = dense(&build(true, &measure.flipped()));
    let deviation = dense(&one_sided)
        .iter()
        .flatten()
        .zip(flipped.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        deviation > 1.0e-3,
        "one-sided kernel must depend on the orientation"
    );
    assert_transpose(&one_sided, &state, "one-sided flux");
}

#[test]
fn two_field_p1_interface_coupling_assembles_segment_mass_matrices_on_the_shared_nodes() {
    let mesh = unit_square(2);
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    let left = (0..mesh.cells().len())
        .filter(|&cell| {
            let vertices = &mesh.cell(CellId(cell)).unwrap().vertices;
            let centroid_x = vertices
                .iter()
                .map(|v| mesh.vertices()[v.0][0])
                .sum::<f64>()
                / vertices.len() as f64;
            centroid_x < 0.5
        })
        .map(CellId)
        .collect::<BTreeSet<_>>();
    let measure = InterfaceMeasure::between(&topology, &left).unwrap();
    assert_eq!(
        measure.pairs().len(),
        2,
        "two interface segments at x = 0.5"
    );
    for pair in measure.pairs() {
        assert!(left.contains(&pair.minus.cell));
        assert!(!left.contains(&pair.plus.cell));
    }
    let space = InterfaceSpace::new(
        mesh.clone(),
        vec![
            TraceFieldSpec {
                symbol: A,
                order: 1,
                components: 1,
                continuous: true,
            },
            TraceFieldSpec {
                symbol: B,
                order: 1,
                components: 1,
                continuous: true,
            },
        ],
    )
    .unwrap();
    let vertices = mesh.vertices().len();
    assert_eq!(space.layout().extent(), 2 * vertices);
    let bindings = vec![
        InterfaceOperand::Trace {
            field: A,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Trace {
            field: B,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Residual {
            field: A,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Residual {
            field: B,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Value,
        },
    ];
    let kernel = InterfaceKernel::new(two_field_kernel(), bindings).unwrap();
    let operator = InterfaceOperator::new(space, measure.clone(), vec![kernel]).unwrap();

    let n = 2 * vertices;
    let mut expected = vec![vec![0.0; n]; n];
    for pair in measure.pairs() {
        let facet = &topology.facets()[pair.facet.0];
        let length = facet_length(&mesh, &topology, pair.facet.0);
        let nodes = [facet.vertices[0].0, facet.vertices[1].0];
        for (i, &row) in nodes.iter().enumerate() {
            for (j, &column) in nodes.iter().enumerate() {
                let mass = length / 6.0 * if i == j { 2.0 } else { 1.0 };
                expected[row][column] += mass;
                expected[row][vertices + column] -= mass;
                expected[vertices + row][column] -= mass;
                expected[vertices + row][vertices + column] += mass;
            }
        }
    }
    assert_matrix_close(&dense(&operator), &expected, 1.0e-13, "two-field interface");

    // Equal constants on both sides: zero residual. Different constants: a jump.
    let mut state = vec![0.0; n];
    state[..vertices].fill(0.7);
    state[vertices..].fill(0.7);
    let mut residual = vec![0.0; n];
    operator.residual(&state, &mut residual).unwrap();
    assert!(residual.iter().all(|value| value.abs() <= 1.0e-13));
    state[vertices..].fill(0.2);
    operator.residual(&state, &mut residual).unwrap();
    assert!(residual.iter().any(|value| value.abs() > 1.0e-3));
    assert_transpose(&operator, &probe(n, 2.2), "two-field interface");

    // One-sided cell operands cannot be exchanged: the receipt is refused typed.
    let error = operator.swap_symmetry_receipt(&state, 1.0e-12).unwrap_err();
    assert!(
        matches!(error, FinitumError::UnsupportedRealization(_)),
        "{error}"
    );
}

#[test]
fn gradient_trace_consistency_term_on_a_discontinuous_p1_field_matches_the_closed_form() {
    let mesh = unit_square(2);
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    let measure = InterfaceMeasure::interior(&topology).unwrap();
    let space = InterfaceSpace::new(
        mesh.clone(),
        vec![TraceFieldSpec {
            symbol: U,
            order: 1,
            components: 1,
            continuous: false,
        }],
    )
    .unwrap();
    let cells = mesh.cells().len();
    assert_eq!(space.layout().extent(), 3 * cells);
    let bindings = vec![
        InterfaceOperand::Trace {
            field: U,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Gradient,
        },
        InterfaceOperand::Trace {
            field: U,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Gradient,
        },
        InterfaceOperand::Normal,
        InterfaceOperand::Residual {
            field: U,
            side: FacetSide::Minus,
            evaluation: TraceEvaluation::Value,
        },
        InterfaceOperand::Residual {
            field: U,
            side: FacetSide::Plus,
            evaluation: TraceEvaluation::Value,
        },
    ];
    let kernel = InterfaceKernel::new(gradient_average_kernel(), bindings).unwrap();
    let operator = InterfaceOperator::new(space.clone(), measure.clone(), vec![kernel]).unwrap();

    // u = x in the cell-owned nodal representation (node k of cell c is its k-th vertex).
    let mut state = vec![0.0; 3 * cells];
    for (cell_index, cell) in mesh.cells().iter().enumerate() {
        for (node, vertex) in cell.vertices.iter().enumerate() {
            state[cell_index * 3 + node] = mesh.vertices()[vertex.0][0];
        }
    }
    let mut residual = vec![0.0; 3 * cells];
    operator.residual(&state, &mut residual).unwrap();

    // Closed form: grad u = (1, 0) on both sides, so r_minus = -n_x, r_plus = +n_x, each
    // scattered as `integral_F phi_i ds = |F| / 2` onto the two facet nodes of its cell.
    let mut expected = vec![0.0; 3 * cells];
    for pair in measure.pairs() {
        let facet = &topology.facets()[pair.facet.0];
        let length = facet_length(&mesh, &topology, pair.facet.0);
        let a = &mesh.vertices()[facet.vertices[0].0];
        let b = &mesh.vertices()[facet.vertices[1].0];
        let tangent = [b[0] - a[0], b[1] - a[1]];
        let mut normal = [tangent[1] / length, -tangent[0] / length];
        let minus_cell = mesh.cell(pair.minus.cell).unwrap();
        let opposite = &mesh.vertices()[minus_cell.vertices[pair.minus.local_facet].0];
        if normal[0] * (opposite[0] - a[0]) + normal[1] * (opposite[1] - a[1]) > 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        for (incidence, sign) in [(pair.minus, -1.0), (pair.plus, 1.0)] {
            let cell = mesh.cell(incidence.cell).unwrap();
            for (node, vertex) in cell.vertices.iter().enumerate() {
                if facet.vertices.contains(vertex) {
                    expected[incidence.cell.0 * 3 + node] += sign * normal[0] * length / 2.0;
                }
            }
        }
    }
    for (index, (actual, expected)) in residual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-13,
            "gradient consistency at {index}: {actual} != {expected}"
        );
    }
    assert_transpose(&operator, &state, "gradient consistency");
    let receipt = operator.swap_symmetry_receipt(&state, 1.0e-12).unwrap();
    assert!(receipt.within_tolerance, "{receipt:?}");
    let flipped = InterfaceOperator::new(
        space,
        measure.flipped(),
        vec![
            InterfaceKernel::new(
                gradient_average_kernel(),
                operator.kernels().next().unwrap().operands().to_vec(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut flipped_residual = vec![0.0; 3 * cells];
    flipped.residual(&state, &mut flipped_residual).unwrap();
    for (a, b) in residual.iter().zip(&flipped_residual) {
        assert!((a - b).abs() <= 1.0e-13);
    }
}

#[test]
fn interface_binding_refusals_are_typed() {
    let mesh = unit_square(2);
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    let measure = InterfaceMeasure::interior(&topology).unwrap();

    // Wrong side, wrong parity, read/write mismatch, wrong binding count, no residual.
    let mut wrong_side = u_value_bindings();
    wrong_side[0] = InterfaceOperand::Trace {
        field: U,
        side: FacetSide::Plus,
        evaluation: TraceEvaluation::Value,
    };
    assert!(matches!(
        InterfaceKernel::new(jump_penalty_kernel(1.0), wrong_side),
        Err(FinitumError::InvalidRealization(_))
    ));
    let mut normal_as_even = flux_bindings([1.0, 0.0]);
    normal_as_even[2] = InterfaceOperand::Coordinates;
    assert!(matches!(
        InterfaceKernel::new(flux_kernel(false), normal_as_even),
        Err(FinitumError::InvalidRealization(_))
    ));
    let mut output_as_input = u_value_bindings();
    output_as_input[2] = InterfaceOperand::Trace {
        field: U,
        side: FacetSide::Minus,
        evaluation: TraceEvaluation::Value,
    };
    assert!(matches!(
        InterfaceKernel::new(jump_penalty_kernel(1.0), output_as_input),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        InterfaceKernel::new(jump_penalty_kernel(1.0), u_value_bindings()[..3].to_vec()),
        Err(FinitumError::InvalidRealization(_))
    ));
    let mut no_residual = u_value_bindings();
    no_residual[2] = InterfaceOperand::Ignored;
    no_residual[3] = InterfaceOperand::Ignored;
    assert!(matches!(
        InterfaceKernel::new(jump_penalty_kernel(1.0), no_residual),
        Err(FinitumError::InvalidRealization(_))
    ));

    // Shape mismatch (a gradient bound to a scalar operand) and an unknown field are caught at
    // realization time.
    let mut gradient_scalar = u_value_bindings();
    gradient_scalar[0] = InterfaceOperand::Trace {
        field: U,
        side: FacetSide::Minus,
        evaluation: TraceEvaluation::Gradient,
    };
    let kernel = InterfaceKernel::new(jump_penalty_kernel(1.0), gradient_scalar).unwrap();
    assert!(matches!(
        InterfaceOperator::new(p0_space(&mesh), measure.clone(), vec![kernel]),
        Err(FinitumError::InvalidRealization(_))
    ));
    let mut unknown_field = u_value_bindings();
    unknown_field[1] = InterfaceOperand::Trace {
        field: A,
        side: FacetSide::Plus,
        evaluation: TraceEvaluation::Value,
    };
    let kernel = InterfaceKernel::new(jump_penalty_kernel(1.0), unknown_field).unwrap();
    assert!(matches!(
        InterfaceOperator::new(p0_space(&mesh), measure.clone(), vec![kernel]),
        Err(FinitumError::InvalidRealization(_))
    ));

    // Exterior facets and repeated facets are not an interface measure.
    let exterior = topology.exterior().next().unwrap();
    assert!(matches!(
        topology.oriented_pair(exterior.id, exterior.minus().cell),
        Err(FinitumError::InvalidRealization(_))
    ));
    let pair = measure.pairs()[0];
    assert!(matches!(
        InterfaceMeasure::from_pairs(vec![pair, pair]),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        InterfaceMeasure::from_pairs(Vec::new()),
        Err(FinitumError::InvalidRealization(_))
    ));

    // Unsupported trace fields.
    for spec in [
        TraceFieldSpec {
            symbol: U,
            order: 0,
            components: 1,
            continuous: true,
        },
        TraceFieldSpec {
            symbol: U,
            order: 2,
            components: 1,
            continuous: false,
        },
        TraceFieldSpec {
            symbol: U,
            order: 3,
            components: 1,
            continuous: true,
        },
    ] {
        assert!(matches!(
            InterfaceSpace::new(mesh.clone(), vec![spec]),
            Err(FinitumError::UnsupportedRealization(_))
        ));
    }

    // `over_layout` checks the layout's extents against the trace field.
    let vertices = mesh.vertices().len();
    let layout = BlockLayout::new([(A, vertices, 2), (U, mesh.cells().len(), 1)]).unwrap();
    let space = InterfaceSpace::over_layout(
        mesh.clone(),
        vec![TraceFieldSpec {
            symbol: U,
            order: 0,
            components: 1,
            continuous: false,
        }],
        layout.clone(),
    )
    .unwrap();
    assert_eq!(space.layout().extent(), 2 * vertices + mesh.cells().len());
    let kernel = InterfaceKernel::new(jump_penalty_kernel(1.0), u_value_bindings()).unwrap();
    let operator = InterfaceOperator::new(space, measure.clone(), vec![kernel]).unwrap();
    let mut output = vec![0.0; operator.dimension()];
    operator
        .apply_action(&probe(operator.dimension(), 0.5), &mut output)
        .unwrap();
    assert!(output[..2 * vertices].iter().all(|v| *v == 0.0));
    assert!(output[2 * vertices..].iter().any(|v| v.abs() > 1.0e-6));
    assert!(matches!(
        InterfaceSpace::over_layout(
            mesh.clone(),
            vec![TraceFieldSpec {
                symbol: A,
                order: 1,
                components: 1,
                continuous: true,
            }],
            layout,
        ),
        Err(FinitumError::ArtifactMismatch(_))
    ));
}
