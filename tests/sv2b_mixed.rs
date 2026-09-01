//! SV2-B1/B4: mixed product layouts and block operator composition (`finitum::mixed`).
//!
//! Fixture: a vector P2 field ("field_a", 2 components) and a scalar P1 field ("field_b", 1
//! component) on the unit square, coupled by a generic `GradientGradient` diagonal block on
//! field_a and a `DivergenceValue` off-diagonal coupling between field_a and field_b -- the
//! standard `[[A, B^T], [B, 0]]` saddle-point shape, built entirely from structural machinery
//! (no named physics anywhere in this file or in `finitum::mixed`).

use finitum::{
    AffineMap, BlockCoupling, BlockLayout, BlockNullspaceCandidate, Cell, CellId, CouplingKind,
    FieldSpec, FinitumError, Mesh, MixedOperator, MixedSpace, VertexId, simplex_basis,
};
use methodus::{BlockLinearOperator, LinearOperator, check_properties_consistency};
use scientia::SymbolId;

const FIELD_A: SymbolId = SymbolId(0);
const FIELD_B: SymbolId = SymbolId(1);

fn unit_square_mesh(subdivisions: usize) -> Mesh {
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
    Mesh::new(2, vertices, cells).unwrap()
}

fn fixture_space(subdivisions: usize) -> MixedSpace {
    MixedSpace::new(
        unit_square_mesh(subdivisions),
        vec![
            FieldSpec {
                symbol: FIELD_A,
                order: 2,
                components: 2,
            },
            FieldSpec {
                symbol: FIELD_B,
                order: 1,
                components: 1,
            },
        ],
    )
    .unwrap()
}

fn fixture_couplings() -> Vec<BlockCoupling> {
    vec![
        BlockCoupling {
            test: FIELD_A,
            trial: FIELD_A,
            kind: CouplingKind::GradientGradient,
            scale: 1.0,
        },
        BlockCoupling {
            test: FIELD_A,
            trial: FIELD_B,
            kind: CouplingKind::DivergenceValue,
            scale: 1.0,
        },
    ]
}

#[test]
fn mixed_layout_has_no_overlap_or_gap_and_orders_deterministically() {
    let space = fixture_space(3);
    let layout = space.layout();
    let blocks = layout.blocks();
    assert_eq!(blocks.len(), 2);
    // Contiguous, non-overlapping offsets in declaration order.
    let mut expected_offset = 0usize;
    for block in blocks {
        assert_eq!(block.offset, expected_offset);
        expected_offset += block.extent;
    }
    assert_eq!(layout.extent(), expected_offset);
    assert_eq!(
        layout.block(FIELD_A).unwrap().extent,
        space.dof_map(FIELD_A).unwrap().dof_count()
    );
    assert_eq!(
        layout.block(FIELD_B).unwrap().extent,
        space.dof_map(FIELD_B).unwrap().dof_count()
    );

    // Deterministic: an independently reconstructed space produces a byte-identical layout.
    let repeated = fixture_space(3);
    assert_eq!(layout, repeated.layout());
}

#[test]
fn mixed_space_refuses_malformed_field_shapes() {
    let mesh = unit_square_mesh(1);
    assert!(matches!(
        MixedSpace::new(mesh.clone(), Vec::new()).unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    assert!(matches!(
        MixedSpace::new(
            mesh.clone(),
            vec![FieldSpec {
                symbol: FIELD_A,
                order: 3,
                components: 1,
            }],
        )
        .unwrap_err(),
        FinitumError::UnsupportedRealization(_)
    ));
    assert!(matches!(
        MixedSpace::new(
            mesh.clone(),
            vec![FieldSpec {
                symbol: FIELD_A,
                order: 1,
                components: 3,
            }],
        )
        .unwrap_err(),
        FinitumError::UnsupportedRealization(_)
    ));
    assert!(matches!(
        MixedSpace::new(
            mesh,
            vec![
                FieldSpec {
                    symbol: FIELD_A,
                    order: 1,
                    components: 1,
                },
                FieldSpec {
                    symbol: FIELD_A,
                    order: 2,
                    components: 1,
                },
            ],
        )
        .unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
}

#[test]
fn mixed_operator_refuses_malformed_couplings() {
    let space = fixture_space(1);
    assert!(matches!(
        MixedOperator::new(fixture_space(1), Vec::new()).unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    // GradientGradient across two distinct fields is refused.
    assert!(matches!(
        MixedOperator::new(
            fixture_space(1),
            vec![BlockCoupling {
                test: FIELD_A,
                trial: FIELD_B,
                kind: CouplingKind::GradientGradient,
                scale: 1.0,
            }],
        )
        .unwrap_err(),
        FinitumError::UnsupportedRealization(_)
    ));
    // DivergenceValue with a scalar test field is refused.
    assert!(matches!(
        MixedOperator::new(
            fixture_space(1),
            vec![BlockCoupling {
                test: FIELD_B,
                trial: FIELD_A,
                kind: CouplingKind::DivergenceValue,
                scale: 1.0,
            }],
        )
        .unwrap_err(),
        FinitumError::UnsupportedRealization(_)
    ));
    // DivergenceValue on a single field is refused.
    assert!(matches!(
        MixedOperator::new(
            fixture_space(1),
            vec![BlockCoupling {
                test: FIELD_A,
                trial: FIELD_A,
                kind: CouplingKind::DivergenceValue,
                scale: 1.0,
            }],
        )
        .unwrap_err(),
        FinitumError::UnsupportedRealization(_)
    ));
    // Non-finite scale is refused.
    assert!(matches!(
        MixedOperator::new(
            space,
            vec![BlockCoupling {
                test: FIELD_A,
                trial: FIELD_A,
                kind: CouplingKind::GradientGradient,
                scale: f64::NAN,
            }],
        )
        .unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
}

/// SV2-B4: the block/coupling-composed matrix-free action and its column-probed assembly agree
/// with a genuinely independent, separately derived monolithic reference to near machine
/// precision. The reference is built in this test with its own quadrature rule (a 3-point
/// degree-2 triangle rule, distinct from `finitum::element`'s own 6-point degree-4 rule) and its
/// own nested loops -- it does not call into `finitum::mixed`'s private local-matrix builders.
#[test]
fn block_apply_agrees_with_an_independently_assembled_monolithic_reference() {
    let space = fixture_space(2);
    let couplings = fixture_couplings();
    let reference = independent_reference_matrix(&space, &couplings);
    let operator = MixedOperator::new(space, couplings).unwrap();
    let dimension = operator.dimension();
    assert_eq!(reference.len(), dimension * dimension);

    // Matrix-free action vs. the reference matrix-vector product, several probe vectors.
    for seed in 0..5u64 {
        let input = pseudo_random_vector(dimension, seed);
        let mut actual = vec![0.0; dimension];
        operator.apply_action(&input, &mut actual).unwrap();
        let expected = dense_matvec(&reference, dimension, &input);
        assert_close(&actual, &expected, 5.0e-11);
    }

    // Column-probed assembly vs. the reference matrix, entrywise.
    let assembled = operator.assemble().unwrap();
    let mut dense_assembled = vec![0.0; dimension * dimension];
    for row in 0..assembled.rows() {
        for entry in assembled.row_offsets()[row]..assembled.row_offsets()[row + 1] {
            dense_assembled[row * dimension + assembled.column_indices()[entry]] =
                assembled.values()[entry];
        }
    }
    assert_close(&dense_assembled, &reference, 5.0e-11);

    // Declared symmetric by construction; the reference independently confirms it.
    assert_eq!(operator.symmetry(), methodus::OperatorSymmetry::Symmetric);
    check_properties_consistency(&operator).unwrap();
    for row in 0..dimension {
        for column in 0..dimension {
            assert!(
                (reference[row * dimension + column] - reference[column * dimension + row]).abs()
                    < 1.0e-11,
                "independent reference is not symmetric at ({row}, {column})"
            );
        }
    }

    let block_layout = operator.block_layout();
    assert_eq!(block_layout.blocks().len(), 2);
    assert_eq!(block_layout.dimension(), dimension);
}

/// SV2-B1: the pressure-nullspace representation, resolved against this fixture's `field_b`
/// block, matches the mathematically expected structure -- exactly zero on `field_b` rows and
/// on every interior `field_a` dof, and genuinely nonzero on at least one boundary `field_a`
/// dof -- demonstrating that the "pure-Dirichlet pressure mode" wording is literal: the vector
/// is only in the *reduced* (post-Dirichlet-elimination) system's kernel, not the unconstrained
/// monolithic one this representation-only module builds. `verify_in_kernel` therefore correctly
/// refuses to certify it against the unconstrained operator.
#[test]
fn pressure_like_nullspace_candidate_round_trips_and_matches_expected_structure() {
    let space = fixture_space(3);
    let couplings = fixture_couplings();
    let layout: BlockLayout = space.layout().clone();
    let boundary_a = boundary_field_a_dofs(&space);
    let operator = MixedOperator::new(space, couplings).unwrap();

    let candidate = BlockNullspaceCandidate::constant(FIELD_B, "field_a carries no Dirichlet data");
    let mode = candidate.resolve(&layout).unwrap();
    let field_b = layout.block(FIELD_B).unwrap();
    let expected_value = 1.0 / (field_b.extent as f64).sqrt();
    for (index, value) in mode.vector().iter().enumerate() {
        if (field_b.offset..field_b.offset + field_b.extent).contains(&index) {
            assert!((value - expected_value).abs() < 1.0e-15);
        } else {
            assert_eq!(*value, 0.0);
        }
    }

    // Round trip: the typed declaration serializes/deserializes and reconstructs the identical
    // vector deterministically from the layout, without carrying the vector itself as data.
    let encoded = serde_json::to_string(&candidate).unwrap();
    let decoded: BlockNullspaceCandidate = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, candidate);
    let mode_from_round_trip = decoded.resolve(&layout).unwrap();
    assert_eq!(mode_from_round_trip.vector(), mode.vector());

    // Not certified against the unconstrained operator (real discriminative power).
    assert!(!mode.verify_in_kernel(&operator, 1.0e-9).unwrap());

    let mut residual = vec![0.0; operator.dimension()];
    operator.apply_action(mode.vector(), &mut residual).unwrap();
    let field_a = layout.block(FIELD_A).unwrap();
    for (index, value) in residual
        .iter()
        .enumerate()
        .skip(field_b.offset)
        .take(field_b.extent)
    {
        assert!(value.abs() < 1.0e-12, "field_b residual row {index}");
    }
    let mut max_boundary_residual: f64 = 0.0;
    for (local, is_boundary) in boundary_a.iter().enumerate().take(field_a.extent) {
        let global = field_a.offset + local;
        if *is_boundary {
            max_boundary_residual = max_boundary_residual.max(residual[global].abs());
        } else {
            assert!(
                residual[global].abs() < 1.0e-9,
                "interior field_a residual row {global} = {}",
                residual[global]
            );
        }
    }
    assert!(
        max_boundary_residual > 1.0e-6,
        "expected a genuinely nonzero boundary field_a residual, got max {max_boundary_residual}"
    );
}

#[test]
fn nullspace_candidate_refuses_unknown_or_non_scalar_blocks_and_bad_tolerances() {
    let space = fixture_space(1);
    let couplings = fixture_couplings();
    let layout: BlockLayout = space.layout().clone();
    let operator = MixedOperator::new(space, couplings).unwrap();

    assert!(matches!(
        BlockNullspaceCandidate::constant(SymbolId(99), "absent")
            .resolve(&layout)
            .unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    assert!(matches!(
        BlockNullspaceCandidate::constant(FIELD_A, "field_a is vector-valued")
            .resolve(&layout)
            .unwrap_err(),
        FinitumError::UnsupportedRealization(_)
    ));

    let mode = BlockNullspaceCandidate::constant(FIELD_B, "ok")
        .resolve(&layout)
        .unwrap();
    assert!(matches!(
        mode.verify_in_kernel(&operator, f64::NAN).unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    assert!(matches!(
        mode.verify_in_kernel(&operator, -1.0).unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
}

fn boundary_field_a_dofs(space: &MixedSpace) -> Vec<bool> {
    let node_points = quadratic_simplex_node_points_for(space);
    let components = space.field(FIELD_A).unwrap().components;
    let mut boundary = vec![false; node_points.len() * components];
    for (node, point) in node_points.iter().enumerate() {
        let on_boundary = point[0] <= 1.0e-12
            || point[1] <= 1.0e-12
            || point[0] >= 1.0 - 1.0e-12
            || point[1] >= 1.0 - 1.0e-12;
        for component in 0..components {
            boundary[node * components + component] = on_boundary;
        }
    }
    boundary
}

fn quadratic_simplex_node_points_for(space: &MixedSpace) -> Vec<Vec<f64>> {
    finitum::quadratic_simplex_node_points(space.mesh())
}

fn dense_matvec(matrix: &[f64], dimension: usize, input: &[f64]) -> Vec<f64> {
    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| matrix[row * dimension + column] * input[column])
                .sum()
        })
        .collect()
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

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: {actual} != {expected} within {tolerance}"
        );
    }
}

/// Independent monolithic reference, built with this test's own 3-point degree-2 triangle
/// quadrature and its own nested loops -- deliberately not sharing code with
/// `finitum::mixed`'s (private) local-matrix builders, which use a different (6-point
/// degree-4) quadrature rule. Both rules integrate these polynomial integrands (degree <= 2)
/// exactly, so entrywise agreement to near machine precision is a genuine two-derivation check.
fn independent_reference_matrix(space: &MixedSpace, couplings: &[BlockCoupling]) -> Vec<f64> {
    let dimension = space.layout().extent();
    let mesh_dimension = space.mesh().dimension();
    let mut matrix = vec![0.0; dimension * dimension];
    let quadrature = independent_triangle_quadrature();
    for cell in 0..space.mesh().cells().len() {
        let map = AffineMap::from_cell(space.mesh(), CellId(cell)).unwrap();
        for coupling in couplings {
            match coupling.kind {
                CouplingKind::GradientGradient => {
                    let field = space.field(coupling.test).unwrap();
                    let block = space.layout().block(coupling.test).unwrap();
                    let restriction = &space.dof_map(coupling.test).unwrap().restrictions()[cell];
                    let basis_count = restriction.dofs.len() / field.components;
                    for &(x, y, weight) in &quadrature {
                        let (_, gradients) =
                            simplex_basis(mesh_dimension, field.order, &[x, y]).unwrap();
                        let physical = gradients
                            .iter()
                            .map(|gradient| map.covariant_piola(gradient).unwrap())
                            .collect::<Vec<_>>();
                        let scale = weight * map.determinant();
                        for i in 0..basis_count {
                            for j in 0..basis_count {
                                let dot = (0..mesh_dimension)
                                    .map(|axis| physical[i][axis] * physical[j][axis])
                                    .sum::<f64>();
                                for component in 0..field.components {
                                    let row = block.offset
                                        + restriction.dofs[i * field.components + component].0;
                                    let column = block.offset
                                        + restriction.dofs[j * field.components + component].0;
                                    matrix[row * dimension + column] +=
                                        coupling.scale * scale * dot;
                                }
                            }
                        }
                    }
                }
                CouplingKind::DivergenceValue => {
                    let test_field = space.field(coupling.test).unwrap();
                    let trial_field = space.field(coupling.trial).unwrap();
                    let test_block = space.layout().block(coupling.test).unwrap();
                    let trial_block = space.layout().block(coupling.trial).unwrap();
                    let test_restriction =
                        &space.dof_map(coupling.test).unwrap().restrictions()[cell];
                    let trial_restriction =
                        &space.dof_map(coupling.trial).unwrap().restrictions()[cell];
                    let test_basis_count = test_restriction.dofs.len() / test_field.components;
                    let trial_basis_count = trial_restriction.dofs.len();
                    for &(x, y, weight) in &quadrature {
                        let (_, test_gradients) =
                            simplex_basis(mesh_dimension, test_field.order, &[x, y]).unwrap();
                        let (trial_values, _) =
                            simplex_basis(mesh_dimension, trial_field.order, &[x, y]).unwrap();
                        let physical_test_gradients = test_gradients
                            .iter()
                            .map(|gradient| map.covariant_piola(gradient).unwrap())
                            .collect::<Vec<_>>();
                        let scale = weight * map.determinant();
                        for (i, gradient) in physical_test_gradients
                            .iter()
                            .enumerate()
                            .take(test_basis_count)
                        {
                            for (component, &divergence_contribution) in gradient.iter().enumerate()
                            {
                                let row = test_block.offset
                                    + test_restriction.dofs[i * mesh_dimension + component].0;
                                for (j, &trial_value) in
                                    trial_values.iter().enumerate().take(trial_basis_count)
                                {
                                    let column = trial_block.offset + trial_restriction.dofs[j].0;
                                    let value = coupling.scale
                                        * scale
                                        * divergence_contribution
                                        * trial_value;
                                    matrix[row * dimension + column] += value;
                                    matrix[column * dimension + row] += value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    matrix
}

/// A 3-point, degree-2-exact symmetric quadrature for the reference triangle, distinct from
/// `finitum::element`'s own 6-point degree-4 rule.
fn independent_triangle_quadrature() -> Vec<(f64, f64, f64)> {
    const AREA: f64 = 0.5;
    let a = 2.0 / 3.0;
    let b = 1.0 / 6.0;
    [(b, b), (a, b), (b, a)]
        .into_iter()
        .map(|(x, y)| (x, y, AREA / 3.0))
        .collect()
}
