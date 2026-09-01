//! SV2-B1/B4: mixed product layouts and block operator composition (`finitum::mixed`).
//!
//! Fixture: a vector P2 field ("field_a", 2 components) and a scalar P1 field ("field_b", 1
//! component) on the unit square, coupled by a generic `GradientGradient` diagonal block on
//! field_a and a `DivergenceValue` off-diagonal coupling between field_a and field_b -- the
//! standard `[[A, B^T], [B, 0]]` saddle-point shape, built entirely from structural machinery
//! (no named physics anywhere in this file or in `finitum::mixed`).

use finitum::{
    AffineMap, BlockCoupling, BlockEssentialValue, BlockLayout, BlockNullspaceCandidate, Cell,
    CellId, CouplingKind, FieldSpec, FinitumError, Mesh, MixedOperator, MixedSpace, VertexId,
    essential_constraints_for_blocks, simplex_basis,
};
use methodus::{
    BlockLinearOperator, EvaluationContext, LinearOperator, MinresConfig, NullspaceProjector,
    check_properties_consistency, solve_minres,
};
use scientia::SymbolId;

const FIELD_A: SymbolId = SymbolId(0);
const FIELD_B: SymbolId = SymbolId(1);
const FIELD_C: SymbolId = SymbolId(2);

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
    let reference =
        independent_reference_matrix(&space, &couplings, &independent_triangle_quadrature());
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

/// SV2-B4: Dirichlet elimination of the boundary `field_a` degrees of freedom completes the
/// honest demonstration `pressure_like_nullspace_candidate_round_trips_and_matches_expected_structure`
/// leaves open above -- the constant pressure mode is only in the kernel of the *reduced*
/// system, not the raw unconstrained one. This builds that reduced system with
/// `MixedOperator::reduced` over `finitum::essential_constraints_for_blocks` and shows
/// `verify_in_kernel` now passes against it, and that the reduced action matches an
/// independently-derived dense reduced reference: the same independent monolithic reference
/// `block_apply_agrees_with_an_independently_assembled_monolithic_reference` uses, with the
/// identity-row/zero-column elimination transform applied directly in this test -- not through
/// any code path shared with `MixedOperator::apply_reduced_action`.
#[test]
fn dirichlet_elimination_makes_the_pressure_nullspace_candidate_verify_against_the_reduced_operator()
 {
    let space = fixture_space(3);
    let couplings = fixture_couplings();
    let reference =
        independent_reference_matrix(&space, &couplings, &independent_triangle_quadrature());
    let layout: BlockLayout = space.layout().clone();
    let boundary_a = boundary_field_a_dofs(&space);
    let field_a = layout.block(FIELD_A).unwrap();
    let components = space.field(FIELD_A).unwrap().components;
    let operator = MixedOperator::new(space, couplings).unwrap();
    let dimension = operator.dimension();
    assert_eq!(reference.len(), dimension * dimension);

    let constrained_dofs = boundary_a
        .iter()
        .enumerate()
        .filter(|(_, constrained)| **constrained)
        .map(|(local, _)| field_a.offset + local)
        .collect::<Vec<_>>();
    assert!(
        !constrained_dofs.is_empty(),
        "fixture must have at least one boundary field_a dof"
    );
    let constraints = essential_constraints_for_blocks(
        &layout,
        boundary_a
            .iter()
            .enumerate()
            .filter(|(_, constrained)| **constrained)
            .map(|(local, _)| BlockEssentialValue {
                block: FIELD_A,
                entity: local / components,
                component: local % components,
                value: 0.0,
            }),
    )
    .unwrap();

    let dense_reduced_reference = eliminate_dense(&reference, dimension, &constrained_dofs);

    let reduced = operator.reduced(constraints).unwrap();
    for seed in 0..5u64 {
        let input = pseudo_random_vector(dimension, seed + 200);
        let mut actual = vec![0.0; dimension];
        reduced
            .apply(&EvaluationContext::default(), &input, &mut actual)
            .unwrap();
        let expected = dense_matvec(&dense_reduced_reference, dimension, &input);
        assert_close(&actual, &expected, 5.0e-11);
    }

    assert_eq!(reduced.symmetry(), methodus::OperatorSymmetry::Symmetric);
    check_properties_consistency(&reduced).unwrap();

    let candidate = BlockNullspaceCandidate::constant(FIELD_B, "field_a carries no Dirichlet data");
    let mode = candidate.resolve(&layout).unwrap();
    assert!(
        mode.verify_in_kernel(&reduced, 1.0e-9).unwrap(),
        "the constant pressure mode should now verify in the kernel of the reduced operator"
    );
}

/// SV2-B4 acceptance (b): a `methodus::solve_minres` solve over `MixedOperator::reduced`'s
/// output, using the resolved `ConstantModeProjector`, converges on the reduced saddle-point
/// system and matches an independently-derived dense reduced reference (the same elimination
/// transform as the test above, applied to a right-hand side generated from a known solution).
#[test]
fn minres_converges_on_the_reduced_saddle_point_system_and_matches_a_dense_reduced_reference() {
    let space = fixture_space(3);
    let couplings = fixture_couplings();
    let reference =
        independent_reference_matrix(&space, &couplings, &independent_triangle_quadrature());
    let layout: BlockLayout = space.layout().clone();
    let boundary_a = boundary_field_a_dofs(&space);
    let field_a = layout.block(FIELD_A).unwrap();
    let field_b = layout.block(FIELD_B).unwrap();
    let components = space.field(FIELD_A).unwrap().components;
    let operator = MixedOperator::new(space, couplings).unwrap();
    let dimension = operator.dimension();

    let constrained_dofs = boundary_a
        .iter()
        .enumerate()
        .filter(|(_, constrained)| **constrained)
        .map(|(local, _)| field_a.offset + local)
        .collect::<Vec<_>>();
    let constraints = essential_constraints_for_blocks(
        &layout,
        boundary_a
            .iter()
            .enumerate()
            .filter(|(_, constrained)| **constrained)
            .map(|(local, _)| BlockEssentialValue {
                block: FIELD_A,
                entity: local / components,
                component: local % components,
                value: 0.0,
            }),
    )
    .unwrap();

    let dense_reduced_reference = eliminate_dense(&reference, dimension, &constrained_dofs);

    // A known solution respecting the homogeneous Dirichlet data and lying in the orthogonal
    // complement of the declared constant-pressure nullspace (zero-mean on field_b), so the
    // generated right-hand side is exactly consistent and MINRES's projected iterates converge
    // to it rather than to some nullspace-shifted variant.
    let mut x_true = pseudo_random_vector(dimension, 777);
    for &t in &constrained_dofs {
        x_true[t] = 0.0;
    }
    let pressure_mean = x_true[field_b.offset..field_b.offset + field_b.extent]
        .iter()
        .sum::<f64>()
        / field_b.extent as f64;
    for value in &mut x_true[field_b.offset..field_b.offset + field_b.extent] {
        *value -= pressure_mean;
    }

    let right_hand_side = dense_matvec(&dense_reduced_reference, dimension, &x_true);

    let reduced = operator.reduced(constraints).unwrap();
    let mode = BlockNullspaceCandidate::constant(FIELD_B, "field_a carries no Dirichlet data")
        .resolve(&layout)
        .unwrap();
    assert!(mode.verify_in_kernel(&reduced, 1.0e-9).unwrap());

    let config = MinresConfig {
        max_iterations: 2_000,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-10,
    };
    let report = solve_minres(
        &reduced,
        None,
        Some(mode.projector() as &dyn NullspaceProjector),
        &EvaluationContext::default(),
        &right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap();
    assert!(
        report.converged,
        "minres did not converge on the reduced system"
    );

    assert_close(&report.solution, &x_true, 1.0e-6);
    let recovered = dense_matvec(&dense_reduced_reference, dimension, &report.solution);
    assert_close(&recovered, &right_hand_side, 1.0e-6);
}

#[test]
fn reduced_refuses_a_constraint_set_of_the_wrong_dimension() {
    let space = fixture_space(1);
    let couplings = fixture_couplings();
    let operator = MixedOperator::new(space, couplings).unwrap();
    let mismatched = essential_constraints_for_blocks(
        &BlockLayout::new(vec![(FIELD_A, 3, 1)]).unwrap(),
        Vec::new(),
    )
    .unwrap();
    assert!(matches!(
        operator.reduced(mismatched).unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
}

#[test]
fn essential_constraints_for_blocks_refuses_unknown_blocks_out_of_range_entries_and_conflicts() {
    let space = fixture_space(1);
    let layout: BlockLayout = space.layout().clone();

    assert!(matches!(
        essential_constraints_for_blocks(
            &layout,
            vec![BlockEssentialValue {
                block: SymbolId(99),
                entity: 0,
                component: 0,
                value: 0.0,
            }],
        )
        .unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    let field_a = layout.block(FIELD_A).unwrap();
    assert!(matches!(
        essential_constraints_for_blocks(
            &layout,
            vec![BlockEssentialValue {
                block: FIELD_A,
                entity: field_a.entity_count,
                component: 0,
                value: 0.0,
            }],
        )
        .unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    assert!(matches!(
        essential_constraints_for_blocks(
            &layout,
            vec![BlockEssentialValue {
                block: FIELD_A,
                entity: 0,
                component: field_a.component_count,
                value: 0.0,
            }],
        )
        .unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    assert!(matches!(
        essential_constraints_for_blocks(
            &layout,
            vec![BlockEssentialValue {
                block: FIELD_A,
                entity: 0,
                component: 0,
                value: f64::NAN,
            }],
        )
        .unwrap_err(),
        FinitumError::InvalidRealization(_)
    ));
    // Two declarations targeting the same degree of freedom are refused via `ConstraintSet::new`.
    assert!(
        essential_constraints_for_blocks(
            &layout,
            vec![
                BlockEssentialValue {
                    block: FIELD_A,
                    entity: 0,
                    component: 0,
                    value: 0.0,
                },
                BlockEssentialValue {
                    block: FIELD_A,
                    entity: 0,
                    component: 0,
                    value: 1.0,
                },
            ],
        )
        .is_err()
    );
}

/// SV2-B4: `MixedOperator::digest()` is content-addressed over space (mesh and field specs) and
/// couplings, mirroring `RealizationPlan::digest()` -- unlike
/// `krasis::block_solve::operator_identity`'s shape-only identity, it distinguishes two
/// operators that share a block layout but differ in coupling scale or mesh refinement.
#[test]
fn digest_is_content_addressed_over_space_and_couplings() {
    let operator_a = MixedOperator::new(fixture_space(2), fixture_couplings()).unwrap();
    let operator_b = MixedOperator::new(fixture_space(2), fixture_couplings()).unwrap();
    assert_eq!(operator_a.digest(), operator_b.digest());

    let mut rescaled = fixture_couplings();
    rescaled[0].scale = 2.0;
    let operator_c = MixedOperator::new(fixture_space(2), rescaled).unwrap();
    assert_ne!(operator_a.digest(), operator_c.digest());

    let operator_d = MixedOperator::new(fixture_space(4), fixture_couplings()).unwrap();
    assert_ne!(operator_a.digest(), operator_d.digest());
}

fn fixture_space_p2_scalar_trial(subdivisions: usize) -> MixedSpace {
    MixedSpace::new(
        unit_square_mesh(subdivisions),
        vec![
            FieldSpec {
                symbol: FIELD_A,
                order: 2,
                components: 2,
            },
            FieldSpec {
                symbol: FIELD_C,
                order: 2,
                components: 1,
            },
        ],
    )
    .unwrap()
}

fn fixture_couplings_p2_scalar_trial() -> Vec<BlockCoupling> {
    vec![
        BlockCoupling {
            test: FIELD_A,
            trial: FIELD_A,
            kind: CouplingKind::GradientGradient,
            scale: 1.0,
        },
        BlockCoupling {
            test: FIELD_A,
            trial: FIELD_C,
            kind: CouplingKind::DivergenceValue,
            scale: 1.0,
        },
    ]
}

/// SV2-B1's own deferred item: `local_divergence_value`'s test/trial order parameters are
/// independent (order-generic), but every fixture above pairs a P2 vector test field with a P1
/// scalar trial field. This exercises the previously-unexercised P2 vector / P2 *scalar* trial
/// pairing against the same independently-derived monolithic reference the P2/P1 fixture uses.
#[test]
fn divergence_value_coupling_agrees_with_reference_for_a_p2_scalar_trial_field() {
    let space = fixture_space_p2_scalar_trial(2);
    let couplings = fixture_couplings_p2_scalar_trial();
    let reference = independent_reference_matrix(
        &space,
        &couplings,
        &independent_higher_order_triangle_quadrature(),
    );
    let operator = MixedOperator::new(space, couplings).unwrap();
    let dimension = operator.dimension();
    assert_eq!(reference.len(), dimension * dimension);

    for seed in 0..5u64 {
        let input = pseudo_random_vector(dimension, seed + 300);
        let mut actual = vec![0.0; dimension];
        operator.apply_action(&input, &mut actual).unwrap();
        let expected = dense_matvec(&reference, dimension, &input);
        assert_close(&actual, &expected, 5.0e-11);
    }

    let assembled = operator.assemble().unwrap();
    let mut dense_assembled = vec![0.0; dimension * dimension];
    for row in 0..assembled.rows() {
        for entry in assembled.row_offsets()[row]..assembled.row_offsets()[row + 1] {
            dense_assembled[row * dimension + assembled.column_indices()[entry]] =
                assembled.values()[entry];
        }
    }
    assert_close(&dense_assembled, &reference, 5.0e-11);
    assert_eq!(operator.symmetry(), methodus::OperatorSymmetry::Symmetric);
    check_properties_consistency(&operator).unwrap();

    let block_layout = operator.block_layout();
    assert_eq!(block_layout.blocks().len(), 2);
    assert_eq!(block_layout.dimension(), dimension);
}

/// Applies the identity-row/zero-column Dirichlet elimination transform directly to a dense
/// matrix: row `t` becomes the identity row, column `t` is zeroed everywhere else. This is the
/// dense-matrix definition of what `MixedOperator::apply_reduced_action` computes matrix-free,
/// derived independently in this test rather than by calling into `finitum::mixed`.
fn eliminate_dense(matrix: &[f64], dimension: usize, constrained: &[usize]) -> Vec<f64> {
    let mut eliminated = matrix.to_vec();
    for &t in constrained {
        for column in 0..dimension {
            eliminated[t * dimension + column] = if column == t { 1.0 } else { 0.0 };
        }
        for row in 0..dimension {
            if row != t {
                eliminated[row * dimension + t] = 0.0;
            }
        }
    }
    eliminated
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

/// Independent monolithic reference, built with a caller-supplied quadrature rule and this
/// test's own nested loops -- deliberately not sharing code with `finitum::mixed`'s (private)
/// local-matrix builders, which use a different (6-point degree-4) quadrature rule. Callers pick
/// a rule exact enough for their fixture's integrand degree; see
/// [`independent_triangle_quadrature`] and [`independent_higher_order_triangle_quadrature`].
fn independent_reference_matrix(
    space: &MixedSpace,
    couplings: &[BlockCoupling],
    quadrature: &[(f64, f64, f64)],
) -> Vec<f64> {
    let dimension = space.layout().extent();
    let mesh_dimension = space.mesh().dimension();
    let mut matrix = vec![0.0; dimension * dimension];
    for cell in 0..space.mesh().cells().len() {
        let map = AffineMap::from_cell(space.mesh(), CellId(cell)).unwrap();
        for coupling in couplings {
            match coupling.kind {
                CouplingKind::GradientGradient => {
                    let field = space.field(coupling.test).unwrap();
                    let block = space.layout().block(coupling.test).unwrap();
                    let restriction = &space.dof_map(coupling.test).unwrap().restrictions()[cell];
                    let basis_count = restriction.dofs.len() / field.components;
                    for &(x, y, weight) in quadrature {
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
                    for &(x, y, weight) in quadrature {
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

/// A 9-point triangle quadrature built by a Duffy (collapsed-square) transform of 3-point
/// Gauss-Legendre in each direction (`x = s * (1 - t)`, `y = t`, Jacobian `1 - t`) -- a
/// fundamentally different construction from both [`independent_triangle_quadrature`]'s
/// symmetric barycentric rule and `finitum::element`'s own 6-point Dunavant-style symmetric
/// rule. Exact well beyond the degree-3 integrand a P2-vector/P2-scalar `DivergenceValue`
/// coupling produces (`div(P2) * P2` has total degree `1 + 2 = 3`), needed because
/// [`independent_triangle_quadrature`] is calibrated only for the degree-2 integrands the
/// crate's other fixtures produce.
fn independent_higher_order_triangle_quadrature() -> Vec<(f64, f64, f64)> {
    // 3-point Gauss-Legendre nodes/weights on [-1, 1], mapped to [0, 1].
    let root = (3.0_f64 / 5.0).sqrt();
    let nodes_1d = [(-root, 5.0 / 9.0), (0.0, 8.0 / 9.0), (root, 5.0 / 9.0)];
    let mut points = Vec::with_capacity(9);
    for (gs, ws) in nodes_1d {
        let s = (gs + 1.0) / 2.0;
        let weight_s = ws / 2.0;
        for (gt, wt) in nodes_1d {
            let t = (gt + 1.0) / 2.0;
            let weight_t = wt / 2.0;
            let x = s * (1.0 - t);
            let y = t;
            let weight = weight_s * weight_t * (1.0 - t);
            points.push((x, y, weight));
        }
    }
    points
}
