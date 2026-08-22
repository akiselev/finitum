use finitum::{
    AffineConstraint, Cell, ConstraintSet, DofId, ExactSequence, FacetTopology, Mesh,
    MeshRefinementSample, NonmatchingTransfer, PatchCheckReport, TransferConservationReport,
    VerificationCheckKind, VerificationSubject, VertexId, WeightedDof, check_constraint_work,
    check_exact_sequence, check_global_transpose, check_mesh_refinement, check_nodal_patch,
    check_transfer_conservation,
};
use methodus::{ComparisonTolerance, CsrMatrix};
use std::collections::HashMap;

const TOLERANCE: ComparisonTolerance = ComparisonTolerance {
    absolute: 1.0e-13,
    relative: 1.0e-13,
};

#[test]
fn exact_sequence_reports_dimension_complete_stages() {
    let triangle = Mesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
        }],
    )
    .unwrap();
    let topology = FacetTopology::from_mesh(&triangle).unwrap();
    let sequence = ExactSequence::simplex(&triangle, &topology).unwrap();
    let two_dimensional = check_exact_sequence(2, &sequence).unwrap();
    assert!(two_dimensional.validate(&sequence).unwrap().accepted);
    assert!(two_dimensional.body.stage_complete);
    assert_eq!(two_dimensional.body.expected_stage_count, 2);
    assert_eq!(two_dimensional.body.observed_stage_count, 2);
    assert_eq!(two_dimensional.body.divergence_shape, None);

    let tetrahedron = Mesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        }],
    )
    .unwrap();
    let topology = FacetTopology::from_mesh(&tetrahedron).unwrap();
    let complete = ExactSequence::simplex(&tetrahedron, &topology).unwrap();
    assert!(
        check_exact_sequence(3, &complete)
            .unwrap()
            .validate(&complete)
            .unwrap()
            .accepted
    );
    let mut missing_divergence = complete;
    missing_divergence.divergence = None;
    let missing = check_exact_sequence(3, &missing_divergence).unwrap();
    assert!(!missing.body.stage_complete);
    assert!(!missing.body.accepted);
    assert_eq!(missing.body.expected_stage_count, 3);
    assert_eq!(missing.body.observed_stage_count, 2);
    assert_eq!(missing.body.divergence_curl_zero, None);
}

#[test]
fn reports_are_canonical_kind_bound_and_tamper_evident() {
    let mut first_map = HashMap::new();
    first_map.insert("alpha", 1_u64);
    first_map.insert("beta", 2_u64);
    let mut opposite_order = HashMap::new();
    opposite_order.insert("beta", 2_u64);
    opposite_order.insert("alpha", 1_u64);
    assert_eq!(
        VerificationSubject::from_serializable("map", &first_map).unwrap(),
        VerificationSubject::from_serializable("map", &opposite_order).unwrap()
    );

    let constraints = ConstraintSet::new(
        3,
        [AffineConstraint {
            target: DofId(2),
            dependencies: vec![
                WeightedDof {
                    dof: DofId(0),
                    weight: 0.25,
                },
                WeightedDof {
                    dof: DofId(1),
                    weight: 0.75,
                },
            ],
            offset: 4.0,
        }],
    )
    .unwrap();
    let first = check_constraint_work(
        &constraints,
        &[1.2, -0.7, 99.0],
        &[0.3, 1.1, -2.0],
        TOLERANCE,
    )
    .unwrap();
    let repeated = check_constraint_work(
        &constraints,
        &[1.2, -0.7, 99.0],
        &[0.3, 1.1, -2.0],
        TOLERANCE,
    )
    .unwrap();
    assert!(first.validate(&constraints).unwrap().accepted);
    assert_eq!(first.header.report_digest, repeated.header.report_digest);

    let mut tampered = first.clone();
    tampered.body.unconstrained[0] += 0.5;
    assert!(tampered.validate(&constraints).is_err());
    let mut cross_kind = first.clone();
    cross_kind.header.check_kind = VerificationCheckKind::TransferConservation;
    assert!(cross_kind.validate(&constraints).is_err());

    let transfer = NonmatchingTransfer::lagrange(vec![0.0, 1.0], vec![0.0, 0.4, 1.0]).unwrap();
    let transfer_report = check_transfer_conservation(
        &transfer,
        &[1.0, 3.0],
        &[0.7, -1.2, 0.5],
        &[0.2, 0.6, 0.2],
        TOLERANCE,
    )
    .unwrap();
    assert!(transfer_report.validate(&transfer).unwrap().accepted);
    assert_ne!(first.header.check_kind, transfer_report.header.check_kind);
    assert_ne!(
        first.header.report_digest,
        transfer_report.header.report_digest
    );
    let constraint_json = serde_json::to_value(&first).unwrap();
    assert!(serde_json::from_value::<TransferConservationReport>(constraint_json).is_err());

    let patch_mesh = segment_mesh(2);
    let exact = [2.0, 3.0, 4.0];
    let patch = check_nodal_patch(&patch_mesh, 1, &exact, TOLERANCE, |point| {
        vec![2.0 + 2.0 * point[0]]
    })
    .unwrap();
    let encoded = serde_json::to_vec(&patch).unwrap();
    let decoded: PatchCheckReport = serde_json::from_slice(&encoded).unwrap();
    assert!(decoded.validate(&patch_mesh).unwrap().accepted);
    let mut changed = decoded;
    changed.body.comparison.accepted = false;
    assert!(changed.validate(&patch_mesh).is_err());
}

#[test]
fn synthetic_checks_detect_wrong_values_transpose_and_nonrefinement() {
    let mesh = segment_mesh(2);
    let wrong = [2.0, 3.25, 4.0];
    assert!(
        !check_nodal_patch(&mesh, 1, &wrong, TOLERANCE, |point| {
            vec![2.0 + 2.0 * point[0]]
        })
        .unwrap()
        .body
        .comparison
        .accepted
    );

    let forward = CsrMatrix::from_triplets(
        2,
        3,
        vec![(0, 0, 2.0), (0, 2, -1.0), (1, 1, 4.0), (1, 2, 0.5)],
    )
    .unwrap();
    let wrong_transpose = CsrMatrix::from_triplets(3, 2, vec![(0, 0, 2.0), (1, 1, 4.0)]).unwrap();
    let subject = VerificationSubject::from_serializable("hostile-transpose", &forward).unwrap();
    assert!(
        !check_global_transpose(
            subject,
            &forward,
            &wrong_transpose,
            &[0.3, -0.8],
            &[1.1, 0.2, -0.4],
            TOLERANCE,
        )
        .unwrap()
        .body
        .comparison
        .accepted
    );

    let coarse = segment_mesh(1);
    let medium = segment_mesh(2);
    let fine = segment_mesh(4);
    let refinement = check_mesh_refinement(
        &[
            MeshRefinementSample {
                mesh: &coarse,
                error: 1.0,
            },
            MeshRefinementSample {
                mesh: &medium,
                error: 0.25,
            },
            MeshRefinementSample {
                mesh: &fine,
                error: 0.0625,
            },
        ],
        1.9,
    )
    .unwrap();
    assert!(
        refinement
            .validate(&[
                MeshRefinementSample {
                    mesh: &coarse,
                    error: 1.0,
                },
                MeshRefinementSample {
                    mesh: &medium,
                    error: 0.25,
                },
                MeshRefinementSample {
                    mesh: &fine,
                    error: 0.0625,
                },
            ])
            .unwrap()
            .accepted
    );
    assert!((refinement.body.convergence.fitted_order - 2.0).abs() < 1.0e-14);
    assert!(
        check_mesh_refinement(
            &[
                MeshRefinementSample {
                    mesh: &coarse,
                    error: 1.0,
                },
                MeshRefinementSample {
                    mesh: &coarse,
                    error: 0.5,
                },
            ],
            1.0,
        )
        .is_err()
    );
}

fn segment_mesh(cells: usize) -> Mesh {
    Mesh::new(
        1,
        (0..=cells)
            .map(|index| vec![index as f64 / cells as f64])
            .collect(),
        (0..cells)
            .map(|index| Cell {
                vertices: vec![VertexId(index), VertexId(index + 1)],
            })
            .collect(),
    )
    .unwrap()
}
