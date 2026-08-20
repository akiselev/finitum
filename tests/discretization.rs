use finitum::{
    AffineConstraint, Cell, ConstraintSet, DofId, DofMap, ElementRestriction, FinitumError, Mesh,
    PreparedElement, QuadraturePoint, VertexId, WeightedDof,
};

#[test]
fn fixture_triangle_and_constraints_validate() {
    let mesh = Mesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
        }],
    )
    .unwrap();
    assert_eq!(mesh.cells().len(), 1);

    let dofs = DofMap::new(
        3,
        vec![ElementRestriction {
            dofs: vec![DofId(0), DofId(1), DofId(2)],
        }],
    )
    .unwrap();
    assert_eq!(dofs.dof_count(), 3);

    let constraints = ConstraintSet::new(
        3,
        [AffineConstraint {
            target: DofId(2),
            dependencies: vec![
                WeightedDof {
                    dof: DofId(0),
                    weight: 0.5,
                },
                WeightedDof {
                    dof: DofId(1),
                    weight: 0.5,
                },
            ],
            offset: 0.0,
        }],
    )
    .unwrap();
    assert_eq!(constraints.expand(&[2.0, 4.0, 0.0]).unwrap()[2], 3.0);
    assert_eq!(
        constraints.expand(&[2.0]).unwrap_err(),
        FinitumError::ConstraintInputLength {
            actual: 1,
            expected: 3,
        }
    );
}

#[test]
fn mesh_rejects_repeated_simplex_vertices_and_nonfinite_coordinates() {
    let duplicate = Mesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(1)],
        }],
    )
    .unwrap_err();
    assert_eq!(
        duplicate,
        FinitumError::DuplicateCellVertex { cell: 0, vertex: 1 }
    );

    let nonfinite = Mesh::new(1, vec![vec![f64::NAN], vec![1.0]], Vec::new()).unwrap_err();
    assert_eq!(
        nonfinite,
        FinitumError::NonFiniteCoordinate { vertex: 0, axis: 0 }
    );
}

#[test]
fn dof_map_rejects_empty_and_repeated_local_restrictions() {
    assert_eq!(
        DofMap::new(1, vec![ElementRestriction { dofs: Vec::new() }]).unwrap_err(),
        FinitumError::EmptyRestriction { restriction: 0 }
    );
    assert_eq!(
        DofMap::new(
            2,
            vec![ElementRestriction {
                dofs: vec![DofId(0), DofId(0)],
            }],
        )
        .unwrap_err(),
        FinitumError::DuplicateRestrictionDof {
            restriction: 0,
            dof: 0,
        }
    );
}

#[test]
fn constraint_set_rejects_ambiguous_or_nonfinite_data() {
    assert_eq!(
        ConstraintSet::new(
            1,
            [AffineConstraint {
                target: DofId(0),
                dependencies: Vec::new(),
                offset: f64::INFINITY,
            }],
        )
        .unwrap_err(),
        FinitumError::InvalidConstraintCoefficient { target: 0 }
    );

    let duplicate_dependency = ConstraintSet::new(
        2,
        [AffineConstraint {
            target: DofId(1),
            dependencies: vec![
                WeightedDof {
                    dof: DofId(0),
                    weight: 0.5,
                },
                WeightedDof {
                    dof: DofId(0),
                    weight: 0.5,
                },
            ],
            offset: 0.0,
        }],
    )
    .unwrap_err();
    assert_eq!(
        duplicate_dependency,
        FinitumError::DuplicateConstraintDependency {
            target: 1,
            dependency: 0,
        }
    );

    assert_eq!(
        ConstraintSet::new(
            2,
            [
                AffineConstraint {
                    target: DofId(0),
                    dependencies: vec![WeightedDof {
                        dof: DofId(1),
                        weight: 1.0,
                    }],
                    offset: 0.0,
                },
                AffineConstraint {
                    target: DofId(1),
                    dependencies: vec![WeightedDof {
                        dof: DofId(0),
                        weight: 1.0,
                    }],
                    offset: 0.0,
                },
            ],
        )
        .unwrap_err(),
        FinitumError::ConstraintCycle(0)
    );

    let constraints = ConstraintSet::new(
        2,
        [AffineConstraint {
            target: DofId(1),
            dependencies: vec![WeightedDof {
                dof: DofId(0),
                weight: f64::MAX,
            }],
            offset: 0.0,
        }],
    )
    .unwrap();
    assert_eq!(
        constraints.expand(&[f64::NAN, 0.0]).unwrap_err(),
        FinitumError::NonFiniteConstraintInput(0)
    );
    assert_eq!(
        constraints.expand(&[f64::MAX, 0.0]).unwrap_err(),
        FinitumError::NonFiniteConstraintResult(1)
    );
}

#[test]
fn prepared_element_rejects_shape_and_nonfinite_tables() {
    let quadrature = vec![QuadraturePoint {
        coordinates: vec![0.25, 0.25],
        weight: 0.5,
    }];
    assert!(matches!(
        PreparedElement::new(2, 3, quadrature.clone(), vec![1.0; 2], vec![0.0; 6]),
        Err(FinitumError::InvalidElementShape(_))
    ));
    assert!(matches!(
        PreparedElement::new(4, 1, quadrature.clone(), vec![1.0], vec![0.0; 4]),
        Err(FinitumError::InvalidElementShape(_))
    ));
    assert_eq!(
        PreparedElement::new(
            2,
            3,
            quadrature,
            vec![f64::INFINITY, 0.0, 0.0],
            vec![0.0; 6],
        )
        .unwrap_err(),
        FinitumError::NonFiniteElementData {
            location: "basis value 0".into(),
        }
    );
}
