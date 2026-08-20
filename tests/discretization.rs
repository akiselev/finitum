use finitum::{
    AffineConstraint, Cell, ConstraintSet, DofId, DofMap, ElementRestriction, Mesh, VertexId,
    WeightedDof,
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
}
