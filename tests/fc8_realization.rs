use finitum::{
    AffineMap, BlockLayout, Cell, CellId, CompatibleDofMaps, ExactSequence, FacetTopology, Mesh,
    VertexId, static_condense,
};
use resolvent::SymbolId;

fn tetra_pair() -> Mesh {
    Mesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![0.5, 3.0, 0.0],
            vec![0.0, 0.25, 4.0],
            vec![0.0, 0.0, -1.0],
        ],
        vec![
            Cell {
                vertices: vec![VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
            },
            Cell {
                vertices: vec![VertexId(0), VertexId(2), VertexId(1), VertexId(4)],
            },
        ],
    )
    .unwrap()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[test]
fn elasticity_vector_tensor_blocks_preserve_component_ownership() {
    let displacement = SymbolId(0);
    let stress = SymbolId(1);
    let layout = BlockLayout::new([(displacement, 4, 3), (stress, 2, 6)]).unwrap();
    assert_eq!(layout.extent(), 24);
    let mut global = (0..layout.extent())
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    assert_eq!(
        layout.gather(&global, displacement, &[1, 3]).unwrap(),
        vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
    );
    layout
        .scatter_add(&mut global, stress, &[1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    assert_eq!(
        layout.values(&global, stress).unwrap()[6..],
        [19.0, 21.0, 23.0, 25.0, 27.0, 29.0]
    );
}

#[test]
fn stokes_product_layout_and_static_condensation_match_the_full_local_system() {
    let velocity = SymbolId(0);
    let pressure = SymbolId(1);
    let layout = BlockLayout::new([(velocity, 3, 2), (pressure, 1, 1)]).unwrap();
    assert_eq!(layout.extent(), 7);
    assert_eq!(layout.block(velocity).unwrap().component_count, 2);
    assert_eq!(layout.block(pressure).unwrap().offset, 6);

    let matrix = vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0];
    let rhs = vec![1.0, 2.0, 3.0];
    let condensed = static_condense(3, &matrix, &rhs, &[0]).unwrap();
    assert_eq!(condensed.trace_dofs, [1, 2]);
    assert_eq!(condensed.schur, [2.75, -0.25, -0.25, 1.75]);
    assert_eq!(condensed.rhs, [1.75, 2.75]);
    let trace = [0.5, 1.5];
    let complete = condensed.recover(&trace).unwrap();
    assert!((complete[0] + 0.25).abs() < 1e-14);
    let full_residual = (0..3)
        .map(|row| {
            (0..3)
                .map(|column| matrix[row * 3 + column] * complete[column])
                .sum::<f64>()
                - rhs[row]
        })
        .collect::<Vec<_>>();
    assert!(full_residual[0].abs() < 1e-14);
    for row in 0..2 {
        let condensed_residual = (0..2)
            .map(|column| condensed.schur[row * 2 + column] * trace[column])
            .sum::<f64>()
            - condensed.rhs[row];
        assert!((condensed_residual - full_residual[condensed.trace_dofs[row]]).abs() < 1e-14);
    }
}

#[test]
fn darcy_hdiv_mapping_preserves_flux_and_oriented_shared_facets() {
    let mesh = tetra_pair();
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    let compatible = CompatibleDofMaps::simplex(&mesh, &topology).unwrap();
    assert_eq!(compatible.hcurl_dof_count, 9);
    assert_eq!(compatible.hdiv_dof_count, 7);
    let shared = topology.interior().next().unwrap();
    assert_eq!(shared.incidences.len(), 2);
    assert_eq!(
        shared.incidences[0].orientation,
        -shared.incidences[1].orientation
    );
    let shared_dof = shared.id.0;
    let signs = compatible
        .hdiv
        .iter()
        .map(|restriction| {
            let local = restriction
                .dofs
                .iter()
                .position(|dof| dof.0 == shared_dof)
                .unwrap();
            restriction.orientations[local]
        })
        .collect::<Vec<_>>();
    assert_eq!(signs[0], -signs[1]);

    let map = AffineMap::from_cell(&mesh, CellId(0)).unwrap();
    let reference_value = [1.5, -0.25, 2.0];
    let reference_normal = [0.0, 0.0, 1.0];
    let physical_value = map.contravariant_piola(&reference_value).unwrap();
    let physical_scaled_normal = map.covariant_piola(&reference_normal).unwrap();
    let physical_scaled_normal = physical_scaled_normal
        .iter()
        .map(|value| value * map.determinant())
        .collect::<Vec<_>>();
    assert!(
        (dot(&physical_value, &physical_scaled_normal) - dot(&reference_value, &reference_normal))
            .abs()
            < 1e-13
    );
    assert!((map.map_hdiv_divergence(3.0) * map.determinant() - 3.0).abs() < 1e-13);
}

#[test]
fn dg_transport_uses_each_interior_facet_once_with_conservative_oriented_scatter() {
    let mesh = Mesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ],
        vec![
            Cell {
                vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
            },
            Cell {
                vertices: vec![VertexId(1), VertexId(3), VertexId(2)],
            },
        ],
    )
    .unwrap();
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    assert_eq!(topology.interior().count(), 1);
    assert_eq!(topology.exterior().count(), 4);
    let facet = topology.interior().next().unwrap();
    let forward = topology
        .oriented_pair(facet.id, facet.incidences[0].cell)
        .unwrap();
    let reverse = topology
        .oriented_pair(facet.id, facet.incidences[1].cell)
        .unwrap();
    assert_eq!(forward.minus, reverse.plus);
    assert_eq!(forward.plus, reverse.minus);
    assert_eq!(forward.relative_orientation(), -1);
    let minus_state = 3.0;
    let plus_state = 1.0;
    let normal_speed = 2.0;
    let numerical_flux = normal_speed * (minus_state - plus_state);
    let mut residual = [0.0; 2];
    residual[facet.minus().cell.0] += numerical_flux;
    residual[facet.plus().unwrap().cell.0] -= numerical_flux;
    assert_eq!(residual, [4.0, -4.0]);
    assert_eq!(residual.iter().sum::<f64>(), 0.0);
}

#[test]
fn maxwell_hcurl_mapping_and_exact_sequence_commute() {
    let mesh = tetra_pair();
    let topology = FacetTopology::from_mesh(&mesh).unwrap();
    let sequence = ExactSequence::simplex(&mesh, &topology).unwrap();
    assert!(sequence.curl.product_is_zero(&sequence.gradient));
    assert_eq!(
        sequence.gradient.rank() + sequence.curl.rank(),
        sequence.gradient.rows()
    );
    assert!(
        sequence
            .divergence
            .as_ref()
            .unwrap()
            .product_is_zero(&sequence.curl)
    );
    assert_eq!(
        sequence.curl.rank() + sequence.divergence.as_ref().unwrap().rank(),
        sequence.curl.rows()
    );

    let map = AffineMap::from_cell(&mesh, CellId(0)).unwrap();
    let reference_value = [0.75, -1.25, 0.5];
    let reference_tangent = [0.2, 0.3, -0.1];
    let physical_value = map.covariant_piola(&reference_value).unwrap();
    let origin = map.physical_point(&[0.0, 0.0, 0.0]).unwrap();
    let endpoint = map.physical_point(&reference_tangent).unwrap();
    let physical_tangent = endpoint
        .iter()
        .zip(origin)
        .map(|(endpoint, origin)| endpoint - origin)
        .collect::<Vec<_>>();
    assert!(
        (dot(&physical_value, &physical_tangent) - dot(&reference_value, &reference_tangent)).abs()
            < 1e-13
    );
    let mapped_curl = map.map_hcurl_curl(&[1.0, 2.0, 3.0]).unwrap();
    assert!(mapped_curl.iter().all(|value| value.is_finite()));
}
