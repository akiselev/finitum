use finitum::{
    CellId, FieldSampler, LinearFieldSamples, MeshProfile, ResultMesh, SampledFamily,
    inspection_surface, realize,
};
#[test]
fn saved_scalar_and_vector_fields_and_cut_surfaces_match_owner_sampling() {
    for dimension in [2usize, 3] {
        for components in [1, dimension] {
            let tagged = realize(&MeshProfile::SimplexBox {
                dimension: dimension as u8,
                extent: vec![[0.0, 1.0]; dimension],
                subdivisions: vec![2; dimension],
            })
            .unwrap();
            let mesh = &tagged.mesh;
            let values: Vec<_> = mesh
                .vertices()
                .iter()
                .flat_map(|p| {
                    (0..components).map(move |c| {
                        2.0 + c as f64
                            + p.iter()
                                .enumerate()
                                .map(|(i, x)| (i + 1) as f64 * x)
                                .sum::<f64>()
                    })
                })
                .collect();
            let sampler = FieldSampler::new(
                mesh,
                SampledFamily::Lagrange {
                    order: 1,
                    components,
                },
                &values,
            )
            .unwrap();
            let saved = LinearFieldSamples::capture(&sampler).unwrap();
            let geometry = ResultMesh::capture(mesh);
            saved.validate(&geometry).unwrap();
            for cut in [None, Some(0.0), Some(0.37), Some(0.5), Some(1.0)] {
                let surface = inspection_surface(&geometry, cut).unwrap();
                if cut != Some(0.0) {
                    assert!(!surface.triangles.is_empty());
                }
                for p in surface.points {
                    assert!(cut.is_none_or(|cut| p.coordinates[0] <= cut + 1e-12));
                    let actual = saved.value_at_reference(p.cell, &p.reference).unwrap();
                    let reference = sampler.value_at(CellId(p.cell), &p.coordinates).unwrap();
                    for (a, b) in actual.iter().zip(reference) {
                        assert!((a - b).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
#[test]
fn mismatched_mesh_invalid_extents_and_unsupported_basis_refuse() {
    let mesh = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]; 2],
        subdivisions: vec![1; 2],
    })
    .unwrap()
    .mesh;
    let values = vec![1.0; mesh.vertices().len()];
    let sampler = FieldSampler::new(
        &mesh,
        SampledFamily::Lagrange {
            order: 1,
            components: 1,
        },
        &values,
    )
    .unwrap();
    let saved = LinearFieldSamples::capture(&sampler).unwrap();
    let geometry = ResultMesh::capture(&mesh);
    let mut bad = saved.clone();
    bad.mesh_identity.push('x');
    assert!(bad.validate(&geometry).is_err());
    let mut bad = saved.clone();
    bad.cell_values[0].pop();
    assert!(bad.validate(&geometry).is_err());
    let mut bad = saved;
    bad.cell_values[0][0][0] = f64::NAN;
    assert!(bad.validate(&geometry).is_err());
    let values = vec![0.0; mesh.cells().len()];
    let p0 = FieldSampler::new(&mesh, SampledFamily::CellConstant, &values).unwrap();
    assert!(LinearFieldSamples::capture(&p0).is_err());
    let mut bad = geometry;
    bad.cells[0][0] = 999;
    assert!(inspection_surface(&bad, None).is_err());
}
