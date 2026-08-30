//! GX-C1: `MeshProfile` realization tests — structured simplex boxes and CAD-provider
//! wrappers under one `TaggedMesh` contract.

use std::collections::BTreeSet;

use cadabra_provider::{
    AnalyticFamily, AnalyticProvider, DifferentiabilityDisposition, FamilyRequest, ProviderFrame,
    RectangleProvider, RectangleRequest,
};
use finitum::{
    AffineMap, CellId, FacetTopology, FinitumError, Mesh, MeshProfile, MeshProvenance, RegionTagId,
    realize, refine_uniform,
};

fn rectangle(revision: u64) -> RectangleProvider {
    let request = RectangleRequest::try_new(
        "fixture/gx-c1/plate",
        revision,
        ProviderFrame::world(),
        2.0,
        1.0,
    )
    .expect("valid rectangle request");
    match RectangleProvider::admit(request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth rectangle, got {other:?}"),
    }
}

fn annulus(inner: f64, outer: f64, revision: u64) -> AnalyticProvider {
    let request = FamilyRequest::try_new(
        "fixture/gx-c1/ring",
        revision,
        ProviderFrame::world(),
        AnalyticFamily::Annulus {
            inner_radius: inner,
            outer_radius: outer,
        },
    )
    .expect("valid annulus request");
    match AnalyticProvider::admit(request) {
        DifferentiabilityDisposition::Smooth { value, .. } => value,
        other => panic!("expected smooth annulus, got {other:?}"),
    }
}

fn assert_positive_orientation(mesh: &Mesh) {
    for cell in 0..mesh.cells().len() {
        let map = AffineMap::from_cell(mesh, CellId(cell)).expect("affine map");
        assert!(
            map.determinant() > 0.0,
            "cell {cell} has non-positive determinant {}",
            map.determinant()
        );
    }
}

fn total_measure(mesh: &Mesh) -> f64 {
    (0..mesh.cells().len())
        .map(|cell| {
            AffineMap::from_cell(mesh, CellId(cell))
                .unwrap()
                .volume_scale()
        })
        .sum()
}

#[test]
fn segment_box_has_expected_topology_and_orientation() {
    let profile = MeshProfile::SimplexBox {
        dimension: 1,
        extent: vec![[0.0, 2.0]],
        subdivisions: vec![4],
    };
    let tagged = realize(&profile).unwrap();
    assert_eq!(tagged.mesh.vertices().len(), 5);
    assert_eq!(tagged.mesh.cells().len(), 4);
    assert_positive_orientation(&tagged.mesh);
    assert_eq!(tagged.tags.cell_regions.len(), 4);
    assert!(
        tagged
            .tags
            .cell_regions
            .iter()
            .all(|tag| tag.as_str() == "interior")
    );
    assert_eq!(
        tagged.tags.facet_regions[&RegionTagId::new("x_min")].len(),
        1
    );
    assert_eq!(
        tagged.tags.facet_regions[&RegionTagId::new("x_max")].len(),
        1
    );
    assert_eq!(tagged.provenance, MeshProvenance::Profile);
}

#[test]
fn square_box_matches_sinbad_square_discretization() {
    let n = 3usize;
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![n, n],
    };
    let tagged = realize(&profile).unwrap();
    assert_positive_orientation(&tagged.mesh);

    // Reproduces sinbad's poisson.rs `square_discretization` inline: same row-major,
    // x-fastest vertex order and [ll,lr,ur]/[ll,ur,ul] triangle pattern, so `SimplexBox`
    // realizations stay digest-comparable with that hand-rolled mesh on the same input.
    let width = n + 1;
    let mut expected_vertices = Vec::new();
    for row in 0..=n {
        for column in 0..=n {
            expected_vertices.push(vec![column as f64 / n as f64, row as f64 / n as f64]);
        }
    }
    let mut expected_cells = Vec::new();
    for row in 0..n {
        for column in 0..n {
            let lower_left = row * width + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            expected_cells.push(vec![lower_left, lower_right, upper_right]);
            expected_cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    assert_eq!(tagged.mesh.vertices(), expected_vertices.as_slice());
    let actual_cells = tagged
        .mesh
        .cells()
        .iter()
        .map(|cell| {
            cell.vertices
                .iter()
                .map(|vertex| vertex.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual_cells, expected_cells);
}

#[test]
fn cube_box_matches_kuhn_decomposition_as_tet_vertex_sets() {
    let n = 2usize;
    let profile = MeshProfile::SimplexBox {
        dimension: 3,
        extent: vec![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![n, n, n],
    };
    let tagged = realize(&profile).unwrap();
    assert_positive_orientation(&tagged.mesh);
    assert_eq!(tagged.mesh.cells().len(), n * n * n * 6);

    // Reproduces tests/sv2_elasticity.rs's `brick_tets`/`cube_mesh` construction and compares
    // realized tetrahedra as vertex SETS per brick: `SimplexBox` reorders two of the six tets'
    // last two vertices for positive orientation, so exact tuple order is not preserved, but
    // the six tetrahedra themselves (and their shared main diagonal) are identical.
    let width = n + 1;
    let index = |x: usize, y: usize, z: usize| z * width * width + y * width + x;
    fn brick_tets(o: [usize; 3]) -> Vec<[[usize; 3]; 4]> {
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
    let mut expected_bricks = Vec::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let mut brick = brick_tets([x, y, z])
                    .into_iter()
                    .map(|tet| {
                        tet.into_iter()
                            .map(|[cx, cy, cz]| index(cx, cy, cz))
                            .collect::<BTreeSet<_>>()
                    })
                    .collect::<Vec<_>>();
                brick.sort();
                expected_bricks.push(brick);
            }
        }
    }
    let mut actual_bricks = Vec::new();
    for chunk in tagged.mesh.cells().chunks(6) {
        let mut brick = chunk
            .iter()
            .map(|cell| {
                cell.vertices
                    .iter()
                    .map(|vertex| vertex.0)
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        brick.sort();
        actual_bricks.push(brick);
    }
    assert_eq!(actual_bricks, expected_bricks);
}

#[test]
fn box_realization_refuses_the_one_million_item_cap() {
    let profile = MeshProfile::SimplexBox {
        dimension: 3,
        extent: vec![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![200, 200, 200],
    };
    assert!(matches!(
        realize(&profile),
        Err(FinitumError::MeshProfileUnsupported(_))
    ));
}

#[test]
fn box_realization_refuses_invalid_extent_and_subdivisions() {
    let bad_extent = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[1.0, 0.0], [0.0, 1.0]],
        subdivisions: vec![2, 2],
    };
    assert!(realize(&bad_extent).is_err());

    let bad_subdivisions = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![0, 2],
    };
    assert!(realize(&bad_subdivisions).is_err());

    let wrong_arity = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]],
        subdivisions: vec![2, 2],
    };
    assert!(realize(&wrong_arity).is_err());
}

#[test]
fn cad_rectangle_profile_tags_boundaries_and_regions() {
    let provider = rectangle(1);
    let profile = MeshProfile::CadRectangle {
        provider: provider.clone(),
        expected_revision: 1,
        subdivisions: [3, 2],
    };
    let tagged = realize(&profile).unwrap();
    assert_positive_orientation(&tagged.mesh);
    assert_eq!(tagged.tags.cell_regions.len(), tagged.mesh.cells().len());
    assert!(
        tagged
            .tags
            .cell_regions
            .iter()
            .all(|tag| tag.as_str() == provider.snapshot().region.id.as_str())
    );
    for boundary in &provider.snapshot().boundaries {
        let tag = RegionTagId::new(boundary.id.as_str());
        assert!(!tagged.tags.facet_regions[&tag].is_empty());
    }
}

#[test]
fn cad_family_profile_tags_annulus_boundaries_as_closed_loops() {
    let provider = annulus(0.5, 2.0, 1);
    let profile = MeshProfile::CadFamily {
        provider: provider.clone(),
        expected_revision: 1,
        subdivisions: [3, 8],
    };
    let tagged = realize(&profile).unwrap();
    assert_positive_orientation(&tagged.mesh);
    for boundary in &provider.snapshot().boundaries {
        let tag = RegionTagId::new(boundary.id.as_str());
        // Angular direction is periodic; every one of the 8 angular columns contributes one
        // exterior facet, including the wrap-around edge back to column zero.
        assert_eq!(tagged.tags.facet_regions[&tag].len(), 8);
    }
}

#[test]
fn refine_uniform_preserves_measure_and_tag_partition_across_dimensions() {
    let profiles = [
        MeshProfile::SimplexBox {
            dimension: 1,
            extent: vec![[0.0, 3.0]],
            subdivisions: vec![3],
        },
        MeshProfile::SimplexBox {
            dimension: 2,
            extent: vec![[0.0, 1.0], [0.0, 2.0]],
            subdivisions: vec![2, 3],
        },
        MeshProfile::SimplexBox {
            dimension: 3,
            extent: vec![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            subdivisions: vec![2, 2, 2],
        },
    ];
    for profile in profiles {
        let coarse = realize(&profile).unwrap();
        let fine = refine_uniform(&coarse).unwrap();
        assert_positive_orientation(&fine.mesh);
        assert_eq!(
            fine.mesh.cells().len(),
            coarse.mesh.cells().len() * 2usize.pow(coarse.mesh.dimension() as u32)
        );
        assert!((total_measure(&coarse.mesh) - total_measure(&fine.mesh)).abs() < 1.0e-9);

        for tagged in [&coarse, &fine] {
            let exterior_count = FacetTopology::from_mesh(&tagged.mesh)
                .unwrap()
                .exterior()
                .count();
            let total_tagged: usize = tagged.tags.facet_regions.values().map(Vec::len).sum();
            assert_eq!(
                total_tagged, exterior_count,
                "every exterior facet must be tagged by exactly one box face"
            );
        }
        assert_eq!(
            fine.provenance,
            MeshProvenance::Refined {
                parent: coarse.digest.clone(),
                level: 1,
            }
        );

        let refined_again = refine_uniform(&fine).unwrap();
        assert_eq!(refined_again.provenance.level(), 2);
    }
}

#[test]
fn refine_uniform_refuses_non_simplex_and_inconsistent_tags() {
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![2, 2],
    };
    let mut tagged = realize(&profile).unwrap();
    tagged.tags.cell_regions.pop();
    assert!(matches!(
        refine_uniform(&tagged),
        Err(FinitumError::MeshProfileUnsupported(_))
    ));
}
