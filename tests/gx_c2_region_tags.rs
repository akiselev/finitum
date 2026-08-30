//! GX-C2: `RegionTags`-derived essential constraints and boundary-partition checks.

use finitum::{
    DofId, FieldSource, FinitumError, MeshProfile, RegionMap, RegionTagId,
    check_boundary_partition, essential_constraints_from, realize, vector_nodal_dof_map,
};
use scientia::{
    BoundaryPartitionRequirement, DeclarationId, DomainId, EssentialConstraintRequirement,
    RegionId, SymbolId,
};

fn square(n: usize) -> finitum::TaggedMesh {
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![n, n],
    };
    realize(&profile).unwrap()
}

fn boundary_vertices(n: usize) -> Vec<usize> {
    let width = n + 1;
    (0..width * width)
        .filter(|&index| {
            let row = index / width;
            let column = index % width;
            row == 0 || row == n || column == 0 || column == n
        })
        .collect()
}

fn all_faces_map(region: RegionId) -> RegionMap {
    let mut map = RegionMap::new();
    map.insert(
        region,
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("x_max"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
        ],
    );
    map
}

#[test]
fn essential_constraints_from_tags_matches_index_arithmetic_boundary_set() {
    let n = 4usize;
    let mesh = square(n);
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 1).unwrap();
    let region_map = all_faces_map(RegionId(0));
    let requirement = EssentialConstraintRequirement {
        argument: SymbolId(0),
        region: RegionId(0),
        condition: DeclarationId(0),
    };
    let constraints = essential_constraints_from(
        &mesh,
        &dof_map,
        &[requirement],
        &region_map,
        &[FieldSource::constant(vec![5.0])],
    )
    .unwrap();

    let expected = boundary_vertices(n);
    let mut actual_targets = constraints
        .constraints()
        .map(|constraint| constraint.target.0)
        .collect::<Vec<_>>();
    actual_targets.sort_unstable();
    assert_eq!(actual_targets, expected);
    for constraint in constraints.constraints() {
        assert!(constraint.dependencies.is_empty());
        assert_eq!(constraint.offset, 5.0);
    }
}

#[test]
fn essential_constraints_from_supports_vector_fields_and_sampled_sources() {
    let n = 3usize;
    let mesh = square(n);
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 2).unwrap();
    // x_min and x_max are opposite (disjoint) faces, so the two requirements below never
    // contest the same vertex.
    let mut region_map = RegionMap::new();
    region_map.insert(RegionId(0), [RegionTagId::new("x_min")]);
    region_map.insert(RegionId(1), [RegionTagId::new("x_max")]);
    let requirements = vec![
        EssentialConstraintRequirement {
            argument: SymbolId(0),
            region: RegionId(0),
            condition: DeclarationId(0),
        },
        EssentialConstraintRequirement {
            argument: SymbolId(1),
            region: RegionId(1),
            condition: DeclarationId(1),
        },
    ];
    let values = vec![
        FieldSource::constant(vec![1.0, 2.0]),
        FieldSource::sampled(|coordinates: &[f64]| vec![coordinates[0], coordinates[1]]),
    ];
    let constraints =
        essential_constraints_from(&mesh, &dof_map, &requirements, &region_map, &values).unwrap();

    // x_min vertices: column 0, every row; constant vector [1.0, 2.0].
    let width = n + 1;
    for row in 0..=n {
        let vertex = row * width;
        let zero = vec![0.0; dof_map.dof_count()];
        let residual_x = constraints
            .equation_residual(&zero, DofId(vertex * 2))
            .unwrap();
        let residual_y = constraints
            .equation_residual(&zero, DofId(vertex * 2 + 1))
            .unwrap();
        assert!((residual_x + 1.0).abs() < 1.0e-12);
        assert!((residual_y + 2.0).abs() < 1.0e-12);
    }

    // x_max vertices: column n, sampled coordinate value equals the vertex's own coordinates.
    for row in 0..=n {
        let vertex = row * width + n;
        let y = row as f64 / n as f64;
        let zero = vec![0.0; dof_map.dof_count()];
        let residual_x = constraints
            .equation_residual(&zero, DofId(vertex * 2))
            .unwrap();
        let residual_y = constraints
            .equation_residual(&zero, DofId(vertex * 2 + 1))
            .unwrap();
        assert!((residual_x + 1.0).abs() < 1.0e-12);
        assert!((residual_y + y).abs() < 1.0e-12);
    }
}

#[test]
fn essential_constraints_from_refuses_unmapped_region() {
    let mesh = square(2);
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 1).unwrap();
    let region_map = RegionMap::new();
    let requirement = EssentialConstraintRequirement {
        argument: SymbolId(0),
        region: RegionId(0),
        condition: DeclarationId(0),
    };
    let result = essential_constraints_from(
        &mesh,
        &dof_map,
        &[requirement],
        &region_map,
        &[FieldSource::constant(vec![1.0])],
    );
    assert!(matches!(
        result,
        Err(FinitumError::RealizationRegionUnmapped(_))
    ));
}

#[test]
fn essential_constraints_from_refuses_conflicting_corner_values() {
    let mesh = square(2);
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 1).unwrap();
    let mut region_map = RegionMap::new();
    region_map.insert(RegionId(0), [RegionTagId::new("x_min")]);
    region_map.insert(RegionId(1), [RegionTagId::new("y_min")]);
    let requirements = vec![
        EssentialConstraintRequirement {
            argument: SymbolId(0),
            region: RegionId(0),
            condition: DeclarationId(0),
        },
        EssentialConstraintRequirement {
            argument: SymbolId(0),
            region: RegionId(1),
            condition: DeclarationId(1),
        },
    ];
    // x_min and y_min share the corner vertex (0, 0); different constants conflict there.
    let values = vec![
        FieldSource::constant(vec![1.0]),
        FieldSource::constant(vec![2.0]),
    ];
    let result = essential_constraints_from(&mesh, &dof_map, &requirements, &region_map, &values);
    assert!(matches!(
        result,
        Err(FinitumError::ConflictingRegionValue { .. })
    ));
}

#[test]
fn check_boundary_partition_accepts_a_full_cover_and_refuses_gaps_and_overlaps() {
    let mesh = square(3);
    let full_map = all_faces_map(RegionId(0));
    let requirement = BoundaryPartitionRequirement {
        domain: DomainId(0),
        exterior_regions: vec![RegionId(0)],
    };
    let report = check_boundary_partition(&mesh, &requirement, &full_map).unwrap();
    assert!(report.exterior_facet_count > 0);
    assert_eq!(report.region_count, 1);

    let mut gapped_map = RegionMap::new();
    gapped_map.insert(
        RegionId(0),
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("x_max"),
            RegionTagId::new("y_min"),
            // y_max intentionally left uncovered.
        ],
    );
    let gap_result = check_boundary_partition(&mesh, &requirement, &gapped_map);
    assert!(matches!(
        gap_result,
        Err(FinitumError::RealizationPartitionFailed { uncovered, .. }) if uncovered > 0
    ));

    let mut overlapping_map = RegionMap::new();
    overlapping_map.insert(
        RegionId(0),
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("x_max"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
        ],
    );
    overlapping_map.insert(RegionId(1), [RegionTagId::new("x_min")]);
    let overlap_requirement = BoundaryPartitionRequirement {
        domain: DomainId(0),
        exterior_regions: vec![RegionId(0), RegionId(1)],
    };
    let overlap_result = check_boundary_partition(&mesh, &overlap_requirement, &overlapping_map);
    assert!(matches!(
        overlap_result,
        Err(FinitumError::RealizationPartitionFailed { overlapping, .. }) if overlapping > 0
    ));
}
