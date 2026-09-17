use finitum::{
    BlockLayout, ConnectionRealizationPlan, MeshProfile, RegionTagId, SystemQuadrature,
    SystemRealizationPlan, realize,
};
use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    NoImports, Registries, compile_system, compile_system_operator, resolve_module_closure,
};
fn meshes(n: usize) -> [finitum::TaggedMesh; 2] {
    [0, 1].map(|side| {
        realize(&MeshProfile::SimplexBox {
            dimension: 3,
            extent: vec![
                [side as f64 * 0.5, (side + 1) as f64 * 0.5],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            subdivisions: vec![n; 3],
        })
        .unwrap()
    })
}
#[test]
fn matching_checks_coverage_normals_mesh_identity_and_open_boundary_admission() {
    let closure = resolve_module_closure(
        include_str!("fixtures/two-material-conduction.res"),
        &NoImports,
    )
    .unwrap();
    let compiled = compile_system(
        &closure,
        Registries::new(
            &UnitRegistry::si_bootstrap(),
            &QuantityKindRegistry::si_bootstrap(),
        ),
        "TwoMaterials",
    )
    .unwrap();
    let operators = compile_system_operator(&compiled).unwrap();
    let system = &compiled.system;
    let relation = &system.connections[0];
    let [left, right] = meshes(2);
    let a = &left.tags.facet_regions[&RegionTagId::new("x_max")];
    let b = &right.tags.facet_regions[&RegionTagId::new("x_min")];
    let matching = ConnectionRealizationPlan::matching(
        system,
        relation,
        [&left.mesh, &right.mesh],
        [a, b],
        1e-12,
    )
    .unwrap();
    assert_eq!(matching.vertex_pairs().len(), 9);
    assert_eq!(matching.facet_pairs().len(), 8);
    for (side, mesh) in [&left.mesh, &right.mesh].into_iter().enumerate() {
        let model = operators.model_systems[side].1.clone();
        assert!(
            model.blocks[0]
                .form
                .receipt
                .boundary_terms
                .iter()
                .any(|t| matches!(
                    t.disposition,
                    scientia::BoundaryTermDisposition::Open { .. }
                ))
        );
        let layout =
            BlockLayout::new(vec![(model.field_order[0], mesh.vertices().len(), 1)]).unwrap();
        assert!(
            SystemRealizationPlan::new(model.clone(), mesh.clone(), layout.clone())
                .unwrap_err()
                .to_string()
                .contains("CONNECTION_UNCLOSED")
        );
        let mut other_model = model.clone();
        other_model.model = "AnotherModelInTheSameModule".into();
        assert!(
            SystemRealizationPlan::with_connections(
                other_model,
                mesh.clone(),
                layout.clone(),
                SystemQuadrature::Richest,
                std::slice::from_ref(&matching),
            )
            .is_err()
        );
        let mut changed = model.clone();
        changed.source_semantic_digest = scientia::Digest::blake3(b"different scientific source");
        assert!(
            SystemRealizationPlan::with_connections(
                changed,
                mesh.clone(),
                layout.clone(),
                SystemQuadrature::Richest,
                std::slice::from_ref(&matching),
            )
            .is_err()
        );
        SystemRealizationPlan::with_connections(
            model,
            mesh.clone(),
            layout,
            SystemQuadrature::Richest,
            std::slice::from_ref(&matching),
        )
        .unwrap();
    }
    let partial = &b[..b.len() - 1];
    assert!(
        ConnectionRealizationPlan::matching(
            system,
            relation,
            [&left.mesh, &right.mesh],
            [a, partial],
            1e-12
        )
        .is_err()
    );
    assert!(
        ConnectionRealizationPlan::matching(
            system,
            relation,
            [&left.mesh, &left.mesh],
            [a, a],
            1e-12
        )
        .unwrap_err()
        .to_string()
        .contains("normals")
    );
    let [_, fine] = meshes(4);
    assert!(
        ConnectionRealizationPlan::matching(
            system,
            relation,
            [&left.mesh, &fine.mesh],
            [a, &fine.tags.facet_regions[&RegionTagId::new("x_min")]],
            1e-12
        )
        .is_err()
    );
    let mut swapped = relation.clone();
    swapped.balance_coefficients = [1, -1];
    assert!(
        ConnectionRealizationPlan::matching(
            system,
            &swapped,
            [&left.mesh, &right.mesh],
            [a, b],
            1e-12
        )
        .is_err()
    );
}
