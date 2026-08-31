//! GX-C6: Dirichlet constraints from region tags plus [`FieldSource`] data, including
//! `Kernel`-valued boundary functions and per-component selection.

use finitum::{
    ComponentSelection, DofId, FieldSource, MeshProfile, RegionMap, RegionTagId, realize,
    vector_nodal_dof_map,
};
use quantitas::{Dimension, QuantityKindId, UnitRegistry};
use scientia::scientific::{
    FrameSemantics, OutOfValidityPolicy, PropertyDomain, PropertyEvidence, PropertyInput,
    PropertyLocality, PropertyModel, PropertyOutput, PropertySignature, TensorSymmetry, ValueShape,
};
use scientia::{
    DeclarationId, DerivativeContract, EssentialConstraintRequirement, PropertyDefinition,
    RegionId, SymbolId, lower_property_kernel, parse_expression,
};

fn scalar_input(name: &str) -> PropertyInput {
    PropertyInput {
        name: name.into(),
        quantity_kind: QuantityKindId::new("Dimensionless"),
        dimension: Dimension::DIMENSIONLESS,
        shape: ValueShape::Scalar,
        physical_min: None,
        physical_max: None,
        nominal: None,
    }
}

/// A boundary function `g(x) = 2.0 + 3.0 * x`, evaluated with no derivative contract (boundary
/// data never needs a tangent).
fn boundary_function_definition() -> PropertyDefinition {
    PropertyDefinition {
        signature: PropertySignature {
            id: "boundary_flux".into(),
            inputs: vec![scalar_input("x")],
            output: PropertyOutput {
                quantity_kind: QuantityKindId::new("Dimensionless"),
                dimension: Dimension::DIMENSIONLESS,
                shape: ValueShape::Scalar,
                symmetry: TensorSymmetry::None,
                frame: FrameSemantics::Scalar,
            },
            locality: PropertyLocality::Pointwise,
            differentiability: DerivativeContract::AnalyticProvided,
        },
        model: PropertyModel::Expression(parse_expression("2.0 + 3.0 * x").unwrap()),
        domain: PropertyDomain {
            physical_bounds: vec![],
            validity_bounds: vec![],
            phase_constraints: vec![],
            composition_constraints: vec![],
            assumptions: vec![],
            out_of_validity: OutOfValidityPolicy::Warn,
        },
        evidence: PropertyEvidence {
            sources: vec![],
            dataset_digest: None,
            fit_digest: None,
            uncertainty: None,
            notes: Default::default(),
        },
    }
}

#[test]
fn kernel_valued_boundary_matches_closed_form_at_vertices() {
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 2.0], [0.0, 1.0]],
        subdivisions: vec![4, 2],
    };
    let mesh = realize(&profile).unwrap();
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 1).unwrap();

    let kernel = lower_property_kernel(
        &boundary_function_definition(),
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let source = FieldSource::kernel(kernel).unwrap();

    let mut region_map = RegionMap::new();
    region_map.insert(RegionId(0), [RegionTagId::new("y_min")]);
    let requirement = EssentialConstraintRequirement {
        argument: SymbolId(0),
        region: RegionId(0),
        condition: DeclarationId(0),
    };
    let constraints = finitum::essential_constraints_from(
        &mesh,
        &dof_map,
        &[requirement],
        &region_map,
        &[source],
    )
    .unwrap();

    let mut checked = 0;
    for constraint in constraints.constraints() {
        let vertex = &mesh.mesh.vertices()[constraint.target.0];
        // Only y_min (y == 0.0) vertices were tagged.
        assert!((vertex[1] - 0.0).abs() <= 1.0e-12);
        let expected = 2.0 + 3.0 * vertex[0];
        assert!(
            (constraint.offset - expected).abs() <= 1.0e-9,
            "vertex {vertex:?}: got {}, expected {expected}",
            constraint.offset
        );
        checked += 1;
    }
    // Five vertices along y_min for subdivisions [4, 2] (x = 0, 0.5, 1.0, 1.5, 2.0).
    assert_eq!(checked, 5);
}

#[test]
fn per_component_selection_constrains_only_the_selected_components() {
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![2, 2],
    };
    let mesh = realize(&profile).unwrap();
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 2).unwrap();

    let mut region_map = RegionMap::new();
    region_map.insert(RegionId(0), [RegionTagId::new("x_min")]);
    let requirement = EssentialConstraintRequirement {
        argument: SymbolId(0),
        region: RegionId(0),
        condition: DeclarationId(0),
    };
    let source = FieldSource::constant(vec![7.0, 9.0]);

    // Only component 0 (the x-component) is selected.
    let constraints = finitum::essential_constraints_from_selected(
        &mesh,
        &dof_map,
        std::slice::from_ref(&requirement),
        &region_map,
        std::slice::from_ref(&source),
        &[ComponentSelection::Only(vec![0])],
    )
    .unwrap();

    let x_min_vertices = mesh
        .mesh
        .vertices()
        .iter()
        .enumerate()
        .filter(|(_, vertex)| (vertex[0] - 0.0).abs() <= 1.0e-12)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert!(!x_min_vertices.is_empty());
    for vertex in &x_min_vertices {
        let x_dof = DofId(vertex * 2);
        let y_dof = DofId(vertex * 2 + 1);
        assert!(
            constraints.constraints().any(|c| c.target == x_dof),
            "x-component of vertex {vertex} should be constrained"
        );
        assert!(
            constraints.constraints().all(|c| c.target != y_dof),
            "y-component of vertex {vertex} should be free"
        );
    }

    // `ComponentSelection::All` (the default via `essential_constraints_from`) constrains both.
    let both = finitum::essential_constraints_from(
        &mesh,
        &dof_map,
        &[requirement],
        &region_map,
        &[source],
    )
    .unwrap();
    for vertex in &x_min_vertices {
        let y_dof = DofId(vertex * 2 + 1);
        assert!(
            both.constraints().any(|c| c.target == y_dof),
            "y-component should be constrained when every component is selected"
        );
    }
}

#[test]
fn conflicting_and_unmapped_refusals_still_hold() {
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![2, 2],
    };
    let mesh = realize(&profile).unwrap();
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 1).unwrap();
    let region_map = RegionMap::new();
    let requirement = EssentialConstraintRequirement {
        argument: SymbolId(0),
        region: RegionId(0),
        condition: DeclarationId(0),
    };
    let result = finitum::essential_constraints_from_selected(
        &mesh,
        &dof_map,
        &[requirement],
        &region_map,
        &[FieldSource::constant(vec![1.0])],
        &[ComponentSelection::All],
    );
    assert!(matches!(
        result,
        Err(finitum::FinitumError::RealizationRegionUnmapped(_))
    ));
}
