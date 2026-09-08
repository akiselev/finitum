//! GX-CONTRACTS C11.22 (Sinbad lane need): essential normal-trace data on an RT0 field through
//! `essential_constraints_from_system`, so a genuinely impermeable mixed-Darcy wall
//! (`flux . n = 0`, an essential condition on the H(div) normal trace) is realizable instead
//! of the naturally closed zero-pressure wall the corpus now declares.
//!
//! Evidence on the `13-mixed-darcy.res` snapshot (unit cube, RT0-P0):
//! 1. constraining every wall's flux DOF to zero puts the constant pressure mode into the
//!    kernel of the reduced operator (the physical impermeable-box statement; the naturally
//!    closed system is full rank, as the E6 test records), and the reduced operator stays
//!    symmetric under MINRES with the pressure projector;
//! 2. a uniform outward datum `flux . n = g` on every wall lifts to a flux whose total
//!    divergence, measured through the `mass_balance` row's public block action, is
//!    `g * (surface area)` -- the divergence theorem fixes the orientation sign and the
//!    facet-measure scaling of the constrained DOFs;
//! 3. an interior facet in the region, a nodal source, and a vector datum are refused typed.

use finitum::{
    BlockLayout, CompatibleDofMaps, FacetTopology, FieldSource, FinitumError, InstanceId,
    PointEvaluation, RegionMap, RegionTagId, SystemConstitutiveInput,
    SystemEssentialConstraintRequirement, SystemOperator, SystemRealizationPlan, TaggedMesh,
    essential_constraints_from_system, facet_membership_from, realize,
};
use methodus::{EvaluationContext, LinearOperator, MinresConfig, NullspaceProjector, solve_minres};
use quantitas::UnitRegistry;
use scientia::{
    DeclarationId, EssentialConstraintRequirement, InputSourceRequirement, OperatorSystem,
    RegionId, SemanticMeasure, SymbolId, compile_operator_system, compile_semantics,
};
use std::collections::BTreeMap;

const DARCY: &str = include_str!("fixtures/corpus/13-mixed-darcy.res");
const MOBILITY_INVERSE: f64 = 2.3;

struct Realized {
    operator: SystemOperator,
    mesh: TaggedMesh,
    flux: SymbolId,
    pressure: SymbolId,
    wall_region: RegionId,
    region_map: RegionMap,
}

fn constitutive(system: &OperatorSystem) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                let components = input.shape.iter().product::<usize>().max(1);
                let binding = match input.source {
                    InputSourceRequirement::Basis => continue,
                    InputSourceRequirement::ModelDefinedConstitutive { .. } => {
                        SystemConstitutiveInput::new(
                            block.equation.clone(),
                            integral.integral_index,
                            input.id,
                            1,
                            "mobility_inverse",
                            move |_: &PointEvaluation| vec![MOBILITY_INVERSE],
                            move |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                        )
                    }
                    _ => SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "zero",
                        move |_: &PointEvaluation| vec![0.0; components],
                        move |_: &PointEvaluation, _: &PointEvaluation| vec![0.0; components],
                    ),
                };
                constitutive.push(binding.unwrap());
            }
        }
    }
    constitutive
}

fn realize_darcy(subdivisions: usize) -> Realized {
    let compilation = compile_semantics(DARCY, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "MixedDarcy",
        &["darcy_law", "mass_balance"],
    )
    .unwrap();
    let row = |equation: &str| {
        system
            .blocks
            .iter()
            .find(|block| block.equation == equation)
            .unwrap()
            .row
    };
    let (flux, pressure) = (row("darcy_law"), row("mass_balance"));
    let mesh = realize(&finitum::MeshProfile::SimplexBox {
        dimension: 3,
        extent: vec![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions; 3],
    })
    .unwrap();
    let facets = FacetTopology::from_mesh(&mesh.mesh).unwrap();
    let compatible = CompatibleDofMaps::simplex(&mesh.mesh, &facets).unwrap();
    let layout = BlockLayout::new([
        (flux, compatible.hdiv_dof_count, 1),
        (pressure, mesh.mesh.cells().len(), 1),
    ])
    .unwrap();
    let plan = SystemRealizationPlan::new(system.clone(), mesh.mesh.clone(), layout).unwrap();
    let wall_region = system
        .blocks
        .iter()
        .find(|block| block.equation == "darcy_law")
        .unwrap()
        .factorization
        .integrals
        .iter()
        .find_map(|integral| match integral.measure {
            SemanticMeasure::ExteriorFacet { region } => Some(region),
            _ => None,
        })
        .expect("the snapshot declares the walls as an exterior-facet region");
    let mut region_map = RegionMap::new();
    region_map.insert(
        wall_region,
        ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"].map(RegionTagId::new),
    );
    let facet_regions = facet_membership_from(&mesh, &region_map, [wall_region]).unwrap();
    let operator = plan
        .bind_kernels_with_facets(
            constitutive(&system),
            BTreeMap::from([("mass_balance".to_string(), -1.0)]),
            facet_regions,
        )
        .unwrap();
    Realized {
        operator,
        mesh,
        flux,
        pressure,
        wall_region,
        region_map,
    }
}

/// The snapshot declares no essential requirement of its own (its walls are a Neumann
/// integral), so the impermeable statement is authored here as the typed requirement a
/// `dirichlet flux . n = g` closure would compile to: the RT0 argument on the wall region.
fn wall_requirement(
    realized: &Realized,
    value: FieldSource,
) -> SystemEssentialConstraintRequirement {
    SystemEssentialConstraintRequirement {
        field: realized.flux,
        requirement: EssentialConstraintRequirement {
            argument: realized.flux,
            region: realized.wall_region,
            condition: DeclarationId(0),
        },
        value,
    }
}

#[test]
fn impermeable_walls_as_rt0_essential_constraints_restore_the_constant_pressure_kernel() {
    let realized = realize_darcy(2);
    let requirement = wall_requirement(&realized, FieldSource::constant([0.0]));
    let constraints = essential_constraints_from_system(
        &realized.operator,
        &realized.mesh,
        &realized.region_map,
        &[requirement],
    )
    .unwrap();
    let facets = FacetTopology::from_mesh(&realized.mesh.mesh).unwrap();
    assert_eq!(
        constraints.constraints().count(),
        facets.exterior().count(),
        "one constrained flux DOF per wall facet"
    );
    for constraint in constraints.constraints() {
        assert!(constraint.dependencies.is_empty());
        assert_eq!(constraint.offset, 0.0);
        assert!(facets.facets()[constraint.target.0].is_exterior());
    }

    let reduced = realized.operator.reduced(constraints).unwrap();
    assert_eq!(
        realized.operator.prove_symmetry(1.0e-9).unwrap(),
        methodus::OperatorSymmetry::Symmetric
    );
    let candidates = realized.operator.nullspace_candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].block, realized.pressure);
    let mode = candidates[0].resolve(realized.operator.layout()).unwrap();
    assert!(
        mode.verify_in_kernel(&reduced, 1.0e-9).unwrap(),
        "with every wall impermeable the constant pressure is a genuine kernel mode"
    );

    // The reduced system still solves under MINRES with the projector (a consistent
    // right-hand side manufactured from a zero-mean-pressure state).
    let dimension = realized.operator.dimension();
    let pressure_block = realized.operator.layout().block(realized.pressure).unwrap();
    let mut x_true = (0..dimension)
        .map(|index| ((index as f64 + 0.3) * 0.618_034).sin())
        .collect::<Vec<_>>();
    for constraint in reduced.constraints().constraints() {
        x_true[constraint.target.0] = 0.0;
    }
    let mean = x_true[pressure_block.offset..pressure_block.offset + pressure_block.extent]
        .iter()
        .sum::<f64>()
        / pressure_block.extent as f64;
    for value in &mut x_true[pressure_block.offset..pressure_block.offset + pressure_block.extent] {
        *value -= mean;
    }
    let mut rhs = vec![0.0; dimension];
    reduced
        .apply(&EvaluationContext::default(), &x_true, &mut rhs)
        .unwrap();
    let report = solve_minres(
        &reduced,
        None,
        Some(mode.projector() as &dyn NullspaceProjector),
        &EvaluationContext::reproducible(),
        &rhs,
        &vec![0.0; dimension],
        &MinresConfig {
            max_iterations: 4 * dimension,
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-10,
        },
    )
    .unwrap();
    assert!(report.converged);
    for (solved, expected) in report.solution.iter().zip(&x_true) {
        assert!((solved - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn uniform_outward_flux_datum_lifts_to_the_divergence_theorem_total() {
    let realized = realize_darcy(2);
    let g = 0.7;
    let requirement = wall_requirement(&realized, FieldSource::constant([g]));
    let motion = finitum::prescribed_values_from_system_by_variable(
        &realized.operator,
        &realized.mesh,
        &[(InstanceId(0), &realized.region_map)],
        &[finitum::SystemVariablePrescribedValue {
            variable: realized
                .operator
                .system_ids()
                .variable(InstanceId(0), realized.flux)
                .unwrap(),
            requirement: requirement.requirement.clone(),
            value: FieldSource::constant([g]),
            rate: FieldSource::constant([0.0]),
            origin: finitum::InputOrigin::Slot("wall-flux".into()),
        }],
    )
    .unwrap_err();
    assert!(
        matches!(motion, FinitumError::UnsupportedRealization(message) if message.contains("RT0"))
    );

    let constraints = essential_constraints_from_system(
        &realized.operator,
        &realized.mesh,
        &realized.region_map,
        &[requirement],
    )
    .unwrap();
    let dimension = realized.operator.dimension();
    // Lift: the constrained DOFs at their values, everything else zero.
    let lifted = constraints.expand(&vec![0.0; dimension]).unwrap();
    let flux_block = realized.operator.layout().block(realized.flux).unwrap();
    let lifted_flux = lifted[flux_block.offset..flux_block.offset + flux_block.extent].to_vec();
    assert!(lifted_flux.iter().any(|value| value.abs() > 1.0e-6));

    // `mass_balance` row applied to the lifted flux: per cell `-(integral q div(flux))`
    // (the row carries `equation_sign = -1`); summing over the P0 test functions gives
    // `-integral_Omega div(flux) = -g * |boundary|`.
    let ids = realized.operator.system_ids();
    let row = ids.residual(InstanceId(0), "mass_balance").unwrap();
    let column = ids.variable(InstanceId(0), realized.flux).unwrap();
    let mut divergence = vec![0.0; realized.mesh.mesh.cells().len()];
    realized
        .operator
        .block_action(row, column, &lifted_flux, &mut divergence)
        .unwrap();
    let total = divergence.iter().sum::<f64>();
    let surface = 6.0;
    assert!(
        (total + g * surface).abs() <= 1.0e-12 * (g * surface),
        "total divergence {total} should equal -{g} * {surface}"
    );

    // Refusals: a vector datum, a nodal source, and an interior facet in the region.
    assert!(matches!(
        essential_constraints_from_system(
            &realized.operator,
            &realized.mesh,
            &realized.region_map,
            &[wall_requirement(
                &realized,
                FieldSource::constant([g, 0.0, 0.0])
            )],
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        essential_constraints_from_system(
            &realized.operator,
            &realized.mesh,
            &realized.region_map,
            &[wall_requirement(
                &realized,
                FieldSource::Nodal(vec![0.0; realized.mesh.mesh.vertices().len()])
            )],
        ),
        Err(FinitumError::UnsupportedRealization(_))
    ));
    let facets = FacetTopology::from_mesh(&realized.mesh.mesh).unwrap();
    let interior = facets.interior().next().unwrap().id;
    let mut tagged = realized.mesh.clone();
    tagged
        .tags
        .facet_regions
        .insert(RegionTagId::new("x_min"), vec![interior]);
    assert!(matches!(
        essential_constraints_from_system(
            &realized.operator,
            &tagged,
            &realized.region_map,
            &[wall_requirement(&realized, FieldSource::constant([0.0]))],
        ),
        Err(FinitumError::UnsupportedRealization(_))
    ));
}
