//! W7 package 3: runtime inf-sup checker for realized mixed pairs.
//!
//! Drives corpus copies (`tests/fixtures/w7_infsup/`) through the real Scientia ->
//! `SystemRealizationPlan::bind_kernels` path and estimates the discrete inf-sup constant with
//! `finitum::estimate_inf_sup`: Taylor-Hood Stokes (P2-P1) and RT0-P0 mixed Darcy are stable
//! with a mesh-robust constant; the same Stokes model with a P1 velocity (the P1-P1 pair) is
//! refused deterministically with spurious pressure modes. The pairing is derived from
//! Scientia's structural `OperatorStructure`, never from a field name.

use finitum::{
    BlockCoupling, BlockLayout, CompatibleDofMaps, ConstraintSet, CouplingKind, DofId,
    FacetTopology, FieldSource, FieldSpec, FinitumError, InfSupConfig, InfSupInstability,
    InfSupNorm, InfSupPairing, InfSupVerdict, Mesh, MeshProfile, MixedOperator, MixedSpace,
    PointEvaluation, RegionMap, RegionTagId, SystemConstitutiveInput,
    SystemEssentialConstraintRequirement, SystemOperator, SystemRealizationPlan, WeightedDof,
    essential_constraints_from_system, estimate_inf_sup, facet_membership_from,
    quadratic_simplex_dof_map, realize, require_inf_sup_stable,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, RegionId, SemanticMeasure,
    SymbolId, compile_operator_system, compile_semantics, derive_operator_structure_for_system,
};
use std::collections::BTreeMap;

const STOKES: &str = include_str!("fixtures/corpus/25-stokes.res");
const STOKES_P1P1: &str = include_str!("fixtures/w7_infsup/25-stokes-p1p1.res");
const DARCY: &str = include_str!("fixtures/corpus/13-mixed-darcy.res");
const POISSON: &str = include_str!("fixtures/corpus/01-poisson.res");

const MU: f64 = 1.7;
const DARCY_MOBILITY_INVERSE: f64 = 2.3;

struct CompiledPair {
    system: OperatorSystem,
    constrained: SymbolId,
    multiplier: SymbolId,
}

fn compile_pair(source: &str, model: &str, equations: [&str; 2]) -> CompiledPair {
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(&compilation.semantic, model, &equations).unwrap();
    let row = |equation: &str| {
        system
            .blocks
            .iter()
            .find(|block| block.equation == equation)
            .unwrap_or_else(|| panic!("{equation} block"))
            .row
    };
    CompiledPair {
        constrained: row(equations[0]),
        multiplier: row(equations[1]),
        system,
    }
}

fn unit_square(subdivisions: usize) -> finitum::TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

fn unit_cube(subdivisions: usize) -> finitum::TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 3,
        extent: vec![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions, subdivisions],
    })
    .unwrap()
}

fn stokes_layout(
    mesh: &Mesh,
    velocity_order: u8,
    velocity: SymbolId,
    pressure: SymbolId,
) -> BlockLayout {
    let velocity_nodes = match velocity_order {
        1 => mesh.vertices().len(),
        2 => quadratic_simplex_dof_map(mesh, 2).unwrap().dof_count() / 2,
        _ => unreachable!(),
    };
    BlockLayout::new([
        (velocity, velocity_nodes, 2),
        (pressure, mesh.vertices().len(), 1),
    ])
    .unwrap()
}

/// Shape-resolved (no physics-name dispatch) constitutive bindings, mirroring
/// `tests/sv2b4_system_stokes.rs`: the `[2,2]` tensor input is the linear viscous stress, the
/// 2-vector is the body force (zero here).
fn stokes_constitutive(system: &OperatorSystem) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let components = input.shape.iter().product::<usize>().max(1);
                let binding = if components == 4 {
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "w7-infsup/viscosity",
                        |evaluation: &PointEvaluation| {
                            evaluation
                                .values(DerivativeEvaluation::SymmetricGradient)
                                .expect("active symmetric-gradient input")
                                .iter()
                                .map(|value| 2.0 * MU * value)
                                .collect()
                        },
                        |_evaluation: &PointEvaluation, direction: &PointEvaluation| {
                            direction
                                .values(DerivativeEvaluation::SymmetricGradient)
                                .expect("active symmetric-gradient direction")
                                .iter()
                                .map(|value| 2.0 * MU * value)
                                .collect()
                        },
                    )
                } else {
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "w7-infsup/zero",
                        move |_evaluation: &PointEvaluation| vec![0.0; components],
                        move |_evaluation: &PointEvaluation, _direction: &PointEvaluation| {
                            vec![0.0; components]
                        },
                    )
                };
                constitutive.push(binding.unwrap());
            }
        }
    }
    constitutive
}

fn walls_region_map(region: RegionId, dimension: usize) -> RegionMap {
    let mut map = RegionMap::new();
    let tags = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"];
    map.insert(
        region,
        tags[..2 * dimension]
            .iter()
            .map(|tag| RegionTagId::new(*tag)),
    );
    map
}

struct RealizedStokes {
    operator: SystemOperator,
    constraints: ConstraintSet,
}

fn realize_stokes(
    source: &str,
    velocity_order: u8,
    subdivisions: usize,
    signed: bool,
) -> (CompiledPair, RealizedStokes) {
    let compiled = compile_pair(source, "StokesFlow", ["momentum", "incompressibility"]);
    let mesh = unit_square(subdivisions);
    let layout = stokes_layout(
        &mesh.mesh,
        velocity_order,
        compiled.constrained,
        compiled.multiplier,
    );
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let equation_sign = if signed {
        BTreeMap::from([("incompressibility".to_string(), -1.0)])
    } else {
        BTreeMap::new()
    };
    let operator = plan
        .bind_kernels(stokes_constitutive(&compiled.system), equation_sign)
        .unwrap();
    let requirement = compiled
        .system
        .blocks
        .iter()
        .find(|block| block.equation == "momentum")
        .unwrap()
        .factorization
        .essential_constraints
        .first()
        .expect("momentum declares the walls essential constraint")
        .clone();
    let region_map = walls_region_map(requirement.region, 2);
    let constraints = essential_constraints_from_system(
        &operator,
        &mesh,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: compiled.constrained,
            requirement,
            value: FieldSource::constant(vec![0.0, 0.0]),
        }],
    )
    .unwrap();
    (
        compiled,
        RealizedStokes {
            operator,
            constraints,
        },
    )
}

fn pressure_config(declared_kernel_dimension: usize) -> InfSupConfig {
    InfSupConfig {
        declared_kernel_dimension,
        ..InfSupConfig::default()
    }
}

/// Taylor-Hood (P2-P1) Stokes from the corpus copy: the pairing derived from Scientia's
/// structure is (velocity, pressure); with velocity fixed on every wall exactly one constant
/// pressure mode is in the kernel (declared), and the energy/L2-norm inf-sup constant stays
/// bounded away from zero under refinement (the checker's mesh-robustness signal).
#[test]
fn taylor_hood_stokes_is_inf_sup_stable_with_a_mesh_robust_constant() {
    let mut constants = Vec::new();
    for subdivisions in [2usize, 4] {
        let (compiled, realized) = realize_stokes(STOKES, 2, subdivisions, true);
        let structure = derive_operator_structure_for_system(&compiled.system, None).unwrap();
        let pairing = InfSupPairing::from_structure(&structure).unwrap();
        assert_eq!(pairing.constrained, compiled.constrained);
        assert_eq!(pairing.multiplier, compiled.multiplier);
        assert_eq!(
            pairing,
            InfSupPairing::from_structure(realized.operator.structure()).unwrap()
        );

        let mass = realized.operator.mass_matrix(compiled.multiplier).unwrap();
        let estimate = estimate_inf_sup(
            &realized.operator,
            realized.operator.layout(),
            Some(&realized.constraints),
            pairing,
            &InfSupNorm::Gram(mass),
            &pressure_config(1),
        )
        .unwrap();
        assert_eq!(estimate.verdict, InfSupVerdict::Stable, "{estimate:?}");
        assert_eq!(estimate.kernel_dimension, 1);
        assert_eq!(estimate.spurious_mode_count, 0);
        assert_eq!(estimate.count_deficit, 0);
        assert_eq!(
            estimate.multiplier_dimension,
            (subdivisions + 1) * (subdivisions + 1)
        );
        assert!(estimate.constrained_dimension > estimate.multiplier_dimension);
        require_inf_sup_stable(&estimate).unwrap();
        let constant = estimate.inf_sup_constant.unwrap();
        assert!(constant > 0.2, "constant {constant}");
        constants.push(constant);

        // Undeclared kernel: the same estimate reports the constant mode as spurious and the
        // refusal is deterministic (same input, same typed error).
        let undeclared = estimate_inf_sup(
            &realized.operator,
            realized.operator.layout(),
            Some(&realized.constraints),
            pairing,
            &InfSupNorm::Euclidean,
            &pressure_config(0),
        )
        .unwrap();
        assert_eq!(
            undeclared.verdict,
            InfSupVerdict::Unstable(InfSupInstability::SpuriousModes {
                observed: 1,
                declared: 0
            })
        );
        let first = require_inf_sup_stable(&undeclared).unwrap_err();
        let second = require_inf_sup_stable(&undeclared).unwrap_err();
        assert_eq!(first, second);
        assert!(matches!(first, FinitumError::InfSupUnstable(_)));
        assert!(first.to_string().starts_with("INF_SUP_UNSTABLE"));
    }
    let ratio = constants[1] / constants[0];
    assert!(
        (0.5..=2.0).contains(&ratio),
        "inf-sup constant should be mesh-robust: {constants:?}"
    );
}

/// The estimate depends on the multiplier-row coupling block alone, so the unsigned corpus
/// system (whose momentum/incompressibility blocks are exact negatives relative to one
/// divergence integral) and the sign-corrected symmetric one give the identical record.
#[test]
fn inf_sup_estimate_is_independent_of_the_equation_sign_gauge() {
    let (compiled, unsigned) = realize_stokes(STOKES, 2, 2, false);
    let (_, signed) = realize_stokes(STOKES, 2, 2, true);
    let pairing = InfSupPairing::new(compiled.constrained, compiled.multiplier).unwrap();
    let mass = signed.operator.mass_matrix(compiled.multiplier).unwrap();
    let estimate = |realized: &RealizedStokes| {
        estimate_inf_sup(
            &realized.operator,
            realized.operator.layout(),
            Some(&realized.constraints),
            pairing,
            &InfSupNorm::Gram(mass.clone()),
            &pressure_config(1),
        )
        .unwrap()
    };
    let unsigned = estimate(&unsigned);
    let signed = estimate(&signed);
    assert_eq!(unsigned.identity, signed.identity);
    assert_eq!(unsigned, signed);

    // The already-reduced operator (identity rows / zero columns on the constrained DOFs)
    // with the same constraint set removed gives the identical record.
    let reduced = signed_reduced(&compiled, STOKES);
    assert_eq!(reduced.identity, signed.identity);
}

fn signed_reduced(compiled: &CompiledPair, source: &str) -> finitum::InfSupEstimate {
    let (_, realized) = realize_stokes(source, 2, 2, true);
    let reduced = realized
        .operator
        .reduced(realized.constraints.clone())
        .unwrap();
    let mass = realized.operator.mass_matrix(compiled.multiplier).unwrap();
    estimate_inf_sup(
        &reduced,
        realized.operator.layout(),
        Some(&realized.constraints),
        InfSupPairing::new(compiled.constrained, compiled.multiplier).unwrap(),
        &InfSupNorm::Gram(mass),
        &pressure_config(1),
    )
    .unwrap()
}

/// The same corpus model with `H1(order=1)` velocity (equal-order P1-P1): spurious pressure
/// modes appear on every mesh and the refusal is deterministic. Finding recorded by this
/// test (not assumed): on the structured diagonal `SimplexBox` triangulation with velocity
/// fixed on every wall, the P1-P1 pressure kernel has dimension 8 (one legitimate constant
/// mode plus seven spurious modes) at 4x4 and 6x6 alike -- the count argument alone (more
/// pressure than free velocity DOFs) explains the 2x2 mesh only, while the 6x6 mesh has 50
/// free velocity DOFs against 49 pressures and is rank deficient regardless (8x8, with 98
/// against 81, gave the same kernel dimension 8 when probed).
#[test]
fn p1_p1_stokes_is_refused_with_spurious_pressure_modes() {
    for subdivisions in [2usize, 4, 6] {
        let (compiled, realized) = realize_stokes(STOKES_P1P1, 1, subdivisions, true);
        let pairing = InfSupPairing::from_structure(realized.operator.structure()).unwrap();
        assert_eq!(pairing.constrained, compiled.constrained);
        let mass = realized.operator.mass_matrix(compiled.multiplier).unwrap();
        let estimate = estimate_inf_sup(
            &realized.operator,
            realized.operator.layout(),
            Some(&realized.constraints),
            pairing,
            &InfSupNorm::Gram(mass),
            &pressure_config(1),
        )
        .unwrap();
        assert!(estimate.spurious_mode_count > 0, "{estimate:?}");
        assert_eq!(
            estimate.verdict,
            InfSupVerdict::Unstable(InfSupInstability::SpuriousModes {
                observed: estimate.kernel_dimension,
                declared: 1
            })
        );
        assert_eq!(estimate.inf_sup_constant, None);
        assert_eq!(
            estimate.multiplier_dimension,
            (subdivisions + 1) * (subdivisions + 1)
        );
        assert_eq!(
            estimate.constrained_dimension,
            2 * (subdivisions - 1) * (subdivisions - 1)
        );
        match subdivisions {
            2 => {
                assert_eq!(estimate.count_deficit, 7);
                assert!(estimate.kernel_dimension >= 7);
            }
            4 => {
                assert_eq!(estimate.count_deficit, 7);
                assert_eq!(estimate.kernel_dimension, 8);
            }
            _ => {
                assert_eq!(estimate.count_deficit, 0);
                assert_eq!(estimate.kernel_dimension, 8);
                assert_eq!(estimate.spurious_mode_count, 7);
            }
        }
        let error = require_inf_sup_stable(&estimate).unwrap_err();
        assert_eq!(error, require_inf_sup_stable(&estimate).unwrap_err());
        match error {
            FinitumError::InfSupUnstable(message) => {
                assert!(message.contains("spurious multiplier mode"), "{message}");
            }
            other => panic!("expected INF_SUP_UNSTABLE, got {other:?}"),
        }
    }
}

fn darcy_constitutive(system: &OperatorSystem) -> Vec<SystemConstitutiveInput> {
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
                            move |_evaluation: &PointEvaluation| vec![DARCY_MOBILITY_INVERSE],
                            move |_evaluation: &PointEvaluation, _direction: &PointEvaluation| {
                                vec![0.0]
                            },
                        )
                    }
                    _ => SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "zero",
                        move |_evaluation: &PointEvaluation| vec![0.0; components],
                        move |_evaluation: &PointEvaluation, _direction: &PointEvaluation| {
                            vec![0.0; components]
                        },
                    ),
                };
                constitutive.push(binding.unwrap());
            }
        }
    }
    constitutive
}

/// RT0-P0 mixed Darcy on a tetrahedral unit cube (the E6 fixture): stable with no kernel at
/// all (the corpus's impermeable boundary term leaves the system full rank, as `STATUS.md`
/// records), in both the P0 mass norm and the Euclidean norm.
#[test]
fn rt0_p0_darcy_is_inf_sup_stable() {
    let compiled = compile_pair(DARCY, "MixedDarcy", ["darcy_law", "mass_balance"]);
    let mesh = unit_cube(2);
    let facets = FacetTopology::from_mesh(&mesh.mesh).unwrap();
    let compatible = CompatibleDofMaps::simplex(&mesh.mesh, &facets).unwrap();
    let layout = BlockLayout::new([
        (compiled.constrained, compatible.hdiv_dof_count, 1),
        (compiled.multiplier, mesh.mesh.cells().len(), 1),
    ])
    .unwrap();
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let region = compiled
        .system
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
        .expect("darcy_law declares an exterior-facet Neumann boundary integral");
    let facet_regions =
        facet_membership_from(&mesh, &walls_region_map(region, 3), [region]).unwrap();
    let operator = plan
        .bind_kernels_with_facets(
            darcy_constitutive(&compiled.system),
            BTreeMap::from([("mass_balance".to_string(), -1.0)]),
            facet_regions,
        )
        .unwrap();
    let pairing = InfSupPairing::from_structure(operator.structure()).unwrap();
    assert_eq!(pairing.constrained, compiled.constrained);
    assert_eq!(pairing.multiplier, compiled.multiplier);

    let mass = operator.mass_matrix(compiled.multiplier).unwrap();
    // P0 mass is the diagonal of cell volumes, summing to the unit cube's volume.
    let cells = mesh.mesh.cells().len();
    let trace = (0..cells)
        .map(|cell| mass[cell * cells + cell])
        .sum::<f64>();
    assert!((trace - 1.0).abs() < 1.0e-12, "trace {trace}");
    for row in 0..cells {
        for column in 0..cells {
            if row != column {
                assert_eq!(mass[row * cells + column], 0.0);
            }
        }
    }

    for norm in [InfSupNorm::Gram(mass), InfSupNorm::Euclidean] {
        let estimate = estimate_inf_sup(
            &operator,
            operator.layout(),
            None,
            pairing,
            &norm,
            &pressure_config(0),
        )
        .unwrap();
        assert_eq!(estimate.verdict, InfSupVerdict::Stable, "{estimate:?}");
        assert_eq!(estimate.kernel_dimension, 0);
        assert_eq!(estimate.constrained_dimension, compatible.hdiv_dof_count);
        assert_eq!(estimate.multiplier_dimension, cells);
        assert!(estimate.inf_sup_constant.unwrap() > 0.1, "{estimate:?}");
        require_inf_sup_stable(&estimate).unwrap();
    }

    // Refusal: the RT0 field's mass matrix is not realized.
    assert!(matches!(
        operator.mass_matrix(compiled.constrained),
        Err(FinitumError::UnsupportedRealization(_))
    ));
}

/// Typed refusals: a non-saddle structure has no pairing, a same-field pairing, a Gram norm
/// of the wrong extent, an affine (dependency-carrying) constraint, and a nonzero multiplier
/// diagonal block (a stabilized pairing).
#[test]
fn inf_sup_refusals_are_typed() {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let poisson = compile_operator_system(&compilation.semantic, "Poisson", &["balance"]).unwrap();
    let structure = derive_operator_structure_for_system(&poisson, None).unwrap();
    assert!(matches!(
        InfSupPairing::from_structure(&structure),
        Err(FinitumError::UnsupportedRealization(_))
    ));

    let (compiled, realized) = realize_stokes(STOKES, 2, 2, true);
    assert!(matches!(
        InfSupPairing::new(compiled.multiplier, compiled.multiplier),
        Err(FinitumError::InvalidRealization(_))
    ));
    let pairing = InfSupPairing::new(compiled.constrained, compiled.multiplier).unwrap();
    assert!(matches!(
        estimate_inf_sup(
            &realized.operator,
            realized.operator.layout(),
            Some(&realized.constraints),
            pairing,
            &InfSupNorm::Gram(vec![1.0; 4]),
            &pressure_config(1),
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    let dimension = realized.operator.dimension();
    let affine = ConstraintSet::new(
        dimension,
        vec![finitum::AffineConstraint {
            target: DofId(0),
            dependencies: vec![WeightedDof {
                dof: DofId(1),
                weight: 0.5,
            }],
            offset: 0.0,
        }],
    )
    .unwrap();
    assert!(matches!(
        estimate_inf_sup(
            &realized.operator,
            realized.operator.layout(),
            Some(&affine),
            pairing,
            &InfSupNorm::Euclidean,
            &pressure_config(1),
        ),
        Err(FinitumError::UnsupportedRealization(_))
    ));

    // A structural P1-P1 operator with a Laplacian on the pressure block (a stabilized-looking
    // pairing) is refused: the constraint-only estimate does not judge it.
    let mesh = unit_square(2);
    let velocity = SymbolId(0);
    let pressure = SymbolId(1);
    let space = MixedSpace::new(
        mesh.mesh.clone(),
        vec![
            FieldSpec {
                symbol: velocity,
                order: 1,
                components: 2,
            },
            FieldSpec {
                symbol: pressure,
                order: 1,
                components: 1,
            },
        ],
    )
    .unwrap();
    let stabilized = MixedOperator::new(
        space,
        vec![
            BlockCoupling {
                test: velocity,
                trial: velocity,
                kind: CouplingKind::GradientGradient,
                scale: 1.0,
            },
            BlockCoupling {
                test: pressure,
                trial: pressure,
                kind: CouplingKind::GradientGradient,
                scale: 0.01,
            },
            BlockCoupling {
                test: velocity,
                trial: pressure,
                kind: CouplingKind::DivergenceValue,
                scale: -1.0,
            },
        ],
    )
    .unwrap();
    let layout = stabilized.space().layout().clone();
    let error = estimate_inf_sup(
        &stabilized,
        &layout,
        None,
        InfSupPairing::new(velocity, pressure).unwrap(),
        &InfSupNorm::Euclidean,
        &InfSupConfig::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error, FinitumError::UnsupportedRealization(_)),
        "{error}"
    );
    assert!(error.to_string().contains("nonzero diagonal block"));
}
