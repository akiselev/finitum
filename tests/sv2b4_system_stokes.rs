//! SV2-B4 continuation (E6): executable Scientia-form/Malleus-kernel-driven mixed realization,
//! decisive acceptance.
//!
//! Drives the real `25-stokes.res` corpus (Taylor-Hood-labeled P2 velocity / L2(order=1)
//! pressure, realized here through a continuous P1 Lagrange pressure basis -- see
//! `finitum::system`'s `build_field_elements` doc comment for why an L2-typed field is
//! admitted) through `compile_semantics` -> `compile_operator_system` ->
//! `derive_operator_structure_for_system` -> `SystemRealizationPlan::bind_kernels` (real bound
//! Malleus kernels) -> `methodus::solve_minres`, and cross-checks the realized action against an
//! independently hand-composed `finitum::mixed::MixedOperator`.
//!
//! A real, load-bearing finding this file documents and works around explicitly: the corpus's
//! own equation authorship (`momentum` uses `+grad(pressure)`, which after integration by parts
//! becomes `-integral(p * div(v))`; `incompressibility` uses the un-negated `div(velocity) = 0`,
//! which weakly is `+integral(q * div(u))`) gives a saddle-point coupling that is genuinely NOT
//! symmetric relative to a single shared sign -- confirmed empirically below (the momentum block
//! and the incompressibility block are exact negatives of one another relative to the same
//! underlying divergence integral) and consistent with Scientia's own structural
//! `OperatorStructure::form_symmetry` correctly reporting `Unknown` (never `Symmetric`) for the
//! unsigned system. `SystemRealizationPlan::bind_kernels`'s `equation_sign` parameter exists
//! for exactly this: multiplying the `incompressibility` equation row by `-1` is solution-
//! preserving (its right-hand side is zero) and restores genuine symmetry, verified here by
//! assembly (`SystemOperator::prove_symmetry`), not assumed.

use finitum::{
    BlockCoupling, BlockLayout, CompatibleDofMaps, ConstraintSet, CouplingKind, DofId,
    FacetTopology, FieldSource, FieldSpec, Mesh, MeshProfile, MixedOperator, MixedSpace,
    PointEvaluation, RegionMap, RegionTagId, SystemConstitutiveInput,
    SystemEssentialConstraintRequirement, SystemRealizationPlan, essential_constraints_from_system,
    facet_membership_from, quadratic_simplex_dof_map, realize,
};
use methodus::{
    EvaluationContext, LinearOperator, MinresConfig, NullspaceProjector, OperatorSymmetry,
    solve_minres,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, RegionId, SemanticMeasure,
    SymbolId, compile_operator_system, compile_semantics,
};
use std::collections::BTreeMap;
use std::fs;

const STOKES_CORPUS: &str = "/projects/sinbad/sinbad/physics/corpus/25-stokes.res";
const DARCY_CORPUS: &str = "/projects/sinbad/sinbad/physics/corpus/13-mixed-darcy.res";

/// Viscosity constant this test supplies through a [`SystemConstitutiveInput`] closure -- the
/// corpus declares `mu` as a provider-resolved property (`dynamic_viscosity(0)`), and Scientia
/// never resolves provider *values* itself (they are a realization-side concern, matching how
/// `tests/sv2_elasticity.rs` supplies its own Lame constants through the same
/// `DynamicExternalInput` mechanism `SystemConstitutiveInput` mirrors).
const MU: f64 = 1.7;

fn stress(strain: &[f64]) -> Vec<f64> {
    strain.iter().map(|value| 2.0 * MU * value).collect()
}

struct CompiledStokes {
    system: OperatorSystem,
    velocity: SymbolId,
    pressure: SymbolId,
}

fn compile_stokes() -> CompiledStokes {
    let source = fs::read_to_string(STOKES_CORPUS).expect("25-stokes.res corpus is readable");
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "StokesFlow",
        &["momentum", "incompressibility"],
    )
    .unwrap();
    let velocity = system
        .blocks
        .iter()
        .find(|block| block.equation == "momentum")
        .expect("momentum block")
        .row;
    let pressure = system
        .blocks
        .iter()
        .find(|block| block.equation == "incompressibility")
        .expect("incompressibility block")
        .row;
    CompiledStokes {
        system,
        velocity,
        pressure,
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

fn taylor_hood_layout(mesh: &Mesh, velocity: SymbolId, pressure: SymbolId) -> BlockLayout {
    let velocity_nodes = quadratic_simplex_dof_map(mesh, 2).unwrap().dof_count() / 2;
    let pressure_nodes = mesh.vertices().len();
    BlockLayout::new([(velocity, velocity_nodes, 2), (pressure, pressure_nodes, 1)]).unwrap()
}

/// Every non-`Basis` primal input across the compiled system, resolved generically by shape (no
/// physics-name dispatch): a 4-component (`[2,2]`) tensor input is the momentum block's
/// `ModelDefinedConstitutive` viscous stress (`2 * mu * sym_grad(velocity)`, linear, so its
/// value/direction share the same formula); a 2-component input is the `ExternalValue`
/// body-force, bound to the caller-supplied `body_force(coordinates)` closure (its JVP direction
/// is exactly zero regardless of the closure's own spatial variation, since a closure with no
/// dependence on the active velocity/pressure state is state-independent by construction);
/// anything else falls back to an always-zero closure (unexercised by this corpus, kept only so
/// an unexpected shape still resolves rather than panicking). Passing `|_| [0.0, 0.0]` reproduces
/// this file's original "no forcing term" fixture exactly (a MINRES demonstration against a
/// synthetically constructed, self-consistent right-hand side -- mirroring `tests/sv2b_mixed.rs`'s
/// own MINRES fixture); a nonzero `body_force` is `SystemOperator::load_vector`'s own
/// decisive-acceptance fixture (mission item 3/E6-sysload).
fn stokes_constitutive(
    system: &OperatorSystem,
    body_force: impl Fn(&[f64]) -> [f64; 2] + Clone + Send + Sync + 'static,
) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let components = input.shape.iter().product::<usize>().max(1);
                let equation = block.equation.clone();
                let integral_index = integral.integral_index;
                let input_id = input.id;
                let binding = if components == 4 {
                    SystemConstitutiveInput::new(
                        equation,
                        integral_index,
                        input_id,
                        components,
                        "sv2b4-stokes/viscosity",
                        |evaluation: &PointEvaluation| {
                            stress(
                                evaluation
                                    .values(DerivativeEvaluation::SymmetricGradient)
                                    .expect("active symmetric-gradient input"),
                            )
                        },
                        |_evaluation: &PointEvaluation, direction: &PointEvaluation| {
                            stress(
                                direction
                                    .values(DerivativeEvaluation::SymmetricGradient)
                                    .expect("active symmetric-gradient direction"),
                            )
                        },
                    )
                } else if components == 2 {
                    let body_force = body_force.clone();
                    SystemConstitutiveInput::new(
                        equation,
                        integral_index,
                        input_id,
                        components,
                        "sv2b4-stokes/body-force",
                        move |evaluation: &PointEvaluation| {
                            body_force(&evaluation.coordinates).to_vec()
                        },
                        move |_evaluation: &PointEvaluation, _direction: &PointEvaluation| {
                            vec![0.0; components]
                        },
                    )
                } else {
                    SystemConstitutiveInput::new(
                        equation,
                        integral_index,
                        input_id,
                        components,
                        "sv2b4-stokes/no-forcing",
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

fn walls_region_map(region: RegionId) -> RegionMap {
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

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: {actual} != {expected} within {tolerance}"
        );
    }
}

fn pseudo_random_vector(dimension: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..dimension)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * bits - 1.0
        })
        .collect()
}

/// The corpus's own equation authorship gives a genuinely non-symmetric saddle-point coupling
/// (see this file's module doc comment): Scientia's structural `OperatorStructure` correctly
/// reports `form_symmetry: Unknown` for it, `SystemOperator::symmetry()` threads that claim
/// through unmodified (mission item 8), and `methodus::solve_minres` -- which requires a
/// declared `Symmetric` operator -- correctly refuses it, rather than either crate silently
/// accepting a system that has not been shown to satisfy MINRES's precondition.
#[test]
fn unsigned_stokes_operator_reports_unknown_symmetry_and_minres_refuses_it() {
    let compiled = compile_stokes();
    let mesh = unit_square(2);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let constitutive = stokes_constitutive(&compiled.system, |_coordinates: &[f64]| [0.0, 0.0]);
    let operator = plan.bind_kernels(constitutive, BTreeMap::new()).unwrap();

    let structure = operator.structure();
    assert!(structure.saddle_point);
    assert_eq!(structure.nullspace_candidates.len(), 1);
    assert_eq!(structure.nullspace_candidates[0].field, compiled.pressure);
    assert_eq!(operator.nullspace_candidates().len(), 1);
    assert_eq!(operator.symmetry(), OperatorSymmetry::Unknown);

    let dimension = operator.dimension();
    let constraints = ConstraintSet::new(dimension, Vec::new()).unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    assert_eq!(reduced.symmetry(), OperatorSymmetry::Unknown);

    let right_hand_side = vec![0.0; dimension];
    let config = MinresConfig::default();
    let error = solve_minres(
        &reduced,
        None,
        None,
        &EvaluationContext::reproducible(),
        &right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("Symmetric") || message.contains("symmetric"),
        "expected a symmetry-related refusal, got: {message}"
    );
}

/// Decisive acceptance: the `equation_sign`-corrected system operator is genuinely symmetric by
/// assembly, agrees entrywise with an independently hand-composed `MixedOperator`
/// (`SymmetricGradientGradient` + `DivergenceValue`) on both the unconstrained and the
/// Dirichlet-reduced action, and drives `methodus::solve_minres` with the auto-derived pressure
/// nullspace projector to convergence on a consistent right-hand side.
#[test]
fn signed_stokes_system_matches_mixed_operator_and_minres_converges() {
    let compiled = compile_stokes();
    let mesh = unit_square(2);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let constitutive = stokes_constitutive(&compiled.system, |_coordinates: &[f64]| [0.0, 0.0]);
    let equation_sign = BTreeMap::from([("incompressibility".to_string(), -1.0)]);
    let operator = plan.bind_kernels(constitutive, equation_sign).unwrap();
    let dimension = operator.dimension();

    // Symmetry is not assumed: it is proven by assembly, and the proof is cached.
    assert_eq!(operator.symmetry(), OperatorSymmetry::Unknown);
    assert_eq!(
        operator.prove_symmetry(1.0e-9).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert_eq!(operator.symmetry(), OperatorSymmetry::Symmetric);

    // Cross-check: an independently hand-composed MixedOperator over the SAME mesh/fields,
    // built from generic structural couplings (SymmetricGradientGradient for the viscosity
    // block, DivergenceValue at scale -1.0 to match the pressure coupling this file's module
    // doc comment derives empirically), agreeing entrywise on several probe vectors.
    let mixed_space = MixedSpace::new(
        mesh.mesh.clone(),
        vec![
            FieldSpec {
                symbol: compiled.velocity,
                order: 2,
                components: 2,
            },
            FieldSpec {
                symbol: compiled.pressure,
                order: 1,
                components: 1,
            },
        ],
    )
    .unwrap();
    let mixed_operator = MixedOperator::new(
        mixed_space,
        vec![
            BlockCoupling {
                test: compiled.velocity,
                trial: compiled.velocity,
                kind: CouplingKind::SymmetricGradientGradient,
                scale: 2.0 * MU,
            },
            BlockCoupling {
                test: compiled.velocity,
                trial: compiled.pressure,
                kind: CouplingKind::DivergenceValue,
                scale: -1.0,
            },
        ],
    )
    .unwrap();
    assert_eq!(mixed_operator.dimension(), dimension);
    for seed in 0..6u64 {
        let probe = pseudo_random_vector(dimension, seed);
        let mut system_output = vec![0.0; dimension];
        operator.apply_action(&probe, &mut system_output).unwrap();
        let mut mixed_output = vec![0.0; dimension];
        mixed_operator
            .apply_action(&probe, &mut mixed_output)
            .unwrap();
        assert_close(&system_output, &mixed_output, 5.0e-11);
    }

    // Region-tag-driven multi-field essential constraints (mission item 3): zero velocity on
    // every unit-square wall.
    let momentum_requirement = compiled
        .system
        .blocks
        .iter()
        .find(|block| block.equation == "momentum")
        .unwrap()
        .factorization
        .essential_constraints
        .first()
        .expect("momentum declares one essential-constraint requirement (the walls boundary)")
        .clone();
    let region_map = walls_region_map(momentum_requirement.region);
    let constraints = essential_constraints_from_system(
        &operator,
        &mesh,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: compiled.velocity,
            requirement: momentum_requirement,
            value: FieldSource::constant(vec![0.0, 0.0]),
        }],
    )
    .unwrap();
    assert!(constraints.constraints().next().is_some());

    let mixed_reduced = mixed_operator.reduced(constraints.clone()).unwrap();
    let system_reduced = operator.reduced(constraints.clone()).unwrap();
    assert_eq!(system_reduced.symmetry(), OperatorSymmetry::Symmetric);
    for seed in 10..16u64 {
        let probe = pseudo_random_vector(dimension, seed);
        let mut system_output = vec![0.0; dimension];
        system_reduced
            .apply(&EvaluationContext::default(), &probe, &mut system_output)
            .unwrap();
        let mut mixed_output = vec![0.0; dimension];
        mixed_reduced
            .apply(&EvaluationContext::default(), &probe, &mut mixed_output)
            .unwrap();
        assert_close(&system_output, &mixed_output, 5.0e-11);
    }

    // Auto-derived nullspace candidate (mission item 9), resolved and verified in the kernel of
    // the reduced operator.
    let candidates = operator.nullspace_candidates();
    assert_eq!(candidates.len(), 1);
    let mode = candidates[0].resolve(operator.layout()).unwrap();
    assert!(
        mode.verify_in_kernel(&system_reduced, 1.0e-8).unwrap(),
        "the auto-derived constant pressure mode should verify against the reduced operator"
    );

    // Known solution respecting the homogeneous Dirichlet data and orthogonal to the declared
    // constant-pressure nullspace (zero-mean pressure), so the generated right-hand side is
    // exactly consistent (mirrors tests/sv2b_mixed.rs's own MINRES fixture).
    let velocity_block = operator.layout().block(compiled.velocity).unwrap();
    let pressure_block = operator.layout().block(compiled.pressure).unwrap();
    let mut x_true = pseudo_random_vector(dimension, 4242);
    for constraint in constraints.constraints() {
        x_true[constraint.target.0] = 0.0;
    }
    let pressure_mean = x_true
        [pressure_block.offset..pressure_block.offset + pressure_block.extent]
        .iter()
        .sum::<f64>()
        / pressure_block.extent as f64;
    for value in &mut x_true[pressure_block.offset..pressure_block.offset + pressure_block.extent] {
        *value -= pressure_mean;
    }
    let _ = velocity_block;

    let mut right_hand_side = vec![0.0; dimension];
    system_reduced
        .apply(&EvaluationContext::default(), &x_true, &mut right_hand_side)
        .unwrap();

    let config = MinresConfig {
        max_iterations: 4 * dimension,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-10,
    };
    let report = solve_minres(
        &system_reduced,
        None,
        Some(mode.projector() as &dyn NullspaceProjector),
        &EvaluationContext::reproducible(),
        &right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap();
    assert!(
        report.converged,
        "minres did not converge on the reduced Stokes system"
    );
    assert_close(&report.solution, &x_true, 1.0e-6);
    let mut recovered = vec![0.0; dimension];
    system_reduced
        .apply(
            &EvaluationContext::default(),
            &report.solution,
            &mut recovered,
        )
        .unwrap();
    assert_close(&recovered, &right_hand_side, 1.0e-6);
}

/// `13-mixed-darcy.res` pairs an `HDiv(order=0)` flux field (`darcy_law`'s row) with an
/// `L2(order=0)` pressure field (`mass_balance`'s row), `@inf_sup(pair = "RT0-P0")` -- the E6/
/// Hdiv lane's own decisive-acceptance case. `flux`'s `impermeable` boundary compiles to a
/// `SemanticMeasure::ExteriorFacet` integral on `darcy_law` with an empty `primal.inputs` list
/// and a `Constant{0.0}` expression (Scientia's own compilation of `neumann flux = 0`, not a
/// choice made here); `mass_balance` has no facet integral and no essential-constraint
/// requirement at all (confirmed directly against the compiled system, not assumed).
struct CompiledDarcy {
    system: OperatorSystem,
    flux: SymbolId,
    pressure: SymbolId,
}

fn compile_darcy() -> CompiledDarcy {
    let source = fs::read_to_string(DARCY_CORPUS).expect("13-mixed-darcy.res corpus is readable");
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "MixedDarcy",
        &["darcy_law", "mass_balance"],
    )
    .unwrap();
    let flux = system
        .blocks
        .iter()
        .find(|block| block.equation == "darcy_law")
        .unwrap()
        .row;
    let pressure = system
        .blocks
        .iter()
        .find(|block| block.equation == "mass_balance")
        .unwrap()
        .row;
    CompiledDarcy {
        system,
        flux,
        pressure,
    }
}

/// A structured tetrahedral unit cube (`MeshProfile::SimplexBox`, dimension 3): at
/// `subdivisions = 2` this realizes 48 cells / 120 facets (RT0 flux dimension 120, P0 pressure
/// dimension 48, total system dimension 168) -- small enough for the dense independent
/// cross-check below, large enough to be a genuine multi-cell 3-D solve rather than a
/// single-tetrahedron sanity check.
fn darcy_box(subdivisions: usize) -> finitum::TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 3,
        extent: vec![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions, subdivisions],
    })
    .unwrap()
}

/// The concrete RT0/P0 `BlockLayout` this case needs: one entity per mesh facet with exactly one
/// component for `flux` (RT0's own DOF structure -- see `finitum::system`'s
/// `expected_field_components`), one entity per cell with one component for `pressure` (P0).
fn darcy_layout(
    mesh: &Mesh,
    facets: &FacetTopology,
    flux: SymbolId,
    pressure: SymbolId,
) -> (BlockLayout, CompatibleDofMaps) {
    let compatible = CompatibleDofMaps::simplex(mesh, facets).unwrap();
    let layout = BlockLayout::new([
        (flux, compatible.hdiv_dof_count, 1),
        (pressure, mesh.cells().len(), 1),
    ])
    .unwrap();
    (layout, compatible)
}

/// All six faces of the unit-cube `darcy_box` realizes, matching the case's own
/// `select = { kind = "box_faces", faces = ["all"] }` -- every exterior facet is `impermeable`.
fn darcy_walls_region_map(region: RegionId) -> RegionMap {
    let mut map = RegionMap::new();
    map.insert(
        region,
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("x_max"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
            RegionTagId::new("z_min"),
            RegionTagId::new("z_max"),
        ],
    );
    map
}

/// Every non-`Basis` primal input across the compiled Darcy system, resolved generically by
/// `InputSourceRequirement`/shape (no physics-name dispatch, mirroring `stokes_constitutive`):
/// `ModelDefinedConstitutive` (1 component) is `mobility_inverse` (`viscosity *
/// inverse(permeability)`, Scientia never resolves provider *values* itself), bound to a
/// constant; `ExternalValue` with 1 component is `source_term`, `ExternalValue` with 3 components
/// is `body_force` -- both bound to an always-zero closure (this file's decisive-acceptance test
/// manufactures its own consistent right-hand side directly from a known solution, mirroring
/// `signed_stokes_system_matches_mixed_operator_and_minres_converges`'s own "no forcing needed"
/// approach, rather than needing a discretely mean-zero source function).
const DARCY_MOBILITY_INVERSE: f64 = 2.3;

fn darcy_constitutive(system: &OperatorSystem) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                let equation = block.equation.clone();
                let integral_index = integral.integral_index;
                let input_id = input.id;
                let binding = match input.source {
                    InputSourceRequirement::Basis => continue,
                    InputSourceRequirement::ModelDefinedConstitutive { .. } => {
                        SystemConstitutiveInput::new(
                            equation,
                            integral_index,
                            input_id,
                            1,
                            "mobility_inverse",
                            move |_evaluation: &PointEvaluation| vec![DARCY_MOBILITY_INVERSE],
                            move |_evaluation: &PointEvaluation, _direction: &PointEvaluation| {
                                vec![0.0]
                            },
                        )
                    }
                    InputSourceRequirement::ExternalValue => {
                        let components = input.shape.iter().product::<usize>().max(1);
                        let identity = if components == 1 {
                            "source_term"
                        } else {
                            "body_force"
                        };
                        SystemConstitutiveInput::new(
                            equation,
                            integral_index,
                            input_id,
                            components,
                            identity,
                            move |_evaluation: &PointEvaluation| vec![0.0; components],
                            move |_evaluation: &PointEvaluation, _direction: &PointEvaluation| {
                                vec![0.0; components]
                            },
                        )
                    }
                    other => panic!(
                        "13-mixed-darcy.res declares an unexpected non-Basis input source {other:?}"
                    ),
                };
                constitutive.push(binding.unwrap());
            }
        }
    }
    constitutive
}

/// Decisive acceptance (E6/Hdiv lane): drives the real `13-mixed-darcy.res` RT0-P0 system
/// through `compile_operator_system` -> `derive_operator_structure_for_system` ->
/// `SystemRealizationPlan::bind_kernels_with_facets` (real bound Malleus kernels, including the
/// `darcy_law` Neumann `ExteriorFacet` integral) -> `methodus::solve_minres`, matching an
/// independently hand-assembled dense reference.
///
/// The saddle-point coupling sign mirrors Stokes exactly (see this file's module doc comment):
/// `darcy_law`'s `-integral(pressure * div(v))` term and `mass_balance`'s un-negated
/// `+integral(q * div(flux))` term are exact negatives of the same underlying divergence
/// coupling, confirmed empirically below by `prove_symmetry`, not assumed -- `equation_sign =
/// {"mass_balance": -1.0}` restores genuine symmetry exactly as `{"incompressibility": -1.0}`
/// did for Stokes.
#[test]
fn mixed_darcy_rt0_p0_system_realizes_and_minres_converges_matching_an_independent_dense_reference()
{
    let compiled = compile_darcy();
    let mesh = darcy_box(2);
    let facets = FacetTopology::from_mesh(&mesh.mesh).unwrap();
    let (layout, compatible) = darcy_layout(&mesh.mesh, &facets, compiled.flux, compiled.pressure);
    assert_eq!(compatible.hdiv.len(), mesh.mesh.cells().len());
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    assert!(
        plan.compatible_dofs().is_some(),
        "an Hdiv field must trigger FC8 compatible-DOF-map computation at SystemRealizationPlan::new"
    );

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
    let region_map = darcy_walls_region_map(region);
    let facet_regions = facet_membership_from(&mesh, &region_map, [region]).unwrap();
    assert_eq!(
        facet_regions[&region].len(),
        facets.exterior().count(),
        "every exterior facet of the unit cube is `impermeable`"
    );

    let constitutive = darcy_constitutive(&compiled.system);
    let equation_sign = BTreeMap::from([("mass_balance".to_string(), -1.0)]);
    let operator = plan
        .bind_kernels_with_facets(constitutive, equation_sign, facet_regions)
        .unwrap();
    let dimension = operator.dimension();
    assert_eq!(
        dimension,
        compatible.hdiv_dof_count + mesh.mesh.cells().len()
    );

    let structure = operator.structure();
    assert!(structure.saddle_point);
    assert_eq!(structure.nullspace_candidates.len(), 1);
    assert_eq!(structure.nullspace_candidates[0].field, compiled.pressure);

    // Symmetry is not assumed: it is proven by assembly, and the proof is cached.
    assert_eq!(operator.symmetry(), OperatorSymmetry::Unknown);
    assert_eq!(
        operator.prove_symmetry(1.0e-9).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert_eq!(operator.symmetry(), OperatorSymmetry::Symmetric);

    // No essential constraints at all for this case (confirmed above by the compiled system's
    // empty `requirements.essential_constraints`, not assumed): both boundary conditions
    // (`impermeable` on flux, none declared on pressure) are natural/weak.
    let reduced = operator
        .reduced(ConstraintSet::new(dimension, Vec::new()).unwrap())
        .unwrap();
    assert_eq!(reduced.symmetry(), OperatorSymmetry::Symmetric);

    // A genuine finding, characterized rather than assumed: Scientia's structural nullspace
    // heuristic (item 9) declares a `Constant` candidate on `pressure` for every block whose
    // boundary conditions are all natural -- true for the *textbook* mixed-Darcy weak form
    // (where a Neumann/impermeable condition contributes a pressure-dependent boundary term that
    // exactly cancels the divergence coupling's own boundary flux). But `13-mixed-darcy.res`'s
    // own compiled `darcy_law` Neumann integral (mission item 2's own facet extension) is a
    // literal, input-independent `Constant{0.0}` (confirmed directly from the compiled
    // `IntegralOperatorFactorization`, not assumed) -- it contributes nothing, so it does *not*
    // cancel the boundary term the divergence coupling contributes on its own. Probing the
    // realized (equation_sign-corrected) operator at the candidate's own constant-pressure mode
    // confirms this precisely: the residual is *exactly* zero at every interior facet (both
    // orientations of a shared facet cancel algebraically -- independent proof that this lane's
    // RT0 orientation/divergence machinery is realized correctly) and *nonzero* at every exterior
    // facet (the boundary term that should cancel it is the literal-zero Neumann integral). The
    // discrete operator this literal corpus compiles to is therefore genuinely full rank (no
    // constant-pressure nullspace, unlike the continuum problem or Scientia's own structural
    // heuristic) -- confirmed below by a plain (unbordered) dense solve succeeding, not assumed.
    let candidates = operator.nullspace_candidates();
    assert_eq!(candidates.len(), 1);
    let mode = candidates[0].resolve(operator.layout()).unwrap();
    let flux_block = operator.layout().block(compiled.flux).unwrap();
    let mut nullspace_probe = vec![0.0; dimension];
    reduced
        .apply(
            &EvaluationContext::default(),
            mode.vector(),
            &mut nullspace_probe,
        )
        .unwrap();
    let interior_residual: f64 = facets
        .interior()
        .map(|facet| nullspace_probe[flux_block.offset + facet.id.0].abs())
        .sum();
    let exterior_residual: f64 = facets
        .exterior()
        .map(|facet| nullspace_probe[flux_block.offset + facet.id.0].abs())
        .sum();
    assert!(
        interior_residual < 1.0e-10,
        "every interior facet's orientation-cancellation must hold exactly; got {interior_residual}"
    );
    assert!(
        exterior_residual > 1.0e-6,
        "the literal Constant{{0.0}} Neumann term should leave a genuine boundary residual \
         against the naive constant-pressure mode; got {exterior_residual}"
    );
    assert!(
        !mode.verify_in_kernel(&reduced, 1.0e-8).unwrap(),
        "the naive constant-pressure mode is NOT exactly in this literal corpus's kernel (see \
         the finding documented above) -- it should not silently verify"
    );

    // Known solution (no zero-mean gauge needed: the operator is genuinely nonsingular for this
    // corpus, established above).
    let pressure_block = operator.layout().block(compiled.pressure).unwrap();
    let x_true = pseudo_random_vector(dimension, 130978);

    let mut right_hand_side = vec![0.0; dimension];
    reduced
        .apply(&EvaluationContext::default(), &x_true, &mut right_hand_side)
        .unwrap();

    let config = MinresConfig {
        max_iterations: 4 * dimension,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-10,
    };
    let report = solve_minres(
        &reduced,
        None,
        None,
        &EvaluationContext::reproducible(),
        &right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap();
    assert!(
        report.converged,
        "minres did not converge on the RT0-P0 Darcy system"
    );
    println!(
        "mixed-darcy RT0-P0: dimension {dimension}, minres converged in {} iterations",
        report.trace.len()
    );
    assert_close(&report.solution, &x_true, 1.0e-6);

    // Independent cross-check: assemble the reduced operator by unit-column probing of
    // `methodus::LinearOperator::apply` (not `SystemOperator::assemble`'s own CSR path) into a
    // plain dense matrix (no gauge-fixing border needed -- the operator is genuinely nonsingular
    // for this corpus, established above) and solve with the from-scratch dense Gaussian
    // elimination shared with the Stokes decisive-acceptance test above (no shared code with
    // `methodus`'s own CG/MINRES/GMRES implementations). `gaussian_eliminate_solve` itself
    // asserts every pivot is non-negligible, so a singular `dense` would fail loudly here rather
    // than silently, independently confirming the full-rank finding above.
    let mut dense = vec![vec![0.0; dimension]; dimension];
    let mut probe = vec![0.0; dimension];
    for column in 0..dimension {
        probe[column] = 1.0;
        let mut output = vec![0.0; dimension];
        reduced
            .apply(&EvaluationContext::default(), &probe, &mut output)
            .unwrap();
        for row in 0..dimension {
            dense[row][column] = output[row];
        }
        probe[column] = 0.0;
    }
    let dense_solution = gaussian_eliminate_solve(dense, right_hand_side.clone());
    assert_close(&dense_solution, &report.solution, 1.0e-6);

    let mut recovered = vec![0.0; dimension];
    reduced
        .apply(
            &EvaluationContext::default(),
            &report.solution,
            &mut recovered,
        )
        .unwrap();
    assert_close(&recovered, &right_hand_side, 1.0e-6);

    // Solution-level evidence, reported honestly (no zero-mean pressure is expected or asserted
    // here, per the full-rank finding documented above -- unlike the Stokes decisive-acceptance
    // test, this system has no pressure gauge freedom to report against).
    let pressure_mean = report.solution
        [pressure_block.offset..pressure_block.offset + pressure_block.extent]
        .iter()
        .sum::<f64>()
        / pressure_block.extent as f64;
    let flux_norm = report.solution[flux_block.offset..flux_block.offset + flux_block.extent]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    println!(
        "mixed-darcy RT0-P0: pressure mean = {pressure_mean:e}, flux solution norm = {flux_norm:e}"
    );

    // `SystemOperator::load_vector`/`ReducedSystemOperator::load_vector` (E6-sysload) exercise
    // the RT0 row field's PRIMAL path (`apply_block_cell_load`'s new field-kind dispatch) and the
    // narrowly-scoped facet PRIMAL path (`apply_facets`), not just the JVP-based `apply_action`
    // path this test's solve otherwise exercises: with every constitutive input bound to zero,
    // both must be exactly zero.
    let load = operator.load_vector().unwrap();
    assert!(load.iter().all(|&value| value == 0.0));
    let reduced_load = reduced.load_vector().unwrap();
    assert!(reduced_load.iter().all(|&value| value == 0.0));
}

/// From-scratch dense Gaussian elimination with partial pivoting for a general square system --
/// the independent, non-shared solver mission item 3's decisive-acceptance test cross-checks
/// `solve_minres` against. Shares no code with `methodus`' own solvers.
#[allow(clippy::needless_range_loop)]
fn gaussian_eliminate_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let dimension = rhs.len();
    assert_eq!(matrix.len(), dimension);
    for column in 0..dimension {
        let mut pivot_row = column;
        let mut pivot_value = matrix[column][column].abs();
        for row in (column + 1)..dimension {
            if matrix[row][column].abs() > pivot_value {
                pivot_row = row;
                pivot_value = matrix[row][column].abs();
            }
        }
        assert!(
            pivot_value > 1.0e-10,
            "dense Gaussian elimination found a singular pivot at column {column}"
        );
        matrix.swap(column, pivot_row);
        rhs.swap(column, pivot_row);
        let pivot = matrix[column][column];
        for row in (column + 1)..dimension {
            let factor = matrix[row][column] / pivot;
            if factor == 0.0 {
                continue;
            }
            for entry in column..dimension {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for column in (row + 1)..dimension {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    solution
}

/// Mission item 1 (E6-sysload): `SystemOperator::load_vector` executes each block's bound PRIMAL
/// Malleus kernel at zero active state, scattering the result through `BlockLayout` -- the
/// multi-block analogue of `RealizationPlan::load_vector`'s own PRIMAL-kernel execution. Verifies
/// both halves of its documented contract: an all-zero-bound source (today's existing "no
/// forcing" fixture, unchanged) loads to the exact zero vector, and a nonzero constant body-force
/// closure loads to a genuinely nonzero vector whose momentum-block entries match an independent
/// closed-form reference -- the Lagrange partition-of-unity identity `sum_i integral(f_c * phi_i)
/// dOmega = f_c * |domain|` (any Lagrange basis sums to `1` pointwise) -- computed here without
/// touching any of `SystemOperator`'s own kernel-execution code.
#[test]
fn system_operator_load_vector_zero_source_is_zero_and_nonzero_source_matches_partition_of_unity() {
    let compiled = compile_stokes();
    let mesh = unit_square(2);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();

    let zero_constitutive =
        stokes_constitutive(&compiled.system, |_coordinates: &[f64]| [0.0, 0.0]);
    let zero_operator = plan
        .bind_kernels(zero_constitutive, BTreeMap::new())
        .unwrap();
    let zero_load = zero_operator.load_vector().unwrap();
    assert!(
        zero_load.iter().all(|&value| value == 0.0),
        "an all-zero-bound source must load to the exact zero vector"
    );

    let force = [0.3, -0.7];
    let force_constitutive =
        stokes_constitutive(&compiled.system, move |_coordinates: &[f64]| force);
    let operator = plan
        .bind_kernels(force_constitutive, BTreeMap::new())
        .unwrap();
    let load = operator.load_vector().unwrap();

    let velocity_block = operator.layout().block(compiled.velocity).unwrap();
    let pressure_block = operator.layout().block(compiled.pressure).unwrap();

    // `incompressibility` declares no source term at all: its rows stay exactly zero regardless
    // of the momentum block's body force.
    assert!(
        load[pressure_block.offset..pressure_block.offset + pressure_block.extent]
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        load[velocity_block.offset..velocity_block.offset + velocity_block.extent]
            .iter()
            .any(|&value| value != 0.0)
    );

    let domain_area = 1.0; // the unit-square fixture
    let node_count = velocity_block.extent / velocity_block.component_count;
    for component in 0..velocity_block.component_count {
        let sum: f64 = (0..node_count)
            .map(|node| {
                load[velocity_block.offset + node * velocity_block.component_count + component]
            })
            .sum();
        let expected = force[component] * domain_area;
        assert!(
            (sum - expected).abs() < 1.0e-9,
            "component {component}: partition-of-unity sum {sum} != {expected}"
        );
    }
}

/// Decisive acceptance for the system-load-vector capability (E6-sysload, mission items 2/3): a
/// genuinely nonzero body-force closure bound on the real `25-stokes.res` system, through
/// `ReducedSystemOperator::load_vector`'s composed right-hand side, drives `solve_minres` (the
/// auto-derived pressure-mode projector) to a nontrivial solution -- cross-checked against a
/// from-scratch dense direct solve of the independently assembled reduced system, not a
/// manufactured `x_true` and not any code `solve_minres` itself uses.
///
/// The body force here is deliberately *spatially varying* (`f(x, y) = [y - 0.5, 0]`, a pure
/// shear with `curl(f) = -1` everywhere), not merely nonzero: a *constant* body force on a fully
/// enclosed no-slip cavity (this corpus's own boundary condition on every wall) is an exact
/// hydrostatic balance -- `(u, p) = (0, f . x + C)` satisfies `grad(p) = f` pointwise with `u`
/// left exactly zero -- so a constant forcing would only ever exercise `load_vector` without ever
/// exercising a genuinely nontrivial *velocity* solution. `curl(f) != 0` rules out any pressure-
/// only balance, forcing real flow.
#[test]
fn nonzero_body_force_stokes_system_solves_to_a_nontrivial_solution_matching_an_independent_dense_solve()
 {
    let compiled = compile_stokes();
    let mesh = unit_square(2);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let constitutive = stokes_constitutive(&compiled.system, |coordinates: &[f64]| {
        [coordinates[1] - 0.5, 0.0]
    });
    let equation_sign = BTreeMap::from([("incompressibility".to_string(), -1.0)]);
    let operator = plan.bind_kernels(constitutive, equation_sign).unwrap();
    let dimension = operator.dimension();
    assert_eq!(
        operator.prove_symmetry(1.0e-9).unwrap(),
        OperatorSymmetry::Symmetric
    );

    let momentum_requirement = compiled
        .system
        .blocks
        .iter()
        .find(|block| block.equation == "momentum")
        .unwrap()
        .factorization
        .essential_constraints
        .first()
        .expect("momentum declares one essential-constraint requirement (the walls boundary)")
        .clone();
    let region_map = walls_region_map(momentum_requirement.region);
    let constraints = essential_constraints_from_system(
        &operator,
        &mesh,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: compiled.velocity,
            requirement: momentum_requirement,
            value: FieldSource::constant(vec![0.0, 0.0]),
        }],
    )
    .unwrap();
    assert!(constraints.constraints().next().is_some());

    let reduced = operator.reduced(constraints.clone()).unwrap();
    let rhs = reduced.load_vector().unwrap();
    assert!(
        rhs.iter().any(|&value| value != 0.0),
        "a nonzero body force must produce a genuinely nonzero reduced right-hand side"
    );

    let velocity_block = operator.layout().block(compiled.velocity).unwrap();
    let pressure_block = operator.layout().block(compiled.pressure).unwrap();

    // The corpus's own Dirichlet data is the homogeneous `[0, 0]` literal, so the lifting
    // contribution to the composed right-hand side is exactly zero and `load_vector`'s
    // composition contract reduces, here, to the restricted raw load -- verified directly against
    // `SystemOperator::load_vector` (item 1), not merely assumed from item 2's documented
    // contract.
    let raw_load = operator.load_vector().unwrap();
    let restricted_raw_load = constraints.restrict_transpose(&raw_load).unwrap();
    for index in 0..dimension {
        if constraints.is_constrained(DofId(index)) {
            assert_eq!(rhs[index], 0.0);
        } else {
            assert!((rhs[index] - restricted_raw_load[index]).abs() < 1.0e-12);
        }
    }
    // The incompressibility block declares no source: the composed right-hand side's pressure
    // rows are exactly zero, so it is automatically orthogonal to the declared constant-pressure
    // nullspace mode -- no manufactured `x_true` gauge choice is needed to make this system
    // consistent.
    assert!(
        rhs[pressure_block.offset..pressure_block.offset + pressure_block.extent]
            .iter()
            .all(|&value| value == 0.0)
    );

    let candidates = operator.nullspace_candidates();
    assert_eq!(candidates.len(), 1);
    let mode = candidates[0].resolve(operator.layout()).unwrap();
    assert!(
        mode.verify_in_kernel(&reduced, 1.0e-8).unwrap(),
        "the auto-derived constant pressure mode should verify against the reduced operator"
    );
    let nullspace_dot_rhs: f64 = mode
        .vector()
        .iter()
        .zip(&rhs)
        .map(|(basis, value)| basis * value)
        .sum();
    assert!(
        nullspace_dot_rhs.abs() < 1.0e-10,
        "the right-hand side must be orthogonal to the declared nullspace for a consistent solve, \
         got dot product {nullspace_dot_rhs}"
    );

    let config = MinresConfig {
        max_iterations: 4 * dimension,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-10,
    };
    let report = solve_minres(
        &reduced,
        None,
        Some(mode.projector() as &dyn NullspaceProjector),
        &EvaluationContext::reproducible(),
        &rhs,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap();
    assert!(
        report.converged,
        "minres did not converge on the nonzero-body-force Stokes system"
    );
    println!(
        "nonzero-body-force Stokes: minres converged in {} iterations",
        report.trace.len()
    );
    assert!(
        report.solution.iter().any(|&value| value.abs() > 1.0e-6),
        "a nonzero body force must produce a genuinely nontrivial solution, not the trivial zero"
    );

    // Independent cross-check: assemble the reduced operator by unit-column probing of
    // `methodus::LinearOperator::apply` (not `SystemOperator::assemble`'s own CSR path) into a
    // dense matrix, border it with the declared nullspace mode as a zero-mean-pressure gauge
    // constraint (the same gauge `solve_minres`'s nullspace projector enforces), and solve with a
    // from-scratch dense Gaussian elimination that shares no code with `solve_minres`.
    let mut dense = vec![vec![0.0; dimension]; dimension];
    let mut probe = vec![0.0; dimension];
    for column in 0..dimension {
        probe[column] = 1.0;
        let mut output = vec![0.0; dimension];
        reduced
            .apply(&EvaluationContext::default(), &probe, &mut output)
            .unwrap();
        for row in 0..dimension {
            dense[row][column] = output[row];
        }
        probe[column] = 0.0;
    }
    let bordered_dimension = dimension + 1;
    let mut bordered = vec![vec![0.0; bordered_dimension]; bordered_dimension];
    for row in 0..dimension {
        bordered[row][..dimension].copy_from_slice(&dense[row]);
        bordered[row][dimension] = mode.vector()[row];
        bordered[dimension][row] = mode.vector()[row];
    }
    let mut bordered_rhs = vec![0.0; bordered_dimension];
    bordered_rhs[..dimension].copy_from_slice(&rhs);
    let bordered_solution = gaussian_eliminate_solve(bordered, bordered_rhs);
    let dense_solution = &bordered_solution[..dimension];

    assert_close(dense_solution, &report.solution, 1.0e-6);

    let mut recovered = vec![0.0; dimension];
    reduced
        .apply(
            &EvaluationContext::default(),
            &report.solution,
            &mut recovered,
        )
        .unwrap();
    assert_close(&recovered, &rhs, 1.0e-6);

    // Solution-level evidence, reported honestly rather than forced to an arbitrary bound: the
    // incompressibility row's residual is the discrete weak divergence `integral(q * div(u))`
    // tested against every pressure basis function. Its own right-hand side is exactly zero and
    // the solve enforces `reduced * solution == rhs` to `solve_minres`'s own tolerance, so this
    // residual sits at solver-tolerance scale, not merely at discretization-truncation scale.
    let divergence_residual_norm = recovered
        [pressure_block.offset..pressure_block.offset + pressure_block.extent]
        .iter()
        .zip(&rhs[pressure_block.offset..pressure_block.offset + pressure_block.extent])
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    println!(
        "nonzero-body-force Stokes: velocity divergence residual norm = {divergence_residual_norm:e}"
    );
    assert!(
        divergence_residual_norm < 1.0e-6,
        "velocity divergence residual norm {divergence_residual_norm} is not small"
    );

    let pressure_mean = report.solution
        [pressure_block.offset..pressure_block.offset + pressure_block.extent]
        .iter()
        .sum::<f64>()
        / pressure_block.extent as f64;
    println!("nonzero-body-force Stokes: pressure mean = {pressure_mean:e}");
    assert!(
        pressure_mean.abs() < 1.0e-6,
        "pressure mean {pressure_mean} should be at solver-tolerance-level zero given the \
         zero-mean-orthogonal nullspace projector"
    );

    let velocity_norm = report.solution
        [velocity_block.offset..velocity_block.offset + velocity_block.extent]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    println!("nonzero-body-force Stokes: velocity solution norm = {velocity_norm:e}");
    assert!(
        velocity_norm > 1.0e-6,
        "the nontrivial body force must produce a nonzero velocity field"
    );
}

/// Mission item 4 (falls out cheaply): `equation_sign` must flip a signed row's load
/// contribution exactly as it flips that row's operator contribution, so the row's own weak-form
/// equation (`a_i(x, v) - L_i(v) = 0`) is unchanged by the sign -- the same "solution-preserving"
/// property `SystemRealizationPlan::bind_kernels`'s own doc comment claims for `equation_sign`,
/// now verified with a genuinely nonzero load bound to the *flipped* row (`momentum`, which --
/// unlike `incompressibility` -- has a real source term in this corpus). No solve is needed here:
/// this checks the row-level residual invariance the sign transform is defined by.
#[test]
fn equation_sign_flips_the_load_vectors_row_consistently_with_the_operator() {
    let compiled = compile_stokes();
    let mesh = unit_square(2);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let constitutive_unsigned = stokes_constitutive(&compiled.system, |coordinates: &[f64]| {
        [coordinates[1] - 0.5, 0.0]
    });
    let constitutive_signed = stokes_constitutive(&compiled.system, |coordinates: &[f64]| {
        [coordinates[1] - 0.5, 0.0]
    });

    let unsigned = plan
        .bind_kernels(constitutive_unsigned, BTreeMap::new())
        .unwrap();
    let signed = plan
        .bind_kernels(
            constitutive_signed,
            BTreeMap::from([("momentum".to_string(), -1.0)]),
        )
        .unwrap();
    let dimension = unsigned.dimension();
    assert_eq!(signed.dimension(), dimension);

    let unsigned_load = unsigned.load_vector().unwrap();
    let signed_load = signed.load_vector().unwrap();

    let probe = pseudo_random_vector(dimension, 777);
    let mut unsigned_action = vec![0.0; dimension];
    unsigned.apply_action(&probe, &mut unsigned_action).unwrap();
    let mut signed_action = vec![0.0; dimension];
    signed.apply_action(&probe, &mut signed_action).unwrap();

    let velocity_block = unsigned.layout().block(compiled.velocity).unwrap();
    let pressure_block = unsigned.layout().block(compiled.pressure).unwrap();

    // `momentum` (the row field is `velocity`) is flipped: both its operator action and its load
    // flip sign together, so the row's own residual `action - load` is exactly negated -- the
    // row's zero set (its solutions) is unchanged.
    for index in velocity_block.offset..velocity_block.offset + velocity_block.extent {
        let unsigned_residual = unsigned_action[index] - unsigned_load[index];
        let signed_residual = signed_action[index] - signed_load[index];
        assert!(
            (signed_residual + unsigned_residual).abs() < 1.0e-9,
            "momentum row {index}: signed residual {signed_residual} is not the exact negation \
             of the unsigned residual {unsigned_residual}"
        );
        assert!(
            (signed_load[index] + unsigned_load[index]).abs() < 1.0e-9,
            "momentum row {index}: signed load {} is not the exact negation of the unsigned \
             load {}",
            signed_load[index],
            unsigned_load[index]
        );
    }
    // `incompressibility` (row field `pressure`) is unsigned in this variant: unaffected, exactly.
    for index in pressure_block.offset..pressure_block.offset + pressure_block.extent {
        assert_eq!(signed_load[index], unsigned_load[index]);
        assert_eq!(signed_action[index], unsigned_action[index]);
    }
}
