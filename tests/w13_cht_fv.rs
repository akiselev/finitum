//! W13 A4 (SV2C L9a, SV9 E5 stage 1): prescribed-velocity convection–diffusion on TensorBox
//! FV (Graetz flow between parallel plates) and the single-mesh two-region realization
//! (decision D5, gate 1b). Every acceptance bar below is fixed with its basis before the run.
use std::collections::BTreeMap;
use std::sync::Arc;

use finitum::{
    FieldSource, FinitumError, FvRealization, FvRegionProperties, FvRegions, FvTopology, RegionMap,
    RegionTagId, TensorAxis, TensorBox,
};
use methodus::{
    BdfConfig, BdfOrder, BdfState, DaeOperator, EvaluationContext, ForcingPolicy, GmresConfig,
    KrylovMethod, NewtonKrylovConfig, NonlinearOperator, NumericError, StepOutcome, bdf_step,
    solve_newton_krylov,
};
use quantitas::UnitRegistry;
use scientia::{
    BoundaryFluxSemantics, ConservationLawProgram, ConservationLawSystem, FvMethodIntent,
    GradientScheme, LimiterFamily, NumericalFluxIntent, PositivityIntent, ReconstructionIntent,
    ReconstructionVariables, SourceTreatment, compile_conservation_law_system, compile_semantics,
    lower_conservation_law,
};

const CHT: &str = include_str!("fixtures/fv-cht.res");

/// Plate gap `H`, mean velocity `Ū`, diffusivity `α` (ρc_p = 1): `Pe = Ū D_h/α = 100`.
const H: f64 = 1.0;
const U_MEAN: f64 = 1.0;
const ALPHA: f64 = 0.02;
const D_H: f64 = 2.0 * H;
const LENGTH: f64 = 6.0;
/// Shah & London fully developed parallel-plate Nusselt numbers (D_h based, Pe → ∞; the
/// axial-conduction correction at Pe = 100 is below 0.1%).
const NU_T: f64 = 7.541;
const NU_H: f64 = 8.235;

fn cx() -> EvaluationContext {
    EvaluationContext::reproducible()
}

fn law(
    model: &str,
    flux: NumericalFluxIntent,
    reconstruction: ReconstructionIntent,
) -> (ConservationLawSystem, Arc<ConservationLawProgram>) {
    let module = compile_semantics(CHT, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("{diagnostics:#?}"))
        .semantic;
    let system = compile_conservation_law_system(&module, model, "balance", "T").unwrap();
    let intent = FvMethodIntent {
        numerical_flux: flux,
        reconstruction,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    };
    let program = lower_conservation_law(&module, &system, &intent, &[]).unwrap();
    (system, Arc::new(program))
}

fn muscl() -> ReconstructionIntent {
    ReconstructionIntent::Muscl {
        variables: ReconstructionVariables::Conserved,
        gradient: GradientScheme::LeastSquares,
        limiter: LimiterFamily::None,
    }
}

/// Boundary names onto TensorBox tags.
fn regions(system: &ConservationLawSystem) -> RegionMap {
    let mut map = RegionMap::new();
    for boundary in &system.boundaries {
        let (BoundaryFluxSemantics::GhostState { region, .. }
        | BoundaryFluxSemantics::PrescribedFlux { region, .. }) = boundary;
        let tags: &[&str] = match boundary.region_name() {
            "inlet" | "west" | "left" => &["x_min"],
            "outlet" | "east" | "right" | "interface" => &["x_max"],
            "walls" => &["y_min", "y_max"],
            other => panic!("unexpected boundary {other}"),
        };
        map.insert(*region, tags.iter().map(|tag| RegionTagId::new(*tag)));
    }
    map
}

fn data(values: &[(&str, f64)]) -> BTreeMap<String, f64> {
    values.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn poiseuille(y: f64) -> f64 {
    6.0 * U_MEAN * (y / H) * (1.0 - y / H)
}

/// `ux` = Poiseuille inside the channel `0 ≤ y ≤ H` (zero outside, in walls), `uy` = 0.
fn velocity() -> BTreeMap<String, FieldSource> {
    BTreeMap::from([
        (
            "ux".to_string(),
            FieldSource::fallible(|point: &[f64], _| {
                let y = point[1];
                Ok(vec![if (0.0..=H).contains(&y) {
                    poiseuille(y)
                } else {
                    0.0
                }])
            }),
        ),
        ("uy".to_string(), FieldSource::constant([0.0])),
    ])
}

fn channel(nx: usize, ny: usize) -> Arc<FvTopology> {
    Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![
                TensorAxis::uniform(0.0, LENGTH, nx),
                TensorAxis::uniform(0.0, H, ny),
            ],
        })
        .unwrap(),
    )
}

/// One fluid region (`k = α`, `C = 1`): the uniform diffusion of a single-region problem.
fn fluid_only(topology: &FvTopology) -> FvRegions {
    FvRegions {
        cell_regions: vec![RegionTagId::new("fluid"); topology.cell_count()],
        properties: BTreeMap::from([(
            RegionTagId::new("fluid"),
            FvRegionProperties {
                conductivity: ALPHA,
                capacity: 1.0,
                stagnant: false,
            },
        )]),
    }
}

struct Steady<'a>(&'a FvRealization);

impl NonlinearOperator for Steady<'_> {
    fn dimension(&self) -> usize {
        DaeOperator::dimension(self.0)
    }
    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let rate = vec![0.0; state.len()];
        DaeOperator::residual(self.0, context, 0.0, state, &rate, output)
    }
    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; state.len()];
        DaeOperator::jacobian_vector_product(
            self.0, context, 0.0, state, &zero, direction, &zero, output,
        )
    }
}

/// Steady state through Newton–GMRES (unpreconditioned, restart 200).
fn steady(operator: &FvRealization, initial: f64) -> Vec<f64> {
    let n = DaeOperator::dimension(operator);
    let report = solve_newton_krylov(
        &Steady(operator),
        &cx(),
        &vec![initial; n],
        &KrylovMethod::Gmres(GmresConfig {
            max_iterations: 40_000,
            restart: 200,
            absolute_tolerance: 1e-14,
            relative_tolerance: 1e-12,
        }),
        None,
        None,
        &NewtonKrylovConfig {
            max_iterations: 8,
            absolute_tolerance: 1e-13,
            relative_tolerance: 1e-11,
            forcing: ForcingPolicy::Constant { forcing: 1e-12 },
            ..NewtonKrylovConfig::default()
        },
    )
    .unwrap();
    assert!(report.converged, "Newton-GMRES did not converge");
    report.state
}

fn residual(operator: &FvRealization, state: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; state.len()];
    DaeOperator::residual(
        operator,
        &cx(),
        0.0,
        state,
        &vec![0.0; state.len()],
        &mut out,
    )
    .unwrap();
    out
}

fn jvp(operator: &FvRealization, state: &[f64], direction: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; state.len()];
    let zero = vec![0.0; state.len()];
    DaeOperator::jacobian_vector_product(
        operator,
        &cx(),
        0.0,
        state,
        &zero,
        direction,
        &zero,
        &mut out,
    )
    .unwrap();
    out
}

/// Dense Gaussian elimination of the linear steady problem `R(u) = 0` (exact to round-off).
fn dense_steady(operator: &FvRealization) -> Vec<f64> {
    let n = DaeOperator::dimension(operator);
    let zero = vec![0.0; n];
    let b = residual(operator, &zero);
    let mut matrix = vec![vec![0.0; n + 1]; n];
    for column in 0..n {
        let mut e = vec![0.0; n];
        e[column] = 1.0;
        for (row, value) in jvp(operator, &zero, &e).into_iter().enumerate() {
            matrix[row][column] = value;
        }
    }
    for row in 0..n {
        matrix[row][n] = -b[row];
    }
    solve_dense(matrix)
}

fn solve_dense(mut matrix: Vec<Vec<f64>>) -> Vec<f64> {
    let n = matrix.len();
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|a, b| matrix[*a][pivot].abs().total_cmp(&matrix[*b][pivot].abs()))
            .unwrap();
        matrix.swap(pivot, best);
        let (head, tail) = matrix.split_at_mut(pivot + 1);
        let pivot_row = &head[pivot];
        for row in tail {
            let factor = row[pivot] / pivot_row[pivot];
            for (value, above) in row[pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *value -= factor * above;
            }
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let tail = (row + 1..n)
            .map(|c| matrix[row][c] * solution[c])
            .sum::<f64>();
        solution[row] = (matrix[row][n] - tail) / matrix[row][row];
    }
    solution
}

fn code(error: &FinitumError) -> String {
    let message = error.to_string();
    message
        .split(':')
        .find(|part| part.trim_start().starts_with("FV_"))
        .map(|part| part.trim().to_string())
        .unwrap_or(message)
}

/// Per-column bulk (mixing-cup) temperature and mean wall flux (outward from the fluid).
struct Columns {
    x: Vec<f64>,
    bulk: Vec<f64>,
    wall_flux: Vec<f64>,
    /// Mean of the two wall-adjacent cell values.
    near_wall: Vec<f64>,
}

fn columns(operator: &FvRealization, state: &[f64], nx: usize, ny: usize) -> Columns {
    let topology = operator.topology();
    let centroids = topology.cell_centroids();
    let fluxes = operator.face_fluxes(0.0, state).unwrap();
    let mut out = Columns {
        x: vec![0.0; nx],
        bulk: vec![0.0; nx],
        wall_flux: vec![0.0; nx],
        near_wall: vec![0.0; nx],
    };
    for i in 0..nx {
        let (mut weighted, mut weight) = (0.0, 0.0);
        for j in 0..ny {
            let cell = i + j * nx;
            let u = poiseuille(centroids[2 * cell + 1]);
            weighted += u * state[cell];
            weight += u;
        }
        out.x[i] = centroids[2 * i];
        out.bulk[i] = weighted / weight;
    }
    for tag in ["y_min", "y_max"] {
        let block = &topology.boundary()[&RegionTagId::new(tag)];
        for (face, flux) in fluxes.boundary[&RegionTagId::new(tag)].iter().enumerate() {
            let cell = block.minus[face];
            out.wall_flux[cell % nx] += 0.5 * flux;
            out.near_wall[cell % nx] += 0.5 * state[cell];
        }
    }
    out
}

fn mean_over(x: &[f64], values: &[f64], range: std::ops::RangeInclusive<f64>) -> f64 {
    let picked = x
        .iter()
        .zip(values)
        .filter(|(x, _)| range.contains(x))
        .map(|(_, v)| *v)
        .collect::<Vec<_>>();
    assert!(!picked.is_empty());
    picked.iter().sum::<f64>() / picked.len() as f64
}

/// Fully developed isothermal-wall Nusselt number over `4 ≤ x ≤ 5`, with the steady state.
fn graetz_isothermal(
    flux: NumericalFluxIntent,
    reconstruction: ReconstructionIntent,
    nx: usize,
    ny: usize,
) -> (f64, FvRealization, Vec<f64>) {
    let (system, program) = law("Graetz2D", flux, reconstruction);
    let topology = channel(nx, ny);
    let operator = FvRealization::with_face_fields(
        &system,
        program,
        topology.clone(),
        &regions(&system),
        &data(&[("t_in", 1.0), ("t_wall", 0.0)]),
        &velocity(),
    )
    .unwrap()
    .with_regions(fluid_only(&topology))
    .unwrap();
    let state = steady(&operator, 0.5);
    let c = columns(&operator, &state, nx, ny);
    let local = (0..nx)
        .map(|i| c.wall_flux[i] * D_H / (ALPHA * (c.bulk[i] - 0.0)))
        .collect::<Vec<_>>();
    (mean_over(&c.x, &local, 4.0..=5.0), operator, state)
}

#[test]
fn graetz_isothermal_walls_reach_the_fully_developed_nusselt_number() {
    // Bar (fixed before running): 2% at ny = 64. Basis: second-order cross-stream TPFA error
    // (~(1/64)²) plus the axial numerical diffusion of upwinding (u·dx/2 ≈ 0.025 vs α = 0.02,
    // i.e. an effective Pe ≈ 44, whose plate Nusselt shift is well under 1%).
    let (upwind, operator, state) = graetz_isothermal(
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
        120,
        64,
    );
    println!("isothermal Nu, LLF upwind, 120x64: {upwind:.5}");
    assert!((upwind - NU_T).abs() / NU_T <= 0.02, "upwind Nu {upwind}");

    // The discrete balance and the per-face wall flux extraction.
    let receipt = operator.conservation_receipt(0.0, &state).unwrap();
    assert!(receipt.balanced, "{receipt:?}");
    let fluxes = operator.face_fluxes(0.0, &state).unwrap();
    let topology = operator.topology();
    let mut net = 0.0;
    let mut scale = 0.0;
    for (tag, block) in topology.boundary() {
        for (area, flux) in block.area.iter().zip(&fluxes.boundary[tag]) {
            net += area * flux;
            scale += (area * flux).abs();
        }
    }
    // Steady: the boundary flows cancel up to the solver residual. Bar 1e-8 of the boundary
    // flow scale (Newton relative tolerance 1e-11, n ≈ 7680 cells, ‖·‖₁ ≤ √n‖·‖₂).
    assert!(
        net.abs() <= 1e-8 * scale,
        "net boundary flow {net} of {scale}"
    );
    let residual_sum = residual(&operator, &state).iter().sum::<f64>();
    assert!((residual_sum - net).abs() <= 1e-12 * scale.max(1.0));

    let (fine, _, _) = graetz_isothermal(NumericalFluxIntent::LocalLaxFriedrichs, muscl(), 120, 64);
    println!("isothermal Nu, MUSCL LSQ, 120x64: {fine:.5}");
    assert!((fine - NU_T).abs() / NU_T <= 0.02, "MUSCL Nu {fine}");
}

/// Pre-registered W13 A4 bar, kept unchanged: "MUSCL's error against 7.541 strictly decreases
/// from ny = 16 to 32 to 64". A4's setup (Pe = 100, window 4 <= x <= 5, i.e. x/(D_h Pe) =
/// 0.02-0.025) converged at second order to about 7.548, so its premise failed. W13 A7 changed
/// only the physical SETUP (user decision 2026-10-19), chosen from theory before running once:
/// - Pe = Ū D_h/α = 1000 (α = 0.002). Axial conduction enters the fully developed parallel-plate
///   Nu_T as O(Pe⁻²) (Shah & London; the Pe = 100 shift is below 0.1%), so at Pe = 1000 it is
///   below 1e-5 relative, far under the ny = 64 discretization error (~1e-4).
/// - Window 200 <= x <= 250, x* = x/(D_h Pe) = 0.10-0.125, well past the thermal entrance
///   (Shah & London: fully developed for x* ≳ 0.1). With H = Ū = 1 the bulk decays as
///   exp(−4 Nu x*); the second Graetz mode (λ₁²/λ₀² ≈ 9) decays as exp(−4·60·x*) relative to
///   the first, ≈ e^{−24} at x* = 0.1.
/// - Channel length 260 (outlet 10 cells past the window; the outflow influence length α/Ū =
///   0.002 is far below a cell), nx = 260 (Δx = 1). In the developed region T = φ(y)e^{−kx};
///   unlimited MUSCL advection and TPFA axial diffusion are linear and act on e^{−kx} as exact
///   multiples, so the axial grid only rescales k and leaves the cross-stream eigenfunction
///   (hence Nu) unchanged up to the O(αk/Ū) ≈ 3e-5 axial-conduction term.
#[test]
fn muscl_isothermal_nusselt_error_decreases_under_refinement() {
    const ALPHA_DEVELOPED: f64 = 0.002;
    const LENGTH_DEVELOPED: f64 = 260.0;
    const NX_DEVELOPED: usize = 260;
    let nusselt = |ny: usize| {
        let (system, program) = law("Graetz2D", NumericalFluxIntent::LocalLaxFriedrichs, muscl());
        let topology = Arc::new(
            FvTopology::from_tensor_box(&TensorBox {
                axes: vec![
                    TensorAxis::uniform(0.0, LENGTH_DEVELOPED, NX_DEVELOPED),
                    TensorAxis::uniform(0.0, H, ny),
                ],
            })
            .unwrap(),
        );
        let mut fluid = fluid_only(&topology);
        for properties in fluid.properties.values_mut() {
            properties.conductivity = ALPHA_DEVELOPED;
        }
        let operator = FvRealization::with_face_fields(
            &system,
            program,
            topology.clone(),
            &regions(&system),
            &data(&[("t_in", 1.0), ("t_wall", 0.0)]),
            &velocity(),
        )
        .unwrap()
        .with_regions(fluid)
        .unwrap();
        let state = steady(&operator, 0.5);
        let c = columns(&operator, &state, NX_DEVELOPED, ny);
        let local = (0..NX_DEVELOPED)
            .map(|i| c.wall_flux[i] * D_H / (ALPHA_DEVELOPED * c.bulk[i]))
            .collect::<Vec<_>>();
        mean_over(&c.x, &local, 200.0..=250.0)
    };
    let values = [16, 32, 64].map(nusselt);
    println!("developed isothermal Nu (Pe = 1000), MUSCL LSQ, ny = 16/32/64: {values:?}");
    assert!(
        (values[1] - NU_T).abs() < (values[0] - NU_T).abs(),
        "{values:?}"
    );
    assert!(
        (values[2] - NU_T).abs() < (values[1] - NU_T).abs(),
        "{values:?}"
    );
}

#[test]
fn graetz_uniform_wall_flux_reaches_the_fully_developed_nusselt_number() {
    // Bar (fixed before running): 2% at ny = 64, upwind; same basis as the isothermal case.
    // The wall temperature is the half-cell extrapolation T_P + q (δ·n)/α consistent with the
    // prescribed face flux.
    let (nx, ny) = (120, 64);
    let q_in = ALPHA;
    let (system, program) = law(
        "GraetzFlux2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
    );
    let topology = channel(nx, ny);
    let operator = FvRealization::with_face_fields(
        &system,
        program,
        topology.clone(),
        &regions(&system),
        &data(&[("t_in", 0.0), ("q_out", -q_in)]),
        &velocity(),
    )
    .unwrap()
    .with_regions(fluid_only(&topology))
    .unwrap();
    let state = steady(&operator, 0.0);
    let c = columns(&operator, &state, nx, ny);
    let half = 0.5 * H / ny as f64;
    let local = (0..nx)
        .map(|i| {
            assert_eq!(
                c.wall_flux[i], -q_in,
                "prescribed wall flux is the face flux"
            );
            let wall = c.near_wall[i] + q_in * half / ALPHA;
            q_in * D_H / (ALPHA * (wall - c.bulk[i]))
        })
        .collect::<Vec<_>>();
    let nu = mean_over(&c.x, &local, 4.0..=5.0);
    println!("uniform-flux Nu, LLF upwind, 120x64: {nu:.5}");
    assert!((nu - NU_H).abs() / NU_H <= 0.02, "Nu_H {nu}");
    assert!(operator.conservation_receipt(0.0, &state).unwrap().balanced);
}

#[test]
fn transient_bdf_relaxes_to_the_steady_graetz_state() {
    // Bar (fixed before running): max |T(t=400) − T_steady| ≤ 1e-6. Basis: the slowest decay
    // (wash-out near the walls, u ≈ 0.35 at the first cell, and cross-stream diffusion
    // π²α/H² ≈ 0.2) is ≥ 0.05/unit time, so the transient is below e^{-20} ≈ 2e-9 by t = 400.
    let (nx, ny) = (30, 8);
    let (system, program) = law(
        "Graetz2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
    );
    let topology = channel(nx, ny);
    let operator = FvRealization::with_face_fields(
        &system,
        program,
        topology.clone(),
        &regions(&system),
        &data(&[("t_in", 1.0), ("t_wall", 0.0)]),
        &velocity(),
    )
    .unwrap()
    .with_regions(fluid_only(&topology))
    .unwrap();
    let target = dense_steady(&operator);
    let config = BdfConfig {
        order: BdfOrder::One,
        absolute_tolerance: 1e-3,
        relative_tolerance: 1e-3,
        maximum_step: 4.0,
        ..BdfConfig::default()
    };
    let mut state = BdfState::initialize(&operator, &cx(), 0.0, vec![0.0; nx * ny]).unwrap();
    let mut step: f64 = 0.05;
    let mut steps = 0;
    while state.time < 400.0 {
        steps += 1;
        assert!(steps < 5_000, "BDF did not reach t = 400");
        let size = step.min(400.0 - state.time);
        match bdf_step(&operator, &cx(), &state, size, &config).unwrap() {
            StepOutcome::Accepted(accepted) => {
                step = accepted.suggested_step.min(4.0);
                state = accepted.state;
                let balance = operator
                    .conservation_receipt(state.time, &state.values)
                    .unwrap();
                assert!(balance.balanced, "{balance:?}");
            }
            StepOutcome::Rejected(rejected) => step = rejected.suggested_step,
        }
    }
    let error = state
        .values
        .iter()
        .zip(&target)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("BDF1 to t=400 in {steps} attempts: max |T - T_steady| = {error:e}");
    assert!(error <= 1e-6, "{error}");
}

fn slab_regions(topology: &FvTopology, split: f64, solid_stagnant: bool) -> FvRegions {
    let centroids = topology.cell_centroids();
    let dimension = topology.dimension();
    FvRegions {
        cell_regions: (0..topology.cell_count())
            .map(|cell| {
                RegionTagId::new(if centroids[cell * dimension] < split {
                    "a"
                } else {
                    "b"
                })
            })
            .collect(),
        properties: BTreeMap::from([
            (
                RegionTagId::new("a"),
                FvRegionProperties {
                    conductivity: 1.0,
                    capacity: 1.0,
                    stagnant: false,
                },
            ),
            (
                RegionTagId::new("b"),
                FvRegionProperties {
                    conductivity: 4.0,
                    capacity: 3.0,
                    stagnant: solid_stagnant,
                },
            ),
        ]),
    }
}

/// The two-material conduction case: `k = 1` on [0, 0.5] at 400 K, `k = 4` on [0.5, 1] at
/// 300 K: `q = 100/(0.5/1 + 0.5/4) = 160 W/m²`, interface `400 − 160·0.5 = 320 K`.
fn two_material(x: f64) -> f64 {
    if x <= 0.5 {
        400.0 - 160.0 * x
    } else {
        320.0 - 40.0 * (x - 0.5)
    }
}

#[test]
fn two_slab_conduction_is_exact_with_harmonic_face_conductivity() {
    // Bar (fixed before running): 1e-9 K and 1e-9 W/m². Basis: TPFA with the series
    // (harmonic) half-cell conductance is exact for a piecewise-linear profile whose kink lies
    // on a face; the dense solve of ≤ 40 unknowns at 400 K with condition ≲ 1e3 leaves
    // ≲ 1e3·400·ε ≈ 1e-10.
    let (system, program) = law(
        "Slab1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
    );
    let nodes = |graded: bool| {
        let left = (0..=8).map(|i| {
            let s = i as f64 / 8.0;
            0.5 * if graded { s * s } else { s }
        });
        let right = (1..=12).map(|i| {
            let s = i as f64 / 12.0;
            0.5 + 0.5 * if graded { s.sqrt() } else { s }
        });
        left.chain(right).collect::<Vec<_>>()
    };
    for graded in [false, true] {
        let topology = Arc::new(
            FvTopology::from_tensor_box(&TensorBox {
                axes: vec![TensorAxis::graded(nodes(graded))],
            })
            .unwrap(),
        );
        let operator = FvRealization::new(
            &system,
            program.clone(),
            topology.clone(),
            &regions(&system),
            &data(&[("ux", 0.0), ("t_left", 400.0), ("t_right", 300.0)]),
        )
        .unwrap()
        .with_regions(slab_regions(&topology, 0.5, true))
        .unwrap();
        let state = dense_steady(&operator);
        for (cell, value) in state.iter().enumerate() {
            let x = topology.cell_centroids()[cell];
            assert!(
                (value - two_material(x)).abs() <= 1e-9,
                "graded {graded}: {x} {value}"
            );
        }
        let interface = operator.region_interface_fluxes(0.0, &state).unwrap();
        assert_eq!(interface.interfaces.len(), 1);
        let flow = &interface.interfaces[0];
        assert_eq!(
            (flow.from.as_str(), flow.to.as_str(), flow.faces),
            ("a", "b", 1)
        );
        assert!((flow.flow - 160.0).abs() <= 1e-9, "{flow:?}");
        let fluxes = operator.face_fluxes(0.0, &state).unwrap();
        assert!((fluxes.boundary[&RegionTagId::new("x_min")][0] + 160.0).abs() <= 1e-9);
        assert!((fluxes.boundary[&RegionTagId::new("x_max")][0] - 160.0).abs() <= 1e-9);
        let receipt = operator.receipt();
        let regions = receipt.regions.as_ref().unwrap();
        assert_eq!(regions.scheme, "two_point_harmonic");
        assert_eq!(regions.interface_faces, 1);
        assert_eq!(regions.regions[&RegionTagId::new("a")].cells, 8);
        assert!(receipt.diffusion.is_none());
    }

    // Capacity enters the rate: one BDF1 step from a uniform 350 K obeys
    // Σ C_i|Ω_i|(T¹ − T⁰)/Δt = −Σ_b |f|F_b(T¹). Bar 1e-9 relative (linear problem, one exact
    // dense Newton step).
    let topology = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![TensorAxis::uniform(0.0, 1.0, 20)],
        })
        .unwrap(),
    );
    let operator = FvRealization::new(
        &system,
        program.clone(),
        topology.clone(),
        &regions(&system),
        &data(&[("ux", 0.0), ("t_left", 400.0), ("t_right", 300.0)]),
    )
    .unwrap()
    .with_regions(slab_regions(&topology, 0.5, true))
    .unwrap();
    let initial = vec![350.0; 20];
    let state = BdfState::initialize(&operator, &cx(), 0.0, initial.clone()).unwrap();
    let config = BdfConfig {
        order: BdfOrder::One,
        ..BdfConfig::default()
    };
    let dt = 1e-3;
    let StepOutcome::Accepted(step) = bdf_step(&operator, &cx(), &state, dt, &config).unwrap()
    else {
        panic!("BDF1 step rejected");
    };
    let capacity = |cell: usize| if cell < 10 { 1.0 } else { 3.0 };
    let storage = (0..20)
        .map(|cell| {
            capacity(cell)
                * topology.cell_volumes()[cell]
                * (step.state.values[cell] - initial[cell])
                / dt
        })
        .sum::<f64>();
    let fluxes = operator.face_fluxes(dt, &step.state.values).unwrap();
    let outflow = fluxes.boundary.values().map(|v| v[0]).sum::<f64>();
    assert!(
        (storage + outflow).abs() <= 1e-9 * outflow.abs().max(storage.abs()),
        "storage {storage} outflow {outflow}"
    );
    use methodus::ExplicitOperator;
    assert_eq!(
        operator.linear_invariants()[0],
        (0..20)
            .map(|cell| capacity(cell) * topology.cell_volumes()[cell])
            .collect::<Vec<_>>()
    );
}

#[test]
fn zero_velocity_limit_reproduces_two_material_conduction_in_2d() {
    // Bar (fixed before running): 1e-9 K / W/m² (same basis as the 1-D slab).
    let (system, program) = law(
        "Conjugate2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
    );
    let topology = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![
                TensorAxis::uniform(0.0, 1.0, 8),
                TensorAxis::uniform(0.0, 1.0, 3),
            ],
        })
        .unwrap(),
    );
    let fields = |ux: f64| {
        BTreeMap::from([
            ("ux".to_string(), FieldSource::constant([ux])),
            ("uy".to_string(), FieldSource::constant([0.0])),
        ])
    };
    let bind = |ux: f64| {
        FvRealization::with_face_fields(
            &system,
            program.clone(),
            topology.clone(),
            &regions(&system),
            &data(&[("t_west", 400.0), ("t_east", 300.0)]),
            &fields(ux),
        )
        .unwrap()
        .with_regions(slab_regions(&topology, 0.5, true))
    };
    let operator = bind(0.0).unwrap();
    let state = dense_steady(&operator);
    for (cell, value) in state.iter().enumerate() {
        let x = topology.cell_centroids()[2 * cell];
        assert!((value - two_material(x)).abs() <= 1e-9, "{x} {value}");
    }
    let interface = operator.region_interface_fluxes(0.0, &state).unwrap();
    let flow = &interface.interfaces[0];
    assert_eq!(flow.faces, 3);
    assert!((flow.area - 1.0).abs() <= 1e-15);
    assert!((flow.flow / flow.area - 160.0).abs() <= 1e-9, "{flow:?}");
    let receipt = operator.receipt();
    assert_eq!(receipt.face_fields.len(), 2);
    assert!(
        receipt
            .face_fields
            .iter()
            .all(|field| field.faces == 8 * 3 * 2 + 8 + 3)
    );

    // A velocity that does not vanish next to the stagnant region is refused.
    let error = bind(1.0).unwrap_err();
    assert_eq!(code(&error), "FV_REGION_NOT_STAGNANT");
}

#[test]
fn conjugate_graetz_conserves_energy_across_the_fluid_wall_interface() {
    // Single-mesh conjugate Graetz (gate 1b): fluid channel 0 ≤ y ≤ 1 (k = α, C = 1) between
    // stagnant walls of thickness 0.25 (k = 5α, C = 2), isothermal outer wall surfaces.
    // Bars (fixed before running): the heat entering both walls through the region interface
    // equals the heat leaving the wall cells through the boundary to 1e-8 of the boundary flow
    // scale (Newton-GMRES relative tolerance 1e-11); the region JVP satisfies the VJP transpose
    // identity to 64ε·scale (exact linear transpose).
    let (nx, ny_fluid, ny_wall) = (60, 16, 4);
    let mut y = (0..=ny_wall)
        .map(|j| -0.25 + 0.25 * j as f64 / ny_wall as f64)
        .collect::<Vec<_>>();
    y.extend((1..=ny_fluid).map(|j| j as f64 / ny_fluid as f64));
    y.extend((1..=ny_wall).map(|j| 1.0 + 0.25 * j as f64 / ny_wall as f64));
    let ny = y.len() - 1;
    let topology = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![TensorAxis::uniform(0.0, LENGTH, nx), TensorAxis::graded(y)],
        })
        .unwrap(),
    );
    let (system, program) = law(
        "Graetz2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
    );
    let centroids = topology.cell_centroids().to_vec();
    let solid = |cell: usize| !(0.0..=H).contains(&centroids[2 * cell + 1]);
    let regions_spec = FvRegions {
        cell_regions: (0..topology.cell_count())
            .map(|cell| RegionTagId::new(if solid(cell) { "wall" } else { "fluid" }))
            .collect(),
        properties: BTreeMap::from([
            (
                RegionTagId::new("fluid"),
                FvRegionProperties {
                    conductivity: ALPHA,
                    capacity: 1.0,
                    stagnant: false,
                },
            ),
            (
                RegionTagId::new("wall"),
                FvRegionProperties {
                    conductivity: 5.0 * ALPHA,
                    capacity: 2.0,
                    stagnant: true,
                },
            ),
        ]),
    };
    let operator = FvRealization::with_face_fields(
        &system,
        program,
        topology.clone(),
        &regions(&system),
        &data(&[("t_in", 1.0), ("t_wall", 0.0)]),
        &velocity(),
    )
    .unwrap()
    .with_regions(regions_spec)
    .unwrap();
    let state = steady(&operator, 0.5);
    assert!(operator.conservation_receipt(0.0, &state).unwrap().balanced);
    let interface = operator.region_interface_fluxes(0.0, &state).unwrap();
    assert_eq!(interface.interfaces.len(), 1);
    let flow = &interface.interfaces[0];
    assert_eq!((flow.from.as_str(), flow.to.as_str()), ("fluid", "wall"));
    assert_eq!(flow.faces, 2 * nx);
    let fluxes = operator.face_fluxes(0.0, &state).unwrap();
    let (mut wall_out, mut scale) = (0.0, 0.0);
    for (tag, block) in topology.boundary() {
        for (face, value) in fluxes.boundary[tag].iter().enumerate() {
            let term = block.area[face] * value;
            scale += term.abs();
            if solid(block.minus[face]) {
                wall_out += term;
            }
        }
    }
    println!(
        "conjugate Graetz: fluid->wall {:.9}, wall boundary outflow {wall_out:.9}",
        flow.flow
    );
    assert!(flow.flow > 0.0);
    assert!(
        (flow.flow - wall_out).abs() <= 1e-8 * scale,
        "{} vs {wall_out}",
        flow.flow
    );
    // The wall-adjacent fluid cells sit between the interface and the cold outer surface.
    assert!(
        state
            .iter()
            .all(|value| (-1e-9..=1.0 + 1e-9).contains(value))
    );

    let n = state.len();
    let probe = |seed: f64| {
        (0..n)
            .map(|i| ((i as f64 + seed) * 0.618_034).sin())
            .collect::<Vec<_>>()
    };
    let (v, w) = (probe(0.3), probe(1.7));
    let jv = jvp(&operator, &state, &v);
    let mut jtw = vec![0.0; n];
    operator.state_vjp(0.0, &state, &w, &mut jtw).unwrap();
    let left = w.iter().zip(&jv).map(|(a, b)| a * b).sum::<f64>();
    let right = jtw.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
    let scale = w.iter().zip(&jv).map(|(a, b)| (a * b).abs()).sum::<f64>();
    assert!(
        (left - right).abs() <= 64.0 * f64::EPSILON * scale,
        "{left} {right}"
    );
    let _ = ny;
}

#[test]
fn face_field_and_region_refusals_are_typed() {
    let (system, program) = law(
        "Conjugate2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
    );
    let topology = channel(4, 2);
    let base = data(&[("t_west", 1.0), ("t_east", 0.0)]);
    let bind = |fields: BTreeMap<String, FieldSource>, data: &BTreeMap<String, f64>| {
        FvRealization::with_face_fields(
            &system,
            program.clone(),
            topology.clone(),
            &regions(&system),
            data,
            &fields,
        )
    };
    let zero = || FieldSource::constant([0.0]);
    let two = |a: FieldSource, b: FieldSource| {
        BTreeMap::from([("ux".to_string(), a), ("uy".to_string(), b)])
    };
    let checks: Vec<(&str, Result<FvRealization, FinitumError>)> = vec![
        ("FV_FACE_FIELD_INVALID", bind(BTreeMap::new(), &base)),
        (
            "FV_DATUM_MISSING",
            bind(BTreeMap::from([("ux".to_string(), zero())]), &base),
        ),
        (
            "FV_DATUM_UNKNOWN",
            bind(
                BTreeMap::from([
                    ("ux".to_string(), zero()),
                    ("uy".to_string(), zero()),
                    ("nope".to_string(), zero()),
                ]),
                &base,
            ),
        ),
        (
            "FV_FACE_FIELD_INVALID",
            bind(two(zero(), FieldSource::constant([0.0, 1.0])), &base),
        ),
        (
            "FV_FACE_FIELD_INVALID",
            bind(two(zero(), FieldSource::nodal(vec![0.0; 3])), &base),
        ),
        (
            "FV_FACE_FIELD_INVALID",
            bind(
                two(
                    zero(),
                    FieldSource::fallible(|_: &[f64], _| Ok(vec![f64::NAN])),
                ),
                &base,
            ),
        ),
        (
            "FV_FACE_FIELD_INVALID",
            bind(two(zero(), zero()), &{
                let mut both = base.clone();
                both.insert("ux".into(), 0.0);
                both
            }),
        ),
    ];
    for (expected, result) in checks {
        assert_eq!(code(&result.unwrap_err()), expected);
    }
    let operator = bind(two(zero(), zero()), &base).unwrap();
    let mut properties = fluid_only(&topology);
    properties.cell_regions.pop();
    assert_eq!(
        code(
            &bind(two(zero(), zero()), &base)
                .unwrap()
                .with_regions(properties)
                .unwrap_err()
        ),
        "FV_REGION_SHAPE"
    );
    let mut unknown = fluid_only(&topology);
    unknown.cell_regions[0] = RegionTagId::new("ghost");
    assert_eq!(
        code(
            &bind(two(zero(), zero()), &base)
                .unwrap()
                .with_regions(unknown)
                .unwrap_err()
        ),
        "FV_REGION_UNKNOWN"
    );
    for (conductivity, capacity) in [(-1.0, 1.0), (1.0, 0.0), (f64::NAN, 1.0)] {
        let mut bad = fluid_only(&topology);
        bad.properties.values_mut().for_each(|p| {
            p.conductivity = conductivity;
            p.capacity = capacity;
        });
        assert_eq!(
            code(
                &bind(two(zero(), zero()), &base)
                    .unwrap()
                    .with_regions(bad)
                    .unwrap_err()
            ),
            "FV_REGION_PROPERTY_INVALID"
        );
    }
    let twice = operator.with_regions(fluid_only(&topology)).unwrap();
    assert_eq!(
        code(&twice.with_regions(fluid_only(&topology)).unwrap_err()),
        "FV_DIFFUSION_INVALID"
    );
    // Uniform nonzero velocity next to a stagnant region.
    let uniform = FvRealization::new(
        &system,
        program.clone(),
        topology.clone(),
        &regions(&system),
        &data(&[("ux", 1.0), ("uy", 0.0), ("t_west", 1.0), ("t_east", 0.0)]),
    )
    .unwrap();
    let mut stagnant = fluid_only(&topology);
    stagnant
        .properties
        .values_mut()
        .for_each(|p| p.stagnant = true);
    assert_eq!(
        code(&uniform.with_regions(stagnant).unwrap_err()),
        "FV_REGION_NOT_STAGNANT"
    );
}
