//! W13 A7 (SV2C L8c): compressible Euler on the multi-component FV realization — Sod star
//! region against Toro, conservation per component, Zhang–Shu positivity on Toro test 2, the
//! isentropic vortex order ladder, authored HLLC/Chandrashekar fluxes, slip walls, D6 refusals.
//!
//! Every acceptance bar below is fixed before the run, with its basis next to it.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use finitum::{
    FV_NONSMOOTH_DERIVATIVE_REFUSED, FvDerivativeDisposition, FvSystemRealization, FvTopology,
    RegionMap, RegionTagId, TensorAxis, TensorBox,
};
use methodus::{
    Admissibility, DaeOperator, EvaluationContext, ExplicitConfig, ExplicitIntegrationReport,
    ExplicitOperator, ExplicitScheme, ExplicitState, NumericError, StepPolicy, integrate_explicit,
};
use quantitas::UnitRegistry;
use scientia::{
    ConservationEquationSpec, ConservationSystemSpec, FV_STATE_INADMISSIBLE, FvMethodIntent,
    FvSystemSample, GradientScheme, LimiterFamily, MultiComponentConservationLaw,
    MultiComponentProgram, NumericalFluxIntent, PositivityIntent, ReconstructionIntent,
    ReconstructionVariables, SemanticModule, SignalSpeedSpec, SourceTreatment,
    compile_multi_component_law, compile_semantics, lower_multi_component_law,
};

const EULER: &str = include_str!("fixtures/fv-euler.res");
const HLLC: &str = include_str!("../../scientia/tests/fixtures/euler-hllc.res");
const CHANDRASHEKAR: &str = include_str!("../../scientia/tests/fixtures/euler-chandrashekar.res");
const GAMMA: f64 = 1.4;
/// Scientia's authored-flux claim tolerance for these fixtures (its sv2c1b basis).
const AUTHORED_TOLERANCE: f64 = 1e-10;
/// Zhang–Shu thresholds `ε` for density and pressure.
const EPSILON: f64 = 1e-10;
/// Toro (2009) Table 4.2, test 1 (Sod): star pressure and velocity.
const P_STAR: f64 = 0.30313;
const U_STAR: f64 = 0.92745;

fn cx() -> EvaluationContext {
    EvaluationContext::default()
}

fn module(source: &str) -> SemanticModule {
    compile_semantics(source, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("{diagnostics:#?}"))
        .semantic
}

fn law(module: &SemanticModule, model: &str, admissible: &[&str]) -> MultiComponentConservationLaw {
    let spec = ConservationSystemSpec {
        equations: [
            ("mass", "density"),
            ("momentum_balance", "momentum"),
            ("energy", "total_energy"),
        ]
        .map(|(equation, conserved)| ConservationEquationSpec {
            equation: equation.into(),
            conserved: conserved.into(),
        })
        .to_vec(),
        signal_speeds: SignalSpeedSpec {
            min: "speed_min".into(),
            max: "speed_max".into(),
        },
        admissible: admissible.iter().map(|name| (*name).into()).collect(),
    };
    compile_multi_component_law(module, model, &spec).unwrap()
}

fn muscl(limiter: LimiterFamily) -> ReconstructionIntent {
    ReconstructionIntent::Muscl {
        variables: ReconstructionVariables::Conserved,
        gradient: GradientScheme::LeastSquares,
        limiter,
    }
}

fn zhang_shu() -> PositivityIntent {
    PositivityIntent::ZhangShuScaling {
        thresholds: vec![EPSILON, EPSILON],
    }
}

/// Conserved state from primitive `(ρ, velocity, p)`.
fn conserved(rho: f64, velocity: &[f64], p: f64) -> Vec<f64> {
    let kinetic = velocity.iter().map(|v| v * v).sum::<f64>();
    let mut state = vec![rho];
    state.extend(velocity.iter().map(|v| rho * v));
    state.push(p / (GAMMA - 1.0) + 0.5 * rho * kinetic);
    state
}

fn pressure(u: &[f64]) -> f64 {
    let d = u.len() - 2;
    let kinetic = (0..d).map(|i| u[1 + i] * u[1 + i]).sum::<f64>();
    (GAMMA - 1.0) * (u[d + 1] - 0.5 * kinetic / u[0])
}

/// Deterministic admissible samples (as Scientia's sv2c1b tests draw them).
fn samples(dimension: usize, count: usize) -> Vec<FvSystemSample> {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let data = BTreeMap::from([("gamma".to_owned(), GAMMA)]);
    let mut out = Vec::new();
    while out.len() < count {
        let mut state = || {
            let rho = 0.2 + 1.8 * next();
            let velocity = (0..dimension)
                .map(|_| -1.5 + 3.0 * next())
                .collect::<Vec<_>>();
            let p = 0.1 + 1.9 * next();
            conserved(rho, &velocity, p)
        };
        let (minus, plus) = (state(), state());
        let angle = std::f64::consts::TAU * next();
        out.push(FvSystemSample {
            minus,
            plus,
            normal: if dimension == 1 {
                vec![if angle.cos() >= 0.0 { 1.0 } else { -1.0 }]
            } else {
                vec![angle.cos(), angle.sin()]
            },
            data: data.clone(),
        });
    }
    out
}

struct Case {
    law: MultiComponentConservationLaw,
    program: Arc<MultiComponentProgram>,
}

fn case(
    source: &str,
    model: &str,
    admissible: &[&str],
    flux: NumericalFluxIntent,
    reconstruction: ReconstructionIntent,
    positivity: PositivityIntent,
) -> Case {
    let module = module(source);
    let law = law(&module, model, admissible);
    let intent = FvMethodIntent {
        numerical_flux: flux,
        reconstruction,
        source: SourceTreatment::Pointwise,
        positivity,
    };
    let program = lower_multi_component_law(
        &module,
        &law,
        &intent,
        &samples(law.spatial_dimension.into(), 24),
    )
    .unwrap();
    Case {
        law,
        program: Arc::new(program),
    }
}

fn authored(name: &str) -> NumericalFluxIntent {
    NumericalFluxIntent::Authored {
        declaration: name.into(),
        tolerance: AUTHORED_TOLERANCE,
    }
}

fn line(min: f64, max: f64, cells: usize, periodic: bool) -> Arc<FvTopology> {
    let axis = TensorAxis::uniform(min, max, cells);
    Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![if periodic { axis.periodic() } else { axis }],
        })
        .unwrap(),
    )
}

fn square(axes: [(f64, f64, usize); 2], periodic: bool) -> Arc<FvTopology> {
    Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: axes
                .map(|(min, max, cells)| {
                    let axis = TensorAxis::uniform(min, max, cells);
                    if periodic { axis.periodic() } else { axis }
                })
                .to_vec(),
        })
        .unwrap(),
    )
}

/// `left`/`right` onto `x_min`/`x_max`; `wall` onto every box tag.
fn regions(law: &MultiComponentConservationLaw) -> RegionMap {
    let mut map = RegionMap::new();
    for boundary in &law.boundaries {
        let tags: &[&str] = match boundary.region_name.as_str() {
            "left" => &["x_min"],
            "right" => &["x_max"],
            "wall" if law.spatial_dimension == 1 => &["x_min", "x_max"],
            "wall" => &["x_min", "x_max", "y_min", "y_max"],
            other => panic!("unexpected boundary {other}"),
        };
        map.insert(
            boundary.region,
            tags.iter().map(|tag| RegionTagId::new(*tag)),
        );
    }
    map
}

fn gamma() -> BTreeMap<String, f64> {
    BTreeMap::from([("gamma".to_owned(), GAMMA)])
}

fn realize(case: &Case, topology: &Arc<FvTopology>) -> FvSystemRealization {
    FvSystemRealization::new(
        &case.law,
        case.program.clone(),
        topology.clone(),
        &regions(&case.law),
        &gamma(),
    )
    .unwrap()
}

/// Component-major state from per-cell conserved states.
fn layout(cells: &[Vec<f64>]) -> Vec<f64> {
    let m = cells[0].len();
    (0..m)
        .flat_map(|k| cells.iter().map(move |cell| cell[k]))
        .collect()
}

fn cell_state(state: &[f64], cells: usize, m: usize, cell: usize) -> Vec<f64> {
    (0..m).map(|k| state[k * cells + cell]).collect()
}

fn cfl_config(scheme: ExplicitScheme) -> ExplicitConfig {
    ExplicitConfig {
        scheme,
        step_policy: StepPolicy::Cfl {
            target: 0.9,
            max: 1.0,
            min_step: 1e-12,
            max_step: 1.0,
        },
    }
}

fn integrate(
    operator: &(impl ExplicitOperator + ?Sized),
    initial: Vec<f64>,
    final_time: f64,
    scheme: ExplicitScheme,
) -> Result<ExplicitIntegrationReport, methodus::SolveError> {
    let state = ExplicitState::initialize(operator, &cx(), 0.0, initial)?;
    integrate_explicit(
        operator,
        &cx(),
        &state,
        final_time,
        &cfl_config(scheme),
        100_000,
    )
}

/// Every step's invariant balance within the established FV bound `(cells + 4·stages)·ε·scale`
/// (Methodus balance residual of a scatter that telescopes exactly; SV2-C2a basis). Returns the
/// accumulated per-invariant bound `Σ_steps (cells + 4·stages)·ε·scale`.
fn check_balances(
    report: &ExplicitIntegrationReport,
    cells: usize,
    scheme: ExplicitScheme,
) -> Vec<f64> {
    let k = (cells + 4 * scheme.stages()) as f64 * f64::EPSILON;
    let mut total = vec![0.0; report.steps[0].invariants.len()];
    for step in &report.steps {
        assert!(step.balance_corrected);
        for (index, balance) in step.invariants.iter().enumerate() {
            assert!(
                balance.balance_residual.abs() <= k * balance.scale,
                "invariant {index}: {balance:?}"
            );
            total[index] += k * balance.scale;
        }
    }
    total
}

/// A Riemann problem started from rest raises the wave speed within the first step, so the
/// integrator may retry an early step on its CFL check (`CflExceeded`, by design; at most two). Recorded after the
/// first run showed one such retry: no other rejection (in particular no inadmissible stage) is
/// accepted.
fn only_cfl_retries(report: &ExplicitIntegrationReport) {
    for rejected in &report.rejections {
        assert!(
            matches!(
                rejected.rejection,
                methodus::ExplicitRejection::CflExceeded { .. }
            ),
            "{rejected:?}"
        );
    }
    assert!(report.rejections.len() <= 2, "{:?}", report.rejections);
    eprintln!("CFL retries {}", report.rejections.len());
}

fn totals(state: &[f64], volumes: &[f64]) -> Vec<f64> {
    let cells = volumes.len();
    (0..state.len() / cells)
        .map(|k| {
            state[k * cells..(k + 1) * cells]
                .iter()
                .zip(volumes)
                .map(|(u, v)| u * v)
                .sum()
        })
        .collect()
}

/// Max relative star-region error of `p` and `u` over cells with centroid in `[0.55, 0.80]`.
fn star_errors(
    state: &[f64],
    centroids: &[f64],
    stride: usize,
    cells: usize,
    m: usize,
) -> (f64, f64) {
    let (mut p_error, mut u_error) = (0.0_f64, 0.0_f64);
    let mut seen = 0;
    for cell in 0..cells {
        let x = centroids[cell * stride];
        if !(0.55..=0.80).contains(&x) {
            continue;
        }
        let u = cell_state(state, cells, m, cell);
        p_error = p_error.max((pressure(&u) - P_STAR).abs() / P_STAR);
        u_error = u_error.max((u[1] / u[0] - U_STAR).abs() / U_STAR);
        seen += 1;
    }
    assert!(seen > 10);
    (p_error, u_error)
}

/// Sod (Toro test 1) on [0, 1], 400 cells, transmissive ends, LLF, SSP-RK3 at CFL 0.9 to
/// t = 0.2. Bars, fixed before running, for the star window [0.55, 0.80]: at t = 0.2 the
/// rarefaction tail is at x = 0.486, the contact at 0.686 and the shock at 0.850 (exact
/// solution), so the window is ≥ 0.05 from the shock and 0.064 (≈ 3 LLF smearing widths
/// √(ν t), ν ≈ λΔx/2) from the tail; p and u are continuous across the contact. First order:
/// relative error ≤ 2e-2; MUSCL (minmod): ≤ 1e-2 and no larger than first order. Toro's
/// values carry 5 digits (≤ 2e-5 relative), far below both bars. Mass and energy have exactly
/// zero boundary flux while the waves stay inside, so their drift is bounded by the summed
/// per-step balance bound; momentum gains exactly (p_L − p_R)·t = 0.18.
#[test]
fn sod_star_region_matches_toro_and_conserves_each_component() {
    let cells = 400;
    let topology = line(0.0, 1.0, cells, false);
    let initial = layout(
        &(0..cells)
            .map(|i| {
                if (i as f64 + 0.5) / (cells as f64) < 0.5 {
                    conserved(1.0, &[0.0], 1.0)
                } else {
                    conserved(0.125, &[0.0], 0.1)
                }
            })
            .collect::<Vec<_>>(),
    );
    let scheme = ExplicitScheme::SspRk3;
    let mut first_order = None;
    for (reconstruction, bar) in [
        (ReconstructionIntent::Constant, 2e-2),
        (muscl(LimiterFamily::Minmod), 1e-2),
    ] {
        let case = case(
            EULER,
            "Euler1D",
            &["mass_density", "pressure"],
            NumericalFluxIntent::LocalLaxFriedrichs,
            reconstruction.clone(),
            PositivityIntent::None,
        );
        let operator = realize(&case, &topology);
        let report = integrate(&operator, initial.clone(), 0.2, scheme).unwrap();
        only_cfl_retries(&report);
        let end = &report.state.values;
        let (p_error, u_error) = star_errors(end, topology.cell_centroids(), 1, cells, 3);
        eprintln!(
            "Sod {reconstruction:?}: steps {}, max rel error p {p_error:.3e}, u {u_error:.3e}",
            report.steps.len()
        );
        assert!(p_error <= bar && u_error <= bar, "{p_error} {u_error}");
        match first_order {
            None => first_order = Some(p_error.max(u_error)),
            Some(previous) => assert!(p_error.max(u_error) <= previous),
        }
        let bound = check_balances(&report, cells, scheme);
        let (start, finish) = (
            totals(&initial, topology.cell_volumes()),
            totals(end, topology.cell_volumes()),
        );
        assert!((finish[0] - start[0]).abs() <= bound[0], "mass");
        assert!((finish[2] - start[2]).abs() <= bound[2], "energy");
        let steps = report.steps.len() as f64;
        assert!(
            (finish[1] - 0.18).abs() <= bound[1] + steps * f64::EPSILON,
            "momentum {}",
            finish[1]
        );
        for receipt in operator.conservation_receipts(0.2, end).unwrap() {
            assert!(receipt.balanced, "{receipt:?}");
        }
        let receipt = operator.receipt();
        assert_eq!(receipt.components, 3);
        assert_eq!(receipt.layout, "component_major");
    }
}

/// An explicit operator that records the smallest density and pressure of every state the
/// integrator asks about (every stage state of every attempted step).
struct Watch<'a> {
    inner: &'a FvSystemRealization,
    minimum: Mutex<(f64, f64)>,
}

impl ExplicitOperator for Watch<'_> {
    fn dimension(&self) -> usize {
        ExplicitOperator::dimension(self.inner)
    }
    fn rhs(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.inner.rhs(context, time, state, output)
    }
    fn stable_step(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        self.inner.stable_step(context, time, state)
    }
    fn admissibility(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Admissibility, NumericError> {
        let cells = state.len() / 3;
        let mut minimum = self.minimum.lock().unwrap();
        for cell in 0..cells {
            let u = cell_state(state, cells, 3, cell);
            minimum.0 = minimum.0.min(u[0]);
            minimum.1 = minimum.1.min(pressure(&u));
        }
        ExplicitOperator::admissibility(self.inner, context, time, state)
    }
    fn linear_invariants(&self) -> &[Vec<f64>] {
        self.inner.linear_invariants()
    }
    fn invariant_rates(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<bool, NumericError> {
        self.inner.invariant_rates(context, time, state, output)
    }
}

/// Toro test 2 (ρ, u, p = 1, −2, 0.4 | 1, 2, 0.4) on [0, 1], 200 cells, LLF, t = 0.15. Without
/// positivity scaling the unlimited MUSCL face state left of the jump has p < 0 at t = 0
/// (E = 3, m_f = −3 ⇒ p = 0.4(3 − 4.5)), which is refused typed `FV_STATE_INADMISSIBLE`. With
/// Zhang–Shu (unlimited and minmod) the run must finish with no rejected step and every stage
/// state with ρ, p > ε: basis — the stable step is h/(2λ) per face pair, so CFL ≤ 1 meets the
/// Zhang–Shu MUSCL condition λΔt/h ≤ 1/2, and SSP-RK3 stages are convex combinations of
/// forward-Euler steps.
#[test]
fn toro_test_two_is_refused_without_and_positive_with_zhang_shu() {
    let cells = 200;
    let topology = line(0.0, 1.0, cells, false);
    let initial = layout(
        &(0..cells)
            .map(|i| {
                let velocity = if (i as f64 + 0.5) / (cells as f64) < 0.5 {
                    -2.0
                } else {
                    2.0
                };
                conserved(1.0, &[velocity], 0.4)
            })
            .collect::<Vec<_>>(),
    );

    let unscaled = case(
        EULER,
        "Euler1D",
        &["mass_density", "pressure"],
        NumericalFluxIntent::LocalLaxFriedrichs,
        muscl(LimiterFamily::None),
        PositivityIntent::None,
    );
    let operator = realize(&unscaled, &topology);
    let mut output = vec![0.0; initial.len()];
    let Err(NumericError::Evaluation { code, origin, .. }) =
        operator.rhs(&cx(), 0.0, &initial, &mut output)
    else {
        panic!("unlimited MUSCL face state must be refused");
    };
    assert_eq!(code, FV_STATE_INADMISSIBLE);
    assert_eq!(origin, "fv/face_state");
    let error = integrate(&operator, initial.clone(), 0.15, ExplicitScheme::SspRk3).unwrap_err();
    assert!(
        format!("{error:?}").contains(FV_STATE_INADMISSIBLE),
        "{error:?}"
    );
    // The cell-average hook refuses a negative-pressure average.
    let mut bad = initial.clone();
    bad[2 * cells + 7] = 1.0; // E = 1 < ½ m²/ρ = 2 ⇒ p < 0
    assert!(matches!(
        ExplicitOperator::admissibility(&operator, &cx(), 0.0, &bad).unwrap(),
        Admissibility::Inadmissible { index: Some(7), .. }
    ));

    for limiter in [LimiterFamily::None, LimiterFamily::Minmod] {
        let scaled = case(
            EULER,
            "Euler1D",
            &["mass_density", "pressure"],
            NumericalFluxIntent::LocalLaxFriedrichs,
            muscl(limiter.clone()),
            zhang_shu(),
        );
        let operator = realize(&scaled, &topology);
        assert_eq!(
            operator.receipt().positivity.unwrap().thresholds,
            [EPSILON, EPSILON]
        );
        let watch = Watch {
            inner: &operator,
            minimum: Mutex::new((f64::INFINITY, f64::INFINITY)),
        };
        let report = integrate(&watch, initial.clone(), 0.15, ExplicitScheme::SspRk3).unwrap();
        let (rho, p) = *watch.minimum.lock().unwrap();
        eprintln!(
            "Toro 2 {limiter:?} + Zhang-Shu: steps {}, rejections {}, min rho {rho:.4e}, min p {p:.4e}",
            report.steps.len(),
            report.rejections.len()
        );
        assert!(report.rejections.is_empty(), "{:?}", report.rejections);
        assert!(rho > EPSILON && p > EPSILON);
        check_balances(&report, cells, ExplicitScheme::SspRk3);
    }
}

/// Isentropic vortex (Shu 1998; β = 5, free stream ρ = u = v = p = 1) in the periodic square
/// [0, 10]², exact solution the translated vortex. Cell averages by 3×3 Gauss (O(h⁶)). MUSCL
/// unlimited least squares (central on a uniform tensor mesh, second order), LLF, SSP-RK3 at
/// CFL 0.9 (Δt ∝ h, time error O(h³)), t = 1, ladder 32/64/128. Bar (brief, fixed): L1
/// density error order ≥ 1.8 between the two finest levels; every level conserves each
/// component per step within the FV balance bound.
#[test]
fn isentropic_vortex_muscl_unlimited_is_second_order() {
    let case = case(
        EULER,
        "Euler2D",
        &[],
        NumericalFluxIntent::LocalLaxFriedrichs,
        muscl(LimiterFamily::None),
        PositivityIntent::None,
    );
    let beta = 5.0;
    let exact = |x: f64, y: f64, t: f64| {
        let wrap = |d: f64| d - 10.0 * (d / 10.0).round();
        let (dx, dy) = (wrap(x - 5.0 - t), wrap(y - 5.0 - t));
        let r2 = dx * dx + dy * dy;
        let f = beta / (2.0 * std::f64::consts::PI) * ((1.0 - r2) / 2.0).exp();
        let (u, v) = (1.0 - f * dy, 1.0 + f * dx);
        let temperature = 1.0
            - (GAMMA - 1.0) * beta * beta
                / (8.0 * GAMMA * std::f64::consts::PI * std::f64::consts::PI)
                * (1.0 - r2).exp();
        let rho = temperature.powf(1.0 / (GAMMA - 1.0));
        conserved(rho, &[u, v], rho * temperature)
    };
    let gauss = [
        (-(0.6_f64.sqrt()), 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (0.6_f64.sqrt(), 5.0 / 9.0),
    ];
    let averages = |n: usize, t: f64| {
        let h = 10.0 / n as f64;
        let cells = (0..n * n)
            .map(|cell| {
                let (i, j) = (cell % n, cell / n);
                let mut sum = [0.0; 4];
                for (a, wa) in gauss {
                    for (b, wb) in gauss {
                        let x = (i as f64 + 0.5 + 0.5 * a) * h;
                        let y = (j as f64 + 0.5 + 0.5 * b) * h;
                        for (slot, value) in sum.iter_mut().zip(exact(x, y, t)) {
                            *slot += 0.25 * wa * wb * value;
                        }
                    }
                }
                sum.to_vec()
            })
            .collect::<Vec<_>>();
        layout(&cells)
    };
    let mut errors = Vec::new();
    for n in [32, 64, 128] {
        let topology = square([(0.0, 10.0, n), (0.0, 10.0, n)], true);
        let operator = realize(&case, &topology);
        let report = integrate(&operator, averages(n, 0.0), 1.0, ExplicitScheme::SspRk3).unwrap();
        check_balances(&report, n * n, ExplicitScheme::SspRk3);
        let reference = averages(n, 1.0);
        let h2 = (10.0 / n as f64).powi(2);
        let error = (0..n * n)
            .map(|cell| (report.state.values[cell] - reference[cell]).abs() * h2)
            .sum::<f64>()
            / 100.0;
        eprintln!(
            "vortex n = {n}: steps {}, L1(rho) = {error:.4e}",
            report.steps.len()
        );
        errors.push(error);
    }
    let orders = errors
        .windows(2)
        .map(|pair| (pair[0] / pair[1]).log2())
        .collect::<Vec<_>>();
    eprintln!("vortex orders {orders:?}");
    assert!(orders[1] >= 1.8, "{orders:?}");
}

/// Authored catalog fluxes. HLLC (MUSCL minmod) on Sod in a periodic 2-D strip [0, 2] × [0, 1]
/// (800 × 2 cells, Δx = 1/400 as above; Riemann problems at 0.5 and, mirrored, 1.5 whose
/// waves stay in [1.15, 1.74] at t = 0.2): star-window bar 1e-2 (the MUSCL bar above), every
/// component conserved per step. Chandrashekar (entropy conservative): a uniform state is a
/// steady state to round-off (bound below).
#[test]
fn authored_hllc_and_chandrashekar_fluxes_realize() {
    let hllc = case(
        HLLC,
        "IdealGasHllc2D",
        &[],
        authored("hllc"),
        muscl(LimiterFamily::Minmod),
        PositivityIntent::None,
    );
    let (nx, ny) = (800, 2);
    let topology = square([(0.0, 2.0, nx), (0.0, 1.0, ny)], true);
    let cells = nx * ny;
    let initial = layout(
        &(0..cells)
            .map(|cell| {
                let x = topology.cell_centroids()[2 * cell];
                if (0.5..1.5).contains(&x) {
                    conserved(0.125, &[0.0, 0.0], 0.1)
                } else {
                    conserved(1.0, &[0.0, 0.0], 1.0)
                }
            })
            .collect::<Vec<_>>(),
    );
    let operator = realize(&hllc, &topology);
    assert_eq!(
        operator.receipt().kernels[0].derivative,
        FvDerivativeDisposition::Refused {
            code: FV_NONSMOOTH_DERIVATIVE_REFUSED.into()
        }
    );
    let report = integrate(&operator, initial, 0.2, ExplicitScheme::SspRk3).unwrap();
    only_cfl_retries(&report);
    check_balances(&report, cells, ExplicitScheme::SspRk3);
    let end = &report.state.values;
    let (mut p_error, mut u_error) = (0.0_f64, 0.0_f64);
    for cell in 0..cells {
        let x = topology.cell_centroids()[2 * cell];
        if (0.55..=0.80).contains(&x) {
            let u = cell_state(end, cells, 4, cell);
            p_error = p_error.max((pressure(&u) - P_STAR).abs() / P_STAR);
            u_error = u_error.max((u[1] / u[0] - U_STAR).abs() / U_STAR);
        }
    }
    eprintln!("HLLC Sod: max rel error p {p_error:.3e}, u {u_error:.3e}");
    assert!(p_error <= 1e-2 && u_error <= 1e-2);

    let chandrashekar = case(
        CHANDRASHEKAR,
        "IdealGasChandrashekar2D",
        &[],
        authored("chandrashekar"),
        ReconstructionIntent::Constant,
        PositivityIntent::None,
    );
    let topology = square([(0.0, 1.0, 16), (0.0, 1.0, 16)], true);
    let operator = realize(&chandrashekar, &topology);
    let uniform = layout(&vec![conserved(1.3, &[0.7, -0.4], 0.9); 256]);
    let mut rhs = vec![1.0; uniform.len()];
    operator.rhs(&cx(), 0.0, &uniform, &mut rhs).unwrap();
    // Free-stream preservation to round-off. (A first "bitwise zero" bar was wrong: every face
    // flux is bitwise identical, but a cell accumulates Fx + Fy − Fx − Fy, which is not
    // associative.) Four accumulated terms of size ≤ |f|·max|F*_k| give
    // |rhs_k| ≤ 4ε·Σ_f |f||F*_k| / |Ω| = 16ε·max|F*_k| / h.
    let fluxes = operator.face_fluxes(0.0, &uniform).unwrap();
    for (k, flux) in fluxes.iter().enumerate() {
        let peak = flux
            .periodic
            .iter()
            .chain(&flux.interior)
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        let bound = 16.0 * f64::EPSILON * peak * 16.0;
        let worst = rhs[k * 256..(k + 1) * 256]
            .iter()
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        eprintln!(
            "Chandrashekar free stream component {k}: max |rhs| {worst:.3e}, bound {bound:.3e}"
        );
        assert!(worst <= bound);
    }
    assert_eq!(
        operator.receipt().residual_derivative,
        FvDerivativeDisposition::Smooth
    );
}

/// D6 on Euler faces: the residual JVP through LLF, HLL and HLLC face fluxes (abs/min/max
/// kinks), a Sweby limiter or a Zhang–Shu scaling is the typed `FV_NONSMOOTH_DERIVATIVE_REFUSED`
/// at its origin; the primal still evaluates. Chandrashekar (select only) differentiates: its
/// JVP matches a central difference of the residual within 1e-6·(1 + max|JVP|) (basis: step
/// 1e-6 ⇒ truncation ~1e-12·|R'''|, round-off ~ε|R|/1e-6 ~ 1e-9; the Ismail–Roe series and log
/// branches agree to ~w⁴ ≤ 1e-12 so a branch switch inside the stencil is below the bar).
#[test]
fn euler_face_derivatives_refuse_through_kinks_and_chandrashekar_differentiates() {
    let topology1 = line(0.0, 1.0, 16, false);
    let state1 = layout(
        &(0..16)
            .map(|i| {
                conserved(
                    1.0 + 0.1 * i as f64,
                    &[0.3 - 0.05 * i as f64],
                    1.0 + 0.02 * i as f64,
                )
            })
            .collect::<Vec<_>>(),
    );
    let refused = |operator: &FvSystemRealization, state: &[f64], origin: &str| {
        let n = state.len();
        let mut output = vec![0.0; n];
        let Err(NumericError::Evaluation {
            code, origin: at, ..
        }) = operator.jacobian_vector_product(
            &cx(),
            0.0,
            state,
            &vec![0.0; n],
            &vec![1.0; n],
            &vec![0.0; n],
            &mut output,
        )
        else {
            panic!("JVP must be refused at {origin}");
        };
        assert_eq!(code, FV_NONSMOOTH_DERIVATIVE_REFUSED);
        assert_eq!(at, origin);
        assert!(matches!(
            operator.receipt().residual_derivative,
            FvDerivativeDisposition::Refused { .. }
        ));
        operator.rhs(&cx(), 0.0, state, &mut output).unwrap();
    };
    for flux in [
        NumericalFluxIntent::LocalLaxFriedrichs,
        NumericalFluxIntent::Hll,
    ] {
        let case = case(
            EULER,
            "Euler1D",
            &[],
            flux,
            ReconstructionIntent::Constant,
            PositivityIntent::None,
        );
        refused(&realize(&case, &topology1), &state1, "fv/face_flux");
    }
    let limited = case(
        EULER,
        "Euler1D",
        &["mass_density", "pressure"],
        NumericalFluxIntent::LocalLaxFriedrichs,
        muscl(LimiterFamily::VanLeer),
        PositivityIntent::None,
    );
    refused(&realize(&limited, &topology1), &state1, "fv/limiter");
    let scaled = case(
        EULER,
        "Euler1D",
        &["mass_density", "pressure"],
        NumericalFluxIntent::LocalLaxFriedrichs,
        muscl(LimiterFamily::None),
        zhang_shu(),
    );
    refused(&realize(&scaled, &topology1), &state1, "fv/positivity");

    let topology2 = square([(0.0, 1.0, 6), (0.0, 1.0, 5)], true);
    let state2 = layout(
        &(0..30)
            .map(|c| {
                let s = c as f64;
                conserved(
                    1.0 + 0.2 * (0.7 * s).sin(),
                    &[0.4 * (0.3 * s).cos(), -0.2 + 0.01 * s],
                    1.0 + 0.1 * (0.5 * s).cos(),
                )
            })
            .collect::<Vec<_>>(),
    );
    let hllc = case(
        HLLC,
        "IdealGasHllc2D",
        &[],
        authored("hllc"),
        ReconstructionIntent::Constant,
        PositivityIntent::None,
    );
    refused(&realize(&hllc, &topology2), &state2, "fv/face_flux");

    for reconstruction in [ReconstructionIntent::Constant, muscl(LimiterFamily::None)] {
        let chandrashekar = case(
            CHANDRASHEKAR,
            "IdealGasChandrashekar2D",
            &[],
            authored("chandrashekar"),
            reconstruction,
            PositivityIntent::None,
        );
        let operator = realize(&chandrashekar, &topology2);
        let n = state2.len();
        let direction = (0..n)
            .map(|i| ((i * 7 % 11) as f64 - 5.0) / 5.0 * state2[i].abs().max(0.1) * 0.1)
            .collect::<Vec<_>>();
        let rate = vec![0.0; n];
        let mut jvp = vec![0.0; n];
        operator
            .jacobian_vector_product(&cx(), 0.0, &state2, &rate, &direction, &rate, &mut jvp)
            .unwrap();
        let step = 1e-6;
        let residual = |sign: f64| {
            let shifted = state2
                .iter()
                .zip(&direction)
                .map(|(u, d)| u + sign * step * d)
                .collect::<Vec<_>>();
            let mut out = vec![0.0; n];
            operator
                .residual(&cx(), 0.0, &shifted, &rate, &mut out)
                .unwrap();
            out
        };
        let (plus, minus) = (residual(1.0), residual(-1.0));
        let scale = 1.0 + jvp.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let defect = (0..n)
            .map(|i| ((plus[i] - minus[i]) / (2.0 * step) - jvp[i]).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("Chandrashekar JVP defect {defect:.3e} (scale {scale:.3e})");
        assert!(defect <= 1e-6 * scale, "{defect}");
    }
}

/// Slip walls. A gas at rest in a closed box is a bitwise steady state (1-D and 2-D, the 2-D
/// ghost `m − 2(m·n)n` lowered from `.res`). For any state the wall mass and energy fluxes are
/// exactly zero (ghost m·n = −m·n, equal ρ, p), so a 1-D pressure pulse conserves mass and
/// energy within the summed per-step balance bound. Identity and binding refusals.
#[test]
fn slip_walls_identity_and_binding_refusals() {
    let wall1 = case(
        EULER,
        "EulerWall1D",
        &["mass_density", "pressure"],
        NumericalFluxIntent::LocalLaxFriedrichs,
        muscl(LimiterFamily::Minmod),
        PositivityIntent::None,
    );
    let cells = 100;
    let topology = line(0.0, 1.0, cells, false);
    let operator = realize(&wall1, &topology);
    let rest = layout(&vec![conserved(1.0, &[0.0], 1.0); cells]);
    let mut rhs = vec![1.0; rest.len()];
    operator.rhs(&cx(), 0.0, &rest, &mut rhs).unwrap();
    assert!(rhs.iter().all(|value| *value == 0.0));
    let pulse = layout(
        &(0..cells)
            .map(|i| {
                let x = (i as f64 + 0.5) / cells as f64;
                conserved(1.0, &[0.0], 1.0 + 0.5 * (-((x - 0.3) / 0.05).powi(2)).exp())
            })
            .collect::<Vec<_>>(),
    );
    let report = integrate(&operator, pulse.clone(), 0.6, ExplicitScheme::SspRk3).unwrap();
    let bound = check_balances(&report, cells, ExplicitScheme::SspRk3);
    let (start, finish) = (
        totals(&pulse, topology.cell_volumes()),
        totals(&report.state.values, topology.cell_volumes()),
    );
    assert!((finish[0] - start[0]).abs() <= bound[0]);
    assert!((finish[2] - start[2]).abs() <= bound[2]);

    let wall2 = case(
        EULER,
        "EulerWall2D",
        &[],
        NumericalFluxIntent::LocalLaxFriedrichs,
        ReconstructionIntent::Constant,
        PositivityIntent::None,
    );
    let topology2 = square([(0.0, 1.0, 5), (0.0, 1.0, 4)], false);
    let operator2 = realize(&wall2, &topology2);
    let rest2 = layout(&vec![conserved(0.8, &[0.0, 0.0], 0.6); 20]);
    let mut rhs2 = vec![1.0; rest2.len()];
    operator2.rhs(&cx(), 0.0, &rest2, &mut rhs2).unwrap();
    assert!(rhs2.iter().all(|value| *value == 0.0));
    let moving = layout(
        &(0..20)
            .map(|c| {
                conserved(
                    1.0 + 0.03 * c as f64,
                    &[0.5 - 0.04 * c as f64, 0.2 + 0.03 * c as f64],
                    1.0,
                )
            })
            .collect::<Vec<_>>(),
    );
    let fluxes = operator2.face_fluxes(0.0, &moving).unwrap();
    for k in [0, 3] {
        for values in fluxes[k].boundary.values() {
            assert!(
                values.iter().all(|value| *value == 0.0),
                "component {k}: {values:?}"
            );
        }
    }
    for receipt in operator2.conservation_receipts(0.0, &moving).unwrap() {
        assert!(receipt.balanced);
    }

    // Identity: deterministic, data-sensitive; binding refusals.
    let again = realize(&wall2, &topology2);
    assert_eq!(again.digest(), operator2.digest());
    let other = FvSystemRealization::new(
        &wall2.law,
        wall2.program.clone(),
        topology2.clone(),
        &regions(&wall2.law),
        &BTreeMap::from([("gamma".to_owned(), 1.67)]),
    )
    .unwrap();
    assert_ne!(other.digest(), operator2.digest());
    let error = |result: Result<FvSystemRealization, finitum::FinitumError>| {
        result.unwrap_err().to_string()
    };
    assert!(
        error(FvSystemRealization::new(
            &wall2.law,
            wall2.program.clone(),
            topology2.clone(),
            &regions(&wall2.law),
            &BTreeMap::new()
        ))
        .contains("FV_DATUM_MISSING")
    );
    assert!(
        error(FvSystemRealization::new(
            &wall2.law,
            wall2.program.clone(),
            topology2.clone(),
            &RegionMap::new(),
            &gamma()
        ))
        .contains("FV_BOUNDARY_UNMAPPED")
    );
    assert!(
        error(FvSystemRealization::new(
            &wall1.law,
            wall1.program.clone(),
            topology2.clone(),
            &regions(&wall1.law),
            &gamma()
        ))
        .contains("FV_DIMENSION_MISMATCH")
    );
    assert!(
        error(FvSystemRealization::new(
            &wall1.law,
            wall2.program.clone(),
            topology2,
            &regions(&wall2.law),
            &gamma()
        ))
        .contains("FV_IDENTITY_MISMATCH")
    );
}

/// JIT parity (feature `cranelift`): every new system kernel — multi-output face fluxes (LLF,
/// authored HLLC and Chandrashekar), ghosts (transmissive, slip wall), speeds, admissible
/// quantities, Zhang–Shu θ and the Chandrashekar JVP — runs natively with no fallback, and the
/// rhs, CFL, face fluxes and JVP equal the interpreter's bitwise (bar: bitwise, the Malleus
/// strict-numerics contract; NaN never arises on these admissible states).
#[cfg(feature = "cranelift")]
#[test]
fn native_backend_matches_the_interpreter_bitwise() {
    use finitum::{ExecutionBackend, with_execution_backend};
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let toro = layout(
        &(0..64)
            .map(|i| conserved(1.0, &[if i < 32 { -2.0 } else { 2.0 }], 0.4))
            .collect::<Vec<_>>(),
    );
    let field2 = |cells: usize| {
        layout(
            &(0..cells)
                .map(|c| {
                    let s = c as f64;
                    conserved(
                        1.0 + 0.3 * (0.37 * s).sin(),
                        &[0.5 * (0.21 * s).cos(), -0.3 + 0.02 * (s % 7.0)],
                        1.0 + 0.2 * (0.53 * s).cos(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    };
    let cases = [
        (
            case(
                EULER,
                "Euler1D",
                &["mass_density", "pressure"],
                NumericalFluxIntent::LocalLaxFriedrichs,
                muscl(LimiterFamily::None),
                zhang_shu(),
            ),
            line(0.0, 1.0, 64, false),
            toro.clone(),
        ),
        (
            case(
                EULER,
                "Euler1D",
                &["mass_density", "pressure"],
                NumericalFluxIntent::Hll,
                muscl(LimiterFamily::Minmod),
                zhang_shu(),
            ),
            line(0.0, 1.0, 64, false),
            toro,
        ),
        (
            case(
                HLLC,
                "IdealGasHllc2D",
                &[],
                authored("hllc"),
                muscl(LimiterFamily::Minmod),
                PositivityIntent::None,
            ),
            square([(0.0, 1.0, 7), (0.0, 1.0, 6)], true),
            field2(42),
        ),
        (
            case(
                CHANDRASHEKAR,
                "IdealGasChandrashekar2D",
                &[],
                authored("chandrashekar"),
                muscl(LimiterFamily::None),
                PositivityIntent::None,
            ),
            square([(0.0, 1.0, 7), (0.0, 1.0, 6)], true),
            field2(42),
        ),
        (
            case(
                EULER,
                "EulerWall2D",
                &["mass_density", "pressure"],
                NumericalFluxIntent::LocalLaxFriedrichs,
                ReconstructionIntent::Constant,
                PositivityIntent::None,
            ),
            square([(0.0, 1.0, 7), (0.0, 1.0, 6)], false),
            field2(42),
        ),
    ];
    let mut fallbacks = 0;
    for (case, topology, state) in &cases {
        let interpreted =
            with_execution_backend(ExecutionBackend::Interpreter, || realize(case, topology));
        let native =
            with_execution_backend(ExecutionBackend::Cranelift, || realize(case, topology));
        let receipt = native.execution_receipt();
        eprintln!(
            "{}: native {}, interpreted {}, fallbacks {}",
            case.program.face_flux.kernel.name,
            receipt.native,
            receipt.interpreted,
            receipt.fallbacks
        );
        assert_eq!(receipt.backend, ExecutionBackend::Cranelift);
        for kernel in &receipt.kernels {
            if let finitum::KernelEngine::Fallback { reason } = &kernel.engine {
                eprintln!("  fallback {}: {reason}", kernel.name);
            }
        }
        fallbacks += receipt.fallbacks;
        assert_eq!(native.digest(), interpreted.digest());
        assert_eq!(native.receipt(), interpreted.receipt());
        let n = state.len();
        let rhs = |operator: &FvSystemRealization| {
            let mut out = vec![0.0; n];
            operator.rhs(&cx(), 0.0, state, &mut out).unwrap();
            out
        };
        assert_eq!(bits(&rhs(&native)), bits(&rhs(&interpreted)));
        assert_eq!(
            native.cfl(0.0, state).unwrap(),
            interpreted.cfl(0.0, state).unwrap()
        );
        assert_eq!(
            native.face_fluxes(0.0, state).unwrap(),
            interpreted.face_fluxes(0.0, state).unwrap()
        );
        if native.receipt().residual_derivative == FvDerivativeDisposition::Smooth {
            let direction = (0..n)
                .map(|i| ((i * 5 % 13) as f64 - 6.0) / 60.0)
                .collect::<Vec<_>>();
            let jvp = |operator: &FvSystemRealization| {
                let mut out = vec![0.0; n];
                operator
                    .jacobian_vector_product(
                        &cx(),
                        0.0,
                        state,
                        &vec![0.0; n],
                        &direction,
                        &vec![0.0; n],
                        &mut out,
                    )
                    .unwrap();
                out
            };
            assert_eq!(bits(&jvp(&native)), bits(&jvp(&interpreted)));
        }
    }
    eprintln!("JIT fallbacks across the cases: {fallbacks}");
}

/// RECORDED FAILING (W13 A7; bar fixed before the first run and kept): every new system kernel
/// runs natively. Measured: the authored HLLC face flux and the Chandrashekar face JVP are
/// refused by malleus-cranelift with `unsupported backend capability: "expression budget"` and
/// run on the interpreter (bitwise parity above still holds, trivially for those two). This is a
/// Malleus JIT limit (no Malleus edits in W13 Task A); recorded as a cross-repo need.
#[cfg(feature = "cranelift")]
#[test]
fn every_new_system_kernel_runs_natively() {
    use finitum::{ExecutionBackend, with_execution_backend};
    let topology = square([(0.0, 1.0, 4), (0.0, 1.0, 3)], true);
    let mut refused = Vec::new();
    for case in [
        case(
            HLLC,
            "IdealGasHllc2D",
            &[],
            authored("hllc"),
            ReconstructionIntent::Constant,
            PositivityIntent::None,
        ),
        case(
            CHANDRASHEKAR,
            "IdealGasChandrashekar2D",
            &[],
            authored("chandrashekar"),
            ReconstructionIntent::Constant,
            PositivityIntent::None,
        ),
    ] {
        let native =
            with_execution_backend(ExecutionBackend::Cranelift, || realize(&case, &topology));
        for kernel in native.execution_receipt().kernels {
            if let finitum::KernelEngine::Fallback { reason } = kernel.engine {
                refused.push(format!("{}: {reason}", kernel.name));
            }
        }
    }
    assert!(refused.is_empty(), "{refused:?}");
}
