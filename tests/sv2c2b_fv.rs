//! SV2-C2b: MUSCL reconstruction (least-squares gradients, Sweby and multidimensional
//! limiters), two-point diffusion, the smooth JVP/VJP chain and its D6 refusals, and the
//! pinned identities of SV2-C2a first-order realizations.
use std::sync::Arc;

use finitum::{
    FV_NONSMOOTH_DERIVATIVE_REFUSED, FinitumError, FvDerivativeDisposition, FvDiffusion, FvLimiter,
    FvRealization, FvReconstruction, FvTopology, MeshProfile, RegionMap, RegionTagId, TensorAxis,
    TensorBox,
};
use methodus::{DaeOperator, EvaluationContext, ExplicitOperator};
use quantitas::UnitRegistry;
use scientia::{
    ConservationLawProgram, ConservationLawSystem, FvMethodIntent, GradientScheme, LimiterFamily,
    NumericalFluxIntent, PositivityIntent, ReconstructionIntent, ReconstructionVariables,
    SemanticModule, SourceTreatment, compile_conservation_law_system, compile_semantics,
    lower_conservation_law,
};

const SCALAR: &str = include_str!("fixtures/fv-scalar.res");
const MUSCL: &str = include_str!("fixtures/fv-muscl.res");

fn module(source: &str) -> SemanticModule {
    compile_semantics(source, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("{diagnostics:#?}"))
        .semantic
}

fn muscl(gradient: GradientScheme, limiter: LimiterFamily) -> ReconstructionIntent {
    ReconstructionIntent::Muscl {
        variables: ReconstructionVariables::Conserved,
        gradient,
        limiter,
    }
}

fn law(
    source: &str,
    model: &str,
    numerical_flux: NumericalFluxIntent,
    reconstruction: ReconstructionIntent,
) -> (ConservationLawSystem, Arc<ConservationLawProgram>) {
    let module = module(source);
    let system = compile_conservation_law_system(&module, model, "balance", "u").unwrap();
    let intent = FvMethodIntent {
        numerical_flux,
        reconstruction,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    };
    let program = lower_conservation_law(&module, &system, &intent, &[]).unwrap();
    (system, Arc::new(program))
}

fn regions(system: &ConservationLawSystem) -> RegionMap {
    let mut map = RegionMap::new();
    for boundary in &system.boundaries {
        let (scientia::BoundaryFluxSemantics::GhostState { region, .. }
        | scientia::BoundaryFluxSemantics::PrescribedFlux { region, .. }) = boundary;
        let tag = match boundary.region_name() {
            "left" => "x_min",
            "right" => "x_max",
            "bottom" => "y_min",
            "top" => "y_max",
            other => panic!("unexpected boundary {other}"),
        };
        map.insert(*region, [RegionTagId::new(tag)]);
    }
    map
}

fn realization(
    source: &str,
    model: &str,
    flux: NumericalFluxIntent,
    reconstruction: ReconstructionIntent,
    topology: &Arc<FvTopology>,
    data: &[(&str, f64)],
) -> Result<FvRealization, FinitumError> {
    let (system, program) = law(source, model, flux, reconstruction);
    FvRealization::new(
        &system,
        program,
        topology.clone(),
        &regions(&system),
        &data.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    )
}

fn line(cells: usize, periodic: bool) -> Arc<FvTopology> {
    let axis = TensorAxis::uniform(0.0, 1.0, cells);
    let axis = if periodic { axis.periodic() } else { axis };
    Arc::new(FvTopology::from_tensor_box(&TensorBox { axes: vec![axis] }).unwrap())
}

fn simplex_square(subdivisions: usize) -> Arc<FvTopology> {
    Arc::new(
        FvTopology::from_profile(&MeshProfile::SimplexBox {
            dimension: 2,
            extent: vec![[0.0, 1.0], [0.0, 1.0]],
            subdivisions: vec![subdivisions, subdivisions],
        })
        .unwrap(),
    )
}

fn cx() -> EvaluationContext {
    EvaluationContext::reproducible()
}

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut seed = seed | 1;
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn code(error: &FinitumError) -> String {
    let message = error.to_string();
    message
        .split(':')
        .find(|part| part.trim_start().starts_with("FV_"))
        .map(|part| part.trim().to_string())
        .unwrap_or(message)
}

fn residual(operator: &FvRealization, state: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; state.len()];
    let rate = vec![0.0; state.len()];
    DaeOperator::residual(operator, &cx(), 0.0, state, &rate, &mut out).unwrap();
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

fn rhs(operator: &FvRealization, state: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; state.len()];
    ExplicitOperator::rhs(operator, &cx(), 0.0, state, &mut out).unwrap();
    out
}

/// One SSP-RK3 (Shu–Osher) step.
fn ssp_rk3(operator: &FvRealization, state: &[f64], step: f64) -> Vec<f64> {
    let stage = |base: &[f64], weight: f64, current: &[f64]| {
        let rate = rhs(operator, current);
        base.iter()
            .zip(current)
            .zip(&rate)
            .map(|((b, c), r)| (1.0 - weight) * b + weight * (c + step * r))
            .collect::<Vec<_>>()
    };
    let one = stage(state, 1.0, state);
    let two = stage(state, 0.25, &one);
    stage(state, 2.0 / 3.0, &two)
}

#[test]
fn first_order_identities_are_pinned() {
    // Captured from W12 L4 head b935dd2 before any SV2-C2b change.
    for (model, flux, periodic, data, digest, receipt) in [
        (
            "Burgers1D",
            NumericalFluxIntent::Hll,
            true,
            ("k", 0.3),
            "d4e8bc9aab85db722c26899cf82c9478e9304e609725a2d966ea496b09634299",
            "b78d546e18aa3cc3ecc402a46f20e4060c7b0c8b3c5b2a1e3f36b0bec992638c",
        ),
        (
            "InflowAdvection1D",
            NumericalFluxIntent::LocalLaxFriedrichs,
            false,
            ("a", 1.0),
            "65b1b74beb4f0d8bb47f5ac1542ced94f475f0e46145f33efc2c107a5e8d04ea",
            "4c95bfb0ebd3cd3a5a9f6bee7a200bb12ebb017ec7c93dabd8e6d9bf613adc96",
        ),
    ] {
        let operator = realization(
            SCALAR,
            model,
            flux,
            ReconstructionIntent::Constant,
            &line(8, periodic),
            &[data],
        )
        .unwrap();
        assert_eq!(operator.digest().hex, digest, "{model}");
        let bytes = serde_json::to_vec(&operator.receipt()).unwrap();
        assert_eq!(blake3::hash(&bytes).to_hex().as_str(), receipt, "{model}");
        assert!(operator.reconstruction().is_none());
    }
}

#[test]
fn least_squares_gradients_are_exact_for_linear_fields_on_tensor_and_simplex_meshes() {
    let graded = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![
                TensorAxis::graded(vec![0.0, 0.1, 0.35, 0.5, 0.8, 1.0]),
                TensorAxis::uniform(0.0, 2.0, 4),
            ],
        })
        .unwrap(),
    );
    for topology in [graded, simplex_square(4)] {
        let reconstruction = FvReconstruction::new(topology.clone(), FvLimiter::Unlimited).unwrap();
        assert_eq!(reconstruction.degenerate_cells(), 0);
        assert!(reconstruction.derivative_refusal().is_none());
        let field = |x: &[f64]| 0.5 + 2.0 * x[0] - 3.0 * x[1];
        let state = topology
            .cell_centroids()
            .chunks(2)
            .map(field)
            .collect::<Vec<_>>();
        let slopes = reconstruction.slopes(&state).unwrap();
        for gradient in slopes.chunks(2) {
            assert!((gradient[0] - 2.0).abs() < 1e-12 && (gradient[1] + 3.0).abs() < 1e-12);
        }
        // Linear reconstruction reproduces the field at every face centroid, both sides.
        let blocks = [topology.interior(), topology.periodic()]
            .into_iter()
            .chain(topology.boundary().values());
        for (block, states) in blocks.zip(reconstruction.face_states(&state).unwrap()) {
            for face in 0..block.len() {
                let exact = field(&block.centroid[2 * face..2 * face + 2]);
                assert!((states.minus[face] - exact).abs() < 1e-12);
                if let Some(plus) = states.plus.get(face) {
                    assert!((plus - exact).abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn multidimensional_limiters_bound_simplex_face_states_by_neighbours() {
    let topology = simplex_square(8);
    let state = topology
        .cell_centroids()
        .chunks(2)
        .map(|x| if x[0] + 0.3 * x[1] < 0.55 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let unlimited = FvReconstruction::new(topology.clone(), FvLimiter::Unlimited)
        .unwrap()
        .face_states(&state)
        .unwrap();
    let overshoot = unlimited
        .iter()
        .flat_map(|block| block.minus.iter().chain(&block.plus))
        .any(|value| !(-1e-12..=1.0 + 1e-12).contains(value));
    assert!(
        overshoot,
        "unlimited reconstruction should overshoot a step"
    );
    for (limiter, bounded) in [
        (FvLimiter::BarthJespersen, true),
        (FvLimiter::Venkatakrishnan { k: 0.3 }, false),
    ] {
        let reconstruction = FvReconstruction::new(topology.clone(), limiter).unwrap();
        let (refusal, _) = reconstruction.derivative_refusal().unwrap();
        assert_eq!(refusal, FV_NONSMOOTH_DERIVATIVE_REFUSED);
        // Barth-Jespersen is strictly bounded; Venkatakrishnan trades an O(ε²) overshoot for
        // smoothness, ε² = (k h)³ ≈ 1.8e-5 here.
        let slack = if bounded { 1e-12 } else { 1e-3 };
        for block in reconstruction.face_states(&state).unwrap() {
            for value in block.minus.iter().chain(&block.plus) {
                assert!((-slack..=1.0 + slack).contains(value), "{value}");
            }
        }
        // A linear field is far from its bounds only in the interior; it stays bounded.
        let ramp = topology
            .cell_centroids()
            .chunks(2)
            .map(|x| x[0])
            .collect::<Vec<_>>();
        assert!(
            reconstruction
                .slopes(&ramp)
                .unwrap()
                .iter()
                .all(|s| s.is_finite())
        );
    }
    assert_eq!(
        code(
            &FvReconstruction::new(topology.clone(), FvLimiter::Venkatakrishnan { k: 0.0 })
                .unwrap_err()
        ),
        "FV_LIMITER_UNSUPPORTED"
    );
}

#[test]
fn smooth_chain_satisfies_the_transpose_identity_and_a_second_order_taylor_test() {
    let unlimited = || muscl(GradientScheme::LeastSquares, LimiterFamily::None);
    let cases = [
        (
            realization(
                SCALAR,
                "Burgers1D",
                NumericalFluxIntent::Central,
                unlimited(),
                &line(16, true),
                &[("k", 0.3)],
            )
            .unwrap(),
            "burgers",
        ),
        (
            realization(
                SCALAR,
                "InflowAdvection1D",
                NumericalFluxIntent::LocalLaxFriedrichs,
                unlimited(),
                &line(16, false),
                &[("a", 1.0)],
            )
            .unwrap()
            .with_diffusion(FvDiffusion { coefficient: 0.05 })
            .unwrap(),
            "inflow+diffusion",
        ),
        (
            realization(
                MUSCL,
                "WalledAdvection2D",
                NumericalFluxIntent::LocalLaxFriedrichs,
                unlimited(),
                &simplex_square(4),
                &[("ax", 1.0), ("ay", 0.5)],
            )
            .unwrap(),
            "simplex-2d",
        ),
    ];
    for (operator, label) in &cases {
        let cells = operator.dimension();
        let receipt = operator.receipt();
        assert_eq!(receipt.reconstruction, "muscl");
        assert_eq!(
            receipt.muscl.as_ref().unwrap().derivative,
            FvDerivativeDisposition::Smooth
        );
        let state = values(cells, 3)
            .iter()
            .map(|v| 0.5 + 0.4 * v)
            .collect::<Vec<_>>();
        let v = values(cells, 5);
        let w = values(cells, 7);
        let jv = jvp(operator, &state, &v);
        let mut jtw = vec![0.0; cells];
        operator.state_vjp(0.0, &state, &w, &mut jtw).unwrap();
        let (left, right) = (dot(&jv, &w), dot(&v, &jtw));
        let scale =
            jv.iter().map(|x| x.abs()).sum::<f64>() + jtw.iter().map(|x| x.abs()).sum::<f64>();
        assert!(
            (left - right).abs() <= 64.0 * f64::EPSILON * scale,
            "{label}: <Jv,w> = {left}, <v,J^T w> = {right}"
        );
        // Taylor: ‖R(u + h v) − R(u) − h Jv‖ = O(h²).
        let base = residual(operator, &state);
        let remainder = |h: f64| {
            let shifted = state
                .iter()
                .zip(&v)
                .map(|(u, d)| u + h * d)
                .collect::<Vec<_>>();
            residual(operator, &shifted)
                .iter()
                .zip(&base)
                .zip(&jv)
                .map(|((r, b), j)| (r - b - h * j).abs())
                .fold(0.0, f64::max)
        };
        let (coarse, fine) = (remainder(1e-2), remainder(5e-3));
        if *label == "burgers" {
            let order = (coarse / fine).log2();
            assert!((1.9..2.1).contains(&order), "{label}: Taylor order {order}");
        } else {
            // A linear operator: the remainder is round-off.
            assert!(coarse < 1e-12 && fine < 1e-12, "{label}: {coarse} {fine}");
        }
    }
}

#[test]
fn limited_reconstruction_refuses_derivatives_through_malleus_and_scientia() {
    let ring = line(12, true);
    let state = values(12, 11);
    for limiter in [
        LimiterFamily::Minmod,
        LimiterFamily::VanLeer,
        LimiterFamily::MonotonizedCentral,
        LimiterFamily::Superbee,
    ] {
        let operator = realization(
            SCALAR,
            "PeriodicAdvection1D",
            NumericalFluxIntent::LocalLaxFriedrichs,
            muscl(GradientScheme::Axis1D, limiter.clone()),
            &ring,
            &[("a", 1.0)],
        )
        .unwrap();
        let mut out = vec![0.0; 12];
        let zero = vec![0.0; 12];
        let error = DaeOperator::jacobian_vector_product(
            &operator,
            &cx(),
            0.0,
            &state,
            &zero,
            &state,
            &zero,
            &mut out,
        )
        .unwrap_err();
        let text = format!("{error:?}");
        assert!(
            text.contains(FV_NONSMOOTH_DERIVATIVE_REFUSED) && text.contains("fv/limiter"),
            "{text}"
        );
        let vjp = operator
            .state_vjp(0.0, &state, &state, &mut out)
            .unwrap_err();
        assert!(matches!(vjp, FinitumError::InputEvaluation(_)), "{vjp}");
        // The primal is unaffected.
        assert!(residual(&operator, &state).iter().all(|r| r.is_finite()));
        let receipt = operator.receipt();
        let muscl = receipt.muscl.unwrap();
        assert_eq!(muscl.limiter, limiter);
        assert_eq!(
            muscl.derivative,
            FvDerivativeDisposition::Refused {
                code: FV_NONSMOOTH_DERIVATIVE_REFUSED.into()
            }
        );
        assert_eq!(
            muscl.limiter_kernel_digest.as_ref(),
            Some(&operator.program().limiter.as_ref().unwrap().digest)
        );
        assert_eq!(receipt.max_nonorthogonality, Some(0.0));
        // Scientia refuses the same limiter derivative at its level (D6 consistency).
        let kernel = operator.program().limiter.as_ref().unwrap();
        let request = malleus::DerivativeRequest {
            mode: malleus::DerivativeMode::Jvp,
            independent_operands: vec![
                kernel
                    .operand(&scientia::FvOperandMeaning::SlopeRatio)
                    .unwrap(),
            ],
            dependent_operands: vec![kernel.operand(&scientia::FvOperandMeaning::Output).unwrap()],
        };
        assert_eq!(
            kernel.differentiate(&request).unwrap_err().code,
            FV_NONSMOOTH_DERIVATIVE_REFUSED
        );
    }
}

fn total_variation(values: &[f64]) -> f64 {
    let n = values.len();
    (0..n)
        .map(|i| (values[(i + 1) % n] - values[i]).abs())
        .sum()
}

#[test]
fn sweby_limiters_are_tvd_on_a_periodic_square_wave_under_ssp_rk3() {
    let cells = 100;
    let ring = line(cells, true);
    let initial = (0..cells)
        .map(|i| {
            let x = (i as f64 + 0.5) / cells as f64;
            if (0.2..0.5).contains(&x) { 1.0 } else { 0.0 }
        })
        .collect::<Vec<_>>();
    let run = |limiter: LimiterFamily| {
        let operator = realization(
            SCALAR,
            "PeriodicAdvection1D",
            NumericalFluxIntent::LocalLaxFriedrichs,
            muscl(GradientScheme::Axis1D, limiter),
            &ring,
            &[("a", 1.0)],
        )
        .unwrap();
        let mut state = initial.clone();
        let mut worst_tv_growth = f64::NEG_INFINITY;
        let (mut low, mut high) = (0.0_f64, 1.0_f64);
        for _ in 0..120 {
            let step = operator.cfl(0.0, &state).unwrap().stable_step;
            let next = ssp_rk3(&operator, &state, step);
            worst_tv_growth = worst_tv_growth.max(total_variation(&next) - total_variation(&state));
            for value in &next {
                low = low.min(*value);
                high = high.max(*value);
            }
            state = next;
        }
        (worst_tv_growth, low, high)
    };
    for limiter in [
        LimiterFamily::Minmod,
        LimiterFamily::VanLeer,
        LimiterFamily::MonotonizedCentral,
        LimiterFamily::Superbee,
    ] {
        let (growth, low, high) = run(limiter.clone());
        assert!(growth <= 1e-12, "{limiter:?}: TV grew by {growth}");
        assert!(
            low >= -1e-12 && high <= 1.0 + 1e-12,
            "{limiter:?}: [{low}, {high}]"
        );
    }
    // Unlimited MUSCL is not TVD: the control fails the same check.
    let (growth, low, high) = run(LimiterFamily::None);
    assert!(growth > 1e-6 && (low < -1e-6 || high > 1.0 + 1e-6));
}

#[test]
fn muscl_converges_at_second_order_on_smooth_periodic_advection() {
    // L1 error against the exact translate of sin(2πx) at t = 0.5, SSP-RK3 at the stable step
    // (time error O(h³)); the basis is smooth linear advection, where limiters clip only near
    // the two extrema.
    let error = |reconstruction: ReconstructionIntent, cells: usize| {
        let ring = line(cells, true);
        let operator = realization(
            SCALAR,
            "PeriodicAdvection1D",
            NumericalFluxIntent::LocalLaxFriedrichs,
            reconstruction,
            &ring,
            &[("a", 1.0)],
        )
        .unwrap();
        let h = 1.0 / cells as f64;
        let average = |x0: f64, x1: f64| {
            let tau = std::f64::consts::TAU;
            ((tau * x0).cos() - (tau * x1).cos()) / (tau * (x1 - x0))
        };
        let mut state = (0..cells)
            .map(|i| average(i as f64 * h, (i + 1) as f64 * h))
            .collect::<Vec<_>>();
        let end = 0.5;
        let base = operator.cfl(0.0, &state).unwrap().stable_step;
        let steps = (end / base).ceil() as usize;
        let step = end / steps as f64;
        for _ in 0..steps {
            state = ssp_rk3(&operator, &state, step);
        }
        (0..cells)
            .map(|i| (state[i] - average(i as f64 * h - end, (i + 1) as f64 * h - end)).abs() * h)
            .sum::<f64>()
    };
    let order = |reconstruction: ReconstructionIntent| {
        let (coarse, fine) = (
            error(reconstruction.clone(), 128),
            error(reconstruction, 256),
        );
        (coarse / fine).log2()
    };
    let first = order(ReconstructionIntent::Constant);
    let unlimited = order(muscl(GradientScheme::LeastSquares, LimiterFamily::None));
    let van_leer = order(muscl(GradientScheme::Axis1D, LimiterFamily::VanLeer));
    let mc = order(muscl(
        GradientScheme::Axis1D,
        LimiterFamily::MonotonizedCentral,
    ));
    println!(
        "observed L1 orders (128->256): first {first:.3} unlimited {unlimited:.3} van Leer {van_leer:.3} MC {mc:.3}"
    );
    assert!(first >= 0.9, "{first}");
    assert!(unlimited >= 1.9, "{unlimited}");
    assert!(van_leer >= 1.7, "{van_leer}");
    assert!(mc >= 1.7, "{mc}");
}

/// Solve the linear steady problem `R(u) = 0` by assembling `J` from JVP columns.
fn steady(operator: &FvRealization) -> Vec<f64> {
    let n = operator.dimension();
    let zero = vec![0.0; n];
    let b = residual(operator, &zero);
    let mut matrix = vec![vec![0.0; n + 1]; n];
    for column in 0..n {
        let mut e = vec![0.0; n];
        e[column] = 1.0;
        let jc = jvp(operator, &zero, &e);
        for row in 0..n {
            matrix[row][column] = jc[row];
        }
    }
    for row in 0..n {
        matrix[row][n] = -b[row];
    }
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

#[test]
fn two_point_diffusion_matches_steady_analytic_solutions() {
    let diffusive = |a: f64, kappa: f64, cells: usize| {
        realization(
            MUSCL,
            "DirichletAdvection1D",
            NumericalFluxIntent::Central,
            ReconstructionIntent::Constant,
            &line(cells, false),
            &[("a", a)],
        )
        .unwrap()
        .with_diffusion(FvDiffusion { coefficient: kappa })
        .unwrap()
    };
    // Pure diffusion between u(0) = 0 and u(1) = 1: the linear profile is reproduced exactly.
    let operator = diffusive(0.0, 0.7, 10);
    let solution = steady(&operator);
    for (cell, value) in solution.iter().enumerate() {
        assert!((value - (cell as f64 + 0.5) / 10.0).abs() < 1e-13);
    }
    let receipt = operator.receipt();
    let diffusion = receipt.diffusion.unwrap();
    assert_eq!(
        (diffusion.scheme.as_str(), diffusion.coefficient),
        ("two_point", 0.7)
    );
    assert_eq!(diffusion.max_nonorthogonality, 0.0);
    assert!(receipt.muscl.is_none());
    assert_ne!(
        operator.digest(),
        realization(
            MUSCL,
            "DirichletAdvection1D",
            NumericalFluxIntent::Central,
            ReconstructionIntent::Constant,
            &line(10, false),
            &[("a", 0.0)],
        )
        .unwrap()
        .digest()
    );
    // Advection–diffusion boundary layer, Pe = 10: u = (e^{Pe x} − 1)/(e^{Pe} − 1).
    let peclet: f64 = 10.0;
    let exact = |x: f64| ((peclet * x).exp() - 1.0) / (peclet.exp() - 1.0);
    let error = |cells: usize| {
        let solution = steady(&diffusive(1.0, 1.0 / peclet, cells));
        solution
            .iter()
            .enumerate()
            .map(|(i, u)| (u - exact((i as f64 + 0.5) / cells as f64)).abs())
            .fold(0.0, f64::max)
    };
    let (coarse, fine) = (error(40), error(80));
    let order = (coarse / fine).log2();
    println!("TPFA + central boundary layer max error {coarse:e} -> {fine:e}, order {order:.3}");
    assert!(fine < 2e-3 && order >= 1.8, "{coarse} {fine} {order}");
    // The CFL bound adds the transmissibilities: a boundary cell, with its half-cell face
    // `2κ/h`, limits it to h / (2a + 3κ/h).
    let operator = diffusive(1.0, 0.1, 10);
    let cfl = operator.cfl(0.0, &[0.0; 10]).unwrap();
    assert!(
        (cfl.stable_step - 0.1 / (2.0 + 3.0)).abs() < 1e-15,
        "{cfl:?}"
    );
    assert!(matches!(cfl.limiting_cell, Some(0 | 9)));
}

#[test]
fn simplex_refusals_and_binding_shapes() {
    let square = simplex_square(3);
    assert!(square.max_nonorthogonality() > 1e-3);
    let walled = |reconstruction: ReconstructionIntent| {
        realization(
            MUSCL,
            "WalledAdvection2D",
            NumericalFluxIntent::LocalLaxFriedrichs,
            reconstruction,
            &square,
            &[("ax", 1.0), ("ay", 0.5)],
        )
    };
    let first = walled(ReconstructionIntent::Constant).unwrap();
    assert_eq!(
        code(
            &first
                .with_diffusion(FvDiffusion { coefficient: 0.1 })
                .unwrap_err()
        ),
        "FV_TPFA_NONORTHOGONAL"
    );
    assert_eq!(
        code(&walled(muscl(GradientScheme::LeastSquares, LimiterFamily::Minmod)).unwrap_err()),
        "FV_LIMITER_UNSUPPORTED"
    );
    assert_eq!(
        code(&walled(muscl(GradientScheme::Axis1D, LimiterFamily::None)).unwrap_err()),
        "FV_GRADIENT_UNSUPPORTED"
    );
    assert_eq!(
        code(&walled(muscl(GradientScheme::GreenGauss, LimiterFamily::None)).unwrap_err()),
        "FV_GRADIENT_UNSUPPORTED"
    );
    let unlimited = walled(muscl(GradientScheme::LeastSquares, LimiterFamily::None)).unwrap();
    let receipt = unlimited.receipt();
    assert!(receipt.max_nonorthogonality.unwrap() > 1e-3);
    assert_eq!(receipt.muscl.unwrap().degenerate_gradient_cells, 0);
    // MUSCL keeps the conservation receipt balanced on the simplex mesh.
    let state = values(unlimited.dimension(), 23);
    assert!(
        unlimited
            .conservation_receipt(0.0, &state)
            .unwrap()
            .balanced
    );
    let tensor = line(4, false);
    let operator = realization(
        MUSCL,
        "DirichletAdvection1D",
        NumericalFluxIntent::Central,
        ReconstructionIntent::Constant,
        &tensor,
        &[("a", 1.0)],
    )
    .unwrap();
    assert_eq!(
        code(
            &operator
                .with_diffusion(FvDiffusion { coefficient: -1.0 })
                .unwrap_err()
        ),
        "FV_DIFFUSION_INVALID"
    );
}

#[cfg(feature = "cranelift")]
#[path = "support/backend.rs"]
mod backend;

/// W13 A1: MUSCL reconstruction with every limiter family Scientia lowers (Barth-Jespersen and
/// Venkatakrishnan are refused at lowering, `FV_LIMITER_UNSUPPORTED`) and every gradient
/// scheme, and two-point diffusion, run bitwise equal on Cranelift (tensor 1-D, simplex 2-D).
#[cfg(feature = "cranelift")]
#[test]
fn cranelift_backend_is_bitwise_equal_on_muscl_limiters_and_tpfa() {
    let limiters = [
        LimiterFamily::None,
        LimiterFamily::Minmod,
        LimiterFamily::VanLeer,
        LimiterFamily::MonotonizedCentral,
        LimiterFamily::Superbee,
    ];
    for limiter in limiters {
        for (gradient, simplex) in [
            (GradientScheme::Axis1D, false),
            (GradientScheme::LeastSquares, false),
            (GradientScheme::LeastSquares, true),
            (GradientScheme::GreenGauss, true),
        ] {
            let label = format!("{limiter:?}/{gradient:?}/simplex={simplex}");
            let ((interpreted, receipt_i), (native, receipt_n)) = backend::on_both_backends(|| {
                let built = if simplex {
                    realization(
                        MUSCL,
                        "WalledAdvection2D",
                        NumericalFluxIntent::LocalLaxFriedrichs,
                        muscl(gradient, limiter.clone()),
                        &simplex_square(4),
                        &[("ax", 1.0), ("ay", 0.5)],
                    )
                } else {
                    realization(
                        SCALAR,
                        "InflowAdvection1D",
                        NumericalFluxIntent::LocalLaxFriedrichs,
                        muscl(gradient, limiter.clone()),
                        &line(16, false),
                        &[("a", 1.0)],
                    )
                    .and_then(|operator| operator.with_diffusion(FvDiffusion { coefficient: 0.05 }))
                };
                let operator = match built {
                    Ok(operator) => operator,
                    Err(error) => return (vec![Err(error.to_string())], None),
                };
                let cells = operator.dimension();
                let state = values(cells, 41);
                let step = state
                    .iter()
                    .enumerate()
                    .map(|(index, _)| if index < cells / 2 { 1.0 } else { -0.0 })
                    .collect::<Vec<_>>();
                let direction = values(cells, 43);
                let mut outcomes = Vec::new();
                for state in [&state, &step] {
                    let zero = vec![0.0; cells];
                    let mut out = vec![0.0; cells];
                    outcomes.push(backend::outcome(
                        DaeOperator::residual(&operator, &cx(), 0.0, state, &zero, &mut out)
                            .map(|()| out.clone()),
                    ));
                    outcomes.push(backend::outcome(
                        DaeOperator::jacobian_vector_product(
                            &operator,
                            &cx(),
                            0.0,
                            state,
                            &zero,
                            &direction,
                            &zero,
                            &mut out,
                        )
                        .map(|()| out.clone()),
                    ));
                    outcomes.push(backend::outcome(
                        ExplicitOperator::rhs(&operator, &cx(), 0.0, state, &mut out)
                            .map(|()| out.clone()),
                    ));
                }
                (outcomes, Some(operator.execution_receipt()))
            });
            backend::assert_bitwise_outcomes(&label, &interpreted, &native);
            match (receipt_i, receipt_n) {
                (Some(left), Some(right)) => backend::assert_native_receipts(&label, &left, &right),
                (None, None) => {}
                other => panic!("{label}: construction differs by backend: {other:?}"),
            }
        }
    }
}
