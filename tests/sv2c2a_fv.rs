//! SV2-C2a: finite-volume topology (tensor and simplex), first-order realization of Scientia
//! conservation-law programs, conservation receipt, CFL and Methodus explicit/BDF stepping.
use std::collections::BTreeMap;
use std::sync::Arc;

use finitum::{
    AxisSpacing, FvCellShape, FvConservationReceipt, FvDerivativeDisposition, FvFaceFluxes,
    FvRealization, FvTopology, MeshProfile, RegionMap, RegionTagId, TensorAxis, TensorBox,
    discrete_flux_balance, realize, scatter_face_fluxes,
};
use methodus::{
    BdfConfig, BdfOrder, BdfState, DaeOperator, EvaluationContext, ExplicitConfig,
    ExplicitOperator, ExplicitScheme, ExplicitState, NumericError, StepOutcome, StepPolicy,
    bdf_step, integrate_explicit,
};
use quantitas::UnitRegistry;
use scientia::{
    ConservationLawProgram, ConservationLawSystem, FvMethodIntent, GradientScheme, LimiterFamily,
    NumericalFluxIntent, PositivityIntent, ReconstructionIntent, ReconstructionVariables,
    SemanticModule, SourceTreatment, compile_conservation_law_system, compile_semantics,
    lower_conservation_law,
};

const FIXTURE: &str = include_str!("fixtures/fv-scalar.res");

fn module() -> SemanticModule {
    compile_semantics(FIXTURE, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("{diagnostics:#?}"))
        .semantic
}

fn intent(numerical_flux: NumericalFluxIntent) -> FvMethodIntent {
    FvMethodIntent {
        numerical_flux,
        reconstruction: ReconstructionIntent::Constant,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    }
}

fn law(
    module: &SemanticModule,
    model: &str,
    flux: NumericalFluxIntent,
) -> (ConservationLawSystem, Arc<ConservationLawProgram>) {
    let system = compile_conservation_law_system(module, model, "balance", "u").unwrap();
    let program = lower_conservation_law(module, &system, &intent(flux), &[]).unwrap();
    (system, Arc::new(program))
}

/// Map the fixture's `left`/`right` boundaries onto the box tags.
fn regions(system: &ConservationLawSystem) -> RegionMap {
    let mut map = RegionMap::new();
    for boundary in &system.boundaries {
        let (scientia::BoundaryFluxSemantics::GhostState { region, .. }
        | scientia::BoundaryFluxSemantics::PrescribedFlux { region, .. }) = boundary;
        let tag = match boundary.region_name() {
            "left" => "x_min",
            "right" => "x_max",
            other => panic!("unexpected boundary {other}"),
        };
        map.insert(*region, [RegionTagId::new(tag)]);
    }
    map
}

fn realization(
    model: &str,
    flux: NumericalFluxIntent,
    topology: &Arc<FvTopology>,
    data: &[(&str, f64)],
) -> Result<FvRealization, finitum::FinitumError> {
    let module = module();
    let (system, program) = law(&module, model, flux);
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

fn cx() -> EvaluationContext {
    EvaluationContext::reproducible()
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
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

#[test]
fn tensor_topology_numbering_periodic_pairing_and_invariants() {
    let one = line(4, false);
    assert_eq!(one.shape(), FvCellShape::Tensor);
    assert_eq!(one.cell_volumes(), &[0.25; 4]);
    assert_eq!(one.interior().minus, [0, 1, 2]);
    assert_eq!(one.interior().plus, [1, 2, 3]);
    assert_eq!(one.periodic().len(), 0);
    let tags = one
        .boundary()
        .keys()
        .map(RegionTagId::as_str)
        .collect::<Vec<_>>();
    assert_eq!(tags, ["x_max", "x_min"]);
    assert_eq!(one.boundary()[&RegionTagId::new("x_min")].normal[0], [-1.0]);
    assert_eq!(one.boundary()[&RegionTagId::new("x_max")].minus, [3]);
    let report = one.check_invariants().unwrap();
    assert_eq!((report.faces_per_cell, report.max_closure_defect), (2, 0.0));

    let ring = line(5, true);
    assert_eq!(ring.boundary().len(), 0);
    assert_eq!((ring.periodic().minus[0], ring.periodic().plus[0]), (4, 0));
    assert_eq!(ring.periodic().translation, [1.0]);
    assert!((ring.periodic().delta[0] - 0.2).abs() < 1e-15);
    ring.check_invariants().unwrap();
    // A one-cell ring pairs the cell with itself across the wrap.
    line(1, true).check_invariants().unwrap();

    // 2-D graded in x, periodic in x, uniform in y.
    let spec = TensorBox {
        axes: vec![
            TensorAxis::graded(vec![0.0, 0.1, 0.4, 1.0]).periodic(),
            TensorAxis::uniform(-1.0, 1.0, 2),
        ],
    };
    let plane = FvTopology::from_tensor_box(&spec).unwrap();
    assert_eq!(plane.cell_count(), 6);
    assert!((plane.cell_volumes()[2] - 0.6).abs() < 1e-15);
    assert_eq!(&plane.cell_centroids()[2 * 3..2 * 4], &[0.05, 0.5]);
    // x faces: 2 interior per row; y faces: 3 interior between the rows.
    assert_eq!(plane.interior().len(), 4 + 3);
    assert_eq!(plane.interior().minus[..4], [0, 1, 3, 4]);
    assert_eq!(plane.interior().minus[4..], [0, 1, 2]);
    assert_eq!(plane.interior().plus[4..], [3, 4, 5]);
    assert_eq!(plane.periodic().minus, [2, 5]);
    assert_eq!(plane.periodic().plus, [0, 3]);
    assert_eq!(plane.periodic().translation, [1.0, 0.0, 1.0, 0.0]);
    assert!((plane.periodic().delta[0] - (0.3 + 0.05)).abs() < 1e-15);
    let tags = plane
        .boundary()
        .keys()
        .map(RegionTagId::as_str)
        .collect::<Vec<_>>();
    assert_eq!(tags, ["y_max", "y_min"]);
    assert_eq!(
        plane.boundary()[&RegionTagId::new("y_min")].area,
        [0.1, 0.30000000000000004, 0.6]
    );
    let report = plane.check_invariants().unwrap();
    assert_eq!(
        (
            report.interior_faces,
            report.periodic_faces,
            report.boundary_faces
        ),
        (7, 2, 6)
    );
    assert_eq!(report.max_closure_defect, 0.0);

    // Deterministic identity; spacing changes it.
    assert_eq!(
        FvTopology::from_tensor_box(&spec).unwrap().digest(),
        plane.digest()
    );
    let mut other = spec.clone();
    other.axes[1].spacing = AxisSpacing::Uniform {
        min: -1.0,
        max: 1.5,
        cells: 2,
    };
    assert_ne!(
        FvTopology::from_tensor_box(&other).unwrap().digest(),
        plane.digest()
    );

    // TensorBox is an FV profile; the simplex realizer refuses it.
    let profile = MeshProfile::TensorBox(spec);
    assert!(matches!(
        realize(&profile),
        Err(finitum::FinitumError::MeshProfileUnsupported(_))
    ));
    assert_eq!(
        FvTopology::from_profile(&profile).unwrap().digest(),
        plane.digest()
    );
    for bad in [
        TensorBox { axes: vec![] },
        TensorBox {
            axes: vec![TensorAxis::graded(vec![0.0, 0.0, 1.0])],
        },
        TensorBox {
            axes: vec![TensorAxis::uniform(0.0, 1.0, 0)],
        },
    ] {
        assert!(FvTopology::from_tensor_box(&bad).is_err());
    }
}

#[test]
fn simplex_topology_invariants_hold_in_one_two_and_three_dimensions() {
    for (dimension, subdivisions) in [(1u8, vec![5]), (2, vec![3, 2]), (3, vec![2, 2, 2])] {
        let extent = (0..dimension)
            .map(|axis| [0.0, 1.0 + axis as f64])
            .collect::<Vec<_>>();
        let profile = MeshProfile::SimplexBox {
            dimension,
            extent: extent.clone(),
            subdivisions,
        };
        let mesh = realize(&profile).unwrap();
        let topology = FvTopology::from_profile(&profile).unwrap();
        assert_eq!(topology.shape(), FvCellShape::Simplex);
        assert_eq!(topology.source_digest(), &mesh.digest);
        let report = topology.check_invariants().unwrap();
        assert_eq!(report.faces_per_cell, usize::from(dimension) + 1);
        assert!(report.max_closure_defect <= report.closure_bound);
        let total_volume = topology.cell_volumes().iter().sum::<f64>();
        let box_volume = extent.iter().map(|[a, b]| b - a).product::<f64>();
        assert!((total_volume - box_volume).abs() < 1e-12);
        // Each interior face is listed once, with its normal pointing minus -> plus.
        let interior = topology.interior();
        let d = usize::from(dimension);
        for face in 0..interior.len() {
            let along = (0..d)
                .map(|axis| interior.normal[axis][face] * interior.delta[face * d + axis])
                .sum::<f64>();
            assert!(along > 0.0);
        }
        let mut pairs = (0..interior.len())
            .map(|face| (interior.minus[face], interior.plus[face]))
            .collect::<Vec<_>>();
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs.len(), interior.len());
        // Boundary area per box face equals that face's measure.
        for (tag, block) in topology.boundary() {
            let axis = ["x", "y", "z"]
                .iter()
                .position(|name| tag.as_str().starts_with(name))
                .unwrap();
            let expected = (0..d)
                .filter(|other| *other != axis)
                .map(|other| extent[other][1] - extent[other][0])
                .product::<f64>();
            assert!(
                (block.area.iter().sum::<f64>() - expected).abs() < 1e-12,
                "{tag}"
            );
        }
    }
}

/// The lowered LLF flux for `F = a u` in 1-D, in the lowering's own operation order
/// (Scientia `lowered_llf_equals_the_hand_formula_bitwise`).
fn hand_llf(a: f64, um: f64, up: f64, n: f64) -> f64 {
    let fm = (a * um) * n;
    let fp = (a * up) * n;
    let speed = (a * n).abs();
    0.5 * (fm + fp) - (0.5 * speed.max(speed)) * (up - um)
}

#[test]
fn residual_equals_a_hand_written_first_order_upwind_scheme() {
    let a = 1.75;
    let cells = 9;
    let topology = line(cells, true);
    let operator = realization(
        "PeriodicAdvection1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &topology,
        &[("a", a)],
    )
    .unwrap();
    let state = values(cells, 7);
    let rate = values(cells, 11);
    let mut residual = vec![0.0; cells];
    DaeOperator::residual(&operator, &cx(), 0.0, &state, &rate, &mut residual).unwrap();

    // Hand scheme, same op order: R_i = h·u̇_i, then faces in order (interior i|i+1, then the
    // periodic face last|first): owner += F, neighbour -= F (area 1 in 1-D).
    let volumes = topology.cell_volumes();
    let h = 1.0 / cells as f64;
    assert!(volumes.iter().all(|v| (v - h).abs() <= 2.0 * f64::EPSILON));
    let mut hand = rate
        .iter()
        .zip(volumes)
        .map(|(r, v)| v * r)
        .collect::<Vec<_>>();
    let mut faces = (0..cells - 1).map(|i| (i, i + 1)).collect::<Vec<_>>();
    faces.push((cells - 1, 0));
    for (minus, plus) in faces {
        let flux = 1.0 * hand_llf(a, state[minus], state[plus], 1.0);
        hand[minus] += flux;
        hand[plus] -= flux;
    }
    for (observed, expected) in residual.iter().zip(&hand) {
        assert!(same(*observed, *expected), "{observed} vs {expected}");
    }

    // Against pure upwind `F = a u_minus` (a > 0): LLF equals upwind in exact arithmetic;
    // basis: each LLF flux rounds within 4ε|a|max|u| of upwind, and each cell adds 3 terms.
    let scale = state.iter().fold(0.0_f64, |m, u| m.max(u.abs())) * a + h;
    for i in 0..cells {
        let left = (i + cells - 1) % cells;
        let upwind = volumes[i] * rate[i] + a * state[i] - a * state[left];
        assert!((residual[i] - upwind).abs() <= 16.0 * f64::EPSILON * scale);
    }

    // Explicit rhs is -(Σ flux)/h.
    let mut rhs = vec![0.0; cells];
    ExplicitOperator::rhs(&operator, &cx(), 0.0, &state, &mut rhs).unwrap();
    let mut zero_rate = vec![0.0; cells];
    DaeOperator::residual(
        &operator,
        &cx(),
        0.0,
        &state,
        &vec![0.0; cells],
        &mut zero_rate,
    )
    .unwrap();
    for ((r, z), v) in rhs.iter().zip(&zero_rate).zip(volumes) {
        assert!(same(*r, -*z / v));
    }

    // CFL: λ = |a| on both faces of each cell, so the step is min_i |Ω_i| / (2|a|).
    let cfl = operator.cfl(0.0, &state).unwrap();
    let expected = volumes
        .iter()
        .map(|v| v / (a + a))
        .fold(f64::INFINITY, f64::min);
    assert_eq!(cfl.max_face_speed, a);
    assert_eq!(cfl.stable_step, expected);
    assert_eq!(
        ExplicitOperator::stable_step(&operator, &cx(), 0.0, &state).unwrap(),
        Some(expected)
    );
}

#[test]
fn ghost_and_prescribed_boundaries_replace_the_face_flux() {
    let a = 0.5;
    let cells = 4;
    let topology = line(cells, false);
    let operator = realization(
        "InflowAdvection1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &topology,
        &[("a", a)],
    )
    .unwrap();
    let state = vec![0.2, 0.4, -0.3, 0.8];
    let fluxes = operator.face_fluxes(0.0, &state).unwrap();
    // x_min: ghost u = 1 on the plus side, outward normal -1.
    assert!(same(
        fluxes.boundary[&RegionTagId::new("x_min")][0],
        hand_llf(a, state[0], 1.0, -1.0)
    ));
    // x_max: the prescribed outward flux a·u replaces the face flux.
    assert!(same(
        fluxes.boundary[&RegionTagId::new("x_max")][0],
        a * state[3]
    ));
    let mut residual = vec![0.0; cells];
    DaeOperator::residual(&operator, &cx(), 0.0, &state, &[0.0; 4], &mut residual).unwrap();
    // Upwind with inflow 1 and outflow a·u: R_i = a u_i − a u_{i−1} up to LLF rounding.
    let upstream = [1.0, state[0], state[1], state[2]];
    for i in 0..cells {
        assert!((residual[i] - (a * state[i] - a * upstream[i])).abs() <= 1e-15);
    }
    let receipt = operator.conservation_receipt(0.0, &state).unwrap();
    assert!(receipt.balanced, "{receipt:?}");
    assert!((receipt.boundary_flux_sum - (a * state[3] - a)).abs() <= 1e-15);

    // Mass rate declared to Methodus is the net inflow.
    let mut rate = [0.0];
    assert!(ExplicitOperator::invariant_rates(&operator, &cx(), 0.0, &state, &mut rate).unwrap());
    assert!((rate[0] - (a - a * state[3])).abs() <= 1e-15);
}

fn mutated_receipt(
    topology: &FvTopology,
    fluxes: &FvFaceFluxes,
    leak: f64,
) -> FvConservationReceipt {
    // A deliberately non-conservative scatter: the neighbour loses only (1 - leak)·|f|F*.
    let mut residual = vec![0.0; topology.cell_count()];
    for (block, values) in [
        (topology.interior(), &fluxes.interior),
        (topology.periodic(), &fluxes.periodic),
    ] {
        for face in 0..block.len() {
            let term = block.area[face] * values[face];
            residual[block.minus[face]] += term;
            residual[block.plus[face]] -= (1.0 - leak) * term;
        }
    }
    for (block, values) in topology.boundary().values().zip(fluxes.boundary.values()) {
        for face in 0..block.len() {
            residual[block.minus[face]] += block.area[face] * values[face];
        }
    }
    discrete_flux_balance(topology, fluxes, &residual).unwrap()
}

#[test]
fn conservation_receipt_passes_and_a_non_conservative_scatter_fails_it() {
    let cells = 32;
    let topology = line(cells, true);
    let operator = realization(
        "Burgers1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &topology,
        &[("k", 0.3)],
    )
    .unwrap();
    let state = values(cells, 3).iter().map(|v| 1.0 + v).collect::<Vec<_>>();
    let receipt = operator.conservation_receipt(0.0, &state).unwrap();
    assert!(receipt.balanced, "{receipt:?}");
    assert_eq!(receipt.boundary_flux_sum, 0.0);
    assert!(receipt.relative_imbalance <= receipt.relative_bound);

    let fluxes = operator.face_fluxes(0.0, &state).unwrap();
    let mut honest = vec![0.0; cells];
    scatter_face_fluxes(operator.topology(), &fluxes, &mut honest).unwrap();
    assert!(
        discrete_flux_balance(operator.topology(), &fluxes, &honest)
            .unwrap()
            .balanced
    );
    for leak in [1e-3, 1e-9] {
        let broken = mutated_receipt(operator.topology(), &fluxes, leak);
        assert!(!broken.balanced, "leak {leak}: {broken:?}");
        assert!(broken.relative_imbalance > 1e3 * broken.relative_bound);
    }

    // 2-D: free stream is preserved to round-off (a cell's x and y terms interleave in face
    // order, so cancellation is not exact; basis: 4 terms of size ≤ |F|·|f| ≤ 1 per cell, over
    // volumes ≥ 0.04) and the receipt balances on a random state.
    let plane = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![
                TensorAxis::uniform(0.0, 1.0, 5).periodic(),
                TensorAxis::graded(vec![0.0, 0.2, 0.7, 1.0]).periodic(),
            ],
        })
        .unwrap(),
    );
    let operator = realization(
        "PeriodicAdvection2D",
        NumericalFluxIntent::Hll,
        &plane,
        &[("ax", 1.0), ("ay", -0.5)],
    )
    .unwrap();
    let mut rhs = vec![1.0; 15];
    ExplicitOperator::rhs(&operator, &cx(), 0.0, &[0.7; 15], &mut rhs).unwrap();
    assert!(
        rhs.iter()
            .all(|value| value.abs() <= 4.0 * f64::EPSILON / 0.04),
        "{rhs:?}"
    );
    let state = values(15, 5);
    assert!(operator.conservation_receipt(0.0, &state).unwrap().balanced);
    let fluxes = operator.face_fluxes(0.0, &state).unwrap();
    assert!(!mutated_receipt(operator.topology(), &fluxes, 1e-6).balanced);
}

#[test]
fn explicit_periodic_advection_conserves_mass_to_the_methodus_bound() {
    let cells = 64;
    let topology = line(cells, true);
    let operator = realization(
        "PeriodicAdvection1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &topology,
        &[("a", 1.0)],
    )
    .unwrap();
    let initial = (0..cells)
        .map(|i| {
            let x = (i as f64 + 0.5) / cells as f64;
            if (0.25..0.5).contains(&x) { 1.0 } else { 0.1 }
        })
        .collect::<Vec<_>>();
    let mass = |values: &[f64]| values.iter().map(|v| v / cells as f64).sum::<f64>();
    for scheme in [ExplicitScheme::ForwardEuler, ExplicitScheme::SspRk3] {
        let state = ExplicitState::initialize(&operator, &cx(), 0.0, initial.clone()).unwrap();
        let config = ExplicitConfig {
            scheme,
            step_policy: StepPolicy::Cfl {
                target: 0.9,
                max: 1.0,
                min_step: 1e-9,
                max_step: 1.0,
            },
        };
        let report = integrate_explicit(&operator, &cx(), &state, 0.5, &config, 10_000).unwrap();
        assert!(report.rejections.is_empty());
        let k = (cells + 4 * scheme.stages()) as f64;
        for step in &report.steps {
            assert!(step.balance_corrected);
            let balance = &step.invariants[0];
            assert!(
                balance.balance_residual.abs() <= k * f64::EPSILON * balance.scale,
                "{scheme:?}: {balance:?}"
            );
            assert!(step.cfl_realized.unwrap() <= 1.0);
        }
        let end = &report.state.values;
        assert!((mass(end) - mass(&initial)).abs() <= 1e-13);
        // First-order SSP upwind is TVD and keeps the data in [0.1, 1].
        assert!(end.iter().all(|v| (0.1 - 1e-12..=1.0 + 1e-12).contains(v)));
    }
}

#[test]
fn jvp_is_exact_for_smooth_fluxes_and_refused_through_flux_kinks() {
    let cells = 12;
    let topology = line(cells, false);
    let state = values(cells, 13);
    let direction = values(cells, 17);
    let rate_direction = values(cells, 19);
    let fd = |operator: &FvRealization, step: f64| {
        let shifted = |sign: f64| {
            let y = state
                .iter()
                .zip(&direction)
                .map(|(y, d)| y + sign * step * d)
                .collect::<Vec<_>>();
            let r = rate_direction
                .iter()
                .map(|d| sign * step * d)
                .collect::<Vec<_>>();
            let mut out = vec![0.0; cells];
            DaeOperator::residual(operator, &cx(), 0.0, &y, &r, &mut out).unwrap();
            out
        };
        let (plus, minus) = (shifted(1.0), shifted(-1.0));
        plus.iter()
            .zip(&minus)
            .map(|(p, m)| (p - m) / (2.0 * step))
            .collect::<Vec<_>>()
    };
    let jvp = |operator: &FvRealization| -> Result<Vec<f64>, NumericError> {
        let mut out = vec![0.0; cells];
        DaeOperator::jacobian_vector_product(
            operator,
            &cx(),
            0.0,
            &state,
            &vec![0.0; cells],
            &direction,
            &rate_direction,
            &mut out,
        )?;
        Ok(out)
    };

    // Linear advection with LLF: the |a| switch reads only data, so the state JVP is smooth
    // (and the operator is linear: central differences are exact up to rounding).
    let advection = realization(
        "InflowAdvection1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &topology,
        &[("a", 0.8)],
    )
    .unwrap();
    for (j, f) in jvp(&advection).unwrap().iter().zip(fd(&advection, 0.5)) {
        assert!((j - f).abs() <= 1e-14, "{j} vs {f}");
    }
    let dispositions = advection
        .receipt()
        .kernels
        .iter()
        .map(|kernel| kernel.derivative.clone())
        .collect::<Vec<_>>();
    // Face flux, then the CFL speed (not part of the residual), then both boundary kernels.
    assert_eq!(
        dispositions,
        [
            FvDerivativeDisposition::Smooth,
            FvDerivativeDisposition::NotDifferentiated,
            FvDerivativeDisposition::Smooth,
            FvDerivativeDisposition::Smooth,
        ]
    );

    // Central Burgers is smooth: the JVP matches central differences to O(h²).
    let ring = line(cells, true);
    let central = realization(
        "Burgers1D",
        NumericalFluxIntent::Central,
        &ring,
        &[("k", 0.3)],
    )
    .unwrap();
    for (j, f) in jvp(&central).unwrap().iter().zip(fd(&central, 1e-5)) {
        assert!((j - f).abs() <= 1e-8, "{j} vs {f}");
    }

    // LLF Burgers: λ = max(|u⁻|, |u⁺|) is a kink in the state, so the JVP is a typed refusal
    // (D6) while the primal residual still evaluates.
    let llf = realization(
        "Burgers1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &ring,
        &[("k", 0.3)],
    )
    .unwrap();
    let error = jvp(&llf).unwrap_err();
    assert_eq!(
        error.evaluation_code(),
        Some("FV_NONSMOOTH_DERIVATIVE_REFUSED"),
        "{error}"
    );
    assert_eq!(
        llf.receipt().kernels[0].derivative,
        FvDerivativeDisposition::Refused {
            code: "FV_NONSMOOTH_DERIVATIVE_REFUSED".into()
        }
    );
    let mut out = vec![0.0; cells];
    DaeOperator::residual(&llf, &cx(), 0.0, &state, &vec![0.0; cells], &mut out).unwrap();
}

#[test]
fn implicit_bdf_steps_the_fv_operator() {
    let cells = 16;
    let topology = line(cells, true);
    let operator = realization(
        "Burgers1D",
        NumericalFluxIntent::Central,
        &topology,
        &[("k", 0.3)],
    )
    .unwrap();
    let initial = (0..cells)
        .map(|i| 1.0 + 0.1 * (2.0 * std::f64::consts::PI * (i as f64 + 0.5) / cells as f64).sin())
        .collect::<Vec<_>>();
    let state = BdfState::initialize(&operator, &cx(), 0.0, initial.clone()).unwrap();
    let config = BdfConfig {
        order: BdfOrder::One,
        ..BdfConfig::default()
    };
    let StepOutcome::Accepted(step) = bdf_step(&operator, &cx(), &state, 1e-3, &config).unwrap()
    else {
        panic!("BDF1 step rejected");
    };
    // Mass decays by the source alone: d/dt Σ h u = -k Σ h u, so BDF1 gives m/(1 + kΔt).
    let mass = |values: &[f64]| values.iter().sum::<f64>() / cells as f64;
    let expected = mass(&initial) / (1.0 + 0.3 * 1e-3);
    assert!((mass(&step.state.values) - expected).abs() <= 1e-9);
}

#[test]
fn realization_identity_and_binding_refusals() {
    let ring = line(8, true);
    let one = realization("Burgers1D", NumericalFluxIntent::Hll, &ring, &[("k", 0.3)]).unwrap();
    let two = realization("Burgers1D", NumericalFluxIntent::Hll, &ring, &[("k", 0.3)]).unwrap();
    assert_eq!(one.digest(), two.digest());
    let other = realization("Burgers1D", NumericalFluxIntent::Hll, &ring, &[("k", 0.4)]).unwrap();
    assert_ne!(one.digest(), other.digest());
    let receipt = one.receipt();
    assert_eq!(receipt.schema, "finitum-fv-realization/1");
    assert_eq!(&receipt.topology_digest, ring.digest());
    assert_eq!(&receipt.program_digest, &one.program().artifact_digest);

    let code = |result: Result<FvRealization, finitum::FinitumError>| {
        let message = result.unwrap_err().to_string();
        message
            .split(':')
            .find(|part| part.trim_start().starts_with("FV_"))
            .map(|part| part.trim().to_string())
            .unwrap_or(message)
    };
    let plane = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![
                TensorAxis::uniform(0.0, 1.0, 2),
                TensorAxis::uniform(0.0, 1.0, 2),
            ],
        })
        .unwrap(),
    );
    assert_eq!(
        code(realization(
            "Burgers1D",
            NumericalFluxIntent::Hll,
            &plane,
            &[("k", 0.3)]
        )),
        "FV_DIMENSION_MISMATCH"
    );
    assert_eq!(
        code(realization(
            "Burgers1D",
            NumericalFluxIntent::Hll,
            &ring,
            &[]
        )),
        "FV_DATUM_MISSING"
    );
    assert_eq!(
        code(realization(
            "Burgers1D",
            NumericalFluxIntent::Hll,
            &ring,
            &[("k", 0.3), ("q", 1.0)]
        )),
        "FV_DATUM_UNKNOWN"
    );
    // A bounded line under a law without boundaries leaves its tags unclosed.
    assert_eq!(
        code(realization(
            "Burgers1D",
            NumericalFluxIntent::Hll,
            &line(4, false),
            &[("k", 0.3)]
        )),
        "FV_BOUNDARY_UNCLOSED"
    );
    // A periodic ring has no `x_min`/`x_max` tags for the declared boundaries.
    assert_eq!(
        code(realization(
            "InflowAdvection1D",
            NumericalFluxIntent::Hll,
            &ring,
            &[("a", 1.0)]
        )),
        "FV_BOUNDARY_TAG_UNKNOWN"
    );
    // MUSCL reconstruction is SV2-C2b.
    let module = module();
    let system = compile_conservation_law_system(&module, "Burgers1D", "balance", "u").unwrap();
    let muscl = FvMethodIntent {
        reconstruction: ReconstructionIntent::Muscl {
            variables: ReconstructionVariables::Conserved,
            gradient: GradientScheme::Axis1D,
            limiter: LimiterFamily::Minmod,
        },
        ..intent(NumericalFluxIntent::LocalLaxFriedrichs)
    };
    let program = lower_conservation_law(&module, &system, &muscl, &[]).unwrap();
    assert_eq!(
        code(FvRealization::new(
            &system,
            Arc::new(program),
            ring.clone(),
            &RegionMap::new(),
            &BTreeMap::from([("k".to_string(), 0.3)]),
        )),
        "FV_RECONSTRUCTION_UNSUPPORTED"
    );
}
