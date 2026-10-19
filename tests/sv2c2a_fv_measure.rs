//! SV2-C2a: repeated FV evaluations over the reused scratch are bitwise deterministic, and the
//! per-face cost of the batched face loop is printed (method and numbers in STATUS.md).
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use finitum::{FvRealization, FvTopology, RegionMap, RegionTagId, TensorAxis, TensorBox};
use methodus::{DaeOperator, EvaluationContext, ExplicitOperator};
use quantitas::UnitRegistry;
use scientia::{
    BoundaryFluxSemantics, FvMethodIntent, NumericalFluxIntent, PositivityIntent,
    ReconstructionIntent, SourceTreatment, compile_conservation_law_system, compile_semantics,
    lower_conservation_law,
};

#[test]
fn repeated_evaluation_is_deterministic_and_measured() {
    let module = compile_semantics(
        include_str!("fixtures/fv-scalar.res"),
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap()
    .semantic;
    let system =
        compile_conservation_law_system(&module, "InflowAdvection1D", "balance", "u").unwrap();
    let intent = FvMethodIntent {
        numerical_flux: NumericalFluxIntent::LocalLaxFriedrichs,
        reconstruction: ReconstructionIntent::Constant,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    };
    let program = lower_conservation_law(&module, &system, &intent, &[]).unwrap();
    let mut regions = RegionMap::new();
    for boundary in &system.boundaries {
        let (BoundaryFluxSemantics::GhostState { region, .. }
        | BoundaryFluxSemantics::PrescribedFlux { region, .. }) = boundary;
        let tag = if boundary.region_name() == "left" {
            "x_min"
        } else {
            "x_max"
        };
        regions.insert(*region, [RegionTagId::new(tag)]);
    }
    let cells = 100_000;
    let topology = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![TensorAxis::uniform(0.0, 1.0, cells)],
        })
        .unwrap(),
    );
    let operator = FvRealization::new(
        &system,
        Arc::new(program),
        topology,
        &regions,
        &BTreeMap::from([("a".to_string(), 0.8)]),
    )
    .unwrap();
    let cx = EvaluationContext::reproducible();
    let state = (0..cells)
        .map(|i| (i as f64 * 1e-3).sin())
        .collect::<Vec<_>>();
    let direction = (0..cells)
        .map(|i| (i as f64 * 7e-3).cos())
        .collect::<Vec<_>>();
    let zero = vec![0.0; cells];
    let mut output = vec![0.0; cells];
    let mut rate = [0.0];
    let evaluate = |output: &mut [f64], rate: &mut [f64]| {
        let mut all = Vec::new();
        DaeOperator::residual(&operator, &cx, 0.0, &state, &zero, output).unwrap();
        all.extend_from_slice(output);
        DaeOperator::jacobian_vector_product(
            &operator, &cx, 0.0, &state, &zero, &direction, &zero, output,
        )
        .unwrap();
        all.extend_from_slice(output);
        ExplicitOperator::rhs(&operator, &cx, 0.0, &state, output).unwrap();
        all.extend_from_slice(output);
        all.push(
            ExplicitOperator::stable_step(&operator, &cx, 0.0, &state)
                .unwrap()
                .unwrap(),
        );
        ExplicitOperator::invariant_rates(&operator, &cx, 0.0, &state, rate).unwrap();
        all.push(rate[0]);
        all.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
    };
    let first = evaluate(&mut output, &mut rate);
    assert_eq!(evaluate(&mut output, &mut rate), first);

    let repeats = 20;
    let start = Instant::now();
    for _ in 0..repeats {
        DaeOperator::residual(&operator, &cx, 0.0, &state, &zero, &mut output).unwrap();
    }
    let faces = (cells + 1) as f64;
    println!(
        "FV residual, 1-D LLF, {cells} cells: {:.2} ns/face",
        start.elapsed().as_nanos() as f64 / (repeats as f64 * faces)
    );
}
