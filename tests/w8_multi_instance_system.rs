//! W8 lane F-MI: the multi-instance `SystemRealizationPlan` (`sinbad/ARCHITECTURE.md` §2.3,
//! §2.6, §6, §8; GX-CONTRACTS C12.9 "lane A2 exact need", items 1-4).
//!
//! The two-instance `Electrothermal` system (`physics.electrical` + `physics.thermal` +
//! `systems.electrothermal`, the Sinbad module snapshots under `fixtures/corpus/modules/`)
//! realized as ONE plan on a shared mesh is compared block by block with the monolithic
//! `08-electrothermal-joule` corpus model realized as a one-instance plan: residuals, full
//! JVPs, every `(row, column)` block (both cross blocks through `Q = joule_heat`), transposes,
//! `linearize`/`assemble`, receipts, keyed essential constraints and typed failures.

use finitum::{
    BindPath, BlockLayout, CoefficientLayout, DistributedCoefficient, ExternalInput, FieldSource,
    FinitumError, InputEvaluationError, InputOrigin, InstanceId, MeshProfile, PointEvaluation,
    RegionMap, RegionTagId, SYSTEM_REALIZATION_COMPOSED_DIGEST_SCHEMA, SysResId, SysVarId,
    SystemConstitutiveInput, SystemDistributedCoefficient, SystemEssentialConstraintRequirement,
    SystemExternalInput, SystemOperator, SystemQuadrature, SystemRealizationPlan,
    SystemVariableEssentialConstraint, TaggedMesh, essential_constraints_from_system,
    essential_constraints_from_system_by_variable_at, realize,
};
use methodus::{LinearOperator, OperatorStructureHint, OperatorSymmetry};
use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, IntegralOperatorFactorization, Registries,
    SemanticModel, SymbolId, SystemOperatorCompilation, TensorInputRole, compile_operator_system,
    compile_semantics, compile_system, compile_system_operator, resolve_module_closure,
};
use std::collections::BTreeMap;

const MONOLITHIC_08: &str = include_str!("fixtures/corpus/08-electrothermal-joule.res");
const ELECTRICAL: &str = include_str!("fixtures/corpus/modules/physics.electrical.res");
const THERMAL: &str = include_str!("fixtures/corpus/modules/physics.thermal.res");
const ELECTROTHERMAL: &str = include_str!("fixtures/corpus/modules/systems.electrothermal.res");

const TWO_HEAT: &str = r#"
module systems.two_heat;
use physics.thermal.{HeatConduction};

pub system TwoHeat {
    domain body { dimension = 2; coordinates = cartesian; }
    instance a: HeatConduction(body = body);
    instance b: HeatConduction(body = body);
}
"#;

const AGREEMENT: f64 = 1.0e-10;
const IDENTITY: f64 = 1.0e-12;
const TIME: f64 = 0.3;
const RATE_SHIFT: f64 = 0.7;

// The case bindings of `cases/electrothermal-system.toml` / `08-electrothermal-joule.toml`.
fn sigma(t: f64) -> f64 {
    1.0 / (1.0 + 0.004 * (t - 300.0))
}
fn d_sigma(t: f64, dt: f64) -> f64 {
    -0.004 * dt / ((1.0 + 0.004 * (t - 300.0)) * (1.0 + 0.004 * (t - 300.0)))
}
fn conductivity(t: f64) -> f64 {
    1.0 + 0.01 * (t - 300.0)
}
fn d_conductivity(dt: f64) -> f64 {
    0.01 * dt
}
const RHO: f64 = 1.0;
const CP: f64 = 1.0;

fn registries() -> (UnitRegistry, QuantityKindRegistry) {
    (
        UnitRegistry::si_bootstrap(),
        QuantityKindRegistry::si_bootstrap(),
    )
}

fn modules() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("physics.electrical".to_string(), ELECTRICAL.to_string()),
        ("physics.thermal".to_string(), THERMAL.to_string()),
    ])
}

struct Composed {
    compilation: scientia::SystemCompilation,
    operator: SystemOperatorCompilation,
}

fn compile_composed(root: &str, name: &str) -> Composed {
    let closure = resolve_module_closure(root, &modules()).unwrap();
    let (units, kinds) = registries();
    let compilation = compile_system(&closure, Registries::new(&units, &kinds), name).unwrap();
    let operator = compile_system_operator(&compilation).unwrap();
    Composed {
        compilation,
        operator,
    }
}

fn unit_square(subdivisions: usize) -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

fn symbol(model: &SemanticModel, name: &str) -> SymbolId {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name)
        .map(|symbol| symbol.id)
        .unwrap_or_else(|| panic!("model {} has no symbol {name}", model.name))
}

fn symbol_name(model: &SemanticModel, symbol: SymbolId) -> &str {
    &model.symbols[symbol.index()].name
}

/// The active input of `symbol` with evaluation kind `derivative` in `integral`.
fn active_input(
    integral: &IntegralOperatorFactorization,
    symbol: SymbolId,
    derivative: DerivativeEvaluation,
) -> scientia::TensorInputId {
    integral
        .primal
        .inputs
        .iter()
        .find(|input| {
            input.source == InputSourceRequirement::Basis
                && input.role == TensorInputRole::Active
                && input.binding.symbol == symbol
                && input.binding.evaluation.derivative == derivative
        })
        .map(|input| input.id)
        .unwrap_or_else(|| panic!("no {derivative:?} active input for {symbol}"))
}

fn probe_vector(dimension: usize, seed: f64, scale: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| scale * ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Asserts agreement and reports the measured worst relative discrepancy (`--nocapture`).
fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: extent");
    let mut worst = 0.0_f64;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let scale = expected.abs().max(1.0);
        let discrepancy = (actual - expected).abs() / scale;
        worst = worst.max(discrepancy);
        assert!(
            discrepancy <= tolerance,
            "{what} at {index}: {actual} != {expected} (tolerance {tolerance})"
        );
    }
    eprintln!(
        "agreement {what}: worst relative discrepancy {worst:.3e} (max |expected| {:.3e})",
        max_abs(expected)
    );
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()))
}

// ---------------------------------------------------------------------------------------------
// The monolithic 08 model, hand-closed exactly as Sinbad's dual closures close it.
// ---------------------------------------------------------------------------------------------

struct Monolithic {
    operator: SystemOperator,
    v: SymbolId,
    t: SymbolId,
}

fn monolithic_08(tagged: &TaggedMesh, quadrature: SystemQuadrature) -> Monolithic {
    let compilation = compile_semantics(MONOLITHIC_08, &UnitRegistry::si_bootstrap()).unwrap();
    let model = compilation.semantic.models[0].clone();
    let system = compile_operator_system(
        &compilation.semantic,
        "ElectrothermalJoule",
        &["electrical", "thermal"],
    )
    .unwrap();
    let (v, t) = (symbol(&model, "V"), symbol(&model, "T"));
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = symbol_name(&model, input.binding.symbol).to_string();
                let equation = block.equation.clone();
                let index = integral.integral_index;
                let built = match name.as_str() {
                    "current_density" => {
                        // `-sigma(T) grad V`, the constitutive law as one opaque vector input.
                        let t_in = active_input(integral, t, DerivativeEvaluation::Value);
                        let gv_in = active_input(integral, v, DerivativeEvaluation::Gradient);
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            2,
                            "mono/current_density",
                            move |point: &PointEvaluation| {
                                let t = point.input_values(t_in).unwrap()[0];
                                let gv = point.input_values(gv_in).unwrap();
                                gv.iter().map(|g| -sigma(t) * g).collect()
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                let t = point.input_values(t_in).unwrap()[0];
                                let dt = direction.input_values(t_in).unwrap()[0];
                                let gv = point.input_values(gv_in).unwrap();
                                let dgv = direction.input_values(gv_in).unwrap();
                                gv.iter()
                                    .zip(dgv)
                                    .map(|(g, dg)| -(d_sigma(t, dt) * g + sigma(t) * dg))
                                    .collect()
                            },
                        )
                    }
                    "k" => {
                        let t_in = active_input(integral, t, DerivativeEvaluation::Value);
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "mono/k",
                            move |point: &PointEvaluation| {
                                vec![conductivity(point.input_values(t_in).unwrap()[0])]
                            },
                            move |_: &PointEvaluation, direction: &PointEvaluation| {
                                vec![d_conductivity(direction.input_values(t_in).unwrap()[0])]
                            },
                        )
                    }
                    "rho" | "cp" => {
                        let value = if name == "rho" { RHO } else { CP };
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            format!("mono/{name}"),
                            move |_: &PointEvaluation| vec![value],
                            |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                        )
                    }
                    "joule" => {
                        let t_in = active_input(integral, t, DerivativeEvaluation::Value);
                        let gv_in = active_input(integral, v, DerivativeEvaluation::Gradient);
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "mono/joule",
                            move |point: &PointEvaluation| {
                                let t = point.input_values(t_in).unwrap()[0];
                                let gv = point.input_values(gv_in).unwrap();
                                vec![sigma(t) * dot(gv, gv)]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                let t = point.input_values(t_in).unwrap()[0];
                                let dt = direction.input_values(t_in).unwrap()[0];
                                let gv = point.input_values(gv_in).unwrap();
                                let dgv = direction.input_values(gv_in).unwrap();
                                vec![d_sigma(t, dt) * dot(gv, gv) + 2.0 * sigma(t) * dot(gv, dgv)]
                            },
                        )
                    }
                    other => panic!("unexpected monolithic non-basis input {other}"),
                };
                constitutive.push(built.unwrap());
            }
        }
    }
    let vertices = tagged.mesh.vertices().len();
    let layout = BlockLayout::new([(v, vertices, 1), (t, vertices, 1)]).unwrap();
    let plan =
        SystemRealizationPlan::with_quadrature(system, tagged.mesh.clone(), layout, quadrature)
            .unwrap();
    let operator = plan.bind_kernels(constitutive, BTreeMap::new()).unwrap();
    Monolithic { operator, v, t }
}

// ---------------------------------------------------------------------------------------------
// The composed two-instance system: closures per instance, keyed by residual / output.
// ---------------------------------------------------------------------------------------------

struct ComposedRealization {
    composed: Composed,
    operator: SystemOperator,
    electrical: InstanceId,
    thermal: InstanceId,
    v: SysVarId,
    t: SysVarId,
    electrical_row: SysResId,
    thermal_row: SysResId,
}

/// Every closure of the composed electrothermal system; `refuse` optionally makes one of them
/// (the thermal `k` on the row path, or the electrical output's `sigma`) refuse typed at the
/// first point of `refuse_cell`.
fn composed_closures(
    composed: &Composed,
    plan: &SystemRealizationPlan,
    refuse: Option<(&'static str, usize)>,
) -> Vec<SystemConstitutiveInput> {
    let ids = plan.system_ids();
    let system = &composed.compilation.system;
    let mut closures = Vec::new();
    for residual in ids.residuals() {
        let model = composed
            .compilation
            .model(scientia::InstanceId(residual.instance.0));
        let system_1 = plan.instance_system(residual.instance).unwrap();
        let block = system_1
            .blocks
            .iter()
            .find(|block| block.equation == residual.equation)
            .unwrap();
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = symbol_name(model, input.binding.symbol).to_string();
                let index = integral.integral_index;
                let built = match name.as_str() {
                    "current_density" => {
                        // The provider-input path: `sigma(temperature)` with `temperature`
                        // bound to `thermal.temperature`, read as `PointEvaluation::bound`.
                        let temperature = symbol(model, "temperature");
                        let gv_in = active_input(
                            integral,
                            symbol(model, "V"),
                            DerivativeEvaluation::Gradient,
                        );
                        SystemConstitutiveInput::try_new_for_residual(
                            residual.id,
                            index,
                            input.id,
                            2,
                            "composed/electrical/current_density",
                            move |point: &PointEvaluation| {
                                let t = point.bound_values(temperature).ok_or_else(|| {
                                    InputEvaluationError::new(
                                        "TEST_UNBOUND",
                                        InputOrigin::Slot("electrical/input/temperature".into()),
                                        "no bound temperature at the point",
                                    )
                                })?[0];
                                let gv = point.input_values(gv_in).unwrap();
                                Ok(gv.iter().map(|g| -sigma(t) * g).collect())
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                let t = point.bound_values(temperature).unwrap()[0];
                                let dt = direction.bound_values(temperature).unwrap()[0];
                                let gv = point.input_values(gv_in).unwrap();
                                let dgv = direction.input_values(gv_in).unwrap();
                                Ok(gv
                                    .iter()
                                    .zip(dgv)
                                    .map(|(g, dg)| -(d_sigma(t, dt) * g + sigma(t) * dg))
                                    .collect())
                            },
                        )
                    }
                    "k" => {
                        let t_in =
                            active_input(integral, symbol(model, "T"), DerivativeEvaluation::Value);
                        let refuse_cell = match refuse {
                            Some(("k", cell)) => Some(cell),
                            _ => None,
                        };
                        SystemConstitutiveInput::try_new_for_residual(
                            residual.id,
                            index,
                            input.id,
                            1,
                            "composed/thermal/k",
                            move |point: &PointEvaluation| {
                                if refuse_cell == Some(point.cell.0) {
                                    return Err(InputEvaluationError::new(
                                        "RUN_PROPERTY_REFUSED",
                                        InputOrigin::Slot(
                                            "thermal/provider/thermal_conductivity".into(),
                                        ),
                                        "the test refuses this point",
                                    ));
                                }
                                Ok(vec![conductivity(point.input_values(t_in).unwrap()[0])])
                            },
                            move |_: &PointEvaluation, direction: &PointEvaluation| {
                                Ok(vec![d_conductivity(
                                    direction.input_values(t_in).unwrap()[0],
                                )])
                            },
                        )
                    }
                    "rho" | "cp" => {
                        let value = if name == "rho" { RHO } else { CP };
                        SystemConstitutiveInput::try_new_for_residual(
                            residual.id,
                            index,
                            input.id,
                            1,
                            format!("composed/thermal/{name}"),
                            move |_: &PointEvaluation| Ok(vec![value]),
                            |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
                        )
                    }
                    "Q" => continue, // closed by `bind thermal.Q <- electrical.joule_heat`
                    other => panic!("unexpected composed non-basis input {other}"),
                };
                closures.push(built.unwrap());
            }
        }
    }
    // The producer output `electrical.joule_heat` reads `sigma` (through the bound temperature).
    for output in &system.outputs {
        let model = composed.compilation.model(output.instance);
        let kernels = &composed.operator.output_kernels[output.id.index()];
        let integral = &kernels.factorization.integrals[0];
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = symbol_name(model, input.binding.symbol).to_string();
            assert_eq!(
                name, "sigma",
                "the only closed output input is joule_heat's sigma"
            );
            let temperature = symbol(model, "temperature");
            let refuse_cell = match refuse {
                Some(("output-sigma", cell)) => Some(cell),
                _ => None,
            };
            closures.push(
                SystemConstitutiveInput::try_new_for_output(
                    InstanceId(output.instance.0),
                    output.id,
                    integral.integral_index,
                    input.id,
                    1,
                    "composed/electrical.joule_heat/sigma",
                    move |point: &PointEvaluation| {
                        if refuse_cell == Some(point.cell.0) {
                            return Err(InputEvaluationError::new(
                                "RUN_PROPERTY_REFUSED",
                                InputOrigin::ExpressionPath("electrical.joule_heat.sigma".into()),
                                "the test refuses this point",
                            ));
                        }
                        let t = point.bound_values(temperature).unwrap()[0];
                        Ok(vec![sigma(t)])
                    },
                    move |point: &PointEvaluation, direction: &PointEvaluation| {
                        let t = point.bound_values(temperature).unwrap()[0];
                        let dt = direction.bound_values(temperature).unwrap()[0];
                        Ok(vec![d_sigma(t, dt)])
                    },
                )
                .unwrap(),
            );
        }
    }
    closures
}

fn composed_plan(
    composed: &Composed,
    tagged: &TaggedMesh,
    quadrature: SystemQuadrature,
) -> SystemRealizationPlan {
    let ids = finitum::SystemIdMap::from_scientia(&composed.operator.operator).unwrap();
    let vertices = tagged.mesh.vertices().len();
    let layout = BlockLayout::new_keyed(
        ids.variables()
            .iter()
            .map(|variable| (variable.id, variable.local, vertices, 1)),
    )
    .unwrap();
    SystemRealizationPlan::composed(&composed.operator, tagged.mesh.clone(), layout, quadrature)
        .unwrap()
}

fn composed_electrothermal(
    tagged: &TaggedMesh,
    quadrature: SystemQuadrature,
) -> ComposedRealization {
    let composed = compile_composed(ELECTROTHERMAL, "Electrothermal");
    let plan = composed_plan(&composed, tagged, quadrature);
    let closures = composed_closures(&composed, &plan, None);
    let operator = plan.bind_kernels(closures, BTreeMap::new()).unwrap();
    let ids = plan.system_ids().clone();
    let instance = |name: &str| {
        ids.instances()
            .iter()
            .find(|record| record.name == name)
            .map(|record| record.instance)
            .unwrap()
    };
    let (electrical, thermal) = (instance("electrical"), instance("thermal"));
    let v = ids
        .variable(
            electrical,
            symbol(
                composed
                    .compilation
                    .model(scientia::InstanceId(electrical.0)),
                "V",
            ),
        )
        .unwrap();
    let t = ids
        .variable(
            thermal,
            symbol(
                composed.compilation.model(scientia::InstanceId(thermal.0)),
                "T",
            ),
        )
        .unwrap();
    ComposedRealization {
        composed,
        operator,
        electrical,
        thermal,
        v,
        t,
        electrical_row: ids.residual(electrical, "electrical").unwrap(),
        thermal_row: ids.residual(thermal, "thermal").unwrap(),
    }
}

/// `OriginMap` permutation: composed index -> monolithic index (same P1 nodal maps, so the
/// blocks map by field name).
fn permutation(composed: &ComposedRealization, mono: &Monolithic) -> Vec<usize> {
    let composed_layout = composed.operator.layout();
    let mono_layout = mono.operator.layout();
    let mut map = vec![usize::MAX; composed_layout.extent()];
    for (variable, mono_symbol) in [(composed.v, mono.v), (composed.t, mono.t)] {
        let from = composed_layout.block_by_variable(variable).unwrap();
        let to = mono_layout.block(mono_symbol).unwrap();
        assert_eq!(from.extent, to.extent);
        for offset in 0..from.extent {
            map[from.offset + offset] = to.offset + offset;
        }
    }
    assert!(map.iter().all(|index| *index != usize::MAX));
    map
}

fn permute(values: &[f64], map: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for (from, to) in map.iter().enumerate() {
        out[*to] = values[from];
    }
    out
}

/// Sampled states in the monolithic layout: zero, a perturbation, a nontrivial state.
fn states(mono: &Monolithic) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
    let dimension = mono.operator.dimension();
    let layout = mono.operator.layout();
    let v_block = layout.block(mono.v).unwrap();
    let t_block = layout.block(mono.t).unwrap();
    let mesh = mono.operator.plan().mesh();
    let mut nontrivial = vec![0.0; dimension];
    for (vertex, coordinates) in mesh.vertices().iter().enumerate() {
        nontrivial[v_block.offset + vertex] = 1.0 - coordinates[0] + 0.2 * coordinates[1];
        nontrivial[t_block.offset + vertex] =
            300.0 + 20.0 * (3.0 * coordinates[0] + coordinates[1]).sin();
    }
    let mut perturbed = vec![0.0; dimension];
    for vertex in 0..mesh.vertices().len() {
        perturbed[v_block.offset + vertex] = 0.05 * ((vertex as f64) * 1.7).sin();
        perturbed[t_block.offset + vertex] = 300.0 + 0.5 * ((vertex as f64) * 0.9).cos();
    }
    let rate = probe_vector(dimension, 2.5, 3.0);
    vec![
        ("zero", vec![0.0; dimension], vec![0.0; dimension]),
        ("perturbed", perturbed, rate.clone()),
        ("nontrivial", nontrivial, rate),
    ]
}

#[test]
fn composed_electrothermal_matches_monolithic_08_per_block_and_cross_block() {
    for quadrature in [SystemQuadrature::Barycenter, SystemQuadrature::Richest] {
        let tagged = unit_square(3);
        let mono = monolithic_08(&tagged, quadrature);
        let composed = composed_electrothermal(&tagged, quadrature);
        let map = permutation(&composed, &mono);
        let inverse = {
            let mut inverse = vec![0; map.len()];
            for (from, to) in map.iter().enumerate() {
                inverse[*to] = from;
            }
            inverse
        };
        let dimension = mono.operator.dimension();
        assert_eq!(composed.operator.dimension(), dimension);

        let mono_rows = [SysResId(0), SysResId(1)];
        let mono_columns = [SysVarId(mono.v.0), SysVarId(mono.t.0)];
        let composed_rows = [composed.electrical_row, composed.thermal_row];
        let composed_columns = [composed.v, composed.t];

        for (label, state_m, rate_m) in states(&mono) {
            let state_c = permute(&state_m, &inverse);
            let rate_c = permute(&rate_m, &inverse);
            // Residual.
            let mut expected = vec![0.0; dimension];
            mono.operator
                .residual(TIME, &state_m, &rate_m, &mut expected)
                .unwrap();
            let mut actual = vec![0.0; dimension];
            composed
                .operator
                .residual(TIME, &state_c, &rate_c, &mut actual)
                .unwrap();
            assert_close(
                &permute(&actual, &map),
                &expected,
                AGREEMENT,
                &format!("{quadrature:?} residual at {label}"),
            );
            // Full JVP along a state and a rate direction.
            let direction_m = probe_vector(dimension, 0.3, 1.0);
            let rate_direction_m = probe_vector(dimension, 1.1, 0.5);
            let direction_c = permute(&direction_m, &inverse);
            let rate_direction_c = permute(&rate_direction_m, &inverse);
            mono.operator
                .jacobian_vector_product(
                    TIME,
                    &state_m,
                    &rate_m,
                    &direction_m,
                    &rate_direction_m,
                    &mut expected,
                )
                .unwrap();
            composed
                .operator
                .jacobian_vector_product(
                    TIME,
                    &state_c,
                    &rate_c,
                    &direction_c,
                    &rate_direction_c,
                    &mut actual,
                )
                .unwrap();
            assert_close(
                &permute(&actual, &map),
                &expected,
                AGREEMENT,
                &format!("{quadrature:?} JVP at {label}"),
            );
            // Every (row, column) block, the two cross blocks included.
            for (r, (mono_row, composed_row)) in mono_rows.iter().zip(&composed_rows).enumerate() {
                for (c, (mono_column, composed_column)) in
                    mono_columns.iter().zip(&composed_columns).enumerate()
                {
                    let extent = mono
                        .operator
                        .layout()
                        .block_by_variable(*mono_column)
                        .unwrap()
                        .extent;
                    let row_extent = mono
                        .operator
                        .layout()
                        .block_by_variable(*mono_row_variable(&mono, r))
                        .unwrap()
                        .extent;
                    let block_direction = probe_vector(extent, 0.7 + c as f64, 1.0);
                    let mut expected = vec![0.0; row_extent];
                    mono.operator
                        .block_jacobian_vector_product(
                            *mono_row,
                            *mono_column,
                            TIME,
                            &state_m,
                            &rate_m,
                            &block_direction,
                            RATE_SHIFT,
                            &mut expected,
                        )
                        .unwrap();
                    let mut actual = vec![0.0; row_extent];
                    composed
                        .operator
                        .block_jacobian_vector_product(
                            *composed_row,
                            *composed_column,
                            TIME,
                            &state_c,
                            &rate_c,
                            &block_direction,
                            RATE_SHIFT,
                            &mut actual,
                        )
                        .unwrap();
                    assert_close(
                        &actual,
                        &expected,
                        AGREEMENT,
                        &format!("{quadrature:?} block ({r}, {c}) at {label}"),
                    );
                    if r != c && label == "nontrivial" {
                        assert!(
                            max_abs(&actual) > 1.0e-6,
                            "{quadrature:?} cross block ({r}, {c}) at {label} is zero"
                        );
                    }
                }
            }
            // The two cross blocks on a unit direction at an interior node.
            if label == "nontrivial" {
                let interior = tagged
                    .mesh
                    .vertices()
                    .iter()
                    .position(|x| x[0] > 0.3 && x[0] < 0.7 && x[1] > 0.3 && x[1] < 0.7)
                    .expect("the 3x3 box has an interior vertex");
                for (row, column, what) in [
                    (composed.thermal_row, composed.v, "dR_thermal/dV"),
                    (composed.electrical_row, composed.t, "dR_electrical/dT"),
                ] {
                    let vertices = tagged.mesh.vertices().len();
                    let mut unit = vec![0.0; vertices];
                    unit[interior] = 1.0;
                    let mut actual = vec![0.0; vertices];
                    composed
                        .operator
                        .block_jacobian_vector_product(
                            row,
                            column,
                            TIME,
                            &state_c,
                            &rate_c,
                            &unit,
                            RATE_SHIFT,
                            &mut actual,
                        )
                        .unwrap();
                    let (mono_row, mono_column) = if what == "dR_thermal/dV" {
                        (SysResId(1), SysVarId(mono.v.0))
                    } else {
                        (SysResId(0), SysVarId(mono.t.0))
                    };
                    let mut expected = vec![0.0; vertices];
                    mono.operator
                        .block_jacobian_vector_product(
                            mono_row,
                            mono_column,
                            TIME,
                            &state_m,
                            &rate_m,
                            &unit,
                            RATE_SHIFT,
                            &mut expected,
                        )
                        .unwrap();
                    assert!(
                        max_abs(&actual) > 1.0e-6,
                        "{what} on a unit direction is zero"
                    );
                    assert_close(&actual, &expected, AGREEMENT, what);
                }
            }
        }
    }
}

fn mono_row_variable(mono: &Monolithic, row: usize) -> &SysVarId {
    let rows = [mono.v, mono.t];
    let symbol = rows[row];
    &mono.operator.layout().block(symbol).unwrap().variable
}

#[test]
fn composed_bind_transposes_and_linearizations_are_exact() {
    let tagged = unit_square(3);
    let quadrature = SystemQuadrature::Richest;
    let mono = monolithic_08(&tagged, quadrature);
    let composed = composed_electrothermal(&tagged, quadrature);
    let map = permutation(&composed, &mono);
    let inverse = {
        let mut inverse = vec![0; map.len()];
        for (from, to) in map.iter().enumerate() {
            inverse[*to] = from;
        }
        inverse
    };
    let dimension = mono.operator.dimension();
    let (_, state_m, rate_m) = states(&mono).pop().unwrap();
    let state = permute(&state_m, &inverse);
    let rate = permute(&rate_m, &inverse);

    // Adjoint identity of the composed operator: <J d, w> = <d, J^T w> with the rate shift.
    let direction = probe_vector(dimension, 0.3, 1.0);
    let rate_direction = direction
        .iter()
        .map(|value| RATE_SHIFT * value)
        .collect::<Vec<_>>();
    let adjoint = probe_vector(dimension, 4.2, 1.0);
    let mut forward = vec![0.0; dimension];
    composed
        .operator
        .jacobian_vector_product(
            TIME,
            &state,
            &rate,
            &direction,
            &rate_direction,
            &mut forward,
        )
        .unwrap();
    let mut transposed = vec![0.0; dimension];
    composed
        .operator
        .vector_jacobian_product_shifted(TIME, &state, &rate, &adjoint, RATE_SHIFT, &mut transposed)
        .unwrap();
    let left = dot(&forward, &adjoint);
    let right = dot(&direction, &transposed);
    assert!(
        (left - right).abs() <= IDENTITY * left.abs().max(right.abs()).max(1.0),
        "composed adjoint identity: {left} vs {right}"
    );
    // ... and equal to the monolithic transpose after the permutation.
    let mut expected = vec![0.0; dimension];
    mono.operator
        .vector_jacobian_product_shifted(
            TIME,
            &state_m,
            &rate_m,
            &permute(&adjoint, &map),
            RATE_SHIFT,
            &mut expected,
        )
        .unwrap();
    assert_close(
        &permute(&transposed, &map),
        &expected,
        AGREEMENT,
        "VJP vs monolithic",
    );

    // Block transposes of both cross blocks satisfy the adjoint identity.
    let vertices = tagged.mesh.vertices().len();
    for (row, column) in [
        (composed.thermal_row, composed.v),
        (composed.electrical_row, composed.t),
    ] {
        let block_direction = probe_vector(vertices, 0.9, 1.0);
        let block_adjoint = probe_vector(vertices, 3.3, 1.0);
        let mut forward = vec![0.0; vertices];
        composed
            .operator
            .block_jacobian_vector_product(
                row,
                column,
                TIME,
                &state,
                &rate,
                &block_direction,
                RATE_SHIFT,
                &mut forward,
            )
            .unwrap();
        let mut transposed = vec![0.0; vertices];
        composed
            .operator
            .block_vector_jacobian_product(
                row,
                column,
                TIME,
                &state,
                &rate,
                &block_adjoint,
                RATE_SHIFT,
                &mut transposed,
            )
            .unwrap();
        let left = dot(&forward, &block_adjoint);
        let right = dot(&block_direction, &transposed);
        assert!(
            (left - right).abs() <= IDENTITY * left.abs().max(right.abs()).max(1.0),
            "block ({row}, {column}) adjoint identity: {left} vs {right}"
        );
        assert!(max_abs(&transposed) > 1.0e-6);
    }

    // `linearize` / `assemble` agree with the JVP, on the composed and the monolithic side.
    let linearized = composed
        .operator
        .linearize(TIME, &state, &rate, RATE_SHIFT)
        .unwrap();
    let assembled = linearized.assemble().unwrap();
    let mut through_matrix = vec![0.0; dimension];
    assembled
        .apply(
            &methodus::EvaluationContext::default(),
            &direction,
            &mut through_matrix,
        )
        .unwrap();
    assert_close(
        &through_matrix,
        &forward,
        IDENTITY,
        "assembled composed Jacobian",
    );
    let mono_assembled = mono
        .operator
        .linearize(TIME, &state_m, &rate_m, RATE_SHIFT)
        .unwrap()
        .assemble()
        .unwrap();
    let mut mono_through = vec![0.0; dimension];
    mono_assembled
        .apply(
            &methodus::EvaluationContext::default(),
            &permute(&direction, &map),
            &mut mono_through,
        )
        .unwrap();
    assert_close(
        &permute(&through_matrix, &map),
        &mono_through,
        AGREEMENT,
        "assembled composed vs monolithic",
    );
    // The Jacobian at the nontrivial state is nonsymmetric (the two cross blocks are not
    // transposes of each other), while the zero-point linear view -- where both cross blocks
    // vanish with `grad V = 0` -- proves Symmetric by assembly; the structural claim of a
    // composed operator is `Unknown` until proven.
    let mut forward_w = vec![0.0; dimension];
    linearized
        .apply(
            &methodus::EvaluationContext::default(),
            &adjoint,
            &mut forward_w,
        )
        .unwrap();
    let asymmetry = (dot(&forward, &adjoint) - dot(&forward_w, &direction)).abs();
    assert!(
        asymmetry > 1.0e-6,
        "the composed Jacobian is nonsymmetric: {asymmetry}"
    );
    assert!(composed.operator.assemble().is_ok());
    assert_eq!(composed.operator.symmetry(), OperatorSymmetry::Unknown);
    assert_eq!(
        composed.operator.prove_symmetry(1.0e-12).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert_eq!(composed.operator.symmetry(), OperatorSymmetry::Symmetric);
}

#[test]
fn composed_receipts_name_both_instances_and_the_binds() {
    let tagged = unit_square(2);
    let composed = composed_electrothermal(&tagged, SystemQuadrature::Barycenter);
    let operator = &composed.operator;
    let ids = operator.system_ids();
    assert_eq!(ids.instances().len(), 2);
    assert_eq!(
        ids.residual_path(composed.thermal_row).as_deref(),
        Some("thermal.thermal")
    );
    assert_eq!(
        ids.variable_path(composed.v).unwrap(),
        format!("electrical/{}", {
            let model = composed
                .composed
                .compilation
                .model(scientia::InstanceId(composed.electrical.0));
            symbol(model, "V")
        })
    );

    // Methodus block names carry the instance.
    let OperatorStructureHint::Block { layout, .. } = operator.properties().structure().clone()
    else {
        panic!("a system operator is block structured");
    };
    let names = layout
        .blocks()
        .iter()
        .map(|block| block.name().to_string())
        .collect::<Vec<_>>();
    assert!(
        names
            .iter()
            .any(|name| name.starts_with("electrical/field_")),
        "{names:?}"
    );
    assert!(
        names.iter().any(|name| name.starts_with("thermal/field_")),
        "{names:?}"
    );

    // The plan identity is the composed `/3` schema; the per-instance `/1` artifacts are there.
    let plan = operator.plan();
    assert!(plan.system_operator().is_some());
    assert_eq!(
        plan.instance_system(composed.electrical).unwrap().model,
        "ElectricalConduction"
    );
    assert_eq!(
        plan.instance_system(composed.thermal).unwrap().model,
        "HeatConduction"
    );
    assert_eq!(
        SYSTEM_REALIZATION_COMPOSED_DIGEST_SCHEMA,
        "finitum-system-realization/3"
    );
    assert!(operator.instance_structure(InstanceId(1)).is_some());
    assert!(operator.instance_structure(InstanceId(2)).is_none());

    // Binds: the kernel-input Q bind runs a composition; the provider-input temperature bind
    // carries none; both name their rows and columns (both cross blocks exist in the `/2`).
    let binds = operator.binds();
    assert_eq!(
        binds.len(),
        2,
        "{:?}",
        binds
            .iter()
            .map(|bind| (&bind.consumer_slot, &bind.output))
            .collect::<Vec<_>>()
    );
    let q = binds
        .iter()
        .find(|bind| bind.consumer_slot == "thermal/input/Q")
        .unwrap();
    assert_eq!(q.path, BindPath::KernelInput);
    assert_eq!(q.output, "electrical.joule_heat");
    assert_eq!(q.compositions.len(), 1);
    assert_eq!(q.jvp_compositions.len(), 1);
    assert_eq!(q.rows, vec![composed.thermal_row]);
    assert_eq!(q.columns, vec![composed.v]);
    let temperature = binds
        .iter()
        .find(|bind| bind.consumer_slot == "electrical/input/temperature")
        .unwrap();
    assert_eq!(temperature.path, BindPath::ProviderInput);
    assert!(temperature.compositions.is_empty());
    assert_eq!(temperature.rows, vec![composed.electrical_row]);
    assert_eq!(temperature.columns, vec![composed.t]);
    // The binds precede in dependency order: temperature (read by joule_heat's sigma) first.
    assert_eq!(binds[0].consumer_slot, "electrical/input/temperature");

    // Capability / artifact receipts over the reduced operator.
    let constraints = finitum::ConstraintSet::new(operator.dimension(), []).unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    let capability = reduced.capability();
    // Every instance's element requirements (its fields and its input fields): V and
    // `temperature` of the electrical instance, T and Q of the thermal one.
    assert_eq!(capability.elements.len(), 4);
    assert!(
        !capability
            .representation_kinds
            .contains(&finitum::RepresentationKind::PartialAssembly)
    );
    assert!(matches!(
        operator.partial_assembly(1),
        Err(FinitumError::RepresentationUnsupported { .. })
    ));
    let artifact = reduced.artifact();
    assert_eq!(artifact.instances.len(), 2);
    assert_eq!(artifact.binds.len(), 2);
    assert_eq!(artifact.blocks.len(), 2);
    assert_eq!(artifact.blocks[1].equation, "thermal.thermal");
    assert_eq!(artifact.blocks[1].residual, Some(composed.thermal_row));
    assert!(
        artifact
            .external_inputs
            .iter()
            .all(|input| input.residual.is_some())
    );
    let json = serde_json::to_string(&artifact).unwrap();
    assert!(json.contains("\"binds\""));
    assert!(json.contains("thermal/input/Q"));
    // The all-table agreement report needs partial assembly, which a bind chain cannot
    // store: refused typed, naming the consumer row. Matrix-free, assembled and element
    // assembly agree.
    let probe = probe_vector(operator.dimension(), 0.6, 1.0);
    let tolerance = methodus::ComparisonTolerance {
        absolute: 1.0e-12,
        relative: 1.0e-12,
    };
    match finitum::check_system_realization_agreement(&reduced, &probe, 4, tolerance) {
        Err(FinitumError::RepresentationUnsupported {
            representation,
            equation,
            reason,
            ..
        }) => {
            assert_eq!(representation, finitum::RepresentationKind::PartialAssembly);
            assert!(
                equation == "electrical.electrical" || equation == "thermal.thermal",
                "{equation}"
            );
            assert!(reason.contains("same-mesh bind on `"), "{reason}");
        }
        other => panic!("expected a typed partial-assembly refusal, got {other:?}"),
    }
    let context = methodus::EvaluationContext::default();
    let mut matrix_free = vec![0.0; operator.dimension()];
    reduced.apply(&context, &probe, &mut matrix_free).unwrap();
    let mut assembled = vec![0.0; operator.dimension()];
    reduced
        .assemble()
        .unwrap()
        .apply(&context, &probe, &mut assembled)
        .unwrap();
    let mut element = vec![0.0; operator.dimension()];
    reduced
        .element_assembly(4)
        .unwrap()
        .apply(&context, &probe, &mut element)
        .unwrap();
    assert_close(
        &assembled,
        &matrix_free,
        IDENTITY,
        "assembled vs matrix-free",
    );
    assert_close(
        &element,
        &matrix_free,
        IDENTITY,
        "element assembly vs matrix-free",
    );
}

#[test]
fn two_instances_of_one_model_realize_as_distinct_blocks() {
    let tagged = unit_square(2);
    let composed = compile_composed(TWO_HEAT, "TwoHeat");
    let plan = composed_plan(&composed, &tagged, SystemQuadrature::Barycenter);
    let ids = plan.system_ids().clone();
    assert_eq!(ids.variables().len(), 2);
    assert_ne!(ids.variables()[0].id, ids.variables()[1].id);
    assert_eq!(
        ids.variables()[0].local,
        ids.variables()[1].local,
        "same model, same symbol"
    );
    let model = composed.compilation.model(scientia::InstanceId(0));
    let t_symbol = symbol(model, "T");
    let t_in_of = |integral: &IntegralOperatorFactorization| {
        active_input(integral, t_symbol, DerivativeEvaluation::Value)
    };
    // Closures per instance: the same laws, `Q` a constant source that differs per instance so
    // the two copies are distinguishable.
    let mut closures = Vec::new();
    for residual in ids.residuals() {
        let system = plan.instance_system(residual.instance).unwrap();
        let block = &system.blocks[0];
        let q_value = 1.0 + residual.instance.0 as f64;
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = symbol_name(model, input.binding.symbol).to_string();
                let index = integral.integral_index;
                let built = match name.as_str() {
                    "k" => {
                        let t_in = t_in_of(integral);
                        SystemConstitutiveInput::try_new_for_residual(
                            residual.id,
                            index,
                            input.id,
                            1,
                            "two/k",
                            move |point: &PointEvaluation| {
                                Ok(vec![conductivity(point.input_values(t_in).unwrap()[0])])
                            },
                            move |_: &PointEvaluation, direction: &PointEvaluation| {
                                Ok(vec![d_conductivity(
                                    direction.input_values(t_in).unwrap()[0],
                                )])
                            },
                        )
                    }
                    "rho" | "cp" => SystemConstitutiveInput::try_new_for_residual(
                        residual.id,
                        index,
                        input.id,
                        1,
                        format!("two/{name}"),
                        |_: &PointEvaluation| Ok(vec![1.0]),
                        |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
                    ),
                    "Q" => SystemConstitutiveInput::try_new_for_residual(
                        residual.id,
                        index,
                        input.id,
                        1,
                        format!("two/Q={q_value}"),
                        move |_: &PointEvaluation| Ok(vec![q_value]),
                        |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
                    ),
                    other => panic!("unexpected input {other}"),
                };
                closures.push(built.unwrap());
            }
        }
    }
    // An equation-name key is ambiguous on this plan and refused typed.
    let ambiguous = SystemConstitutiveInput::new(
        "thermal",
        0,
        closures[0].input,
        1,
        "ambiguous",
        |_: &PointEvaluation| vec![0.0],
        |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
    )
    .unwrap();
    let mut with_ambiguous = closures.clone();
    with_ambiguous.push(ambiguous);
    assert!(matches!(
        plan.bind_kernels(with_ambiguous, BTreeMap::new()),
        Err(FinitumError::InvalidRealization(message)) if message.contains("2 instances")
    ));
    let operator = plan.bind_kernels(closures, BTreeMap::new()).unwrap();
    assert!(operator.binds().is_empty());
    let composed_symmetry = operator.symmetry();
    assert!(
        operator.dof_map(t_symbol).is_none(),
        "the symbol is ambiguous"
    );
    assert!(
        operator
            .dof_map_by_variable(ids.variables()[1].id)
            .is_some()
    );

    // Each instance's residual equals the one-instance realization of the same model on the
    // same mesh with its own state.
    let vertices = tagged.mesh.vertices().len();
    let state = probe_vector(2 * vertices, 0.4, 5.0)
        .iter()
        .map(|value| 300.0 + value)
        .collect::<Vec<_>>();
    let rate = probe_vector(2 * vertices, 1.4, 2.0);
    let mut residual = vec![0.0; 2 * vertices];
    operator
        .residual(TIME, &state, &rate, &mut residual)
        .unwrap();
    let direction = probe_vector(2 * vertices, 2.2, 1.0);
    let mut jvp = vec![0.0; 2 * vertices];
    operator
        .jacobian_vector_product(TIME, &state, &rate, &direction, &rate, &mut jvp)
        .unwrap();
    for instance in [InstanceId(0), InstanceId(1)] {
        let system = plan.instance_system(instance).unwrap().clone();
        let single_layout = BlockLayout::new([(t_symbol, vertices, 1)]).unwrap();
        let single_plan = SystemRealizationPlan::with_quadrature(
            system,
            tagged.mesh.clone(),
            single_layout,
            SystemQuadrature::Barycenter,
        )
        .unwrap();
        let q_value = 1.0 + instance.0 as f64;
        let mut single_closures = Vec::new();
        for block in &single_plan.system().blocks {
            for integral in &block.factorization.integrals {
                for input in &integral.primal.inputs {
                    if input.source == InputSourceRequirement::Basis {
                        continue;
                    }
                    let name = symbol_name(model, input.binding.symbol).to_string();
                    let built = match name.as_str() {
                        "k" => {
                            let t_in = t_in_of(integral);
                            SystemConstitutiveInput::new(
                                block.equation.clone(),
                                integral.integral_index,
                                input.id,
                                1,
                                "one/k",
                                move |point: &PointEvaluation| {
                                    vec![conductivity(point.input_values(t_in).unwrap()[0])]
                                },
                                move |_: &PointEvaluation, direction: &PointEvaluation| {
                                    vec![d_conductivity(direction.input_values(t_in).unwrap()[0])]
                                },
                            )
                        }
                        "rho" | "cp" => SystemConstitutiveInput::new(
                            block.equation.clone(),
                            integral.integral_index,
                            input.id,
                            1,
                            "one/c",
                            |_: &PointEvaluation| vec![1.0],
                            |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                        ),
                        "Q" => SystemConstitutiveInput::new(
                            block.equation.clone(),
                            integral.integral_index,
                            input.id,
                            1,
                            "one/Q",
                            move |_: &PointEvaluation| vec![q_value],
                            |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                        ),
                        other => panic!("unexpected input {other}"),
                    };
                    single_closures.push(built.unwrap());
                }
            }
        }
        let single = single_plan
            .bind_kernels(single_closures, BTreeMap::new())
            .unwrap();
        assert_eq!(
            composed_symmetry,
            single.symmetry(),
            "block-diagonal: the structural claim of the instances"
        );
        let variable = ids.variable(instance, t_symbol).unwrap();
        let block = operator.layout().block_by_variable(variable).unwrap();
        let slice = |vector: &[f64]| vector[block.offset..block.offset + block.extent].to_vec();
        let mut expected = vec![0.0; vertices];
        single
            .residual(TIME, &slice(&state), &slice(&rate), &mut expected)
            .unwrap();
        assert_close(
            &slice(&residual),
            &expected,
            IDENTITY,
            "duplicate-instance residual",
        );
        single
            .jacobian_vector_product(
                TIME,
                &slice(&state),
                &slice(&rate),
                &slice(&direction),
                &slice(&rate),
                &mut expected,
            )
            .unwrap();
        assert_close(&slice(&jvp), &expected, IDENTITY, "duplicate-instance JVP");
    }
    // The two instances differ (their sources do), so the blocks are not copies of each other.
    let a = &residual[..vertices];
    let b = &residual[vertices..];
    assert!((0..vertices).any(|i| (a[i] - b[i]).abs() > 1.0e-6));
}

#[test]
fn keyed_essential_constraints_touch_only_the_named_instance_and_reduce() {
    let tagged = unit_square(2);
    let composed = composed_electrothermal(&tagged, SystemQuadrature::Barycenter);
    let operator = &composed.operator;
    let electrical_system = operator
        .plan()
        .instance_system(composed.electrical)
        .unwrap();
    // The electrical instance's anode/cathode Dirichlet requirements, resolved through the
    // electrical instance's own region map (`electrical.anode` -> x_min, `electrical.cathode`
    // -> x_max); the thermal instance has none and needs no map.
    let mut region_map = RegionMap::new();
    let mut requirements = Vec::new();
    for block in &electrical_system.blocks {
        for (index, requirement) in block.factorization.essential_constraints.iter().enumerate() {
            let tag = if index == 0 { "x_min" } else { "x_max" };
            region_map.insert(requirement.region, [RegionTagId::new(tag)]);
            requirements.push(SystemVariableEssentialConstraint {
                variable: composed.v,
                requirement: requirement.clone(),
                value: FieldSource::constant([1.0 - index as f64]),
            });
        }
    }
    assert_eq!(requirements.len(), 2);
    let constraints = essential_constraints_from_system_by_variable_at(
        operator,
        &tagged,
        &[(composed.electrical, &region_map)],
        &requirements,
        TIME,
    )
    .unwrap();
    let v_block = operator.layout().block_by_variable(composed.v).unwrap();
    let constraint_list = constraints.constraints().cloned().collect::<Vec<_>>();
    assert!(!constraint_list.is_empty());
    for constraint in &constraint_list {
        assert!(
            constraint.target.0 >= v_block.offset
                && constraint.target.0 < v_block.offset + v_block.extent,
            "constraint {constraint:?} is outside the electrical V block"
        );
    }
    let boundary_vertices = tagged
        .mesh
        .vertices()
        .iter()
        .filter(|x| x[0] < 1.0e-12 || x[0] > 1.0 - 1.0e-12)
        .count();
    assert_eq!(constraint_list.len(), boundary_vertices);
    let reduced = operator.reduced(constraints.clone()).unwrap();
    assert_eq!(reduced.rows(), operator.dimension());
    assert_eq!(
        reduced.constraints().constraints().count(),
        boundary_vertices
    );
    // The symbol-keyed form answers only symbols unique across instances: `V` and `T` are
    // both `SymbolId(0)` of their models, so on this plan it refuses typed instead of
    // guessing an instance.
    assert_eq!(
        v_block.symbol,
        operator
            .layout()
            .block_by_variable(composed.t)
            .unwrap()
            .symbol
    );
    assert!(matches!(
        essential_constraints_from_system(
            operator,
            &tagged,
            &region_map,
            &requirements
                .iter()
                .map(|requirement| SystemEssentialConstraintRequirement {
                    field: v_block.symbol,
                    requirement: requirement.requirement.clone(),
                    value: requirement.value.clone(),
                })
                .collect::<Vec<_>>(),
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    // An instance without a region map is refused typed, naming the region and instance.
    assert!(matches!(
        essential_constraints_from_system_by_variable_at(
            operator,
            &tagged,
            &[(composed.thermal, &region_map)],
            &requirements,
            TIME
        ),
        Err(FinitumError::RealizationRegionUnmapped(message)) if message.contains("instance#0")
    ));
    // A stored table on one instance's row is a distributed coefficient of the composed
    // operator: coefficient JVP and VJP are adjoint.
    let thermal_system = operator.plan().instance_system(composed.thermal).unwrap();
    let thermal_model = composed
        .composed
        .compilation
        .model(scientia::InstanceId(composed.thermal.0));
    let (integral_index, rho) = thermal_system
        .blocks
        .iter()
        .flat_map(|block| block.factorization.integrals.iter())
        .find_map(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .find(|input| {
                    input.source != InputSourceRequirement::Basis
                        && symbol_name(thermal_model, input.binding.symbol) == "rho"
                })
                .map(|input| (integral.integral_index, input.id))
        })
        .unwrap();
    let plan = operator.plan();
    let cells = tagged.mesh.cells().len();
    let points = plan.quadrature().unwrap().len();
    let table = ExternalInput::new(integral_index, rho, 1, vec![RHO; cells * points]).unwrap();
    let mut closures = composed_closures(&composed.composed, plan, None);
    closures.retain(|closure| closure.identity() != "composed/thermal/rho");
    let with_table = plan
        .bind_kernels_with_inputs(
            closures,
            vec![SystemExternalInput {
                residual: composed.thermal_row,
                input: table,
            }],
            BTreeMap::new(),
            BTreeMap::new(),
        )
        .unwrap();
    let coefficient = SystemDistributedCoefficient {
        residual: composed.thermal_row,
        coefficient: DistributedCoefficient {
            integral_index,
            input: rho,
            layout: CoefficientLayout::Cell,
        },
    };
    let design_dimension = with_table.coefficient_dimension(&coefficient).unwrap();
    assert_eq!(design_dimension, cells);
    let dimension = with_table.dimension();
    let state = probe_vector(dimension, 0.4, 1.0)
        .iter()
        .map(|value| 300.0 + value)
        .collect::<Vec<_>>();
    let rate = probe_vector(dimension, 1.4, 2.0);
    let design_direction = probe_vector(design_dimension, 0.2, 1.0);
    let adjoint = probe_vector(dimension, 5.0, 1.0);
    let mut forward = vec![0.0; dimension];
    with_table
        .coefficient_jacobian_vector_product(
            TIME,
            &state,
            &rate,
            &coefficient,
            &design_direction,
            &mut forward,
        )
        .unwrap();
    let mut transposed = vec![0.0; design_dimension];
    with_table
        .coefficient_vector_jacobian_product(
            TIME,
            &state,
            &rate,
            &coefficient,
            &adjoint,
            &mut transposed,
        )
        .unwrap();
    let left = dot(&forward, &adjoint);
    let right = dot(&design_direction, &transposed);
    assert!(
        max_abs(&forward) > 1.0e-6,
        "rho scales the thermal mass term"
    );
    assert!(
        (left - right).abs() <= IDENTITY * left.abs().max(right.abs()).max(1.0),
        "coefficient adjoint identity: {left} vs {right}"
    );
    // The residual with the table equals the residual with the closure.
    let mut with_table_residual = vec![0.0; dimension];
    with_table
        .residual(TIME, &state, &rate, &mut with_table_residual)
        .unwrap();
    let mut with_closure_residual = vec![0.0; dimension];
    operator
        .residual(TIME, &state, &rate, &mut with_closure_residual)
        .unwrap();
    assert_close(
        &with_table_residual,
        &with_closure_residual,
        IDENTITY,
        "table vs closure",
    );
    // The reduced operator's DAE view evaluates through the bind chain.
    let mut reduced_residual = vec![0.0; dimension];
    reduced
        .residual(TIME, &state, &rate, &mut reduced_residual)
        .unwrap();
    assert!(reduced_residual.iter().all(|value| value.is_finite()));
}

#[test]
fn fallible_closures_on_the_composed_path_surface_typed_with_their_origin() {
    let tagged = unit_square(2);
    let composed = compile_composed(ELECTROTHERMAL, "Electrothermal");
    let plan = composed_plan(&composed, &tagged, SystemQuadrature::Barycenter);
    let dimension = plan.layout().extent();
    let state = probe_vector(dimension, 0.4, 1.0)
        .iter()
        .map(|value| 300.0 + value)
        .collect::<Vec<_>>();
    let rate = vec![0.0; dimension];
    let refused_cell = 3;
    for (site, origin) in [
        ("k", "thermal/provider/thermal_conductivity"),
        ("output-sigma", "electrical.joule_heat.sigma"),
    ] {
        let closures = composed_closures(&composed, &plan, Some((site, refused_cell)));
        let operator = plan.bind_kernels(closures, BTreeMap::new()).unwrap();
        let mut output = vec![0.0; dimension];
        let error = operator
            .residual(TIME, &state, &rate, &mut output)
            .unwrap_err();
        let FinitumError::InputEvaluation(failure) = &error else {
            panic!("{site}: expected a typed input evaluation failure, got {error}");
        };
        assert_eq!(failure.code, "RUN_PROPERTY_REFUSED");
        assert_eq!(failure.origin.to_string(), origin);
        assert_eq!(failure.cell(), Some(finitum::CellId(refused_cell)));
        assert_eq!(failure.time(), Some(TIME));
        assert!(failure.point().is_some());
        assert_eq!(error.code(), Some("RUN_PROPERTY_REFUSED"));
        // The JVP and the VJP refuse the same way (never NaN).
        let direction = probe_vector(dimension, 1.0, 1.0);
        assert!(matches!(
            operator.jacobian_vector_product(TIME, &state, &rate, &direction, &rate, &mut output),
            Err(FinitumError::InputEvaluation(_))
        ));
        assert!(matches!(
            operator.vector_jacobian_product(TIME, &state, &rate, &direction, &mut output),
            Err(FinitumError::InputEvaluation(_))
        ));
    }
}

#[test]
fn composed_bind_time_refusals_are_typed() {
    let tagged = unit_square(2);
    let composed = compile_composed(ELECTROTHERMAL, "Electrothermal");
    let plan = composed_plan(&composed, &tagged, SystemQuadrature::Barycenter);
    let closures = composed_closures(&composed, &plan, None);
    // Missing the output kernel's closure.
    let without_output = closures
        .iter()
        .filter(|closure| closure.identity() != "composed/electrical.joule_heat/sigma")
        .cloned()
        .collect::<Vec<_>>();
    assert!(matches!(
        plan.bind_kernels(without_output, BTreeMap::new()),
        Err(FinitumError::UnsupportedRealization(message))
            if message.contains("electrical.joule_heat") && message.contains("try_new_for_output")
    ));
    // Binding the bound input `Q` as a closure as well is refused.
    let ids = plan.system_ids();
    let thermal = ids
        .instances()
        .iter()
        .find(|record| record.name == "thermal")
        .unwrap()
        .instance;
    let thermal_row = ids.residual(thermal, "thermal").unwrap();
    let thermal_model = composed.compilation.model(scientia::InstanceId(thermal.0));
    let q_symbol = symbol(thermal_model, "Q");
    let (integral_index, q_input) = plan
        .instance_system(thermal)
        .unwrap()
        .blocks
        .iter()
        .flat_map(|block| block.factorization.integrals.iter())
        .find_map(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .find(|input| input.binding.symbol == q_symbol)
                .map(|input| (integral.integral_index, input.id))
        })
        .unwrap();
    let mut with_q = closures.clone();
    with_q.push(
        SystemConstitutiveInput::try_new_for_residual(
            thermal_row,
            integral_index,
            q_input,
            1,
            "bogus/Q",
            |_: &PointEvaluation| Ok(vec![0.0]),
            |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
        )
        .unwrap(),
    );
    assert!(matches!(
        plan.bind_kernels(with_q, BTreeMap::new()),
        Err(FinitumError::InvalidRealization(message)) if message.contains("closed by the bind")
    ));
    // A one-instance plan refuses an output-keyed closure.
    let single = compile_semantics(MONOLITHIC_08, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &single.semantic,
        "ElectrothermalJoule",
        &["electrical", "thermal"],
    )
    .unwrap();
    let model = &single.semantic.models[0];
    let vertices = tagged.mesh.vertices().len();
    let layout = BlockLayout::new([
        (symbol(model, "V"), vertices, 1),
        (symbol(model, "T"), vertices, 1),
    ])
    .unwrap();
    let single_plan = SystemRealizationPlan::new(system, tagged.mesh.clone(), layout).unwrap();
    let output_keyed = SystemConstitutiveInput::try_new_for_output(
        InstanceId(0),
        scientia::OutputId(0),
        0,
        q_input,
        1,
        "bogus/output",
        |_: &PointEvaluation| Ok(vec![0.0]),
        |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
    )
    .unwrap();
    assert!(matches!(
        single_plan.bind_kernels(vec![output_keyed], BTreeMap::new()),
        Err(FinitumError::InvalidRealization(message)) if message.contains("no same-mesh binds")
    ));
    // A `/2` artifact of a one-instance implicit system is admitted by `composed` too.
    let closure =
        resolve_module_closure(MONOLITHIC_08, &BTreeMap::<String, String>::new()).unwrap();
    let (units, kinds) = registries();
    let implicit = scientia::compile_model_system(
        &closure,
        Registries::new(&units, &kinds),
        "ElectrothermalJoule",
    )
    .unwrap();
    let implicit_operator = compile_system_operator(&implicit).unwrap();
    let implicit_ids = finitum::SystemIdMap::from_scientia(&implicit_operator.operator).unwrap();
    let implicit_layout = BlockLayout::new_keyed(
        implicit_ids
            .variables()
            .iter()
            .map(|variable| (variable.id, variable.local, vertices, 1)),
    )
    .unwrap();
    let implicit_plan = SystemRealizationPlan::composed(
        &implicit_operator,
        tagged.mesh.clone(),
        implicit_layout,
        SystemQuadrature::Richest,
    )
    .unwrap();
    assert!(implicit_plan.system_operator().is_some());
    assert_eq!(implicit_plan.system_ids().instances().len(), 1);
    assert_ne!(
        implicit_plan.artifact_digest(),
        single_plan.artifact_digest(),
        "/3 vs /2 identity"
    );
}

#[test]
fn one_output_can_feed_two_consumers_with_complete_derivatives() {
    // One electrical output supplies two independent thermal residuals. This exercises
    // output-closure reuse and accumulation back through both outgoing bind paths.
    let source = ELECTROTHERMAL
        .replace(
            "instance thermal: thermal.HeatConduction(body = body);",
            "instance thermal: thermal.HeatConduction(body = body);\n    instance extra: thermal.HeatConduction(body = body);",
        )
        .replace(
            "bind thermal.Q <- electrical.joule_heat;",
            "bind thermal.Q <- electrical.joule_heat;\n    bind extra.Q <- electrical.joule_heat;",
        );
    let compiled = compile_composed(&source, "Electrothermal");
    let tagged = unit_square(2);
    let plan = composed_plan(&compiled, &tagged, SystemQuadrature::Richest);
    let operator = plan
        .bind_kernels(composed_closures(&compiled, &plan, None), BTreeMap::new())
        .unwrap();
    let n = operator.dimension();
    let mut state = vec![0.0; n];
    let mut thermal_ranges = Vec::new();
    for variable in plan.system_ids().variables() {
        let block = plan.layout().block_by_variable(variable.id).unwrap();
        let model = compiled
            .compilation
            .model(scientia::InstanceId(variable.instance.0));
        let is_temperature = symbol_name(model, variable.local) == "T";
        if is_temperature {
            thermal_ranges.push(block.offset..block.offset + block.extent);
        }
        for (local, value) in state[block.offset..block.offset + block.extent]
            .iter_mut()
            .enumerate()
        {
            *value = if is_temperature {
                302.0 + 0.2 * local as f64
            } else {
                0.3 * local as f64
            };
        }
    }
    let rate = vec![0.0; n];
    let mut residual = vec![0.0; n];
    operator
        .residual(TIME, &state, &rate, &mut residual)
        .unwrap();
    assert_eq!(thermal_ranges.len(), 2);
    assert_close(
        &residual[thermal_ranges[0].clone()],
        &residual[thermal_ranges[1].clone()],
        IDENTITY,
        "fan-out thermal residuals",
    );
    let direction = probe_vector(n, 0.3, 1.0);
    let rate_direction = direction.iter().map(|x| RATE_SHIFT * x).collect::<Vec<_>>();
    let mut tangent = vec![0.0; n];
    operator
        .jacobian_vector_product(
            TIME,
            &state,
            &rate,
            &direction,
            &rate_direction,
            &mut tangent,
        )
        .unwrap();
    let epsilon = 1.0e-5;
    let plus = state
        .iter()
        .zip(&direction)
        .map(|(x, d)| x + epsilon * d)
        .collect::<Vec<_>>();
    let minus = state
        .iter()
        .zip(&direction)
        .map(|(x, d)| x - epsilon * d)
        .collect::<Vec<_>>();
    let rate_plus = rate_direction
        .iter()
        .map(|d| epsilon * d)
        .collect::<Vec<_>>();
    let rate_minus = rate_direction
        .iter()
        .map(|d| -epsilon * d)
        .collect::<Vec<_>>();
    let mut forward = vec![0.0; n];
    let mut backward = vec![0.0; n];
    operator
        .residual(TIME, &plus, &rate_plus, &mut forward)
        .unwrap();
    operator
        .residual(TIME, &minus, &rate_minus, &mut backward)
        .unwrap();
    let finite_difference = forward
        .iter()
        .zip(&backward)
        .map(|(a, b)| (a - b) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    assert_close(
        &tangent,
        &finite_difference,
        1.0e-7,
        "fan-out finite difference",
    );
    let adjoint = probe_vector(n, 4.2, 1.0);
    let mut transpose = vec![0.0; n];
    operator
        .vector_jacobian_product_shifted(TIME, &state, &rate, &adjoint, RATE_SHIFT, &mut transpose)
        .unwrap();
    assert_close(
        &[dot(&tangent, &adjoint)],
        &[dot(&direction, &transpose)],
        AGREEMENT,
        "fan-out transpose",
    );
}

#[test]
fn self_referential_output_bind_is_refused_before_evaluation() {
    let source = r#"
module systems.self_heat;
use physics.thermal.{HeatConduction};
pub system SelfHeat {
    domain body { dimension = 2; coordinates = cartesian; }
    instance thermal: HeatConduction(body = body);
    bind thermal.Q <- thermal.heating;
}
"#;
    let thermal = THERMAL.replace(
        "output temperature: ThermodynamicTemperature on body = T;",
        "output temperature: ThermodynamicTemperature on body = T;\n    output heating: VolumetricHeatSource on body = Q + rho * cp * dt(T);",
    );
    let mut sources = modules();
    sources.insert("physics.thermal".into(), thermal);
    let closure = resolve_module_closure(source, &sources).unwrap();
    let (units, kinds) = registries();
    let compilation =
        compile_system(&closure, Registries::new(&units, &kinds), "SelfHeat").unwrap();
    let operator = compile_system_operator(&compilation).unwrap();
    let tagged = unit_square(1);
    let ids = finitum::SystemIdMap::from_scientia(&operator.operator).unwrap();
    let layout = BlockLayout::new_keyed(
        ids.variables()
            .iter()
            .map(|v| (v.id, v.local, tagged.mesh.vertices().len(), 1)),
    )
    .unwrap();
    let result =
        SystemRealizationPlan::composed(&operator, tagged.mesh, layout, SystemQuadrature::Richest);
    assert!(
        matches!(result, Err(FinitumError::UnsupportedRealization(message)) if message.contains("algebraic loop"))
    );
}
