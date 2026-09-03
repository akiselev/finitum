//! SC-W1 Finitum side, items 1 and 2 (`sinbad/ARCHITECTURE.md` §2.3, §2.4, §8): system-level
//! ids (`SysVarId`/`SysResId` with Finitum's own origin table `SystemIdMap`) keying the
//! `BlockLayout`, and public per-`(row, column)` block actions and transposes on
//! `SystemOperator`.
//!
//! Fixture: the hermetic two-field transient nonlinear system of
//! `tests/w7_p_system_state_dependent.rs` (property tangents crossing the fields), so the
//! off-diagonal blocks are genuinely state-dependent chain-rule blocks.

use finitum::{
    BlockLayout, FinitumError, InstanceId, MeshProfile, PointEvaluation, SysResId, SysVarId,
    SystemConstitutiveInput, SystemIdMap, SystemOperator, SystemRealizationPlan, TaggedMesh,
    realize,
};
use methodus::{EvaluationContext, LinearOperator, TransposableOperator};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, SemanticModel, SymbolId,
    TensorInputRole, compile_operator_system, compile_semantics,
};
use std::collections::BTreeMap;

const COUPLED: &str = r#"
module w7_sc_w1.coupled;
model Coupled {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field a: state scalar H1(order=1) on Omega { time_role = differential; };
  field b: state scalar H1(order=1) on Omega { time_role = differential; };
  property ka = diffusivity_a(b);
  property kb = diffusivity_b(a);
  property ca = capacity_a(a);
  source fa: VolumetricSource;
  source fb: VolumetricSource;
  equation ea on Omega { ca * dt(a) - div(ka * grad(a)) = fa; }
  equation eb on Omega { dt(b) - div(kb * grad(b)) + a * b = fb; }
  boundary walls_a on boundary("walls") { dirichlet a = 0; }
  boundary walls_b on boundary("walls") { dirichlet b = 0; }
}
"#;

const IDENTITY_TOLERANCE: f64 = 1.0e-12;

fn ka(b: f64) -> f64 {
    1.0 + 0.3 * b * b
}
fn d_ka(b: f64, db: f64) -> f64 {
    0.6 * b * db
}
fn kb(a: f64) -> f64 {
    1.0 + 0.5 * a
}
fn d_kb(_a: f64, da: f64) -> f64 {
    0.5 * da
}
fn ca(a: f64) -> f64 {
    1.0 + 0.2 * a * a
}
fn d_ca(a: f64, da: f64) -> f64 {
    0.4 * a * da
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn probe_vector(dimension: usize, seed: f64, scale: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| scale * ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

fn assert_identity(left: f64, right: f64, what: &str) {
    let scale = left.abs().max(right.abs()).max(1.0e-300);
    assert!(
        (left - right).abs() <= IDENTITY_TOLERANCE * scale,
        "{what}: {left} != {right}, relative discrepancy {}",
        (left - right).abs() / scale
    );
}

fn assert_close(actual: &[f64], expected: &[f64], relative: f64, what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= relative * scale,
            "{what} at {index}: {actual} != {expected}"
        );
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

struct Compiled {
    model: SemanticModel,
    system: OperatorSystem,
}

fn compile() -> Compiled {
    let compilation = compile_semantics(COUPLED, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(&compilation.semantic, "Coupled", &["ea", "eb"]).unwrap();
    Compiled {
        model: compilation.semantic.models[0].clone(),
        system,
    }
}

fn symbol(model: &SemanticModel, name: &str) -> SymbolId {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name)
        .map(|symbol| symbol.id)
        .unwrap_or_else(|| panic!("model has no symbol {name}"))
}

fn closure_constitutive(compiled: &Compiled) -> Vec<SystemConstitutiveInput> {
    let model = &compiled.model;
    let mut constitutive = Vec::new();
    for block in &compiled.system.blocks {
        for integral in &block.factorization.integrals {
            let value_input = |field: &str| {
                let field = symbol(model, field);
                integral
                    .primal
                    .inputs
                    .iter()
                    .find(|input| {
                        input.source == InputSourceRequirement::Basis
                            && input.role == TensorInputRole::Active
                            && input.binding.symbol == field
                            && input.binding.evaluation.derivative == DerivativeEvaluation::Value
                    })
                    .map(|input| input.id)
                    .expect("the property's field has a Value-kind active input here")
            };
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = model.symbols[input.binding.symbol.index()].name.as_str();
                let equation = block.equation.clone();
                let index = integral.integral_index;
                let built = match name {
                    "ka" => {
                        let b = value_input("b");
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "w7_sc_w1/ka",
                            move |point: &PointEvaluation| {
                                vec![ka(point.input_values(b).unwrap()[0])]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                vec![d_ka(
                                    point.input_values(b).unwrap()[0],
                                    direction.input_values(b).unwrap()[0],
                                )]
                            },
                        )
                    }
                    "kb" => {
                        let a = value_input("a");
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "w7_sc_w1/kb",
                            move |point: &PointEvaluation| {
                                vec![kb(point.input_values(a).unwrap()[0])]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                vec![d_kb(
                                    point.input_values(a).unwrap()[0],
                                    direction.input_values(a).unwrap()[0],
                                )]
                            },
                        )
                    }
                    "ca" => {
                        let a = value_input("a");
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "w7_sc_w1/ca",
                            move |point: &PointEvaluation| {
                                vec![ca(point.input_values(a).unwrap()[0])]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                vec![d_ca(
                                    point.input_values(a).unwrap()[0],
                                    direction.input_values(a).unwrap()[0],
                                )]
                            },
                        )
                    }
                    "fa" | "fb" => {
                        let value = if name == "fa" { 1.0 } else { 0.5 };
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            format!("w7_sc_w1/{name}"),
                            move |_: &PointEvaluation| vec![value],
                            |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                        )
                    }
                    other => panic!("unexpected non-basis input {other}"),
                };
                constitutive.push(built.unwrap());
            }
        }
    }
    constitutive
}

fn build(compiled: &Compiled, tagged: &TaggedMesh) -> SystemOperator {
    let a = symbol(&compiled.model, "a");
    let b = symbol(&compiled.model, "b");
    let vertex_count = tagged.mesh.vertices().len();
    let layout = BlockLayout::new([(a, vertex_count, 1), (b, vertex_count, 1)]).unwrap();
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), tagged.mesh.clone(), layout).unwrap();
    plan.bind_kernels(closure_constitutive(compiled), BTreeMap::new())
        .unwrap()
}

/// The one-instance map is the identity: `SysVarId(symbol.0)` per field, `SysResId(k)` per
/// equation block, root paths unprefixed, and the plan's layout is keyed by the same ids.
#[test]
fn one_instance_system_ids_are_the_identity_maps_and_key_the_layout() {
    let compiled = compile();
    let a = symbol(&compiled.model, "a");
    let b = symbol(&compiled.model, "b");
    let ids = SystemIdMap::one_instance(&compiled.system).unwrap();
    assert_eq!(ids.instances().len(), 1);
    assert_eq!(ids.instances()[0].instance, InstanceId(0));
    assert_eq!(ids.instances()[0].model, "Coupled");
    assert_eq!(
        ids.instances()[0].semantic_digest,
        compiled.system.source_semantic_digest
    );
    assert_eq!(ids.variables().len(), 2);
    assert_eq!(ids.variable(InstanceId(0), a), Some(SysVarId(a.0)));
    assert_eq!(ids.variable(InstanceId(0), b), Some(SysVarId(b.0)));
    assert_eq!(ids.variable_origin(SysVarId(a.0)).unwrap().local, a);
    assert_eq!(ids.residuals().len(), 2);
    assert_eq!(ids.residual(InstanceId(0), "ea"), Some(SysResId(0)));
    assert_eq!(ids.residual(InstanceId(0), "eb"), Some(SysResId(1)));
    assert_eq!(ids.residual_origin(SysResId(1)).unwrap().row, b);
    assert_eq!(ids.residual_path(SysResId(0)).as_deref(), Some("ea"));
    assert_eq!(ids.variable_path(SysVarId(a.0)), Some(a.to_string()));
    assert_eq!(ids.residual(InstanceId(1), "ea"), None);
    assert_eq!(ids.residual_origin(SysResId(7)), None);
    assert_eq!(
        ids.identity(),
        SystemIdMap::one_instance(&compiled.system)
            .unwrap()
            .identity()
    );

    let tagged = unit_square(2);
    let operator = build(&compiled, &tagged);
    assert_eq!(operator.system_ids(), &ids);
    assert_eq!(operator.plan().system_ids(), &ids);
    let layout = operator.layout();
    for variable in ids.variables() {
        let block = layout.block_by_variable(variable.id).unwrap();
        assert_eq!(block.symbol, variable.local);
        assert_eq!(block.variable, variable.id);
        assert_eq!(layout.block(variable.local).unwrap(), block);
    }
    assert_eq!(
        layout.variables().collect::<Vec<_>>(),
        vec![SysVarId(a.0), SysVarId(b.0)]
    );
    let vector = probe_vector(layout.extent(), 1.0, 1.0);
    assert_eq!(
        layout.values_by_variable(&vector, SysVarId(b.0)).unwrap(),
        layout.values(&vector, b).unwrap()
    );
}

/// Two instances of one model compose to dense ids in instance-then-local order; a keyed
/// layout over them is addressable by system variable only (the shared symbols are
/// ambiguous), and the origin table records every instance's receipt.
#[test]
fn composed_system_ids_are_dense_and_key_a_two_instance_layout() {
    let compiled = compile();
    let a = symbol(&compiled.model, "a");
    let b = symbol(&compiled.model, "b");
    let (low, high) = if a < b { (a, b) } else { (b, a) };
    let ids =
        SystemIdMap::compose(&[("left", &compiled.system), ("right", &compiled.system)]).unwrap();
    assert_eq!(ids.instances().len(), 2);
    assert_eq!(ids.instances()[1].name, "right");
    assert_eq!(ids.instances()[1].model, "Coupled");
    assert_eq!(
        ids.variables().iter().map(|v| v.id).collect::<Vec<_>>(),
        (0..4).map(SysVarId).collect::<Vec<_>>()
    );
    assert_eq!(ids.variable(InstanceId(0), low), Some(SysVarId(0)));
    assert_eq!(ids.variable(InstanceId(0), high), Some(SysVarId(1)));
    assert_eq!(ids.variable(InstanceId(1), low), Some(SysVarId(2)));
    assert_eq!(ids.variable(InstanceId(1), high), Some(SysVarId(3)));
    assert_eq!(ids.residual(InstanceId(0), "eb"), Some(SysResId(1)));
    assert_eq!(ids.residual(InstanceId(1), "ea"), Some(SysResId(2)));
    assert_eq!(ids.residual_path(SysResId(3)).as_deref(), Some("right.eb"));
    assert_eq!(ids.variable_path(SysVarId(2)), Some(format!("right/{low}")));
    assert_ne!(
        ids.identity(),
        SystemIdMap::one_instance(&compiled.system)
            .unwrap()
            .identity()
    );

    let layout = BlockLayout::new_keyed(
        ids.variables()
            .iter()
            .map(|variable| (variable.id, variable.local, 5, 1)),
    )
    .unwrap();
    assert_eq!(layout.extent(), 20);
    assert_eq!(layout.block_by_variable(SysVarId(3)).unwrap().offset, 15);
    assert_eq!(layout.block_by_variable(SysVarId(3)).unwrap().symbol, high);
    assert!(
        layout.block(a).is_none() && layout.block(b).is_none(),
        "a repeated per-model symbol is not addressable by symbol"
    );
    assert!(layout.values(&[0.0; 20], a).is_err());
    assert_eq!(
        layout
            .values_by_variable(&[0.0; 20], SysVarId(2))
            .unwrap()
            .len(),
        5
    );

    assert!(matches!(
        BlockLayout::new_keyed([(SysVarId(0), a, 3, 1), (SysVarId(0), b, 3, 1)]),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        SystemIdMap::compose(&[("same", &compiled.system), ("same", &compiled.system)]),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        SystemIdMap::compose(&[]),
        Err(FinitumError::InvalidRealization(_))
    ));
}

/// Every `(row, column)` block action equals the matching slice of the full JVP with the
/// direction masked to the column; the blocks of one row sum to the row of the full JVP; each
/// block transpose is the exact transpose of the block action; the off-diagonal blocks are
/// nonzero at a nonzero state (the chain-rule tangents) and vanish at zero.
#[test]
fn block_actions_partition_the_full_jacobian_and_transpose_exactly() {
    let compiled = compile();
    let tagged = unit_square(3);
    let operator = build(&compiled, &tagged);
    let ids = operator.system_ids().clone();
    let layout = operator.layout().clone();
    let dimension = operator.dimension();
    let time = 0.3;
    let state = probe_vector(dimension, 0.1, 0.8);
    let rate = probe_vector(dimension, 2.7, 0.4);
    let rate_shift = 2.5;

    for residual in ids.residuals() {
        let row_block = layout.block(residual.row).unwrap().clone();
        let mut row_sum = vec![0.0; row_block.extent];
        let direction = probe_vector(dimension, 5.3, 1.0);
        for variable in ids.variables() {
            let column_block = layout.block_by_variable(variable.id).unwrap().clone();
            let column_direction =
                direction[column_block.offset..column_block.offset + column_block.extent].to_vec();

            // Block JVP against the full JVP with a masked direction.
            let mut masked = vec![0.0; dimension];
            masked[column_block.offset..column_block.offset + column_block.extent]
                .copy_from_slice(&column_direction);
            let masked_rate = masked.iter().map(|v| rate_shift * v).collect::<Vec<_>>();
            let mut full = vec![0.0; dimension];
            operator
                .jacobian_vector_product(time, &state, &rate, &masked, &masked_rate, &mut full)
                .unwrap();
            let mut block_output = vec![0.0; row_block.extent];
            operator
                .block_jacobian_vector_product(
                    residual.id,
                    variable.id,
                    time,
                    &state,
                    &rate,
                    &column_direction,
                    rate_shift,
                    &mut block_output,
                )
                .unwrap();
            assert_close(
                &block_output,
                &full[row_block.offset..row_block.offset + row_block.extent],
                1.0e-13,
                &format!("block ({}, {}) JVP", residual.id, variable.id),
            );
            for (sum, value) in row_sum.iter_mut().zip(&block_output) {
                *sum += value;
            }

            // Exact transpose.
            let adjoint = probe_vector(row_block.extent, 9.1, 1.0);
            let mut cotangent = vec![0.0; column_block.extent];
            operator
                .block_vector_jacobian_product(
                    residual.id,
                    variable.id,
                    time,
                    &state,
                    &rate,
                    &adjoint,
                    rate_shift,
                    &mut cotangent,
                )
                .unwrap();
            assert_identity(
                dot(&adjoint, &block_output),
                dot(&cotangent, &column_direction),
                &format!("block ({}, {}) transpose", residual.id, variable.id),
            );

            let off_diagonal = residual.row != variable.local;
            if off_diagonal {
                assert!(
                    block_output.iter().any(|v| v.abs() > 1.0e-6),
                    "off-diagonal block ({}, {}) must carry the chain-rule tangent",
                    residual.id,
                    variable.id
                );
                let zero = vec![0.0; dimension];
                let mut at_zero = vec![0.0; row_block.extent];
                operator
                    .block_action(residual.id, variable.id, &column_direction, &mut at_zero)
                    .unwrap();
                assert!(
                    at_zero.iter().all(|v| v.abs() <= 1.0e-14),
                    "off-diagonal block ({}, {}) vanishes at the zero state",
                    residual.id,
                    variable.id
                );
                let mut transpose_at_zero = vec![0.0; column_block.extent];
                operator
                    .block_transpose_action(
                        residual.id,
                        variable.id,
                        &adjoint,
                        &mut transpose_at_zero,
                    )
                    .unwrap();
                assert!(transpose_at_zero.iter().all(|v| v.abs() <= 1.0e-14));
                let _ = zero;
            }
        }
        let rate_direction = direction.iter().map(|v| rate_shift * v).collect::<Vec<_>>();
        let mut full = vec![0.0; dimension];
        operator
            .jacobian_vector_product(time, &state, &rate, &direction, &rate_direction, &mut full)
            .unwrap();
        assert_close(
            &row_sum,
            &full[row_block.offset..row_block.offset + row_block.extent],
            1.0e-13,
            &format!("row {} block sum", residual.id),
        );
    }
}

/// `SystemOperator::block_operator` is a rectangular Methodus operator whose action and
/// transpose are the block JVP/VJP; refusals are typed.
#[test]
fn system_block_operator_is_a_transposable_methodus_operator_and_refuses_bad_shapes() {
    let compiled = compile();
    let tagged = unit_square(2);
    let operator = build(&compiled, &tagged);
    let ids = operator.system_ids().clone();
    let dimension = operator.dimension();
    let state = probe_vector(dimension, 0.4, 0.7);
    let rate = probe_vector(dimension, 1.9, 0.3);
    let row = ids.residual(InstanceId(0), "ea").unwrap();
    let a = symbol(&compiled.model, "a");
    let b = symbol(&compiled.model, "b");
    let column = ids.variable(InstanceId(0), b).unwrap();
    let block = operator
        .block_operator(row, column, 0.2, &state, &rate, 1.5)
        .unwrap();
    let rows = operator.layout().block(a).unwrap().extent;
    let columns = operator.layout().block(b).unwrap().extent;
    assert_eq!((block.rows(), block.columns()), (rows, columns));
    assert_eq!((block.row(), block.column()), (row, column));
    assert_eq!(block.rate_shift(), 1.5);
    let context = EvaluationContext::default();
    let x = probe_vector(columns, 3.3, 1.0);
    let y = probe_vector(rows, 7.7, 1.0);
    let mut ax = vec![0.0; rows];
    block.apply(&context, &x, &mut ax).unwrap();
    let mut aty = vec![0.0; columns];
    block.apply_transpose(&context, &y, &mut aty).unwrap();
    assert_identity(dot(&y, &ax), dot(&aty, &x), "block operator transpose");
    assert!(ax.iter().any(|v| v.abs() > 1.0e-6));

    assert!(matches!(
        operator.block_jacobian_vector_product(
            SysResId(9),
            column,
            0.0,
            &state,
            &rate,
            &x,
            0.0,
            &mut vec![0.0; rows]
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        operator.block_action(row, SysVarId(42), &x, &mut vec![0.0; rows]),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        operator.block_action(row, column, &x[..columns - 1], &mut vec![0.0; rows]),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        operator.block_transpose_action(row, column, &y, &mut vec![0.0; columns + 1]),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(
        operator
            .block_operator(row, column, f64::NAN, &state, &rate, 0.0)
            .is_err()
    );
}
