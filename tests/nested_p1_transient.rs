//! W9 Finitum SC-W3 package 2: the proof checks Krasis needs to compose a transient `nested_p1`
//! connection -- fine-side primal trace continuity at any state, and the dual-residual
//! transpose-work balance identity at any state and rate -- exercised against real bound
//! operators over the `two-material-conduction.res` heat fixture. `trace_defects` had no test
//! coverage at all before this file, steady or transient.
//!
//! Differential/algebraic row classification, rate expansion (`ConstraintSet::expand_homogeneous`)
//! and transpose restriction (`ConstraintSet::restrict_transpose`) are already generic over any
//! `AffineConstraint` weight, so a `nested_p1` connection's constraints already carry exactly the
//! same shape Krasis's row classification and DAE composition (`krasis/src/connected.rs`)
//! consume from a `matching` connection: no realization-plan change was needed to make that
//! true, only the proof that it holds.
use finitum::{
    BlockLayout, ConnectionRealizationPlan, ConstraintSet, Mesh, MeshProfile, PointEvaluation,
    RegionTagId, SystemConstitutiveInput, SystemOperator, SystemQuadrature, SystemRealizationPlan,
    TaggedMesh, realize,
};
use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, IntegralOperatorFactorization, NoImports,
    OperatorSystem, Registries, SymbolId, TensorInputId, TensorInputRole, compile_system,
    compile_system_operator, resolve_module_closure,
};
use std::collections::BTreeMap;

fn meshes(n: usize) -> [TaggedMesh; 2] {
    [0, 1].map(|side| {
        realize(&MeshProfile::SimplexBox {
            dimension: 3,
            extent: vec![
                [side as f64 * 0.5, (side + 1) as f64 * 0.5],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            subdivisions: vec![n; 3],
        })
        .unwrap()
    })
}

fn active_input(
    integral: &IntegralOperatorFactorization,
    symbol: SymbolId,
    derivative: DerivativeEvaluation,
) -> TensorInputId {
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

/// The single non-basis input of `model`'s one equation is the whole flux `q = -k * grad(T)`.
fn constant_flux(model: &OperatorSystem, conductivity: f64) -> Vec<SystemConstitutiveInput> {
    let t_symbol = model.field_order[0];
    let mut built = Vec::new();
    for block in &model.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let grad_t = active_input(integral, t_symbol, DerivativeEvaluation::Gradient);
                built.push(
                    SystemConstitutiveInput::try_new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        3,
                        "nested_p1_transient/q",
                        move |point: &PointEvaluation| {
                            Ok(point
                                .input_values(grad_t)
                                .unwrap()
                                .iter()
                                .map(|g| -conductivity * g)
                                .collect())
                        },
                        move |_point: &PointEvaluation, direction: &PointEvaluation| {
                            Ok(direction
                                .input_values(grad_t)
                                .unwrap()
                                .iter()
                                .map(|dg| -conductivity * dg)
                                .collect())
                        },
                    )
                    .unwrap(),
                );
            }
        }
    }
    built
}

/// Binds `Conduction` over `mesh` with a constant conductivity and no essential constraints, so
/// the only structure under test is the connection's own weighted trace. Returns the field's
/// `SymbolId` alongside the bound operator (needed for `mass_matrix`).
fn conduction_operator(
    model: OperatorSystem,
    mesh: &Mesh,
    nested: &ConnectionRealizationPlan,
    conductivity: f64,
) -> (SystemOperator, SymbolId) {
    let field = model.field_order[0];
    let layout = BlockLayout::new(vec![(field, mesh.vertices().len(), 1)]).unwrap();
    let built = constant_flux(&model, conductivity);
    let plan = SystemRealizationPlan::with_connections(
        model,
        mesh.clone(),
        layout,
        SystemQuadrature::Richest,
        std::slice::from_ref(nested),
    )
    .unwrap();
    (plan.bind_kernels(built, BTreeMap::new()).unwrap(), field)
}

/// A spatially affine field, so P1 interpolation reproduces it exactly at every vertex.
fn affine(p: &[f64], base: f64, gy: f64, gz: f64) -> f64 {
    base + gy * p[1] + gz * p[2]
}

fn dense_matvec(matrix: &[f64], extent: usize, vector: &[f64]) -> Vec<f64> {
    (0..extent)
        .map(|row| {
            (0..extent)
                .map(|column| matrix[row * extent + column] * vector[column])
                .sum()
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn deterministic_probe(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|index| ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

struct Fixture {
    nested: ConnectionRealizationPlan,
    coarse: TaggedMesh,
    fine: TaggedMesh,
    coarse_operator: SystemOperator,
    fine_operator: SystemOperator,
    coarse_field: SymbolId,
    fine_field: SymbolId,
}

fn fixture() -> Fixture {
    let closure = resolve_module_closure(
        include_str!("fixtures/two-material-conduction.res"),
        &NoImports,
    )
    .unwrap();
    let compiled = compile_system(
        &closure,
        Registries::new(
            &UnitRegistry::si_bootstrap(),
            &QuantityKindRegistry::si_bootstrap(),
        ),
        "TwoMaterials",
    )
    .unwrap();
    let operators = compile_system_operator(&compiled).unwrap();
    let system = &compiled.system;
    let relation = &system.connections[0];
    let [coarse, _] = meshes(1);
    let [_, fine] = meshes(2);
    let a = &coarse.tags.facet_regions[&RegionTagId::new("x_max")];
    let b = &fine.tags.facet_regions[&RegionTagId::new("x_min")];
    let nested = ConnectionRealizationPlan::nested_p1(
        system,
        relation,
        [&coarse.mesh, &fine.mesh],
        [a, b],
        1e-12,
    )
    .unwrap();
    let (coarse_operator, coarse_field) = conduction_operator(
        operators.model_systems[0].1.clone(),
        &coarse.mesh,
        &nested,
        2.0,
    );
    let (fine_operator, fine_field) = conduction_operator(
        operators.model_systems[1].1.clone(),
        &fine.mesh,
        &nested,
        3.0,
    );
    Fixture {
        nested,
        coarse,
        fine,
        coarse_operator,
        fine_operator,
        coarse_field,
        fine_field,
    }
}

/// Fine-side primal trace continuity at any state: a constant trace, and a linear-in-time trace
/// sampled at two different times, both reproduced exactly through the map.
#[test]
fn fine_side_trace_continuity_holds_for_constant_and_linear_in_time_traces() {
    let f = fixture();
    let empty_a = ConstraintSet::new(f.coarse_operator.dimension(), vec![]).unwrap();
    let empty_b = ConstraintSet::new(f.fine_operator.dimension(), vec![]).unwrap();
    let reduced_a = f.coarse_operator.reduced(empty_a).unwrap();
    let reduced_b = f.fine_operator.reduced(empty_b).unwrap();

    let value0 = |p: &[f64]| affine(p, 300.0, 2.0, -3.0);
    let rate = |p: &[f64]| affine(p, 5.0, 0.25, -0.75);

    // Constant trace (rate zero everywhere): value0 alone.
    let values_a: Vec<f64> = f.coarse.mesh.vertices().iter().map(|p| value0(p)).collect();
    let values_b: Vec<f64> = f.fine.mesh.vertices().iter().map(|p| value0(p)).collect();
    let zero_a = vec![0.0; f.coarse_operator.dimension()];
    let zero_b = vec![0.0; f.fine_operator.dimension()];
    let (continuity, _) = f
        .nested
        .trace_defects(
            [&reduced_a, &reduced_b],
            [&values_a, &values_b],
            [&zero_a, &zero_b],
        )
        .unwrap();
    assert!(
        continuity <= 1e-12,
        "constant trace continuity defect {continuity:e} exceeds 1e-12"
    );

    // Linear-in-time trace value(x, t) = value0(x) + rate(x) * t, checked at two different times:
    // the map carries no time dependence of its own, so continuity must hold at each snapshot.
    for &time in &[0.3_f64, 1.7_f64] {
        let values_a: Vec<f64> = f
            .coarse
            .mesh
            .vertices()
            .iter()
            .map(|p| value0(p) + rate(p) * time)
            .collect();
        let values_b: Vec<f64> = f
            .fine
            .mesh
            .vertices()
            .iter()
            .map(|p| value0(p) + rate(p) * time)
            .collect();
        let (continuity, _) = f
            .nested
            .trace_defects(
                [&reduced_a, &reduced_b],
                [&values_a, &values_b],
                [&zero_a, &zero_b],
            )
            .unwrap();
        assert!(
            continuity <= 1e-12,
            "linear-in-time trace continuity defect {continuity:e} at t={time} exceeds 1e-12"
        );
    }

    // The rate field alone (what `dt` of the linear-in-time trace equals at every t) also
    // reproduces exactly through the same weighted map -- rate expansion is the same primal
    // interpolation as value expansion, since a trace constraint's offset is always zero.
    let rate_a: Vec<f64> = f.coarse.mesh.vertices().iter().map(|p| rate(p)).collect();
    let rate_b: Vec<f64> = f.fine.mesh.vertices().iter().map(|p| rate(p)).collect();
    let (rate_continuity, _) = f
        .nested
        .trace_defects(
            [&reduced_a, &reduced_b],
            [&rate_a, &rate_b],
            [&zero_a, &zero_b],
        )
        .unwrap();
    assert!(
        rate_continuity <= 1e-12,
        "rate trace continuity defect {rate_continuity:e} exceeds 1e-12"
    );
}

/// The dual-residual transpose-work balance identity at any state and rate: `trace_defects`'s
/// own balance number must equal the same accumulation `ConstraintSet::restrict_transpose`
/// performs, whether the residual pair is an arbitrary state-like vector or a genuinely
/// storage-bearing (mass-matrix) vector on both sides.
#[test]
fn dual_residual_balance_identity_matches_restrict_transpose_for_state_and_storage_residuals() {
    let f = fixture();
    let empty_a = ConstraintSet::new(f.coarse_operator.dimension(), vec![]).unwrap();
    let empty_b = ConstraintSet::new(f.fine_operator.dimension(), vec![]).unwrap();
    let reduced_a = f.coarse_operator.reduced(empty_a).unwrap();
    let reduced_b = f.fine_operator.reduced(empty_b).unwrap();
    let pair = f.nested.constraints([&reduced_a, &reduced_b]).unwrap();

    let dim_a = f.coarse_operator.dimension();
    let dim_b = f.fine_operator.dimension();

    let value0 = |p: &[f64]| affine(p, 300.0, 2.0, -3.0);
    let values_a: Vec<f64> = f.coarse.mesh.vertices().iter().map(|p| value0(p)).collect();
    let values_b: Vec<f64> = f.fine.mesh.vertices().iter().map(|p| value0(p)).collect();

    let masters: std::collections::BTreeSet<usize> = pair
        .constraints()
        .flat_map(|c| c.dependencies.iter().map(|d| d.dof.0))
        .collect();
    assert!(
        !masters.is_empty(),
        "nested_p1 plan produced no dependencies"
    );

    let check = |label: &str, residual_a: &[f64], residual_b: &[f64]| {
        let (continuity, balance) = f
            .nested
            .trace_defects(
                [&reduced_a, &reduced_b],
                [&values_a, &values_b],
                [residual_a, residual_b],
            )
            .unwrap();
        assert!(
            continuity <= 1e-12,
            "{label}: continuity defect {continuity:e} exceeds 1e-12"
        );
        let mut full_residual = residual_a.to_vec();
        full_residual.extend_from_slice(residual_b);
        let restricted = pair.restrict_transpose(&full_residual).unwrap();
        let expected_balance = masters
            .iter()
            .map(|&master| restricted[master].abs())
            .fold(0.0_f64, f64::max);
        let scale = expected_balance.max(1.0);
        let relative = (balance - expected_balance).abs() / scale;
        assert!(
            relative <= 1e-12,
            "{label}: trace_defects balance {balance} != restrict_transpose reference \
             {expected_balance} (relative {relative:e})"
        );
    };

    // At an arbitrary state-like residual pair (no physical meaning required for this identity
    // to hold -- it is an algebraic property of the transpose restriction itself).
    let residual_a = deterministic_probe(dim_a, 0.0);
    let residual_b = deterministic_probe(dim_b, 7.0);
    check("state residual", &residual_a, &residual_b);

    // At a genuinely storage-bearing residual on both sides: the L2 mass (Gram) matrix action
    // on a rate field is exactly the discrete term a `rho * cp * dt(T)` storage equation
    // contributes, on each side's own mesh and conductivity-independent mass matrix.
    let rate = |p: &[f64]| affine(p, 5.0, 0.25, -0.75);
    let rate_a: Vec<f64> = f.coarse.mesh.vertices().iter().map(|p| rate(p)).collect();
    let rate_b: Vec<f64> = f.fine.mesh.vertices().iter().map(|p| rate(p)).collect();
    let mass_a = f.coarse_operator.mass_matrix(f.coarse_field).unwrap();
    let mass_b = f.fine_operator.mass_matrix(f.fine_field).unwrap();
    let storage_residual_a = dense_matvec(&mass_a, dim_a, &rate_a);
    let storage_residual_b = dense_matvec(&mass_b, dim_b, &rate_b);
    check(
        "storage (mass) residual",
        &storage_residual_a,
        &storage_residual_b,
    );

    // Cross-check the same identity directly through `ConstraintSet::expand_homogeneous` /
    // `restrict_transpose` (the primitives `crate::constraint::apply_constrained_action` and
    // Krasis's own DAE composition both use), confirming it is the general adjoint identity
    // <R x, y> = <x, Rᵀ y> and not an artifact of `trace_defects`'s own bookkeeping.
    let mut full_rate = rate_a.clone();
    full_rate.extend_from_slice(&rate_b);
    let mut full_storage_residual = storage_residual_a.clone();
    full_storage_residual.extend_from_slice(&storage_residual_b);
    let expanded_rate = pair.expand_homogeneous(&full_rate).unwrap();
    let restricted_residual = pair.restrict_transpose(&full_storage_residual).unwrap();
    let left = dot(&expanded_rate, &full_storage_residual);
    let right = dot(&full_rate, &restricted_residual);
    let scale = left.abs().max(right.abs()).max(1.0);
    let relative = (left - right).abs() / scale;
    assert!(
        relative <= 1e-12,
        "adjoint identity <Rx,y> = <x,Rty> failed for the storage residual: {left} != {right} \
         (relative {relative:e})"
    );
}
