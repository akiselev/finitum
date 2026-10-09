//! Shared two-material `Conduction` fixtures for the connected-diagonal proofs
//! (`nested_p1_diagonal.rs`, `planar_p1.rs`): compilation, bound operators with constant or
//! storage-bearing state-dependent closures, and the bounded exhaustive reduced-JVP probe of
//! `diag(Rᵀ J R)` that is the oracle for `ConnectionRealizationPlan::system_linearized_diagonal`.
use finitum::{
    AffineConstraint, BlockLayout, ConnectionRealizationPlan, ConstraintSet, DofId, Mesh,
    MeshProfile, PointEvaluation, ReducedSystemOperator, SystemConstitutiveInput, SystemOperator,
    SystemQuadrature, SystemRealizationPlan, TaggedMesh, realize,
};
use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, IntegralOperatorFactorization, NoImports,
    OperatorSystem, Registries, SymbolId, SystemCompilation, SystemOperatorCompilation,
    TensorInputId, TensorInputRole, compile_system, compile_system_operator,
    resolve_module_closure,
};
use std::collections::{BTreeMap, BTreeSet};

pub const STEADY: &str = include_str!("../fixtures/two-material-conduction.res");
pub const TRANSIENT: &str = include_str!("../fixtures/two-material-transient-conduction.res");

pub fn compile(source: &str) -> (SystemCompilation, SystemOperatorCompilation) {
    let closure = resolve_module_closure(source, &NoImports).unwrap();
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
    (compiled, operators)
}

/// Symbol names of `side`'s model, keyed by symbol id (to bind closures by name).
pub fn symbol_names(
    compiled: &SystemCompilation,
    operators: &SystemOperatorCompilation,
    side: usize,
) -> BTreeMap<SymbolId, String> {
    compiled
        .model(operators.model_systems[side].0)
        .symbols
        .iter()
        .map(|symbol| (symbol.id, symbol.name.clone()))
        .collect()
}

/// The identifier of `symbol`'s active basis input (of the given derivative) in `integral`.
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

/// The left (`x` in `[0, 0.5]`) and right (`x` in `[0.5, 1]`) unit-section boxes, `n`
/// subdivisions per axis; their shared interface is the plane `x == 0.5`.
pub fn meshes(n: usize) -> [TaggedMesh; 2] {
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

/// The single non-basis input of the steady `Conduction`'s one equation is the whole flux
/// `q = -k * grad(T)` (bound as one opaque vector constitutive): a constant `k` makes the
/// interior Jacobian exact and state-independent, linear in `grad(T)` alone.
pub fn constant_flux(model: &OperatorSystem, conductivity: f64) -> Vec<SystemConstitutiveInput> {
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
                        "connected_conduction/q",
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

/// Coefficients of the storage-bearing `Conduction`: `k(T) = k0 * (1 + k1 * T)`, `rho`, `cp`.
#[derive(Clone, Copy)]
pub struct Storage {
    pub k0: f64,
    pub k1: f64,
    pub rho: f64,
    pub cp: f64,
}

/// Every non-basis input of the storage-bearing `Conduction`, bound by symbol name: the whole
/// flux `q = -k(T) * grad(T)` with the STATE-DEPENDENT conductivity `k(T) = k0 * (1 + k1 * T)`
/// (so the Jacobian depends on the linearization point, including the trace entries the
/// connection supplies) and the constant storage coefficients `rho`, `cp` of
/// `rho * cp * dt(T)` (so the Jacobian carries a rate term `rate_shift` must scale).
pub fn storage_closures(
    model: &OperatorSystem,
    names: &BTreeMap<SymbolId, String>,
    storage: Storage,
) -> Vec<SystemConstitutiveInput> {
    let t_symbol = model.field_order[0];
    let mut built = Vec::new();
    for block in &model.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let equation = block.equation.clone();
                let index = integral.integral_index;
                let name = names[&input.binding.symbol].as_str();
                let bound = match name {
                    "q" => {
                        let grad_t =
                            active_input(integral, t_symbol, DerivativeEvaluation::Gradient);
                        let t = active_input(integral, t_symbol, DerivativeEvaluation::Value);
                        let Storage { k0, k1, .. } = storage;
                        SystemConstitutiveInput::try_new(
                            equation,
                            index,
                            input.id,
                            3,
                            "connected_conduction/q(T)",
                            move |point: &PointEvaluation| {
                                let k = k0 * (1.0 + k1 * point.input_values(t).unwrap()[0]);
                                Ok(point
                                    .input_values(grad_t)
                                    .unwrap()
                                    .iter()
                                    .map(|g| -k * g)
                                    .collect())
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                let k = k0 * (1.0 + k1 * point.input_values(t).unwrap()[0]);
                                let dk = k0 * k1 * direction.input_values(t).unwrap()[0];
                                Ok(point
                                    .input_values(grad_t)
                                    .unwrap()
                                    .iter()
                                    .zip(direction.input_values(grad_t).unwrap())
                                    .map(|(g, dg)| -dk * g - k * dg)
                                    .collect())
                            },
                        )
                    }
                    "rho" | "cp" => {
                        let value = if name == "rho" {
                            storage.rho
                        } else {
                            storage.cp
                        };
                        SystemConstitutiveInput::try_new(
                            equation,
                            index,
                            input.id,
                            1,
                            format!("connected_conduction/{name}"),
                            move |_point: &PointEvaluation| Ok(vec![value]),
                            move |_point: &PointEvaluation, _direction: &PointEvaluation| {
                                Ok(vec![0.0])
                            },
                        )
                    }
                    other => panic!("unexpected non-basis input {other}"),
                };
                built.push(bound.unwrap());
            }
        }
    }
    built
}

/// Binds `model` over `mesh` under `plan` with the given closures (scalar nodal P1 layout).
pub fn bind(
    model: OperatorSystem,
    mesh: &Mesh,
    plan: &ConnectionRealizationPlan,
    built: Vec<SystemConstitutiveInput>,
) -> SystemOperator {
    let field = model.field_order[0];
    let layout = BlockLayout::new(vec![(field, mesh.vertices().len(), 1)]).unwrap();
    let plan = SystemRealizationPlan::with_connections(
        model,
        mesh.clone(),
        layout,
        SystemQuadrature::Richest,
        std::slice::from_ref(plan),
    )
    .unwrap();
    plan.bind_kernels(built, BTreeMap::new()).unwrap()
}

/// The first vertex of `mesh` on the plane `x == plane` (an exterior face away from the
/// interface at `x == 0.5`, so never a trace vertex).
pub fn vertex_on(mesh: &Mesh, plane: f64) -> usize {
    mesh.vertices()
        .iter()
        .position(|p| (p[0] - plane).abs() <= 1e-12)
        .unwrap()
}

/// One essential (Dirichlet) constraint at `vertex`, nothing else.
pub fn essential(dimension: usize, vertex: usize, value: f64) -> ConstraintSet {
    ConstraintSet::new(
        dimension,
        [AffineConstraint {
            target: DofId(vertex),
            dependencies: Vec::new(),
            offset: value,
        }],
    )
    .unwrap()
}

/// A spatially affine field, so P1 interpolation reproduces it exactly at every vertex.
pub fn affine(p: &[f64], base: f64, gx: f64, gy: f64, gz: f64) -> f64 {
    base + gx * p[0] + gy * p[1] + gz * p[2]
}

/// `diag(Rᵀ J R)` of the connected reduced system by bounded exhaustive probing: every unit
/// direction `e` of the connection-reduced space is prolongated through the pair
/// (`d = R e`) and pushed through each REDUCED operator's own `jacobian_vector_product` with
/// the rate direction `rate_shift * d` (the shifted column `linearized_diagonal` probes), the
/// physical outputs are restricted by `Rᵀ`, and the target rows are the pair's own equation
/// rows `direction_residual` -- exactly the composition Krasis's
/// `ConnectedSystemOperator::jacobian_vector_product` performs over its leaves, including its
/// linearization point: the connection-reduced `states`/`rates` expanded through the pair
/// first, so operator B is linearized at the weighted trace of operator A's masters, not at
/// whatever the caller's slice holds at the target rows.
pub fn exhaustive_diagonal(
    pair: &ConstraintSet,
    operators: [&ReducedSystemOperator; 2],
    time: f64,
    states: [&[f64]; 2],
    rates: [&[f64]; 2],
    rate_shift: f64,
) -> Vec<f64> {
    let dim_a = operators[0].operator().dimension();
    let dimension = pair.dof_count();
    let mut reduced_state = states[0].to_vec();
    reduced_state.extend_from_slice(states[1]);
    let mut reduced_rate = rates[0].to_vec();
    reduced_rate.extend_from_slice(rates[1]);
    let state = pair.expand(&reduced_state).unwrap();
    let rate = pair.expand_homogeneous(&reduced_rate).unwrap();
    let (state_a, state_b) = state.split_at(dim_a);
    let (rate_a, rate_b) = rate.split_at(dim_a);
    let mut unit = vec![0.0; dimension];
    let mut expected = vec![0.0; dimension];
    for probe in 0..dimension {
        unit[probe] = 1.0;
        let delta = pair.expand_homogeneous(&unit).unwrap();
        let rate_delta: Vec<f64> = delta.iter().map(|d| rate_shift * d).collect();
        let (delta_a, delta_b) = delta.split_at(dim_a);
        let (rate_delta_a, rate_delta_b) = rate_delta.split_at(dim_a);
        let mut out_a = vec![0.0; dim_a];
        let mut out_b = vec![0.0; dimension - dim_a];
        operators[0]
            .jacobian_vector_product(time, state_a, rate_a, delta_a, rate_delta_a, &mut out_a)
            .unwrap();
        operators[1]
            .jacobian_vector_product(time, state_b, rate_b, delta_b, rate_delta_b, &mut out_b)
            .unwrap();
        let mut physical = out_a;
        physical.extend(out_b);
        let mut restricted = pair.restrict_transpose(&physical).unwrap();
        for constraint in pair.constraints() {
            restricted[constraint.target.0] =
                pair.direction_residual(&unit, constraint.target).unwrap();
        }
        expected[probe] = restricted[probe];
        unit[probe] = 0.0;
    }
    expected
}

/// Entrywise agreement to `1e-9` relative; returns the worst relative discrepancy.
pub fn assert_diagonal_matches(label: &str, diagonal: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(diagonal.len(), expected.len(), "{label}: extent");
    let mut worst = 0.0_f64;
    for (row, (&got, &want)) in diagonal.iter().zip(expected).enumerate() {
        let relative = (got - want).abs() / want.abs().max(1.0);
        worst = worst.max(relative);
        assert!(
            relative <= 1e-9,
            "{label}: row {row}: diagonal {got} != exhaustive reduced-JVP probe {want} \
             (relative {relative:.3e})"
        );
    }
    worst
}

/// Non-vacuity: at least one interface-adjacent master row differs from the unconnected
/// operator's own diagonal (a formula that silently ignores the connection would fail here),
/// every connection-target row is the identity row, and every essential row of either operator
/// is the identity row where the unreduced physical diagonal is not `1.0`.
pub fn assert_row_structure(
    label: &str,
    pair: &ConstraintSet,
    diagonal: &[f64],
    plain: [&[f64]; 2],
    essential_rows: [usize; 2],
) {
    let dim_a = plain[0].len();
    let masters: BTreeSet<usize> = pair
        .constraints()
        .flat_map(|c| c.dependencies.iter().map(|d| d.dof.0))
        .collect();
    assert!(
        !masters.is_empty(),
        "{label}: plan produced no dependencies"
    );
    assert!(
        masters.iter().any(|&master| {
            (diagonal[master] - plain[0][master]).abs() > 1e-9 * plain[0][master].abs().max(1.0)
        }),
        "{label}: connected diagonal at every master row equals the unconnected operator's own \
         diagonal; the interface cross term was never exercised"
    );
    for constraint in pair.constraints() {
        assert_eq!(diagonal[constraint.target.0], 1.0, "{label}: target row");
    }
    for (side, offset) in [(0, 0), (1, dim_a)] {
        let row = essential_rows[side];
        assert!(
            (plain[side][row] - 1.0).abs() > 1e-9,
            "{label}: operator {side}'s physical diagonal at its essential row is already 1.0; \
             the identity-row claim would be vacuous"
        );
        assert_eq!(
            diagonal[offset + row],
            1.0,
            "{label}: operator {side}'s essential row {row}"
        );
    }
}
