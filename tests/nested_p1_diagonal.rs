//! W9 Finitum SC-W3 package 1: `ConnectionRealizationPlan::system_linearized_diagonal` proved
//! against bounded exhaustive unit-vector probing of the same two-operator connected reduced
//! Jacobian (`diag(Rᵀ J R)`). The oracle is the two REDUCED operators' own
//! `jacobian_vector_product` (so each operator's essential rows are seen as the identity rows
//! they are), driven with the rate direction `rate_shift · d` `linearized_diagonal` probes, and
//! composed through the connection exactly as Krasis's
//! `ConnectedSystemOperator::jacobian_vector_product` composes its leaves (the linearization
//! point expanded through the pair first). Two shapes:
//!
//! - the steady `nested_p1` heat fixture, with one essential constraint per side;
//! - a storage-bearing (`rho * cp * dt(T)`), state-dependent-conductivity `nested_p1` fixture,
//!   where the diagonal must change with `rate_shift` and must be linearized at the
//!   connection-expanded point, not at the caller's raw fine-trace entries.
use finitum::{
    AffineConstraint, BlockLayout, ConnectionRealizationPlan, ConstraintSet, DofId, Mesh,
    MeshProfile, PointEvaluation, ReducedSystemOperator, RegionTagId, SystemConstitutiveInput,
    SystemOperator, SystemQuadrature, SystemRealizationPlan, TaggedMesh, realize,
};
use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, IntegralOperatorFactorization, NoImports,
    OperatorSystem, Registries, SymbolId, SystemCompilation, SystemOperatorCompilation,
    TensorInputId, TensorInputRole, compile_system, compile_system_operator,
    resolve_module_closure,
};
use std::collections::{BTreeMap, BTreeSet};

const STEADY: &str = include_str!("fixtures/two-material-conduction.res");
const TRANSIENT: &str = include_str!("fixtures/two-material-transient-conduction.res");

fn compile(source: &str) -> (SystemCompilation, SystemOperatorCompilation) {
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

/// The single non-basis input of the steady `Conduction`'s one equation is the whole flux
/// `q = -k * grad(T)` (bound as one opaque vector constitutive, mirroring how `-sigma(T) *
/// grad(V)` is bound in the corpus 08 fixture): a constant `k` makes the interior Jacobian
/// exact and state-independent, linear in `grad(T)` alone.
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
                        "nested_p1_diagonal/q",
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
struct Storage {
    k0: f64,
    k1: f64,
    rho: f64,
    cp: f64,
}

/// Every non-basis input of the storage-bearing `Conduction`, bound by symbol name: the whole
/// flux `q = -k(T) * grad(T)` with the STATE-DEPENDENT conductivity `k(T) = k0 * (1 + k1 * T)`
/// (so the Jacobian depends on the linearization point, including the fine-trace entries the
/// connection supplies) and the constant storage coefficients `rho`, `cp` of
/// `rho * cp * dt(T)` (so the Jacobian carries a rate term `rate_shift` must scale).
fn storage_closures(
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
                            "nested_p1_diagonal/q(T)",
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
                            format!("nested_p1_diagonal/{name}"),
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
fn bind(
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
fn vertex_on(mesh: &Mesh, plane: f64) -> usize {
    mesh.vertices()
        .iter()
        .position(|p| (p[0] - plane).abs() <= 1e-12)
        .unwrap()
}

/// One essential (Dirichlet) constraint at `vertex`, nothing else.
fn essential(dimension: usize, vertex: usize, value: f64) -> ConstraintSet {
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
fn affine(p: &[f64], base: f64, gx: f64, gy: f64, gz: f64) -> f64 {
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
fn exhaustive_diagonal(
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
fn assert_diagonal_matches(label: &str, diagonal: &[f64], expected: &[f64]) -> f64 {
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
fn assert_row_structure(
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

#[test]
fn nested_p1_diagonal_matches_bounded_exhaustive_probing() {
    let (compiled, operators) = compile(STEADY);
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

    let coarse_model = operators.model_systems[0].1.clone();
    let fine_model = operators.model_systems[1].1.clone();
    let coarse_operator = bind(
        coarse_model.clone(),
        &coarse.mesh,
        &nested,
        constant_flux(&coarse_model, 2.0),
    );
    let fine_operator = bind(
        fine_model.clone(),
        &fine.mesh,
        &nested,
        constant_flux(&fine_model, 3.0),
    );

    let dim_a = coarse_operator.dimension();
    let dim_b = fine_operator.dimension();
    // One essential constraint per side on a non-interface vertex (the far exterior faces).
    let essential_a = vertex_on(&coarse.mesh, 0.0);
    let essential_b = vertex_on(&fine.mesh, 1.0);
    let reduced_a = coarse_operator
        .reduced(essential(dim_a, essential_a, 300.0))
        .unwrap();
    let reduced_b = fine_operator
        .reduced(essential(dim_b, essential_b, 350.0))
        .unwrap();

    let state_a = vec![0.0; dim_a];
    let state_b = vec![0.0; dim_b];
    let rate_a = vec![0.0; dim_a];
    let rate_b = vec![0.0; dim_b];
    let rate_shift = 1.0;

    let diagonal = nested
        .system_linearized_diagonal(
            [&reduced_a, &reduced_b],
            0.0,
            [&state_a, &state_b],
            [&rate_a, &rate_b],
            rate_shift,
        )
        .unwrap()
        .expect("linear conduction over cell-only kernels admits a cell-local diagonal");

    let pair = nested.constraints([&reduced_a, &reduced_b]).unwrap();
    let dimension = dim_a + dim_b;
    assert_eq!(diagonal.len(), dimension);
    assert_eq!(pair.dof_count(), dimension);

    let expected = exhaustive_diagonal(
        &pair,
        [&reduced_a, &reduced_b],
        0.0,
        [&state_a, &state_b],
        [&rate_a, &rate_b],
        rate_shift,
    );
    let worst = assert_diagonal_matches("steady nested_p1", &diagonal, &expected);
    eprintln!("steady nested_p1 diagonal: worst relative discrepancy {worst:.3e}");

    let plain_a = coarse_operator
        .linearized_diagonal(0.0, &state_a, &rate_a, rate_shift)
        .unwrap()
        .expect("coarse operator admits its own cell-local diagonal");
    let plain_b = fine_operator
        .linearized_diagonal(0.0, &state_b, &rate_b, rate_shift)
        .unwrap()
        .expect("fine operator admits its own cell-local diagonal");
    assert_row_structure(
        "steady nested_p1",
        &pair,
        &diagonal,
        [&plain_a, &plain_b],
        [essential_a, essential_b],
    );
}

/// The storage-bearing, state-dependent fixture: the diagonal must carry `rate_shift` times the
/// storage mass (the `rate_direction[column] = rate_shift` plumbing of `linearized_block` and
/// `linearized_diagonal`), must still match the shifted probe at every shift, and must be
/// linearized at the connection-expanded point -- the caller's fine-trace (target) entries are
/// deliberately wrong here, and only the pair expansion makes operator B's Jacobian right.
#[test]
fn nested_p1_diagonal_sees_rate_shift_and_expands_the_point_through_the_connection() {
    let (compiled, operators) = compile(TRANSIENT);
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

    let storage = [
        Storage {
            k0: 2.0,
            k1: 1.0e-3,
            rho: 1.5,
            cp: 0.5,
        },
        Storage {
            k0: 3.0,
            k1: 2.0e-3,
            rho: 2.0,
            cp: 0.75,
        },
    ];
    let names = |side: usize| -> BTreeMap<SymbolId, String> {
        compiled
            .model(operators.model_systems[side].0)
            .symbols
            .iter()
            .map(|symbol| (symbol.id, symbol.name.clone()))
            .collect()
    };
    let coarse_model = operators.model_systems[0].1.clone();
    let fine_model = operators.model_systems[1].1.clone();
    let coarse_operator = bind(
        coarse_model.clone(),
        &coarse.mesh,
        &nested,
        storage_closures(&coarse_model, &names(0), storage[0]),
    );
    let fine_operator = bind(
        fine_model.clone(),
        &fine.mesh,
        &nested,
        storage_closures(&fine_model, &names(1), storage[1]),
    );
    let coarse_field = coarse_model.field_order[0];

    let dim_a = coarse_operator.dimension();
    let dim_b = fine_operator.dimension();
    let essential_a = vertex_on(&coarse.mesh, 0.0);
    let essential_b = vertex_on(&fine.mesh, 1.0);
    let reduced_a = coarse_operator
        .reduced(essential(dim_a, essential_a, 300.0))
        .unwrap();
    let reduced_b = fine_operator
        .reduced(essential(dim_b, essential_b, 350.0))
        .unwrap();
    let pair = nested.constraints([&reduced_a, &reduced_b]).unwrap();
    let targets: BTreeSet<usize> = pair.constraints().map(|c| c.target.0 - dim_a).collect();
    let masters: BTreeSet<usize> = pair
        .constraints()
        .flat_map(|c| c.dependencies.iter().map(|d| d.dof.0))
        .collect();

    // The connection-reduced point: an affine temperature and rate sampled on both meshes, the
    // essential rows holding their prescribed values, and the fine-trace (target) rows holding
    // a deliberately WRONG value (0 K, where the expansion gives ~300 K) -- a connection-reduced
    // state owns no meaning there, and `system_linearized_diagonal` must not read it.
    let value0 = |p: &[f64]| affine(p, 300.0, 20.0, 10.0, -5.0);
    let rate0 = |p: &[f64]| affine(p, 5.0, 1.0, 0.25, -0.75);
    let mut state_a: Vec<f64> = coarse.mesh.vertices().iter().map(|p| value0(p)).collect();
    let rate_a: Vec<f64> = coarse.mesh.vertices().iter().map(|p| rate0(p)).collect();
    state_a[essential_a] = 300.0;
    let mut state_b: Vec<f64> = fine.mesh.vertices().iter().map(|p| value0(p)).collect();
    let mut rate_b: Vec<f64> = fine.mesh.vertices().iter().map(|p| rate0(p)).collect();
    state_b[essential_b] = 350.0;
    for &target in &targets {
        state_b[target] = 0.0;
        rate_b[target] = 0.0;
    }
    let time = 0.3;

    // Non-vacuity of the expansion: linearized at the raw caller slice, operator B's own
    // diagonal differs at the trace rows from the one at the pair-expanded point.
    let mut reduced_point = state_a.clone();
    reduced_point.extend_from_slice(&state_b);
    let expanded = pair.expand(&reduced_point).unwrap();
    let (raw_state, raw_rate) = reduced_b
        .physical_state_and_rate(time, &state_b, &rate_b)
        .unwrap();
    let (expanded_state, _) = reduced_b
        .physical_state_and_rate(time, &expanded[dim_a..], &rate_b)
        .unwrap();
    let plain_raw = fine_operator
        .linearized_diagonal(time, &raw_state, &raw_rate, 0.0)
        .unwrap()
        .unwrap();
    let plain_expanded = fine_operator
        .linearized_diagonal(time, &expanded_state, &raw_rate, 0.0)
        .unwrap()
        .unwrap();
    assert!(
        targets.iter().any(|&row| {
            (plain_raw[row] - plain_expanded[row]).abs() > 1e-6 * plain_expanded[row].abs()
        }),
        "the fine operator's Jacobian does not depend on its trace values; the expansion proof \
         would be vacuous"
    );

    let shifts = [0.0, 3.0];
    let mut diagonals = Vec::new();
    for &rate_shift in &shifts {
        let diagonal = nested
            .system_linearized_diagonal(
                [&reduced_a, &reduced_b],
                time,
                [&state_a, &state_b],
                [&rate_a, &rate_b],
                rate_shift,
            )
            .unwrap()
            .expect("cell-only storage-bearing conduction admits a cell-local diagonal");
        let expected = exhaustive_diagonal(
            &pair,
            [&reduced_a, &reduced_b],
            time,
            [&state_a, &state_b],
            [&rate_a, &rate_b],
            rate_shift,
        );
        let label = format!("storage nested_p1 at rate_shift {rate_shift}");
        let worst = assert_diagonal_matches(&label, &diagonal, &expected);
        eprintln!("{label}: worst relative discrepancy {worst:.3e}");
        diagonals.push(diagonal);
    }

    // The diagonal changes with `rate_shift` at every free and master row (every vertex carries
    // storage mass), and on an interior coarse row the change is exactly the shift times the
    // row's own storage mass `rho * cp * M[i, i]` -- an independent cross-check through
    // `SystemOperator::mass_matrix`, not through the JVP machinery under test.
    let mass_a = coarse_operator.mass_matrix(coarse_field).unwrap();
    let mut interior_rows = 0;
    for row in 0..dim_a + dim_b {
        let is_identity = (row < dim_a && row == essential_a)
            || (row >= dim_a && (row - dim_a == essential_b || targets.contains(&(row - dim_a))));
        if is_identity {
            assert_eq!(diagonals[0][row], 1.0);
            assert_eq!(diagonals[1][row], 1.0);
            continue;
        }
        let change = diagonals[1][row] - diagonals[0][row];
        assert!(
            change > 1e-9 * diagonals[0][row].abs().max(1.0),
            "row {row}: diagonal did not grow with rate_shift ({} -> {})",
            diagonals[0][row],
            diagonals[1][row]
        );
        if row < dim_a && !masters.contains(&row) {
            let storage_mass = storage[0].rho * storage[0].cp * mass_a[row * dim_a + row];
            let want = (shifts[1] - shifts[0]) * storage_mass;
            assert!(
                (change - want).abs() <= 1e-9 * want.abs(),
                "row {row}: rate_shift change {change} != shift * rho * cp * M_ii = {want}"
            );
            interior_rows += 1;
        }
    }
    assert!(
        interior_rows > 0,
        "no interior coarse row to cross-check against the mass"
    );

    let plain_a = coarse_operator
        .linearized_diagonal(time, &expanded[..dim_a], &rate_a, shifts[0])
        .unwrap()
        .unwrap();
    assert_row_structure(
        "storage nested_p1",
        &pair,
        &diagonals[0],
        [&plain_a, &plain_expanded],
        [essential_a, essential_b],
    );
}
