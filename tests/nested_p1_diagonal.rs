//! W9 Finitum SC-W3 package 1: `ConnectionRealizationPlan::system_linearized_diagonal` on the
//! nested_p1 heat fixture, proved against bounded exhaustive unit-vector probing of the same
//! two-operator connected reduced action (`diag(Rᵀ J R)`).
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

/// The single non-basis input of `model`'s one equation is the whole flux `q = -k * grad(T)`
/// (bound as one opaque vector constitutive, mirroring how `-sigma(T) * grad(V)` is bound in
/// the corpus 08 fixture): a constant `k` makes the interior Jacobian exact and state-
/// independent, linear in `grad(T)` alone.
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

/// Binds `Conduction` over `mesh` with a constant conductivity and no essential constraints, so
/// the only structure under test is the connection's own weighted trace.
fn conduction_operator(
    model: OperatorSystem,
    mesh: &Mesh,
    nested: &ConnectionRealizationPlan,
    conductivity: f64,
) -> SystemOperator {
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
    plan.bind_kernels(built, BTreeMap::new()).unwrap()
}

#[test]
fn nested_p1_diagonal_matches_bounded_exhaustive_probing() {
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

    let coarse_operator = conduction_operator(
        operators.model_systems[0].1.clone(),
        &coarse.mesh,
        &nested,
        2.0,
    );
    let fine_operator = conduction_operator(
        operators.model_systems[1].1.clone(),
        &fine.mesh,
        &nested,
        3.0,
    );

    let dim_a = coarse_operator.dimension();
    let dim_b = fine_operator.dimension();
    let empty_a = ConstraintSet::new(dim_a, vec![]).unwrap();
    let empty_b = ConstraintSet::new(dim_b, vec![]).unwrap();
    let reduced_a = coarse_operator.reduced(empty_a).unwrap();
    let reduced_b = fine_operator.reduced(empty_b).unwrap();

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

    // Bounded exhaustive probing: every reduced unit direction through the *combined* two-
    // operator reduced action, replicated here from public `ConstraintSet` primitives exactly as
    // `crate::constraint::apply_constrained_action` composes them internally.
    let mut unit = vec![0.0; dimension];
    let mut worst_relative = 0.0_f64;
    for probe in 0..dimension {
        unit[probe] = 1.0;
        let homogeneous = pair.expand_homogeneous(&unit).unwrap();
        let (ha, hb) = homogeneous.split_at(dim_a);
        let mut za = vec![0.0; dim_a];
        let mut zb = vec![0.0; dim_b];
        coarse_operator.apply_action(ha, &mut za).unwrap();
        fine_operator.apply_action(hb, &mut zb).unwrap();
        let mut physical = za;
        physical.extend(zb);
        let mut expected = pair.restrict_transpose(&physical).unwrap();
        for constraint in pair.constraints() {
            expected[constraint.target.0] =
                pair.direction_residual(&unit, constraint.target).unwrap();
        }
        let scale = expected[probe].abs().max(1.0);
        let relative = (diagonal[probe] - expected[probe]).abs() / scale;
        worst_relative = worst_relative.max(relative);
        assert!(
            relative <= 1e-9,
            "probe {probe}: diagonal {} != exhaustive reduced action {} (relative {relative:.3e})",
            diagonal[probe],
            expected[probe]
        );
        unit[probe] = 0.0;
    }
    eprintln!("nested_p1 diagonal: worst relative discrepancy {worst_relative:.3e}");

    // A master row touching the interface must carry a nonzero cross term, or this proof would
    // pass vacuously against a formula that silently ignores the connection.
    let masters: std::collections::BTreeSet<usize> = pair
        .constraints()
        .flat_map(|c| c.dependencies.iter().map(|d| d.dof.0))
        .collect();
    assert!(
        !masters.is_empty(),
        "nested_p1 plan produced no dependencies"
    );
    let plain_diagonal_a = coarse_operator
        .linearized_diagonal(0.0, &state_a, &rate_a, rate_shift)
        .unwrap()
        .expect("coarse operator admits its own cell-local diagonal");
    assert!(
        masters.iter().any(
            |&master| (diagonal[master] - plain_diagonal_a[master]).abs()
                > 1e-9 * plain_diagonal_a[master].abs().max(1.0)
        ),
        "connected diagonal at every master row equals the unconnected operator's own diagonal; \
         the interface cross term was never exercised"
    );

    // Every connection target (slave) row is preserved as the reduced system's own identity row.
    for constraint in pair.constraints() {
        assert_eq!(diagonal[constraint.target.0], 1.0);
    }
}
