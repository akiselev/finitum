//! W9 Finitum SC-W3 package 1: `ConnectionRealizationPlan::system_linearized_diagonal` proved
//! against bounded exhaustive unit-vector probing of the same two-operator connected reduced
//! Jacobian (`diag(Rᵀ J R)`). The oracle is the two REDUCED operators' own
//! `jacobian_vector_product` (so each operator's essential rows are seen as the identity rows
//! they are), driven with the rate direction `rate_shift · d` `linearized_diagonal` probes, and
//! composed through the connection exactly as Krasis's
//! `ConnectedSystemOperator::jacobian_vector_product` composes its leaves (the linearization
//! point expanded through the pair first). Three shapes:
//!
//! - the steady `nested_p1` heat fixture, with one essential constraint per side;
//! - a storage-bearing (`rho * cp * dt(T)`), state-dependent-conductivity `nested_p1` fixture,
//!   where the diagonal must change with `rate_shift` and must be linearized at the
//!   connection-expanded point, not at the caller's raw fine-trace entries;
//! - the `matching` plan, where the diagonal must also equal the sum-of-leaf-diagonals
//!   convention Krasis's `ConnectedSystemOperator::jacobian_diagonal` uses for weight-1 pairs.
//!
//! The fixtures, closures and the probe live in `support/connected_conduction.rs`, shared with
//! the `planar_p1` proofs.
#[path = "support/connected_conduction.rs"]
mod connected_conduction;
use connected_conduction::*;
use finitum::{ConnectionRealizationPlan, RegionTagId};
use methodus::{DaeOperator, EvaluationContext};
use std::collections::BTreeSet;

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
    let names = |side: usize| symbol_names(&compiled, &operators, side);
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

/// The `matching` plan path: the same exhaustive probe, plus equality with the
/// sum-of-leaf-diagonals convention Krasis's `ConnectedSystemOperator::jacobian_diagonal` uses
/// for weight-1 bijective pairs (`Rᵀ` of the concatenated leaf `DaeOperator::jacobian_diagonal`s,
/// targets set to the identity row).
#[test]
fn matching_diagonal_matches_probing_and_the_sum_of_leaf_diagonals_convention() {
    let (compiled, operators) = compile(STEADY);
    let system = &compiled.system;
    let relation = &system.connections[0];
    let [left, right] = meshes(2);
    let a = &left.tags.facet_regions[&RegionTagId::new("x_max")];
    let b = &right.tags.facet_regions[&RegionTagId::new("x_min")];
    let matching = ConnectionRealizationPlan::matching(
        system,
        relation,
        [&left.mesh, &right.mesh],
        [a, b],
        1e-12,
    )
    .unwrap();
    assert!(!matching.is_nonmatching());

    let left_model = operators.model_systems[0].1.clone();
    let right_model = operators.model_systems[1].1.clone();
    let left_operator = bind(
        left_model.clone(),
        &left.mesh,
        &matching,
        constant_flux(&left_model, 2.5),
    );
    let right_operator = bind(
        right_model.clone(),
        &right.mesh,
        &matching,
        constant_flux(&right_model, 3.0),
    );
    let dim_a = left_operator.dimension();
    let dim_b = right_operator.dimension();
    let essential_a = vertex_on(&left.mesh, 0.0);
    let essential_b = vertex_on(&right.mesh, 1.0);
    let reduced_a = left_operator
        .reduced(essential(dim_a, essential_a, 300.0))
        .unwrap();
    let reduced_b = right_operator
        .reduced(essential(dim_b, essential_b, 350.0))
        .unwrap();
    let pair = matching.constraints([&reduced_a, &reduced_b]).unwrap();
    assert!(
        pair.constraints()
            .all(|c| c.dependencies.len() == 1 && c.dependencies[0].weight == 1.0)
    );

    let state_a = vec![0.0; dim_a];
    let state_b = vec![0.0; dim_b];
    let rate_a = vec![0.0; dim_a];
    let rate_b = vec![0.0; dim_b];
    let rate_shift = 1.0;
    let diagonal = matching
        .system_linearized_diagonal(
            [&reduced_a, &reduced_b],
            0.0,
            [&state_a, &state_b],
            [&rate_a, &rate_b],
            rate_shift,
        )
        .unwrap()
        .expect("linear conduction over cell-only kernels admits a cell-local diagonal");
    let expected = exhaustive_diagonal(
        &pair,
        [&reduced_a, &reduced_b],
        0.0,
        [&state_a, &state_b],
        [&rate_a, &rate_b],
        rate_shift,
    );
    let worst = assert_diagonal_matches("matching", &diagonal, &expected);
    eprintln!("matching diagonal: worst relative discrepancy {worst:.3e}");

    // Krasis's convention for weight-1 bijective pairs: concatenate the leaves' own reduced
    // `DaeOperator::jacobian_diagonal`s, restrict by the pair's transpose, identity targets.
    let context = EvaluationContext::reproducible();
    let mut leaves =
        DaeOperator::jacobian_diagonal(&reduced_a, &context, 0.0, &state_a, &rate_a, rate_shift)
            .unwrap()
            .expect("left leaf diagonal");
    leaves.extend(
        DaeOperator::jacobian_diagonal(&reduced_b, &context, 0.0, &state_b, &rate_b, rate_shift)
            .unwrap()
            .expect("right leaf diagonal"),
    );
    let mut krasis = pair.restrict_transpose(&leaves).unwrap();
    for constraint in pair.constraints() {
        krasis[constraint.target.0] = 1.0;
    }
    for (row, (&got, &want)) in diagonal.iter().zip(&krasis).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "row {row}: connected diagonal {got} != sum-of-leaf-diagonals convention {want}"
        );
    }

    let plain_a = left_operator
        .linearized_diagonal(0.0, &state_a, &rate_a, rate_shift)
        .unwrap()
        .unwrap();
    let plain_b = right_operator
        .linearized_diagonal(0.0, &state_b, &rate_b, rate_shift)
        .unwrap()
        .unwrap();
    assert_row_structure(
        "matching",
        &pair,
        &diagonal,
        [&plain_a, &plain_b],
        [essential_a, essential_b],
    );
}
