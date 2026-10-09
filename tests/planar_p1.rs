//! W10 Finitum (W9 package 3): `ConnectionRealizationPlan::planar_p1` -- planar NON-nested
//! triangulated-interface coverage proved through exact pairwise triangle intersections, and
//! mortar-type (biorthogonal dual-basis) weighted P1 constraints from the intersection
//! integrals, emitted in the `ConstraintSet` shape `nested_p1` emits. Every proof prints its
//! measured worst discrepancy:
//!
//! 1. coverage: `Σ_{a,b} |T_a ∩ T_b| = |Γ_A| = |Γ_B|` (both the unit square, an oracle
//!    independent of the constructor) on a square split along opposite diagonals, an irregular
//!    pair whose interior target vertices lie on no master vertex or edge, and the nested pair;
//! 2. affine reproduction at every target vertex (`1e-12`) on both non-nested pairs;
//! 3. nested consistency: entrywise equality (`1e-12`) with `nested_p1`'s weights on the pair it
//!    admits (a projection reproduces functions already in the target space);
//! 4. the transpose work identity `<R x, y> = <x, Rᵀ y>` and total dual-load conservation;
//! 5. `trace_defects` continuity and dual-residual balance through the new map;
//! 6. `system_linearized_diagonal` against the bounded exhaustive reduced-JVP probe, steady and
//!    on the storage-bearing fixture at `rate_shift` 0 and 3 -- the general-weights cross term
//!    with rows of several non-trivial weights;
//! 7. the typed refusals, each by its message.
#[path = "support/connected_conduction.rs"]
mod connected_conduction;
#[path = "support/planar_meshes.rs"]
mod planar_meshes;
use connected_conduction::*;
use planar_meshes::*;

use finitum::{
    ConnectionRealizationPlan, ConstraintSet, FacetId, FacetTopology, FinitumError, Mesh,
    MeshProfile, realize,
};
use scientia::SystemCompilation;
use std::collections::{BTreeMap, BTreeSet};

const TOLERANCE: f64 = 1e-12;

fn interface(mesh: &Mesh) -> Vec<FacetId> {
    plane_facets(mesh, 0, 0.5)
}

fn planar(
    compiled: &SystemCompilation,
    coarse: &Mesh,
    fine: &Mesh,
) -> Result<ConnectionRealizationPlan, FinitumError> {
    ConnectionRealizationPlan::planar_p1(
        &compiled.system,
        &compiled.system.connections[0],
        [coarse, fine],
        [&interface(coarse), &interface(fine)],
        TOLERANCE,
    )
}

fn nested(
    compiled: &SystemCompilation,
    coarse: &Mesh,
    fine: &Mesh,
) -> Result<ConnectionRealizationPlan, FinitumError> {
    ConnectionRealizationPlan::nested_p1(
        &compiled.system,
        &compiled.system.connections[0],
        [coarse, fine],
        [&interface(coarse), &interface(fine)],
        TOLERANCE,
    )
}

/// The unit square split along opposite diagonals: the left 1x1x1 box against the right 1x1x1
/// box turned a quarter turn about the `x` axis through the face centre. The four corner
/// vertices coincide; the two triangulations cross.
fn opposite_diagonals() -> (Mesh, Mesh) {
    let [coarse, right] = meshes(1);
    let fine = quarter_turn_about_x(&right.mesh, [0.5, 0.5]);
    assert_same_orientation(&right.mesh, &fine);
    (coarse.mesh, fine)
}

/// The left 2x2x2 box (8 master triangles) against the right 3x3x3 box deformed in-plane by
/// `(0.05, -0.07) sin(pi y) sin(pi z)`: corners and boundary vertices stay on the square's
/// boundary, every interior target vertex lies on no master vertex, grid line or diagonal.
fn irregular() -> (Mesh, Mesh) {
    let [coarse, _] = meshes(2);
    let [_, right] = meshes(3);
    let fine = deformed_in_plane(&right.mesh, [0.05, -0.07]);
    assert_same_orientation(&right.mesh, &fine);
    (coarse.mesh, fine)
}

/// The pair `nested_p1` admits: left 1x1x1 against right 2x2x2.
fn nested_pair() -> (Mesh, Mesh) {
    let [coarse, _] = meshes(1);
    let [_, fine] = meshes(2);
    (coarse.mesh, fine.mesh)
}

fn interface_vertices(mesh: &Mesh) -> BTreeSet<usize> {
    let topology = FacetTopology::from_mesh(mesh).unwrap();
    interface(mesh)
        .iter()
        .flat_map(|id| topology.facets()[id.0].vertices.iter().map(|v| v.0))
        .collect()
}

fn distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]))
        .clamp(0.0, 1.0);
    ((p[0] - a[0] - t * d[0]).powi(2) + (p[1] - a[1] - t * d[1]).powi(2)).sqrt()
}

/// Every interior target vertex of the irregular pair is at least `1e-2` from every master
/// vertex and every master edge: the pair is genuinely irregular, not accidentally aligned.
fn assert_irregular(coarse: &Mesh, fine: &Mesh) {
    let yz = |mesh: &Mesh, v: usize| [mesh.vertices()[v][1], mesh.vertices()[v][2]];
    let topology = FacetTopology::from_mesh(coarse).unwrap();
    let master_edges: Vec<([f64; 2], [f64; 2])> = interface(coarse)
        .iter()
        .flat_map(|id| {
            let t = &topology.facets()[id.0].vertices;
            (0..3).map(move |i| (yz(coarse, t[i].0), yz(coarse, t[(i + 1) % 3].0)))
        })
        .collect();
    let mut interior = 0;
    for v in interface_vertices(fine) {
        let p = yz(fine, v);
        if p.iter()
            .any(|c| c.abs() <= 1e-12 || (c - 1.0).abs() <= 1e-12)
        {
            continue;
        }
        interior += 1;
        for (a, b) in &master_edges {
            let gap = distance_to_segment(p, *a, *b);
            assert!(
                gap > 1e-2,
                "target vertex {p:?} lies within {gap:e} of a master edge"
            );
        }
    }
    assert_eq!(
        interior, 4,
        "the 3x3 target face has four interior vertices"
    );
}

fn probe(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|index| ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

#[test]
fn planar_coverage_proves_exact_pairwise_intersection_totals() {
    let (compiled, _) = compile(STEADY);
    let irregular_pair = irregular();
    assert_irregular(&irregular_pair.0, &irregular_pair.1);
    for (label, (coarse, fine), expected_pairs) in [
        ("opposite diagonals", opposite_diagonals(), Some(4)),
        ("irregular", irregular_pair, None),
        ("nested", nested_pair(), Some(8)),
    ] {
        let plan = planar(&compiled, &coarse, &fine).unwrap();
        assert!(plan.is_nonmatching());
        assert_eq!(plan.trace_rows().len(), interface_vertices(&fine).len());
        let report = plan
            .planar_coverage()
            .expect("planar_p1 carries its coverage proof");
        // Independent oracle: both traces triangulate the unit square.
        for side in 0..2 {
            assert!(
                (report.area[side] - 1.0).abs() <= 1e-12,
                "{label}: side {side} area {}",
                report.area[side]
            );
        }
        assert!(
            (report.intersection_total - 1.0).abs() <= 1e-10,
            "{label}: intersections total {}",
            report.intersection_total
        );
        assert_eq!(report.relative_tolerance, 1e-10);
        assert!(report.worst_relative_defect <= 1e-12);
        assert!(report.worst_constant_defect <= 1e-12);
        assert!(report.area_floor > 0.0 && report.area_floor <= 2.0 * TOLERANCE);
        if let Some(pairs) = expected_pairs {
            assert_eq!(report.pairs, pairs, "{label}: pairs above the floor");
        } else {
            assert!(report.pairs > 18, "{label}: {} pairs", report.pairs);
        }
        eprintln!(
            "{label}: areas {:?}, Σ|T_a∩T_b| = {}, worst relative coverage defect {:.3e}, worst \
             row-sum defect {:.3e}, {} pairs, {} sub-floor slivers classified, area floor {:.3e}",
            report.area,
            report.intersection_total,
            report.worst_relative_defect,
            report.worst_constant_defect,
            report.pairs,
            report.slivers_classified,
            report.area_floor
        );
    }
    // Non-vacuity: `nested_p1` refuses both non-nested pairs `planar_p1` admits, and its own
    // admitted pair still produces no coverage report.
    for (coarse, fine) in [opposite_diagonals(), irregular()] {
        assert!(nested(&compiled, &coarse, &fine).is_err());
    }
    let (coarse, fine) = nested_pair();
    assert!(
        nested(&compiled, &coarse, &fine)
            .unwrap()
            .planar_coverage()
            .is_none()
    );
}

#[test]
fn planar_p1_reproduces_affine_traces_on_non_nested_pairs() {
    let (compiled, _) = compile(STEADY);
    for (label, (coarse, fine)) in [
        ("opposite diagonals", opposite_diagonals()),
        ("irregular", irregular()),
    ] {
        let plan = planar(&compiled, &coarse, &fine).unwrap();
        let mut worst = 0.0_f64;
        let mut worst_sum = 0.0_f64;
        let mut nontrivial_rows = 0;
        let mut negative_weights = 0;
        for (target, row) in plan.trace_rows() {
            worst_sum = worst_sum.max((row.iter().map(|(_, w)| w).sum::<f64>() - 1.0).abs());
            nontrivial_rows += usize::from(
                row.iter()
                    .filter(|(_, w)| (w - 1.0).abs() > 1e-9 && w.abs() > 1e-9)
                    .count()
                    >= 2,
            );
            negative_weights += row.iter().filter(|(_, w)| *w < -1e-9).count();
            for [base, gx, gy, gz] in [[300.0, 20.0, 2.0, -3.0], [0.01, -1.0, -0.2, 0.3]] {
                let expected = affine(&fine.vertices()[target], base, gx, gy, gz);
                let value: f64 = row
                    .iter()
                    .map(|(v, w)| w * affine(&coarse.vertices()[*v], base, gx, gy, gz))
                    .sum();
                worst = worst.max((value - expected).abs());
            }
        }
        assert!(
            worst <= 1e-12,
            "{label}: affine reproduction defect {worst:e}"
        );
        assert!(worst_sum <= 1e-12, "{label}: row-sum defect {worst_sum:e}");
        assert!(
            nontrivial_rows > 0,
            "{label}: no row with several non-trivial weights"
        );
        eprintln!(
            "{label}: worst affine reproduction defect {worst:.3e}, worst row-sum defect \
             {worst_sum:.3e}, {nontrivial_rows} rows with several non-trivial weights, \
             {negative_weights} negative weights"
        );
    }
}

#[test]
fn planar_p1_weights_coincide_with_nested_p1_on_a_nested_pair() {
    let (compiled, _) = compile(STEADY);
    let (coarse, fine) = nested_pair();
    let nested = nested(&compiled, &coarse, &fine).unwrap();
    let planar = planar(&compiled, &coarse, &fine).unwrap();
    let as_map = |plan: &ConnectionRealizationPlan| -> BTreeMap<(usize, usize), f64> {
        plan.trace_rows()
            .into_iter()
            .flat_map(|(target, row)| row.into_iter().map(move |(s, w)| ((target, s), w)))
            .collect()
    };
    let (reference, weights) = (as_map(&nested), as_map(&planar));
    assert_eq!(nested.trace_rows().len(), 9);
    assert_eq!(planar.trace_rows().len(), 9);
    let keys: BTreeSet<(usize, usize)> = reference.keys().chain(weights.keys()).copied().collect();
    let mut worst = 0.0_f64;
    let mut one_sided = 0;
    for key in keys {
        let a = reference.get(&key).copied().unwrap_or(0.0);
        let b = weights.get(&key).copied().unwrap_or(0.0);
        worst = worst.max((a - b).abs());
        one_sided += usize::from(reference.contains_key(&key) != weights.contains_key(&key));
    }
    assert!(worst <= 1e-12, "entrywise weight discrepancy {worst:e}");
    assert_eq!(one_sided, 0, "weights present on one side only");
    eprintln!(
        "nested consistency: worst entrywise weight discrepancy {worst:.3e} over {} entries, \
         {one_sided} present on one side only (absent = 0)",
        reference.len().max(weights.len())
    );
    assert_ne!(nested.identity(), planar.identity());
}

#[test]
fn planar_constraints_satisfy_the_transpose_work_identity_and_conserve_the_total_dual_load() {
    let (compiled, operators) = compile(STEADY);
    let (coarse, fine) = irregular();
    let plan = planar(&compiled, &coarse, &fine).unwrap();
    let models = [
        operators.model_systems[0].1.clone(),
        operators.model_systems[1].1.clone(),
    ];
    let a = bind(
        models[0].clone(),
        &coarse,
        &plan,
        constant_flux(&models[0], 2.0),
    );
    let b = bind(
        models[1].clone(),
        &fine,
        &plan,
        constant_flux(&models[1], 3.0),
    );
    let reduced_a = a
        .reduced(ConstraintSet::new(a.dimension(), vec![]).unwrap())
        .unwrap();
    let reduced_b = b
        .reduced(ConstraintSet::new(b.dimension(), vec![]).unwrap())
        .unwrap();
    let pair = plan.constraints([&reduced_a, &reduced_b]).unwrap();
    let dimension = pair.dof_count();
    assert_eq!(dimension, a.dimension() + b.dimension());
    assert!(pair.constraints().all(|c| c.offset == 0.0));
    assert!(pair.constraints().any(|c| c.dependencies.len() >= 3
        && c.dependencies.iter().all(|d| (d.weight - 1.0).abs() > 1e-9)));

    let x = probe(dimension, 0.0);
    let y = probe(dimension, 7.0);
    let rx = pair.expand_homogeneous(&x).unwrap();
    let rty = pair.restrict_transpose(&y).unwrap();
    let (left, right) = (dot(&rx, &y), dot(&x, &rty));
    let relative = (left - right).abs() / left.abs().max(right.abs()).max(1.0);
    assert!(
        relative <= 1e-12,
        "<Rx,y> = {left} != <x,Rᵀy> = {right} (relative {relative:e})"
    );

    // Total dual load: fine (target) loads transposed to the masters sum to the same total,
    // because every mortar row reproduces constants (`<R 1, f> = <1, Rᵀ f>`).
    let mut loads = vec![0.0; dimension];
    for (index, constraint) in pair.constraints().enumerate() {
        loads[constraint.target.0] = 1.0 + 0.1 * index as f64;
    }
    let transposed = pair.restrict_transpose(&loads).unwrap();
    let (before, after) = (loads.iter().sum::<f64>(), transposed.iter().sum::<f64>());
    let load_defect = (before - after).abs() / before.abs();
    assert!(load_defect <= 1e-12, "total dual load {before} -> {after}");
    for constraint in pair.constraints() {
        assert_eq!(transposed[constraint.target.0], 0.0);
    }
    eprintln!(
        "transpose identity: relative defect {relative:.3e}; total dual load: relative defect \
         {load_defect:.3e}"
    );
}

#[test]
fn trace_defects_hold_for_constant_and_linear_traces_through_the_planar_map() {
    let (compiled, operators) = compile(STEADY);
    let (coarse, fine) = irregular();
    let plan = planar(&compiled, &coarse, &fine).unwrap();
    let models = [
        operators.model_systems[0].1.clone(),
        operators.model_systems[1].1.clone(),
    ];
    let a = bind(
        models[0].clone(),
        &coarse,
        &plan,
        constant_flux(&models[0], 2.0),
    );
    let b = bind(
        models[1].clone(),
        &fine,
        &plan,
        constant_flux(&models[1], 3.0),
    );
    let reduced_a = a
        .reduced(ConstraintSet::new(a.dimension(), vec![]).unwrap())
        .unwrap();
    let reduced_b = b
        .reduced(ConstraintSet::new(b.dimension(), vec![]).unwrap())
        .unwrap();
    let pair = plan.constraints([&reduced_a, &reduced_b]).unwrap();
    let (dim_a, dim_b) = (a.dimension(), b.dimension());
    let zero_a = vec![0.0; dim_a];
    let zero_b = vec![0.0; dim_b];

    let mut worst_continuity = 0.0_f64;
    for (label, trace) in [
        ("constant", [300.0, 0.0, 0.0, 0.0]),
        ("linear", [300.0, 20.0, 2.0, -3.0]),
    ] {
        let [base, gx, gy, gz] = trace;
        let values_a: Vec<f64> = coarse
            .vertices()
            .iter()
            .map(|p| affine(p, base, gx, gy, gz))
            .collect();
        let values_b: Vec<f64> = fine
            .vertices()
            .iter()
            .map(|p| affine(p, base, gx, gy, gz))
            .collect();
        let (continuity, _) = plan
            .trace_defects(
                [&reduced_a, &reduced_b],
                [&values_a, &values_b],
                [&zero_a, &zero_b],
            )
            .unwrap();
        assert!(
            continuity <= 1e-12,
            "{label} trace continuity defect {continuity:e}"
        );
        worst_continuity = worst_continuity.max(continuity);
    }

    let values_a: Vec<f64> = coarse
        .vertices()
        .iter()
        .map(|p| affine(p, 300.0, 20.0, 2.0, -3.0))
        .collect();
    let values_b: Vec<f64> = fine
        .vertices()
        .iter()
        .map(|p| affine(p, 300.0, 20.0, 2.0, -3.0))
        .collect();
    let masters: BTreeSet<usize> = pair
        .constraints()
        .flat_map(|c| c.dependencies.iter().map(|d| d.dof.0))
        .collect();
    let mut worst_balance = 0.0_f64;
    let mut check = |label: &str, residual_a: &[f64], residual_b: &[f64]| {
        let (_, balance) = plan
            .trace_defects(
                [&reduced_a, &reduced_b],
                [&values_a, &values_b],
                [residual_a, residual_b],
            )
            .unwrap();
        let mut full = residual_a.to_vec();
        full.extend_from_slice(residual_b);
        let restricted = pair.restrict_transpose(&full).unwrap();
        let expected = masters
            .iter()
            .map(|&m| restricted[m].abs())
            .fold(0.0_f64, f64::max);
        let relative = (balance - expected).abs() / expected.max(1.0);
        assert!(
            relative <= 1e-12,
            "{label}: balance {balance} != restrict_transpose reference {expected}"
        );
        worst_balance = worst_balance.max(relative);
    };
    check("state residual", &probe(dim_a, 0.0), &probe(dim_b, 7.0));
    let rate_a: Vec<f64> = coarse
        .vertices()
        .iter()
        .map(|p| affine(p, 5.0, 1.0, 0.25, -0.75))
        .collect();
    let rate_b: Vec<f64> = fine
        .vertices()
        .iter()
        .map(|p| affine(p, 5.0, 1.0, 0.25, -0.75))
        .collect();
    let mass_a = a.mass_matrix(models[0].field_order[0]).unwrap();
    let mass_b = b.mass_matrix(models[1].field_order[0]).unwrap();
    check(
        "storage (mass) residual",
        &dense_matvec(&mass_a, dim_a, &rate_a),
        &dense_matvec(&mass_b, dim_b, &rate_b),
    );
    eprintln!(
        "trace_defects through planar_p1: worst continuity {worst_continuity:.3e}, worst balance \
         relative defect {worst_balance:.3e}"
    );
}

/// The general-weights cross term `Σ w_β J_B[β,γ] w_γ` is exercised here for the first time
/// with masters that prolongate into several targets with non-trivial weights.
fn assert_general_cross_term(pair: &ConstraintSet) {
    let mut incidence: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for constraint in pair.constraints() {
        for dependency in &constraint.dependencies {
            incidence
                .entry(dependency.dof.0)
                .or_default()
                .push(dependency.weight);
        }
    }
    let general = incidence
        .values()
        .filter(|weights| weights.len() >= 2 && weights.iter().all(|w| (w - 1.0).abs() > 1e-9))
        .count();
    assert!(general > 0, "no master with several non-trivial weights");
    eprintln!("{general} masters prolongate into several targets with non-trivial weights");
}

#[test]
fn planar_p1_diagonal_matches_bounded_exhaustive_probing_on_a_non_nested_pair() {
    let (compiled, operators) = compile(STEADY);
    let (coarse, fine) = irregular();
    let plan = planar(&compiled, &coarse, &fine).unwrap();
    let coarse_model = operators.model_systems[0].1.clone();
    let fine_model = operators.model_systems[1].1.clone();
    // 2.5, not 2.0: with 2.0 the coarse 2x2x2 mesh's physical diagonal at its essential corner
    // is exactly 1.0, which would make the identity-row assertion vacuous.
    let coarse_operator = bind(
        coarse_model.clone(),
        &coarse,
        &plan,
        constant_flux(&coarse_model, 2.5),
    );
    let fine_operator = bind(
        fine_model.clone(),
        &fine,
        &plan,
        constant_flux(&fine_model, 3.0),
    );
    let dim_a = coarse_operator.dimension();
    let dim_b = fine_operator.dimension();
    let essential_a = vertex_on(&coarse, 0.0);
    let essential_b = vertex_on(&fine, 1.0);
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
    let diagonal = plan
        .system_linearized_diagonal(
            [&reduced_a, &reduced_b],
            0.0,
            [&state_a, &state_b],
            [&rate_a, &rate_b],
            rate_shift,
        )
        .unwrap()
        .expect("linear conduction over cell-only kernels admits a cell-local diagonal");
    let pair = plan.constraints([&reduced_a, &reduced_b]).unwrap();
    assert_general_cross_term(&pair);
    let expected = exhaustive_diagonal(
        &pair,
        [&reduced_a, &reduced_b],
        0.0,
        [&state_a, &state_b],
        [&rate_a, &rate_b],
        rate_shift,
    );
    let worst = assert_diagonal_matches("steady planar_p1", &diagonal, &expected);
    eprintln!("steady planar_p1 diagonal: worst relative discrepancy {worst:.3e}");
    let plain_a = coarse_operator
        .linearized_diagonal(0.0, &state_a, &rate_a, rate_shift)
        .unwrap()
        .unwrap();
    let plain_b = fine_operator
        .linearized_diagonal(0.0, &state_b, &rate_b, rate_shift)
        .unwrap()
        .unwrap();
    assert_row_structure(
        "steady planar_p1",
        &pair,
        &diagonal,
        [&plain_a, &plain_b],
        [essential_a, essential_b],
    );
}

#[test]
fn planar_p1_diagonal_sees_rate_shift_and_expands_the_point_through_the_connection() {
    let (compiled, operators) = compile(TRANSIENT);
    let (coarse, fine) = irregular();
    let plan = planar(&compiled, &coarse, &fine).unwrap();
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
    let coarse_model = operators.model_systems[0].1.clone();
    let fine_model = operators.model_systems[1].1.clone();
    let coarse_operator = bind(
        coarse_model.clone(),
        &coarse,
        &plan,
        storage_closures(
            &coarse_model,
            &symbol_names(&compiled, &operators, 0),
            storage[0],
        ),
    );
    let fine_operator = bind(
        fine_model.clone(),
        &fine,
        &plan,
        storage_closures(
            &fine_model,
            &symbol_names(&compiled, &operators, 1),
            storage[1],
        ),
    );
    let dim_a = coarse_operator.dimension();
    let dim_b = fine_operator.dimension();
    let essential_a = vertex_on(&coarse, 0.0);
    let essential_b = vertex_on(&fine, 1.0);
    let reduced_a = coarse_operator
        .reduced(essential(dim_a, essential_a, 300.0))
        .unwrap();
    let reduced_b = fine_operator
        .reduced(essential(dim_b, essential_b, 350.0))
        .unwrap();
    let pair = plan.constraints([&reduced_a, &reduced_b]).unwrap();
    assert_general_cross_term(&pair);
    let targets: BTreeSet<usize> = pair.constraints().map(|c| c.target.0 - dim_a).collect();
    let masters: BTreeSet<usize> = pair
        .constraints()
        .flat_map(|c| c.dependencies.iter().map(|d| d.dof.0))
        .collect();

    // The connection-reduced point: an affine temperature and rate on both meshes, the
    // essential rows at their prescribed values, the target rows deliberately WRONG (0 K).
    let value0 = |p: &[f64]| affine(p, 300.0, 20.0, 10.0, -5.0);
    let rate0 = |p: &[f64]| affine(p, 5.0, 1.0, 0.25, -0.75);
    let mut state_a: Vec<f64> = coarse.vertices().iter().map(|p| value0(p)).collect();
    let rate_a: Vec<f64> = coarse.vertices().iter().map(|p| rate0(p)).collect();
    state_a[essential_a] = 300.0;
    let mut state_b: Vec<f64> = fine.vertices().iter().map(|p| value0(p)).collect();
    let mut rate_b: Vec<f64> = fine.vertices().iter().map(|p| rate0(p)).collect();
    state_b[essential_b] = 350.0;
    for &target in &targets {
        state_b[target] = 0.0;
        rate_b[target] = 0.0;
    }
    let time = 0.3;

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
        let diagonal = plan
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
        let label = format!("storage planar_p1 at rate_shift {rate_shift}");
        let worst = assert_diagonal_matches(&label, &diagonal, &expected);
        eprintln!("{label}: worst relative discrepancy {worst:.3e}");
        diagonals.push(diagonal);
    }

    let mass_a = coarse_operator
        .mass_matrix(coarse_model.field_order[0])
        .unwrap();
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
            "row {row}: diagonal did not grow with rate_shift"
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
    assert!(interior_rows > 0);

    let plain_a = coarse_operator
        .linearized_diagonal(time, &expanded[..dim_a], &rate_a, shifts[0])
        .unwrap()
        .unwrap();
    assert_row_structure(
        "storage planar_p1",
        &pair,
        &diagonals[0],
        [&plain_a, &plain_expanded],
        [essential_a, essential_b],
    );
}

#[test]
fn planar_p1_refuses_non_coplanar_same_facing_overlapping_partial_degenerate_and_interior_traces() {
    let (compiled, _) = compile(STEADY);
    let system = &compiled.system;
    let relation = &system.connections[0];
    let (coarse, fine) = opposite_diagonals();
    let a = interface(&coarse);
    let b = interface(&fine);
    let attempt = |meshes: [&Mesh; 2], facets: [&[FacetId]; 2], tolerance: f64| -> String {
        ConnectionRealizationPlan::planar_p1(system, relation, meshes, facets, tolerance)
            .unwrap_err()
            .to_string()
    };
    let admitted = "CONNECTION_TRACE: planar P1 needs an admitted relation and nonempty 3-D traces";
    assert!(attempt([&coarse, &fine], [&a, &b], 0.0).contains(admitted));
    assert!(attempt([&coarse, &fine], [&a, &[]], TOLERANCE).contains(admitted));
    assert!(
        attempt([&coarse, &fine], [&a, &[b[0], b[0]]], TOLERANCE)
            .contains("CONNECTION_TRACE: duplicate interface facet")
    );
    assert!(
        attempt([&coarse, &fine], [&a, &[FacetId(usize::MAX)]], TOLERANCE)
            .contains("CONNECTION_TRACE: facet out of range")
    );

    // Non-coplanar traces: the target box shifted off the plane by more than the tolerance.
    let shifted = translated(&fine, [1e-6, 0.0, 0.0]);
    assert!(
        attempt([&coarse, &shifted], [&a, &b], TOLERANCE)
            .contains("CONNECTION_TRACE: interface planes do not coincide")
    );
    // Same-facing normals: the master trace against itself.
    assert!(
        attempt([&coarse, &coarse], [&a, &a], TOLERANCE)
            .contains("CONNECTION_TRACE: planar opposite normals required")
    );
    // Overlap within a side: two coplanar exterior triangles of one mesh that overlap.
    let overlapping = fan_mesh_on_plane(
        &[
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.2, 0.2], [0.9, 0.2], [0.2, 0.9]],
        ],
        [1.0, 0.5, 0.5],
    );
    assert!(
        attempt(
            [&coarse, &overlapping],
            [&a, &interface(&overlapping)],
            TOLERANCE
        )
        .contains("CONNECTION_TRACE: overlapping interface triangles")
    );
    // Partial coverage, one polygon not the other, in both directions, and a dropped facet.
    let polygon = "CONNECTION_TRACE: interface triangulations do not cover the same polygon";
    let half = |x: [f64; 2]| {
        realize(&MeshProfile::SimplexBox {
            dimension: 3,
            extent: vec![x, [0.0, 0.5], [0.0, 1.0]],
            subdivisions: vec![1; 3],
        })
        .unwrap()
        .mesh
    };
    let half_fine = half([0.5, 1.0]);
    assert!(
        attempt(
            [&coarse, &half_fine],
            [&a, &interface(&half_fine)],
            TOLERANCE
        )
        .contains(polygon)
    );
    let half_coarse = half([0.0, 0.5]);
    assert!(
        attempt(
            [&half_coarse, &fine],
            [&interface(&half_coarse), &b],
            TOLERANCE
        )
        .contains(polygon)
    );
    let (nested_coarse, nested_fine) = nested_pair();
    let nested_facets = interface(&nested_fine);
    assert!(
        attempt(
            [&nested_coarse, &nested_fine],
            [
                &interface(&nested_coarse),
                &nested_facets[..nested_facets.len() - 1]
            ],
            TOLERANCE
        )
        .contains(polygon)
    );
    // A degenerate triangle below the area floor (a fan vertex 1e-13 off an edge) refuses; the
    // same fan with the vertex 1e-3 off the edge is thin but admitted and covers the square.
    let fan = |lift: f64| {
        fan_mesh_on_plane(
            &[
                [[0.0, 0.0], [1.0, 0.0], [0.5, lift]],
                [[0.5, lift], [1.0, 0.0], [1.0, 1.0]],
                [[0.5, lift], [1.0, 1.0], [0.0, 1.0]],
                [[0.5, lift], [0.0, 1.0], [0.0, 0.0]],
            ],
            [1.0, 0.5, 0.5],
        )
    };
    let sliver = fan(1e-13);
    assert!(
        attempt([&coarse, &sliver], [&a, &interface(&sliver)], TOLERANCE)
            .contains("CONNECTION_TRACE: degenerate interface triangle below the area floor")
    );
    let thin = fan(1e-3);
    let plan = ConnectionRealizationPlan::planar_p1(
        system,
        relation,
        [&coarse, &thin],
        [&a, &interface(&thin)],
        TOLERANCE,
    )
    .unwrap();
    assert_eq!(plan.planar_coverage().unwrap().pairs, 6);
    // A positive opposite-side intersection at or below the floor is classified as empty, not
    // refused: a fan vertex 1e-12 off the master diagonal makes a sliver of area ~2.5e-13
    // against the master triangle across it, below the 1.4e-12 floor; coverage still holds.
    let grazing = fan_mesh_on_plane(
        &[
            [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5 + 1e-12]],
            [[0.5, 0.5 + 1e-12], [1.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.5 + 1e-12], [1.0, 1.0], [0.0, 1.0]],
            [[0.5, 0.5 + 1e-12], [0.0, 1.0], [0.0, 0.0]],
        ],
        [1.0, 0.5, 0.5],
    );
    let report = ConnectionRealizationPlan::planar_p1(
        system,
        relation,
        [&coarse, &grazing],
        [&a, &interface(&grazing)],
        TOLERANCE,
    )
    .unwrap()
    .planar_coverage()
    .cloned()
    .unwrap();
    assert!(report.slivers_classified >= 1, "{report:?}");
    assert!(report.worst_relative_defect <= 1e-10, "{report:?}");
    eprintln!(
        "grazing fan: {} sub-floor slivers classified as empty, worst relative coverage defect \
         {:.3e}",
        report.slivers_classified, report.worst_relative_defect
    );
    // Non-exterior traces: an interior facet carries no outward normal.
    let interior = FacetTopology::from_mesh(&fine)
        .unwrap()
        .interior()
        .next()
        .unwrap()
        .id;
    assert!(attempt([&coarse, &fine], [&a, &[interior]], TOLERANCE).contains("interior"));
}
