//! GX-C4: exterior facet integrals (SV2-B2 pulled forward, bounded to scalar/vector H1(order=1)
//! on simplex meshes, `Value`-only trace evaluation).
//!
//! Manufactured problem: `u(x, y) = sin(pi*x) * cos(pi*y)` on the unit square, `k = 1`. Since
//! `-Δu = 2*pi^2*u`, `f = 2*pi^2*sin(pi*x)*cos(pi*y)` (sampled per point, not a compile-time
//! constant, exercising the same cell `ExternalInput::sampled` path as every other cell fixture).
//! A transcendental exact solution is deliberately chosen over a low-degree polynomial: this
//! structured "criss-cross" `SimplexBox` triangulation is P1-superconvergent (exactly, to machine
//! precision) for polynomial data up to at least degree two, which would make every mesh
//! resolution's nodal error already at floating-point noise and the convergence order
//! unobservable. Dirichlet data `exact_u` is imposed on three sides (`x_min`, `y_min`,
//! `y_max`); a genuinely non-constant Neumann flux `g = du/dx = pi*cos(pi*x)*cos(pi*y)` (a
//! `FieldSource`-backed stored external input sampled at facet centroids) is imposed on the
//! fourth (`x_max`).

use finitum::{
    DofMap, ExternalInput, FacetTopology, FieldSource, MeshProfile, PreparedElement,
    RealizationPlan, RegionMap, RegionTagId, TaggedMesh, essential_constraints_from,
    facet_membership_from, realize, vector_nodal_dof_map,
};
use methodus::{
    ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, EvaluationContext,
    solve_conjugate_gradient,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, RegionId, SemanticMeasure, SemanticModel, compile_semantics,
    derive_variational_form, factor_operator, infer_form_requirements, lower_operator_kernels,
};

const NEUMANN_POISSON: &str = r#"
module gx_c4.neumann_poisson;
model NeumannPoisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  source g: SurfaceFlux;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
  boundary load on boundary("load") { neumann u = g; }
}
"#;

fn exact_u(coordinates: &[f64]) -> f64 {
    (std::f64::consts::PI * coordinates[0]).sin() * (std::f64::consts::PI * coordinates[1]).cos()
}

/// `f = -div(grad(exact_u)) = 2 * pi^2 * exact_u`, since `exact_u` is a `-Δ` eigenfunction with
/// eigenvalue `2 * pi^2`.
fn source_f(coordinates: &[f64]) -> f64 {
    2.0 * std::f64::consts::PI * std::f64::consts::PI * exact_u(coordinates)
}

/// `g = du/dx = pi * cos(pi * x) * cos(pi * y)`, evaluated at the facet centroid (`x = 1` on the
/// `x_max` boundary, but the closed form is general).
fn flux_g(coordinates: &[f64]) -> f64 {
    std::f64::consts::PI
        * (std::f64::consts::PI * coordinates[0]).cos()
        * (std::f64::consts::PI * coordinates[1]).cos()
}

fn region_id(model: &SemanticModel, name: &str) -> RegionId {
    model
        .regions
        .iter()
        .find(|region| region.name == name)
        .unwrap_or_else(|| panic!("model has no region named {name:?}"))
        .id
}

/// Builds the `NeumannPoisson` realization on an `n` by `n` unit-square `SimplexBox`, with `k =
/// 1`, a stored per-point `f` (`source_f`), Dirichlet data `exact_u` on `walls`, and a stored
/// per-facet-point Neumann flux `g` (`flux_g`) on `load`.
fn neumann_plan(n: usize) -> (RealizationPlan, TaggedMesh, DofMap) {
    let compilation = compile_semantics(NEUMANN_POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "NeumannPoisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let model = &compilation.semantic.models[0];

    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![n, n],
    };
    let tagged = realize(&profile).unwrap();
    let dof_map = vector_nodal_dof_map(&tagged.mesh, 1).unwrap();
    let element = PreparedElement::linear_simplex(2).unwrap();

    let walls_region = region_id(model, "walls");
    let load_region = region_id(model, "load");
    let mut region_map = RegionMap::new();
    region_map.insert(
        walls_region,
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
        ],
    );
    region_map.insert(load_region, [RegionTagId::new("x_max")]);

    // Exactly one Dirichlet declaration ("walls") exists; `requirement.argument` is a
    // compiler-synthesized test-function symbol id (`SymbolId::generated_for`), not an index
    // into `model.symbols`, so it is not filtered by field name here.
    let essential_requirement = requirements
        .essential_constraints
        .first()
        .expect("NeumannPoisson declares one Dirichlet condition on u")
        .clone();
    let constraints = essential_constraints_from(
        &tagged,
        &dof_map,
        &[essential_requirement],
        &region_map,
        &[FieldSource::sampled(|coordinates| {
            vec![exact_u(coordinates)]
        })],
    )
    .unwrap();

    let facet_topology = FacetTopology::from_mesh(&tagged.mesh).unwrap();
    let facet_regions = facet_membership_from(&tagged, &region_map, [load_region]).unwrap();

    let mut stored = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.clone();
            match (&integral.measure, name.as_str()) {
                (SemanticMeasure::Cell { .. }, "k") => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &tagged.mesh,
                        &element,
                        |_, _| vec![1.0],
                    )
                    .unwrap(),
                ),
                (SemanticMeasure::Cell { .. }, "f") => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &tagged.mesh,
                        &element,
                        |_, point| vec![source_f(point)],
                    )
                    .unwrap(),
                ),
                (SemanticMeasure::ExteriorFacet { region }, "g") => {
                    let facet_ids = facet_regions
                        .get(region)
                        .expect("load region was resolved above");
                    stored.push(
                        ExternalInput::sampled_on_facets(
                            integral.integral_index,
                            input.id,
                            1,
                            &tagged.mesh,
                            &facet_topology,
                            facet_ids,
                            |_, point| vec![flux_g(point)],
                        )
                        .unwrap(),
                    );
                }
                (measure, other) => {
                    panic!("unexpected external input {other:?} on measure {measure:?}")
                }
            }
        }
    }

    let plan = RealizationPlan::new_with_facets(
        requirements,
        factorization,
        kernels,
        tagged.mesh.clone(),
        element,
        dof_map.clone(),
        constraints,
        stored,
        Vec::new(),
        facet_regions,
    )
    .unwrap();
    (plan, tagged, dof_map)
}

fn cg_config() -> ConjugateGradientConfig {
    ConjugateGradientConfig {
        symmetry_policy: ConjugateGradientSymmetryPolicy::AssumeSymmetric,
        ..ConjugateGradientConfig::default()
    }
}

fn solve(plan: &RealizationPlan) -> Vec<f64> {
    let matrix_free = plan.matrix_free();
    let context = EvaluationContext::reproducible();
    let right_hand_side = plan.load_vector().unwrap();
    let report = solve_conjugate_gradient(
        &matrix_free,
        None,
        &context,
        &right_hand_side,
        &vec![0.0; plan.dimension()],
        &cg_config(),
    )
    .unwrap();
    assert!(
        report.converged,
        "conjugate gradient solve did not converge"
    );
    report.solution
}

fn max_nodal_error(mesh: &TaggedMesh, solution: &[f64]) -> f64 {
    mesh.mesh
        .vertices()
        .iter()
        .zip(solution)
        .map(|(vertex, value)| (value - exact_u(vertex)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn manufactured_neumann_problem_converges_at_the_p1_rate() {
    let mesh_sizes = [2usize, 4, 8, 16, 32];
    let mut errors = Vec::new();
    for n in mesh_sizes {
        let (plan, tagged, _) = neumann_plan(n);
        let solution = solve(&plan);
        errors.push(max_nodal_error(&tagged, &solution));
    }
    for (n, error) in mesh_sizes.iter().zip(&errors) {
        assert!(
            error.is_finite() && *error < 0.5,
            "n={n} error too large: {error}"
        );
    }
    // Every refinement must strictly shrink the max-nodal error.
    for window in errors.windows(2) {
        assert!(
            window[1] < window[0],
            "refinement must shrink the max-nodal error: {window:?}"
        );
    }
    // The coarsest step (n=2 -> n=4) is pre-asymptotic on such a coarse mesh (observed order
    // ~1.34 here); from n=4 onward the observed order climbs cleanly toward the P1 nodal
    // asymptotic rate of 2 (~1.81, ~1.95, ~1.99 observed), so the strict order bound applies only
    // from the second window onward.
    for window in errors[1..].windows(2) {
        let (coarse, fine) = (window[0], window[1]);
        let observed_order = (coarse / fine).log2();
        assert!(
            observed_order >= 1.7,
            "observed convergence order {observed_order} too low: coarse={coarse}, fine={fine}"
        );
    }
}

#[test]
fn facet_load_vector_contribution_matches_the_hand_computed_single_facet_value() {
    // Zero source and zero Dirichlet data isolate the Neumann facet term exactly: with `f = 0`
    // and homogeneous boundary data, `load_vector()` at a free (non-Dirichlet) degree of freedom
    // equals the boundary integral `integral g * v_i ds`, which the bounded single-point facet
    // rule evaluates exactly as `g * v_i(centroid) * |facet|` for the affine P1 trace `v_i`.
    let compilation = compile_semantics(NEUMANN_POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "NeumannPoisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let model = &compilation.semantic.models[0];

    let n = 2usize;
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![n, n],
    };
    let tagged = realize(&profile).unwrap();
    let dof_map = vector_nodal_dof_map(&tagged.mesh, 1).unwrap();
    let element = PreparedElement::linear_simplex(2).unwrap();

    let walls_region = region_id(model, "walls");
    let load_region = region_id(model, "load");
    let mut region_map = RegionMap::new();
    region_map.insert(
        walls_region,
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
        ],
    );
    region_map.insert(load_region, [RegionTagId::new("x_max")]);

    let essential_requirement = requirements.essential_constraints.first().unwrap().clone();
    let constraints = essential_constraints_from(
        &tagged,
        &dof_map,
        &[essential_requirement],
        &region_map,
        &[FieldSource::constant(vec![0.0])],
    )
    .unwrap();

    let facet_topology = FacetTopology::from_mesh(&tagged.mesh).unwrap();
    let facet_regions = facet_membership_from(&tagged, &region_map, [load_region]).unwrap();
    let flux = 3.5_f64;

    let mut stored = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.clone();
            match (&integral.measure, name.as_str()) {
                (SemanticMeasure::Cell { .. }, "k") => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &tagged.mesh,
                        &element,
                        |_, _| vec![1.0],
                    )
                    .unwrap(),
                ),
                (SemanticMeasure::Cell { .. }, "f") => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &tagged.mesh,
                        &element,
                        |_, _| vec![0.0],
                    )
                    .unwrap(),
                ),
                (SemanticMeasure::ExteriorFacet { region }, "g") => {
                    let facet_ids = facet_regions.get(region).unwrap();
                    stored.push(
                        ExternalInput::sampled_on_facets(
                            integral.integral_index,
                            input.id,
                            1,
                            &tagged.mesh,
                            &facet_topology,
                            facet_ids,
                            |_, _| vec![flux],
                        )
                        .unwrap(),
                    );
                }
                (measure, other) => {
                    panic!("unexpected external input {other:?} on measure {measure:?}")
                }
            }
        }
    }

    let load_facets = facet_regions.get(&load_region).unwrap().clone();
    let plan = RealizationPlan::new_with_facets(
        requirements,
        factorization,
        kernels,
        tagged.mesh.clone(),
        element,
        dof_map.clone(),
        constraints,
        stored,
        Vec::new(),
        facet_regions,
    )
    .unwrap();

    let load_vector = plan.load_vector().unwrap();

    // Hand-compute the expected contribution at every vertex: for each `load` facet (a straight
    // segment on `x = 1`), the single-point centroid rule contributes `flux * 0.5 * |facet|` to
    // each of its two endpoint vertices (the P1 trace basis is 0.5 at the segment midpoint for
    // both endpoints), split across cells only through the shared vertex accumulation.
    let mut expected = vec![0.0; dof_map.dof_count()];
    for facet_id in &load_facets {
        let facet = FacetTopology::from_mesh(&tagged.mesh)
            .unwrap()
            .facets()
            .get(facet_id.0)
            .unwrap()
            .clone();
        assert!(facet.is_exterior());
        let vertices = &facet.vertices;
        assert_eq!(vertices.len(), 2, "GX-C4 segment facets have two vertices");
        let a = &tagged.mesh.vertices()[vertices[0].0];
        let b = &tagged.mesh.vertices()[vertices[1].0];
        let length = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        for vertex in vertices {
            expected[vertex.0] += flux * 0.5 * length;
        }
    }

    let is_dirichlet = |dof: usize| -> bool {
        tagged
            .tags
            .facet_regions
            .get(&RegionTagId::new("x_min"))
            .into_iter()
            .chain(tagged.tags.facet_regions.get(&RegionTagId::new("y_min")))
            .chain(tagged.tags.facet_regions.get(&RegionTagId::new("y_max")))
            .flatten()
            .any(|facet_id| {
                FacetTopology::from_mesh(&tagged.mesh)
                    .unwrap()
                    .facets()
                    .get(facet_id.0)
                    .unwrap()
                    .vertices
                    .iter()
                    .any(|vertex| vertex.0 == dof)
            })
    };

    let mut checked_free = 0;
    for dof in 0..dof_map.dof_count() {
        if is_dirichlet(dof) {
            continue;
        }
        checked_free += 1;
        assert!(
            (load_vector[dof] - expected[dof]).abs() <= 1.0e-9,
            "dof {dof}: got {}, expected {}",
            load_vector[dof],
            expected[dof]
        );
    }
    assert!(
        checked_free > 0,
        "fixture must have at least one free load-facet vertex"
    );
}

#[test]
fn adjoint_identity_holds_with_a_facet_term_present() {
    let (plan, _, _) = neumann_plan(2);
    let dimension = plan.dimension();
    let state = vec![0.0; dimension];
    let rate = vec![0.0; dimension];
    let direction = (0..dimension)
        .map(|index| ((index as f64 + 0.7) * 0.618_034).sin())
        .collect::<Vec<_>>();
    let adjoint = (0..dimension)
        .map(|index| ((index as f64 + 3.1) * 0.618_034).sin())
        .collect::<Vec<_>>();
    let zero_rate_direction = vec![0.0; dimension];

    let mut forward = vec![0.0; dimension];
    plan.jacobian_vector_product(
        0.0,
        &state,
        &rate,
        &direction,
        &zero_rate_direction,
        &mut forward,
    )
    .unwrap();
    let mut backward = vec![0.0; dimension];
    plan.vector_jacobian_product(0.0, &state, &rate, &adjoint, &mut backward)
        .unwrap();

    let left = forward
        .iter()
        .zip(&adjoint)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let right = direction
        .iter()
        .zip(&backward)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= 1.0e-9 * scale,
        "adjoint identity mismatch with a facet term present: <Au,v>={left}, <u,A^Tv>={right}"
    );
}

#[test]
fn unmapped_facet_region_is_refused() {
    let compilation = compile_semantics(NEUMANN_POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "NeumannPoisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let model = &compilation.semantic.models[0];

    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![2, 2],
    };
    let tagged = realize(&profile).unwrap();
    let dof_map = vector_nodal_dof_map(&tagged.mesh, 1).unwrap();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let walls_region = region_id(model, "walls");
    let mut region_map = RegionMap::new();
    region_map.insert(
        walls_region,
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
            RegionTagId::new("x_max"),
        ],
    );
    let essential_requirement = requirements.essential_constraints.first().unwrap().clone();
    let constraints = essential_constraints_from(
        &tagged,
        &dof_map,
        &[essential_requirement],
        &region_map,
        &[FieldSource::sampled(|coordinates| {
            vec![exact_u(coordinates)]
        })],
    )
    .unwrap();

    // No facet_regions entry at all for the `load` region: `new_with_facets` must refuse.
    let mut stored = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.clone();
            match (&integral.measure, name.as_str()) {
                (SemanticMeasure::Cell { .. }, "k") => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &tagged.mesh,
                        &element,
                        |_, _| vec![1.0],
                    )
                    .unwrap(),
                ),
                (SemanticMeasure::Cell { .. }, "f") => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &tagged.mesh,
                        &element,
                        |_, _| vec![-2.0],
                    )
                    .unwrap(),
                ),
                // Deliberately supply nothing for `g`; the plan must refuse before that even
                // matters, at region resolution.
                (SemanticMeasure::ExteriorFacet { .. }, "g") => {}
                (measure, other) => {
                    panic!("unexpected external input {other:?} on measure {measure:?}")
                }
            }
        }
    }

    let result = RealizationPlan::new_with_facets(
        requirements,
        factorization,
        kernels,
        tagged.mesh.clone(),
        element,
        dof_map,
        constraints,
        stored,
        Vec::new(),
        std::collections::BTreeMap::new(),
    );
    assert!(matches!(
        result,
        Err(finitum::FinitumError::RealizationRegionUnmapped(_))
    ));
}

#[test]
fn interior_facet_is_refused_by_sampled_on_facets() {
    let profile = MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![2, 2],
    };
    let tagged = realize(&profile).unwrap();
    let facet_topology = FacetTopology::from_mesh(&tagged.mesh).unwrap();
    let interior_facet_id = facet_topology
        .interior()
        .next()
        .expect("a 2x2 SimplexBox has interior facets")
        .id;
    let result = ExternalInput::sampled_on_facets(
        0,
        scientia::TensorInputId(0),
        1,
        &tagged.mesh,
        &facet_topology,
        &[interior_facet_id],
        |_, _| vec![1.0],
    );
    assert!(
        result.is_err(),
        "sampled_on_facets must refuse an interior facet"
    );
}
