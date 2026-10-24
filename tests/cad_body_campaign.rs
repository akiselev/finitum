//! R6D V8, the consumer gate (CADabra PLAN 14.1): Finitum parameter JVP/VJP and the R4 evidence
//! campaign (PLAN 6.3) against a committed CADabra Boolean.
//!
//! The body is the box `[-2, 2]^3` (its x extent `width`) minus a vertical through bore of radius
//! `r` at `(x0, 0)`, committed by the CADabra kernel and bound through Provider V1. Finitum
//! realizes it as an extruded tetrahedral mesh from the published tessellation
//! ([`CadBodyRealization::extruded`]) and assembles two objectives over the realized volume:
//!
//! * `J_w = integral of w dV`, `w = 1 + x + 2 y`: the sum of the P1 load vector of `-div grad u = w`
//!   (no essential rows). Its exact value `V (1 + C_x + 2 C_y)` comes from the provider's face
//!   integrals, so the primal and the shape derivative converge to a known reference.
//! * `J_c = F . u`, the compliance of `-div grad u = 1` with `u = 0` on the bore and natural
//!   conditions on the box: a solved, Finitum-assembled quantity. Its shape derivative is the
//!   adjoint one, `dJ_c = -u . (dR/dX(0) + dR/dX(u)) X'`, and its gradient runs the VJP chain
//!   (`residual_geometry_sensitivity_transpose`, the mesh-motion transpose, the provider's
//!   `boundary_velocity_transpose`).
//!
//! The parameters are the bore radius, the bore position and the box width, each at two designs
//! and two mesh refinements.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use cadabra::{
    BodyEvidence, BooleanOp, BooleanOptions, BooleanOutcome, BooleanRequest, BoxSpec,
    BrepBodyProvider, CylinderSpec, DesignName, DesignSeed, DesignVelocity, DisplayArtifact,
    DisplayQuality, Kernel, KernelError, Placement, PositiveLength, RunContext, Snapshot,
};
use finitum::{
    AffineConstraint, CadBodyRealization, CadBodySupport, ConstraintSet, DofId, ExternalInput,
    ExternalSensitivityInput, FinitumError, GeometryParameterSensitivity, Mesh, PreparedElement,
    RealizationPlan,
};
use methodus::{
    ConjugateGradientConfig, ConjugateGradientSymmetryPolicy, EvaluationContext, LinearOperator,
    solve_conjugate_gradient,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, TensorInputId, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};

/// The Poisson model; the compliance problem adds the bore's essential condition.
fn poisson(problem: Problem) -> String {
    let bore = match problem {
        Problem::Moment => "",
        Problem::Compliance => {
            "  boundary bore on boundary(\"bore\") { dirichlet u = exact_u(); }\n"
        }
    };
    format!(
        "module v8.poisson;\nmodel Poisson {{\n  domain Omega {{ dimension = 3; coordinates = cartesian; }}\n  \
         field u: unknown scalar H1(order=1) on Omega;\n  property k = diffusivity(0);\n  \
         source f: VolumetricSource;\n  equation balance on Omega {{ -div(k * grad(u)) = f; }}\n{bore}}}\n"
    )
}

const TARGET: &str = "gate/target";
const TOOL: &str = "gate/tool";

/// The campaign's parameters, each with the primitive that owns it.
const PARAMETERS: [(&str, DesignName); 3] = [
    (TOOL, DesignName::Radius),
    (TOOL, DesignName::OriginX),
    (TARGET, DesignName::ExtentX),
];

/// The two designs of the campaign.
const DESIGNS: [Design; 2] = [Design::through(0.6, 0.0), Design::through(1.0, 0.3)];

#[derive(Clone, Copy, Debug)]
struct Design {
    width: f64,
    r: f64,
    x0: f64,
    h: f64,
}

impl Design {
    const fn through(r: f64, x0: f64) -> Self {
        Self {
            width: 4.0,
            r,
            x0,
            h: 6.0,
        }
    }

    fn moved(mut self, name: DesignName, step: f64) -> Self {
        match name {
            DesignName::ExtentX => self.width += step,
            DesignName::Radius => self.r += step,
            DesignName::OriginX => self.x0 += step,
            DesignName::Height => self.h += step,
            other => panic!("{other:?}"),
        }
        self
    }
}

/// A mesh refinement: the published chordal tolerance and Finitum's interior generation.
#[derive(Clone, Copy, Debug)]
struct Level {
    chord: f64,
    spacing: f64,
    layers: usize,
}

const COARSE: Level = Level {
    chord: 1e-2,
    spacing: 0.5,
    layers: 8,
};
const FINE: Level = Level {
    chord: 1e-3,
    spacing: 0.25,
    layers: 16,
};

struct Published {
    snapshot: Snapshot,
    provider: BrepBodyProvider,
}

fn publish(d: Design) -> Published {
    let kernel = Kernel::default();
    let mut model = kernel.new_model(b"model/finitum-gate");
    let run = RunContext::new();
    let mut edit = model.begin_edit(model.head()).unwrap();
    let plen = |v: f64| PositiveLength::from_f64(v).unwrap();
    let target = edit
        .add_box(BoxSpec {
            name: TARGET.to_owned(),
            placement: Placement::world_at(-2.0, -2.0, -2.0).unwrap(),
            extents: [plen(d.width), plen(4.0), plen(4.0)],
        })
        .unwrap();
    let tool = edit
        .add_cylinder(CylinderSpec {
            name: TOOL.to_owned(),
            placement: Placement::world_at(d.x0, 0.0, -3.0).unwrap(),
            radius: plen(d.r),
            height: plen(d.h),
        })
        .unwrap();
    let outcome = edit
        .boolean(
            &BooleanRequest {
                op: BooleanOp::Subtract,
                target,
                tools: vec![tool],
                options: BooleanOptions::default(),
            },
            &run,
        )
        .unwrap();
    assert!(
        matches!(outcome, BooleanOutcome::Complete(_)),
        "{d:?}: {outcome:?}"
    );
    let prepared = edit.prepare(&run).map_err(|f| f.error).unwrap();
    let commit = model.publish(prepared).map_err(|f| f.error).unwrap();
    let evidence = commit.provider_evidence();
    let snapshot = model.snapshot();
    let body = &snapshot.bodies().unwrap()[0];
    let provider = snapshot.provider_body(body, &evidence).unwrap();
    Published { snapshot, provider }
}

fn tessellation(published: &Published, chord: f64) -> DisplayArtifact {
    let quality = DisplayQuality {
        chord_tol: chord,
        // Let the chordal sagitta alone set the boundary refinement.
        max_angle: 1.0,
        ..DisplayQuality::default()
    };
    published
        .snapshot
        .tessellate(quality, &RunContext::new())
        .unwrap()
}

fn realize(published: &Published, level: Level) -> CadBodyRealization {
    CadBodyRealization::extruded(
        &published.provider,
        &tessellation(published, level.chord),
        finitum::ExtrusionRefinement {
            spacing: level.spacing,
            layers: level.layers,
        },
    )
    .unwrap()
}

/// The exact `J_w` from the provider's face integrals: `V (1 + C_x + 2 C_y)`.
fn exact_objective(p: &BrepBodyProvider) -> f64 {
    let (v, c) = (p.volume().unwrap(), p.centroid().unwrap());
    v * (1.0 + c[0] + 2.0 * c[1])
}

/// Its exact derivative along `seeds`, from the volume and centroid rates.
fn exact_rate(p: &BrepBodyProvider, seeds: &[DesignSeed]) -> f64 {
    let (v, c) = (p.volume().unwrap(), p.centroid().unwrap());
    let DesignVelocity::Defined(dv) = p.volume_velocity(seeds).unwrap() else {
        panic!("volume rate")
    };
    let DesignVelocity::Defined(dc) = p.centroid_velocity(seeds).unwrap() else {
        panic!("centroid rate")
    };
    dv * (1.0 + c[0] + 2.0 * c[1]) + v * (dc[0] + 2.0 * dc[1])
}

fn w(x: &[f64]) -> f64 {
    1.0 + x[0] + 2.0 * x[1]
}

const GRAD_W: [f64; 3] = [1.0, 2.0, 0.0];

/// Which forcing and essential rows a Poisson plan carries.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Problem {
    /// `f = w`, no essential rows: the load vector sums to `J_w`.
    Moment,
    /// `f = 1`, `u = 0` on the bore: the compliance problem.
    Compliance,
}

struct Plan {
    plan: RealizationPlan,
    /// The stored inputs: (name, integral, input).
    stored: Vec<(String, usize, TensorInputId)>,
}

fn bore_face(provider: &BrepBodyProvider) -> usize {
    provider
        .binding()
        .faces
        .iter()
        .position(|f| f.support == "cylinder")
        .unwrap()
}

fn build(realization: &CadBodyRealization, bore: usize, problem: Problem) -> Plan {
    build_on(realization, realization.mesh().clone(), bore, problem)
}

/// A plan over `mesh`: the realization's mesh or a moved copy with the same cells.
fn build_on(realization: &CadBodyRealization, mesh: Mesh, bore: usize, problem: Problem) -> Plan {
    let compilation = compile_semantics(&poisson(problem), &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let model = &compilation.semantic.models[0];
    let element = PreparedElement::linear_simplex(3).unwrap();
    let mut external = Vec::new();
    let mut stored = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.clone();
            let sampled = match name.as_str() {
                "k" => ExternalInput::try_sampled(
                    integral.integral_index,
                    input.id,
                    1,
                    &mesh,
                    &element,
                    |_, _| Ok(vec![1.0]),
                ),
                "f" => ExternalInput::try_sampled(
                    integral.integral_index,
                    input.id,
                    1,
                    &mesh,
                    &element,
                    |_, point| {
                        Ok(vec![match problem {
                            Problem::Moment => w(point),
                            Problem::Compliance => 1.0,
                        }])
                    },
                ),
                other => panic!("unexpected external input {other}"),
            }
            .unwrap();
            stored.push((name, integral.integral_index, input.id));
            external.push(sampled);
        }
    }
    let essential = match problem {
        Problem::Moment => Vec::new(),
        Problem::Compliance => realization.face_vertices(bore),
    };
    let constraints = ConstraintSet::new(
        mesh.vertices().len(),
        essential.into_iter().map(|v| AffineConstraint {
            target: DofId(v.0),
            dependencies: Vec::new(),
            offset: 0.0,
        }),
    )
    .unwrap();
    let plan = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        realization.nodal_dof_map().unwrap(),
        constraints,
        external,
    )
    .unwrap();
    Plan { plan, stored }
}

/// The design directions of the stored inputs along node velocities `theta`: `k` is constant;
/// `f = w` is a fixed spatial field, so its value at a moving (barycentre) quadrature point
/// changes at `grad w . theta`.
fn sensitivity(
    realization: &CadBodyRealization,
    plan: &Plan,
    problem: Problem,
    theta: Vec<f64>,
) -> GeometryParameterSensitivity {
    let mesh = realization.mesh();
    let element = PreparedElement::linear_simplex(3).unwrap();
    assert_eq!(element.quadrature().len(), 1, "barycentre rule");
    let external_directions = plan
        .stored
        .iter()
        .map(|(name, integral, input)| {
            if name == "f" && problem == Problem::Moment {
                ExternalSensitivityInput::try_sampled(
                    *integral,
                    *input,
                    1,
                    mesh,
                    &element,
                    |cell, _| {
                        let vertices = &mesh.cells()[cell.0].vertices;
                        let rate = (0..3)
                            .map(|axis| {
                                GRAD_W[axis]
                                    * vertices.iter().map(|v| theta[3 * v.0 + axis]).sum::<f64>()
                                    / vertices.len() as f64
                            })
                            .sum::<f64>();
                        Ok(vec![rate])
                    },
                )
                .unwrap()
            } else {
                ExternalSensitivityInput::new(*integral, *input, 1, vec![0.0; mesh.cells().len()])
                    .unwrap()
            }
        })
        .collect();
    GeometryParameterSensitivity {
        parameter_index: 0,
        node_velocities: theta,
        external_directions,
    }
}

fn residual_sensitivity(plan: &Plan, state: &[f64], s: &GeometryParameterSensitivity) -> Vec<f64> {
    let mut out = vec![0.0; plan.plan.dimension()];
    plan.plan
        .residual_geometry_sensitivity(0.0, state, s, &mut out)
        .unwrap();
    out
}

/// `J_w`: the sum of the load vector (the residual at `u = 0` is `-F`).
fn moment(plan: &Plan) -> f64 {
    plan.plan.load_vector().unwrap().iter().sum()
}

/// `dJ_w` along node velocities `theta`.
fn moment_rate(realization: &CadBodyRealization, plan: &Plan, theta: Vec<f64>) -> f64 {
    let zero = vec![0.0; plan.plan.dimension()];
    let s = sensitivity(realization, plan, Problem::Moment, theta);
    -residual_sensitivity(plan, &zero, &s).iter().sum::<f64>()
}

/// Jacobi preconditioning from the element matrices' scattered diagonal (essential rows are
/// identity rows of the operator).
struct Jacobi(Vec<f64>);

impl methodus::Preconditioner for Jacobi {
    fn dimension(&self) -> usize {
        self.0.len()
    }

    fn apply_inverse(
        &self,
        _context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), methodus::NumericError> {
        for ((o, r), d) in output.iter_mut().zip(right_hand_side).zip(&self.0) {
            *o = r / d;
        }
        Ok(())
    }
}

fn solve(plan: &Plan, rhs: &[f64]) -> Vec<f64> {
    let operator = plan.plan.element_assembly(8).unwrap();
    let artifact = plan.plan.artifact();
    let mut diagonal = vec![0.0; operator.rows()];
    for (restriction, matrix) in artifact
        .dofs
        .restrictions()
        .iter()
        .zip(operator.local_matrices())
    {
        let n = restriction.dofs.len();
        for (local, dof) in restriction.dofs.iter().enumerate() {
            diagonal[dof.0] += matrix[local * n + local];
        }
    }
    for constraint in artifact.constraints.constraints() {
        diagonal[constraint.target.0] = 1.0;
    }
    let config = ConjugateGradientConfig {
        max_iterations: 20_000,
        absolute_tolerance: 1e-14,
        relative_tolerance: 1e-13,
        symmetry_policy: ConjugateGradientSymmetryPolicy::AssumeSymmetric,
    };
    let initial = vec![0.0; operator.rows()];
    let report = solve_conjugate_gradient(
        &operator,
        Some(&Jacobi(diagonal)),
        &EvaluationContext::reproducible(),
        rhs,
        &initial,
        &config,
    )
    .unwrap();
    assert!(report.converged, "CG did not converge");
    report.solution
}

struct Compliance {
    value: f64,
    state: Vec<f64>,
}

fn compliance(plan: &Plan) -> Compliance {
    let load = plan.plan.load_vector().unwrap();
    let state = solve(plan, &load);
    Compliance {
        value: load.iter().zip(&state).map(|(f, u)| f * u).sum(),
        state,
    }
}

/// The adjoint shape derivative `dJ_c = -u . (dR/dX(0) + dR/dX(u)) theta`.
fn compliance_rate(
    realization: &CadBodyRealization,
    plan: &Plan,
    c: &Compliance,
    theta: Vec<f64>,
) -> f64 {
    let zero = vec![0.0; plan.plan.dimension()];
    let s = sensitivity(realization, plan, Problem::Compliance, theta);
    let (s0, su) = (
        residual_sensitivity(plan, &zero, &s),
        residual_sensitivity(plan, &c.state, &s),
    );
    -c.state
        .iter()
        .zip(s0.iter().zip(&su))
        .map(|(u, (a, b))| u * (a + b))
        .sum::<f64>()
}

/// The node gradient `dJ_c/dX = -(dR/dX(0)^T u + dR/dX(u)^T u)`.
fn compliance_node_gradient(plan: &Plan, c: &Compliance) -> Vec<f64> {
    let n = 3 * plan.plan.artifact().mesh.vertices().len();
    let zero = vec![0.0; plan.plan.dimension()];
    let (mut g0, mut gu) = (vec![0.0; n], vec![0.0; n]);
    plan.plan
        .residual_geometry_sensitivity_transpose(0.0, &zero, &c.state, &mut g0)
        .unwrap();
    plan.plan
        .residual_geometry_sensitivity_transpose(0.0, &c.state, &c.state, &mut gu)
        .unwrap();
    g0.iter().zip(&gu).map(|(a, b)| -(a + b)).collect()
}

fn seed(namespace: &str, name: DesignName, rate: f64) -> DesignSeed {
    DesignSeed {
        namespace: namespace.to_owned(),
        name,
        rate,
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The mesh moved by `step` along node velocities `theta`, with the same cells.
fn moved_mesh(realization: &CadBodyRealization, theta: &[f64], step: f64) -> Mesh {
    let mut vertices = realization.mesh().vertices().to_vec();
    for (i, v) in vertices.iter_mut().enumerate() {
        for axis in 0..3 {
            v[axis] += step * theta[3 * i + axis];
        }
    }
    Mesh::new(3, vertices, realization.mesh().cells().to_vec()).unwrap()
}

const STEPS: [f64; 3] = [1e-2, 1e-3, 1e-4];

/// Centred differences of `j` about zero and Taylor remainders against `rate`, at [`STEPS`].
fn differences(j0: f64, rate: f64, j: impl Fn(f64) -> f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let mut centred = [0.0; 3];
    let mut errors = [0.0; 3];
    let mut remainders = [0.0; 3];
    for (k, step) in STEPS.into_iter().enumerate() {
        let (plus, minus) = (j(step), j(-step));
        centred[k] = (plus - minus) / (2.0 * step);
        errors[k] = (centred[k] - rate).abs();
        remainders[k] = (plus - j0 - step * rate).abs();
    }
    (centred, errors, remainders)
}

/// Second-order centred differences and a Taylor remainder that falls a hundredfold per decade,
/// down to rounding (`floor`).
fn assert_consistent(what: &str, errors: [f64; 3], remainders: [f64; 3], rate: f64, floor: f64) {
    assert!(
        errors[2] <= 1e-5 * (1.0 + rate.abs()) && errors[1] <= 1e-3 * (1.0 + rate.abs()),
        "{what}: centred errors {errors:?} (rate {rate})"
    );
    for k in 0..2 {
        if errors[k + 1] > floor {
            assert!(
                errors[k] / errors[k + 1] > 50.0,
                "{what}: centred errors {errors:?}"
            );
        }
        if remainders[k + 1] > floor {
            let ratio = remainders[k] / remainders[k + 1];
            assert!(
                (50.0..200.0).contains(&ratio),
                "{what}: Taylor remainders {remainders:?}"
            );
        }
    }
}

/// The distance a parameter's centred CAD difference may sit from the Finitum derivative: the
/// rate is exact for the discrete objective along Finitum's mesh motion, which slides published
/// samples only normally; regenerating the mesh from a published body slides them with their
/// face charts. With the same tessellation topology on both sides, a parameter that moves its
/// samples only normally (radius, width) differs by rounding; the bore position differs by that
/// gauge, which falls with the chordal tolerance.
fn cad_bound(name: DesignName, same_topology: bool, level: Level, rate: f64) -> f64 {
    if same_topology && name != DesignName::OriginX {
        1e-6 * (1.0 + rate.abs())
    } else {
        3.0 * level.chord * (1.0 + rate.abs())
    }
}

/// The R4 campaign for one design at both refinements; returns nothing, asserts everything.
fn campaign(design: Design) {
    let published = publish(design);
    let bore = bore_face(&published.provider);
    let exact = exact_objective(&published.provider);
    // Per level: |J_w - exact|, per parameter |dJ_w - exact rate|, the bore position's CAD
    // gauge for J_w (at step 1e-4) and J_c (at the smallest step that keeps the topology).
    let mut primal = Vec::new();
    let mut rate_errors = Vec::new();
    let mut gauge_w = Vec::new();
    let mut gauge_c = Vec::new();
    for level in [COARSE, FINE] {
        let realization = realize(&published, level);
        let moment_plan = build(&realization, bore, Problem::Moment);
        let j_w = moment(&moment_plan);
        primal.push((j_w - exact).abs());
        let full_compliance = level.chord == COARSE.chord;
        let compliance_plan = build(&realization, bore, Problem::Compliance);
        let j_c = compliance(&compliance_plan);
        let mut level_rate_errors = Vec::new();
        for (namespace, name) in PARAMETERS {
            let seeds = [seed(namespace, name, 1.0)];
            let what = format!("{design:?} {name:?} {level:?}");
            let theta = realization
                .design_velocity(&published.provider, &seeds)
                .unwrap();
            let rate_w = moment_rate(&realization, &moment_plan, theta.clone());
            let exact_rate = exact_rate(&published.provider, &seeds);
            level_rate_errors.push((rate_w - exact_rate).abs());

            // The Finitum chain: the objective on the realized mesh moved along the mesh motion.
            let (_, errors, remainders) = differences(j_w, rate_w, |step| {
                moment(&build_on(
                    &realization,
                    moved_mesh(&realization, &theta, step),
                    bore,
                    Problem::Moment,
                ))
            });
            assert_consistent(
                &format!("{what} J_w moved"),
                errors,
                remainders,
                rate_w,
                1e-8 * (1.0 + j_w.abs()),
            );

            // Through the CAD: publish the moved design, tessellate, realize, assemble.
            let regenerated = |step: f64| {
                let moved = publish(design.moved(name, step));
                let r = realize(&moved, level);
                let same = r.mesh().cells() == realization.mesh().cells();
                (moved, r, same)
            };
            for step in STEPS {
                let ((_, plus, same_plus), (_, minus, same_minus)) =
                    (regenerated(step), regenerated(-step));
                let same = same_plus && same_minus;
                let (j_plus, j_minus) = (
                    moment(&build(&plus, bore, Problem::Moment)),
                    moment(&build(&minus, bore, Problem::Moment)),
                );
                let centred = (j_plus - j_minus) / (2.0 * step);
                let bound = cad_bound(name, same, level, rate_w);
                eprintln!(
                    "{what} step {step:e}: J_w rate {rate_w:.9} (exact {exact_rate:.9}) CAD centred {centred:.9} same topology {same}"
                );
                assert!(
                    (centred - rate_w).abs() <= bound,
                    "{what} step {step}: J_w CAD centred {centred} vs {rate_w} (bound {bound})"
                );
                let remainder = (j_plus - j_w - step * rate_w).abs();
                assert!(
                    remainder <= step * bound + 50.0 * step * step * (1.0 + j_w.abs()),
                    "{what} step {step}: J_w CAD Taylor remainder {remainder}"
                );
                if name == DesignName::OriginX && step == 1e-4 {
                    gauge_w.push((centred - rate_w).abs());
                }
            }

            // Compliance: the full chain at the coarse level, the bore position's gauge at both.
            if full_compliance || name == DesignName::OriginX {
                let rate_c = compliance_rate(&realization, &compliance_plan, &j_c, theta.clone());
                if full_compliance {
                    let (_, errors, remainders) = differences(j_c.value, rate_c, |step| {
                        compliance(&build_on(
                            &realization,
                            moved_mesh(&realization, &theta, step),
                            bore,
                            Problem::Compliance,
                        ))
                        .value
                    });
                    assert_consistent(
                        &format!("{what} J_c moved"),
                        errors,
                        remainders,
                        rate_c,
                        1e-8 * (1.0 + j_c.value),
                    );
                }
                let centred_c = |step: f64| -> Option<f64> {
                    let ((_, plus, same_plus), (_, minus, same_minus)) =
                        (regenerated(step), regenerated(-step));
                    if !(same_plus && same_minus) {
                        eprintln!(
                            "{what} step {step:e}: J_c CAD difference skipped: the tessellation topology changed"
                        );
                        return None;
                    }
                    let centred = (compliance(&build(&plus, bore, Problem::Compliance)).value
                        - compliance(&build(&minus, bore, Problem::Compliance)).value)
                        / (2.0 * step);
                    eprintln!(
                        "{what} step {step:e}: J_c rate {rate_c:.9} CAD centred {centred:.9}"
                    );
                    Some(centred)
                };
                let check = |step: f64, centred: f64| {
                    if name != DesignName::OriginX {
                        assert!(
                            (centred - rate_c).abs()
                                <= 1e-3 * (step / 1e-4).max(1.0) * (1.0 + rate_c.abs()),
                            "{what} step {step}: J_c CAD centred {centred} vs {rate_c}"
                        );
                    }
                };
                if full_compliance {
                    for step in &STEPS[..2] {
                        if let Some(centred) = centred_c(*step) {
                            check(*step, centred);
                        }
                    }
                }
                // A step that crosses no tessellation decision is required: the first of
                // 1e-4, 1e-5, 1e-6 that keeps the topology on both sides.
                let (step, centred) = [1e-4, 1e-5, 1e-6]
                    .into_iter()
                    .find_map(|step| centred_c(step).map(|c| (step, c)))
                    .unwrap_or_else(|| panic!("{what}: every small step changes the topology"));
                check(step, centred);
                if name == DesignName::OriginX {
                    gauge_c.push((centred - rate_c).abs());
                }
            }
        }
        rate_errors.push(level_rate_errors);
        eprintln!(
            "{design:?} {level:?}: {} vertices, {} tetrahedra; J_w {j_w} (exact {exact}), J_c {}",
            realization.mesh().vertices().len(),
            realization.mesh().cells().len(),
            j_c.value
        );
    }
    eprintln!(
        "{design:?}: primal {primal:?} rate errors {rate_errors:?} gauge J_w {gauge_w:?} J_c {gauge_c:?}"
    );
    // Convergence under refinement (tenfold chordal tolerance): the primal and the shape
    // derivative of J_w approach the exact values, first order in the sagitta; the box width
    // moves a planar face, so its rate is exact at both refinements.
    assert!(primal[1] < primal[0] / 4.0, "J_w primal {primal:?}");
    for (k, (_, name)) in PARAMETERS.iter().enumerate() {
        let (coarse, fine) = (rate_errors[0][k], rate_errors[1][k]);
        if *name == DesignName::ExtentX {
            assert!(coarse < 1e-9 && fine < 1e-9, "{name:?}: {coarse} {fine}");
        } else {
            assert!(
                fine < coarse / 4.0,
                "{name:?} rate error {coarse} -> {fine}"
            );
        }
    }
    // The bore position's gauge between Finitum's motion and the regenerated mesh falls too.
    assert!(gauge_w[1] < gauge_w[0] / 4.0, "J_w gauge {gauge_w:?}");
    assert!(
        gauge_c.len() == 2 && gauge_c[1] < gauge_c[0],
        "J_c gauge {gauge_c:?}"
    );
}

#[test]
fn the_r4_campaign_at_the_centred_bore() {
    campaign(DESIGNS[0]);
}

#[test]
fn the_r4_campaign_at_the_offset_bore() {
    campaign(DESIGNS[1]);
}

#[test]
fn the_extruded_realization_associates_every_boundary_node_with_its_b_rep_entity() {
    let design = DESIGNS[1];
    let published = publish(design);
    let realization = realize(&published, COARSE);
    // Deterministic.
    let again = realize(&published, COARSE);
    assert_eq!(realization, again);
    assert_eq!(realization.digest(), again.digest());
    realization.require_source(&published.provider).unwrap();

    let binding = published.provider.binding();
    let mesh = realization.mesh();
    let position = |v: usize| -> [f64; 3] {
        let x = &mesh.vertices()[v];
        [x[0], x[1], x[2]]
    };
    // Every tetrahedron is positively oriented; the volume is the polyhedron's, within the
    // chordal error of the exact volume.
    let mut volume = 0.0;
    for cell in mesh.cells() {
        let [a, b, c, d] = [0, 1, 2, 3].map(|k| position(cell.vertices[k].0));
        let (u, v, w) = (
            [0, 1, 2].map(|k| b[k] - a[k]),
            [0, 1, 2].map(|k| c[k] - a[k]),
            [0, 1, 2].map(|k| d[k] - a[k]),
        );
        let det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]);
        assert!(det > 0.0);
        volume += det / 6.0;
    }
    let exact = published.provider.volume().unwrap();
    assert!(
        (volume - exact).abs() < 2.0 * COARSE.chord * 2.0 * std::f64::consts::PI * design.r * 4.0,
        "volume {volume} vs {exact}"
    );

    // Each node's support names the faces of its incidences, and an edge or vertex those
    // faces share.
    let mut kinds = [0usize; 4];
    for node in realization.nodes() {
        let faces = realization
            .incidences()
            .iter()
            .filter(|i| i.vertex == node.vertex)
            .map(|i| i.face)
            .collect::<Vec<_>>();
        match &node.support {
            CadBodySupport::Interior => {
                kinds[0] += 1;
                assert!(faces.is_empty());
                let x = position(node.vertex.0);
                assert!(x.iter().all(|c| c.abs() < 2.0 - 1e-9));
                assert!((x[0] - design.x0).hypot(x[1]) > design.r);
            }
            CadBodySupport::Face { face } => {
                kinds[1] += 1;
                assert_eq!(faces, vec![*face]);
            }
            CadBodySupport::Edge { edge, faces: pair } => {
                kinds[2] += 1;
                assert_eq!(faces, pair.to_vec());
                let edge = &binding.edges[*edge];
                for face in pair {
                    assert!(edge.faces.contains(&binding.faces[*face].uid));
                }
            }
            CadBodySupport::Vertex {
                vertex,
                faces: list,
            } => {
                kinds[3] += 1;
                assert_eq!(&faces, list);
                assert_eq!(list.len(), 3);
                let x = position(node.vertex.0);
                assert!(
                    (0..3).all(|k| (binding.vertices[*vertex].position[k] - x[k]).abs() < 1e-12)
                );
            }
        }
    }
    // The box's eight corners, and both kinds of nodes elsewhere.
    assert_eq!(kinds[3], 8, "{kinds:?}");
    assert!(kinds.iter().all(|&k| k > 0), "{kinds:?}");

    // Every incidence lies on its face: planes exactly, the bore at its radius (published
    // samples) or on a chord inside it (rim splits).
    let bore = bore_face(&published.provider);
    for incidence in realization.incidences() {
        let x = incidence.point;
        if incidence.face == bore {
            let radius = (x[0] - design.x0).hypot(x[1]);
            assert!(radius <= design.r + 1e-12 && radius >= design.r - COARSE.chord);
            let radial = [(x[0] - design.x0) / radius, x[1] / radius, 0.0];
            // The bore's outward material normal points into the hole.
            assert!(dot(&radial, &incidence.normal) < -1.0 + 1e-3);
        } else {
            let n = incidence.normal;
            let offset = dot(&n, &x);
            assert!(
                (offset - 2.0).abs() < 1e-12,
                "a box face incidence is off its plane: {offset}"
            );
        }
    }

    // Every boundary triangle of the mesh (a tetrahedron face used once) lies on one face.
    let mut uses = std::collections::BTreeMap::<[usize; 3], usize>::new();
    for cell in mesh.cells() {
        let v = cell.vertices.iter().map(|v| v.0).collect::<Vec<_>>();
        for skip in 0..4 {
            let mut f = [0; 3];
            let mut k = 0;
            for (i, &vertex) in v.iter().enumerate() {
                if i != skip {
                    f[k] = vertex;
                    k += 1;
                }
            }
            f.sort_unstable();
            *uses.entry(f).or_default() += 1;
        }
    }
    let faces_of = |v: usize| {
        realization
            .incidences()
            .iter()
            .filter(|i| i.vertex.0 == v)
            .map(|i| i.face)
            .collect::<Vec<_>>()
    };
    let mut boundary = 0;
    for (triangle, count) in uses {
        assert!(count <= 2);
        if count == 1 {
            boundary += 1;
            let common = faces_of(triangle[0])
                .into_iter()
                .filter(|f| faces_of(triangle[1]).contains(f) && faces_of(triangle[2]).contains(f))
                .count();
            assert!(
                common >= 1,
                "boundary triangle {triangle:?} is on no single face"
            );
        }
    }
    assert!(boundary > 0);
    assert!(!realization.face_vertices(bore).is_empty());
}

#[test]
fn the_parameter_vjp_is_the_transpose_of_the_jvp() {
    let published = publish(DESIGNS[1]);
    let provider = &published.provider;
    let realization = realize(&published, COARSE);
    let bore = bore_face(provider);
    let plan = build(&realization, bore, Problem::Compliance);
    let c = compliance(&plan);
    let pseudo = |n: usize, seed: usize| -> Vec<f64> {
        (0..n)
            .map(|i| ((i * seed + 7) % 101) as f64 / 101.0 - 0.5)
            .collect()
    };

    // The mesh motion and its transpose.
    let (incidences, nodes3) = (
        realization.incidences().len(),
        3 * realization.mesh().vertices().len(),
    );
    let (v, g) = (pseudo(incidences, 31), pseudo(nodes3, 17));
    let lhs = dot(&g, &realization.mesh_velocity(&v).unwrap());
    let rhs = dot(&realization.mesh_velocity_transpose(&g).unwrap(), &v);
    assert!(
        (lhs - rhs).abs() <= 1e-12 * (1.0 + lhs.abs()),
        "{lhs} vs {rhs}"
    );

    // The residual's geometry sensitivity and its transpose, at the solved state.
    let a = pseudo(plan.plan.dimension(), 13);
    let theta = pseudo(nodes3, 23);
    let s = sensitivity(&realization, &plan, Problem::Compliance, theta.clone());
    let forward = dot(&a, &residual_sensitivity(&plan, &c.state, &s));
    let mut transposed = vec![0.0; nodes3];
    plan.plan
        .residual_geometry_sensitivity_transpose(0.0, &c.state, &a, &mut transposed)
        .unwrap();
    let back = dot(&transposed, &theta);
    assert!(
        (forward - back).abs() <= 1e-11 * (1.0 + forward.abs()),
        "{forward} vs {back}"
    );

    // The parameter VJP: dJ_c/dp from the node gradient through the mesh-motion transpose and
    // the provider's boundary-velocity transpose, against the JVP of each parameter.
    let node_gradient = compliance_node_gradient(&plan, &c);
    let parameters = PARAMETERS
        .iter()
        .map(|(ns, name)| ((*ns).to_owned(), *name))
        .collect::<Vec<_>>();
    let gradient = realization
        .design_gradient(provider, &node_gradient, &parameters)
        .unwrap();
    let columns = PARAMETERS
        .iter()
        .map(|(ns, name)| {
            realization
                .design_velocity(provider, &[seed(ns, *name, 1.0)])
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (k, column) in columns.iter().enumerate() {
        let jvp = compliance_rate(&realization, &plan, &c, column.clone());
        assert!(
            (gradient[k] - jvp).abs() <= 1e-9 * (1.0 + jvp.abs()),
            "{:?}: VJP {} vs JVP {jvp}",
            PARAMETERS[k],
            gradient[k]
        );
    }

    // Assembled (per-parameter columns) against matrix-free (one combined direction), and the
    // dot-product identity <dJ/dX, X'(direction)> = <dJ/dp, direction>.
    let direction = [0.7, -1.3, 0.4];
    let seeds = PARAMETERS
        .iter()
        .zip(direction)
        .map(|((ns, name), rate)| seed(ns, *name, rate))
        .collect::<Vec<_>>();
    let combined = realization.design_velocity(provider, &seeds).unwrap();
    for (i, value) in combined.iter().enumerate() {
        let assembled: f64 = columns.iter().zip(direction).map(|(c, d)| c[i] * d).sum();
        assert!(
            (value - assembled).abs() <= 1e-13 * (1.0 + value.abs()),
            "node component {i}"
        );
    }
    let lhs = dot(&node_gradient, &combined);
    let rhs = dot(&gradient, &direction);
    assert!(
        (lhs - rhs).abs() <= 1e-10 * (1.0 + lhs.abs()),
        "{lhs} vs {rhs}"
    );

    // Finitum's assembled and matrix-free operators agree on the realized body.
    let x = pseudo(plan.plan.dimension(), 41);
    let context = EvaluationContext::reproducible();
    let (mut free, mut element) = (vec![0.0; x.len()], vec![0.0; x.len()]);
    plan.plan
        .matrix_free()
        .apply(&context, &x, &mut free)
        .unwrap();
    plan.plan
        .element_assembly(8)
        .unwrap()
        .apply(&context, &x, &mut element)
        .unwrap();
    for (a, b) in free.iter().zip(&element) {
        assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()));
    }

    // Wrong lengths are refused.
    assert!(matches!(
        realization.mesh_velocity(&v[1..]),
        Err(FinitumError::InvalidCadGeometry(_))
    ));
    assert!(matches!(
        realization.design_gradient(provider, &node_gradient[1..], &parameters),
        Err(FinitumError::InvalidCadGeometry(_))
    ));
}

#[test]
fn snapped_decisions_and_stale_revisions_are_refused() {
    // The bore's cap flush with the box top (z = -3 + 5 = 2): the flush decision is snapped
    // and the tool's height moves it, so no smooth derivative exists; Finitum refuses with the
    // band exit rather than differentiating the snapped stratum.
    let flush = publish(Design {
        h: 5.0,
        ..Design::through(1.0, 0.0)
    });
    let realization = realize(&flush, COARSE);
    let height = [seed(TOOL, DesignName::Height, 1.0)];
    assert!(matches!(
        realization.design_velocity(&flush.provider, &height),
        Err(FinitumError::SnappedCadDesign { exit: Some(_) })
    ));
    let n = 3 * realization.mesh().vertices().len();
    assert!(matches!(
        realization.design_gradient(
            &flush.provider,
            &vec![1.0; n],
            &[(TOOL.to_owned(), DesignName::Height)]
        ),
        Err(FinitumError::SnappedCadDesign { exit: Some(_) })
    ));
    // The radius moves no snapped decision there.
    realization
        .design_velocity(&flush.provider, &[seed(TOOL, DesignName::Radius, 1.0)])
        .unwrap();

    // Stale revisions: a realization refuses a provider of another revision, and a
    // tessellation of another revision cannot be realized against a provider.
    let other = publish(Design::through(0.8, 0.0));
    assert_eq!(
        realization.require_source(&other.provider),
        Err(FinitumError::StaleBrepRevision)
    );
    assert_eq!(
        realization.design_velocity(&other.provider, &[seed(TOOL, DesignName::Radius, 1.0)]),
        Err(FinitumError::StaleBrepRevision)
    );
    assert_eq!(
        CadBodyRealization::extruded(
            &other.provider,
            &tessellation(&flush, COARSE.chord),
            finitum::ExtrusionRefinement {
                spacing: COARSE.spacing,
                layers: COARSE.layers
            }
        ),
        Err(FinitumError::StaleBrepRevision)
    );
    // The kernel itself refuses to bind a body of one revision against another snapshot.
    let stale = &flush.snapshot.bodies().unwrap()[0];
    assert!(matches!(
        other
            .snapshot
            .provider_body(stale, &BodyEvidence::default()),
        Err(KernelError::WrongRevision)
    ));

    // A blind bore (its cap inside the box) is not an extrusion: refused, not mis-meshed.
    let blind = publish(Design {
        h: 4.0,
        ..Design::through(1.0, 0.0)
    });
    assert!(matches!(
        CadBodyRealization::extruded(
            &blind.provider,
            &tessellation(&blind, COARSE.chord),
            finitum::ExtrusionRefinement {
                spacing: COARSE.spacing,
                layers: COARSE.layers
            }
        ),
        Err(FinitumError::InvalidCadGeometry(_))
    ));
}
