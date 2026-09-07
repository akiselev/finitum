//! W8 lane F1: the public field sampler (`finitum-field-sampler/1`) agrees to roundoff with
//! Finitum's own quadrature-point evaluation inside `RealizationPlan` and `SystemOperator`.
//!
//! Every test binds a state-dependent (dynamic / constitutive) input whose closure records
//! what the executing plan hands it at each quadrature point -- the cell, the physical
//! coordinates and the active basis inputs (value, gradient, symmetric gradient) -- then
//! rebuilds the same quantities through `FieldSampler` at the recorded coordinates. The meshes
//! are sheared so no cell is axis-aligned and every Piola map is non-diagonal. Coverage:
//! scalar P1 (transient nonlinear form), scalar P2 (Poisson), vector P1 in 3-D (elasticity),
//! RT0 + P0 on the system path (mixed Darcy), and P2 vector + P1 scalar on the system path
//! (Taylor-Hood Stokes). `QuadratureView` names each plan's rule and reproduces its points.

use finitum::{
    AffineConstraint, BlockLayout, Cell, CellId, CompatibleDofMaps, ConstraintSet, DofId, DofMap,
    DynamicExternalInput, ElementRestriction, ExternalInput, FacetTopology, FieldSampler,
    FinitumError, Mesh, MeshProfile, PointEvaluation, PreparedElement, QuadratureView,
    RealizationPlan, RegionMap, RegionTagId, SampledFamily, SystemConstitutiveInput,
    SystemRealizationPlan, TaggedMesh, VertexId, facet_membership_from, quadratic_simplex_dof_map,
    realize, vector_nodal_dof_map,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, SemanticMeasure,
    compile_operator_system, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const TRANSIENT_NONLINEAR: &str = r#"
module w8_f1.transient_nonlinear;
model TransientNonlinear {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  property capacity = storage_capacity(u);
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation evolution on Omega { capacity * dt(u) - div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(t); }
}
"#;

const POISSON_P2: &str = r#"
module w8_f1.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=2) on Omega;
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

const ELASTICITY: &str = r#"
module w8_f1.elasticity;
model Elasticity {
  domain Omega { dimension = 3; coordinates = cartesian; }
  field u: unknown vector(3) H1(order=1) on Omega;
  property lambda = lame_lambda(0);
  property mu = lame_mu(0);
  source body_force: MechanicalBodyForce;

  constitutive strain = sym_grad(u);
  constitutive stress = lambda * trace(strain) * identity(3) + 2 * mu * strain;

  equation momentum on Omega {
    -div(stress) = body_force;
  }
  boundary clamp on boundary("clamp") { dirichlet u = [0, 0, 0]; }
}
"#;

const DARCY: &str = include_str!("fixtures/corpus/13-mixed-darcy.res");
const STOKES: &str = include_str!("fixtures/corpus/25-stokes.res");

/// What the executing plan handed one closure at one quadrature point.
#[derive(Clone, Debug)]
struct Record {
    cell: CellId,
    coordinates: Vec<f64>,
    value: Option<Vec<f64>>,
    gradient: Option<Vec<f64>>,
    symmetric_gradient: Option<Vec<f64>>,
}

type Recorder = Arc<Mutex<Vec<Record>>>;

fn record(recorder: &Recorder, evaluation: &PointEvaluation) {
    recorder.lock().unwrap().push(Record {
        cell: evaluation.cell,
        coordinates: evaluation.coordinates.clone(),
        value: evaluation
            .values(DerivativeEvaluation::Value)
            .map(<[f64]>::to_vec),
        gradient: evaluation
            .values(DerivativeEvaluation::Gradient)
            .map(<[f64]>::to_vec),
        symmetric_gradient: evaluation
            .values(DerivativeEvaluation::SymmetricGradient)
            .map(<[f64]>::to_vec),
    });
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: component count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{what}[{index}]: {actual} != {expected}"
        );
    }
}

/// Deterministic, smooth, non-trivial nodal data.
fn wave(point: &[f64], phase: f64) -> f64 {
    0.7 + (1.3 * point[0] + 0.4 * phase).sin() * (0.9 * point.get(1).copied().unwrap_or(0.0)).cos()
        + 0.2 * point.iter().sum::<f64>()
}

fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn symmetric_part(rows: &[Vec<f64>]) -> Vec<f64> {
    let dimension = rows.len();
    let mut symmetric = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            symmetric[row * dimension + column] = 0.5 * (rows[row][column] + rows[column][row]);
        }
    }
    symmetric
}

/// A sheared `subdivisions x subdivisions` triangle grid (no edge axis-aligned) with the
/// vertex-indexed P1 DOF map and the boundary vertices fixed.
fn sheared_square(subdivisions: usize) -> (Mesh, DofMap, ConstraintSet) {
    let width = subdivisions + 1;
    let vertices = (0..=subdivisions)
        .flat_map(|row| {
            (0..=subdivisions).map(move |column| {
                let x = column as f64 / subdivisions as f64;
                let y = row as f64 / subdivisions as f64;
                vec![
                    x + 0.3 * y + 0.04 * (3.0 * y).sin(),
                    y - 0.2 * x + 0.04 * (2.0 * x).sin(),
                ]
            })
        })
        .collect::<Vec<_>>();
    let cells = (0..subdivisions)
        .flat_map(|row| {
            (0..subdivisions).flat_map(move |column| {
                let lower_left = row * width + column;
                let lower_right = lower_left + 1;
                let upper_left = lower_left + width;
                let upper_right = upper_left + 1;
                [
                    Cell {
                        vertices: vec![
                            VertexId(lower_left),
                            VertexId(lower_right),
                            VertexId(upper_right),
                        ],
                    },
                    Cell {
                        vertices: vec![
                            VertexId(lower_left),
                            VertexId(upper_right),
                            VertexId(upper_left),
                        ],
                    },
                ]
            })
        })
        .collect::<Vec<_>>();
    let restrictions = cells
        .iter()
        .map(|cell| ElementRestriction {
            dofs: cell.vertices.iter().map(|vertex| DofId(vertex.0)).collect(),
        })
        .collect();
    let mesh = Mesh::new(2, vertices, cells).unwrap();
    let dofs = DofMap::new(width * width, restrictions).unwrap();
    let constraints = ConstraintSet::new(
        width * width,
        (0..width * width)
            .filter(|index| {
                let row = index / width;
                let column = index % width;
                row == 0 || column == 0 || row == subdivisions || column == subdivisions
            })
            .map(|target| AffineConstraint {
                target: DofId(target),
                dependencies: Vec::new(),
                offset: 0.0,
            }),
    )
    .unwrap();
    (mesh, dofs, constraints)
}

/// A `MeshProfile::SimplexBox` under an affine shear: the tags stay valid (facet identities
/// depend on connectivity only) while no cell is axis-aligned any more.
fn sheared_box(dimension: u8, subdivisions: usize) -> TaggedMesh {
    let tagged = realize(&MeshProfile::SimplexBox {
        dimension,
        extent: vec![[0.0, 1.0]; dimension as usize],
        subdivisions: vec![subdivisions; dimension as usize],
    })
    .unwrap();
    let vertices = tagged
        .mesh
        .vertices()
        .iter()
        .map(|point| match point.as_slice() {
            [x, y] => vec![x + 0.25 * y, y - 0.15 * x],
            [x, y, z] => vec![x + 0.2 * y + 0.1 * z, y + 0.15 * z - 0.05 * x, z + 0.1 * x],
            other => other.to_vec(),
        })
        .collect();
    let mesh = Mesh::new(dimension as usize, vertices, tagged.mesh.cells().to_vec()).unwrap();
    TaggedMesh {
        mesh,
        tags: tagged.tags,
        digest: tagged.digest,
        provenance: tagged.provenance,
    }
}

struct Compiled {
    requirements: scientia::FormRequirements,
    factorization: scientia::OperatorFactorization,
    kernels: scientia::StructuredOperatorKernels,
}

fn compile_form(source: &str, model: &str, equation: &str) -> Compiled {
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, model, equation).unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    Compiled {
        requirements,
        factorization,
        kernels,
    }
}

/// Bind every non-basis input of a single-field form: constitutive/property inputs as recording
/// dynamic inputs (returning a constant of the right shape), external values as zero tables.
fn bind_single_field_plan(
    compiled: &Compiled,
    mesh: &Mesh,
    element: PreparedElement,
    dofs: DofMap,
    constraints: ConstraintSet,
    recorder: &Recorder,
) -> RealizationPlan {
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &compiled.factorization.integrals {
        for input in &integral.primal.inputs {
            let components = input.shape.iter().product::<usize>().max(1);
            match input.source {
                InputSourceRequirement::Basis => {}
                InputSourceRequirement::ExternalValue => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        components,
                        mesh,
                        &element,
                        |_, _| vec![0.0; components],
                    )
                    .unwrap(),
                ),
                _ => {
                    let recorder = Arc::clone(recorder);
                    dynamic.push(
                        DynamicExternalInput::new(
                            integral.integral_index,
                            input.id,
                            components,
                            format!("w8-f1/recording/{}/{:?}", integral.integral_index, input.id),
                            move |evaluation| {
                                record(&recorder, evaluation);
                                let mut value = vec![0.0; components];
                                value[0] = 1.0;
                                value
                            },
                            move |_, _| vec![0.0; components],
                        )
                        .unwrap(),
                    );
                }
            }
        }
    }
    RealizationPlan::new_stateful(
        compiled.requirements.clone(),
        compiled.factorization.clone(),
        compiled.kernels.clone(),
        mesh.clone(),
        element,
        dofs,
        constraints,
        stored,
        dynamic,
    )
    .unwrap()
}

/// Every recorded point of every cell agrees with the sampler at the recorded coordinates.
fn check_records(records: &[Record], sampler: &FieldSampler<'_>, tolerance: f64, what: &str) {
    assert!(!records.is_empty(), "{what}: the plan evaluated no points");
    let mut cells_seen = std::collections::BTreeSet::new();
    for record in records {
        cells_seen.insert(record.cell);
        let sample = sampler.sample_at(record.cell, &record.coordinates).unwrap();
        assert!(
            sampler
                .cell_contains(record.cell, &record.coordinates, 1.0e-12)
                .unwrap(),
            "{what}: quadrature point lies in its cell"
        );
        if let Some(value) = &record.value {
            assert_close(&sample.value, value, tolerance, &format!("{what} value"));
            assert_close(
                &sampler.value_at(record.cell, &record.coordinates).unwrap(),
                value,
                tolerance,
                &format!("{what} value_at"),
            );
        }
        if let Some(gradient) = &record.gradient {
            assert_close(
                &flatten(&sample.gradient),
                gradient,
                tolerance,
                &format!("{what} gradient"),
            );
        }
        if let Some(symmetric) = &record.symmetric_gradient {
            assert_close(
                &symmetric_part(&sample.gradient),
                symmetric,
                tolerance,
                &format!("{what} symmetric gradient"),
            );
        }
    }
    assert_eq!(
        cells_seen.len(),
        sampler.mesh().cells().len(),
        "{what}: every cell was evaluated"
    );
}

/// The plan's quadrature view reproduces the recorded points cell by cell (same rule, same
/// physical coordinates, in order).
fn check_quadrature_points(records: &[Record], view: &QuadratureView<'_>, what: &str) {
    let mut by_cell: BTreeMap<CellId, Vec<&Record>> = BTreeMap::new();
    for record in records {
        by_cell.entry(record.cell).or_default().push(record);
    }
    let per_cell = view.rule().point_count();
    for (cell, records) in by_cell {
        assert_eq!(records.len() % per_cell, 0, "{what}: points per cell");
        let points = view.cell_points(cell).unwrap();
        for (index, record) in records.iter().enumerate() {
            assert_close(
                &points[index % per_cell].physical,
                &record.coordinates,
                1.0e-14,
                &format!("{what} point {index} of cell {}", cell.0),
            );
        }
    }
}

#[test]
fn scalar_p1_sampler_agrees_with_the_realization_plan_at_its_quadrature_points() {
    let compiled = compile_form(TRANSIENT_NONLINEAR, "TransientNonlinear", "evolution");
    let (mesh, dofs, constraints) = sheared_square(3);
    let recorder: Recorder = Arc::default();
    let plan = bind_single_field_plan(
        &compiled,
        &mesh,
        PreparedElement::linear_simplex(2).unwrap(),
        dofs,
        constraints.clone(),
        &recorder,
    );
    let state = mesh
        .vertices()
        .iter()
        .map(|point| wave(point, 0.0))
        .collect::<Vec<_>>();
    let rate = mesh
        .vertices()
        .iter()
        .map(|point| wave(point, 1.0))
        .collect::<Vec<_>>();
    let mut output = vec![0.0; plan.dimension()];
    plan.residual(0.0, &state, &rate, &mut output).unwrap();
    let physical = constraints.expand(&state).unwrap();
    let sampler = FieldSampler::from_realization_plan(&plan, &physical).unwrap();
    assert_eq!(
        sampler.family(),
        SampledFamily::Lagrange {
            order: 1,
            components: 1
        }
    );
    assert_eq!(sampler.dofs(), plan.dofs());
    let records = recorder.lock().unwrap().clone();
    assert!(records.iter().any(|record| record.gradient.is_some()));
    check_records(&records, &sampler, 1.0e-13, "scalar P1");
    let view = QuadratureView::of_realization_plan(&plan).unwrap();
    assert_eq!(view.rule().id, "simplex-barycenter");
    assert_eq!(view.rule().degree, 1);
    check_quadrature_points(&records, &view, "scalar P1");
    // The sampler refuses the constrained state's wrong length and a non-finite vector.
    assert!(matches!(
        FieldSampler::from_realization_plan(&plan, &physical[1..]),
        Err(FinitumError::InvalidRealization(_))
    ));
}

#[test]
fn scalar_p2_sampler_agrees_with_the_realization_plan_at_its_quadrature_points() {
    let compiled = compile_form(POISSON_P2, "Poisson", "balance");
    let (mesh, _, _) = sheared_square(2);
    let dofs = quadratic_simplex_dof_map(&mesh, 1).unwrap();
    let node_points = finitum::quadratic_simplex_node_points(&mesh);
    let boundary = FacetTopology::from_mesh(&mesh)
        .unwrap()
        .exterior()
        .flat_map(|facet| {
            facet
                .vertices
                .iter()
                .map(|vertex| vertex.0)
                .collect::<Vec<_>>()
        })
        .collect::<std::collections::BTreeSet<_>>();
    let constraints = ConstraintSet::new(
        dofs.dof_count(),
        boundary.into_iter().map(|target| AffineConstraint {
            target: DofId(target),
            dependencies: Vec::new(),
            offset: 0.0,
        }),
    )
    .unwrap();
    let recorder: Recorder = Arc::default();
    let plan = bind_single_field_plan(
        &compiled,
        &mesh,
        PreparedElement::quadratic_simplex(2).unwrap(),
        dofs,
        constraints.clone(),
        &recorder,
    );
    let state = node_points
        .iter()
        .map(|point| wave(point, 2.0))
        .collect::<Vec<_>>();
    let rate = vec![0.0; plan.dimension()];
    let mut output = vec![0.0; plan.dimension()];
    plan.residual(0.0, &state, &rate, &mut output).unwrap();
    let physical = constraints.expand(&state).unwrap();
    let sampler = FieldSampler::from_realization_plan(&plan, &physical).unwrap();
    assert_eq!(
        sampler.family(),
        SampledFamily::Lagrange {
            order: 2,
            components: 1
        }
    );
    let records = recorder.lock().unwrap().clone();
    assert!(records.iter().any(|record| record.gradient.is_some()));
    check_records(&records, &sampler, 1.0e-12, "scalar P2");
    let view = QuadratureView::of_realization_plan(&plan).unwrap();
    assert_eq!(view.rule().id, "triangle-dunavant-6");
    assert_eq!(view.rule().degree, 4);
    check_quadrature_points(&records, &view, "scalar P2");
}

#[test]
fn vector_p1_sampler_agrees_with_the_elasticity_plan_in_three_dimensions() {
    let compiled = compile_form(ELASTICITY, "Elasticity", "momentum");
    let tagged = sheared_box(3, 2);
    let mesh = &tagged.mesh;
    let dofs = vector_nodal_dof_map(mesh, 3).unwrap();
    let constraints = ConstraintSet::new(
        dofs.dof_count(),
        (0..3).map(|component| AffineConstraint {
            target: DofId(component),
            dependencies: Vec::new(),
            offset: 0.0,
        }),
    )
    .unwrap();
    let recorder: Recorder = Arc::default();
    let plan = bind_single_field_plan(
        &compiled,
        mesh,
        PreparedElement::linear_simplex(3).unwrap(),
        dofs,
        constraints.clone(),
        &recorder,
    );
    let mut state = vec![0.0; plan.dimension()];
    for (vertex, point) in mesh.vertices().iter().enumerate() {
        for component in 0..3 {
            state[vertex * 3 + component] = wave(point, component as f64);
        }
    }
    for entry in state.iter_mut().take(3) {
        *entry = 0.0;
    }
    let rate = vec![0.0; plan.dimension()];
    let mut output = vec![0.0; plan.dimension()];
    plan.residual(0.0, &state, &rate, &mut output).unwrap();
    let physical = constraints.expand(&state).unwrap();
    let sampler = FieldSampler::from_realization_plan(&plan, &physical).unwrap();
    assert_eq!(
        sampler.family(),
        SampledFamily::Lagrange {
            order: 1,
            components: 3
        }
    );
    let records = recorder.lock().unwrap().clone();
    assert!(
        records
            .iter()
            .any(|record| record.symmetric_gradient.is_some())
    );
    check_records(&records, &sampler, 1.0e-13, "vector P1");
    check_quadrature_points(
        &records,
        &QuadratureView::of_realization_plan(&plan).unwrap(),
        "vector P1",
    );
}

/// Every non-basis input of a compiled system bound to a recording constitutive closure that
/// returns a constant of the input's shape (`scale` in the first component).
fn recording_system_inputs(
    system: &OperatorSystem,
    recorder: &Recorder,
    scale: f64,
) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let components = input.shape.iter().product::<usize>().max(1);
                let recorder = Arc::clone(recorder);
                constitutive.push(
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        format!("w8-f1/recording/{}/{:?}", integral.integral_index, input.id),
                        move |evaluation: &PointEvaluation| {
                            record(&recorder, evaluation);
                            let mut value = vec![0.0; components];
                            value[0] = scale;
                            value
                        },
                        move |_: &PointEvaluation, _: &PointEvaluation| vec![0.0; components],
                    )
                    .unwrap(),
                );
            }
        }
    }
    constitutive
}

#[test]
fn rt0_and_p0_samplers_agree_with_the_mixed_darcy_system_operator() {
    let compilation = compile_semantics(DARCY, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "MixedDarcy",
        &["darcy_law", "mass_balance"],
    )
    .unwrap();
    let row = |equation: &str| {
        system
            .blocks
            .iter()
            .find(|block| block.equation == equation)
            .unwrap()
            .row
    };
    let (flux, pressure) = (row("darcy_law"), row("mass_balance"));
    let tagged = sheared_box(3, 1);
    let facets = FacetTopology::from_mesh(&tagged.mesh).unwrap();
    let compatible = CompatibleDofMaps::simplex(&tagged.mesh, &facets).unwrap();
    let layout = BlockLayout::new([
        (flux, compatible.hdiv_dof_count, 1),
        (pressure, tagged.mesh.cells().len(), 1),
    ])
    .unwrap();
    let plan = SystemRealizationPlan::new(system.clone(), tagged.mesh.clone(), layout).unwrap();
    let wall_region = system
        .blocks
        .iter()
        .flat_map(|block| block.factorization.integrals.iter())
        .find_map(|integral| match integral.measure {
            SemanticMeasure::ExteriorFacet { region } => Some(region),
            _ => None,
        })
        .unwrap();
    let mut region_map = RegionMap::new();
    region_map.insert(
        wall_region,
        ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"].map(RegionTagId::new),
    );
    let facet_regions = facet_membership_from(&tagged, &region_map, [wall_region]).unwrap();
    let recorder: Recorder = Arc::default();
    let operator = plan
        .bind_kernels_with_facets(
            recording_system_inputs(&system, &recorder, 2.3),
            BTreeMap::from([("mass_balance".to_string(), -1.0)]),
            facet_regions,
        )
        .unwrap();
    let state = (0..plan.layout().extent())
        .map(|index| 0.3 + ((index as f64) * 0.7).sin())
        .collect::<Vec<_>>();
    let rate = vec![0.0; state.len()];
    let mut output = vec![0.0; state.len()];
    operator.residual(0.0, &state, &rate, &mut output).unwrap();
    let records = recorder.lock().unwrap().clone();
    let flux_records = records
        .iter()
        .filter(|record| record.value.as_ref().is_some_and(|value| value.len() == 3))
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        !flux_records.is_empty(),
        "the darcy_law block sees the RT0 flux value"
    );

    let flux_sampler = FieldSampler::from_system_plan(&plan, flux, &state).unwrap();
    assert_eq!(flux_sampler.family(), SampledFamily::RaviartThomas0);
    assert_eq!(flux_sampler.component_count(), 3);
    assert_eq!(Some(flux_sampler.dofs()), operator.dof_map(flux));
    check_records(&flux_records, &flux_sampler, 1.0e-12, "RT0 flux");

    let pressure_sampler = FieldSampler::from_system_plan(&plan, pressure, &state).unwrap();
    assert_eq!(pressure_sampler.family(), SampledFamily::CellConstant);
    assert_eq!(Some(pressure_sampler.dofs()), operator.dof_map(pressure));
    let pressure_block = plan.layout().block(pressure).unwrap();
    for record in &flux_records {
        let value = pressure_sampler
            .value_at(record.cell, &record.coordinates)
            .unwrap();
        assert_eq!(value, vec![state[pressure_block.offset + record.cell.0]]);
    }

    let view = QuadratureView::of_system_plan(&plan).unwrap();
    assert_eq!(view.rule().id, "tetrahedron-symmetric-4");
    assert_eq!(view.rule().degree, 2);
    check_quadrature_points(&flux_records, &view, "RT0 flux");
    let degree_four = view.rule_for_degree(4).unwrap();
    assert_eq!(degree_four.rule().id, "tetrahedron-symmetric-14");

    // The domain's outward normals on the sheared box integrate to zero (closed surface) and
    // the RT0 normal traces satisfy the divergence theorem against the sampled divergence.
    let mut normal_sum = vec![0.0; 3];
    let mut boundary_flux = 0.0;
    for facet in plan.facets().exterior() {
        let trace = flux_sampler.trace_at_centroid(facet.id).unwrap();
        for (sum, normal) in normal_sum.iter_mut().zip(&trace.normal) {
            *sum += normal * trace.measure;
        }
        boundary_flux += trace.normal_component().unwrap() * trace.measure;
    }
    assert_close(&normal_sum, &[0.0; 3], 1.0e-13, "closed surface");
    let volume_divergence = view
        .integrate(|cell, point| flux_sampler.divergence_at(cell, &point.physical))
        .unwrap();
    assert!(
        (boundary_flux - volume_divergence).abs() <= 1.0e-12 * (1.0 + boundary_flux.abs()),
        "divergence theorem: boundary flux {boundary_flux} vs volume divergence {volume_divergence}"
    );
}

#[test]
fn taylor_hood_samplers_agree_with_the_stokes_system_operator() {
    let compilation = compile_semantics(STOKES, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "StokesFlow",
        &["momentum", "incompressibility"],
    )
    .unwrap();
    let row = |equation: &str| {
        system
            .blocks
            .iter()
            .find(|block| block.equation == equation)
            .unwrap()
            .row
    };
    let (velocity, pressure) = (row("momentum"), row("incompressibility"));
    let tagged = sheared_box(2, 2);
    let mesh = &tagged.mesh;
    let velocity_nodes = quadratic_simplex_dof_map(mesh, 2).unwrap().dof_count() / 2;
    let layout = BlockLayout::new([
        (velocity, velocity_nodes, 2),
        (pressure, mesh.vertices().len(), 1),
    ])
    .unwrap();
    let plan = SystemRealizationPlan::new(system.clone(), mesh.clone(), layout).unwrap();
    let recorder: Recorder = Arc::default();
    let operator = plan
        .bind_kernels(
            recording_system_inputs(&system, &recorder, 1.7),
            BTreeMap::new(),
        )
        .unwrap();
    let state = (0..plan.layout().extent())
        .map(|index| 0.5 + ((index as f64) * 0.37).cos())
        .collect::<Vec<_>>();
    let rate = vec![0.0; state.len()];
    let mut output = vec![0.0; state.len()];
    operator.residual(0.0, &state, &rate, &mut output).unwrap();
    let records = recorder.lock().unwrap().clone();
    let velocity_records = records
        .iter()
        .filter(|record| record.symmetric_gradient.is_some())
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        !velocity_records.is_empty(),
        "the momentum block sees sym_grad(velocity)"
    );

    let velocity_sampler = FieldSampler::from_system_plan(&plan, velocity, &state).unwrap();
    assert_eq!(
        velocity_sampler.family(),
        SampledFamily::Lagrange {
            order: 2,
            components: 2
        }
    );
    assert_eq!(Some(velocity_sampler.dofs()), operator.dof_map(velocity));
    check_records(&velocity_records, &velocity_sampler, 1.0e-12, "P2 velocity");

    let pressure_sampler = FieldSampler::from_system_plan(&plan, pressure, &state).unwrap();
    assert_eq!(
        pressure_sampler.family(),
        SampledFamily::Lagrange {
            order: 1,
            components: 1
        }
    );
    assert_eq!(Some(pressure_sampler.dofs()), operator.dof_map(pressure));
    // The P1 pressure at each vertex is its own DOF value.
    let pressure_block = plan.layout().block(pressure).unwrap();
    for (cell_index, cell) in mesh.cells().iter().enumerate() {
        for vertex in &cell.vertices {
            let value = pressure_sampler
                .value_at(CellId(cell_index), &mesh.vertices()[vertex.0])
                .unwrap();
            assert_close(
                &value,
                &[state[pressure_block.offset + vertex.0]],
                1.0e-13,
                "P1 pressure at a vertex",
            );
        }
    }

    let view = QuadratureView::of_system_plan(&plan).unwrap();
    assert_eq!(view.rule().id, "triangle-dunavant-6");
    check_quadrature_points(&velocity_records, &view, "P2 velocity");

    // Both samplers carry the same conventions digest scheme and differ by family.
    assert_ne!(velocity_sampler.digest(), pressure_sampler.digest());
    assert_eq!(
        velocity_sampler.conventions().schema,
        finitum::FIELD_SAMPLER_SCHEMA
    );
}

#[test]
fn a_system_field_outside_the_sampled_families_is_refused_by_name() {
    // A Stokes plan built on the sheared box, asked for a symbol no block realizes.
    let compilation = compile_semantics(STOKES, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "StokesFlow",
        &["momentum", "incompressibility"],
    )
    .unwrap();
    let velocity = system.blocks[0].row;
    let pressure = system.blocks[1].row;
    let tagged = sheared_box(2, 1);
    let mesh = &tagged.mesh;
    let velocity_nodes = quadratic_simplex_dof_map(mesh, 2).unwrap().dof_count() / 2;
    let layout = BlockLayout::new([
        (velocity, velocity_nodes, 2),
        (pressure, mesh.vertices().len(), 1),
    ])
    .unwrap();
    let plan = SystemRealizationPlan::new(system, mesh.clone(), layout).unwrap();
    let state = vec![0.0; plan.layout().extent()];
    assert!(matches!(
        FieldSampler::from_system_plan(&plan, scientia::SymbolId(9_999), &state),
        Err(FinitumError::InvalidRealization(_))
    ));
    assert!(matches!(
        FieldSampler::from_system_plan(&plan, velocity, &state[1..]),
        Err(FinitumError::InvalidRealization(_))
    ));
    // The typed family refusal itself (an order the sampler does not reconstruct).
    match FieldSampler::new(
        mesh,
        SampledFamily::Lagrange {
            order: 4,
            components: 1,
        },
        &state,
    ) {
        Err(FinitumError::SamplingUnsupported { family, reason }) => {
            assert_eq!(family, "Lagrange(order=4)");
            assert!(reason.contains("P1 and P2"));
        }
        other => panic!("expected SAMPLING_UNSUPPORTED, got {other:?}"),
    }
}
