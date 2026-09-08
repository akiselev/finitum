//! W8 lane F2 (PLAN §6 W8 decision 3, gate G3): fallible external-input and constitutive
//! callbacks. A callback that meets a typed refusal returns its own `InputEvaluationError`
//! (code, origin, message); Finitum locates it (cell, physical point, time) and propagates it
//! as `FinitumError::InputEvaluation` out of every action and as `NumericError::Evaluation`
//! out of every Methodus trait entry point -- never as a non-finite value.
//!
//! Evidence:
//! 1. a fallible P1 scalar input that refuses at one quadrature point of one cell yields the
//!    typed error with the right cell, point, origin and time from `residual`, JVP, VJP,
//!    `linearize`, `assemble`, `load_vector`, element assembly, the matrix-free Methodus
//!    action and `check_realization_agreement`; the first failure in cell / quadrature-point
//!    / input order wins, deterministically;
//! 2. the same on `SystemOperator` / `ReducedSystemOperator` for a fallible constitutive
//!    input, through the `DaeOperator` and `LinearOperator` impls and a Methodus BDF step;
//! 3. the infallible constructors are the fallible ones with `Ok`: same digest, bitwise the
//!    same actions (the unchanged pre-existing suite is the broader proof);
//! 4. a stored-table builder that refuses fails at construction with a `Table` origin, located
//!    at the cell, point and sampling time (the transient all-table system path);
//! 5. time-aware sampling: `g(t) = t` Dirichlet data sampled at `t = 0.5` give `0.5` on every
//!    constrained DOF on the system and the profile path, and `FieldSource::fallible` sources
//!    reach stored tables at the requested time and constitutive closures at the runtime time.

use std::collections::BTreeMap;
use std::sync::Arc;

use finitum::{
    AffineConstraint, BlockLayout, Cell, CellId, ConstraintSet, DofId, DofMap,
    DynamicExternalInput, ElementRestriction, ExternalInput, FieldSource, FinitumError,
    InputEvaluationError, InputLocation, InputOrigin, Mesh, MeshProfile, PointEvaluation,
    PreparedElement, QuadratureView, REALIZATION_PROPERTY_UNAVAILABLE, RealizationPlan,
    ReducedSystemOperator, RegionMap, RegionTagId, SysResId, SystemConstitutiveInput,
    SystemEssentialConstraintRequirement, SystemExternalInput, SystemOperator, SystemQuadrature,
    SystemRealizationPlan, TaggedMesh, VertexId, cell_centroid, check_realization_agreement,
    essential_constraints_from, essential_constraints_from_at, essential_constraints_from_system,
    essential_constraints_from_system_at, external_inputs_from, external_inputs_from_at, realize,
    system_constitutive_from_sources, vector_nodal_dof_map,
};
use methodus::{
    BdfConfig, BdfOrder, BdfState, ComparisonTolerance, DaeOperator, EvaluationContext,
    LinearOperator, NewtonConfig, NumericError, SolveError, bdf_step,
};
use quantitas::UnitRegistry;
use scientia::scientific::{Interpolation, OutOfValidityPolicy, TableAxis};
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, PropertyTable, SemanticModel,
    SymbolId, TableDerivativePolicy, TensorInputRole, compile_operator_system, compile_semantics,
    derive_variational_form, factor_operator, infer_form_requirements, lower_operator_kernels,
};

const TRANSIENT_NONLINEAR: &str = r#"
module w8_f2.transient_nonlinear;
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

const COUPLED: &str = r#"
module w8_f2.coupled;
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

const TOLERANCE: ComparisonTolerance = ComparisonTolerance {
    absolute: 1.0e-12,
    relative: 1.0e-12,
};
const POINT_TOLERANCE: f64 = 1.0e-12;

/// A refusal predicate over the evaluation Finitum hands the callback.
type Refuse = Arc<dyn Fn(&PointEvaluation) -> bool + Send + Sync>;

enum Binding {
    /// The infallible constructor.
    Infallible,
    /// The fallible constructor, refusing wherever the predicate says so.
    Fallible(Refuse),
}

fn never() -> Binding {
    Binding::Fallible(Arc::new(|_| false))
}

fn probe_vector(dimension: usize, seed: f64, scale: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| scale * ((index as f64 + seed) * 0.7).sin())
        .collect()
}

fn close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| (a - b).abs() <= POINT_TOLERANCE)
}

/// What Finitum must report: the callback's code/origin/message and Finitum's own location.
struct Expected {
    code: &'static str,
    origin: InputOrigin,
    message: &'static str,
    cell: CellId,
    point: Vec<f64>,
    time: f64,
}

/// The failure as a callback raises it -- with deliberately wrong location fields, which
/// Finitum must overwrite with where it actually evaluated.
fn raised(expected: &Expected) -> InputEvaluationError {
    let mut failure =
        InputEvaluationError::new(expected.code, expected.origin.clone(), expected.message);
    failure.location = Some(Box::new(InputLocation {
        cell: Some(CellId(77)),
        point: vec![99.0],
        time: Some(-1.0),
    }));
    failure
}

fn assert_located(error: &FinitumError, expected: &Expected, what: &str) {
    let FinitumError::InputEvaluation(failure) = error else {
        panic!("{what}: expected a typed input failure, got {error}");
    };
    assert_eq!(failure.code, expected.code, "{what}: code");
    assert_eq!(failure.origin, expected.origin, "{what}: origin");
    assert_eq!(failure.message, expected.message, "{what}: message");
    assert_eq!(failure.cell(), Some(expected.cell), "{what}: cell");
    let point = failure.point().expect("located point");
    assert!(
        close(point, &expected.point),
        "{what}: point {point:?}, expected {:?}",
        expected.point
    );
    assert_eq!(failure.time(), Some(expected.time), "{what}: time");
    assert_eq!(
        error.code(),
        Some(expected.code),
        "{what}: FinitumError::code"
    );
    let display = error.to_string();
    assert!(
        display.starts_with(&format!(
            "{} at {}, point (",
            expected.code, expected.origin
        )),
        "{what}: display {display}"
    );
    assert!(
        display.ends_with(&format!(
            ", t = {}, cell {}: {}",
            expected.time, expected.cell.0, expected.message
        )),
        "{what}: display {display}"
    );
}

fn assert_numeric(error: &NumericError, expected: &Expected, what: &str) {
    assert_eq!(
        error.evaluation_code(),
        Some(expected.code),
        "{what}: {error}"
    );
    let NumericError::Evaluation {
        code,
        origin,
        message,
    } = error
    else {
        panic!("{what}: expected NumericError::Evaluation, got {error}");
    };
    assert_eq!(code, expected.code);
    assert_eq!(origin, &expected.origin.to_string());
    assert!(
        message.starts_with("point (")
            && message.ends_with(&format!(
                ", t = {}, cell {}: {}",
                expected.time, expected.cell.0, expected.message
            )),
        "{what}: message {message}"
    );
}

// ---------------------------------------------------------------------------------------------
// P1 single-model fixture (the GX-C3 transient nonlinear plan, `k` bound by hand).
// ---------------------------------------------------------------------------------------------

fn square_discretization(subdivisions: usize) -> (Mesh, DofMap, ConstraintSet) {
    let width = subdivisions + 1;
    let vertices = (0..=subdivisions)
        .flat_map(|row| {
            (0..=subdivisions).map(move |column| {
                vec![
                    column as f64 / subdivisions as f64,
                    row as f64 / subdivisions as f64,
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

fn diffusivity_expectation(cell: CellId, point: Vec<f64>, time: f64) -> Expected {
    Expected {
        code: "RUN_TANGENT_UNAVAILABLE",
        origin: InputOrigin::Provider("diffusivity".into()),
        message: "no tangent for an analytic_provided law",
        cell,
        point,
        time,
    }
}

fn k_value(point: &PointEvaluation) -> Vec<f64> {
    vec![1.0 + 0.2 * point.values(DerivativeEvaluation::Value).unwrap()[0]]
}

fn k_direction(_: &PointEvaluation, direction: &PointEvaluation) -> Vec<f64> {
    vec![0.2 * direction.values(DerivativeEvaluation::Value).unwrap()[0]]
}

fn diffusivity(
    integral_index: usize,
    input: scientia::TensorInputId,
    binding: &Binding,
) -> DynamicExternalInput {
    const IDENTITY: &str = "k=1+0.2u;direction=0.2du/v1";
    match binding {
        Binding::Infallible => {
            DynamicExternalInput::new(integral_index, input, 1, IDENTITY, k_value, k_direction)
                .unwrap()
        }
        Binding::Fallible(refuse) => {
            let value_refuse = refuse.clone();
            let direction_refuse = refuse.clone();
            let failure = || raised(&diffusivity_expectation(CellId(0), Vec::new(), 0.0));
            DynamicExternalInput::try_new(
                integral_index,
                input,
                1,
                IDENTITY,
                move |point| {
                    if value_refuse(point) {
                        Err(failure())
                    } else {
                        Ok(k_value(point))
                    }
                },
                move |point, direction| {
                    if direction_refuse(point) {
                        Err(failure())
                    } else {
                        Ok(k_direction(point, direction))
                    }
                },
            )
            .unwrap()
        }
    }
}

/// The transient nonlinear P1 plan on the 2x2 square (8 cells) with the degree-`degree` rule;
/// `capacity` and `f` are fixed, `k` is bound per `binding`. `capacity_binding` lets one test
/// make the capacity refuse too.
fn p1_plan(degree: u16, binding: &Binding, capacity_refuse: Option<Refuse>) -> RealizationPlan {
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, constraints) = square_discretization(2);
    let element = PreparedElement::linear_simplex_with_degree(2, degree).unwrap();
    let model = &compilation.semantic.models[0];
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.as_str();
            match name {
                "capacity" => {
                    let refuse = capacity_refuse.clone();
                    dynamic.push(
                        DynamicExternalInput::try_new(
                            integral.integral_index,
                            input.id,
                            1,
                            "capacity=1;direction=0/v1",
                            move |point| match &refuse {
                                Some(refuse) if refuse(point) => Err(InputEvaluationError::new(
                                    "RUN_PROPERTY_UNSUPPORTED",
                                    InputOrigin::ExpressionPath(
                                        "TransientNonlinear.evolution.capacity".into(),
                                    ),
                                    "capacity refused",
                                )),
                                _ => Ok(vec![1.0]),
                            },
                            |_, _| Ok(vec![0.0]),
                        )
                        .unwrap(),
                    );
                }
                "k" => dynamic.push(diffusivity(integral.integral_index, input.id, binding)),
                "f" => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        |_, _| vec![0.0],
                    )
                    .unwrap(),
                ),
                other => panic!("unexpected external input {other}"),
            }
        }
    }
    RealizationPlan::new_stateful(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        stored,
        dynamic,
    )
    .unwrap()
}

/// The edge midpoints of `cell` in the degree-2 rule's own order: `(v0, v1)`, `(v1, v2)`,
/// `(v0, v2)`.
fn edge_midpoints(mesh: &Mesh, cell: CellId) -> [Vec<f64>; 3] {
    let vertices = &mesh.cells()[cell.0].vertices;
    let midpoint = |left: usize, right: usize| {
        let a = &mesh.vertices()[vertices[left].0];
        let b = &mesh.vertices()[vertices[right].0];
        vec![0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
    };
    [midpoint(0, 1), midpoint(1, 2), midpoint(0, 2)]
}

#[test]
fn a_p1_input_refusing_at_one_quadrature_point_is_located_and_carried_by_every_action() {
    let (mesh, _, _) = square_discretization(2);
    // Refuse at the second and third quadrature points of cell 3 and everywhere on cell 5; the
    // first failure in cell-then-point order is cell 3's second point.
    let [_, second, third] = edge_midpoints(&mesh, CellId(3));
    let refuse_points = [second.clone(), third];
    let refuse: Refuse = Arc::new(move |point: &PointEvaluation| {
        point.cell == CellId(5)
            || (point.cell == CellId(3)
                && refuse_points
                    .iter()
                    .any(|candidate| close(candidate, &point.coordinates)))
    });
    let plan = p1_plan(2, &Binding::Fallible(refuse), None);
    let dimension = plan.dimension();
    let context = EvaluationContext::reproducible();
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let direction = probe_vector(dimension, 1.1, 1.0);
    let rate_direction = probe_vector(dimension, 3.3, 0.8);
    let mut output = vec![0.0; dimension];

    // The located point is a quadrature point of the plan's own rule.
    let recorded = QuadratureView::of_realization_plan(&plan)
        .unwrap()
        .cell_points(CellId(3))
        .unwrap();
    assert!(close(&recorded[1].physical, &second));

    let at = |time: f64| diffusivity_expectation(CellId(3), second.clone(), time);

    let error = plan.residual(0.3, &state, &rate, &mut output).unwrap_err();
    assert_located(&error, &at(0.3), "residual");
    let again = plan.residual(0.3, &state, &rate, &mut output).unwrap_err();
    assert_eq!(again, error, "the first failure is deterministic");

    let error = plan
        .jacobian_vector_product(0.7, &state, &rate, &direction, &rate_direction, &mut output)
        .unwrap_err();
    assert_located(&error, &at(0.7), "jacobian_vector_product");

    let error = plan
        .vector_jacobian_product(0.9, &state, &rate, &direction, &mut output)
        .unwrap_err();
    assert_located(&error, &at(0.9), "vector_jacobian_product");

    // `linearize` is lazy: the failure surfaces on the first Methodus application.
    let linearized = plan.linearize(0.3, &state, &rate, 1.5).unwrap();
    let error = linearized
        .apply(&context, &direction, &mut output)
        .unwrap_err();
    assert_numeric(&error, &at(0.3), "linearized apply");

    // Zero-state actions evaluate at `t = 0`.
    assert_located(&plan.assemble().unwrap_err(), &at(0.0), "assemble");
    assert_located(&plan.load_vector().unwrap_err(), &at(0.0), "load_vector");
    // Element assembly is eager: it fails at construction.
    assert_located(
        &plan.element_assembly(4).unwrap_err(),
        &at(0.0),
        "element_assembly",
    );
    let error = plan
        .matrix_free()
        .apply(&context, &direction, &mut output)
        .unwrap_err();
    assert_numeric(&error, &at(0.0), "matrix-free Methodus action");
    let error = check_realization_agreement(&plan, &direction, 4, TOLERANCE).unwrap_err();
    assert_located(&error, &at(0.0), "check_realization_agreement");
}

#[test]
fn among_inputs_refusing_at_the_same_point_the_first_in_declaration_order_wins() {
    let (mesh, _, _) = square_discretization(2);
    let [first, _, _] = edge_midpoints(&mesh, CellId(2));
    let target = first.clone();
    let refuse: Refuse = Arc::new(move |point: &PointEvaluation| {
        point.cell == CellId(2) && close(&target, &point.coordinates)
    });
    let plan = p1_plan(2, &Binding::Fallible(refuse.clone()), Some(refuse));
    let dimension = plan.dimension();
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let mut output = vec![0.0; dimension];
    let error = plan.residual(0.1, &state, &rate, &mut output).unwrap_err();
    // Which of `capacity` / `k` Finitum reaches first is the factorization's own order:
    // integrals in order, non-basis inputs in declaration order within each.
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &compilation.semantic.models[0];
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let first_declared = factorization
        .integrals
        .iter()
        .flat_map(|integral| integral.primal.inputs.iter())
        .filter(|input| input.source != InputSourceRequirement::Basis)
        .map(|input| model.symbols[input.binding.symbol.index()].name.as_str())
        .find(|name| *name == "capacity" || *name == "k")
        .unwrap();
    let expected = if first_declared == "capacity" {
        Expected {
            code: "RUN_PROPERTY_UNSUPPORTED",
            origin: InputOrigin::ExpressionPath("TransientNonlinear.evolution.capacity".into()),
            message: "capacity refused",
            cell: CellId(2),
            point: first,
            time: 0.1,
        }
    } else {
        diffusivity_expectation(CellId(2), first, 0.1)
    };
    assert_located(&error, &expected, "residual with two refusing inputs");
    let again = plan.residual(0.1, &state, &rate, &mut output).unwrap_err();
    assert_eq!(again, error);
}

#[test]
fn the_infallible_p1_constructor_is_the_fallible_one_with_ok_bitwise_and_digest_equal() {
    let infallible = p1_plan(1, &Binding::Infallible, None);
    let fallible = p1_plan(1, &never(), None);
    assert_eq!(infallible.digest(), fallible.digest());
    let dimension = infallible.dimension();
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let direction = probe_vector(dimension, 1.1, 1.0);
    let rate_direction = probe_vector(dimension, 3.3, 0.8);
    let mut left = vec![0.0; dimension];
    let mut right = vec![0.0; dimension];
    infallible.residual(0.3, &state, &rate, &mut left).unwrap();
    fallible.residual(0.3, &state, &rate, &mut right).unwrap();
    assert!(left.iter().any(|value| *value != 0.0));
    assert_eq!(bits(&left), bits(&right), "residual");
    infallible
        .jacobian_vector_product(0.3, &state, &rate, &direction, &rate_direction, &mut left)
        .unwrap();
    fallible
        .jacobian_vector_product(0.3, &state, &rate, &direction, &rate_direction, &mut right)
        .unwrap();
    assert_eq!(bits(&left), bits(&right), "jvp");
    infallible
        .vector_jacobian_product(0.3, &state, &rate, &direction, &mut left)
        .unwrap();
    fallible
        .vector_jacobian_product(0.3, &state, &rate, &direction, &mut right)
        .unwrap();
    assert_eq!(bits(&left), bits(&right), "vjp");
    let context = EvaluationContext::reproducible();
    infallible
        .assemble()
        .unwrap()
        .apply(&context, &direction, &mut left)
        .unwrap();
    fallible
        .assemble()
        .unwrap()
        .apply(&context, &direction, &mut right)
        .unwrap();
    assert_eq!(bits(&left), bits(&right), "assembled action");
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

// ---------------------------------------------------------------------------------------------
// System-path fixture (the Batch P coupled transient system, closures by hand).
// ---------------------------------------------------------------------------------------------

struct Compiled {
    model: SemanticModel,
    system: OperatorSystem,
}

fn compile_coupled() -> Compiled {
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

fn unit_square(subdivisions: usize) -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

fn ka_expectation(cell: CellId, point: Vec<f64>, time: f64) -> Expected {
    Expected {
        code: "RUN_CONSTITUTIVE_LAW_UNSUPPORTED",
        origin: InputOrigin::ExpressionPath("Coupled.ea[0].ka".into()),
        message: "diffusivity_a has no realization at this state",
        cell,
        point,
        time,
    }
}

/// Every non-basis input of the coupled system as a closure; `ka` bound per `binding`.
fn coupled_constitutive(compiled: &Compiled, binding: &Binding) -> Vec<SystemConstitutiveInput> {
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
                        let ka = move |point: &PointEvaluation| {
                            let b = point.input_values(b).unwrap()[0];
                            vec![1.0 + 0.3 * b * b]
                        };
                        let d_ka = move |point: &PointEvaluation, direction: &PointEvaluation| {
                            vec![
                                0.6 * point.input_values(b).unwrap()[0]
                                    * direction.input_values(b).unwrap()[0],
                            ]
                        };
                        match binding {
                            Binding::Infallible => SystemConstitutiveInput::new(
                                equation,
                                index,
                                input.id,
                                1,
                                "w8_f2/ka=1+0.3b^2",
                                ka,
                                d_ka,
                            ),
                            Binding::Fallible(refuse) => {
                                let value_refuse = refuse.clone();
                                let direction_refuse = refuse.clone();
                                let failure =
                                    || raised(&ka_expectation(CellId(0), Vec::new(), 0.0));
                                SystemConstitutiveInput::try_new(
                                    equation,
                                    index,
                                    input.id,
                                    1,
                                    "w8_f2/ka=1+0.3b^2",
                                    move |point| {
                                        if value_refuse(point) {
                                            Err(failure())
                                        } else {
                                            Ok(ka(point))
                                        }
                                    },
                                    move |point, direction| {
                                        if direction_refuse(point) {
                                            Err(failure())
                                        } else {
                                            Ok(d_ka(point, direction))
                                        }
                                    },
                                )
                            }
                        }
                    }
                    "kb" => {
                        let a = value_input("a");
                        SystemConstitutiveInput::new(
                            equation,
                            index,
                            input.id,
                            1,
                            "w8_f2/kb=1+0.5a",
                            move |point: &PointEvaluation| {
                                vec![1.0 + 0.5 * point.input_values(a).unwrap()[0]]
                            },
                            move |_: &PointEvaluation, direction: &PointEvaluation| {
                                vec![0.5 * direction.input_values(a).unwrap()[0]]
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
                            "w8_f2/ca=1+0.2a^2",
                            move |point: &PointEvaluation| {
                                let a = point.input_values(a).unwrap()[0];
                                vec![1.0 + 0.2 * a * a]
                            },
                            move |point: &PointEvaluation, direction: &PointEvaluation| {
                                vec![
                                    0.4 * point.input_values(a).unwrap()[0]
                                        * direction.input_values(a).unwrap()[0],
                                ]
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
                            format!("w8_f2/{name}={value}"),
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

fn coupled_operators(
    compiled: &Compiled,
    tagged: &TaggedMesh,
    constitutive: Vec<SystemConstitutiveInput>,
) -> (SystemOperator, ReducedSystemOperator) {
    let model = &compiled.model;
    let a = symbol(model, "a");
    let b = symbol(model, "b");
    let vertex_count = tagged.mesh.vertices().len();
    let layout = BlockLayout::new([(a, vertex_count, 1), (b, vertex_count, 1)]).unwrap();
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), tagged.mesh.clone(), layout).unwrap();
    let operator = plan.bind_kernels(constitutive, BTreeMap::new()).unwrap();
    let mut requirements = Vec::new();
    let mut region_map = RegionMap::new();
    for block in &compiled.system.blocks {
        for requirement in &block.factorization.essential_constraints {
            region_map.insert(
                requirement.region,
                ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
            );
            requirements.push(SystemEssentialConstraintRequirement {
                field: block.row,
                requirement: requirement.clone(),
                value: FieldSource::constant([0.0]),
            });
        }
    }
    let constraints =
        essential_constraints_from_system(&operator, tagged, &region_map, &requirements).unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    (operator, reduced)
}

#[test]
fn a_system_constitutive_refusal_is_located_and_carried_through_the_reduced_dae_operator() {
    let compiled = compile_coupled();
    let tagged = unit_square(3);
    // Refuse on cells 4 and 2: the first failure is cell 2's first quadrature point.
    let refuse: Refuse =
        Arc::new(|point: &PointEvaluation| point.cell == CellId(4) || point.cell == CellId(2));
    let (operator, reduced) = coupled_operators(
        &compiled,
        &tagged,
        coupled_constitutive(&compiled, &Binding::Fallible(refuse)),
    );
    let first_point = QuadratureView::of_system_plan(operator.plan())
        .unwrap()
        .cell_points(CellId(2))
        .unwrap()[0]
        .physical
        .clone();
    let at = |time: f64| ka_expectation(CellId(2), first_point.clone(), time);
    let dimension = operator.dimension();
    let context = EvaluationContext::reproducible();
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let direction = probe_vector(dimension, 1.1, 1.0);
    let rate_direction = probe_vector(dimension, 3.3, 0.8);
    let mut output = vec![0.0; dimension];

    let error = operator
        .residual(0.2, &state, &rate, &mut output)
        .unwrap_err();
    assert_located(&error, &at(0.2), "system residual");
    let again = operator
        .residual(0.2, &state, &rate, &mut output)
        .unwrap_err();
    assert_eq!(again, error, "the first failure is deterministic");
    let error = operator
        .jacobian_vector_product(0.4, &state, &rate, &direction, &rate_direction, &mut output)
        .unwrap_err();
    assert_located(&error, &at(0.4), "system jacobian_vector_product");
    let error = operator
        .vector_jacobian_product(0.6, &state, &rate, &direction, &mut output)
        .unwrap_err();
    assert_located(&error, &at(0.6), "system vector_jacobian_product");
    let linearized = operator.linearize(0.2, &state, &rate, 2.5).unwrap();
    let error = linearized
        .apply(&context, &direction, &mut output)
        .unwrap_err();
    assert_numeric(&error, &at(0.2), "system linearized apply");
    assert_located(
        &operator.assemble().unwrap_err(),
        &at(0.0),
        "system assemble",
    );
    assert_located(
        &operator.load_vector().unwrap_err(),
        &at(0.0),
        "system load_vector",
    );
    assert_located(
        &operator.element_assembly(4).unwrap_err(),
        &at(0.0),
        "system element_assembly",
    );
    let error = LinearOperator::apply(&operator, &context, &direction, &mut output).unwrap_err();
    assert_numeric(&error, &at(0.0), "system LinearOperator::apply");

    // The essential-constraint-eliminated operator: Finitum's own action, the Methodus
    // `DaeOperator` / `LinearOperator` impls, and a whole Methodus BDF step.
    let error = reduced
        .residual(0.2, &state, &rate, &mut output)
        .unwrap_err();
    assert_located(&error, &at(0.2), "reduced residual");
    let error =
        DaeOperator::residual(&reduced, &context, 0.2, &state, &rate, &mut output).unwrap_err();
    assert_numeric(&error, &at(0.2), "reduced DaeOperator::residual");
    let error = DaeOperator::jacobian_vector_product(
        &reduced,
        &context,
        0.2,
        &state,
        &rate,
        &direction,
        &rate_direction,
        &mut output,
    )
    .unwrap_err();
    assert_numeric(
        &error,
        &at(0.2),
        "reduced DaeOperator::jacobian_vector_product",
    );
    let error = LinearOperator::apply(&reduced, &context, &direction, &mut output).unwrap_err();
    assert_numeric(&error, &at(0.0), "reduced LinearOperator::apply");
    let bdf = BdfState {
        time: 0.0,
        values: vec![0.0; dimension],
        previous_values: None,
        previous_step: None,
        accepted_steps: 0,
    };
    let config = BdfConfig {
        order: BdfOrder::One,
        absolute_tolerance: 1.0e-3,
        relative_tolerance: 1.0e-3,
        minimum_step: 1.0e-8,
        maximum_step: 1.0,
        newton: NewtonConfig::default(),
    };
    let error = bdf_step(&reduced, &context, &bdf, 0.05, &config).unwrap_err();
    let SolveError::Numeric(numeric) = &error else {
        panic!("a BDF step over a refusing callback must fail numerically, got {error}");
    };
    assert_eq!(
        numeric.evaluation_code(),
        Some("RUN_CONSTITUTIVE_LAW_UNSUPPORTED")
    );
    let NumericError::Evaluation { origin, .. } = numeric else {
        panic!("expected NumericError::Evaluation, got {numeric}");
    };
    assert_eq!(origin, "Coupled.ea[0].ka");
}

#[test]
fn the_infallible_system_constructor_is_the_fallible_one_with_ok_bitwise_and_digest_equal() {
    let compiled = compile_coupled();
    let tagged = unit_square(3);
    let (infallible, _) = coupled_operators(
        &compiled,
        &tagged,
        coupled_constitutive(&compiled, &Binding::Infallible),
    );
    let (fallible, _) = coupled_operators(
        &compiled,
        &tagged,
        coupled_constitutive(&compiled, &never()),
    );
    assert_eq!(infallible.digest(), fallible.digest());
    let dimension = infallible.dimension();
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let direction = probe_vector(dimension, 1.1, 1.0);
    let rate_direction = probe_vector(dimension, 3.3, 0.8);
    let mut left = vec![0.0; dimension];
    let mut right = vec![0.0; dimension];
    infallible.residual(0.2, &state, &rate, &mut left).unwrap();
    fallible.residual(0.2, &state, &rate, &mut right).unwrap();
    assert!(left.iter().any(|value| *value != 0.0));
    assert_eq!(bits(&left), bits(&right), "system residual");
    infallible
        .jacobian_vector_product(0.2, &state, &rate, &direction, &rate_direction, &mut left)
        .unwrap();
    fallible
        .jacobian_vector_product(0.2, &state, &rate, &direction, &rate_direction, &mut right)
        .unwrap();
    assert_eq!(bits(&left), bits(&right), "system jvp");
    infallible
        .vector_jacobian_product(0.2, &state, &rate, &direction, &mut left)
        .unwrap();
    fallible
        .vector_jacobian_product(0.2, &state, &rate, &direction, &mut right)
        .unwrap();
    assert_eq!(bits(&left), bits(&right), "system vjp");
}

// ---------------------------------------------------------------------------------------------
// Stored tables and time-aware sampling.
// ---------------------------------------------------------------------------------------------

fn walls_region_map(region: scientia::RegionId) -> RegionMap {
    let mut map = RegionMap::new();
    map.insert(
        region,
        ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
    );
    map
}

#[test]
fn a_failing_table_builder_refuses_at_construction_with_a_table_origin_on_the_all_table_transient_path()
 {
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let model = compilation.semantic.models[0].clone();
    let system =
        compile_operator_system(&compilation.semantic, "TransientNonlinear", &["evolution"])
            .unwrap();
    let tagged = unit_square(3);
    let mesh = tagged.mesh.clone();
    let u = symbol(&model, "u");
    let layout = BlockLayout::new([(u, mesh.vertices().len(), 1)]).unwrap();
    let plan = SystemRealizationPlan::with_quadrature(
        system.clone(),
        mesh.clone(),
        layout,
        SystemQuadrature::Barycenter,
    )
    .unwrap();
    // The single-model P1 element tabulates the same barycenter rule (W7 7c A), so it is the
    // table these system-path inputs are sampled on.
    let element = PreparedElement::linear_simplex(2).unwrap();
    assert_eq!(element.quadrature(), plan.quadrature().unwrap().as_slice());
    let block = &system.blocks[0];
    let centroid = cell_centroid(&mesh, CellId(2)).unwrap();

    let mut stored = Vec::new();
    let mut refused = None;
    let mut source_input = None;
    for integral in &block.factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.clone();
            let path = format!(
                "TransientNonlinear.evolution[{}].{name}",
                integral.integral_index
            );
            if name == "f" {
                source_input = Some((integral.integral_index, input.id, path.clone()));
            }
            let table = ExternalInput::try_sampled_at(
                integral.integral_index,
                input.id,
                1,
                &mesh,
                &element,
                0.5,
                |cell, _, time| {
                    if name == "f" && cell == CellId(2) {
                        Err(InputEvaluationError::new(
                            "RUN_PROPERTY_UNSUPPORTED",
                            InputOrigin::ExpressionPath(path.clone()),
                            "the source has no value here",
                        ))
                    } else if name == "f" {
                        Ok(vec![time])
                    } else {
                        Ok(vec![1.0])
                    }
                },
            );
            match table {
                Ok(table) => stored.push(SystemExternalInput {
                    residual: SysResId(0),
                    input: table,
                }),
                Err(error) => {
                    assert_eq!(name, "f");
                    refused = Some(error);
                }
            }
        }
    }
    let error = refused.expect("the f table builder refuses");
    let (f_integral, f_input, f_path) = source_input.unwrap();
    let FinitumError::InputEvaluation(failure) = &error else {
        panic!("expected a typed input failure, got {error}");
    };
    assert_eq!(failure.code, "RUN_PROPERTY_UNSUPPORTED");
    assert_eq!(failure.origin, InputOrigin::Table(f_path.clone()));
    assert_eq!(failure.cell(), Some(CellId(2)));
    let point = failure.point().unwrap();
    assert!(close(point, &centroid), "{point:?} vs {centroid:?}");
    assert_eq!(failure.time(), Some(0.5));
    assert_eq!(error.code(), Some("RUN_PROPERTY_UNSUPPORTED"));
    assert_eq!(
        error.to_string(),
        format!(
            "RUN_PROPERTY_UNSUPPORTED at {f_path} (stored table), point ({}, {}), t = 0.5, \
             cell 2: the source has no value here",
            point[0], point[1]
        )
    );

    // Without the refusal the same builder samples the source at t = 0.5 everywhere, and the
    // all-table operator binds and loads it.
    let f = ExternalInput::try_sampled_at(
        f_integral,
        f_input,
        1,
        &mesh,
        &element,
        0.5,
        |_, _, time| Ok(vec![time]),
    )
    .unwrap();
    assert_eq!(f.values().len(), mesh.cells().len());
    assert!(f.values().iter().all(|value| *value == 0.5));
    stored.push(SystemExternalInput {
        residual: SysResId(0),
        input: f,
    });
    let operator = plan
        .bind_kernels_with_inputs(Vec::new(), stored, BTreeMap::new(), BTreeMap::new())
        .unwrap();
    let load = operator.load_vector().unwrap();
    assert!(load.iter().any(|value| *value != 0.0));

    // The steady form carries no time; the origin is still re-labelled as the table's.
    let steady = ExternalInput::try_sampled(f_integral, f_input, 1, &mesh, &element, |cell, _| {
        if cell == CellId(2) {
            Err(InputEvaluationError::new(
                "RUN_PROPERTY_UNSUPPORTED",
                InputOrigin::Slot("source/f".into()),
                "no value",
            ))
        } else {
            Ok(vec![0.0])
        }
    })
    .unwrap_err();
    let FinitumError::InputEvaluation(failure) = &steady else {
        panic!("expected a typed input failure, got {steady}");
    };
    assert_eq!(failure.origin, InputOrigin::Table("source/f".into()));
    assert_eq!(failure.cell(), Some(CellId(2)));
    assert_eq!(failure.time(), None);
    assert!(
        steady
            .to_string()
            .starts_with("RUN_PROPERTY_UNSUPPORTED at source/f (stored table), point (")
    );
    assert!(steady.to_string().ends_with("), cell 2: no value"));
}

#[test]
fn transient_dirichlet_data_g_of_t_is_sampled_at_the_requested_time_on_both_paths() {
    // System path: the coupled transient fixture with `g(t) = t` on every wall.
    let compiled = compile_coupled();
    let tagged = unit_square(3);
    let (operator, _) = coupled_operators(
        &compiled,
        &tagged,
        coupled_constitutive(&compiled, &Binding::Infallible),
    );
    let mut requirements = Vec::new();
    let mut region_map = RegionMap::new();
    for block in &compiled.system.blocks {
        for requirement in &block.factorization.essential_constraints {
            region_map.insert(
                requirement.region,
                ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
            );
            requirements.push(SystemEssentialConstraintRequirement {
                field: block.row,
                requirement: requirement.clone(),
                value: FieldSource::fallible(|_, time| Ok(vec![time])),
            });
        }
    }
    let at_half =
        essential_constraints_from_system_at(&operator, &tagged, &region_map, &requirements, 0.5)
            .unwrap();
    let mut constrained = 0;
    for constraint in at_half.constraints() {
        assert_eq!(constraint.offset, 0.5);
        assert!(constraint.dependencies.is_empty());
        constrained += 1;
    }
    assert!(constrained > 0);
    let legacy =
        essential_constraints_from_system(&operator, &tagged, &region_map, &requirements).unwrap();
    assert_eq!(legacy.constraints().count(), constrained);
    for constraint in legacy.constraints() {
        assert_eq!(constraint.offset, 0.0);
    }

    // A refusing datum is located at the node (no cell) at the requested time, its origin
    // untouched (a datum is not a stored table).
    requirements[0].value = FieldSource::fallible(|point, _| {
        if point[0] == 0.0 {
            Err(InputEvaluationError::new(
                "RUN_PROPERTY_UNSUPPORTED",
                InputOrigin::Slot("boundary/walls_a".into()),
                "no data on x = 0",
            ))
        } else {
            Ok(vec![0.0])
        }
    });
    let error =
        essential_constraints_from_system_at(&operator, &tagged, &region_map, &requirements, 0.5)
            .unwrap_err();
    let FinitumError::InputEvaluation(failure) = &error else {
        panic!("expected a typed input failure, got {error}");
    };
    assert_eq!(failure.code, "RUN_PROPERTY_UNSUPPORTED");
    assert_eq!(failure.origin, InputOrigin::Slot("boundary/walls_a".into()));
    assert_eq!(failure.cell(), None);
    assert_eq!(failure.point().unwrap()[0], 0.0);
    assert_eq!(failure.time(), Some(0.5));
    let display = error.to_string();
    assert!(
        display.starts_with("RUN_PROPERTY_UNSUPPORTED at boundary/walls_a, point (0, "),
        "{display}"
    );
    assert!(
        display.ends_with("), t = 0.5: no data on x = 0"),
        "{display}"
    );

    // Profile path: the single-model transient plan's `walls` requirement.
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let form_requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &form_requirements).unwrap();
    let dofs = vector_nodal_dof_map(&tagged.mesh, 1).unwrap();
    let region_map = walls_region_map(factorization.essential_constraints[0].region);
    let values = [FieldSource::fallible(|_, time| Ok(vec![time]))];
    let at_half = essential_constraints_from_at(
        &tagged,
        &dofs,
        &factorization.essential_constraints,
        &region_map,
        &values,
        0.5,
    )
    .unwrap();
    assert!(at_half.constraints().count() > 0);
    for constraint in at_half.constraints() {
        assert_eq!(constraint.offset, 0.5);
    }
    let legacy = essential_constraints_from(
        &tagged,
        &dofs,
        &factorization.essential_constraints,
        &region_map,
        &values,
    )
    .unwrap();
    assert_eq!(legacy.constraints().count(), at_half.constraints().count());
    for constraint in legacy.constraints() {
        assert_eq!(constraint.offset, 0.0);
    }
}

#[test]
fn fallible_field_sources_feed_time_sampled_tables_and_runtime_time_constitutive_inputs() {
    // Single-model path: `external_inputs_from_at` samples a Fallible source into a stored
    // table at the given time.
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &compilation.semantic.models[0];
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let (mesh, _, _) = square_discretization(2);
    let element = PreparedElement::linear_simplex(2).unwrap();
    let constant = |value: f64| FieldSource::fallible(move |_, _| Ok(vec![value]));
    let f = symbol(model, "f");
    let sources = vec![
        (symbol(model, "capacity"), constant(1.0)),
        (symbol(model, "k"), constant(1.0)),
        (f, FieldSource::fallible(|_, time| Ok(vec![time]))),
    ];
    let (stored, dynamic) =
        external_inputs_from_at(&factorization, model, &mesh, &element, &sources, 0.5).unwrap();
    assert!(dynamic.is_empty());
    let is_f = |table: &ExternalInput| {
        factorization.integrals.iter().any(|integral| {
            integral.integral_index == table.integral_index
                && integral
                    .primal
                    .inputs
                    .iter()
                    .any(|input| input.id == table.input && input.binding.symbol == f)
        })
    };
    let f_table = stored.iter().find(|table| is_f(table)).unwrap();
    assert_eq!(f_table.values().len(), mesh.cells().len());
    assert!(f_table.values().iter().all(|value| *value == 0.5));
    let (legacy, _) =
        external_inputs_from(&factorization, model, &mesh, &element, &sources).unwrap();
    let legacy_f = legacy.iter().find(|table| is_f(table)).unwrap();
    assert!(legacy_f.values().iter().all(|value| *value == 0.0));

    // A refusing source is a typed, located, table-labelled failure at the first offending
    // cell in cell order.
    let refusing = vec![
        (symbol(model, "capacity"), constant(1.0)),
        (symbol(model, "k"), constant(1.0)),
        (
            f,
            FieldSource::fallible(|point, _| {
                if point[1] > 0.6 {
                    Err(InputEvaluationError::new(
                        "RUN_PROPERTY_UNSUPPORTED",
                        InputOrigin::Provider("volumetric_source".into()),
                        "undefined above y = 0.6",
                    ))
                } else {
                    Ok(vec![0.0])
                }
            }),
        ),
    ];
    let error = external_inputs_from_at(&factorization, model, &mesh, &element, &refusing, 0.25)
        .unwrap_err();
    let FinitumError::InputEvaluation(failure) = &error else {
        panic!("expected a typed input failure, got {error}");
    };
    assert_eq!(
        failure.origin,
        InputOrigin::Table("provider/volumetric_source".into())
    );
    assert!(failure.point().unwrap()[1] > 0.6);
    assert_eq!(failure.time(), Some(0.25));
    let expected_cell = (0..mesh.cells().len())
        .find(|cell| cell_centroid(&mesh, CellId(*cell)).unwrap()[1] > 0.6)
        .unwrap();
    assert_eq!(failure.cell(), Some(CellId(expected_cell)));

    // System path: `system_constitutive_from_sources` binds a Fallible source as a closure
    // evaluated at the runtime point's time.
    let compiled = compile_coupled();
    let tagged = unit_square(3);
    let m = &compiled.model;
    let sources = vec![
        (symbol(m, "ka"), constant(1.0)),
        (symbol(m, "kb"), constant(1.0)),
        (symbol(m, "ca"), constant(1.0)),
        (
            symbol(m, "fa"),
            FieldSource::fallible(|_, time| Ok(vec![time])),
        ),
        (symbol(m, "fb"), constant(0.5)),
    ];
    let constitutive = system_constitutive_from_sources(&compiled.system, m, &sources).unwrap();
    let (operator, _) = coupled_operators(&compiled, &tagged, constitutive);
    let dimension = operator.dimension();
    let zero = vec![0.0; dimension];
    let residual_at = |time: f64| {
        let mut output = vec![0.0; dimension];
        operator.residual(time, &zero, &zero, &mut output).unwrap();
        output
    };
    let r0 = residual_at(0.0);
    let r1 = residual_at(1.0);
    let r_half = residual_at(0.5);
    assert!(r0.iter().zip(&r1).any(|(a, b)| a != b));
    for ((a, b), half) in r0.iter().zip(&r1).zip(&r_half) {
        assert!(
            (0.5 * (a + b) - half).abs() <= 1.0e-14,
            "the residual is affine in the source's time: {a} {b} {half}"
        );
    }

    // A refusing source is located through the operator action, at the runtime time, with
    // its origin untouched (a constitutive closure is not a stored table).
    let sources = vec![
        (symbol(m, "ka"), constant(1.0)),
        (symbol(m, "kb"), constant(1.0)),
        (symbol(m, "ca"), constant(1.0)),
        (
            symbol(m, "fa"),
            FieldSource::fallible(|point, _| {
                if point[0] > 0.9 {
                    Err(InputEvaluationError::new(
                        "RUN_PROPERTY_UNSUPPORTED",
                        InputOrigin::ExpressionPath("Coupled.ea[0].fa".into()),
                        "undefined beyond x = 0.9",
                    ))
                } else {
                    Ok(vec![0.0])
                }
            }),
        ),
        (symbol(m, "fb"), constant(0.5)),
    ];
    let constitutive = system_constitutive_from_sources(&compiled.system, m, &sources).unwrap();
    let (operator, _) = coupled_operators(&compiled, &tagged, constitutive);
    let view = QuadratureView::of_system_plan(operator.plan()).unwrap();
    let (expected_cell, expected_point) = (0..tagged.mesh.cells().len())
        .find_map(|cell| {
            view.cell_points(CellId(cell))
                .unwrap()
                .into_iter()
                .find(|point| point.physical[0] > 0.9)
                .map(|point| (CellId(cell), point.physical))
        })
        .unwrap();
    let mut output = vec![0.0; dimension];
    let error = operator
        .residual(0.7, &zero, &zero, &mut output)
        .unwrap_err();
    let FinitumError::InputEvaluation(failure) = &error else {
        panic!("expected a typed input failure, got {error}");
    };
    assert_eq!(
        failure.origin,
        InputOrigin::ExpressionPath("Coupled.ea[0].fa".into())
    );
    assert_eq!(failure.cell(), Some(expected_cell));
    assert!(close(failure.point().unwrap(), &expected_point));
    assert_eq!(failure.time(), Some(0.7));
}

// ---------------------------------------------------------------------------------------------
// Finitum's own bound property sources.
// ---------------------------------------------------------------------------------------------

/// A linear table over `axis` on `[0, 1]` that refuses outside its range.
fn unit_table(axis: &str) -> FieldSource {
    FieldSource::table(PropertyTable {
        axes: vec![TableAxis {
            name: axis.into(),
            points: vec![0.0, 1.0],
        }],
        values: vec![1.0, 2.0],
        interpolation: Interpolation::Linear,
        derivative_policy: TableDerivativePolicy::PiecewiseConstantSlope,
        out_of_range: OutOfValidityPolicy::Error,
    })
    .unwrap()
}

#[test]
fn finitum_s_own_bound_property_table_refuses_typed_at_runtime_instead_of_nan_or_a_panic() {
    // System path: `ka` tabulated over `b` on [0, 1]; a state with b = 5 leaves the table.
    let compiled = compile_coupled();
    let tagged = unit_square(3);
    let m = &compiled.model;
    let sources = vec![
        (symbol(m, "ka"), unit_table("b")),
        (symbol(m, "kb"), FieldSource::constant([1.0])),
        (symbol(m, "ca"), FieldSource::constant([1.0])),
        (symbol(m, "fa"), FieldSource::constant([1.0])),
        (symbol(m, "fb"), FieldSource::constant([0.5])),
    ];
    let constitutive = system_constitutive_from_sources(&compiled.system, m, &sources).unwrap();
    let (operator, _) = coupled_operators(&compiled, &tagged, constitutive);
    let dimension = operator.dimension();
    let b_block = operator.layout().block(symbol(m, "b")).unwrap().offset;
    let vertex_count = tagged.mesh.vertices().len();
    let mut state = vec![0.0; dimension];
    let rate = vec![0.0; dimension];
    let mut output = vec![0.0; dimension];
    for vertex in 0..vertex_count {
        state[b_block + vertex] = 0.5;
    }
    operator.residual(0.1, &state, &rate, &mut output).unwrap();
    for vertex in 0..vertex_count {
        state[b_block + vertex] = 5.0;
    }
    let error = operator
        .residual(0.1, &state, &rate, &mut output)
        .unwrap_err();
    let FinitumError::InputEvaluation(failure) = &error else {
        panic!("expected a typed input failure, got {error}");
    };
    assert_eq!(failure.code, REALIZATION_PROPERTY_UNAVAILABLE);
    let InputOrigin::ExpressionPath(path) = &failure.origin else {
        panic!(
            "expected an expression-path origin, got {:?}",
            failure.origin
        );
    };
    assert!(
        path.starts_with("Coupled.ea[") && path.ends_with("].ka"),
        "{path}"
    );
    assert_eq!(failure.cell(), Some(CellId(0)));
    assert_eq!(failure.time(), Some(0.1));
    assert_eq!(error.code(), Some(REALIZATION_PROPERTY_UNAVAILABLE));
    assert!(!error.to_string().contains("non-finite"), "{error}");
    let direction = probe_vector(dimension, 1.1, 1.0);
    let error = operator
        .jacobian_vector_product(0.1, &state, &rate, &direction, &rate, &mut output)
        .unwrap_err();
    assert_eq!(error.code(), Some(REALIZATION_PROPERTY_UNAVAILABLE));

    // Single-model path: `k` tabulated over `u`; the dynamic binding used to panic here.
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &compilation.semantic.models[0];
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, constraints) = square_discretization(2);
    let element = PreparedElement::linear_simplex(2).unwrap();
    let sources = vec![
        (symbol(model, "capacity"), FieldSource::constant([1.0])),
        (symbol(model, "k"), unit_table("u")),
        (symbol(model, "f"), FieldSource::constant([0.0])),
    ];
    let (stored, dynamic) =
        external_inputs_from(&factorization, model, &mesh, &element, &sources).unwrap();
    assert_eq!(dynamic.len(), 1);
    let plan = RealizationPlan::new_stateful(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        stored,
        dynamic,
    )
    .unwrap();
    let dimension = plan.dimension();
    let rate = vec![0.0; dimension];
    let mut output = vec![0.0; dimension];
    plan.residual(0.2, &vec![0.5; dimension], &rate, &mut output)
        .unwrap();
    let error = plan
        .residual(0.2, &vec![5.0; dimension], &rate, &mut output)
        .unwrap_err();
    let FinitumError::InputEvaluation(failure) = &error else {
        panic!("expected a typed input failure, got {error}");
    };
    assert_eq!(failure.code, REALIZATION_PROPERTY_UNAVAILABLE);
    let InputOrigin::ExpressionPath(path) = &failure.origin else {
        panic!(
            "expected an expression-path origin, got {:?}",
            failure.origin
        );
    };
    assert!(
        path.starts_with("TransientNonlinear[") && path.ends_with("].k"),
        "{path}"
    );
    assert_eq!(failure.cell(), Some(CellId(0)));
    assert_eq!(failure.time(), Some(0.2));
    assert!(!error.to_string().contains("non-finite"), "{error}");
}
