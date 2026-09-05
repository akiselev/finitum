//! W7 / SC-W1 system-path parity (GX-CONTRACTS C12.6 items (a) and (b)): the one-instance
//! `SystemRealizationPlan` reproduces the single-model `RealizationPlan` surface Sinbad's E7
//! path consumes -- stored `from_coefficient` design tables, `coefficient_dimension`, the
//! coefficient JVP/VJP, `linearize` with an explicit transpose, assembled CSR transposes, and
//! the realization-agreement / capability / artifact receipts -- to roundoff, on the corpus
//! snapshots `01-poisson.res` (linear) and `03-nonlinear-heat.res` (transient nonlinear).

use finitum::{
    BlockLayout, CoefficientLayout, ConstraintSet, DerivativeProduct, DistributedCoefficient,
    DynamicExternalInput, ExternalInput, FieldSource, FinitumError, MeshProfile, PreparedElement,
    QuadraturePoint, RealizationPlan, ReducedSystemOperator, RegionMap, RegionTagId,
    RepresentationKind, SYSTEM_OPERATOR_DIGEST_SCHEMA, SYSTEM_REALIZATION_ARTIFACT_SCHEMA,
    SysResId, SystemConstitutiveInput, SystemDistributedCoefficient,
    SystemEssentialConstraintRequirement, SystemExternalInput, SystemRealizationPlan, TaggedMesh,
    check_realization_agreement, check_system_realization_agreement, essential_constraints_from,
    essential_constraints_from_system, realize, simplex_basis, vector_nodal_dof_map,
};
use methodus::{ComparisonTolerance, EvaluationContext, LinearOperator, TransposableOperator};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, SemanticModel, SymbolId,
    compile_operator_system, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels,
};
use std::collections::BTreeMap;

const POISSON: &str = include_str!("fixtures/corpus/01-poisson.res");
const NONLINEAR_HEAT: &str = include_str!("fixtures/corpus/03-nonlinear-heat.res");

/// Roundoff parity between the two compile paths (both integrate with the same table).
const PARITY: f64 = 1.0e-13;
const IDENTITY: f64 = 1.0e-12;
const SOURCE: f64 = 0.4;
const TOLERANCE: ComparisonTolerance = ComparisonTolerance {
    absolute: 1.0e-12,
    relative: 1.0e-12,
};

fn unit_square(subdivisions: usize) -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

fn walls_region_map(region: scientia::RegionId) -> RegionMap {
    let mut map = RegionMap::new();
    map.insert(
        region,
        ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
    );
    map
}

fn probe_vector(dimension: usize, seed: f64, scale: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| scale * ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

/// `max_i |a_i - b_i| / max(||a||_inf, 1e-300)`; panics above `tolerance`.
fn assert_parity(single: &[f64], system: &[f64], tolerance: f64, what: &str) -> f64 {
    assert_eq!(single.len(), system.len(), "{what}: lengths differ");
    let scale = max_abs(single).max(1.0e-300);
    let difference = single
        .iter()
        .zip(system)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
        / scale;
    assert!(
        difference <= tolerance,
        "{what}: single-model vs system relative difference {difference:e} > {tolerance:e}"
    );
    difference
}

fn assert_identity(left: f64, right: f64, what: &str) {
    let scale = left.abs().max(right.abs()).max(1.0e-300);
    assert!(
        (left - right).abs() <= IDENTITY * scale,
        "{what}: <Au,v>={left}, <u,A^Tv>={right}, relative discrepancy {}",
        (left - right).abs() / scale
    );
}

/// The P1 element tabulated at the system path's shared quadrature, so both paths integrate
/// with one and the same rule (the single-model default is the barycenter rule, which is exact
/// for Poisson's linear integrands but not for the nonlinear-heat ones).
fn shared_p1_element(quadrature: &[QuadraturePoint]) -> PreparedElement {
    let mut values = Vec::new();
    let mut gradients = Vec::new();
    for point in quadrature {
        let (basis, basis_gradients) = simplex_basis(2, 1, &point.coordinates).unwrap();
        values.extend(basis);
        for gradient in basis_gradients {
            gradients.extend(gradient);
        }
    }
    PreparedElement::new(2, 3, quadrature.to_vec(), values, gradients).unwrap()
}

fn symbol_name(model: &SemanticModel, symbol: SymbolId) -> &str {
    model.symbols[symbol.index()].name.as_str()
}

struct Compiled {
    semantic: scientia::SemanticCompilation,
    system: OperatorSystem,
}

fn compile(source: &str, model: &str, equation: &str) -> Compiled {
    let semantic = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(&semantic.semantic, model, &[equation]).unwrap();
    Compiled { semantic, system }
}

/// Both realizations of `01-poisson.res` on `tagged`: the single-model plan with `k` a
/// `from_coefficient` table under `layout` and the one-instance reduced system with the
/// identical table laid out over the shared quadrature, sharing one constraint set.
struct PoissonPair {
    plan: RealizationPlan,
    coefficient: DistributedCoefficient,
    reduced: ReducedSystemOperator,
    system_coefficient: SystemDistributedCoefficient,
}

fn poisson_pair(tagged: &TaggedMesh, layout: CoefficientLayout, design: &[f64]) -> PoissonPair {
    let compiled = compile(POISSON, "Poisson", "balance");
    let semantic = &compiled.semantic.semantic;
    let model = &semantic.models[0];
    let form = derive_variational_form(semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let block = &compiled.system.blocks[0];
    assert_eq!(
        block.factorization.artifact_digest, factorization.artifact_digest,
        "the one-instance system carries the single-model factorization verbatim"
    );
    let mesh = tagged.mesh.clone();
    let unknown = block.row;
    let block_layout = BlockLayout::new([(unknown, mesh.vertices().len(), 1)]).unwrap();
    let system_plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.clone(), block_layout).unwrap();
    let quadrature = system_plan.quadrature().unwrap();
    let element = shared_p1_element(&quadrature);
    let dofs = vector_nodal_dof_map(&mesh, 1).unwrap();
    let region_map = walls_region_map(factorization.essential_constraints[0].region);
    let constraints = essential_constraints_from(
        tagged,
        &dofs,
        &factorization.essential_constraints,
        &region_map,
        &[FieldSource::constant([0.0])],
    )
    .unwrap();
    let residual = system_plan.system_ids().residuals()[0].id;
    assert_eq!(residual, SysResId(0));
    let mut external = Vec::new();
    let mut stored = Vec::new();
    let mut coefficient = None;
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let table = match symbol_name(model, input.binding.symbol) {
                "k" => {
                    coefficient = Some(DistributedCoefficient {
                        integral_index: integral.integral_index,
                        input: input.id,
                        layout,
                    });
                    ExternalInput::from_coefficient_at(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &quadrature,
                        layout,
                        design,
                    )
                    .unwrap()
                }
                "f" => ExternalInput::new(
                    integral.integral_index,
                    input.id,
                    1,
                    vec![SOURCE; mesh.cells().len() * quadrature.len()],
                )
                .unwrap(),
                other => panic!("unexpected external input {other}"),
            };
            // The single-model `from_coefficient` over the shared element is the same table.
            if let Some(c) = &coefficient
                && c.input == input.id
            {
                let single = ExternalInput::from_coefficient(
                    integral.integral_index,
                    input.id,
                    1,
                    &mesh,
                    &element,
                    layout,
                    design,
                )
                .unwrap();
                assert_eq!(single, table);
            }
            external.push(table.clone());
            stored.push(SystemExternalInput {
                residual,
                input: table,
            });
        }
    }
    let coefficient = coefficient.expect("the Poisson form binds k");
    let plan = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints.clone(),
        external,
    )
    .unwrap();
    let operator = system_plan
        .bind_kernels_with_inputs(Vec::new(), stored, BTreeMap::new(), BTreeMap::new())
        .unwrap();
    let system_constraints = essential_constraints_from_system(
        &operator,
        tagged,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: unknown,
            requirement: block.factorization.essential_constraints[0].clone(),
            value: FieldSource::constant([0.0]),
        }],
    )
    .unwrap();
    assert_eq!(
        system_constraints, constraints,
        "the system path derives the single-model essential constraint set"
    );
    let reduced = operator.reduced(system_constraints).unwrap();
    PoissonPair {
        plan,
        coefficient: coefficient.clone(),
        reduced,
        system_coefficient: SystemDistributedCoefficient {
            residual,
            coefficient,
        },
    }
}

fn nodal_design(mesh: &TaggedMesh) -> Vec<f64> {
    mesh.mesh
        .vertices()
        .iter()
        .map(|vertex| 1.0 + 0.5 * vertex[0] + 0.25 * vertex[1] * vertex[1])
        .collect()
}

fn cell_design(mesh: &TaggedMesh) -> Vec<f64> {
    (0..mesh.mesh.cells().len())
        .map(|cell| 1.0 + 0.1 * ((cell as f64) * 0.37).cos())
        .collect()
}

/// Every product Sinbad's E7 path consumes, compared between the two paths at one sampled
/// `(t, u, u_t)`; returns the largest relative difference seen.
struct Products<'a> {
    plan: &'a RealizationPlan,
    coefficient: &'a DistributedCoefficient,
    reduced: &'a ReducedSystemOperator,
    system_coefficient: &'a SystemDistributedCoefficient,
}

fn compare_products(products: &Products<'_>, time: f64, rate_shift: f64, label: &str) -> f64 {
    let Products {
        plan,
        coefficient,
        reduced,
        system_coefficient,
    } = products;
    let dimension = plan.dimension();
    assert_eq!(reduced.rows(), dimension);
    let state = probe_vector(dimension, 0.3, 1.0);
    let rate = probe_vector(dimension, 1.7, 0.5);
    let state_direction = probe_vector(dimension, 2.9, 0.8);
    let rate_direction = probe_vector(dimension, 4.1, 0.6);
    let adjoint = probe_vector(dimension, 5.3, 1.1);
    let design_dimension = plan.coefficient_dimension(coefficient).unwrap();
    assert_eq!(
        reduced.coefficient_dimension(system_coefficient).unwrap(),
        design_dimension,
        "{label}: coefficient_dimension"
    );
    let design_direction = probe_vector(design_dimension, 6.7, 0.9);
    let mut worst = 0.0_f64;
    let mut single = vec![0.0; dimension];
    let mut system = vec![0.0; dimension];

    plan.residual(time, &state, &rate, &mut single).unwrap();
    reduced.residual(time, &state, &rate, &mut system).unwrap();
    worst = worst.max(assert_parity(
        &single,
        &system,
        PARITY,
        &format!("{label}: residual"),
    ));

    plan.jacobian_vector_product(
        time,
        &state,
        &rate,
        &state_direction,
        &rate_direction,
        &mut single,
    )
    .unwrap();
    reduced
        .jacobian_vector_product(
            time,
            &state,
            &rate,
            &state_direction,
            &rate_direction,
            &mut system,
        )
        .unwrap();
    worst = worst.max(assert_parity(
        &single,
        &system,
        PARITY,
        &format!("{label}: JVP"),
    ));

    plan.vector_jacobian_product_shifted(time, &state, &rate, &adjoint, rate_shift, &mut single)
        .unwrap();
    reduced
        .vector_jacobian_product_shifted(time, &state, &rate, &adjoint, rate_shift, &mut system)
        .unwrap();
    worst = worst.max(assert_parity(
        &single,
        &system,
        PARITY,
        &format!("{label}: VJP"),
    ));

    plan.coefficient_jacobian_vector_product(
        time,
        &state,
        &rate,
        coefficient,
        &design_direction,
        &mut single,
    )
    .unwrap();
    reduced
        .coefficient_jacobian_vector_product(
            time,
            &state,
            &rate,
            system_coefficient,
            &design_direction,
            &mut system,
        )
        .unwrap();
    worst = worst.max(assert_parity(
        &single,
        &system,
        PARITY,
        &format!("{label}: coefficient JVP"),
    ));
    let forward_work = dot(&system, &adjoint);

    let mut single_design = vec![0.0; design_dimension];
    let mut system_design = vec![0.0; design_dimension];
    plan.coefficient_vector_jacobian_product(
        time,
        &state,
        &rate,
        coefficient,
        &adjoint,
        &mut single_design,
    )
    .unwrap();
    reduced
        .coefficient_vector_jacobian_product(
            time,
            &state,
            &rate,
            system_coefficient,
            &adjoint,
            &mut system_design,
        )
        .unwrap();
    worst = worst.max(assert_parity(
        &single_design,
        &system_design,
        PARITY,
        &format!("{label}: coefficient VJP"),
    ));
    assert_identity(
        forward_work,
        dot(&design_direction, &system_design),
        &format!("{label}: system coefficient adjoint identity"),
    );

    let context = EvaluationContext::reproducible();
    let single_jacobian = plan.linearize(time, &state, &rate, rate_shift).unwrap();
    let system_jacobian = reduced.linearize(time, &state, &rate, rate_shift).unwrap();
    single_jacobian
        .apply(&context, &state_direction, &mut single)
        .unwrap();
    system_jacobian
        .apply(&context, &state_direction, &mut system)
        .unwrap();
    worst = worst.max(assert_parity(
        &single,
        &system,
        PARITY,
        &format!("{label}: linearized action"),
    ));
    let forward_work = dot(&system, &adjoint);
    single_jacobian
        .apply_transpose(&context, &adjoint, &mut single)
        .unwrap();
    system_jacobian
        .apply_transpose(&context, &adjoint, &mut system)
        .unwrap();
    worst = worst.max(assert_parity(
        &single,
        &system,
        PARITY,
        &format!("{label}: linearized transpose action"),
    ));
    assert_identity(
        forward_work,
        dot(&state_direction, &system),
        &format!("{label}: system linearized adjoint identity"),
    );

    // Assembled CSR at the linearization point: forward and transposed traversals.
    let assembled = system_jacobian.assemble().unwrap();
    let mut csr = vec![0.0; dimension];
    assembled
        .apply(&context, &state_direction, &mut csr)
        .unwrap();
    single_jacobian
        .apply(&context, &state_direction, &mut single)
        .unwrap();
    worst = worst.max(assert_parity(
        &single,
        &csr,
        PARITY,
        &format!("{label}: assembled action"),
    ));
    assembled
        .apply_transpose(&context, &adjoint, &mut csr)
        .unwrap();
    single_jacobian
        .apply_transpose(&context, &adjoint, &mut single)
        .unwrap();
    worst = worst.max(assert_parity(
        &single,
        &csr,
        PARITY,
        &format!("{label}: assembled transpose action"),
    ));
    eprintln!("{label}: worst single-model vs system relative difference {worst:e}");
    worst
}

#[test]
fn one_instance_poisson_reproduces_the_single_model_products_on_cell_and_vertex_layouts() {
    let tagged = unit_square(4);
    for (layout, design) in [
        (CoefficientLayout::Cell, cell_design(&tagged)),
        (CoefficientLayout::Vertex, nodal_design(&tagged)),
    ] {
        let pair = poisson_pair(&tagged, layout, &design);
        let worst = compare_products(
            &Products {
                plan: &pair.plan,
                coefficient: &pair.coefficient,
                reduced: &pair.reduced,
                system_coefficient: &pair.system_coefficient,
            },
            0.0,
            0.0,
            &format!("poisson {layout:?}"),
        );
        assert!(worst <= PARITY);

        // Zero-point assembled CSR transposes agree between the paths as well.
        let context = EvaluationContext::reproducible();
        let dimension = pair.plan.dimension();
        let probe = probe_vector(dimension, 7.1, 1.0);
        let single = pair.plan.assemble().unwrap().transpose().unwrap();
        let system = pair.reduced.assemble().unwrap();
        let mut left = vec![0.0; dimension];
        let mut right = vec![0.0; dimension];
        single.apply(&context, &probe, &mut left).unwrap();
        system
            .apply_transpose(&context, &probe, &mut right)
            .unwrap();
        assert_parity(&left, &right, PARITY, "poisson zero-point CSR transpose");
        assert_eq!(
            pair.reduced.operator().plan().quadrature().unwrap(),
            pair.reduced.operator().quadrature()
        );
    }
}

#[test]
fn one_instance_poisson_reproduces_the_realization_agreement_report_and_receipts() {
    let tagged = unit_square(3);
    let design = nodal_design(&tagged);
    let pair = poisson_pair(&tagged, CoefficientLayout::Vertex, &design);
    let dimension = pair.plan.dimension();
    let probe: Vec<f64> = (0..dimension)
        .map(|index| ((index as f64) + 1.0).sin())
        .collect();
    let single = check_realization_agreement(&pair.plan, &probe, 4, TOLERANCE).unwrap();
    let system = check_system_realization_agreement(&pair.reduced, &probe, 4, TOLERANCE).unwrap();
    assert_eq!(system.header.subject.identity, "system-operator");
    assert_eq!(
        &system.header.subject.digest,
        pair.reduced.operator().digest()
    );
    for (name, left, right) in [
        (
            "matrix-free",
            &single.body.matrix_free_output,
            &system.body.matrix_free_output,
        ),
        (
            "assembled",
            &single.body.assembled_output,
            &system.body.assembled_output,
        ),
        (
            "element-assembled",
            &single.body.element_assembled_output,
            &system.body.element_assembled_output,
        ),
        (
            "partial-assembled",
            &single.body.partial_assembled_output,
            &system.body.partial_assembled_output,
        ),
    ] {
        assert_parity(left, right, PARITY, &format!("agreement {name} output"));
    }
    for (name, left, right) in [
        ("assembled", &single.body.assembled, &system.body.assembled),
        (
            "element",
            &single.body.element_assembled,
            &system.body.element_assembled,
        ),
        (
            "partial",
            &single.body.partial_assembled,
            &system.body.partial_assembled,
        ),
    ] {
        assert!(left.accepted && right.accepted, "{name} verdict");
        eprintln!(
            "agreement {name}: single max abs error {:e}, system {:e}",
            left.maximum_absolute_error, right.maximum_absolute_error
        );
        assert!(
            (left.maximum_absolute_error - right.maximum_absolute_error).abs() <= 1.0e-14,
            "{name} maximum absolute error {} vs {}",
            left.maximum_absolute_error,
            right.maximum_absolute_error
        );
    }

    // The single-model path declares `Unknown` until proven by assembly; the system path
    // carries Scientia's structural claim (C5.4/C5.5). Once proven, both declare `Symmetric`.
    assert_eq!(
        pair.plan.capability().symmetry,
        methodus::OperatorSymmetry::Unknown
    );
    assert_eq!(
        pair.plan.prove_symmetry(1.0e-12).unwrap(),
        methodus::OperatorSymmetry::Symmetric
    );
    let single = pair.plan.capability();
    let system = pair.reduced.capability();
    assert_eq!(system.schema, single.schema);
    assert_eq!(system.topology_dimension, single.topology_dimension);
    assert_eq!(system.elements, single.elements);
    assert_eq!(system.measures, single.measures);
    assert_eq!(system.constraint_kinds, single.constraint_kinds);
    assert_eq!(system.representation_kinds, single.representation_kinds);
    assert_eq!(system.derivative_products, single.derivative_products);
    assert!(
        system
            .derivative_products
            .contains(&DerivativeProduct::CoefficientVjp)
    );
    assert!(
        system
            .representation_kinds
            .contains(&RepresentationKind::PartialAssembly)
    );
    assert_eq!(system.symmetry, single.symmetry);
    assert_eq!(
        system.receipt.source_requirements_digest,
        single.receipt.source_requirements_digest
    );
    assert_eq!(
        system.receipt.source_factorization_digest,
        single.receipt.source_factorization_digest
    );
    assert_eq!(
        system.receipt.source_kernels_digest,
        single.receipt.source_kernels_digest
    );
    assert_eq!(
        &system.receipt.realization_digest,
        pair.reduced.operator().digest()
    );
    assert_ne!(system.digest, single.digest, "distinct realization digests");

    let artifact = pair.reduced.artifact();
    let single_artifact = pair.plan.artifact();
    assert_eq!(artifact.schema, SYSTEM_REALIZATION_ARTIFACT_SCHEMA);
    assert_eq!(&artifact.artifact_digest, pair.reduced.operator().digest());
    assert_eq!(artifact.blocks.len(), 1);
    assert_eq!(artifact.blocks[0].residual, Some(SysResId(0)));
    assert_eq!(
        artifact.blocks[0].source_factorization_digest,
        single_artifact.source_factorization_digest
    );
    assert_eq!(artifact.fields.len(), 1);
    assert_eq!(artifact.fields[0].dofs, single_artifact.dofs);
    assert_eq!(artifact.constraints, single_artifact.constraints);
    assert_eq!(artifact.mesh, single_artifact.mesh);
    let mut system_inputs = artifact
        .external_inputs
        .iter()
        .map(|input| input.input.clone())
        .collect::<Vec<_>>();
    let mut single_inputs = single_artifact.external_inputs.clone();
    let key = |input: &finitum::RealizationExternalInput| match input {
        finitum::RealizationExternalInput::Stored { input, .. }
        | finitum::RealizationExternalInput::Dynamic { input, .. } => *input,
    };
    system_inputs.sort_by_key(key);
    single_inputs.sort_by_key(key);
    assert_eq!(system_inputs, single_inputs);
    let serialized = serde_json::to_string(&artifact).unwrap();
    assert!(serialized.contains(SYSTEM_REALIZATION_ARTIFACT_SCHEMA));
    assert!(SYSTEM_OPERATOR_DIGEST_SCHEMA.ends_with("/2"));
}

#[test]
fn the_design_vector_is_part_of_the_system_operator_identity() {
    let tagged = unit_square(2);
    let design = cell_design(&tagged);
    let mut perturbed = design.clone();
    perturbed[0] += 1.0e-3;
    let a = poisson_pair(&tagged, CoefficientLayout::Cell, &design);
    let b = poisson_pair(&tagged, CoefficientLayout::Cell, &perturbed);
    let c = poisson_pair(&tagged, CoefficientLayout::Cell, &design);
    assert_ne!(a.reduced.operator().digest(), b.reduced.operator().digest());
    assert_eq!(a.reduced.operator().digest(), c.reduced.operator().digest());
    assert_ne!(a.plan.digest(), b.plan.digest());
}

type Scalar = fn(f64) -> f64;
type Slope = fn(f64, f64) -> f64;

/// Both realizations of `03-nonlinear-heat.res`: `rho = 1 + 0.2 T`, `cp = 1 + 0.3 T^2`,
/// `k = 1 + 0.1 T` as closures with exact directional derivatives on both paths, `Q` a
/// `from_coefficient` cell-layout design table (the coefficient).
fn nonlinear_heat_pair(tagged: &TaggedMesh, design: &[f64]) -> PoissonPair {
    let compiled = compile(NONLINEAR_HEAT, "NonlinearHeat", "energy");
    let semantic = &compiled.semantic.semantic;
    let model = &semantic.models[0];
    let form = derive_variational_form(semantic, "NonlinearHeat", "energy").unwrap();
    let requirements = infer_form_requirements(semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let block = &compiled.system.blocks[0];
    assert_eq!(
        block.factorization.artifact_digest,
        factorization.artifact_digest
    );
    let mesh = tagged.mesh.clone();
    let unknown = block.row;
    let block_layout = BlockLayout::new([(unknown, mesh.vertices().len(), 1)]).unwrap();
    let system_plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.clone(), block_layout).unwrap();
    let quadrature = system_plan.quadrature().unwrap();
    let element = shared_p1_element(&quadrature);
    let dofs = vector_nodal_dof_map(&mesh, 1).unwrap();
    let region_map = walls_region_map(factorization.essential_constraints[0].region);
    let constraints = essential_constraints_from(
        tagged,
        &dofs,
        &factorization.essential_constraints,
        &region_map,
        &[FieldSource::constant([0.0])],
    )
    .unwrap();
    let residual = SysResId(0);
    let value_of = |evaluation: &finitum::PointEvaluation| {
        evaluation.values(DerivativeEvaluation::Value).unwrap()[0]
    };
    let mut stored_single = Vec::new();
    let mut dynamic_single = Vec::new();
    let mut stored_system = Vec::new();
    let mut constitutive = Vec::new();
    let mut coefficient = None;
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = symbol_name(model, input.binding.symbol);
            let (identity, value, direction): (&str, Scalar, Slope) = match name {
                "rho" => ("rho=1+0.2T", |t| 1.0 + 0.2 * t, |_, dt| 0.2 * dt),
                "cp" => ("cp=1+0.3T^2", |t| 1.0 + 0.3 * t * t, |t, dt| 0.6 * t * dt),
                "k" => ("k=1+0.1T", |t| 1.0 + 0.1 * t, |_, dt| 0.1 * dt),
                "Q" => {
                    coefficient = Some(DistributedCoefficient {
                        integral_index: integral.integral_index,
                        input: input.id,
                        layout: CoefficientLayout::Cell,
                    });
                    let table = ExternalInput::from_coefficient_at(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &quadrature,
                        CoefficientLayout::Cell,
                        design,
                    )
                    .unwrap();
                    stored_single.push(table.clone());
                    stored_system.push(SystemExternalInput {
                        residual,
                        input: table,
                    });
                    continue;
                }
                other => panic!("unexpected external input {other}"),
            };
            dynamic_single.push(
                DynamicExternalInput::new(
                    integral.integral_index,
                    input.id,
                    1,
                    identity,
                    move |evaluation| vec![value(value_of(evaluation))],
                    move |evaluation, d| vec![direction(value_of(evaluation), value_of(d))],
                )
                .unwrap(),
            );
            constitutive.push(
                SystemConstitutiveInput::new(
                    block.equation.clone(),
                    integral.integral_index,
                    input.id,
                    1,
                    identity,
                    move |evaluation| vec![value(value_of(evaluation))],
                    move |evaluation, d| vec![direction(value_of(evaluation), value_of(d))],
                )
                .unwrap(),
            );
        }
    }
    let coefficient = coefficient.expect("the energy form binds Q");
    let plan = RealizationPlan::new_stateful(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints.clone(),
        stored_single,
        dynamic_single,
    )
    .unwrap();
    let operator = system_plan
        .bind_kernels_with_inputs(
            constitutive,
            stored_system,
            BTreeMap::new(),
            BTreeMap::new(),
        )
        .unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    PoissonPair {
        plan,
        coefficient: coefficient.clone(),
        reduced,
        system_coefficient: SystemDistributedCoefficient {
            residual,
            coefficient,
        },
    }
}

#[test]
fn one_instance_nonlinear_heat_reproduces_the_single_model_products_at_a_nonzero_state() {
    let tagged = unit_square(3);
    let design = cell_design(&tagged);
    let pair = nonlinear_heat_pair(&tagged, &design);
    let products = Products {
        plan: &pair.plan,
        coefficient: &pair.coefficient,
        reduced: &pair.reduced,
        system_coefficient: &pair.system_coefficient,
    };
    for rate_shift in [0.0, 2.5] {
        let worst = compare_products(
            &products,
            0.4,
            rate_shift,
            &format!("nonlinear heat shift {rate_shift}"),
        );
        assert!(worst <= PARITY);
    }
    // The system's capability is honest about the closure-bound realization: no partial
    // assembly, but the coefficient products on the stored `Q` table.
    let capability = pair.reduced.capability();
    assert!(
        !capability
            .representation_kinds
            .contains(&RepresentationKind::PartialAssembly)
    );
    assert!(
        capability
            .representation_kinds
            .contains(&RepresentationKind::ElementAssembly)
    );
    assert!(
        capability
            .derivative_products
            .contains(&DerivativeProduct::CoefficientJvp)
    );
    assert!(matches!(
        pair.reduced.partial_assembly(4),
        Err(FinitumError::UnsupportedRealization(_))
    ));
}

#[test]
fn system_coefficient_bindings_are_refused_typed() {
    let tagged = unit_square(2);
    let pair = nonlinear_heat_pair(&tagged, &cell_design(&tagged));
    let reduced = &pair.reduced;
    let dimension = reduced.rows();
    let zero = vec![0.0; dimension];
    let mut output = vec![0.0; dimension];
    // A closure-bound input is not a design vector.
    let block = &reduced.operator().plan().system().blocks[0];
    let closure_input = block
        .factorization
        .integrals
        .iter()
        .flat_map(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .map(move |input| (integral, input))
        })
        .find(|(_, input)| {
            input.source != InputSourceRequirement::Basis
                && input.id != pair.system_coefficient.coefficient.input
        })
        .map(|(integral, input)| SystemDistributedCoefficient {
            residual: SysResId(0),
            coefficient: DistributedCoefficient {
                integral_index: integral.integral_index,
                input: input.id,
                layout: CoefficientLayout::Cell,
            },
        })
        .unwrap();
    assert!(matches!(
        reduced.coefficient_dimension(&closure_input),
        Err(FinitumError::UnsupportedRealization(_))
    ));
    // An unknown residual.
    let unknown = SystemDistributedCoefficient {
        residual: SysResId(7),
        coefficient: pair.system_coefficient.coefficient.clone(),
    };
    assert!(matches!(
        reduced.coefficient_dimension(&unknown),
        Err(FinitumError::InvalidRealization(_))
    ));
    // A wrong-length direction.
    assert!(matches!(
        reduced.coefficient_jacobian_vector_product(
            0.0,
            &zero,
            &zero,
            &pair.system_coefficient,
            &[1.0],
            &mut output
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    // A stored table of the wrong extent, a basis input, and a double binding refuse at bind.
    let compiled = compile(POISSON, "Poisson", "balance");
    let mesh = tagged.mesh.clone();
    let block = &compiled.system.blocks[0];
    let layout = BlockLayout::new([(block.row, mesh.vertices().len(), 1)]).unwrap();
    let plan = SystemRealizationPlan::new(compiled.system.clone(), mesh.clone(), layout).unwrap();
    let integral = &block.factorization.integrals[0];
    let external = integral
        .primal
        .inputs
        .iter()
        .find(|input| input.source != InputSourceRequirement::Basis)
        .unwrap();
    let basis = integral
        .primal
        .inputs
        .iter()
        .find(|input| input.source == InputSourceRequirement::Basis)
        .unwrap();
    let short = ExternalInput::new(integral.integral_index, external.id, 1, vec![1.0; 3]).unwrap();
    assert!(matches!(
        plan.bind_kernels_with_inputs(
            Vec::new(),
            vec![SystemExternalInput {
                residual: SysResId(0),
                input: short
            }],
            BTreeMap::new(),
            BTreeMap::new()
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    let point_count = plan.quadrature().unwrap().len();
    let on_basis = ExternalInput::new(
        integral.integral_index,
        basis.id,
        1,
        vec![1.0; mesh.cells().len() * point_count],
    )
    .unwrap();
    assert!(matches!(
        plan.bind_kernels_with_inputs(
            Vec::new(),
            vec![SystemExternalInput {
                residual: SysResId(0),
                input: on_basis
            }],
            BTreeMap::new(),
            BTreeMap::new()
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    let table = ExternalInput::new(
        integral.integral_index,
        external.id,
        1,
        vec![1.0; mesh.cells().len() * point_count],
    )
    .unwrap();
    assert!(matches!(
        plan.bind_kernels_with_inputs(
            Vec::new(),
            vec![
                SystemExternalInput {
                    residual: SysResId(0),
                    input: table.clone()
                },
                SystemExternalInput {
                    residual: SysResId(0),
                    input: table
                }
            ],
            BTreeMap::new(),
            BTreeMap::new()
        ),
        Err(FinitumError::InvalidRealization(_))
    ));
    // The physical (unreduced) operator exposes the same products without constraints.
    let physical = pair.reduced.operator();
    let mut a = vec![0.0; dimension];
    physical
        .coefficient_jacobian_vector_product(
            0.0,
            &zero,
            &zero,
            &pair.system_coefficient,
            &vec![
                1.0;
                physical
                    .coefficient_dimension(&pair.system_coefficient)
                    .unwrap()
            ],
            &mut a,
        )
        .unwrap();
    assert!(max_abs(&a) > 0.0);
    let _ = ConstraintSet::new(dimension, []).unwrap();
}
