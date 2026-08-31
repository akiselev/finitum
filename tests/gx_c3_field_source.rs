//! GX-C3: kernel/table-backed `FieldSource` inputs and the `external_inputs_from` builder.

use finitum::{
    AffineConstraint, Cell, ConstraintSet, DofId, DofMap, DynamicExternalInput, ElementRestriction,
    ExternalInput, FieldSource, Mesh, PreparedElement, RealizationPlan, VertexId,
};
use quantitas::{Dimension, QuantityKindId, UnitRegistry};
use scientia::scientific::{
    FrameSemantics, OutOfValidityPolicy, PropertyDomain, PropertyEvidence, PropertyInput,
    PropertyLocality, PropertyModel, PropertyOutput, PropertySignature, TableAxis, TensorSymmetry,
    ValueShape,
};
use scientia::{
    DerivativeContract, PropertyDefinition, PropertyKernel, PropertyTable, SemanticModel, SymbolId,
    TableDerivativePolicy, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels, lower_property_kernel, parse_expression,
};

const TRANSIENT_NONLINEAR: &str = r#"
module gx_c3.transient_nonlinear;
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

fn symbol_id(model: &SemanticModel, name: &str) -> SymbolId {
    let index = model
        .symbols
        .iter()
        .position(|symbol| symbol.name == name)
        .unwrap_or_else(|| panic!("model has no symbol named {name:?}"));
    SymbolId(index as u32)
}

fn scalar_input(name: &str) -> PropertyInput {
    PropertyInput {
        name: name.into(),
        quantity_kind: QuantityKindId::new("Dimensionless"),
        dimension: Dimension::DIMENSIONLESS,
        shape: ValueShape::Scalar,
        physical_min: None,
        physical_max: None,
        nominal: None,
    }
}

fn conductivity_definition() -> PropertyDefinition {
    PropertyDefinition {
        signature: PropertySignature {
            id: "diffusivity".into(),
            inputs: vec![scalar_input("u")],
            output: PropertyOutput {
                quantity_kind: QuantityKindId::new("Dimensionless"),
                dimension: Dimension::DIMENSIONLESS,
                shape: ValueShape::Scalar,
                symmetry: TensorSymmetry::None,
                frame: FrameSemantics::Scalar,
            },
            locality: PropertyLocality::Pointwise,
            differentiability: DerivativeContract::Symbolic,
        },
        model: PropertyModel::Expression(parse_expression("1.0 + 0.2 * u").unwrap()),
        domain: PropertyDomain {
            physical_bounds: vec![],
            validity_bounds: vec![],
            phase_constraints: vec![],
            composition_constraints: vec![],
            assumptions: vec![],
            out_of_validity: OutOfValidityPolicy::Warn,
        },
        evidence: PropertyEvidence {
            sources: vec![],
            dataset_digest: None,
            fit_digest: None,
            uncertainty: None,
            notes: Default::default(),
        },
    }
}

fn conductivity_kernel() -> PropertyKernel {
    lower_property_kernel(&conductivity_definition(), &UnitRegistry::si_bootstrap()).unwrap()
}

/// Builds the `TransientNonlinear` plan via [`finitum::external_inputs_from`], with `k`
/// backed by a [`FieldSource::Kernel`] wrapping `1.0 + 0.2 * u` (matching the FC7 gate's hand
/// closure exactly), `capacity` a constant `1.0`, and `f` a zero sampler.
fn kernel_backed_plan() -> RealizationPlan {
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, constraints) = square_discretization(2);
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let sources = vec![
        (
            symbol_id(model, "capacity"),
            FieldSource::constant(vec![1.0]),
        ),
        (
            symbol_id(model, "k"),
            FieldSource::kernel(conductivity_kernel()).unwrap(),
        ),
        (symbol_id(model, "f"), FieldSource::sampled(|_| vec![0.0])),
    ];
    let (stored, dynamic) =
        finitum::external_inputs_from(&factorization, model, &mesh, &element, &sources).unwrap();
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

/// The FC7 gate's hand-written closures, verbatim, for the same model.
fn hand_closure_plan() -> RealizationPlan {
    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, constraints) = square_discretization(2);
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == scientia::InputSourceRequirement::Basis {
                continue;
            }
            let name = &model.symbols[input.binding.symbol.index()].name;
            match name.as_str() {
                "capacity" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        "capacity=1;direction=0/v1",
                        |_| vec![1.0],
                        |_, _| vec![0.0],
                    )
                    .unwrap(),
                ),
                "k" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        "k=1+0.2u;direction=0.2du/v1",
                        |evaluation| {
                            vec![
                                1.0 + 0.2
                                    * evaluation
                                        .values(scientia::DerivativeEvaluation::Value)
                                        .unwrap()[0],
                            ]
                        },
                        |_, direction| {
                            vec![
                                0.2 * direction
                                    .values(scientia::DerivativeEvaluation::Value)
                                    .unwrap()[0],
                            ]
                        },
                    )
                    .unwrap(),
                ),
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

#[test]
fn kernel_field_source_reproduces_hand_closure_residual_and_jvp() {
    let kernel_plan = kernel_backed_plan();
    let hand_plan = hand_closure_plan();
    assert_eq!(kernel_plan.dimension(), hand_plan.dimension());

    let state = vec![0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0];
    let rate = vec![0.0, 0.0, 0.0, 0.0, -0.17, 0.0, 0.0, 0.0, 0.0];
    let state_direction = vec![0.0, 0.0, 0.0, 0.0, 0.73, 0.0, 0.0, 0.0, 0.0];
    let rate_direction = vec![0.0, 0.0, 0.0, 0.0, -0.41, 0.0, 0.0, 0.0, 0.0];

    let mut kernel_residual = vec![0.0; kernel_plan.dimension()];
    let mut hand_residual = vec![0.0; hand_plan.dimension()];
    kernel_plan
        .residual(0.3, &state, &rate, &mut kernel_residual)
        .unwrap();
    hand_plan
        .residual(0.3, &state, &rate, &mut hand_residual)
        .unwrap();
    for (kernel, hand) in kernel_residual.iter().zip(&hand_residual) {
        assert!(
            (kernel - hand).abs() <= 1.0e-12,
            "residual mismatch: kernel={kernel}, hand={hand}"
        );
    }

    let mut kernel_jvp = vec![0.0; kernel_plan.dimension()];
    let mut hand_jvp = vec![0.0; hand_plan.dimension()];
    kernel_plan
        .jacobian_vector_product(
            0.3,
            &state,
            &rate,
            &state_direction,
            &rate_direction,
            &mut kernel_jvp,
        )
        .unwrap();
    hand_plan
        .jacobian_vector_product(
            0.3,
            &state,
            &rate,
            &state_direction,
            &rate_direction,
            &mut hand_jvp,
        )
        .unwrap();
    for (kernel, hand) in kernel_jvp.iter().zip(&hand_jvp) {
        assert!(
            (kernel - hand).abs() <= 1.0e-12,
            "JVP mismatch: kernel={kernel}, hand={hand}"
        );
    }
}

#[test]
fn kernel_field_source_refuses_missing_tangent() {
    let mut definition = conductivity_definition();
    definition.signature.differentiability = DerivativeContract::AnalyticProvided;
    let kernel = lower_property_kernel(&definition, &UnitRegistry::si_bootstrap()).unwrap();
    assert!(kernel.tangents.is_empty());
    let field_source = FieldSource::kernel(kernel).unwrap();

    let compilation =
        compile_semantics(TRANSIENT_NONLINEAR, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let (mesh, _, _) = square_discretization(2);
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let sources = vec![
        (
            symbol_id(model, "capacity"),
            FieldSource::constant(vec![1.0]),
        ),
        (symbol_id(model, "k"), field_source),
        (symbol_id(model, "f"), FieldSource::sampled(|_| vec![0.0])),
    ];
    let result = finitum::external_inputs_from(&factorization, model, &mesh, &element, &sources);
    assert!(matches!(
        result,
        Err(finitum::FinitumError::RealizationTangentUnavailable(_))
    ));
}

fn linear_table() -> PropertyTable {
    PropertyTable {
        axes: vec![TableAxis {
            name: "x".into(),
            points: vec![0.0, 1.0, 2.0, 3.0],
        }],
        values: vec![10.0, 12.0, 16.0, 22.0],
        interpolation: scientia::scientific::Interpolation::Linear,
        derivative_policy: TableDerivativePolicy::PiecewiseConstantSlope,
        out_of_range: OutOfValidityPolicy::Error,
    }
}

/// Exercises [`FieldSource::table`]'s interpolation through the public
/// `essential_constraints_from` boundary path (the interpolator itself is crate-private): a
/// 1-D segment mesh offset from the table's grid puts one tagged endpoint strictly between grid
/// points (an interpolation) and the other outside the table's range (a refusal).
#[test]
fn table_field_source_interpolates_linearly_and_refuses_out_of_range() {
    use finitum::{MeshProfile, RegionMap, RegionTagId, realize, vector_nodal_dof_map};

    let profile = MeshProfile::SimplexBox {
        dimension: 1,
        extent: vec![[0.5, 3.5]],
        subdivisions: vec![3],
    };
    let mesh = realize(&profile).unwrap();
    let dof_map = vector_nodal_dof_map(&mesh.mesh, 1).unwrap();

    // x_min = 0.5, strictly between table grid points 0.0 and 1.0: value = 10 + 0.5*(12-10) = 11.
    let mut region_map = RegionMap::new();
    region_map.insert(scientia::RegionId(0), [RegionTagId::new("x_min")]);
    let requirement = scientia::EssentialConstraintRequirement {
        argument: SymbolId(0),
        region: scientia::RegionId(0),
        condition: scientia::DeclarationId(0),
    };
    let constraints = finitum::essential_constraints_from(
        &mesh,
        &dof_map,
        std::slice::from_ref(&requirement),
        &region_map,
        &[FieldSource::table(linear_table()).unwrap()],
    )
    .unwrap();
    let value = constraints
        .constraints()
        .find(|constraint| constraint.target == DofId(0))
        .expect("x_min vertex is DOF 0")
        .offset;
    assert!((value - 11.0).abs() <= 1.0e-12, "got {value}");

    // x_max = 3.5, outside the table's [0.0, 3.0] range with `OutOfValidityPolicy::Error`.
    let mut region_map = RegionMap::new();
    region_map.insert(scientia::RegionId(0), [RegionTagId::new("x_max")]);
    let result = finitum::essential_constraints_from(
        &mesh,
        &dof_map,
        &[requirement],
        &region_map,
        &[FieldSource::table(linear_table()).unwrap()],
    );
    assert!(result.is_err(), "out-of-range table lookup must refuse");
}

#[test]
fn field_source_identity_digests_differ_for_different_data() {
    let constant_a = FieldSource::constant(vec![1.0]);
    let constant_b = FieldSource::constant(vec![2.0]);
    assert_ne!(constant_a.identity(), constant_b.identity());

    let table_a = FieldSource::table(linear_table()).unwrap();
    let mut different_table = linear_table();
    different_table.values[0] = 999.0;
    let table_b = FieldSource::table(different_table).unwrap();
    assert_ne!(table_a.identity(), table_b.identity());
    assert_ne!(constant_a.identity(), table_a.identity());

    let kernel_a = FieldSource::kernel(conductivity_kernel()).unwrap();
    let mut different_definition = conductivity_definition();
    different_definition.model =
        PropertyModel::Expression(parse_expression("2.0 + 0.5 * u").unwrap());
    let kernel_b_raw =
        lower_property_kernel(&different_definition, &UnitRegistry::si_bootstrap()).unwrap();
    let kernel_b = FieldSource::kernel(kernel_b_raw).unwrap();
    assert_ne!(kernel_a.identity(), kernel_b.identity());
}
