use finitum::{
    AcceleratorLayout, AffineConstraint, Cell, CellBatchLayout, ConstraintSet, DofId, DofMap,
    ElementRestriction, EmbeddedQuadraturePolicy, EmbeddedSegmentQuadrature, ExternalInput,
    HangingNodeConstraint, Mesh, MortarInterface, NonmatchingTransfer, PreparedElement,
    RealizationPlan, TensorProductBasis, VariableOrderSegmentElements, VertexId,
};
use quantitas::UnitRegistry;
use resolvent::{
    InputSourceRequirement, TensorInputRole, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};
use solverang::{
    ConjugateGradientConfig, EvaluationContext, LinearOperator, OperatorSymmetry, SolveError,
    solve_conjugate_gradient,
};

const POISSON: &str = r#"
module fc9.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

#[test]
fn fc9_contracts_refuse_ambiguous_or_malformed_data() {
    assert!(NonmatchingTransfer::lagrange(vec![0.0, 0.0], vec![0.5]).is_err());
    assert!(
        MortarInterface::lagrange(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.5], vec![0.0]).is_err()
    );
    assert!(HangingNodeConstraint::linear(DofId(1), DofId(1), DofId(2), 0.5).is_err());
    assert!(AcceleratorLayout::new(2, 1, 0).is_err());
    assert!(CellBatchLayout::new(2, 0).is_err());
    assert!(TensorProductBasis::new(2, 2, 2, vec![1.0; 3], vec![1.0; 4]).is_err());
    assert!(EmbeddedQuadraturePolicy::new(" ", 2, 0.0).is_err());
}

#[test]
fn nonmatching_mortar_reproduces_traces_and_preserves_interface_work() {
    let mortar = MortarInterface::lagrange(
        vec![0.0, 1.0],
        vec![0.0, 0.5, 1.0],
        vec![0.112_701_665_379_258_3, 0.5, 0.887_298_334_620_741_7],
        vec![5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0],
    )
    .unwrap();
    let minus = [1.0, 3.0];
    let plus = [0.5, 1.25, 2.5];
    let (minus_trace, plus_trace) = mortar.traces(&minus, &plus).unwrap();
    for (point, value) in [0.112_701_665_379_258_3, 0.5, 0.887_298_334_620_741_7]
        .into_iter()
        .zip(&minus_trace)
    {
        assert!((value - (1.0 + 2.0 * point)).abs() < 1.0e-14);
    }
    let flux = [0.7, -0.2, 1.1];
    let (minus_residual, plus_residual) = mortar.scatter_flux(&flux).unwrap();
    let discrete_work = dot(&minus, &minus_residual) + dot(&plus, &plus_residual);
    let mortar_work = flux
        .iter()
        .zip(mortar.quadrature_weights())
        .zip(minus_trace.iter().zip(&plus_trace))
        .map(|((flux, weight), (minus, plus))| weight * flux * (minus - plus))
        .sum::<f64>();
    assert!((discrete_work - mortar_work).abs() < 1.0e-14);
    assert!((minus_residual.iter().sum::<f64>() + plus_residual.iter().sum::<f64>()).abs() < 1e-14);
}

#[test]
fn variable_order_segments_and_affine_dependencies_preserve_polynomials() {
    let mesh = Mesh::new(
        1,
        vec![vec![0.0], vec![0.5], vec![1.0]],
        vec![
            Cell {
                vertices: vec![VertexId(0), VertexId(1)],
            },
            Cell {
                vertices: vec![VertexId(1), VertexId(2)],
            },
        ],
    )
    .unwrap();
    let dofs = DofMap::new(
        5,
        vec![
            ElementRestriction {
                dofs: vec![DofId(0), DofId(1)],
            },
            ElementRestriction {
                dofs: vec![DofId(1), DofId(3), DofId(4), DofId(2)],
            },
        ],
    )
    .unwrap();
    let elements =
        VariableOrderSegmentElements::lagrange_segments(&mesh, &dofs, vec![1, 3]).unwrap();
    assert_eq!(elements.order(finitum::CellId(0)), Some(1));
    assert_eq!(
        elements.element(finitum::CellId(1)).unwrap().basis_count(),
        4
    );
    for cell in 0..elements.cell_count() {
        let element = elements.element(finitum::CellId(cell)).unwrap();
        for point in 0..element.quadrature().len() {
            let value_sum = (0..element.basis_count())
                .map(|basis| element.basis_value(point, basis).unwrap())
                .sum::<f64>();
            let gradient_sum = (0..element.basis_count())
                .map(|basis| element.basis_gradient(point, basis).unwrap()[0])
                .sum::<f64>();
            assert!((value_sum - 1.0).abs() < 2.0e-14);
            assert!(gradient_sum.abs() < 5.0e-14);
        }
    }

    let hanging = HangingNodeConstraint::linear(DofId(4), DofId(1), DofId(2), 2.0 / 3.0).unwrap();
    let constraints = ConstraintSet::new(5, [hanging.into_affine()]).unwrap();
    let expanded = constraints.expand(&[0.0, 2.0, 5.0, 0.0, 99.0]).unwrap();
    assert!((expanded[4] - 4.0).abs() < 1.0e-14);
}

#[test]
fn sum_factorization_and_accelerator_packing_match_dense_meaning() {
    let interpolation = vec![0.75, 0.25, 0.25, 0.75];
    let derivative = vec![-1.0, 1.0, -1.0, 1.0];
    let basis = TensorProductBasis::new(2, 2, 2, interpolation, derivative).unwrap();
    let nodal = [1.0, -2.0, 3.0, 4.0];
    let evaluation = basis.evaluate(&nodal).unwrap();
    let points = [0.25, 0.75];
    let shape = |point: f64| [1.0 - point, point];
    let gradient = [-1.0, 1.0];
    for (x_index, x) in points.iter().copied().enumerate() {
        for (y_index, y) in points.iter().copied().enumerate() {
            let point = x_index * 2 + y_index;
            let mut dense_value = 0.0;
            let mut dense_dx = 0.0;
            let mut dense_dy = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    let value = nodal[i * 2 + j];
                    dense_value += shape(x)[i] * shape(y)[j] * value;
                    dense_dx += gradient[i] * shape(y)[j] * value;
                    dense_dy += shape(x)[i] * gradient[j] * value;
                }
            }
            assert!((evaluation.values[point] - dense_value).abs() < 1.0e-14);
            assert!((evaluation.gradients[point * 2] - dense_dx).abs() < 1.0e-14);
            assert!((evaluation.gradients[point * 2 + 1] - dense_dy).abs() < 1.0e-14);
        }
    }

    let layout = AcceleratorLayout::new(5, 3, 4).unwrap();
    let entity_major = (0..15).map(|value| value as f64 - 3.0).collect::<Vec<_>>();
    let packed = layout.pack(&entity_major).unwrap();
    assert_eq!(packed.len(), 24);
    // entity 4 -> batch 1/lane 0; component 2 -> (1 * 3 + 2) * 4 + 0 = 20.
    assert_eq!(packed[20], entity_major[14]);
    assert_eq!(layout.unpack(&packed).unwrap(), entity_major);
    assert!(packed[15..].contains(&0.0));
}

#[test]
fn embedded_segment_quadrature_integrates_the_clipped_domain() {
    let requirement = EmbeddedQuadraturePolicy::new("phi=x-0.5/v1", 4, 1.0e-8).unwrap();
    let clipped =
        EmbeddedSegmentQuadrature::from_linear_level_set([0.0, 2.0], [-1.0, 3.0], &requirement)
            .unwrap();
    assert_eq!(clipped.active_interval(), Some([0.0, 0.5]));
    assert_eq!(clipped.interface_coordinate(), Some(0.5));
    assert!((clipped.active_measure() - 0.5).abs() < 1.0e-14);
    let linear = clipped
        .points()
        .iter()
        .map(|point| point.weight * point.coordinates[0])
        .sum::<f64>();
    let quadratic = clipped
        .points()
        .iter()
        .map(|point| point.weight * point.coordinates[0].powi(2))
        .sum::<f64>();
    assert!((linear - 0.125).abs() < 1.0e-14);
    assert!((quadratic - 1.0 / 24.0).abs() < 1.0e-14);
}

#[test]
fn partial_assembly_preserves_interpreter_jvp_and_affine_constraints() {
    let plan = poisson_plan_with_hanging_constraint();
    let partial = plan.partial_assembly(4).unwrap();
    let element_assembled = plan.element_assembly(4).unwrap();
    let assembled = plan.assemble().unwrap();
    assert_eq!(
        plan.matrix_free().symmetry(),
        OperatorSymmetry::Nonsymmetric
    );
    assert_eq!(partial.symmetry(), OperatorSymmetry::Nonsymmetric);
    assert_eq!(element_assembled.symmetry(), OperatorSymmetry::Nonsymmetric);
    assert_eq!(assembled.symmetry(), OperatorSymmetry::Nonsymmetric);
    let context = EvaluationContext::reproducible();
    let mut target_column = vec![0.0; plan.dimension()];
    target_column[5] = 1.0;
    let target_column = optimized_action(&assembled, &context, &target_column);
    let mut master_column = vec![0.0; plan.dimension()];
    master_column[1] = 1.0;
    let master_column = optimized_action(&assembled, &context, &master_column);
    assert_eq!(master_column[5], -0.5);
    assert_eq!(target_column[1], 0.0);
    let right_hand_side = vec![1.0; plan.dimension()];
    let initial = vec![0.0; plan.dimension()];
    let cg_error = solve_conjugate_gradient(
        &partial,
        None,
        &context,
        &right_hand_side,
        &initial,
        &ConjugateGradientConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(
        cg_error,
        SolveError::InvalidConfiguration { ref reason } if reason.contains("nonsymmetric")
    ));
    assert_eq!(partial.batches().cell_count(), 18);
    assert_eq!(partial.batches().batch_count(), 5);
    assert_eq!(partial.stored_point_action_count(), 18);
    assert_eq!(
        partial.source_factorization_digest(),
        plan.source_factorization_digest()
    );
    let input = (0..plan.dimension())
        .map(|index| 0.3 * index as f64 - 1.1)
        .collect::<Vec<_>>();
    let mut reference = vec![0.0; plan.dimension()];
    let mut optimized = vec![0.0; plan.dimension()];
    let mut element_output = vec![0.0; plan.dimension()];
    plan.matrix_free()
        .apply(&context, &input, &mut reference)
        .unwrap();
    partial.apply(&context, &input, &mut optimized).unwrap();
    element_assembled
        .apply(&context, &input, &mut element_output)
        .unwrap();
    assert_close(&reference, &optimized, 2.0e-14);
    assert_close(&reference, &element_output, 2.0e-14);

    let state = (0..plan.dimension())
        .map(|index| 0.2 * index as f64 - 0.7)
        .collect::<Vec<_>>();
    let direction = (0..plan.dimension())
        .map(|index| 0.11 * index as f64 - 0.4)
        .collect::<Vec<_>>();
    let zero = vec![0.0; plan.dimension()];
    let mut jvp = vec![0.0; plan.dimension()];
    plan.jacobian_vector_product(0.0, &state, &zero, &direction, &zero, &mut jvp)
        .unwrap();
    let epsilon = 1.0e-6;
    let plus = state
        .iter()
        .zip(&direction)
        .map(|(state, direction)| state + epsilon * direction)
        .collect::<Vec<_>>();
    let minus = state
        .iter()
        .zip(&direction)
        .map(|(state, direction)| state - epsilon * direction)
        .collect::<Vec<_>>();
    let mut plus_residual = vec![0.0; plan.dimension()];
    let mut minus_residual = vec![0.0; plan.dimension()];
    plan.residual(0.0, &plus, &zero, &mut plus_residual)
        .unwrap();
    plan.residual(0.0, &minus, &zero, &mut minus_residual)
        .unwrap();
    let finite_difference = plus_residual
        .iter()
        .zip(&minus_residual)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    assert_close(&jvp, &finite_difference, 2.0e-9);
    assert_close(
        &jvp,
        &optimized_action(&partial, &context, &direction),
        2.0e-14,
    );
    let target = 5;
    assert!(
        (jvp[target] - (direction[target] - 0.5 * direction[1] - 0.5 * direction[9])).abs() < 1e-14
    );
}

fn poisson_plan_with_hanging_constraint() -> RealizationPlan {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let subdivisions = 3;
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
    let mut constraints = (0..width * width)
        .filter(|index| {
            let row = index / width;
            let column = index % width;
            row == 0 || column == 0 || row == subdivisions || column == subdivisions
        })
        .map(|target| AffineConstraint {
            target: DofId(target),
            dependencies: Vec::new(),
            offset: 0.0,
        })
        .collect::<Vec<_>>();
    constraints.push(
        HangingNodeConstraint::linear(DofId(5), DofId(1), DofId(9), 0.5)
            .unwrap()
            .into_affine(),
    );
    let constraints = ConstraintSet::new(width * width, constraints).unwrap();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let external = factorization
        .integrals
        .iter()
        .flat_map(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .filter(|input| input.source != InputSourceRequirement::Basis)
                .map(|input| {
                    assert_ne!(input.role, TensorInputRole::Active);
                    let name = &model.symbols[input.binding.symbol.index()].name;
                    let value = match name.as_str() {
                        "k" => 1.0,
                        "f" => 0.4,
                        other => panic!("unexpected external input {other}"),
                    };
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        move |_, _| vec![value],
                    )
                    .unwrap()
                })
        })
        .collect();
    RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        external,
    )
    .unwrap()
}

fn optimized_action(
    operator: &impl LinearOperator,
    context: &EvaluationContext,
    input: &[f64],
) -> Vec<f64> {
    let mut output = vec![0.0; operator.rows()];
    operator.apply(context, input, &mut output).unwrap();
    output
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }
}
