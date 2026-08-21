use finitum::{
    AffineConstraint, Cell, ConstraintSet, DofId, DofMap, DynamicExternalInput, ElementRestriction,
    ExternalInput, FinitumError, Mesh, PreparedElement, QuadraturePoint, RealizationPlan, VertexId,
    WeightedDof,
};
use quantitas::UnitRegistry;
use resolvent::{
    DerivativeEvaluation, InputSourceRequirement, TensorInputId, TensorInputRole,
    compile_semantics, derive_variational_form, factor_operator, infer_form_requirements,
    lower_operator_kernels,
};
use solverang::{
    ConjugateGradientConfig, EvaluationContext, LinearOperator, solve_conjugate_gradient,
};

const POISSON: &str = r#"
module fc6.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

const TRANSIENT_NONLINEAR: &str = r#"
module fc7.transient_nonlinear;
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

#[test]
fn fixture_triangle_and_constraints_validate() {
    let mesh = Mesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
        }],
    )
    .unwrap();
    assert_eq!(mesh.cells().len(), 1);

    let dofs = DofMap::new(
        3,
        vec![ElementRestriction {
            dofs: vec![DofId(0), DofId(1), DofId(2)],
        }],
    )
    .unwrap();
    assert_eq!(dofs.dof_count(), 3);

    let constraints = ConstraintSet::new(
        3,
        [AffineConstraint {
            target: DofId(2),
            dependencies: vec![
                WeightedDof {
                    dof: DofId(0),
                    weight: 0.5,
                },
                WeightedDof {
                    dof: DofId(1),
                    weight: 0.5,
                },
            ],
            offset: 0.0,
        }],
    )
    .unwrap();
    assert_eq!(constraints.expand(&[2.0, 4.0, 0.0]).unwrap()[2], 3.0);
    assert_eq!(
        constraints.expand(&[2.0]).unwrap_err(),
        FinitumError::ConstraintInputLength {
            actual: 1,
            expected: 3,
        }
    );
}

#[test]
fn mesh_rejects_repeated_simplex_vertices_and_nonfinite_coordinates() {
    let duplicate = Mesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(1)],
        }],
    )
    .unwrap_err();
    assert_eq!(
        duplicate,
        FinitumError::DuplicateCellVertex { cell: 0, vertex: 1 }
    );

    let nonfinite = Mesh::new(1, vec![vec![f64::NAN], vec![1.0]], Vec::new()).unwrap_err();
    assert_eq!(
        nonfinite,
        FinitumError::NonFiniteCoordinate { vertex: 0, axis: 0 }
    );
}

#[test]
fn dof_map_rejects_empty_and_repeated_local_restrictions() {
    assert_eq!(
        DofMap::new(1, vec![ElementRestriction { dofs: Vec::new() }]).unwrap_err(),
        FinitumError::EmptyRestriction { restriction: 0 }
    );
    assert_eq!(
        DofMap::new(
            2,
            vec![ElementRestriction {
                dofs: vec![DofId(0), DofId(0)],
            }],
        )
        .unwrap_err(),
        FinitumError::DuplicateRestrictionDof {
            restriction: 0,
            dof: 0,
        }
    );
}

#[test]
fn constraint_set_rejects_ambiguous_or_nonfinite_data() {
    assert_eq!(
        ConstraintSet::new(
            1,
            [AffineConstraint {
                target: DofId(0),
                dependencies: Vec::new(),
                offset: f64::INFINITY,
            }],
        )
        .unwrap_err(),
        FinitumError::InvalidConstraintCoefficient { target: 0 }
    );

    let duplicate_dependency = ConstraintSet::new(
        2,
        [AffineConstraint {
            target: DofId(1),
            dependencies: vec![
                WeightedDof {
                    dof: DofId(0),
                    weight: 0.5,
                },
                WeightedDof {
                    dof: DofId(0),
                    weight: 0.5,
                },
            ],
            offset: 0.0,
        }],
    )
    .unwrap_err();
    assert_eq!(
        duplicate_dependency,
        FinitumError::DuplicateConstraintDependency {
            target: 1,
            dependency: 0,
        }
    );

    assert_eq!(
        ConstraintSet::new(
            2,
            [
                AffineConstraint {
                    target: DofId(0),
                    dependencies: vec![WeightedDof {
                        dof: DofId(1),
                        weight: 1.0,
                    }],
                    offset: 0.0,
                },
                AffineConstraint {
                    target: DofId(1),
                    dependencies: vec![WeightedDof {
                        dof: DofId(0),
                        weight: 1.0,
                    }],
                    offset: 0.0,
                },
            ],
        )
        .unwrap_err(),
        FinitumError::ConstraintCycle(0)
    );

    let constraints = ConstraintSet::new(
        2,
        [AffineConstraint {
            target: DofId(1),
            dependencies: vec![WeightedDof {
                dof: DofId(0),
                weight: f64::MAX,
            }],
            offset: 0.0,
        }],
    )
    .unwrap();
    assert_eq!(
        constraints.expand(&[f64::NAN, 0.0]).unwrap_err(),
        FinitumError::NonFiniteConstraintInput(0)
    );
    assert_eq!(
        constraints.expand(&[f64::MAX, 0.0]).unwrap_err(),
        FinitumError::NonFiniteConstraintResult(1)
    );
}

#[test]
fn prepared_element_rejects_shape_and_nonfinite_tables() {
    let quadrature = vec![QuadraturePoint {
        coordinates: vec![0.25, 0.25],
        weight: 0.5,
    }];
    assert!(matches!(
        PreparedElement::new(2, 3, quadrature.clone(), vec![1.0; 2], vec![0.0; 6]),
        Err(FinitumError::InvalidElementShape(_))
    ));
    assert!(matches!(
        PreparedElement::new(4, 1, quadrature.clone(), vec![1.0], vec![0.0; 4]),
        Err(FinitumError::InvalidElementShape(_))
    ));
    assert_eq!(
        PreparedElement::new(
            2,
            3,
            quadrature,
            vec![f64::INFINITY, 0.0, 0.0],
            vec![0.0; 6],
        )
        .unwrap_err(),
        FinitumError::NonFiniteElementData {
            location: "basis value 0".into(),
        }
    );
}

#[test]
fn dynamic_external_input_requires_an_explicit_identity() {
    assert!(matches!(
        DynamicExternalInput::new(0, TensorInputId(0), 1, " ", |_| vec![1.0], |_, _| vec![0.0],),
        Err(FinitumError::InvalidRealization(_))
    ));
}

#[test]
fn fc6_binds_generated_kernels_and_assembled_matrix_free_actions_agree() {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, constraints) = square_discretization(2);
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
                        "f" => 1.0,
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
    let plan = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        external,
    )
    .unwrap();
    let matrix_free = plan.matrix_free();
    let assembled = plan.assemble().unwrap();
    assert_eq!(
        matrix_free.source_factorization_digest(),
        assembled.source_factorization_digest()
    );

    let context = EvaluationContext::reproducible();
    let direction = (0..plan.dimension())
        .map(|index| index as f64 - 2.5)
        .collect::<Vec<_>>();
    let mut matrix_free_output = vec![0.0; plan.dimension()];
    let mut assembled_output = vec![0.0; plan.dimension()];
    matrix_free
        .apply(&context, &direction, &mut matrix_free_output)
        .unwrap();
    assembled
        .apply(&context, &direction, &mut assembled_output)
        .unwrap();
    assert_close(&matrix_free_output, &assembled_output, 1.0e-14);
    for boundary in [0, 1, 2, 3, 5, 6, 7, 8] {
        assert_eq!(matrix_free_output[boundary], direction[boundary]);
    }

    let right_hand_side = plan.load_vector().unwrap();
    let report = solve_conjugate_gradient(
        &matrix_free,
        None,
        &context,
        &right_hand_side,
        &vec![0.0; plan.dimension()],
        &ConjugateGradientConfig::default(),
    )
    .unwrap();
    assert!(report.converged);
    assert!(report.solution[4] > 0.0);
}

#[test]
fn fc6_linear_patch_is_exact_on_a_nonuniform_sheared_mesh() {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();

    // A sheared unit square with an off-center interior vertex. The four determinants differ,
    // and every cell Jacobian has an off-diagonal entry, so neither a transposed pullback nor an
    // omitted/permuted determinant can hide behind an orthogonal uniform mesh.
    let vertices = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.35, 1.0],
        vec![1.35, 1.0],
        vec![0.573, 0.58],
    ];
    let cells = vec![
        Cell {
            vertices: vec![VertexId(0), VertexId(1), VertexId(4)],
        },
        Cell {
            vertices: vec![VertexId(1), VertexId(3), VertexId(4)],
        },
        Cell {
            vertices: vec![VertexId(3), VertexId(2), VertexId(4)],
        },
        Cell {
            vertices: vec![VertexId(2), VertexId(0), VertexId(4)],
        },
    ];
    let determinants = cells
        .iter()
        .map(|cell| {
            let origin = &vertices[cell.vertices[0].0];
            let first = &vertices[cell.vertices[1].0];
            let second = &vertices[cell.vertices[2].0];
            (first[0] - origin[0]) * (second[1] - origin[1])
                - (second[0] - origin[0]) * (first[1] - origin[1])
        })
        .collect::<Vec<_>>();
    assert!(determinants.windows(2).all(|pair| pair[0] != pair[1]));
    assert_ne!(vertices[4][0] - vertices[0][0], 0.0);

    let restrictions = cells
        .iter()
        .map(|cell| ElementRestriction {
            dofs: cell.vertices.iter().map(|vertex| DofId(vertex.0)).collect(),
        })
        .collect();
    let mesh = Mesh::new(2, vertices.clone(), cells).unwrap();
    let dofs = DofMap::new(vertices.len(), restrictions).unwrap();
    let exact = |point: &[f64]| 0.7 - 1.25 * point[0] + 0.8 * point[1];
    let constraints = ConstraintSet::new(
        vertices.len(),
        (0..4).map(|target| AffineConstraint {
            target: DofId(target),
            dependencies: Vec::new(),
            offset: exact(&vertices[target]),
        }),
    )
    .unwrap();
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
                    let name = &model.symbols[input.binding.symbol.index()].name;
                    let value = match name.as_str() {
                        "k" => 1.0,
                        "f" => 0.0,
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
    let plan = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        external,
    )
    .unwrap();
    let right_hand_side = plan.load_vector().unwrap();
    let context = EvaluationContext::reproducible();
    let matrix_free = plan.matrix_free();
    let assembled = plan.assemble().unwrap();
    for operator in [&matrix_free as &dyn LinearOperator, &assembled] {
        let report = solve_conjugate_gradient(
            operator,
            None,
            &context,
            &right_hand_side,
            &vec![0.0; plan.dimension()],
            &ConjugateGradientConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        let expected = vertices
            .iter()
            .map(|point| exact(point))
            .collect::<Vec<_>>();
        assert_close(&report.solution, &expected, 1.0e-12);
    }
}

#[test]
fn fc6_refuses_a_kernel_bundle_from_another_factorization() {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let mut kernels = lower_operator_kernels(&factorization).unwrap();
    kernels.source_factorization_digest.hex = "not-the-parent".into();
    let (mesh, dofs, constraints) = square_discretization(2);
    let error = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        PreparedElement::linear_simplex(2).unwrap(),
        dofs,
        constraints,
        Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(error, FinitumError::ArtifactMismatch(_)));
}

#[test]
fn fc7_runtime_state_rate_and_property_chain_rule_match_finite_differences() {
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
            if input.source == InputSourceRequirement::Basis {
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
                                    * evaluation.values(DerivativeEvaluation::Value).unwrap()[0],
                            ]
                        },
                        |_, direction| {
                            vec![0.2 * direction.values(DerivativeEvaluation::Value).unwrap()[0]]
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
    assert_eq!(plan.digest().algorithm, "blake3");
    let state = vec![0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0];
    let rate = vec![0.0, 0.0, 0.0, 0.0, -0.17, 0.0, 0.0, 0.0, 0.0];
    let state_direction = vec![0.0, 0.0, 0.0, 0.0, 0.73, 0.0, 0.0, 0.0, 0.0];
    let rate_direction = vec![0.0, 0.0, 0.0, 0.0, -0.41, 0.0, 0.0, 0.0, 0.0];
    let mut analytic = vec![0.0; plan.dimension()];
    plan.jacobian_vector_product(
        0.3,
        &state,
        &rate,
        &state_direction,
        &rate_direction,
        &mut analytic,
    )
    .unwrap();
    let epsilon = 1.0e-6;
    let plus_state = state
        .iter()
        .zip(&state_direction)
        .map(|(value, direction)| value + epsilon * direction)
        .collect::<Vec<_>>();
    let minus_state = state
        .iter()
        .zip(&state_direction)
        .map(|(value, direction)| value - epsilon * direction)
        .collect::<Vec<_>>();
    let plus_rate = rate
        .iter()
        .zip(&rate_direction)
        .map(|(value, direction)| value + epsilon * direction)
        .collect::<Vec<_>>();
    let minus_rate = rate
        .iter()
        .zip(&rate_direction)
        .map(|(value, direction)| value - epsilon * direction)
        .collect::<Vec<_>>();
    let mut plus = vec![0.0; plan.dimension()];
    let mut minus = vec![0.0; plan.dimension()];
    plan.residual(0.3, &plus_state, &plus_rate, &mut plus)
        .unwrap();
    plan.residual(0.3, &minus_state, &minus_rate, &mut minus)
        .unwrap();
    let finite_difference = plus
        .iter()
        .zip(&minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    assert_close(&analytic, &finite_difference, 2.0e-9);
}

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

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }
}
