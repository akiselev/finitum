use finitum::{
    BlockLayout, CoefficientLayout, FieldSampler, InputEvaluationError, InputLocation, InputOrigin,
    InstanceId, MeshProfile, QuadratureRule, SysVarId, SystemRealizationPlan, functional::*,
    realize,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, ProviderId, SemanticCompilation, SymbolId, compile_cell_functional,
    compile_operator_system, compile_semantics,
};
use std::collections::BTreeMap;
const SOURCE: &str = r#"
module w8.functional;
model Example {
 domain body { dimension = 2; coordinates = cartesian; }
 field u: unknown scalar H1(order=1) on body;
 input field observed: Dimensionless on body;
 input value offset: Dimensionless;
 provider law(x: Dimensionless) -> Dimensionless { differentiability = analytic_provided; }
 property first = law(u + offset);
 constitutive nested = law(first) * u;
 source f: VolumetricSource;
 equation balance on body { -div(grad(u)) = f; }
 objective misfit { minimize integrate(0.5 * (u-observed) * (u-observed)); }
 observable energy { integrate(nested * nested); }
 observable norm { integrate(u*u); }
 observable gradient { integrate(dot(grad(u),grad(u))); }
}
"#;
fn compiled() -> SemanticCompilation {
    compile_semantics(SOURCE, &UnitRegistry::si_bootstrap()).unwrap()
}
fn symbol(c: &SemanticCompilation, name: &str) -> SymbolId {
    c.semantic.models[0]
        .symbols
        .iter()
        .find(|s| s.name == name)
        .unwrap()
        .id
}
fn plan(c: &SemanticCompilation) -> SystemRealizationPlan {
    let tagged = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]; 2],
        subdivisions: vec![2, 2],
    })
    .unwrap();
    let op = compile_operator_system(&c.semantic, "Example", &["balance"]).unwrap();
    SystemRealizationPlan::new(
        op,
        tagged.mesh.clone(),
        BlockLayout::new([(symbol(c, "u"), tagged.mesh.vertices().len(), 1)]).unwrap(),
    )
    .unwrap()
}
fn functional(
    c: &SemanticCompilation,
    name: &str,
    bindings: BTreeMap<CaptureKey, CaptureBinding>,
    designs: Vec<FunctionalDesign>,
) -> CellFunctionalPlan {
    let graph = BoundPointExpression::new(
        compile_cell_functional(&c.semantic, "Example", name).unwrap(),
        bindings,
    )
    .unwrap();
    CellFunctionalPlan::new(
        &plan(c),
        InstanceId(0),
        graph,
        FunctionalQuadrature {
            rule: QuadratureRule::for_degree(2, 2).unwrap(),
            reason: "exact quadratic functional".into(),
        },
        designs,
    )
    .unwrap()
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-9 * (1.0 + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
#[test]
fn quadratic_value_and_gradient_functionals_are_exact_and_adjoint() {
    let c = compiled();
    for (name, expected) in [("norm", 7.0 / 6.0), ("gradient", 2.0)] {
        let f = functional(&c, name, BTreeMap::new(), vec![]);
        let state = f
            .realization()
            .mesh()
            .vertices()
            .iter()
            .map(|p| p[0] + p[1])
            .collect::<Vec<_>>();
        let n = state.len();
        let zero = vec![0.0; n];
        let d = (0..n).map(|i| (i as f64).sin()).collect::<Vec<_>>();
        let empty = BTreeMap::new();
        close(f.value(0.4, &state, &zero, &empty).unwrap(), expected);
        let tangent = f
            .jvp(0.4, &state, &zero, &empty, &d, &zero, &empty)
            .unwrap();
        let pull = f.vjp(0.4, &state, &zero, &empty, 1.0).unwrap();
        close(tangent, pull.state.iter().zip(&d).map(|(a, b)| a * b).sum());
        assert!(pull.rate.iter().all(|v| *v == 0.0));
        let e = 1e-5;
        let plus = state
            .iter()
            .zip(&d)
            .map(|(a, b)| a + e * b)
            .collect::<Vec<_>>();
        let minus = state
            .iter()
            .zip(&d)
            .map(|(a, b)| a - e * b)
            .collect::<Vec<_>>();
        close(
            tangent,
            (f.value(0.4, &plus, &zero, &empty).unwrap()
                - f.value(0.4, &minus, &zero, &empty).unwrap())
                / (2.0 * e),
        );
    }
}
#[test]
fn cell_and_vertex_design_pullbacks_use_functional_quadrature() {
    let c = compiled();
    let key = CaptureKey::Input(symbol(&c, "observed"));
    for layout in [CoefficientLayout::Cell, CoefficientLayout::Vertex] {
        let f = functional(
            &c,
            "misfit",
            BTreeMap::from([(key, CaptureBinding::Independent)]),
            vec![FunctionalDesign {
                key,
                layout,
                components: 1,
            }],
        );
        let n = f.realization().layout().extent();
        let m = layout.dimension_at(f.realization().mesh(), 3, 1).unwrap();
        let state = (0..n).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let zero = vec![0.0; n];
        let design = BTreeMap::from([(key, (0..m).map(|i| i as f64 * 0.02).collect())]);
        let direction = BTreeMap::from([(
            key,
            (0..m).map(|i| 0.3 + (i as f64).cos()).collect::<Vec<_>>(),
        )]);
        let tangent = f
            .jvp(0.2, &state, &zero, &design, &zero, &zero, &direction)
            .unwrap();
        let pull = f.vjp(0.2, &state, &zero, &design, 1.0).unwrap();
        close(
            tangent,
            pull.design[&key]
                .iter()
                .zip(&direction[&key])
                .map(|(a, b)| a * b)
                .sum(),
        );
        let e = 1e-5;
        let plus = BTreeMap::from([(
            key,
            design[&key]
                .iter()
                .zip(&direction[&key])
                .map(|(a, b)| a + e * b)
                .collect(),
        )]);
        let minus = BTreeMap::from([(
            key,
            design[&key]
                .iter()
                .zip(&direction[&key])
                .map(|(a, b)| a - e * b)
                .collect(),
        )]);
        close(
            tangent,
            (f.value(0.2, &state, &zero, &plus).unwrap()
                - f.value(0.2, &state, &zero, &minus).unwrap())
                / (2.0 * e),
        );
    }
}
fn provider(derivative: PointDerivative) -> PointProvider {
    PointProvider::new(
        "square/1",
        InputOrigin::Slot("provider/law".into()),
        |_, args| Ok(vec![args[0][0] * args[0][0]]),
        derivative,
    )
}
fn point(c: &SemanticCompilation, u: f64, offset: f64) -> PointExpressionPoint {
    PointExpressionPoint {
        location: InputLocation {
            cell: Some(finitum::CellId(0)),
            point: vec![0.2, 0.3],
            time: Some(0.7),
        },
        fields: vec![PointFieldValue {
            symbol: symbol(c, "u"),
            derivative: DerivativeEvaluation::Value,
            values: vec![u],
        }],
        captures: BTreeMap::from([(CaptureKey::Input(symbol(c, "offset")), vec![offset])]),
    }
}
#[test]
fn nested_provider_argument_graph_has_complete_chain_rule() {
    let c = compiled();
    let graph = BoundPointExpression::new(
        compile_cell_functional(&c.semantic, "Example", "energy").unwrap(),
        BTreeMap::from([
            (
                CaptureKey::Provider(ProviderId(0)),
                CaptureBinding::Provider(provider(PointDerivative::available(|_, a, d| {
                    Ok(vec![2.0 * a[0][0] * d[0][0]])
                }))),
            ),
            (
                CaptureKey::Input(symbol(&c, "offset")),
                CaptureBinding::Independent,
            ),
        ]),
    )
    .unwrap();
    let p = point(&c, 1.2, 0.3);
    let d = point(&c, 0.4, -0.2);
    let expected = 1.2_f64.powi(2) * 1.5_f64.powi(8);
    close(graph.value(&p).unwrap()[0], expected);
    let tangent = graph.jvp(&p, &d).unwrap()[0];
    close(
        tangent,
        2.0 * 1.2 * 0.4 * 1.5_f64.powi(8) + 1.2_f64.powi(2) * 8.0 * 1.5_f64.powi(7) * 0.2,
    );
    let pull = graph.vjp(&p, &[1.0]).unwrap();
    close(
        tangent,
        pull.fields[0].values[0] * 0.4
            + pull.captures[&CaptureKey::Input(symbol(&c, "offset"))][0] * (-0.2),
    );
}
#[test]
fn unavailable_derivative_and_callback_failure_keep_origin_and_time() {
    let c = compiled();
    let kernels = compile_cell_functional(&c.semantic, "Example", "energy").unwrap();
    let bindings = |provider| {
        BTreeMap::from([
            (
                CaptureKey::Provider(ProviderId(0)),
                CaptureBinding::Provider(provider),
            ),
            (
                CaptureKey::Input(symbol(&c, "offset")),
                CaptureBinding::Independent,
            ),
        ])
    };
    let graph = BoundPointExpression::new(
        kernels.clone(),
        bindings(provider(PointDerivative::Unavailable {
            reason: "no tangent".into(),
        })),
    )
    .unwrap();
    let p = point(&c, 1.0, 0.0);
    let d = point(&c, 0.0, 0.0);
    for error in [
        graph.jvp(&p, &d).unwrap_err(),
        graph.vjp(&p, &[0.0]).unwrap_err(),
    ] {
        assert_eq!(error.code(), Some("POINT_TANGENT_UNAVAILABLE"));
        let finitum::FinitumError::InputEvaluation(e) = error else {
            panic!()
        };
        assert_eq!(e.time(), Some(0.7));
        assert_eq!(e.origin, InputOrigin::Slot("provider/law".into()));
    }
    let failing = PointProvider::new(
        "failure",
        InputOrigin::Slot("provider/law".into()),
        |_, _| {
            Err(InputEvaluationError::new(
                "MATERIAL_RANGE",
                InputOrigin::ExpressionPath("law.range".into()),
                "outside domain",
            ))
        },
        PointDerivative::ProvablyZero {
            reason: "fixed".into(),
        },
    );
    let graph = BoundPointExpression::new(kernels, bindings(failing)).unwrap();
    let finitum::FinitumError::InputEvaluation(e) = graph.value(&p).unwrap_err() else {
        panic!()
    };
    assert_eq!(e.code, "MATERIAL_RANGE");
    assert_eq!(e.time(), Some(0.7));
    assert_eq!(e.point(), Some(&[0.2, 0.3][..]));
}
#[test]
fn coefficient_weight_indices_refuse_instead_of_panicking() {
    let c = compiled();
    let p = plan(&c);
    let q = p.quadrature().unwrap();
    for layout in [
        CoefficientLayout::Cell,
        CoefficientLayout::Vertex,
        CoefficientLayout::QuadraturePoint,
    ] {
        assert!(layout.weights_at(p.mesh(), &q, usize::MAX, 0).is_err());
        assert!(layout.weights_at(p.mesh(), &q, 0, usize::MAX).is_err());
    }
}
#[test]
fn keyed_sampler_preserves_one_instance_values() {
    let c = compiled();
    let p = plan(&c);
    let values = vec![2.0; p.layout().extent()];
    let by_symbol = FieldSampler::from_system_plan(&p, symbol(&c, "u"), &values).unwrap();
    let keyed =
        FieldSampler::from_system_plan_by_variable(&p, SysVarId(symbol(&c, "u").0), &values)
            .unwrap();
    assert_eq!(
        by_symbol
            .sample_at_reference(finitum::CellId(0), &[0.2, 0.3])
            .unwrap(),
        keyed
            .sample_at_reference(finitum::CellId(0), &[0.2, 0.3])
            .unwrap()
    );
}

#[test]
fn repeated_instance_sampling_and_functional_scatter_are_distinct() {
    let source = SOURCE.replace("model Example", "pub model Example")
        + r#"
pub system Two {
 domain shared { dimension = 2; coordinates = cartesian; }
 instance left: Example(body = shared);
 instance right: Example(body = shared);
}
"#;
    let closure =
        scientia::resolve_module_closure(&source, &BTreeMap::<String, String>::new()).unwrap();
    let units = UnitRegistry::si_bootstrap();
    let kinds = quantitas::QuantityKindRegistry::si_bootstrap();
    let compilation =
        scientia::compile_system(&closure, scientia::Registries::new(&units, &kinds), "Two")
            .unwrap();
    let operator = scientia::compile_system_operator(&compilation).unwrap();
    let mesh = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]; 2],
        subdivisions: vec![1, 1],
    })
    .unwrap()
    .mesh;
    let ids = finitum::SystemIdMap::from_scientia(&operator.operator).unwrap();
    let layout = BlockLayout::new_keyed(
        ids.variables()
            .iter()
            .map(|v| (v.id, v.local, mesh.vertices().len(), 1)),
    )
    .unwrap();
    let plan = SystemRealizationPlan::composed(
        &operator,
        mesh,
        layout,
        finitum::SystemQuadrature::Richest,
    )
    .unwrap();
    let mut state = vec![0.0; plan.layout().extent()];
    for variable in ids.variables() {
        let block = plan.layout().block_by_variable(variable.id).unwrap();
        state[block.offset..block.offset + block.extent].fill(
            if variable.instance == InstanceId(0) {
                2.0
            } else {
                3.0
            },
        );
    }
    let symbol = ids.variables()[0].local;
    assert!(FieldSampler::from_system_plan(&plan, symbol, &state).is_err());
    for variable in ids.variables() {
        let instance = variable.instance;
        let expected = if instance == InstanceId(0) { 2.0 } else { 3.0 };
        let sampler =
            FieldSampler::from_system_plan_by_variable(&plan, variable.id, &state).unwrap();
        close(
            sampler
                .value_at_reference(finitum::CellId(0), &[0.2, 0.3])
                .unwrap()[0],
            expected,
        );
        let module = &compilation
            .compilation(scientia::InstanceId(instance.0))
            .semantic;
        let graph = BoundPointExpression::new(
            compile_cell_functional(module, "Example", "norm").unwrap(),
            BTreeMap::new(),
        )
        .unwrap();
        let f = CellFunctionalPlan::new(
            &plan,
            instance,
            graph,
            FunctionalQuadrature {
                rule: QuadratureRule::for_degree(2, 2).unwrap(),
                reason: "degree two exact".into(),
            },
            vec![],
        )
        .unwrap();
        let zero = vec![0.0; state.len()];
        close(
            f.value(0.0, &state, &zero, &BTreeMap::new()).unwrap(),
            expected * expected,
        );
        let pull = f.vjp(0.0, &state, &zero, &BTreeMap::new(), 1.0).unwrap();
        let block = plan.layout().block_by_variable(variable.id).unwrap();
        assert!(pull.state.iter().enumerate().all(|(i, value)| {
            ((block.offset..block.offset + block.extent).contains(&i)) || *value == 0.0
        }));
        close(pull.state.iter().sum(), 2.0 * expected);
    }
}

#[test]
fn independent_provider_binding_cannot_erase_argument_dependencies() {
    let c = compiled();
    let graph = BoundPointExpression::new(
        compile_cell_functional(&c.semantic, "Example", "energy").unwrap(),
        BTreeMap::from([
            (
                CaptureKey::Provider(ProviderId(0)),
                CaptureBinding::Independent,
            ),
            (
                CaptureKey::Input(symbol(&c, "offset")),
                CaptureBinding::Independent,
            ),
        ]),
    );
    assert!(
        matches!(graph,Err(finitum::FinitumError::InvalidRealization(message)) if message.contains("POINT_PROVIDER_DEPENDENCY"))
    );
}

#[test]
fn rate_pullback_is_separate_from_state() {
    let source = SOURCE.replace("integrate(u*u)", "integrate(dt(u)*dt(u))");
    let c = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let f = functional(&c, "norm", BTreeMap::new(), vec![]);
    let rate = f
        .realization()
        .mesh()
        .vertices()
        .iter()
        .map(|p| p[0] + p[1])
        .collect::<Vec<_>>();
    let zero = vec![0.0; rate.len()];
    let ones = vec![1.0; rate.len()];
    let empty = BTreeMap::new();
    close(f.value(0.3, &zero, &rate, &empty).unwrap(), 7.0 / 6.0);
    close(
        f.jvp(0.3, &zero, &rate, &empty, &zero, &ones, &empty)
            .unwrap(),
        2.0,
    );
    let pull = f.vjp(0.3, &zero, &rate, &empty, 1.0).unwrap();
    assert!(pull.state.iter().all(|v| *v == 0.0));
    close(pull.rate.iter().sum(), 2.0);
}
#[test]
fn functional_from_another_model_in_same_module_is_refused() {
    let source = SOURCE.to_owned()
        + &SOURCE
            .replace("module w8.functional;", "")
            .replace("model Example", "model Other");
    let c = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let other = c
        .semantic
        .models
        .iter()
        .find(|m| m.name == "Other")
        .unwrap();
    let field = other.symbols.iter().find(|s| s.name == "u").unwrap().id;
    let mesh = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]; 2],
        subdivisions: vec![1, 1],
    })
    .unwrap()
    .mesh;
    let operator = compile_operator_system(&c.semantic, "Other", &["balance"]).unwrap();
    let plan = SystemRealizationPlan::new(
        operator,
        mesh.clone(),
        BlockLayout::new([(field, mesh.vertices().len(), 1)]).unwrap(),
    )
    .unwrap();
    let graph = BoundPointExpression::new(
        compile_cell_functional(&c.semantic, "Example", "norm").unwrap(),
        BTreeMap::new(),
    )
    .unwrap();
    assert!(matches!(
        CellFunctionalPlan::new(
            &plan,
            InstanceId(0),
            graph,
            FunctionalQuadrature {
                rule: QuadratureRule::for_degree(2, 2).unwrap(),
                reason: "exact".into()
            },
            vec![]
        ),
        Err(finitum::FinitumError::ArtifactMismatch(_))
    ));
}
#[test]
fn constant_argument_provider_can_be_an_independent_design() {
    let source = SOURCE.replace("integrate(u*u)", "integrate(law(0)*u*u)");
    let c = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let key = CaptureKey::Provider(ProviderId(0));
    let f = functional(
        &c,
        "norm",
        BTreeMap::from([(key, CaptureBinding::Independent)]),
        vec![FunctionalDesign {
            key,
            layout: CoefficientLayout::Cell,
            components: 1,
        }],
    );
    let state = vec![3.0; f.realization().layout().extent()];
    let zero = vec![0.0; state.len()];
    let design = BTreeMap::from([(key, vec![2.0; f.realization().mesh().cells().len()])]);
    close(f.value(0.0, &state, &zero, &design).unwrap(), 18.0);
    let pull = f.vjp(0.0, &state, &zero, &design, 1.0).unwrap();
    close(pull.design[&key].iter().sum(), 9.0);
}

#[test]
fn reverse_overflow_is_a_located_refusal() {
    let c = compiled();
    let graph = BoundPointExpression::new(
        compile_cell_functional(&c.semantic, "Example", "norm").unwrap(),
        BTreeMap::new(),
    )
    .unwrap();
    let p = point(&c, 1e150, 0.0);
    assert!(graph.value(&p).unwrap()[0].is_finite());
    let error = graph.vjp(&p, &[1e160]).unwrap_err();
    let finitum::FinitumError::InputEvaluation(error) = error else {
        panic!("expected located refusal: {error}")
    };
    assert_eq!(error.time(), Some(0.7));
    assert!(matches!(error.origin, InputOrigin::ExpressionPath(_)));
}

#[test]
fn rehashed_cyclic_argument_metadata_is_refused_at_bind() {
    let c = compiled();
    let mut kernels = compile_cell_functional(&c.semantic, "Example", "energy").unwrap();
    kernels
        .root
        .captures
        .iter_mut()
        .find(|capture| capture.provider.is_some())
        .unwrap()
        .arguments
        .push(kernels.expression);
    kernels.artifact_digest = scientia::Digest {
        algorithm: "blake3".into(),
        hex: String::new(),
    };
    let mut payload = serde_json::to_value(&kernels).unwrap();
    fn strip(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                map.retain(|key, _| key != "span" && !key.ends_with("_span"));
                for value in map.values_mut() {
                    strip(value)
                }
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    strip(value)
                }
            }
            _ => {}
        }
    }
    strip(&mut payload);
    kernels.artifact_digest = scientia::Digest::blake3(&serde_json::to_vec(&payload).unwrap());
    kernels.validate_identity().unwrap();
    let graph = BoundPointExpression::new(
        kernels,
        BTreeMap::from([
            (
                CaptureKey::Provider(ProviderId(0)),
                CaptureBinding::Provider(provider(PointDerivative::available(|_, a, d| {
                    Ok(vec![2.0 * a[0][0] * d[0][0]])
                }))),
            ),
            (
                CaptureKey::Input(symbol(&c, "offset")),
                CaptureBinding::Independent,
            ),
        ]),
    );
    assert!(
        matches!(graph,Err(finitum::FinitumError::InvalidRealization(message)) if message.contains("cycle"))
    );
}
#[test]
fn composed_bound_input_is_not_an_independent_functional_design() {
    let source = SOURCE
        .replace("model Example", "pub model Example")
        .replace("-div(grad(u)) = f", "-div(grad(u)) = f * observed")
        .replace(
            "observable norm { integrate(u*u); }",
            "observable norm { integrate(u*u); }\n output sampled: Dimensionless on body = u;",
        )
        + r#"
pub system Two {
 domain shared { dimension = 2; coordinates = cartesian; }
 instance left: Example(body = shared);
 instance right: Example(body = shared);
 bind left.observed <- right.sampled;
}
"#;
    let closure =
        scientia::resolve_module_closure(&source, &BTreeMap::<String, String>::new()).unwrap();
    let units = UnitRegistry::si_bootstrap();
    let kinds = quantitas::QuantityKindRegistry::si_bootstrap();
    let compilation =
        scientia::compile_system(&closure, scientia::Registries::new(&units, &kinds), "Two")
            .unwrap();
    let operator = scientia::compile_system_operator(&compilation).unwrap();
    let mesh = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]; 2],
        subdivisions: vec![1, 1],
    })
    .unwrap()
    .mesh;
    let ids = finitum::SystemIdMap::from_scientia(&operator.operator).unwrap();
    let layout = BlockLayout::new_keyed(
        ids.variables()
            .iter()
            .map(|v| (v.id, v.local, mesh.vertices().len(), 1)),
    )
    .unwrap();
    let plan = SystemRealizationPlan::composed(
        &operator,
        mesh,
        layout,
        finitum::SystemQuadrature::Richest,
    )
    .unwrap();
    let instance = ids
        .instances()
        .iter()
        .find(|instance| instance.name == "left")
        .unwrap()
        .instance;
    let c = compilation.compilation(scientia::InstanceId(instance.0));
    let key = CaptureKey::Input(symbol(c, "observed"));
    let graph = BoundPointExpression::new(
        compile_cell_functional(&c.semantic, "Example", "misfit").unwrap(),
        BTreeMap::from([(key, CaptureBinding::Independent)]),
    )
    .unwrap();
    let result = CellFunctionalPlan::new(
        &plan,
        instance,
        graph,
        FunctionalQuadrature {
            rule: QuadratureRule::for_degree(2, 2).unwrap(),
            reason: "exact".into(),
        },
        vec![FunctionalDesign {
            key,
            layout: CoefficientLayout::Vertex,
            components: 1,
        }],
    );
    assert!(
        matches!(result,Err(finitum::FinitumError::UnsupportedRealization(message)) if message.contains("POINT_BOUND_FUNCTIONAL_UNSUPPORTED"))
    );
}
