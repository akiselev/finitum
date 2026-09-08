use finitum::*;
use methodus::{
    BdfConfig, BdfOrder, BdfState, EvaluationContext, NewtonConfig, StepOutcome, bdf_step,
};
use quantitas::UnitRegistry;
use scientia::{compile_operator_system, compile_semantics};
use std::collections::BTreeMap;
fn fixture() -> (ReducedSystemOperator, Vec<Vec<f64>>) {
    let source = r#"module motion; model Heat {
 domain body { dimension = 2; coordinates = cartesian; }
 field u: state scalar H1(order=1) on body { time_role = differential; };
 source f: VolumetricSource;
 equation evolution on body { (1 + u) * dt(u) - div(grad(u)) = f; }
 }"#;
    let c = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let m = &c.semantic.models[0];
    let u = m.symbols.iter().find(|s| s.name == "u").unwrap().id;
    let f = m.symbols.iter().find(|s| s.name == "f").unwrap().id;
    let system = compile_operator_system(&c.semantic, "Heat", &["evolution"]).unwrap();
    let sources = system_constitutive_from_sources(
        &system,
        m,
        &[(f, FieldSource::fallible(|_, time| Ok(vec![1.0 + time])))],
    )
    .unwrap();
    let mesh = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]; 2],
        subdivisions: vec![2, 2],
    })
    .unwrap()
    .mesh;
    let points = mesh.vertices().to_vec();
    let layout = BlockLayout::new([(u, points.len(), 1)]).unwrap();
    let plan = SystemRealizationPlan::new(system, mesh, layout).unwrap();
    let op = plan.bind_kernels(sources, BTreeMap::new()).unwrap();
    let constraints = ConstraintSet::new(
        points.len(),
        points
            .iter()
            .enumerate()
            .filter(|(_, p)| p.iter().any(|&x| x == 0.0 || x == 1.0))
            .map(|(i, _)| AffineConstraint {
                target: DofId(i),
                dependencies: vec![],
                offset: 0.0,
            }),
    )
    .unwrap();
    (op.reduced(constraints).unwrap(), points)
}
fn moving(op: ReducedSystemOperator, points: &[Vec<f64>]) -> ReducedSystemOperator {
    let values = op
        .constraints()
        .constraints()
        .map(|c| {
            PrescribedEssentialValue::new(
                c.target,
                points[c.target.0].clone(),
                InputOrigin::Slot("boundary/g".into()),
                |time| {
                    Ok(PrescribedValueAndRate {
                        value: time,
                        rate: 1.0,
                    })
                },
            )
        })
        .collect();
    op.with_prescribed_values("g=t; dg/dt=1", values).unwrap()
}
fn near(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < tol, "{x} != {y}");
    }
}
#[test]
fn moving_boundary_mass_contribution_and_bdf_trajectory() {
    let (base, points) = fixture();
    let op = moving(base.clone(), &points);
    let n = points.len();
    let mut rate = vec![1.0; n];
    for c in base.constraints().constraints() {
        rate[c.target.0] = 0.0;
    }
    for time in [0.0, 0.37] {
        let state = vec![time; n];
        let mut residual = vec![0.0; n];
        op.residual(time, &state, &rate, &mut residual).unwrap();
        near(&residual, &vec![0.0; n], 1e-12);
        let (_, physical_rate) = op.physical_state_and_rate(time, &state, &rate).unwrap();
        near(&physical_rate, &vec![1.0; n], 1e-14);
        // Rebuilding values alone is wrong: this static operator drops M_free,boundary*gdot.
        let frozen = base
            .operator()
            .reduced(op.constraints_at(time).unwrap())
            .unwrap();
        frozen.residual(time, &state, &rate, &mut residual).unwrap();
        assert!(residual.iter().any(|x| x.abs() > 0.05));
    }
    let context = EvaluationContext::reproducible();
    let mut state = BdfState::initialize(&op, &context, 0.0, vec![0.0; n]).unwrap();
    let config = BdfConfig {
        order: BdfOrder::One,
        absolute_tolerance: 1e3,
        relative_tolerance: 1e3,
        minimum_step: 1e-8,
        maximum_step: 1.0,
        newton: NewtonConfig::default(),
    };
    for _ in 0..4 {
        match bdf_step(&op, &context, &state, 0.1, &config).unwrap() {
            StepOutcome::Accepted(a) => state = a.state,
            StepOutcome::Rejected(r) => panic!("rejected {}", r.error_estimate),
        };
        near(&state.values, &vec![state.time; n], 1e-10);
    }
}
#[test]
fn state_rate_jvp_fd_and_shifted_transpose() {
    let (base, points) = fixture();
    let op = moving(base, &points);
    let n = points.len();
    let state = (0..n).map(|i| 0.2 + 0.07 * i as f64).collect::<Vec<_>>();
    let rate = vec![0.3; n];
    let d = (0..n).map(|i| (i as f64 + 0.2).sin()).collect::<Vec<_>>();
    let dr = (0..n).map(|i| (i as f64 + 0.6).cos()).collect::<Vec<_>>();
    let seed = (0..n).map(|i| 0.8 - 0.13 * i as f64).collect::<Vec<_>>();
    for time in [0.0, 0.4] {
        let mut action = vec![0.0; n];
        op.jacobian_vector_product(time, &state, &rate, &d, &dr, &mut action)
            .unwrap();
        let eps = 1e-6;
        let mut plus = vec![0.0; n];
        let mut minus = vec![0.0; n];
        for (sign, out) in [(1.0, &mut plus), (-1.0, &mut minus)] {
            let s = state
                .iter()
                .zip(&d)
                .map(|(a, b)| a + sign * eps * b)
                .collect::<Vec<_>>();
            let r = rate
                .iter()
                .zip(&dr)
                .map(|(a, b)| a + sign * eps * b)
                .collect::<Vec<_>>();
            op.residual(time, &s, &r, out).unwrap();
        }
        near(
            &action,
            &plus
                .iter()
                .zip(minus)
                .map(|(a, b)| (a - b) / (2.0 * eps))
                .collect::<Vec<_>>(),
            1e-8,
        );
        for shift in [0.0, 2.7] {
            op.jacobian_vector_product(
                time,
                &state,
                &rate,
                &d,
                &d.iter().map(|x| x * shift).collect::<Vec<_>>(),
                &mut action,
            )
            .unwrap();
            let mut transpose = vec![0.0; n];
            op.vector_jacobian_product_shifted(time, &state, &rate, &seed, shift, &mut transpose)
                .unwrap();
            let lhs: f64 = action.iter().zip(&seed).map(|(a, b)| a * b).sum();
            let rhs: f64 = transpose.iter().zip(&d).map(|(a, b)| a * b).sum();
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }
}
#[test]
fn callback_failures_are_located_and_static_identity_is_preserved() {
    let (base, points) = fixture();
    let original = base.artifact();
    assert_eq!(original.artifact_digest, *base.operator().digest());
    assert!(
        !serde_json::to_string(&original)
            .unwrap()
            .contains("prescribed_motion_identity")
    );
    let op = moving(base.clone(), &points);
    assert_ne!(original.artifact_digest, op.artifact().artifact_digest);
    assert!(op.assemble().is_err());
    assert!(op.element_assembly(1).is_err());
    assert!(op.partial_assembly(1).is_err());
    assert!(op.load_vector().is_err());
    let target = base.constraints().constraints().next().unwrap().target;
    let failure = PrescribedEssentialValue::new(
        target,
        points[target.0].clone(),
        InputOrigin::Slot("g".into()),
        |_| {
            Err(InputEvaluationError::new(
                "RATE_UNAVAILABLE",
                InputOrigin::Provider("g".into()),
                "analytic time derivative absent",
            ))
        },
    );
    let op = base
        .with_prescribed_values("unavailable", vec![failure])
        .unwrap();
    let n = points.len();
    let mut output = vec![0.0; n];
    let error = op
        .residual(0.6, &vec![0.0; n], &vec![0.0; n], &mut output)
        .unwrap_err();
    let FinitumError::InputEvaluation(error) = error else {
        panic!("{error}")
    };
    assert_eq!(error.code, "RATE_UNAVAILABLE");
    assert_eq!(error.time(), Some(0.6));
    assert_eq!(error.point(), Some(points[target.0].as_slice()));
    assert_eq!(error.origin, InputOrigin::Provider("g".into()));
}
#[test]
fn fixed_topology_and_nonfinite_rates_are_checked() {
    let (base, points) = fixture();
    let target = base.constraints().constraints().next().unwrap().target;
    let make = |target| {
        PrescribedEssentialValue::new(target, vec![0.0, 0.0], InputOrigin::Slot("g".into()), |t| {
            Ok(PrescribedValueAndRate {
                value: t * t,
                rate: 2.0 * t,
            })
        })
    };
    assert!(
        base.clone()
            .with_prescribed_values("duplicate", vec![make(target), make(target)])
            .is_err()
    );
    assert!(
        base.clone()
            .with_prescribed_values("missing", vec![make(DofId(points.len()))])
            .is_err()
    );
    let affine = ConstraintSet::new(
        points.len(),
        [AffineConstraint {
            target,
            dependencies: vec![WeightedDof {
                dof: DofId(4),
                weight: 0.5,
            }],
            offset: 0.0,
        }],
    )
    .unwrap();
    assert!(
        base.operator()
            .reduced(affine)
            .unwrap()
            .with_prescribed_values("affine", vec![make(target)])
            .is_err()
    );
    let op = base
        .clone()
        .with_prescribed_values("g=t²", vec![make(target)])
        .unwrap();
    for t in [0.0, 0.7] {
        let (s, r) = op
            .physical_state_and_rate(t, &vec![0.0; points.len()], &vec![0.0; points.len()])
            .unwrap();
        assert_eq!(s[target.0], t * t);
        assert_eq!(r[target.0], 2.0 * t);
    }
    // The contract is a continuous prescribed lifting rate, not a BDF difference of g.
    let time = 0.7;
    let state = vec![0.2; points.len()];
    let mut rate = vec![0.3; points.len()];
    rate[target.0] = 999.0;
    let (physical, physical_rate) = op.physical_state_and_rate(time, &state, &rate).unwrap();
    assert_eq!(physical_rate[target.0], 1.4);
    let mut actual = vec![0.0; points.len()];
    op.residual(time, &state, &rate, &mut actual).unwrap();
    let mut raw = vec![0.0; points.len()];
    op.operator()
        .residual(time, &physical, &physical_rate, &mut raw)
        .unwrap();
    let constraints = op.constraints_at(time).unwrap();
    let mut expected = constraints.restrict_transpose(&raw).unwrap();
    for c in constraints.constraints() {
        expected[c.target.0] = constraints.equation_residual(&state, c.target).unwrap();
    }
    near(&actual, &expected, 1e-12);
    let mut discrete_rate = physical_rate.clone();
    discrete_rate[target.0] = 2.0 * time - 0.1;
    op.operator()
        .residual(time, &physical, &discrete_rate, &mut raw)
        .unwrap();
    let discrete = constraints.restrict_transpose(&raw).unwrap();
    assert!(constraints.constraints().all(|c| c.target != DofId(4)));
    assert!((actual[4] - discrete[4]).abs() > 1e-4);
    let bad = PrescribedEssentialValue::new(
        target,
        points[target.0].clone(),
        InputOrigin::Slot("g".into()),
        |_| {
            Ok(PrescribedValueAndRate {
                value: 1.0,
                rate: f64::NAN,
            })
        },
    );
    let op = base.with_prescribed_values("bad-rate", vec![bad]).unwrap();
    let error = op.constraints_at(0.3).unwrap_err();
    let FinitumError::InputEvaluation(error) = error else {
        panic!("{error}")
    };
    assert_eq!(error.code, "PRESCRIBED_VALUE_NONFINITE");
    assert_eq!(error.time(), Some(0.3));
}

#[test]
fn paired_sources_project_vector_p2_edge_targets_and_check_overlap() {
    let source = r#"module project; model Fields {
      domain body { dimension = 2; coordinates = cartesian; }
      field a: unknown scalar H1(order=1) on body;
      field u: unknown vector(2) H1(order=2) on body;
      source f: ForceDensity;
      equation first on body { -div(grad(a)) = 0; }
      equation second on body { -div(grad(u)) = f; }
      boundary walls on boundary("walls") { dirichlet u = [0,0]; }
    }"#;
    let c = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &c.semantic.models[0];
    let symbol = |name: &str| model.symbols.iter().find(|s| s.name == name).unwrap().id;
    let system = compile_operator_system(&c.semantic, "Fields", &["first", "second"]).unwrap();
    let requirement = system
        .blocks
        .iter()
        .find(|b| b.equation == "second")
        .unwrap()
        .factorization
        .essential_constraints[0]
        .clone();
    let sources = system_constitutive_from_sources(
        &system,
        model,
        &[(symbol("f"), FieldSource::constant(vec![0.0; 2]))],
    )
    .unwrap();
    let mesh = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0]; 2],
        subdivisions: vec![2, 2],
    })
    .unwrap();
    let points = quadratic_simplex_node_points(&mesh.mesh);
    let layout = BlockLayout::new([
        (symbol("a"), mesh.mesh.vertices().len(), 1),
        (symbol("u"), points.len(), 2),
    ])
    .unwrap();
    let plan = SystemRealizationPlan::new(system, mesh.mesh.clone(), layout).unwrap();
    let op = plan.bind_kernels(sources, BTreeMap::new()).unwrap();
    let variable = op
        .system_ids()
        .variable(InstanceId(0), symbol("u"))
        .unwrap();
    let block = op.layout().block_by_variable(variable).unwrap();
    let offset = block.offset;
    let mut regions = RegionMap::new();
    regions.insert(
        requirement.region,
        ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
    );
    let region_maps = [(InstanceId(0), &regions)];
    let zero = SystemVariableEssentialConstraint {
        variable,
        requirement: requirement.clone(),
        value: FieldSource::constant(vec![0.0; 2]),
    };
    let base = op
        .reduced(
            essential_constraints_from_system_by_variable_at(
                &op,
                &mesh,
                &region_maps,
                &[zero],
                0.0,
            )
            .unwrap(),
        )
        .unwrap();
    let paired = SystemVariablePrescribedValue {
        variable,
        requirement,
        value: FieldSource::fallible(|x, t| Ok(vec![x[0] + t * t, x[1] + 2.0 * t * t])),
        rate: FieldSource::fallible(|_, t| Ok(vec![2.0 * t, 4.0 * t])),
        origin: InputOrigin::Slot("u/walls".into()),
    };
    let values = prescribed_values_from_system_by_variable(
        &op,
        &mesh,
        &region_maps,
        &[paired.clone(), paired.clone()],
    )
    .unwrap();
    let moving = base
        .clone()
        .with_prescribed_values("paired", values)
        .unwrap();
    let n = op.dimension();
    let (state, rate) = moving
        .physical_state_and_rate(0.4, &vec![0.0; n], &vec![0.0; n])
        .unwrap();
    assert!(state[..offset].iter().all(|v| *v == 0.0));
    let mut edge_seen = false;
    for c in base.constraints().constraints() {
        let local = c.target.0 - offset;
        let node = local / 2;
        let component = local % 2;
        edge_seen |= node >= mesh.mesh.vertices().len();
        assert_eq!(
            state[c.target.0],
            points[node][component] + (0.4 * 0.4) * (component + 1) as f64
        );
        assert_eq!(rate[c.target.0], 0.8 * (component + 1) as f64);
    }
    assert!(edge_seen);
    let mut conflicting = paired.clone();
    conflicting.rate = FieldSource::constant(vec![3.0, 2.0]);
    let values = prescribed_values_from_system_by_variable(
        &op,
        &mesh,
        &region_maps,
        &[paired.clone(), conflicting],
    )
    .unwrap();
    let error = base
        .clone()
        .with_prescribed_values("conflict", values)
        .unwrap()
        .constraints_at(0.6)
        .unwrap_err();
    let FinitumError::InputEvaluation(error) = error else {
        panic!("{error}")
    };
    assert_eq!(error.code, "PRESCRIBED_SOURCE_CONFLICT");
    assert_eq!(error.time(), Some(0.6));
    let mut refusing = paired;
    refusing.rate = FieldSource::fallible(|_, _| {
        Err(InputEvaluationError::new(
            "ANALYTIC_RATE_MISSING",
            InputOrigin::Provider("u_rate".into()),
            "missing rate",
        ))
    });
    let values =
        prescribed_values_from_system_by_variable(&op, &mesh, &region_maps, &[refusing]).unwrap();
    let error = base
        .with_prescribed_values("missing", values)
        .unwrap()
        .constraints_at(0.7)
        .unwrap_err();
    let FinitumError::InputEvaluation(error) = error else {
        panic!("{error}")
    };
    assert_eq!(error.code, "ANALYTIC_RATE_MISSING");
    assert_eq!(error.time(), Some(0.7));
    assert_eq!(error.origin, InputOrigin::Provider("u_rate".into()));
    assert!(error.point().is_some());
}
