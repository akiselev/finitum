use finitum::{
    BoundaryIntegralRealization, FiniteDifferenceRealization, FiniteVolumeFace,
    FiniteVolumeRealization, MethodRealization, NetworkDaeRealization, ParticlePair,
    ParticleRealization, RadialPairPolynomial,
};
use quantitas::UnitRegistry;
use resolvent::{
    AffineMethodKernelSpec, MethodProgram, compile_boundary_integral_method,
    compile_conservation_law_method, compile_finite_difference_method, compile_network_dae_method,
    compile_particle_method, compile_semantics,
};
use solverang::{DaeOperator, EvaluationContext, verify_dae_jvp};

const SOURCE: &str = r#"
module fixtures.fc10;
model Conservation {
  domain Cells { dimension = 1; coordinates = cartesian; }
  field q: state scalar DG(order=0) on Cells { time_role = differential; };
  property speed = transport_speed(0);
  equation balance on Cells { dt(q) + div(speed * q) = 0; }
}
model Stencil {
  domain Grid { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Grid { time_role = differential; };
  equation diffusion on Grid { dt(u) - div(grad(u)) = 0; }
}
model Network {
  domain Graph { dimension = 0; coordinates = lumped; }
  field voltage: state scalar L2(order=0) on Graph { time_role = differential; };
  field current: state scalar L2(order=0) on Graph { time_role = differential; };
  equation node on Graph { dt(voltage) + current = 0; }
  equation branch on Graph { dt(current) - voltage = 0; }
}
model Particles {
  domain Cloud { dimension = 0; coordinates = particle_space; }
  field positions: state vector(2) L2(order=0) on Cloud { time_role = differential; };
  field velocities: state vector(2) L2(order=0) on Cloud { time_role = differential; };
  property mass = particle_mass(0);
  constitutive pair_forces = radial_pair_force(positions);
  equation kinematics on Cloud { dt(positions) = velocities; }
  equation dynamics on Cloud { mass * dt(velocities) = pair_forces; }
}
model Boundary {
  domain Ambient { dimension = 2; coordinates = cartesian; }
  field density: unknown scalar H1(order=1) on Ambient;
  source incident: IncidentField;
  equation representation on Ambient { -div(grad(density)) = incident; }
  boundary surface on boundary("surface") { robin density = single_layer(density); }
}
"#;

fn module() -> resolvent::SemanticModule {
    compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic
}

fn affine(name: &str, inputs: &[&str], coefficients: &[f64]) -> AffineMethodKernelSpec {
    AffineMethodKernelSpec {
        name: name.into(),
        inputs: inputs.iter().map(|input| (*input).into()).collect(),
        coefficients: coefficients.to_vec(),
        constant: 0.0,
    }
}

fn fv_program() -> MethodProgram {
    compile_conservation_law_method(
        &module(),
        "Conservation",
        "balance",
        "q",
        affine("upwind", &["minus", "plus"], &[1.0, 0.0]),
    )
    .unwrap()
}

fn fd_program() -> MethodProgram {
    let mut stencil = affine(
        "negative_laplacian",
        &["left", "center", "right"],
        &[-1.0, 2.0, -1.0],
    );
    stencil.constant = 0.75;
    compile_finite_difference_method(
        &module(),
        "Stencil",
        "diffusion",
        "u",
        vec![-1, 0, 1],
        stencil,
    )
    .unwrap()
}

fn network_program() -> MethodProgram {
    compile_network_dae_method(
        &module(),
        "Network",
        &["node", "branch"],
        &["voltage", "current"],
    )
    .unwrap()
}

fn particle_program() -> MethodProgram {
    compile_particle_method(
        &module(),
        "Particles",
        &["kinematics", "dynamics"],
        "positions",
        "velocities",
        "pair_forces",
    )
    .unwrap()
}

fn boundary_program() -> MethodProgram {
    compile_boundary_integral_method(
        &module(),
        "Boundary",
        "representation",
        "density",
        "surface",
    )
    .unwrap()
}

fn residual(operator: &MethodRealization, state: &[f64], rate: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; operator.dimension()];
    operator
        .residual(
            &EvaluationContext::reproducible(),
            0.0,
            state,
            rate,
            &mut output,
        )
        .unwrap();
    output
}

#[test]
fn finite_volume_fixture_is_locally_conservative_and_uses_the_compiled_flux() {
    let operator = MethodRealization::FiniteVolume(
        FiniteVolumeRealization::new(
            fv_program(),
            vec![1.0; 3],
            vec![
                FiniteVolumeFace { minus: 0, plus: 1 },
                FiniteVolumeFace { minus: 1, plus: 2 },
                FiniteVolumeFace { minus: 2, plus: 0 },
            ],
        )
        .unwrap(),
    );
    let result = residual(&operator, &[1.0, 2.0, 3.0], &[0.0; 3]);
    assert_eq!(result, vec![-2.0, 1.0, 1.0]);
    assert_eq!(result.iter().sum::<f64>(), 0.0);
    assert!(
        verify_dae_jvp(
            &operator,
            &EvaluationContext::reproducible(),
            0.0,
            &[1.0, 2.0, 3.0],
            &[0.3, -0.2, 0.1],
            &[0.2, -0.7, 0.4],
            &[-0.1, 0.5, 0.9],
            1.0e-6,
        )
        .unwrap()
            < 1.0e-9
    );
}

#[test]
fn finite_difference_fixture_matches_a_periodic_centered_stencil() {
    let operator = MethodRealization::FiniteDifference(
        FiniteDifferenceRealization::new(
            fd_program(),
            vec![vec![3, 0, 1], vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 0]],
            vec![1.0; 4],
        )
        .unwrap(),
    );
    assert_eq!(
        residual(&operator, &[0.0, 1.0, 0.0, -1.0], &[0.0; 4]),
        vec![0.75, 2.75, 0.75, -1.25]
    );
    assert!(
        verify_dae_jvp(
            &operator,
            &EvaluationContext::reproducible(),
            0.0,
            &[0.0, 1.0, 0.0, -1.0],
            &[0.0; 4],
            &[0.2, -0.7, 0.4, 0.8],
            &[-0.1, 0.5, 0.9, -0.3],
            1.0e-6,
        )
        .unwrap()
            < 1.0e-9
    );
}

#[test]
fn network_fixture_matches_its_independent_dense_dae_equations() {
    let operator = MethodRealization::NetworkDae(
        NetworkDaeRealization::new(
            network_program(),
            vec![vec![2.0, 0.0], vec![0.0, 3.0]],
            vec![vec![4.0, -1.0], vec![-1.0, 5.0]],
            vec![7.0, 11.0],
        )
        .unwrap(),
    );
    assert_eq!(
        residual(&operator, &[2.0, -1.0], &[0.5, 2.0]),
        vec![3.0, -12.0]
    );
}

#[test]
fn particle_fixture_has_equal_opposite_pair_forces_and_an_energy_gradient() {
    let particle = ParticleRealization::new(
        particle_program(),
        1,
        vec![1.0, 2.0],
        vec![ParticlePair {
            first: 0,
            second: 1,
        }],
        RadialPairPolynomial {
            coefficients: vec![0.0, 0.5],
        },
    )
    .unwrap();
    let state = vec![-1.0, 1.0, 0.0, 0.0];
    let operator = MethodRealization::Particle(particle.clone());
    let result = residual(&operator, &state, &[0.0; 4]);
    assert_eq!(result, vec![0.0, 0.0, -2.0, 2.0]);
    assert_eq!(result[2] + result[3], 0.0);
    let epsilon = 1.0e-6;
    let plus = vec![-1.0 + epsilon, 1.0, 0.0, 0.0];
    let minus = vec![-1.0 - epsilon, 1.0, 0.0, 0.0];
    let derivative = (particle.potential_energy(&plus).unwrap()
        - particle.potential_energy(&minus).unwrap())
        / (2.0 * epsilon);
    assert!((derivative - result[2]).abs() < 1.0e-9);
    assert!(
        verify_dae_jvp(
            &operator,
            &EvaluationContext::reproducible(),
            0.0,
            &state,
            &[0.1, -0.2, 0.3, -0.4],
            &[0.2, -0.7, 0.4, 0.8],
            &[-0.1, 0.5, 0.9, -0.3],
            1.0e-6,
        )
        .unwrap()
            < 1.0e-9
    );
}

#[test]
fn boundary_integral_fixture_preserves_caller_supplied_quadrature_semantics() {
    let operator = MethodRealization::BoundaryIntegral(
        BoundaryIntegralRealization::new(
            boundary_program(),
            vec![0.5, 0.5],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![0.5, 0.5],
            vec![3.0, 3.0],
        )
        .unwrap(),
    );
    assert_eq!(residual(&operator, &[2.0, 2.0], &[0.0; 2]), vec![0.0, 0.0]);
    assert_eq!(
        residual(&operator, &[2.0, 2.0], &[17.0, -23.0]),
        vec![0.0, 0.0]
    );

    let context = EvaluationContext::reproducible();
    let mut first = vec![0.0; 2];
    operator
        .jacobian_vector_product(
            &context,
            0.0,
            &[2.0, 2.0],
            &[1.0, 2.0],
            &[0.25, -0.75],
            &[3.0, 4.0],
            &mut first,
        )
        .unwrap();
    let mut second = vec![0.0; 2];
    operator
        .jacobian_vector_product(
            &context,
            0.0,
            &[2.0, 2.0],
            &[-5.0, 7.0],
            &[0.25, -0.75],
            &[-11.0, 13.0],
            &mut second,
        )
        .unwrap();
    assert_eq!(first, second);
}
