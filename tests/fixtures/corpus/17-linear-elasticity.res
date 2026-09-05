module corpus.mechanics.linear_elasticity;

model LinearElasticity {
    domain Omega { dimension = 3; coordinates = cartesian; }

    field displacement: unknown vector(3) H1(order=1) on Omega;
    provider lame_lambda(material: selector) -> LameFirstParameter { differentiability = analytic_provided; }
    provider lame_mu(material: selector) -> ShearModulus { differentiability = analytic_provided; }
    provider identity(dimension: selector) -> Dimensionless { shape = tensor(3,3); differentiability = analytic_provided; }
    provider exact_displacement() -> Dimensionless { shape = vector(3); differentiability = analytic_provided; }

    property lambda = lame_lambda(0);
    property mu = lame_mu(0);
    source body_force: MechanicalBodyForce;

    constitutive strain = sym_grad(displacement);
    constitutive stress = lambda * trace(strain) * identity(3) + 2 * mu * strain;

    equation momentum on Omega {
        -div(stress) = body_force;
    }

    boundary clamp on boundary("clamp") {
        dirichlet displacement = exact_displacement();
    }

    observable strain_energy { integrate(0.5 * inner(strain, stress)); }
    observable displacement_magnitude_sq { integrate(dot(displacement, displacement)); }

    @mms(field = displacement);
    @patch_test(field = displacement);
    @rigid_body_modes(count = 6);
    @validation(dataset = "nafems-linear-static");
}
