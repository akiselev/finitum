module corpus.fluids.stokes;

model StokesFlow {
    domain Fluid { dimension = 2; coordinates = cartesian; }

    field velocity: unknown vector(2) H1(order=2) on Fluid;
    field pressure: unknown scalar L2(order=1) on Fluid;

    provider density(material: selector) -> Density { differentiability = analytic_provided; }
    provider dynamic_viscosity(material: selector) -> DynamicViscosity { differentiability = analytic_provided; }

    property rho = density(0);
    property mu = dynamic_viscosity(0);
    source body_force: MechanicalBodyForce;

    constitutive strain_rate = sym_grad(velocity);
    constitutive viscous_stress = 2 * mu * strain_rate;

    equation momentum on Fluid {
        -div(viscous_stress) + grad(pressure) = body_force;
    }

    equation incompressibility on Fluid {
        div(velocity) = 0;
    }

    boundary walls on boundary("walls") {
        dirichlet velocity = [0, 0];
    }

    observable dissipation { integrate(inner(viscous_stress, strain_rate)); }
    observable mass_defect { integrate(div(velocity)); }

    @inf_sup(pair = "Taylor-Hood");
    @validation(dataset = "nist-pfhub");
}
