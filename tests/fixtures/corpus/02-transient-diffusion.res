module corpus.scalar.transient_diffusion;

model TransientDiffusion {
    domain Omega { dimension = 2; coordinates = cartesian; }

    field u: state scalar H1(order=1) on Omega {
        time_role = differential;
    };

    provider storage_capacity(u: Dimensionless) -> SpecificHeat { differentiability = analytic_provided; }
    provider diffusivity(u: Dimensionless) -> Diffusivity { differentiability = analytic_provided; }
    provider initial_u(t: selector) -> Dimensionless { differentiability = analytic_provided; }
    provider exact_u(t: Time) -> Dimensionless { differentiability = analytic_provided; }

    property capacity = storage_capacity(u);
    property k = diffusivity(u);
    source f: VolumetricSource;

    equation evolution on Omega {
        capacity * dt(u) - div(k * grad(u)) = f;
    }

    initial { u = initial_u(0); }

    boundary walls on boundary("walls") {
        dirichlet u = exact_u(t);
    }

    observable inventory { integrate(capacity * u); }

    @mms(field = u);
    @spatial_convergence(order = 2);
    @temporal_convergence(order = 2);
    @validation(dataset = "internal-mms");
}
