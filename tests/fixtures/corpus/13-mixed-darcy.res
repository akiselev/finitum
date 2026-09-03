module corpus.porous.mixed_darcy;

model MixedDarcy {
    domain Omega { dimension = 3; coordinates = cartesian; }

    field flux: unknown vector(3) HDiv(order=0) on Omega;
    field pressure: unknown scalar L2(order=0) on Omega;

    provider permeability_tensor(material: selector) -> Permeability { differentiability = analytic_provided; }
    provider dynamic_viscosity(material: selector) -> DynamicViscosity { differentiability = analytic_provided; }
    provider inverse(permeability: Permeability) -> Permeability { differentiability = analytic_provided; }
    provider normal() -> Dimensionless { shape = vector(3); differentiability = analytic_provided; }
    provider integrate_boundary(region: Dimensionless, flux: Dimensionless) -> VolumetricSource { differentiability = symbolic; }

    property permeability = permeability_tensor(0);
    property viscosity = dynamic_viscosity(0);
    source source_term: MassSource;
    source body_force: BodyForce;

    constitutive mobility_inverse = viscosity * inverse(permeability);

    equation darcy_law on Omega {
        mobility_inverse * flux + grad(pressure) = body_force;
    }

    equation mass_balance on Omega {
        div(flux) = source_term;
    }

    boundary impermeable on boundary("walls") {
        neumann flux = 0;
    }

    observable total_flow { integrate_boundary("outlet", dot(flux, normal())); }

    @inf_sup(pair = "RT0-P0");
    @validation(dataset = "spe10-mrst");
}
