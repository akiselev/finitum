module corpus.coupled.electrothermal;

model ElectrothermalJoule {
    domain Omega { dimension = 2; coordinates = cartesian; }

    field V: unknown scalar H1(order=1) on Omega;
    field T: state scalar H1(order=1) on Omega {
        quantity = ThermodynamicTemperature;
        unit = K;
        nominal = 300 K;
        time_role = differential;
    };

    provider electrical_conductivity(T: ThermodynamicTemperature) -> ElectricalConductivity { differentiability = symbolic; }
    provider density(T: ThermodynamicTemperature) -> Density { differentiability = symbolic; }
    provider specific_heat(T: ThermodynamicTemperature) -> SpecificHeat { differentiability = symbolic; }
    provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity { differentiability = symbolic; }

    property sigma = electrical_conductivity(T);
    property rho = density(T);
    property cp = specific_heat(T);
    property k = thermal_conductivity(T);

    constitutive current_density = -sigma * grad(V);
    source joule = sigma * dot(grad(V), grad(V));

    equation electrical on Omega {
        div(current_density) = 0;
    }

    equation thermal on Omega {
        rho * cp * dt(T) - div(k * grad(T)) = joule;
    }

    // Electrodes (batch P, GX-CONTRACTS C12): a unit potential difference is imposed across the
    // `anode`/`cathode` regions so the electrical unknown is gauged and the Joule source is
    // nonzero; every other boundary is naturally closed (insulated: zero current and heat flux).
    boundary anode on boundary("anode") {
        dirichlet V = 1;
    }

    boundary cathode on boundary("cathode") {
        dirichlet V = 0;
    }

    initial { T = 300 K; }

    observable electrical_power { integrate(joule); }
    observable thermal_energy { integrate(rho * cp * T); }

    @jvp_taylor(block = electrical);
    @jvp_taylor(block = thermal);
    @validation(dataset = "internal-mms");
}
