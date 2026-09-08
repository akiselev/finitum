module physics.thermal;

// SC-W1 composition fixture (Sinbad-owned): the thermal half of the composed electrothermal
// system. `Q` is an open input a `system` closes by `bind`; `ambient` stays case data
// (`thermal/provider/ambient`: a zero-argument provider rather than an `input value`, because
// Scientia's `input value` slots carry no quantity kind yet -- recorded cross-repo need,
// C12.5); `temperature` is the output the electrical half consumes.
pub model HeatConduction {
    domain body { dimension = 2; coordinates = cartesian; }
    field T: state scalar H1(order=1) on body {
        quantity = ThermodynamicTemperature;
        unit = K;
        nominal = 300 K;
        time_role = differential;
    };
    input field Q: VolumetricHeatSource on body;
    provider ambient() -> ThermodynamicTemperature { differentiability = symbolic; }
    provider density(T: ThermodynamicTemperature) -> Density { differentiability = symbolic; }
    provider specific_heat(T: ThermodynamicTemperature) -> SpecificHeat { differentiability = symbolic; }
    provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity { differentiability = symbolic; }
    property rho = density(T);
    property cp = specific_heat(T);
    property k = thermal_conductivity(T);
    equation thermal on body { rho * cp * dt(T) - div(k * grad(T)) = Q; }
    initial { T = ambient(); }
    output temperature: ThermodynamicTemperature on body = T;
}
