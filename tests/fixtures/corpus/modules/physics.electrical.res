module physics.electrical;

// SC-W1 composition fixture (Sinbad-owned, `ARCHITECTURE.md` §3.8 in today's grammar): the
// electrical half of the composed electrothermal system. `temperature` is an open input a
// `system` closes by `bind`; `joule_heat` and `potential` are the outputs it offers.
pub model ElectricalConduction {
    domain conductor { dimension = 2; coordinates = cartesian; }
    field V: unknown scalar H1(order=1) on conductor;
    input field temperature: ThermodynamicTemperature on conductor;
    provider electrical_conductivity(T: ThermodynamicTemperature) -> ElectricalConductivity { differentiability = symbolic; }
    property sigma = electrical_conductivity(temperature);
    constitutive current_density = -sigma * grad(V);
    equation electrical on conductor { div(current_density) = 0; }
    boundary anode on boundary("anode") { dirichlet V = 1; }
    boundary cathode on boundary("cathode") { dirichlet V = 0; }
    output joule_heat: VolumetricHeatSource on conductor = sigma * dot(grad(V), grad(V));
    output potential = V;
}
