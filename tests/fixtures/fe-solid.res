// P1 heat-conduction solid for the W13 A4 FV <-> FE face-flux exchange (Finitum test fixture).
// The interface boundary is natural here; the exchange supplies its flux.
module fixtures.fe_solid;

model Solid {
  domain body { dimension = 2; coordinates = cartesian; }
  field T: unknown scalar H1(order=1) on body {
    quantity = ThermodynamicTemperature; unit = K;
  };
  provider conductivity() -> ThermalConductivity { differentiability = symbolic; }
  property k = conductivity();
  constitutive q = -k * grad(T);
  equation energy on body { div(q) = 0; }
}
