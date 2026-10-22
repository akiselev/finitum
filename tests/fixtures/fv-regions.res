// Scalar conservation laws for the SV2-C2c regionwise FV realization (Finitum test fixture).
// Conserved fields use DG(order=0) (decision D1). Physics stays in the model: the tests bind
// per-region values to the parameters declared here.
module fixtures.fv_regions;

model Conduction1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter a;
  parameter t_hot;
  parameter t_cold;
  constitutive flux = a * u;
  equation balance on Line { dt(u) + div(flux) = 0; }
  boundary left on boundary("left") { dirichlet u = t_hot; }
  boundary right on boundary("right") { dirichlet u = t_cold; }
}

model SourcedAdvection2D {
  domain Square { dimension = 2; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Square { time_role = differential; };
  parameter ax;
  parameter ay;
  parameter q;
  equation balance on Square { dt(u) + div([ax * u, ay * u]) = q; }
}
