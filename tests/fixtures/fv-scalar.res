// Scalar conservation laws for the SV2-C2a first-order FV realization (Finitum test fixture).
// Conserved fields use DG(order=0) (decision D1).
module fixtures.fv_scalar;

model PeriodicAdvection1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter a;
  constitutive flux = a * u;
  equation balance on Line { dt(u) + div(flux) = 0; }
}

model InflowAdvection1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter a;
  constitutive flux = a * u;
  equation balance on Line { dt(u) + div(flux) = 0; }
  boundary inflow on boundary("left") { dirichlet u = 1; }
  boundary outflow on boundary("right") { neumann u = a * u; }
}

model Burgers1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter k;
  constitutive flux = 0.5 * u * u;
  equation balance on Line { dt(u) + div(flux) = -k * u; }
}

model PeriodicAdvection2D {
  domain Square { dimension = 2; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Square { time_role = differential; };
  parameter ax;
  parameter ay;
  equation balance on Square { dt(u) + div([ax * u, ay * u]) = 0; }
}
