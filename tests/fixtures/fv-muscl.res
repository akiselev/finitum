// Scalar conservation laws for the SV2-C2b MUSCL/diffusion realization (Finitum test fixture).
// Conserved fields use DG(order=0) (decision D1).
module fixtures.fv_muscl;

model DirichletAdvection1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter a;
  constitutive flux = a * u;
  equation balance on Line { dt(u) + div(flux) = 0; }
  boundary left on boundary("left") { dirichlet u = 0; }
  boundary right on boundary("right") { dirichlet u = 1; }
}

model WalledAdvection2D {
  domain Square { dimension = 2; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Square { time_role = differential; };
  parameter ax;
  parameter ay;
  equation balance on Square { dt(u) + div([ax * u, ay * u]) = 0; }
  boundary left on boundary("left") { dirichlet u = 0; }
  boundary right on boundary("right") { neumann u = ax * u; }
  boundary bottom on boundary("bottom") { dirichlet u = 0; }
  boundary top on boundary("top") { neumann u = ay * u; }
}
