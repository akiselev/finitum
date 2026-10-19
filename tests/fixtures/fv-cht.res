// Prescribed-velocity convection-diffusion laws for the W13 A4 conjugate-heat-transfer FV
// realization (Finitum test fixture). The velocity components `ux`, `uy` are bound per face
// (a Poiseuille profile); diffusion and per-region capacity are realization data.
module fixtures.fv_cht;

model Graetz2D {
  domain Channel { dimension = 2; coordinates = cartesian; }
  field T: state scalar DG(order=0) on Channel { time_role = differential; };
  parameter ux;
  parameter uy;
  parameter t_in;
  parameter t_wall;
  equation balance on Channel { dt(T) + div([ux * T, uy * T]) = 0; }
  boundary inlet on boundary("inlet") { dirichlet T = t_in; }
  boundary outlet on boundary("outlet") { neumann T = ux * T; }
  boundary walls on boundary("walls") { dirichlet T = t_wall; }
}

model GraetzFlux2D {
  domain Channel { dimension = 2; coordinates = cartesian; }
  field T: state scalar DG(order=0) on Channel { time_role = differential; };
  parameter ux;
  parameter uy;
  parameter t_in;
  parameter q_out;
  equation balance on Channel { dt(T) + div([ux * T, uy * T]) = 0; }
  boundary inlet on boundary("inlet") { dirichlet T = t_in; }
  boundary outlet on boundary("outlet") { neumann T = ux * T; }
  boundary walls on boundary("walls") { neumann T = q_out; }
}

model Conjugate2D {
  domain Box { dimension = 2; coordinates = cartesian; }
  field T: state scalar DG(order=0) on Box { time_role = differential; };
  parameter ux;
  parameter uy;
  parameter t_west;
  parameter t_east;
  equation balance on Box { dt(T) + div([ux * T, uy * T]) = 0; }
  boundary west on boundary("west") { dirichlet T = t_west; }
  boundary east on boundary("east") { dirichlet T = t_east; }
  boundary walls on boundary("walls") { neumann T = 0; }
}

model ExchangeFluid2D {
  domain Box { dimension = 2; coordinates = cartesian; }
  field T: state scalar DG(order=0) on Box { time_role = differential; };
  parameter ux;
  parameter uy;
  parameter t_west;
  equation balance on Box { dt(T) + div([ux * T, uy * T]) = 0; }
  boundary west on boundary("west") { dirichlet T = t_west; }
  boundary walls on boundary("walls") { neumann T = 0; }
  boundary interface on boundary("interface") { neumann T = 0; }
}

model Slab1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field T: state scalar DG(order=0) on Line { time_role = differential; };
  parameter ux;
  parameter t_left;
  parameter t_right;
  equation balance on Line { dt(T) + div(ux * T) = 0; }
  boundary left on boundary("left") { dirichlet T = t_left; }
  boundary right on boundary("right") { dirichlet T = t_right; }
}
