// Ideal-gas compressible Euler for the W13 A7 multi-component FV realization tests (Finitum
// test fixture; product models live in Sinbad). `outer`/`identity` are language intrinsics (no
// provider declarations). Speeds and admissible quantities are constitutive declarations named
// by the caller's ConservationSystemSpec.
module fixtures.fv_euler;

// Transmissive (zero-gradient) ends: the ghost state is the interior state.
model Euler1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field density: state scalar DG(order=0) on Line { time_role = differential; };
  field momentum: state scalar DG(order=0) on Line { time_role = differential; };
  field total_energy: state scalar DG(order=0) on Line { time_role = differential; };
  parameter gamma;
  constitutive velocity = momentum / density;
  constitutive pressure = (gamma - 1) * (total_energy - 0.5 * (momentum * momentum) / density);
  constitutive sound_speed = sqrt(gamma * pressure / density);
  constitutive momentum_flux = momentum * velocity + pressure;
  constitutive energy_flux = (total_energy + pressure) * velocity;
  constitutive speed_min = normal_component(velocity) - sound_speed;
  constitutive speed_max = normal_component(velocity) + sound_speed;
  constitutive mass_density = density;
  equation mass on Line { dt(density) + div(momentum) = 0; }
  equation momentum_balance on Line { dt(momentum) + div(momentum_flux) = 0; }
  equation energy on Line { dt(total_energy) + div(energy_flux) = 0; }
  boundary left_density on boundary("left") { dirichlet density = density; }
  boundary left_momentum on boundary("left") { dirichlet momentum = momentum; }
  boundary left_energy on boundary("left") { dirichlet total_energy = total_energy; }
  boundary right_density on boundary("right") { dirichlet density = density; }
  boundary right_momentum on boundary("right") { dirichlet momentum = momentum; }
  boundary right_energy on boundary("right") { dirichlet total_energy = total_energy; }
}

// Reflective (slip-wall) ends: the ghost mirrors the momentum.
model EulerWall1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field density: state scalar DG(order=0) on Line { time_role = differential; };
  field momentum: state scalar DG(order=0) on Line { time_role = differential; };
  field total_energy: state scalar DG(order=0) on Line { time_role = differential; };
  parameter gamma;
  constitutive velocity = momentum / density;
  constitutive pressure = (gamma - 1) * (total_energy - 0.5 * (momentum * momentum) / density);
  constitutive sound_speed = sqrt(gamma * pressure / density);
  constitutive momentum_flux = momentum * velocity + pressure;
  constitutive energy_flux = (total_energy + pressure) * velocity;
  constitutive speed_min = normal_component(velocity) - sound_speed;
  constitutive speed_max = normal_component(velocity) + sound_speed;
  constitutive mass_density = density;
  equation mass on Line { dt(density) + div(momentum) = 0; }
  equation momentum_balance on Line { dt(momentum) + div(momentum_flux) = 0; }
  equation energy on Line { dt(total_energy) + div(energy_flux) = 0; }
  boundary left_density on boundary("left") { dirichlet density = density; }
  boundary left_momentum on boundary("left") { dirichlet momentum = -momentum; }
  boundary left_energy on boundary("left") { dirichlet total_energy = total_energy; }
  boundary right_density on boundary("right") { dirichlet density = density; }
  boundary right_momentum on boundary("right") { dirichlet momentum = -momentum; }
  boundary right_energy on boundary("right") { dirichlet total_energy = total_energy; }
}

// Periodic square (no boundaries).
model Euler2D {
  domain Square { dimension = 2; coordinates = cartesian; }
  field density: state scalar DG(order=0) on Square { time_role = differential; };
  field momentum: state vector(2) DG(order=0) on Square { time_role = differential; };
  field total_energy: state scalar DG(order=0) on Square { time_role = differential; };
  parameter gamma;
  constitutive velocity = momentum / density;
  constitutive pressure = (gamma - 1) * (total_energy - 0.5 * dot(momentum, momentum) / density);
  constitutive sound_speed = sqrt(gamma * pressure / density);
  constitutive momentum_flux = outer(momentum, velocity) + pressure * identity(2);
  constitutive energy_flux = (total_energy + pressure) * velocity;
  constitutive speed_min = normal_component(velocity) - sound_speed;
  constitutive speed_max = normal_component(velocity) + sound_speed;
  constitutive mass_density = density;
  equation mass on Square { dt(density) + div(momentum) = 0; }
  equation momentum_balance on Square { dt(momentum) + div(momentum_flux) = 0; }
  equation energy on Square { dt(total_energy) + div(energy_flux) = 0; }
}

// A closed box with slip walls: the ghost momentum is m - 2 (m.n) n.
model EulerWall2D {
  domain Square { dimension = 2; coordinates = cartesian; }
  field density: state scalar DG(order=0) on Square { time_role = differential; };
  field momentum: state vector(2) DG(order=0) on Square { time_role = differential; };
  field total_energy: state scalar DG(order=0) on Square { time_role = differential; };
  parameter gamma;
  constitutive velocity = momentum / density;
  constitutive pressure = (gamma - 1) * (total_energy - 0.5 * dot(momentum, momentum) / density);
  constitutive sound_speed = sqrt(gamma * pressure / density);
  constitutive momentum_flux = outer(momentum, velocity) + pressure * identity(2);
  constitutive energy_flux = (total_energy + pressure) * velocity;
  constitutive speed_min = normal_component(velocity) - sound_speed;
  constitutive speed_max = normal_component(velocity) + sound_speed;
  constitutive mass_density = density;
  constitutive nvec = [normal_component([1, 0]), normal_component([0, 1])];
  constitutive mirrored = momentum - 2 * normal_component(momentum) * nvec;
  equation mass on Square { dt(density) + div(momentum) = 0; }
  equation momentum_balance on Square { dt(momentum) + div(momentum_flux) = 0; }
  equation energy on Square { dt(total_energy) + div(energy_flux) = 0; }
  boundary wall_density on boundary("wall") { dirichlet density = density; }
  boundary wall_momentum on boundary("wall") { dirichlet momentum = mirrored; }
  boundary wall_energy on boundary("wall") { dirichlet total_energy = total_energy; }
}
