module systems.electrothermal;
use physics.electrical.{ElectricalConduction};
use physics.thermal as thermal;

// SC-W1 composition fixture (Sinbad-owned, `ARCHITECTURE.md` §3.5/§3.8 in today's grammar):
// two instances on one domain, each closing the other's open input by `bind`. Compiled by
// `cases/electrothermal-system.toml` (`sinbad-case/2`, `kind = "system"`); its slots are
// `electrical/...` and `thermal/...`.
pub system Electrothermal {
    domain body { dimension = 2; coordinates = cartesian; }
    instance electrical: ElectricalConduction(conductor = body);
    instance thermal: thermal.HeatConduction(body = body);
    bind electrical.temperature <- thermal.temperature;
    bind thermal.Q <- electrical.joule_heat;
}
