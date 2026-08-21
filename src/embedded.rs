use crate::{FinitumError, QuadraturePoint};

/// Concrete quadrature policy for a level-set embedded domain.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddedQuadraturePolicy {
    level_set_identity: String,
    quadrature_order: usize,
    minimum_volume_fraction: f64,
}

impl EmbeddedQuadraturePolicy {
    pub fn new(
        level_set_identity: impl Into<String>,
        quadrature_order: usize,
        minimum_volume_fraction: f64,
    ) -> Result<Self, FinitumError> {
        let level_set_identity = level_set_identity.into();
        if level_set_identity.trim().is_empty() {
            return Err(FinitumError::InvalidRealization(
                "embedded-domain level-set identity must not be empty".into(),
            ));
        }
        if !(1..=16).contains(&quadrature_order) {
            return Err(FinitumError::InvalidRealization(
                "embedded-domain quadrature order must be in 1..=16".into(),
            ));
        }
        if !minimum_volume_fraction.is_finite() || !(0.0..1.0).contains(&minimum_volume_fraction) {
            return Err(FinitumError::InvalidRealization(
                "embedded-domain minimum volume fraction must be finite and in [0, 1)".into(),
            ));
        }
        Ok(Self {
            level_set_identity,
            quadrature_order,
            minimum_volume_fraction,
        })
    }

    pub fn level_set_identity(&self) -> &str {
        &self.level_set_identity
    }

    pub fn quadrature_order(&self) -> usize {
        self.quadrature_order
    }

    pub fn minimum_volume_fraction(&self) -> f64 {
        self.minimum_volume_fraction
    }
}

/// Active quadrature for a segment clipped by a linearly interpolated level set.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddedSegmentQuadrature {
    active_interval: Option<[f64; 2]>,
    interface_coordinate: Option<f64>,
    points: Vec<QuadraturePoint>,
}

impl EmbeddedSegmentQuadrature {
    /// Keep the subinterval where the linearly interpolated level set is non-positive.
    pub fn from_linear_level_set(
        endpoints: [f64; 2],
        level_set: [f64; 2],
        policy: &EmbeddedQuadraturePolicy,
    ) -> Result<Self, FinitumError> {
        if endpoints
            .iter()
            .chain(&level_set)
            .any(|value| !value.is_finite())
            || endpoints[1] <= endpoints[0]
        {
            return Err(FinitumError::InvalidRealization(
                "embedded segment needs finite increasing endpoints and finite level-set values"
                    .into(),
            ));
        }
        let (active_interval, interface_coordinate) =
            match (level_set[0] <= 0.0, level_set[1] <= 0.0) {
                (true, true) => (Some(endpoints), None),
                (false, false) => (None, None),
                (left_active, _) => {
                    let fraction = level_set[0] / (level_set[0] - level_set[1]);
                    let interface = endpoints[0] + fraction * (endpoints[1] - endpoints[0]);
                    let interval = if left_active {
                        [endpoints[0], interface]
                    } else {
                        [interface, endpoints[1]]
                    };
                    (Some(interval), Some(interface))
                }
            };
        let mut points = Vec::new();
        if let Some(interval) = active_interval {
            let fraction = (interval[1] - interval[0]) / (endpoints[1] - endpoints[0]);
            if fraction >= policy.minimum_volume_fraction() {
                points = gauss_legendre_interval(policy.quadrature_order(), interval);
            }
        }
        Ok(Self {
            active_interval,
            interface_coordinate,
            points,
        })
    }

    pub fn active_interval(&self) -> Option<[f64; 2]> {
        self.active_interval
    }

    pub fn interface_coordinate(&self) -> Option<f64> {
        self.interface_coordinate
    }

    pub fn points(&self) -> &[QuadraturePoint] {
        &self.points
    }

    pub fn active_measure(&self) -> f64 {
        self.points.iter().map(|point| point.weight).sum()
    }
}

fn gauss_legendre_interval(count: usize, interval: [f64; 2]) -> Vec<QuadraturePoint> {
    let center = 0.5 * (interval[0] + interval[1]);
    let radius = 0.5 * (interval[1] - interval[0]);
    let mut points = Vec::with_capacity(count);
    for root in 0..count.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (root as f64 + 0.75) / (count as f64 + 0.5)).cos();
        let derivative = loop {
            let mut previous = 1.0;
            let mut current = z;
            for degree in 2..=count {
                let next = ((2 * degree - 1) as f64 * z * current - (degree - 1) as f64 * previous)
                    / degree as f64;
                previous = current;
                current = next;
            }
            let derivative = count as f64 * (z * current - previous) / (z * z - 1.0);
            let next = z - current / derivative;
            if (next - z).abs() <= 4.0 * f64::EPSILON {
                z = next;
                break derivative;
            }
            z = next;
        };
        let weight = 2.0 * radius / ((1.0 - z * z) * derivative * derivative);
        points.push(QuadraturePoint {
            coordinates: vec![center - radius * z],
            weight,
        });
        if points.len() < count {
            points.push(QuadraturePoint {
                coordinates: vec![center + radius * z],
                weight,
            });
        }
    }
    points.sort_by(|left, right| left.coordinates[0].total_cmp(&right.coordinates[0]));
    points
}
