use crate::FinitumError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuadraturePoint {
    pub coordinates: Vec<f64>,
    pub weight: f64,
}

/// Concrete basis and quadrature data prepared for one reference element.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreparedElement {
    dimension: usize,
    basis_count: usize,
    quadrature: Vec<QuadraturePoint>,
    /// Quadrature-major table: `basis_values[q * basis_count + basis]`.
    basis_values: Vec<f64>,
    /// Quadrature/basis/dimension-major gradients.
    basis_gradients: Vec<f64>,
}

impl PreparedElement {
    /// P1 simplex basis with barycenter quadrature. This rule is exact for the affine stiffness
    /// integrand and first-degree loads in dimensions one through three.
    pub fn linear_simplex(dimension: usize) -> Result<Self, FinitumError> {
        if !(1..=3).contains(&dimension) {
            return Err(FinitumError::InvalidDimension(dimension));
        }
        let basis_count = dimension + 1;
        let weight = match dimension {
            1 => 1.0,
            2 => 0.5,
            3 => 1.0 / 6.0,
            _ => unreachable!("dimension was checked"),
        };
        let quadrature = vec![QuadraturePoint {
            coordinates: vec![1.0 / basis_count as f64; dimension],
            weight,
        }];
        let basis_values = vec![1.0 / basis_count as f64; basis_count];
        let mut basis_gradients = vec![0.0; basis_count * dimension];
        for axis in 0..dimension {
            basis_gradients[axis] = -1.0;
            basis_gradients[(axis + 1) * dimension + axis] = 1.0;
        }
        Self::new(
            dimension,
            basis_count,
            quadrature,
            basis_values,
            basis_gradients,
        )
    }

    /// Nodal Lagrange segment of the requested order with Gauss-Legendre quadrature.
    ///
    /// The interpolation nodes are equispaced. This is a deterministic reference table, not a
    /// well-conditioned high-order production basis; later high-order realizations should use a
    /// stable nodal family rather than extending this constructor's order limit.
    pub fn lagrange_segment(order: usize) -> Result<Self, FinitumError> {
        if !(1..=16).contains(&order) {
            return Err(FinitumError::InvalidElementShape(format!(
                "segment polynomial order must be in 1..=16, got {order}"
            )));
        }
        let basis_count = order + 1;
        let nodes = (0..=order)
            .map(|index| index as f64 / order as f64)
            .collect::<Vec<_>>();
        let quadrature = gauss_legendre_unit_interval(basis_count);
        let mut basis_values = Vec::with_capacity(basis_count * basis_count);
        let mut basis_gradients = Vec::with_capacity(basis_count * basis_count);
        for point in &quadrature {
            let x = point.coordinates[0];
            for (basis, node) in nodes.iter().enumerate() {
                let mut value = 1.0;
                for (other, other_node) in nodes.iter().enumerate() {
                    if other != basis {
                        value *= (x - other_node) / (node - other_node);
                    }
                }
                let mut derivative = 0.0;
                for omitted in 0..basis_count {
                    if omitted == basis {
                        continue;
                    }
                    let mut term = 1.0 / (node - nodes[omitted]);
                    for (other, other_node) in nodes.iter().enumerate() {
                        if other != basis && other != omitted {
                            term *= (x - other_node) / (node - other_node);
                        }
                    }
                    derivative += term;
                }
                basis_values.push(value);
                basis_gradients.push(derivative);
            }
        }
        Self::new(1, basis_count, quadrature, basis_values, basis_gradients)
    }

    pub fn new(
        dimension: usize,
        basis_count: usize,
        quadrature: Vec<QuadraturePoint>,
        basis_values: Vec<f64>,
        basis_gradients: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if !(1..=3).contains(&dimension) || basis_count == 0 || quadrature.is_empty() {
            return Err(FinitumError::InvalidElementShape(
                "dimension must be in 1..=3; basis and quadrature counts must be non-zero".into(),
            ));
        }
        for (point_index, point) in quadrature.iter().enumerate() {
            if point.coordinates.len() != dimension {
                return Err(FinitumError::InvalidElementShape(format!(
                    "quadrature coordinate has dimension {}, expected {dimension}",
                    point.coordinates.len()
                )));
            }
            if !point.weight.is_finite() || point.coordinates.iter().any(|value| !value.is_finite())
            {
                return Err(FinitumError::NonFiniteElementData {
                    location: format!("quadrature point {point_index}"),
                });
            }
        }
        let values = quadrature.len().checked_mul(basis_count).ok_or_else(|| {
            FinitumError::InvalidElementShape("basis table extent overflows usize".into())
        })?;
        let gradients = values.checked_mul(dimension).ok_or_else(|| {
            FinitumError::InvalidElementShape("gradient table extent overflows usize".into())
        })?;
        if basis_values.len() != values || basis_gradients.len() != gradients {
            return Err(FinitumError::InvalidElementShape(format!(
                "got {} values and {} gradients; expected {values} and {gradients}",
                basis_values.len(),
                basis_gradients.len()
            )));
        }
        if let Some(index) = basis_values.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::NonFiniteElementData {
                location: format!("basis value {index}"),
            });
        }
        if let Some(index) = basis_gradients.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::NonFiniteElementData {
                location: format!("basis gradient {index}"),
            });
        }
        Ok(Self {
            dimension,
            basis_count,
            quadrature,
            basis_values,
            basis_gradients,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn basis_count(&self) -> usize {
        self.basis_count
    }

    pub fn quadrature(&self) -> &[QuadraturePoint] {
        &self.quadrature
    }

    pub fn basis_value(&self, point: usize, basis: usize) -> Option<f64> {
        (point < self.quadrature.len() && basis < self.basis_count)
            .then(|| self.basis_values[point * self.basis_count + basis])
    }

    pub fn basis_gradient(&self, point: usize, basis: usize) -> Option<&[f64]> {
        if point >= self.quadrature.len() || basis >= self.basis_count {
            return None;
        }
        let start = (point * self.basis_count + basis) * self.dimension;
        Some(&self.basis_gradients[start..start + self.dimension])
    }
}

fn gauss_legendre_unit_interval(count: usize) -> Vec<QuadraturePoint> {
    let mut points = Vec::with_capacity(count);
    let half = count.div_ceil(2);
    for root in 0..half {
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
        let weight = 1.0 / ((1.0 - z * z) * derivative * derivative);
        points.push(QuadraturePoint {
            coordinates: vec![0.5 * (1.0 - z)],
            weight,
        });
        if points.len() < count {
            points.push(QuadraturePoint {
                coordinates: vec![0.5 * (1.0 + z)],
                weight,
            });
        }
    }
    points.sort_by(|left, right| left.coordinates[0].total_cmp(&right.coordinates[0]));
    points
}
