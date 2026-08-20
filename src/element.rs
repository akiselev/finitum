use crate::FinitumError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuadraturePoint {
    pub coordinates: Vec<f64>,
    pub weight: f64,
}

/// Concrete basis and quadrature data prepared for one reference element.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    pub fn new(
        dimension: usize,
        basis_count: usize,
        quadrature: Vec<QuadraturePoint>,
        basis_values: Vec<f64>,
        basis_gradients: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if dimension == 0 || basis_count == 0 {
            return Err(FinitumError::InvalidElementShape(
                "dimension and basis count must be non-zero".into(),
            ));
        }
        for point in &quadrature {
            if point.coordinates.len() != dimension {
                return Err(FinitumError::InvalidElementShape(format!(
                    "quadrature coordinate has dimension {}, expected {dimension}",
                    point.coordinates.len()
                )));
            }
        }
        let values = quadrature.len() * basis_count;
        let gradients = values * dimension;
        if basis_values.len() != values || basis_gradients.len() != gradients {
            return Err(FinitumError::InvalidElementShape(format!(
                "got {} values and {} gradients; expected {values} and {gradients}",
                basis_values.len(),
                basis_gradients.len()
            )));
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
