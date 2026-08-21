use crate::{CellId, FinitumError, Mesh};

/// Concrete affine reference-to-physical map used by compatible finite-element pullbacks.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineMap {
    dimension: usize,
    origin: Vec<f64>,
    jacobian: Vec<f64>,
    inverse: Vec<f64>,
    determinant: f64,
}

impl AffineMap {
    pub fn from_cell(mesh: &Mesh, cell_id: CellId) -> Result<Self, FinitumError> {
        let cell = mesh.cell(cell_id).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("mesh has no cell {}", cell_id.0))
        })?;
        let dimension = mesh.dimension();
        let origin = mesh.vertices()[cell.vertices[0].0].clone();
        let mut jacobian = vec![0.0; dimension * dimension];
        for column in 0..dimension {
            let vertex = &mesh.vertices()[cell.vertices[column + 1].0];
            for row in 0..dimension {
                jacobian[row * dimension + column] = vertex[row] - origin[row];
            }
        }
        let (determinant, inverse) = invert(&jacobian, dimension).ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "cell {} has a singular affine geometry map",
                cell_id.0
            ))
        })?;
        if !determinant.is_finite() || inverse.iter().any(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(format!(
                "cell {} has an invalid affine geometry map",
                cell_id.0
            )));
        }
        Ok(Self {
            dimension,
            origin,
            jacobian,
            inverse,
            determinant,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn determinant(&self) -> f64 {
        self.determinant
    }

    pub fn volume_scale(&self) -> f64 {
        self.determinant.abs()
    }

    pub fn physical_point(&self, reference: &[f64]) -> Result<Vec<f64>, FinitumError> {
        self.vector_extent(reference)?;
        Ok((0..self.dimension)
            .map(|row| {
                self.origin[row]
                    + (0..self.dimension)
                        .map(|column| {
                            self.jacobian[row * self.dimension + column] * reference[column]
                        })
                        .sum::<f64>()
            })
            .collect())
    }

    /// H1 gradient / H(curl) covariant Piola map: `J^{-T} value`.
    pub fn covariant_piola(&self, reference: &[f64]) -> Result<Vec<f64>, FinitumError> {
        self.vector_extent(reference)?;
        Ok((0..self.dimension)
            .map(|physical| {
                (0..self.dimension)
                    .map(|reference_axis| {
                        self.inverse[reference_axis * self.dimension + physical]
                            * reference[reference_axis]
                    })
                    .sum()
            })
            .collect())
    }

    /// H(div) contravariant Piola map: `J value / det(J)`.
    pub fn contravariant_piola(&self, reference: &[f64]) -> Result<Vec<f64>, FinitumError> {
        self.vector_extent(reference)?;
        Ok((0..self.dimension)
            .map(|physical| {
                (0..self.dimension)
                    .map(|reference_axis| {
                        self.jacobian[physical * self.dimension + reference_axis]
                            * reference[reference_axis]
                    })
                    .sum::<f64>()
                    / self.determinant
            })
            .collect())
    }

    pub fn map_hcurl_curl(&self, reference: &[f64]) -> Result<Vec<f64>, FinitumError> {
        match (self.dimension, reference) {
            (2, [value]) => Ok(vec![value / self.determinant]),
            (3, values) if values.len() == 3 => Ok((0..3)
                .map(|physical| {
                    (0..3)
                        .map(|reference_axis| {
                            self.jacobian[physical * 3 + reference_axis] * values[reference_axis]
                        })
                        .sum::<f64>()
                        / self.determinant
                })
                .collect()),
            _ => Err(FinitumError::InvalidRealization(format!(
                "H(curl) curl data has extent {}, incompatible with dimension {}",
                reference.len(),
                self.dimension
            ))),
        }
    }

    pub fn map_hdiv_divergence(&self, reference: f64) -> f64 {
        reference / self.determinant
    }

    pub fn map_l2_density(&self, reference: f64) -> f64 {
        reference / self.volume_scale()
    }

    fn vector_extent(&self, values: &[f64]) -> Result<(), FinitumError> {
        if values.len() != self.dimension || values.iter().any(|value| !value.is_finite()) {
            Err(FinitumError::InvalidRealization(format!(
                "mapping vector has extent {}, expected {} finite components",
                values.len(),
                self.dimension
            )))
        } else {
            Ok(())
        }
    }
}

fn invert(matrix: &[f64], dimension: usize) -> Option<(f64, Vec<f64>)> {
    match dimension {
        1 => {
            let determinant = matrix[0];
            (determinant != 0.0).then(|| (determinant, vec![1.0 / determinant]))
        }
        2 => {
            let determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
            (determinant != 0.0).then(|| {
                (
                    determinant,
                    vec![
                        matrix[3] / determinant,
                        -matrix[1] / determinant,
                        -matrix[2] / determinant,
                        matrix[0] / determinant,
                    ],
                )
            })
        }
        3 => {
            let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
                - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
                + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
            (determinant != 0.0).then(|| {
                (
                    determinant,
                    vec![
                        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / determinant,
                        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / determinant,
                        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / determinant,
                        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / determinant,
                        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / determinant,
                        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) / determinant,
                        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / determinant,
                        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / determinant,
                        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / determinant,
                    ],
                )
            })
        }
        _ => None,
    }
}
