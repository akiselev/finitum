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

    /// P2 nodal simplex basis (vertex nodes plus edge-midpoint nodes) with a quadrature rule
    /// exact for the mass-matrix-shaped degree-4 integrand in dimension one and two; dimension
    /// three uses a degree-2-exact rule (see `tetrahedron_degree2_quadrature`), a documented
    /// reference-grade limit matching this crate's existing honesty about quadrature accuracy.
    ///
    /// Basis ordering is `(d+1)` vertex nodes in cell-local vertex order, followed by
    /// `(d+1)*d/2` edge nodes in the nested `(left, right)` pair order used throughout this
    /// crate for canonical edge enumeration (see `crate::topology` and
    /// [`crate::space::quadratic_simplex_dof_map`], which must agree with this ordering for the
    /// DOF map's local restriction to line up with these basis functions).
    pub fn quadratic_simplex(dimension: usize) -> Result<Self, FinitumError> {
        if !(1..=3).contains(&dimension) {
            return Err(FinitumError::InvalidDimension(dimension));
        }
        let basis_count = (dimension + 1) * (dimension + 2) / 2;
        let quadrature = match dimension {
            1 => gauss_legendre_unit_interval(3),
            2 => triangle_degree4_quadrature(),
            3 => tetrahedron_degree2_quadrature(),
            _ => unreachable!("dimension was checked"),
        };
        let mut basis_values = Vec::with_capacity(quadrature.len() * basis_count);
        let mut basis_gradients = Vec::with_capacity(quadrature.len() * basis_count * dimension);
        for point in &quadrature {
            let (values, gradients) = simplex_basis(dimension, 2, &point.coordinates)?;
            basis_values.extend(values);
            for gradient in gradients {
                basis_gradients.extend(gradient);
            }
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

/// Number of P1 (`order == 1`) or P2 (`order == 2`) Lagrange simplex basis functions in
/// `dimension`. Returns `0` for any other order; callers validate order before calling this.
pub(crate) fn simplex_basis_count(dimension: usize, order: u8) -> usize {
    match order {
        1 => dimension + 1,
        2 => (dimension + 1) * (dimension + 2) / 2,
        _ => 0,
    }
}

/// The shared quadrature rule `crate::mixed` evaluates every field's basis at, regardless of
/// that field's own order -- richest-available-for-`dimension` (see
/// [`PreparedElement::quadratic_simplex`]'s per-dimension rule selection), since a lower-order
/// polynomial is trivially integrated exactly by a higher-degree rule.
pub(crate) fn simplex_quadrature(dimension: usize) -> Result<Vec<QuadraturePoint>, FinitumError> {
    match dimension {
        1 => Ok(gauss_legendre_unit_interval(3)),
        2 => Ok(triangle_degree4_quadrature()),
        3 => Ok(tetrahedron_degree2_quadrature()),
        _ => Err(FinitumError::InvalidDimension(dimension)),
    }
}

/// P1 or P2 Lagrange simplex basis values and reference gradients at one explicit reference
/// point, independent of any precomputed quadrature table.
///
/// Barycentric convention matches [`PreparedElement::linear_simplex`]: `lambda[0] = 1 -
/// sum(point)`, `lambda[k] = point[k - 1]` for `k = 1..=dimension`, so `grad(lambda[0])` is `-1`
/// on every axis and `grad(lambda[k])` is the unit vector on axis `k - 1`. Order 2 appends edge
/// nodes `(left, right)` with `left < right` in nested-loop order after the `dimension + 1`
/// vertex nodes, using the standard quadratic simplex formulas `N_i = lambda_i (2 lambda_i - 1)`
/// and `N_{ij} = 4 lambda_i lambda_j`.
///
/// This is the single source of truth for simplex Lagrange basis math shared by
/// [`PreparedElement::quadratic_simplex`]'s own quadrature tabulation and by
/// `crate::mixed`'s cross-block coupling evaluation, which evaluates one field's basis at
/// another field's shared quadrature points. Exposed publicly for point evaluation/
/// post-processing and for testing basis identities (partition of unity, Kronecker-delta
/// nodality, gradient consistency) at points beyond a [`PreparedElement`]'s own tabulated
/// quadrature.
pub fn simplex_basis(
    dimension: usize,
    order: u8,
    point: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), FinitumError> {
    if !(1..=3).contains(&dimension) {
        return Err(FinitumError::InvalidDimension(dimension));
    }
    if point.len() != dimension || point.iter().any(|value| !value.is_finite()) {
        return Err(FinitumError::InvalidElementShape(format!(
            "reference point has dimension {}, expected {dimension} finite coordinates",
            point.len()
        )));
    }
    let vertex_count = dimension + 1;
    let mut lambda = Vec::with_capacity(vertex_count);
    lambda.push(1.0 - point.iter().sum::<f64>());
    lambda.extend_from_slice(point);
    let mut grad_lambda = Vec::with_capacity(vertex_count);
    grad_lambda.push(vec![-1.0; dimension]);
    for axis in 0..dimension {
        let mut gradient = vec![0.0; dimension];
        gradient[axis] = 1.0;
        grad_lambda.push(gradient);
    }
    match order {
        1 => Ok((lambda, grad_lambda)),
        2 => {
            let mut values = Vec::with_capacity(vertex_count + vertex_count * dimension / 2);
            let mut gradients = Vec::with_capacity(values.capacity());
            for i in 0..vertex_count {
                values.push(lambda[i] * (2.0 * lambda[i] - 1.0));
                gradients.push(
                    grad_lambda[i]
                        .iter()
                        .map(|component| (4.0 * lambda[i] - 1.0) * component)
                        .collect(),
                );
            }
            for left in 0..vertex_count {
                for right in left + 1..vertex_count {
                    values.push(4.0 * lambda[left] * lambda[right]);
                    gradients.push(
                        (0..dimension)
                            .map(|axis| {
                                4.0 * (lambda[right] * grad_lambda[left][axis]
                                    + lambda[left] * grad_lambda[right][axis])
                            })
                            .collect(),
                    );
                }
            }
            Ok((values, gradients))
        }
        _ => Err(FinitumError::InvalidElementShape(format!(
            "simplex Lagrange basis order must be 1 or 2, got {order}"
        ))),
    }
}

/// Six-point, degree-4-exact symmetric quadrature for the reference triangle `(0,0), (1,0),
/// (0,1)` (area `1/2`), sufficient to exactly integrate a P2 mass-matrix-shaped (degree-4)
/// integrand. Standard Dunavant/Strang-Fix constants.
fn triangle_degree4_quadrature() -> Vec<QuadraturePoint> {
    const AREA: f64 = 0.5;
    let groups = [
        (0.445948490915965_f64, 0.223381589678011_f64),
        (0.091576213509771_f64, 0.109951743655322_f64),
    ];
    let mut points = Vec::with_capacity(6);
    for (a, weight_fraction) in groups {
        let b = 1.0 - 2.0 * a;
        for coordinates in [[a, b], [b, a], [a, a]] {
            points.push(QuadraturePoint {
                coordinates: coordinates.to_vec(),
                weight: weight_fraction * AREA,
            });
        }
    }
    points
}

/// Four-point, degree-2-exact symmetric quadrature for the reference tetrahedron `(0,0,0),
/// (1,0,0), (0,1,0), (0,0,1)` (volume `1/6`). This under-integrates a P2 mass-matrix-shaped
/// (degree-4) integrand; it is a documented reference-grade limit, matching this crate's existing
/// honesty about quadrature accuracy (compare the P1 barycenter rule).
fn tetrahedron_degree2_quadrature() -> Vec<QuadraturePoint> {
    const VOLUME: f64 = 1.0 / 6.0;
    let a = (5.0 + 3.0 * 5.0_f64.sqrt()) / 20.0;
    let b = (5.0 - 5.0_f64.sqrt()) / 20.0;
    [[a, b, b], [b, a, b], [b, b, a], [b, b, b]]
        .into_iter()
        .map(|coordinates| QuadraturePoint {
            coordinates: coordinates.to_vec(),
            weight: 0.25 * VOLUME,
        })
        .collect()
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
