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
    /// P1 simplex basis with barycenter quadrature: exact for the affine stiffness integrand
    /// and first-degree loads in dimensions one through three, and the rule every authored
    /// per-quadrature-point external table (R3D geometry sensitivities, Sinbad's campaigns)
    /// is sized against -- one point per cell. It under-integrates the P1 mass matrix to a
    /// rank-one local block (GX-CONTRACTS C11.8); a caller realizing a mass-shaped integrand
    /// (`dt(u)`, reaction terms) selects [`Self::linear_simplex_with_degree`]`(dimension, 2)`
    /// instead. The Scientia-system path (`SystemRealizationPlan`) does not use this rule.
    pub fn linear_simplex(dimension: usize) -> Result<Self, FinitumError> {
        Self::linear_simplex_with_degree(dimension, 1)
    }

    /// P1 simplex basis tabulated on the smallest rule of this crate exact for polynomial
    /// `degree`: `0 | 1` is the barycenter rule of [`Self::linear_simplex`]; `2` is 2-point
    /// Gauss on the segment, the three edge midpoints on the triangle, and the symmetric
    /// 4-point rule on the tetrahedron -- exact for the P1 mass matrix and for products of two
    /// P1 quantities (C11.8). Higher degrees are refused typed rather than silently
    /// under-integrated (FC3's `minimum_polynomial_degree` above two is not selected for).
    pub fn linear_simplex_with_degree(dimension: usize, degree: u16) -> Result<Self, FinitumError> {
        if !(1..=3).contains(&dimension) {
            return Err(FinitumError::InvalidDimension(dimension));
        }
        let basis_count = dimension + 1;
        let quadrature = match (degree, dimension) {
            (0 | 1, _) => {
                let weight = match dimension {
                    1 => 1.0,
                    2 => 0.5,
                    3 => 1.0 / 6.0,
                    _ => unreachable!("dimension was checked"),
                };
                vec![QuadraturePoint {
                    coordinates: vec![1.0 / basis_count as f64; dimension],
                    weight,
                }]
            }
            (2, 1) => gauss_legendre_unit_interval(2),
            (2, 2) => [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
                .into_iter()
                .map(|coordinates| QuadraturePoint {
                    coordinates: coordinates.to_vec(),
                    weight: 1.0 / 6.0,
                })
                .collect(),
            (2, 3) => tetrahedron_degree2_quadrature(),
            (degree, _) => {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "P1 simplex quadrature is realized for polynomial degree 0, 1, or 2, \
                     got {degree}"
                )));
            }
        };
        let mut basis_values = Vec::with_capacity(quadrature.len() * basis_count);
        let mut basis_gradients = Vec::with_capacity(quadrature.len() * basis_count * dimension);
        for point in &quadrature {
            let (values, gradients) = simplex_basis(dimension, 1, &point.coordinates)?;
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

/// Number of RT0 (lowest-order Raviart-Thomas) facet-based basis functions on a `dimension`-
/// simplex: exactly one per facet, i.e. `dimension + 1`.
pub(crate) fn rt0_basis_count(dimension: usize) -> usize {
    dimension + 1
}

/// RT0 reference-simplex basis: one facet-based vector basis function per local facet (the
/// facet obtained by omitting local vertex `i`, matching `crate::topology`'s "omitted vertex"
/// facet convention exactly -- the same convention [`crate::topology::CompatibleDofMaps::hdiv`]
/// uses to build its per-cell restrictions/orientations), evaluated at one reference point.
///
/// Uses the standard construction `phi_i(x) = x - p_i`, where `p_i` is reference vertex `i`
/// (`p_0` is the origin, `p_k` is the `k`-th standard basis vector for `k = 1..=dimension`).
/// This satisfies `integral_{F_i} phi_i . n_i = d |K_ref| = 1 / (d - 1)!` (`1` on the triangle,
/// `1/2` on the tetrahedron) for the reference simplex's own outward normal at facet `i`, and
/// zero flux through every other facet -- verified directly (not merely asserted) by this
/// module's own triangle test, and used by `crate::system`'s RT0 essential normal-trace data
/// (C11.22) to convert a facet flux into a DOF value. The reference divergence `div(phi_i) =
/// dimension` is the same constant for every `i` (RT0's basis functions differ only in which
/// vertex is subtracted, and `d/dx_k(x_k - p_i,k) = 1` regardless of `p_i`), so it is returned
/// once rather than per basis function.
///
/// Returns `(values, divergence)`: `values[i]` is `phi_i(point)` (length `dimension`);
/// `divergence` is the shared constant. Piola pushforward (`crate::mapping::AffineMap::
/// contravariant_piola`/`map_hdiv_divergence`) maps these to physical space; per-cell DOF
/// orientation correction is the caller's responsibility (this function is purely reference-
/// space, mesh- and cell-independent).
pub(crate) fn rt0_reference_basis(
    dimension: usize,
    point: &[f64],
) -> Result<(Vec<Vec<f64>>, f64), FinitumError> {
    if !(2..=3).contains(&dimension) {
        return Err(FinitumError::InvalidDimension(dimension));
    }
    if point.len() != dimension || point.iter().any(|value| !value.is_finite()) {
        return Err(FinitumError::InvalidElementShape(format!(
            "reference point has dimension {}, expected {dimension} finite coordinates",
            point.len()
        )));
    }
    let vertex_count = dimension + 1;
    let mut vertices = Vec::with_capacity(vertex_count);
    vertices.push(vec![0.0; dimension]);
    for axis in 0..dimension {
        let mut vertex = vec![0.0; dimension];
        vertex[axis] = 1.0;
        vertices.push(vertex);
    }
    let values = vertices
        .iter()
        .map(|vertex| {
            point
                .iter()
                .zip(vertex)
                .map(|(coordinate, origin)| coordinate - origin)
                .collect::<Vec<_>>()
        })
        .collect();
    Ok((values, dimension as f64))
}

/// Six-point, degree-4-exact symmetric quadrature for the reference triangle `(0,0), (1,0),
/// (0,1)` (area `1/2`), sufficient to exactly integrate a P2 mass-matrix-shaped (degree-4)
/// integrand. Standard Dunavant/Strang-Fix constants.
pub(crate) fn triangle_degree4_quadrature() -> Vec<QuadraturePoint> {
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

pub(crate) fn gauss_legendre_unit_interval(count: usize) -> Vec<QuadraturePoint> {
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

#[cfg(test)]
mod p1_mass_tests {
    use super::*;

    fn reference_mass(element: &PreparedElement) -> Vec<f64> {
        let n = element.basis_count();
        let mut mass = vec![0.0; n * n];
        for (point, quadrature) in element.quadrature().iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    mass[i * n + j] += quadrature.weight
                        * element.basis_value(point, i).unwrap()
                        * element.basis_value(point, j).unwrap();
                }
            }
        }
        mass
    }

    /// The reference P1 mass matrix is `|K| (1 + delta_ij) / ((d + 1)(d + 2))`; the barycenter
    /// rule gives `|K| / (d + 1)^2` everywhere (rank one). C11.8.
    #[test]
    fn linear_simplex_degree_two_integrates_the_p1_mass_matrix_exactly() {
        for dimension in 1..=3usize {
            let barycenter = PreparedElement::linear_simplex(dimension).unwrap();
            assert_eq!(barycenter.quadrature().len(), 1);
            let n = dimension + 1;
            let expected_volume = 1.0 / (1..=dimension).product::<usize>() as f64;
            for value in reference_mass(&barycenter) {
                assert!((value - expected_volume / (n * n) as f64).abs() < 1.0e-15);
            }
            assert!(matches!(
                PreparedElement::linear_simplex_with_degree(dimension, 3),
                Err(FinitumError::UnsupportedRealization(_))
            ));

            let element = PreparedElement::linear_simplex_with_degree(dimension, 2).unwrap();
            assert!(element.quadrature().len() > 1);
            let volume = element
                .quadrature()
                .iter()
                .map(|point| point.weight)
                .sum::<f64>();
            assert!((volume - expected_volume).abs() < 1.0e-15);
            let mass = reference_mass(&element);
            let denominator = ((dimension + 1) * (dimension + 2)) as f64;
            for i in 0..n {
                for j in 0..n {
                    let expected = expected_volume * if i == j { 2.0 } else { 1.0 } / denominator;
                    assert!(
                        (mass[i * n + j] - expected).abs() < 1.0e-15,
                        "dimension {dimension} mass ({i}, {j}) = {} != {expected}",
                        mass[i * n + j]
                    );
                }
            }
            // Partition of unity and constant gradients at every point.
            for point in 0..element.quadrature().len() {
                let sum = (0..n)
                    .map(|basis| element.basis_value(point, basis).unwrap())
                    .sum::<f64>();
                assert!((sum - 1.0).abs() < 1.0e-15);
                for axis in 0..dimension {
                    assert_eq!(element.basis_gradient(point, 0).unwrap()[axis], -1.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    /// Independent reference-normal computation for the standard reference triangle
    /// `(0,0),(1,0),(0,1)`: facet `i` is the facet opposite vertex `i`, matching this module's
    /// own "omitted vertex" convention. Outward unit normals (unnormalized here, scaled by the
    /// facet's own reference length) are hand-derived, not shared with `rt0_reference_basis`'s
    /// own implementation.
    fn triangle_facet_normal_and_length(facet: usize) -> ([f64; 2], f64) {
        match facet {
            0 => (
                [
                    1.0 / std::f64::consts::SQRT_2,
                    1.0 / std::f64::consts::SQRT_2,
                ],
                std::f64::consts::SQRT_2,
            ),
            1 => ([-1.0, 0.0], 1.0),
            2 => ([0.0, -1.0], 1.0),
            _ => unreachable!(),
        }
    }

    #[test]
    fn rt0_reference_basis_reproduces_the_defining_flux_biorthogonality_on_the_triangle() {
        // phi_i . n_i is constant along facet i (RT0's order-0 normal trace); sample at the
        // facet's own midpoint and confirm the flux integral (constant * length) is exactly 1
        // for the owning facet and exactly 0 for the other two, for every facet i.
        let midpoints = [
            [0.5, 0.5], // facet 0 (hypotenuse) midpoint
            [0.0, 0.5], // facet 1 (x=0) midpoint
            [0.5, 0.0], // facet 2 (y=0) midpoint
        ];
        for (facet, midpoint) in midpoints.iter().enumerate() {
            let (values, divergence) = rt0_reference_basis(2, midpoint).unwrap();
            assert_eq!(divergence, 2.0);
            for (basis_index, basis_value) in values.iter().enumerate() {
                let (normal, length) = triangle_facet_normal_and_length(facet);
                let flux = dot(basis_value, &normal) * length;
                let expected = if basis_index == facet { 1.0 } else { 0.0 };
                assert!(
                    (flux - expected).abs() < 1e-12,
                    "facet {facet} basis {basis_index}: flux {flux}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rt0_reference_basis_divergence_matches_a_finite_difference_of_the_values() {
        let point = [0.2, 0.3];
        let h = 1e-6;
        let (values, divergence) = rt0_reference_basis(2, &point).unwrap();
        for basis in &values {
            // div(phi) = d(phi_x)/dx + d(phi_y)/dy; phi_i(x,y) = (x,y) - p_i is affine, so a
            // centered difference is exact up to floating-point roundoff.
            let (plus_x, _) = rt0_reference_basis(2, &[point[0] + h, point[1]]).unwrap();
            let (minus_x, _) = rt0_reference_basis(2, &[point[0] - h, point[1]]).unwrap();
            let (plus_y, _) = rt0_reference_basis(2, &[point[0], point[1] + h]).unwrap();
            let (minus_y, _) = rt0_reference_basis(2, &[point[0], point[1] - h]).unwrap();
            let index = values
                .iter()
                .position(|candidate| candidate == basis)
                .unwrap();
            let d_dx = (plus_x[index][0] - minus_x[index][0]) / (2.0 * h);
            let d_dy = (plus_y[index][1] - minus_y[index][1]) / (2.0 * h);
            assert!((d_dx + d_dy - divergence).abs() < 1e-6);
        }
    }

    #[test]
    fn rt0_reference_basis_refuses_wrong_dimension_and_nonfinite_points() {
        assert!(rt0_reference_basis(1, &[0.5]).is_err());
        assert!(rt0_reference_basis(4, &[0.1, 0.1, 0.1, 0.1]).is_err());
        assert!(rt0_reference_basis(2, &[0.1, f64::NAN]).is_err());
        assert!(rt0_reference_basis(2, &[0.1]).is_err());
    }
}
