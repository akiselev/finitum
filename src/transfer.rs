use crate::FinitumError;
use serde::Serialize;

/// Deterministic one-dimensional nodal interpolation between nonmatching traces.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NonmatchingTransfer {
    source_nodes: Vec<f64>,
    target_nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl NonmatchingTransfer {
    /// Construct the Lagrange interpolation matrix from distinct source nodes.
    pub fn lagrange(
        source_nodes: impl Into<Vec<f64>>,
        target_nodes: impl Into<Vec<f64>>,
    ) -> Result<Self, FinitumError> {
        let source_nodes = source_nodes.into();
        let target_nodes = target_nodes.into();
        if source_nodes.is_empty() || target_nodes.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "nonmatching transfer needs nonempty source and target nodes".into(),
            ));
        }
        if source_nodes
            .iter()
            .chain(&target_nodes)
            .any(|node| !node.is_finite())
        {
            return Err(FinitumError::InvalidRealization(
                "nonmatching transfer nodes must be finite".into(),
            ));
        }
        for (index, node) in source_nodes.iter().enumerate() {
            if source_nodes[..index].contains(node) {
                return Err(FinitumError::InvalidRealization(
                    "nonmatching transfer source nodes must be distinct".into(),
                ));
            }
        }
        let mut weights = Vec::with_capacity(source_nodes.len() * target_nodes.len());
        for target in &target_nodes {
            for (source_index, source) in source_nodes.iter().enumerate() {
                let mut weight = 1.0;
                for (other_index, other) in source_nodes.iter().enumerate() {
                    if source_index != other_index {
                        weight *= (target - other) / (source - other);
                    }
                }
                weights.push(weight);
            }
        }
        Ok(Self {
            source_nodes,
            target_nodes,
            weights,
        })
    }

    pub fn source_nodes(&self) -> &[f64] {
        &self.source_nodes
    }

    pub fn target_nodes(&self) -> &[f64] {
        &self.target_nodes
    }

    pub fn apply(&self, source_values: &[f64]) -> Result<Vec<f64>, FinitumError> {
        validate_values("transfer source", source_values, self.source_nodes.len())?;
        Ok(self
            .weights
            .chunks_exact(self.source_nodes.len())
            .map(|row| {
                row.iter()
                    .zip(source_values)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect())
    }

    /// Apply the weighted transpose used by mortar residual scatter.
    pub fn apply_weighted_transpose(
        &self,
        target_values: &[f64],
        target_weights: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        validate_values("transfer target", target_values, self.target_nodes.len())?;
        validate_values("transfer weights", target_weights, self.target_nodes.len())?;
        let mut source = vec![0.0; self.source_nodes.len()];
        for (target, row) in self
            .weights
            .chunks_exact(self.source_nodes.len())
            .enumerate()
        {
            for (source_index, interpolation) in row.iter().enumerate() {
                source[source_index] +=
                    interpolation * target_weights[target] * target_values[target];
            }
        }
        Ok(source)
    }
}

/// Common mortar trace with independent interpolation from both nonmatching sides.
#[derive(Clone, Debug, PartialEq)]
pub struct MortarInterface {
    minus: NonmatchingTransfer,
    plus: NonmatchingTransfer,
    quadrature_weights: Vec<f64>,
}

impl MortarInterface {
    pub fn lagrange(
        minus_nodes: impl Into<Vec<f64>>,
        plus_nodes: impl Into<Vec<f64>>,
        mortar_nodes: impl Into<Vec<f64>>,
        quadrature_weights: impl Into<Vec<f64>>,
    ) -> Result<Self, FinitumError> {
        let mortar_nodes = mortar_nodes.into();
        let quadrature_weights = quadrature_weights.into();
        validate_values(
            "mortar quadrature weights",
            &quadrature_weights,
            mortar_nodes.len(),
        )?;
        if quadrature_weights.iter().any(|weight| *weight <= 0.0) {
            return Err(FinitumError::InvalidRealization(
                "mortar quadrature weights must be positive".into(),
            ));
        }
        Ok(Self {
            minus: NonmatchingTransfer::lagrange(minus_nodes, mortar_nodes.clone())?,
            plus: NonmatchingTransfer::lagrange(plus_nodes, mortar_nodes)?,
            quadrature_weights,
        })
    }

    pub fn traces(
        &self,
        minus_values: &[f64],
        plus_values: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), FinitumError> {
        Ok((
            self.minus.apply(minus_values)?,
            self.plus.apply(plus_values)?,
        ))
    }

    pub fn jump(
        &self,
        minus_values: &[f64],
        plus_values: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let (minus, plus) = self.traces(minus_values, plus_values)?;
        Ok(minus
            .into_iter()
            .zip(plus)
            .map(|(minus, plus)| minus - plus)
            .collect())
    }

    pub fn average(
        &self,
        minus_values: &[f64],
        plus_values: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let (minus, plus) = self.traces(minus_values, plus_values)?;
        Ok(minus
            .into_iter()
            .zip(plus)
            .map(|(minus, plus)| 0.5 * (minus + plus))
            .collect())
    }

    /// Scatter one oriented mortar flux to the two trace spaces with opposite signs.
    pub fn scatter_flux(&self, flux: &[f64]) -> Result<(Vec<f64>, Vec<f64>), FinitumError> {
        let minus = self
            .minus
            .apply_weighted_transpose(flux, &self.quadrature_weights)?;
        let mut plus = self
            .plus
            .apply_weighted_transpose(flux, &self.quadrature_weights)?;
        for value in &mut plus {
            *value = -*value;
        }
        Ok((minus, plus))
    }

    pub fn quadrature_weights(&self) -> &[f64] {
        &self.quadrature_weights
    }
}

fn validate_values(name: &str, values: &[f64], expected: usize) -> Result<(), FinitumError> {
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        return Err(FinitumError::InvalidRealization(format!(
            "{name} must contain {expected} finite values"
        )));
    }
    Ok(())
}

/// Piecewise P1 interpolation from a triangular surface in physical 3-D space.
/// This is a point-transfer artifact, not proof of complete interface coverage.
/// Targets outside the source surface, degenerate cells and ambiguous overlapping
/// traces refuse. The transpose transfers dual loads; primal interpolation alone
/// does not preserve a volume or surface integral.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SurfaceTransfer {
    source_vertices: Vec<[f64; 3]>,
    source_triangles: Vec<[usize; 3]>,
    target_points: Vec<[f64; 3]>,
    rows: Vec<Vec<(usize, f64)>>,
    tolerance: f64,
}
impl SurfaceTransfer {
    pub fn p1(
        source_vertices: Vec<[f64; 3]>,
        source_triangles: Vec<[usize; 3]>,
        target_points: Vec<[f64; 3]>,
        tolerance: f64,
    ) -> Result<Self, FinitumError> {
        let fail = |message: &str| {
            FinitumError::InvalidRealization(format!("surface transfer: {message}"))
        };
        if source_vertices.is_empty()
            || source_triangles.is_empty()
            || target_points.is_empty()
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || source_vertices
                .iter()
                .chain(&target_points)
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err(fail(
                "nonempty finite geometry and positive tolerance required",
            ));
        }
        let sub = |a: [f64; 3], b: [f64; 3]| std::array::from_fn::<_, 3, _>(|i| a[i] - b[i]);
        let dot = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        let mut geometry = Vec::new();
        for triangle in &source_triangles {
            if triangle.iter().any(|i| *i >= source_vertices.len()) {
                return Err(fail("triangle vertex out of range"));
            }
            let a = source_vertices[triangle[0]];
            let u = sub(source_vertices[triangle[1]], a);
            let v = sub(source_vertices[triangle[2]], a);
            let uu = dot(u, u);
            let uv = dot(u, v);
            let vv = dot(v, v);
            let det = uu * vv - uv * uv;
            if !det.is_finite() || det <= 1e-14 * uu * vv || uu == 0.0 || vv == 0.0 {
                return Err(fail("degenerate or ill-conditioned source triangle"));
            }
            geometry.push((a, u, v, uu, uv, vv, det));
        }
        let mut rows = Vec::new();
        for point in &target_points {
            let mut selected: Option<Vec<(usize, f64)>> = None;
            for (triangle, &(a, u, v, uu, uv, vv, det)) in source_triangles.iter().zip(&geometry) {
                let d = sub(*point, a);
                let du = dot(d, u);
                let dv = dot(d, v);
                let s = (vv * du - uv * dv) / det;
                let t = (uu * dv - uv * du) / det;
                if !s.is_finite() || !t.is_finite() {
                    return Err(fail("nonfinite barycentric coordinates"));
                }
                let projected = std::array::from_fn(|i| a[i] + s * u[i] + t * v[i]);
                let distance = sub(*point, projected);
                // Convert the declared physical tolerance into barycentric tolerances.
                let epsilon = tolerance * (uu.max(vv) / det).sqrt();
                if epsilon >= 1e-6 {
                    return Err(fail("tolerance is too large relative to a source triangle"));
                }
                if dot(distance, distance).sqrt() > tolerance
                    || s < -epsilon
                    || t < -epsilon
                    || s + t > 1.0 + epsilon
                {
                    continue;
                }
                let mut row: Vec<_> = triangle
                    .iter()
                    .copied()
                    .zip([1.0 - s - t, s, t])
                    .filter(|(_, w)| w.abs() > epsilon)
                    .collect();
                let sum: f64 = row.iter().map(|(_, w)| w).sum();
                for (_, w) in &mut row {
                    *w /= sum;
                }
                row.sort_by_key(|(i, _)| *i);
                if let Some(previous) = &selected {
                    if previous.len() != row.len()
                        || previous
                            .iter()
                            .zip(&row)
                            .any(|((a, x), (b, y))| a != b || (x - y).abs() > 1e-10)
                    {
                        return Err(fail("target has ambiguous overlapping source traces"));
                    }
                } else {
                    selected = Some(row);
                }
            }
            rows.push(selected.ok_or_else(|| fail("target is outside the source surface"))?);
        }
        Ok(Self {
            source_vertices,
            source_triangles,
            target_points,
            rows,
            tolerance,
        })
    }
    pub fn rows(&self) -> &[Vec<(usize, f64)>] {
        &self.rows
    }
    pub fn apply(&self, values: &[f64]) -> Result<Vec<f64>, FinitumError> {
        validate_values("surface source", values, self.source_vertices.len())?;
        let result: Vec<f64> = self
            .rows
            .iter()
            .map(|r| r.iter().map(|(i, w)| values[*i] * w).sum())
            .collect();
        validate_values("surface result", &result, self.target_points.len())?;
        Ok(result)
    }
    /// Dual action, satisfying `<P u, f> = <u, P^T f>`. With P preserving
    /// constants, the total dual load is conserved even on nonmatching traces.
    pub fn apply_transpose(&self, loads: &[f64]) -> Result<Vec<f64>, FinitumError> {
        validate_values("surface target loads", loads, self.target_points.len())?;
        let mut values = vec![0.0; self.source_vertices.len()];
        for (row, load) in self.rows.iter().zip(loads) {
            for (index, weight) in row {
                values[*index] += weight * load;
            }
        }
        validate_values(
            "surface transpose result",
            &values,
            self.source_vertices.len(),
        )?;
        Ok(values)
    }
}
