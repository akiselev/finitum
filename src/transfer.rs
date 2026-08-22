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
