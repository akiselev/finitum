use crate::realization::{CellGeometry, apply_basis_adjoint, evaluate_basis_input};
use crate::{ConstraintSet, ElementRestriction, FinitumError, PreparedElement};
use resolvent::{DerivativeEvaluation, Digest, QFunctionInput};
use solverang::{EvaluationContext, LinearOperator, NumericError, OperatorSymmetry};

/// Fixed-width cell batches with explicit inactive padding lanes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellBatchLayout {
    cell_count: usize,
    lane_width: usize,
    lanes: Vec<Option<usize>>,
}

impl CellBatchLayout {
    pub fn new(cell_count: usize, lane_width: usize) -> Result<Self, FinitumError> {
        if cell_count == 0 || lane_width == 0 {
            return Err(FinitumError::InvalidRealization(
                "cell batches need nonzero cell and lane counts".into(),
            ));
        }
        let batch_count = cell_count.div_ceil(lane_width);
        let lane_count = batch_count.checked_mul(lane_width).ok_or_else(|| {
            FinitumError::InvalidRealization("cell-batch extent overflows usize".into())
        })?;
        let lanes = (0..lane_count)
            .map(|lane| (lane < cell_count).then_some(lane))
            .collect();
        Ok(Self {
            cell_count,
            lane_width,
            lanes,
        })
    }

    pub fn cell_count(&self) -> usize {
        self.cell_count
    }

    pub fn lane_width(&self) -> usize {
        self.lane_width
    }

    pub fn batch_count(&self) -> usize {
        self.lanes.len() / self.lane_width
    }

    pub fn batch(&self, index: usize) -> Option<&[Option<usize>]> {
        (index < self.batch_count()).then(|| {
            let start = index * self.lane_width;
            &self.lanes[start..start + self.lane_width]
        })
    }
}

/// Component-major, lane-interleaved packing for accelerator-friendly batches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceleratorLayout {
    entity_count: usize,
    component_count: usize,
    lane_width: usize,
    value_count: usize,
    packed_len: usize,
}

impl AcceleratorLayout {
    pub fn new(
        entity_count: usize,
        component_count: usize,
        lane_width: usize,
    ) -> Result<Self, FinitumError> {
        if entity_count == 0 || component_count == 0 || lane_width == 0 {
            return Err(FinitumError::InvalidRealization(
                "accelerator layout extents and lane width must be nonzero".into(),
            ));
        }
        let value_count = entity_count.checked_mul(component_count).ok_or_else(|| {
            FinitumError::InvalidRealization("accelerator value extent overflows usize".into())
        })?;
        let padded_entities = entity_count
            .div_ceil(lane_width)
            .checked_mul(lane_width)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(
                    "accelerator padded entity extent overflows usize".into(),
                )
            })?;
        let packed_len = padded_entities
            .checked_mul(component_count)
            .ok_or_else(|| {
                FinitumError::InvalidRealization("accelerator packed extent overflows usize".into())
            })?;
        Ok(Self {
            entity_count,
            component_count,
            lane_width,
            value_count,
            packed_len,
        })
    }

    pub fn packed_len(&self) -> usize {
        self.packed_len
    }

    /// Pack entity-major values as batch/component/lane, zeroing inactive lanes.
    pub fn pack(&self, entity_major: &[f64]) -> Result<Vec<f64>, FinitumError> {
        validate_finite_length("accelerator input", entity_major, self.value_count)?;
        let mut packed = vec![0.0; self.packed_len()];
        for entity in 0..self.entity_count {
            let batch = entity / self.lane_width;
            let lane = entity % self.lane_width;
            for component in 0..self.component_count {
                let packed_index =
                    (batch * self.component_count + component) * self.lane_width + lane;
                packed[packed_index] = entity_major[entity * self.component_count + component];
            }
        }
        Ok(packed)
    }

    pub fn unpack(&self, packed: &[f64]) -> Result<Vec<f64>, FinitumError> {
        validate_finite_length("accelerator packed input", packed, self.packed_len())?;
        let mut entity_major = vec![0.0; self.value_count];
        for entity in 0..self.entity_count {
            let batch = entity / self.lane_width;
            let lane = entity % self.lane_width;
            for component in 0..self.component_count {
                let packed_index =
                    (batch * self.component_count + component) * self.lane_width + lane;
                entity_major[entity * self.component_count + component] = packed[packed_index];
            }
        }
        Ok(entity_major)
    }
}

/// Values and reference gradients produced by tensor-product basis application.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorProductEvaluation {
    pub values: Vec<f64>,
    /// Point-major gradients: `gradients[point * dimension + axis]`.
    pub gradients: Vec<f64>,
}

/// Reusable one-dimensional tables applied axis-by-axis (sum factorization).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorProductBasis {
    dimension: usize,
    node_count: usize,
    point_count: usize,
    interpolation: Vec<f64>,
    derivative: Vec<f64>,
}

impl TensorProductBasis {
    pub fn new(
        dimension: usize,
        node_count: usize,
        point_count: usize,
        interpolation: Vec<f64>,
        derivative: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if !(1..=3).contains(&dimension) || node_count == 0 || point_count == 0 {
            return Err(FinitumError::InvalidRealization(
                "tensor basis needs dimension 1..=3 and nonzero extents".into(),
            ));
        }
        let expected = node_count.checked_mul(point_count).ok_or_else(|| {
            FinitumError::InvalidRealization("tensor basis table extent overflows usize".into())
        })?;
        validate_finite_length("interpolation table", &interpolation, expected)?;
        validate_finite_length("derivative table", &derivative, expected)?;
        Ok(Self {
            dimension,
            node_count,
            point_count,
            interpolation,
            derivative,
        })
    }

    pub fn evaluate(&self, nodal_values: &[f64]) -> Result<TensorProductEvaluation, FinitumError> {
        let nodal_count = checked_power(self.node_count, self.dimension)?;
        validate_finite_length("tensor nodal values", nodal_values, nodal_count)?;
        let values = self.apply_tables(nodal_values, None)?;
        let point_count = checked_power(self.point_count, self.dimension)?;
        let gradient_count = point_count.checked_mul(self.dimension).ok_or_else(|| {
            FinitumError::InvalidRealization("tensor gradient extent overflows usize".into())
        })?;
        let mut gradients = vec![0.0; gradient_count];
        for axis in 0..self.dimension {
            let derivative = self.apply_tables(nodal_values, Some(axis))?;
            for (point, value) in derivative.into_iter().enumerate() {
                gradients[point * self.dimension + axis] = value;
            }
        }
        Ok(TensorProductEvaluation { values, gradients })
    }

    fn apply_tables(
        &self,
        nodal_values: &[f64],
        derivative_axis: Option<usize>,
    ) -> Result<Vec<f64>, FinitumError> {
        let mut values = nodal_values.to_vec();
        let mut extents = vec![self.node_count; self.dimension];
        for axis in 0..self.dimension {
            let table = if derivative_axis == Some(axis) {
                &self.derivative
            } else {
                &self.interpolation
            };
            values = contract_axis(&values, &extents, axis, self.point_count, table)?;
            extents[axis] = self.point_count;
        }
        Ok(values)
    }
}

/// Element-assembled realization: dense cell matrices are stored, but no global matrix is formed.
/// Affine dependency constraint rows make the resulting full-coordinate action nonsymmetric.
#[derive(Clone, Debug)]
pub struct ElementAssemblyOperator {
    dimension: usize,
    source_factorization_digest: Digest,
    restrictions: Vec<ElementRestriction>,
    constraints: ConstraintSet,
    local_matrices: Vec<Vec<f64>>,
    batches: CellBatchLayout,
}

impl ElementAssemblyOperator {
    pub(crate) fn new(
        dimension: usize,
        source_factorization_digest: Digest,
        restrictions: Vec<ElementRestriction>,
        constraints: ConstraintSet,
        local_matrices: Vec<Vec<f64>>,
        lane_width: usize,
    ) -> Result<Self, FinitumError> {
        let batches = CellBatchLayout::new(restrictions.len(), lane_width)?;
        Ok(Self {
            dimension,
            source_factorization_digest,
            restrictions,
            constraints,
            local_matrices,
            batches,
        })
    }

    pub fn source_factorization_digest(&self) -> &Digest {
        &self.source_factorization_digest
    }

    pub fn batches(&self) -> &CellBatchLayout {
        &self.batches
    }

    pub fn local_matrices(&self) -> &[Vec<f64>] {
        &self.local_matrices
    }

    fn apply_inner(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        validate_finite_length("element-assembly input", input, self.dimension)?;
        if output.len() != self.dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "element-assembly output must contain {} values",
                self.dimension
            )));
        }
        let physical = self.constraints.expand_homogeneous(input)?;
        let mut physical_output = vec![0.0; self.dimension];
        for batch in 0..self.batches.batch_count() {
            for cell in self.batches.batch(batch).expect("batch index is bounded") {
                let Some(cell) = cell else { continue };
                let restriction = &self.restrictions[*cell];
                let matrix = &self.local_matrices[*cell];
                let local_dimension = restriction.dofs.len();
                for row in 0..local_dimension {
                    let value = (0..local_dimension)
                        .map(|column| {
                            matrix[row * local_dimension + column]
                                * physical[restriction.dofs[column].0]
                        })
                        .sum::<f64>();
                    physical_output[restriction.dofs[row].0] += value;
                }
            }
        }
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] = self
                .constraints
                .direction_residual(input, constraint.target)?;
        }
        if let Some(index) = output.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(format!(
                "element-assembly output is non-finite at index {index}"
            )));
        }
        Ok(())
    }
}

impl LinearOperator for ElementAssemblyOperator {
    fn rows(&self) -> usize {
        self.dimension
    }

    fn columns(&self) -> usize {
        self.dimension
    }

    fn symmetry(&self) -> OperatorSymmetry {
        if self.constraints.has_affine_dependencies() {
            OperatorSymmetry::Nonsymmetric
        } else {
            OperatorSymmetry::Unknown
        }
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.apply_inner(input, output)
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PartialPointAction {
    pub(crate) point: usize,
    pub(crate) scale: f64,
    pub(crate) active_inputs: Vec<QFunctionInput>,
    pub(crate) output_derivative: DerivativeEvaluation,
    pub(crate) output_components: usize,
    /// Row-major point Jacobian from concatenated active evaluations to one QFunction output.
    pub(crate) matrix: Vec<f64>,
}

/// Quadrature-data realization applying `E^T B^T D B E` without cell or global matrices.
/// Affine dependency constraint rows make the resulting full-coordinate action nonsymmetric.
#[derive(Clone, Debug)]
pub struct PartialAssemblyOperator {
    dimension: usize,
    source_factorization_digest: Digest,
    restrictions: Vec<ElementRestriction>,
    constraints: ConstraintSet,
    element: PreparedElement,
    geometries: Vec<CellGeometry>,
    point_actions: Vec<Vec<PartialPointAction>>,
    batches: CellBatchLayout,
}

impl PartialAssemblyOperator {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dimension: usize,
        source_factorization_digest: Digest,
        restrictions: Vec<ElementRestriction>,
        constraints: ConstraintSet,
        element: PreparedElement,
        geometries: Vec<CellGeometry>,
        point_actions: Vec<Vec<PartialPointAction>>,
        lane_width: usize,
    ) -> Result<Self, FinitumError> {
        let batches = CellBatchLayout::new(restrictions.len(), lane_width)?;
        Ok(Self {
            dimension,
            source_factorization_digest,
            restrictions,
            constraints,
            element,
            geometries,
            point_actions,
            batches,
        })
    }

    pub fn source_factorization_digest(&self) -> &Digest {
        &self.source_factorization_digest
    }

    pub fn batches(&self) -> &CellBatchLayout {
        &self.batches
    }

    pub fn stored_point_action_count(&self) -> usize {
        self.point_actions.iter().map(Vec::len).sum()
    }

    fn apply_inner(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        validate_finite_length("partial-assembly input", input, self.dimension)?;
        if output.len() != self.dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "partial-assembly output must contain {} values",
                self.dimension
            )));
        }
        let physical = self.constraints.expand_homogeneous(input)?;
        let mut physical_output = vec![0.0; self.dimension];
        for batch in 0..self.batches.batch_count() {
            for cell in self.batches.batch(batch).expect("batch index is bounded") {
                let Some(cell) = cell else { continue };
                let restriction = &self.restrictions[*cell];
                let local = restriction
                    .dofs
                    .iter()
                    .map(|dof| physical[dof.0])
                    .collect::<Vec<_>>();
                let mut local_output = vec![0.0; restriction.dofs.len()];
                for action in &self.point_actions[*cell] {
                    let mut point_input = Vec::new();
                    for qinput in &action.active_inputs {
                        let values = if qinput.binding.evaluation.derivative
                            == DerivativeEvaluation::TimeDerivative
                        {
                            vec![0.0; checked_product(&qinput.shape, "QFunction input")?]
                        } else {
                            evaluate_basis_input(
                                &self.element,
                                &self.geometries[*cell],
                                action.point,
                                qinput,
                                &local,
                            )?
                        };
                        point_input.extend(values);
                    }
                    let input_components = point_input.len();
                    let mut point_output = vec![0.0; action.output_components];
                    for (row, value) in point_output.iter_mut().enumerate() {
                        *value = (0..input_components)
                            .map(|column| {
                                action.matrix[row * input_components + column] * point_input[column]
                            })
                            .sum();
                    }
                    apply_basis_adjoint(
                        &self.element,
                        &self.geometries[*cell],
                        action.point,
                        &action.output_derivative,
                        &point_output,
                        action.scale,
                        &mut local_output,
                    )?;
                }
                for (local, dof) in restriction.dofs.iter().enumerate() {
                    physical_output[dof.0] += local_output[local];
                }
            }
        }
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] = self
                .constraints
                .direction_residual(input, constraint.target)?;
        }
        if let Some(index) = output.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(format!(
                "partial-assembly output is non-finite at index {index}"
            )));
        }
        Ok(())
    }
}

impl LinearOperator for PartialAssemblyOperator {
    fn rows(&self) -> usize {
        self.dimension
    }

    fn columns(&self) -> usize {
        self.dimension
    }

    fn symmetry(&self) -> OperatorSymmetry {
        if self.constraints.has_affine_dependencies() {
            OperatorSymmetry::Nonsymmetric
        } else {
            OperatorSymmetry::Unknown
        }
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.apply_inner(input, output)
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

fn contract_axis(
    input: &[f64],
    input_extents: &[usize],
    axis: usize,
    output_extent: usize,
    table: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    let mut output_extents = input_extents.to_vec();
    output_extents[axis] = output_extent;
    let output_len = checked_product(&output_extents, "tensor contraction")?;
    let mut output = vec![0.0; output_len];
    for (linear, value) in output.iter_mut().enumerate() {
        let mut coordinates = decode_index(linear, &output_extents);
        let output_coordinate = coordinates[axis];
        let mut sum = 0.0;
        for input_coordinate in 0..input_extents[axis] {
            coordinates[axis] = input_coordinate;
            let input_index = encode_index(&coordinates, input_extents);
            sum += table[output_coordinate * input_extents[axis] + input_coordinate]
                * input[input_index];
        }
        *value = sum;
    }
    Ok(output)
}

fn decode_index(mut linear: usize, extents: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; extents.len()];
    for axis in (0..extents.len()).rev() {
        coordinates[axis] = linear % extents[axis];
        linear /= extents[axis];
    }
    coordinates
}

fn encode_index(coordinates: &[usize], extents: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(extents)
        .fold(0, |linear, (coordinate, extent)| {
            linear * extent + coordinate
        })
}

fn checked_power(base: usize, exponent: usize) -> Result<usize, FinitumError> {
    (0..exponent).try_fold(1usize, |value, _| {
        value.checked_mul(base).ok_or_else(|| {
            FinitumError::InvalidRealization("tensor-product extent overflows usize".into())
        })
    })
}

fn checked_product(extents: &[usize], name: &str) -> Result<usize, FinitumError> {
    extents.iter().try_fold(1usize, |value, extent| {
        value.checked_mul(*extent).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("{name} extent overflows usize"))
        })
    })
}

fn validate_finite_length(name: &str, values: &[f64], expected: usize) -> Result<(), FinitumError> {
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        return Err(FinitumError::InvalidRealization(format!(
            "{name} must contain {expected} finite values"
        )));
    }
    Ok(())
}
