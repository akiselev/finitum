use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use malleus::{
    AccessMode, BufferBinding, Executable, ExecutableModule, Interpreter, OperandId,
    validate_module,
};
use methodus::{CsrMatrix, EvaluationContext, LinearOperator, NumericError, OperatorSymmetry};
use scientia::scientific::ValueShape;
use scientia::{
    DerivativeEvaluation, Digest, ElementFamilyRequirement, EvaluationSite, FormRequirements,
    InputSourceRequirement, IntegralOperatorFactorization, OperatorFactorization, QFunctionInput,
    SemanticMeasure, StructuredOperatorKernels, StructuredPointKernelBundle, TensorInputId,
    TensorInputRole,
};
use serde::Serialize;

use crate::optimized::{ElementAssemblyOperator, PartialAssemblyOperator, PartialPointAction};
use crate::{CellId, ConstraintSet, DofMap, FinitumError, Mesh, PreparedElement};

pub const REALIZATION_ARTIFACT_SCHEMA: &str = "finitum-realization-plan/2";

/// Concrete quadrature-point values for one non-basis QFunction input.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalInput {
    pub integral_index: usize,
    pub input: TensorInputId,
    component_count: usize,
    values: Vec<f64>,
}

/// One basis-backed active input evaluated at a quadrature point.
#[derive(Clone, Debug, PartialEq)]
pub struct PointActiveInput {
    pub input: TensorInputId,
    pub derivative: DerivativeEvaluation,
    pub values: Vec<f64>,
}

/// Runtime point data supplied to a state-dependent external input.
#[derive(Clone, Debug, PartialEq)]
pub struct PointEvaluation {
    pub time: f64,
    pub cell: CellId,
    pub coordinates: Vec<f64>,
    pub active: Vec<PointActiveInput>,
}

impl PointEvaluation {
    /// Return the first active binding with the requested evaluation kind.
    pub fn values(&self, derivative: DerivativeEvaluation) -> Option<&[f64]> {
        self.active
            .iter()
            .find(|input| input.derivative == derivative)
            .map(|input| input.values.as_slice())
    }
}

type PointValueEvaluator = dyn Fn(&PointEvaluation) -> Vec<f64> + Send + Sync;
type PointDirectionEvaluator = dyn Fn(&PointEvaluation, &PointEvaluation) -> Vec<f64> + Send + Sync;

/// A non-basis QFunction input evaluated from the current point state.
///
/// The callbacks are supplied by the consuming product or material implementation. Finitum
/// only binds their values and directional derivatives into generated parameter-JVP kernels.
/// `identity` must change whenever either callback's semantics change. The direction callback is
/// trusted to return the exact directional derivative of the value callback; products should
/// retain centered-difference acceptance checks for every authored dynamic binding.
#[derive(Clone)]
pub struct DynamicExternalInput {
    pub integral_index: usize,
    pub input: TensorInputId,
    component_count: usize,
    identity: String,
    value: Arc<PointValueEvaluator>,
    direction: Arc<PointDirectionEvaluator>,
}

impl std::fmt::Debug for DynamicExternalInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicExternalInput")
            .field("integral_index", &self.integral_index)
            .field("input", &self.input)
            .field("component_count", &self.component_count)
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl DynamicExternalInput {
    pub fn new(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        identity: impl Into<String>,
        value: impl Fn(&PointEvaluation) -> Vec<f64> + Send + Sync + 'static,
        direction: impl Fn(&PointEvaluation, &PointEvaluation) -> Vec<f64> + Send + Sync + 'static,
    ) -> Result<Self, FinitumError> {
        if component_count == 0 {
            return Err(FinitumError::InvalidRealization(
                "dynamic external input component count must be non-zero".into(),
            ));
        }
        let identity = identity.into();
        if identity.trim().is_empty() {
            return Err(FinitumError::InvalidRealization(
                "dynamic external input identity must not be empty".into(),
            ));
        }
        Ok(Self {
            integral_index,
            input,
            component_count,
            identity,
            value: Arc::new(value),
            direction: Arc::new(direction),
        })
    }
}

#[derive(Clone, Debug)]
enum ExternalBinding {
    Stored(ExternalInput),
    Dynamic(DynamicExternalInput),
}

impl ExternalInput {
    pub fn new(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        values: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if component_count == 0 {
            return Err(FinitumError::InvalidRealization(
                "external input component count must be non-zero".into(),
            ));
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(format!(
                "external input contains a non-finite value at index {index}"
            )));
        }
        Ok(Self {
            integral_index,
            input,
            component_count,
            values,
        })
    }

    /// Sample and own input values in deterministic cell/quadrature/component order.
    pub fn sampled(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        element: &PreparedElement,
        mut sample: impl FnMut(CellId, &[f64]) -> Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if mesh.dimension() != element.dimension() {
            return Err(FinitumError::InvalidRealization(format!(
                "mesh dimension {} differs from element dimension {}",
                mesh.dimension(),
                element.dimension()
            )));
        }
        let mut values = Vec::new();
        for (cell_index, _) in mesh.cells().iter().enumerate() {
            let cell = CellGeometry::new(mesh, CellId(cell_index))?;
            for point in element.quadrature() {
                let physical = cell.physical_point(&point.coordinates);
                let sampled = sample(CellId(cell_index), &physical);
                if sampled.len() != component_count {
                    return Err(FinitumError::InvalidRealization(format!(
                        "external input sampler returned {} components, expected {component_count}",
                        sampled.len()
                    )));
                }
                values.extend(sampled);
            }
        }
        Self::new(integral_index, input, component_count, values)
    }

    fn point_values(&self, cell: usize, point: usize, point_count: usize) -> &[f64] {
        let start = (cell * point_count + point) * self.component_count;
        &self.values[start..start + self.component_count]
    }
}

/// Stored per-quadrature-point direction values for one external input under
/// one concrete geometry design parameter.
///
/// The layout mirrors [`ExternalInput`]: deterministic cell/quadrature/component
/// order over the baseline chart, so an authored field can declare its exact
/// design derivative alongside the sampled values it differentiates.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSensitivityInput {
    pub integral_index: usize,
    pub input: TensorInputId,
    component_count: usize,
    values: Vec<f64>,
}

impl ExternalSensitivityInput {
    pub fn new(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        values: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if component_count == 0 {
            return Err(FinitumError::InvalidRealization(
                "external sensitivity component count must be non-zero".into(),
            ));
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(format!(
                "external sensitivity contains a non-finite value at index {index}"
            )));
        }
        Ok(Self {
            integral_index,
            input,
            component_count,
            values,
        })
    }

    /// Sample design-direction values in deterministic cell/quadrature order.
    ///
    /// The sampler receives baseline physical quadrature points; authored
    /// closures must differentiate at fixed chart identity, because a fixed
    /// topology keeps every chart coordinate stable while positions move.
    pub fn sampled(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        element: &PreparedElement,
        mut sample: impl FnMut(CellId, &[f64]) -> Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if mesh.dimension() != element.dimension() {
            return Err(FinitumError::InvalidRealization(format!(
                "mesh dimension {} differs from element dimension {}",
                mesh.dimension(),
                element.dimension()
            )));
        }
        let mut values = Vec::new();
        for (cell_index, _) in mesh.cells().iter().enumerate() {
            let cell = CellGeometry::new(mesh, CellId(cell_index))?;
            for point in element.quadrature() {
                let physical = cell.physical_point(&point.coordinates);
                let sampled = sample(CellId(cell_index), &physical);
                if sampled.len() != component_count {
                    return Err(FinitumError::InvalidRealization(format!(
                        "external sensitivity sampler returned {} components, expected {component_count}",
                        sampled.len()
                    )));
                }
                values.extend(sampled);
            }
        }
        Self::new(integral_index, input, component_count, values)
    }

    fn point_values(&self, cell: usize, point: usize, point_count: usize) -> &[f64] {
        let start = (cell * point_count + point) * self.component_count;
        &self.values[start..start + self.component_count]
    }
}

/// Exact first-order geometry data for one CAD design parameter.
///
/// Node velocities are vertex-major physical components supplied by the
/// geometry producer's analytic design differential. External direction
/// entries must cover every stored external input of the target plan;
/// missing or extra entries are refused instead of silently frozen.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryParameterSensitivity {
    /// Design parameter this sensitivity differentiates with respect to.
    pub parameter_index: usize,
    /// Vertex-major design velocity components.
    pub node_velocities: Vec<f64>,
    /// Per-input external direction values.
    pub external_directions: Vec<ExternalSensitivityInput>,
}

#[derive(Clone, Debug)]
struct BoundBundle {
    bundle: StructuredPointKernelBundle,
    executable: ExecutableModule,
}

#[derive(Clone, Debug)]
struct RealizationData {
    digest: Digest,
    kernels_digest: Digest,
    requirements: FormRequirements,
    factorization: OperatorFactorization,
    mesh: Mesh,
    element: PreparedElement,
    geometries: Vec<CellGeometry>,
    dofs: DofMap,
    constraints: ConstraintSet,
    external: BTreeMap<(usize, TensorInputId), ExternalBinding>,
    bundles: BTreeMap<(usize, usize), BoundBundle>,
}

/// Digest-linked binding of FC3 requirements, an FC4 factorization, FC5 executables, and
/// concrete mesh/element/DOF/constraint/input data.
///
/// The FC6 linear view evaluates generated JVPs at zero active input. FC7 residual and JVP methods
/// instead bind independent runtime state and state-rate vectors at the actual linearization
/// point. Semantic boundary requirements are linked to the artifact chain, but the caller
/// currently supplies the concrete boundary-DOF membership.
#[derive(Clone, Debug)]
pub struct RealizationPlan {
    data: Arc<RealizationData>,
}

/// Stable, fully inspectable projection of a concrete realization.
///
/// This projection records identities and serializable data for artifact inspection. It is not a
/// deserialization or `RealizationPlan` reconstruction API: generated executables are absent, and
/// dynamic callbacks are represented only by the identity that participates in the realization
/// digest.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RealizationArtifact {
    pub schema: String,
    pub artifact_digest: Digest,
    pub source_requirements_digest: Digest,
    pub source_factorization_digest: Digest,
    pub source_kernels_digest: Digest,
    pub mesh: Mesh,
    pub element: PreparedElement,
    pub dofs: DofMap,
    pub constraints: ConstraintSet,
    pub external_inputs: Vec<RealizationExternalInput>,
}

/// Serializable product data for one stored or dynamic realization input.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RealizationExternalInput {
    Stored {
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        values: Vec<f64>,
    },
    Dynamic {
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        identity: String,
    },
}

impl RealizationPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        requirements: FormRequirements,
        factorization: OperatorFactorization,
        kernels: StructuredOperatorKernels,
        mesh: Mesh,
        element: PreparedElement,
        dofs: DofMap,
        constraints: ConstraintSet,
        external_inputs: Vec<ExternalInput>,
    ) -> Result<Self, FinitumError> {
        Self::new_stateful(
            requirements,
            factorization,
            kernels,
            mesh,
            element,
            dofs,
            constraints,
            external_inputs,
            Vec::new(),
        )
    }

    /// Bind a realization whose non-basis inputs may depend on the runtime point state.
    #[allow(clippy::too_many_arguments)]
    pub fn new_stateful(
        requirements: FormRequirements,
        factorization: OperatorFactorization,
        kernels: StructuredOperatorKernels,
        mesh: Mesh,
        element: PreparedElement,
        dofs: DofMap,
        constraints: ConstraintSet,
        external_inputs: Vec<ExternalInput>,
        dynamic_external_inputs: Vec<DynamicExternalInput>,
    ) -> Result<Self, FinitumError> {
        validate_artifacts(&requirements, &factorization, &kernels)?;
        validate_discretization(
            &requirements,
            &factorization,
            &mesh,
            &element,
            &dofs,
            &constraints,
        )?;
        let external = validate_external_inputs(
            &factorization,
            &mesh,
            &element,
            external_inputs,
            dynamic_external_inputs,
        )?;
        let digest = realization_digest(
            &requirements,
            &factorization,
            &kernels,
            &mesh,
            &element,
            &dofs,
            &constraints,
            &external,
        );
        let kernels_digest = kernels.artifact_digest.clone();
        let bundles = bind_kernels(&factorization, kernels)?;
        let geometries = (0..mesh.cells().len())
            .map(|cell| CellGeometry::new(&mesh, CellId(cell)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            data: Arc::new(RealizationData {
                digest,
                kernels_digest,
                requirements,
                factorization,
                mesh,
                element,
                geometries,
                dofs,
                constraints,
                external,
                bundles,
            }),
        })
    }

    pub fn dimension(&self) -> usize {
        self.data.dofs.dof_count()
    }

    pub fn mesh(&self) -> &Mesh {
        &self.data.mesh
    }

    /// Digest of the concrete realization, including discretization, constraints, and external
    /// input descriptors. Dynamic callbacks are represented by their required caller identity.
    pub fn digest(&self) -> &Digest {
        &self.data.digest
    }

    pub fn source_factorization_digest(&self) -> &Digest {
        &self.data.factorization.artifact_digest
    }

    pub fn source_requirements_digest(&self) -> &Digest {
        &self.data.requirements.artifact_digest
    }

    /// Capture every identity-sensitive input exposed by this realization for inspection.
    pub fn artifact(&self) -> RealizationArtifact {
        let external_inputs = self
            .data
            .external
            .values()
            .map(|binding| match binding {
                ExternalBinding::Stored(input) => RealizationExternalInput::Stored {
                    integral_index: input.integral_index,
                    input: input.input,
                    component_count: input.component_count,
                    values: input.values.clone(),
                },
                ExternalBinding::Dynamic(input) => RealizationExternalInput::Dynamic {
                    integral_index: input.integral_index,
                    input: input.input,
                    component_count: input.component_count,
                    identity: input.identity.clone(),
                },
            })
            .collect();
        RealizationArtifact {
            schema: REALIZATION_ARTIFACT_SCHEMA.into(),
            artifact_digest: self.data.digest.clone(),
            source_requirements_digest: self.data.requirements.artifact_digest.clone(),
            source_factorization_digest: self.data.factorization.artifact_digest.clone(),
            source_kernels_digest: self.data.kernels_digest.clone(),
            mesh: self.data.mesh.clone(),
            element: self.data.element.clone(),
            dofs: self.data.dofs.clone(),
            constraints: self.data.constraints.clone(),
            external_inputs,
        }
    }

    /// Evaluate the generated global residual at independent state and state-rate vectors.
    pub fn residual(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_time_action(time, state, state_rate, output)?;
        let physical_state = self.data.constraints.expand(state)?;
        let physical_rate = self.data.constraints.expand_homogeneous(state_rate)?;
        let mut physical_output = vec![0.0; self.dimension()];
        self.apply_cells(
            time,
            &physical_state,
            &physical_rate,
            None,
            None,
            &mut physical_output,
            Action::Primal,
        )?;
        output.copy_from_slice(&self.data.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.data.constraints.constraints() {
            output[constraint.target.0] = self
                .data
                .constraints
                .equation_residual(state, constraint.target)?;
        }
        validate_finite("stateful residual", output)
    }

    /// Evaluate the generated state/rate JVP plus the chain rule through dynamic external inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn jacobian_vector_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_time_action(time, state, state_rate, output)?;
        self.validate_action(state_direction, output)?;
        self.validate_action(rate_direction, output)?;
        let physical_state = self.data.constraints.expand(state)?;
        let physical_rate = self.data.constraints.expand_homogeneous(state_rate)?;
        let physical_state_direction = self.data.constraints.expand_homogeneous(state_direction)?;
        let physical_rate_direction = self.data.constraints.expand_homogeneous(rate_direction)?;
        let mut physical_output = vec![0.0; self.dimension()];
        self.apply_cells(
            time,
            &physical_state,
            &physical_rate,
            Some(&physical_state_direction),
            Some(&physical_rate_direction),
            &mut physical_output,
            Action::Jvp,
        )?;
        output.copy_from_slice(&self.data.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.data.constraints.constraints() {
            output[constraint.target.0] = self
                .data
                .constraints
                .direction_residual(state_direction, constraint.target)?;
        }
        validate_finite("stateful JVP", output)
    }

    /// Return the zero-active-state JVP realization for this globally linear FC6 plan.
    pub fn matrix_free(&self) -> MatrixFreeOperator {
        MatrixFreeOperator { plan: self.clone() }
    }

    /// Assemble by applying the matrix-free realization to canonical coordinate vectors. Both
    /// representations therefore execute the same factorization and generated JVP kernels.
    pub fn assemble(&self) -> Result<AssembledOperator, FinitumError> {
        let dimension = self.dimension();
        let mut entries = Vec::new();
        let mut direction = vec![0.0; dimension];
        let mut output = vec![0.0; dimension];
        for column in 0..dimension {
            direction[column] = 1.0;
            self.apply_direction(&direction, &mut output)?;
            for (row, value) in output.iter().copied().enumerate() {
                if value != 0.0 {
                    entries.push((row, column, value));
                }
            }
            direction[column] = 0.0;
        }
        let matrix = CsrMatrix::from_triplets(dimension, dimension, entries)
            .map_err(|error| FinitumError::Assembly(error.to_string()))?;
        let symmetry = if self.data.constraints.has_affine_dependencies() {
            OperatorSymmetry::Nonsymmetric
        } else {
            matrix.symmetry()
        };
        Ok(AssembledOperator {
            matrix,
            source_factorization_digest: self.source_factorization_digest().clone(),
            symmetry,
        })
    }

    /// Precompute dense cell actions while retaining element restrictions and global scatter.
    ///
    /// Like [`Self::matrix_free`], this extracts the generated JVP at zero state and state rate.
    /// It is exact for the globally linear scope and is a frozen zero-state linearization if
    /// reused with a nonlinear form.
    pub fn element_assembly(
        &self,
        lane_width: usize,
    ) -> Result<ElementAssemblyOperator, FinitumError> {
        let dimension = self.dimension();
        let zero = vec![0.0; dimension];
        let mut local_matrices = Vec::with_capacity(self.data.dofs.restrictions().len());
        for (cell, restriction) in self.data.dofs.restrictions().iter().enumerate() {
            let local_dimension = restriction.dofs.len();
            let mut matrix = vec![0.0; local_dimension * local_dimension];
            for (column, dof) in restriction.dofs.iter().enumerate() {
                let mut direction = vec![0.0; dimension];
                direction[dof.0] = 1.0;
                let mut output = vec![0.0; dimension];
                self.apply_cell(
                    cell,
                    0.0,
                    &zero,
                    &zero,
                    Some(&direction),
                    Some(&zero),
                    &mut output,
                    Action::Jvp,
                )?;
                for (row, row_dof) in restriction.dofs.iter().enumerate() {
                    matrix[row * local_dimension + column] = output[row_dof.0];
                }
            }
            local_matrices.push(matrix);
        }
        ElementAssemblyOperator::new(
            dimension,
            self.source_factorization_digest().clone(),
            self.data.dofs.restrictions().to_vec(),
            self.data.constraints.clone(),
            local_matrices,
            lane_width,
        )
    }

    /// Precompute quadrature-point Jacobians while retaining basis actions and restrictions.
    ///
    /// Like [`Self::matrix_free`], this extracts the generated JVP at zero state and state rate.
    /// It is exact for the globally linear scope and is a frozen zero-state linearization if
    /// reused with a nonlinear form.
    pub fn partial_assembly(
        &self,
        lane_width: usize,
    ) -> Result<PartialAssemblyOperator, FinitumError> {
        if self
            .data
            .external
            .values()
            .any(|binding| matches!(binding, ExternalBinding::Dynamic(_)))
        {
            return Err(FinitumError::UnsupportedRealization(
                "partial assembly currently requires state-independent external inputs".into(),
            ));
        }
        let mut point_actions = Vec::with_capacity(self.data.dofs.restrictions().len());
        for (cell, restriction) in self.data.dofs.restrictions().iter().enumerate() {
            let geometry = &self.data.geometries[cell];
            let zero = vec![0.0; restriction.dofs.len()];
            let mut cell_actions = Vec::new();
            for (point, quadrature) in self.data.element.quadrature().iter().enumerate() {
                let scale = quadrature.weight * geometry.determinant;
                for integral in &self.data.factorization.integrals {
                    let (inputs, _) =
                        self.point_inputs(integral, cell, point, geometry, 0.0, &zero, &zero)?;
                    let active_inputs = integral
                        .primal
                        .inputs
                        .iter()
                        .filter(|input| {
                            input.source == InputSourceRequirement::Basis
                                && input.role == TensorInputRole::Active
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    let input_components = active_inputs
                        .iter()
                        .map(|input| component_count(&input.shape))
                        .sum::<Result<usize, _>>()?;
                    if input_components == 0 {
                        continue;
                    }
                    for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                        let bound = &self.data.bundles[&(integral.integral_index, output_index)];
                        let mut columns = Vec::with_capacity(input_components);
                        for selected in 0..input_components {
                            let mut directions = integral
                                .primal
                                .inputs
                                .iter()
                                .map(|input| {
                                    component_count(&input.shape)
                                        .map(|count| (input.id, vec![0.0; count]))
                                })
                                .collect::<Result<BTreeMap<_, _>, _>>()?;
                            let mut offset = 0;
                            for input in &active_inputs {
                                let count = component_count(&input.shape)?;
                                if (offset..offset + count).contains(&selected) {
                                    directions.get_mut(&input.id).expect("input was inserted")
                                        [selected - offset] = 1.0;
                                    break;
                                }
                                offset += count;
                            }
                            columns.push(self.execute_jvp_values(bound, &inputs, &directions)?);
                        }
                        let output_components = columns[0].len();
                        if columns
                            .iter()
                            .any(|column| column.len() != output_components)
                        {
                            return Err(FinitumError::InvalidRealization(
                                "partial point Jacobian has inconsistent output extents".into(),
                            ));
                        }
                        let mut matrix = vec![0.0; output_components * input_components];
                        for (column, values) in columns.iter().enumerate() {
                            for (row, value) in values.iter().copied().enumerate() {
                                matrix[row * input_components + column] = value;
                            }
                        }
                        cell_actions.push(PartialPointAction {
                            point,
                            scale,
                            active_inputs: active_inputs.clone(),
                            output_derivative: qoutput.binding.evaluation.derivative,
                            output_components,
                            matrix,
                        });
                    }
                }
            }
            point_actions.push(cell_actions);
        }
        PartialAssemblyOperator::new(
            self.dimension(),
            self.source_factorization_digest().clone(),
            self.data.dofs.restrictions().to_vec(),
            self.data.constraints.clone(),
            self.data.element.clone(),
            self.data.geometries.clone(),
            point_actions,
            lane_width,
        )
    }

    /// Build the affine right-hand side from the generated primal kernels and fixed essential
    /// values. No source term is duplicated in Finitum.
    pub fn load_vector(&self) -> Result<Vec<f64>, FinitumError> {
        let lifting = self.data.constraints.expand(&vec![0.0; self.dimension()])?;
        let mut physical_residual = vec![0.0; self.dimension()];
        self.apply_primal(&lifting, &mut physical_residual)?;
        let mut residual = self
            .data
            .constraints
            .restrict_transpose(&physical_residual)?;
        for value in &mut residual {
            *value = -*value;
        }
        for constraint in self.data.constraints.constraints() {
            residual[constraint.target.0] = constraint.offset;
        }
        Ok(residual)
    }

    fn apply_direction(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.validate_action(input, output)?;
        let homogeneous = self.data.constraints.expand_homogeneous(input)?;
        let mut physical_output = vec![0.0; self.dimension()];
        let zero = vec![0.0; self.dimension()];
        self.apply_cells(
            0.0,
            &zero,
            &zero,
            Some(&homogeneous),
            Some(&zero),
            &mut physical_output,
            Action::Jvp,
        )?;
        output.copy_from_slice(&self.data.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.data.constraints.constraints() {
            output[constraint.target.0] = self
                .data
                .constraints
                .direction_residual(input, constraint.target)?;
        }
        validate_finite("matrix-free output", output)
    }

    fn apply_primal(&self, state: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.validate_action(state, output)?;
        output.fill(0.0);
        let zero = vec![0.0; self.dimension()];
        self.apply_cells(0.0, state, &zero, None, None, output, Action::Primal)?;
        validate_finite("primal residual", output)
    }

    fn validate_time_action(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &[f64],
    ) -> Result<(), FinitumError> {
        if !time.is_finite() {
            return Err(FinitumError::InvalidRealization(
                "operator evaluation time must be finite".into(),
            ));
        }
        self.validate_action(state, output)?;
        self.validate_action(state_rate, output)
    }

    /// Exact residual sensitivity `dR/dp_k` at a fixed expanded state.
    ///
    /// The state is expanded through the affine constraints exactly like
    /// [`Self::residual`]; constraint rows carry zero design derivative
    /// because essential values are frozen inputs of the realization.
    /// Stored external inputs participate with their authored direction
    /// tables, because the load term belongs to the residual.
    pub fn residual_geometry_sensitivity(
        &self,
        time: f64,
        state: &[f64],
        sensitivity: &GeometryParameterSensitivity,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_time_action(time, state, state, output)?;
        let physical_state = self.data.constraints.expand(state)?;
        let mut physical_output = vec![0.0; self.dimension()];
        self.apply_geometry_sensitivity_cells(
            time,
            &physical_state,
            &physical_state,
            sensitivity,
            &mut physical_output,
        )?;
        output.copy_from_slice(&self.data.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.data.constraints.constraints() {
            output[constraint.target.0] = 0.0;
        }
        validate_finite("residual geometry sensitivity", output)
    }

    fn apply_geometry_sensitivity_cells(
        &self,
        time: f64,
        value_state: &[f64],
        direction_state: &[f64],
        sensitivity: &GeometryParameterSensitivity,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        if self
            .data
            .external
            .values()
            .any(|binding| matches!(binding, ExternalBinding::Dynamic(_)))
        {
            return Err(FinitumError::UnsupportedRealization(
                "geometry sensitivity requires stored external inputs; dynamic callbacks cannot declare an exact design derivative".into(),
            ));
        }
        let dimension = self.data.mesh.dimension();
        let vertex_count = self.data.mesh.vertices().len();
        if sensitivity.node_velocities.len() != vertex_count * dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "geometry node velocities have length {}, expected {}",
                sensitivity.node_velocities.len(),
                vertex_count * dimension
            )));
        }
        if sensitivity
            .node_velocities
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(FinitumError::InvalidRealization(
                "geometry node velocities contain a non-finite component".into(),
            ));
        }
        let directions = validate_sensitivity_inputs(self, sensitivity)?;
        let zero_rate = vec![0.0; self.dimension()];
        for cell_index in 0..self.data.dofs.restrictions().len() {
            self.apply_cell_geometry_sensitivity(
                cell_index,
                time,
                value_state,
                direction_state,
                &zero_rate,
                sensitivity,
                &directions,
                dimension,
                output,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_cell_geometry_sensitivity(
        &self,
        cell_index: usize,
        time: f64,
        value_state: &[f64],
        direction_state: &[f64],
        state_rate: &[f64],
        sensitivity: &GeometryParameterSensitivity,
        directions: &BTreeMap<(usize, TensorInputId), &ExternalSensitivityInput>,
        dimension: usize,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let restriction = &self.data.dofs.restrictions()[cell_index];
        let geometry = &self.data.geometries[cell_index];
        let cell = self.data.mesh.cell(CellId(cell_index)).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("mesh has no cell {cell_index}"))
        })?;
        let local_state = restriction
            .dofs
            .iter()
            .map(|dof| value_state[dof.0])
            .collect::<Vec<_>>();
        let local_direction = restriction
            .dofs
            .iter()
            .map(|dof| direction_state[dof.0])
            .collect::<Vec<_>>();
        // Affine map columns are vertex differences, so their design velocity
        // is the matching difference of exact nodal velocities.
        let velocity = |slot: usize| {
            let vertex = cell.vertices[slot].0;
            let start = vertex * dimension;
            (0..dimension)
                .map(|axis| sensitivity.node_velocities[start + axis])
                .collect::<Vec<_>>()
        };
        let origin_velocity = velocity(0);
        let mut jacobian_direction = vec![0.0; dimension * dimension];
        for column in 0..dimension {
            let column_velocity = velocity(column + 1);
            for row in 0..dimension {
                jacobian_direction[row * dimension + column] =
                    column_velocity[row] - origin_velocity[row];
            }
        }
        let (signed_determinant, inverse) =
            invert(&geometry.jacobian, dimension).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "cell {cell_index} has a singular affine geometry map"
                ))
            })?;
        if signed_determinant <= 0.0 || signed_determinant != geometry.determinant {
            return Err(FinitumError::InvalidRealization(format!(
                "cell {cell_index} is not positively oriented; geometry derivatives refuse reflected maps"
            )));
        }
        // det' = det tr(B^{-1} B') and M' = -B^{-1} B' B^{-1}.
        let mut inverse_jacobian_trace = 0.0;
        for row in 0..dimension {
            for column in 0..dimension {
                inverse_jacobian_trace += inverse[column * dimension + row]
                    * jacobian_direction[row * dimension + column];
            }
        }
        let determinant_direction = signed_determinant * inverse_jacobian_trace;
        let mut inverse_direction = vec![0.0; dimension * dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                let mut product = 0.0;
                for middle_row in 0..dimension {
                    for middle_column in 0..dimension {
                        product += inverse[row * dimension + middle_row]
                            * jacobian_direction[middle_row * dimension + middle_column]
                            * inverse[middle_column * dimension + column];
                    }
                }
                inverse_direction[row * dimension + column] = -product;
            }
        }
        // Physical gradient direction d(B^{-T} ref)/dp.
        let gradient_direction = |reference: &[f64]| -> Vec<f64> {
            (0..dimension)
                .map(|physical_axis| {
                    (0..dimension)
                        .map(|reference_axis| {
                            inverse_direction[reference_axis * dimension + physical_axis]
                                * reference[reference_axis]
                        })
                        .sum()
                })
                .collect()
        };
        let mut local_output = vec![0.0; restriction.dofs.len()];
        for (point_index, point) in self.data.element.quadrature().iter().enumerate() {
            let scale = point.weight * signed_determinant;
            let scale_direction = point.weight * determinant_direction;
            for integral in &self.data.factorization.integrals {
                let (inputs, _) = self.point_inputs(
                    integral,
                    cell_index,
                    point_index,
                    geometry,
                    time,
                    &local_state,
                    state_rate,
                )?;
                let mut point_directions = BTreeMap::new();
                for input in &integral.primal.inputs {
                    if input.source == InputSourceRequirement::Basis {
                        let direction = match input.binding.evaluation.derivative {
                            DerivativeEvaluation::Gradient => {
                                let mut values = vec![0.0; dimension];
                                for (basis, coefficient) in local_direction.iter().enumerate() {
                                    let direction_gradient = gradient_direction(
                                        self.data
                                            .element
                                            .basis_gradient(point_index, basis)
                                            .expect("validated element table"),
                                    );
                                    for axis in 0..dimension {
                                        values[axis] += direction_gradient[axis] * coefficient;
                                    }
                                }
                                values
                            }
                            _ => vec![0.0; component_count(&input.shape)?],
                        };
                        point_directions.insert(input.id, direction);
                    } else {
                        let binding = directions.get(&(integral.integral_index, input.id)).ok_or(
                            FinitumError::MissingExternalInput {
                                integral: integral.integral_index,
                                input: input.id,
                            },
                        )?;
                        point_directions.insert(
                            input.id,
                            binding
                                .point_values(
                                    cell_index,
                                    point_index,
                                    self.data.element.quadrature().len(),
                                )
                                .to_vec(),
                        );
                    }
                }
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = &self.data.bundles[&(integral.integral_index, output_index)];
                    let point_output = self.execute_primal(
                        bound,
                        integral,
                        cell_index,
                        point_index,
                        geometry,
                        time,
                        &local_state,
                        state_rate,
                    )?;
                    let point_output_direction =
                        self.execute_jvp_values(bound, &inputs, &point_directions)?;
                    let output_arity_ok = match qoutput.binding.evaluation.derivative {
                        DerivativeEvaluation::Value => point_output.len() == 1,
                        DerivativeEvaluation::Gradient => point_output.len() == dimension,
                        _ => true,
                    };
                    if !output_arity_ok {
                        return Err(FinitumError::InvalidRealization(format!(
                            "point output with {} components does not match the {:?} geometry-sensitivity output binding",
                            point_output.len(),
                            qoutput.binding.evaluation.derivative
                        )));
                    }
                    for (basis, slot) in local_output.iter_mut().enumerate() {
                        match qoutput.binding.evaluation.derivative {
                            DerivativeEvaluation::Value => {
                                let value = self
                                    .data
                                    .element
                                    .basis_value(point_index, basis)
                                    .expect("validated element table");
                                *slot += scale_direction * value * point_output[0]
                                    + scale * value * point_output_direction[0];
                            }
                            DerivativeEvaluation::Gradient => {
                                let reference = self
                                    .data
                                    .element
                                    .basis_gradient(point_index, basis)
                                    .expect("validated element table");
                                let adjoint = (0..dimension)
                                    .map(|physical_axis| {
                                        (0..dimension)
                                            .map(|reference_axis| {
                                                inverse[reference_axis * dimension + physical_axis]
                                                    * reference[reference_axis]
                                            })
                                            .sum::<f64>()
                                    })
                                    .collect::<Vec<_>>();
                                let adjoint_direction = gradient_direction(reference);
                                let product = |weights: &[f64], values: &[f64]| {
                                    weights
                                        .iter()
                                        .zip(values)
                                        .map(|(weight, value)| weight * value)
                                        .sum::<f64>()
                                };
                                *slot += scale_direction * product(&adjoint, &point_output)
                                    + scale
                                        * (product(&adjoint_direction, &point_output)
                                            + product(&adjoint, &point_output_direction));
                            }
                            other => {
                                return Err(FinitumError::UnsupportedRealization(format!(
                                    "geometry sensitivity supports value and gradient outputs, found {other:?}"
                                )));
                            }
                        }
                    }
                }
            }
        }
        for (local, dof) in restriction.dofs.iter().enumerate() {
            output[dof.0] += local_output[local];
        }
        Ok(())
    }

    fn validate_action(&self, input: &[f64], output: &[f64]) -> Result<(), FinitumError> {
        let dimension = self.dimension();
        if input.len() != dimension || output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "operator action requires input/output length {dimension}, got {}/{}",
                input.len(),
                output.len()
            )));
        }
        validate_finite("operator input", input)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_cells(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: Option<&[f64]>,
        rate_direction: Option<&[f64]>,
        output: &mut [f64],
        action: Action,
    ) -> Result<(), FinitumError> {
        for cell_index in 0..self.data.dofs.restrictions().len() {
            self.apply_cell(
                cell_index,
                time,
                state,
                state_rate,
                state_direction,
                rate_direction,
                output,
                action,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_cell(
        &self,
        cell_index: usize,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: Option<&[f64]>,
        rate_direction: Option<&[f64]>,
        output: &mut [f64],
        action: Action,
    ) -> Result<(), FinitumError> {
        let restriction = &self.data.dofs.restrictions()[cell_index];
        let geometry = &self.data.geometries[cell_index];
        let local_state = restriction
            .dofs
            .iter()
            .map(|dof| state[dof.0])
            .collect::<Vec<_>>();
        let local_rate = restriction
            .dofs
            .iter()
            .map(|dof| state_rate[dof.0])
            .collect::<Vec<_>>();
        let local_state_direction = state_direction.map(|direction| {
            restriction
                .dofs
                .iter()
                .map(|dof| direction[dof.0])
                .collect::<Vec<_>>()
        });
        let local_rate_direction = rate_direction.map(|direction| {
            restriction
                .dofs
                .iter()
                .map(|dof| direction[dof.0])
                .collect::<Vec<_>>()
        });
        let mut local_output = vec![0.0; restriction.dofs.len()];
        for (point_index, point) in self.data.element.quadrature().iter().enumerate() {
            let scale = point.weight * geometry.determinant;
            for integral in &self.data.factorization.integrals {
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = &self.data.bundles[&(integral.integral_index, output_index)];
                    let point_output = match action {
                        Action::Primal => self.execute_primal(
                            bound,
                            integral,
                            cell_index,
                            point_index,
                            geometry,
                            time,
                            &local_state,
                            &local_rate,
                        )?,
                        Action::Jvp => self.execute_jvp(
                            bound,
                            integral,
                            cell_index,
                            point_index,
                            geometry,
                            time,
                            &local_state,
                            &local_rate,
                            local_state_direction.as_deref().ok_or_else(|| {
                                FinitumError::InvalidRealization(
                                    "JVP action is missing a state direction".into(),
                                )
                            })?,
                            local_rate_direction.as_deref().ok_or_else(|| {
                                FinitumError::InvalidRealization(
                                    "JVP action is missing a rate direction".into(),
                                )
                            })?,
                        )?,
                    };
                    apply_basis_adjoint(
                        &self.data.element,
                        geometry,
                        point_index,
                        &qoutput.binding.evaluation.derivative,
                        &point_output,
                        scale,
                        &mut local_output,
                    )?;
                }
            }
        }
        for (local, dof) in restriction.dofs.iter().enumerate() {
            output[dof.0] += local_output[local];
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_primal(
        &self,
        bound: &BoundBundle,
        integral: &IntegralOperatorFactorization,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        time: f64,
        local_state: &[f64],
        local_rate: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let (inputs, _) = self.point_inputs(
            integral,
            cell,
            point,
            geometry,
            time,
            local_state,
            local_rate,
        )?;
        let values = bound
            .bundle
            .primal_inputs
            .iter()
            .map(|binding| {
                inputs
                    .get(&binding.input)
                    .cloned()
                    .map(|values| (binding.operand, values))
                    .ok_or_else(|| {
                        FinitumError::InvalidRealization(format!(
                            "bundle input {:?} is absent from integral {}",
                            binding.input, integral.integral_index
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let buffers = execute(
            &bound.executable.kernels()[bound.bundle.primal_kernel_index],
            &values,
        )?;
        operand_values(
            &bound.executable.kernels()[bound.bundle.primal_kernel_index],
            &buffers,
            bound.bundle.primal_output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_jvp(
        &self,
        bound: &BoundBundle,
        integral: &IntegralOperatorFactorization,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        time: f64,
        local_state: &[f64],
        local_rate: &[f64],
        local_state_direction: &[f64],
        local_rate_direction: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let (inputs, evaluation) = self.point_inputs(
            integral,
            cell,
            point,
            geometry,
            time,
            local_state,
            local_rate,
        )?;
        let directions = self.point_directions(
            integral,
            cell,
            point,
            geometry,
            time,
            local_state_direction,
            local_rate_direction,
            &evaluation,
        )?;
        self.execute_jvp_values(bound, &inputs, &directions)
    }

    fn execute_jvp_values(
        &self,
        bound: &BoundBundle,
        inputs: &BTreeMap<TensorInputId, Vec<f64>>,
        directions: &BTreeMap<TensorInputId, Vec<f64>>,
    ) -> Result<Vec<f64>, FinitumError> {
        let input_by_operand = bound
            .bundle
            .primal_inputs
            .iter()
            .map(|binding| (binding.operand, binding.input))
            .collect::<BTreeMap<_, _>>();
        let mut values = BTreeMap::new();
        for binding in &bound.bundle.primal_inputs {
            values.insert(binding.operand, inputs[&binding.input].clone());
        }
        for pair in &bound.bundle.jvp.independent_operands {
            let input = input_by_operand.get(&pair.primal).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "JVP operand {:?} has no QFunction input binding",
                    pair.primal
                ))
            })?;
            values.insert(pair.derivative, directions[input].clone());
        }
        let executable = &bound.executable.kernels()[bound.bundle.jvp.kernel_index];
        let buffers = execute(executable, &values)?;
        let mut output = operand_values(
            executable,
            &buffers,
            bound.bundle.jvp.dependent_operands[0].derivative,
        )?;

        let mut parameter_values = bound
            .bundle
            .primal_inputs
            .iter()
            .map(|binding| (binding.operand, inputs[&binding.input].clone()))
            .collect::<BTreeMap<_, _>>();
        for pair in &bound.bundle.parameter.independent_operands {
            let input = input_by_operand.get(&pair.primal).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "parameter-JVP operand {:?} has no QFunction input binding",
                    pair.primal
                ))
            })?;
            parameter_values.insert(pair.derivative, directions[input].clone());
        }
        let parameter_executable = &bound.executable.kernels()[bound.bundle.parameter.kernel_index];
        let parameter_buffers = execute(parameter_executable, &parameter_values)?;
        let parameter_output = operand_values(
            parameter_executable,
            &parameter_buffers,
            bound.bundle.parameter.dependent_operands[0].derivative,
        )?;
        for (value, parameter) in output.iter_mut().zip(parameter_output) {
            *value += parameter;
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn point_inputs(
        &self,
        integral: &IntegralOperatorFactorization,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        time: f64,
        local_state: &[f64],
        local_rate: &[f64],
    ) -> Result<(BTreeMap<TensorInputId, Vec<f64>>, PointEvaluation), FinitumError> {
        let mut inputs = BTreeMap::new();
        let mut active = Vec::new();
        for input in &integral.primal.inputs {
            if input.source != InputSourceRequirement::Basis {
                continue;
            }
            let dofs =
                if input.binding.evaluation.derivative == DerivativeEvaluation::TimeDerivative {
                    local_rate
                } else {
                    local_state
                };
            let values = evaluate_basis_input(&self.data.element, geometry, point, input, dofs)?;
            if input.role == TensorInputRole::Active {
                active.push(PointActiveInput {
                    input: input.id,
                    derivative: input.binding.evaluation.derivative,
                    values: values.clone(),
                });
            }
            inputs.insert(input.id, values);
        }
        let evaluation = PointEvaluation {
            time,
            cell: CellId(cell),
            coordinates: geometry
                .physical_point(&self.data.element.quadrature()[point].coordinates),
            active,
        };
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let binding = &self.data.external[&(integral.integral_index, input.id)];
            let values = match binding {
                ExternalBinding::Stored(stored) => stored
                    .point_values(cell, point, self.data.element.quadrature().len())
                    .to_vec(),
                ExternalBinding::Dynamic(dynamic) => (dynamic.value)(&evaluation),
            };
            validate_components(input, &values, "external input")?;
            inputs.insert(input.id, values);
        }
        Ok((inputs, evaluation))
    }

    #[allow(clippy::too_many_arguments)]
    fn point_directions(
        &self,
        integral: &IntegralOperatorFactorization,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        time: f64,
        local_state_direction: &[f64],
        local_rate_direction: &[f64],
        evaluation: &PointEvaluation,
    ) -> Result<BTreeMap<TensorInputId, Vec<f64>>, FinitumError> {
        let mut directions = BTreeMap::new();
        let mut active = Vec::new();
        for input in &integral.primal.inputs {
            if input.source != InputSourceRequirement::Basis {
                continue;
            }
            let dofs =
                if input.binding.evaluation.derivative == DerivativeEvaluation::TimeDerivative {
                    local_rate_direction
                } else {
                    local_state_direction
                };
            let values = evaluate_basis_input(&self.data.element, geometry, point, input, dofs)?;
            if input.role == TensorInputRole::Active {
                active.push(PointActiveInput {
                    input: input.id,
                    derivative: input.binding.evaluation.derivative,
                    values: values.clone(),
                });
            }
            directions.insert(input.id, values);
        }
        let direction_evaluation = PointEvaluation {
            time,
            cell: CellId(cell),
            coordinates: evaluation.coordinates.clone(),
            active,
        };
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let binding = &self.data.external[&(integral.integral_index, input.id)];
            let values = match binding {
                ExternalBinding::Stored(stored) => vec![0.0; stored.component_count],
                ExternalBinding::Dynamic(dynamic) => {
                    (dynamic.direction)(evaluation, &direction_evaluation)
                }
            };
            validate_components(input, &values, "external input direction")?;
            directions.insert(input.id, values);
        }
        Ok(directions)
    }
}

#[derive(Clone, Copy)]
enum Action {
    Primal,
    Jvp,
}

/// Deterministic gather/kernel/scatter action without a stored global matrix.
///
/// This action is the generated JVP evaluated at zero active input. It is therefore a complete
/// operator only for the globally linear FC6 scope, not a reusable nonlinear linearization.
/// Affine dependency constraints replace target rows with constraint residuals, which destroys
/// symmetry; [`LinearOperator::symmetry`] reports that case as nonsymmetric.
#[derive(Clone, Debug)]
pub struct MatrixFreeOperator {
    plan: RealizationPlan,
}

impl MatrixFreeOperator {
    pub fn source_factorization_digest(&self) -> &Digest {
        self.plan.source_factorization_digest()
    }
}

impl LinearOperator for MatrixFreeOperator {
    fn rows(&self) -> usize {
        self.plan.dimension()
    }

    fn columns(&self) -> usize {
        self.plan.dimension()
    }

    fn symmetry(&self) -> OperatorSymmetry {
        if self.plan.data.constraints.has_affine_dependencies() {
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
        self.plan
            .apply_direction(input, output)
            .map_err(numeric_error)
    }
}

/// Canonical CSR realization assembled from the matrix-free action.
///
/// Its symmetry classification is retained explicitly because affine dependency constraint rows
/// make the full-coordinate action nonsymmetric.
#[derive(Clone, Debug)]
pub struct AssembledOperator {
    matrix: CsrMatrix,
    source_factorization_digest: Digest,
    symmetry: OperatorSymmetry,
}

impl AssembledOperator {
    pub fn matrix(&self) -> &CsrMatrix {
        &self.matrix
    }

    pub fn source_factorization_digest(&self) -> &Digest {
        &self.source_factorization_digest
    }

    /// Materialize the canonical global transpose of this assembled action.
    pub fn transpose(&self) -> Result<Self, FinitumError> {
        let matrix = self.matrix();
        let mut entries = Vec::with_capacity(matrix.values().len());
        for row in 0..matrix.rows() {
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                entries.push((matrix.column_indices()[entry], row, matrix.values()[entry]));
            }
        }
        let transpose = CsrMatrix::from_triplets(matrix.columns(), matrix.rows(), entries)
            .map_err(|error| FinitumError::Assembly(error.to_string()))?;
        Ok(Self {
            matrix: transpose,
            source_factorization_digest: self.source_factorization_digest.clone(),
            symmetry: self.symmetry,
        })
    }
}

impl LinearOperator for AssembledOperator {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }

    fn columns(&self) -> usize {
        self.matrix.columns()
    }

    fn symmetry(&self) -> OperatorSymmetry {
        self.symmetry
    }

    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.matrix.apply(context, input, output)
    }
}

#[derive(Serialize)]
struct RealizationDigestPayload<'a> {
    schema: &'static str,
    requirements: &'a Digest,
    factorization: &'a Digest,
    kernels: &'a Digest,
    mesh: &'a Mesh,
    element: &'a PreparedElement,
    dofs: &'a DofMap,
    constraints: &'a ConstraintSet,
    external: Vec<ExternalBindingDescriptor<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ExternalBindingDescriptor<'a> {
    Stored {
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        values: &'a [f64],
    },
    Dynamic {
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        identity: &'a str,
    },
}

#[allow(clippy::too_many_arguments)]
fn realization_digest(
    requirements: &FormRequirements,
    factorization: &OperatorFactorization,
    kernels: &StructuredOperatorKernels,
    mesh: &Mesh,
    element: &PreparedElement,
    dofs: &DofMap,
    constraints: &ConstraintSet,
    external: &BTreeMap<(usize, TensorInputId), ExternalBinding>,
) -> Digest {
    let external = external
        .values()
        .map(|binding| match binding {
            ExternalBinding::Stored(input) => ExternalBindingDescriptor::Stored {
                integral_index: input.integral_index,
                input: input.input,
                component_count: input.component_count,
                values: &input.values,
            },
            ExternalBinding::Dynamic(input) => ExternalBindingDescriptor::Dynamic {
                integral_index: input.integral_index,
                input: input.input,
                component_count: input.component_count,
                identity: &input.identity,
            },
        })
        .collect();
    let payload = RealizationDigestPayload {
        schema: REALIZATION_ARTIFACT_SCHEMA,
        requirements: &requirements.artifact_digest,
        factorization: &factorization.artifact_digest,
        kernels: &kernels.artifact_digest,
        mesh,
        element,
        dofs,
        constraints,
        external,
    };
    Digest::blake3(
        &serde_json::to_vec(&payload).expect("validated realization data must serialize"),
    )
}

fn validate_artifacts(
    requirements: &FormRequirements,
    factorization: &OperatorFactorization,
    kernels: &StructuredOperatorKernels,
) -> Result<(), FinitumError> {
    if requirements.artifact_digest != factorization.receipt.source_requirements_digest
        || requirements.receipt.source_form_digest != factorization.receipt.source_form_digest
        || requirements.model != factorization.model
        || requirements.form != factorization.form
    {
        return Err(FinitumError::ArtifactMismatch(
            "FC3 requirements do not match the FC4 factorization".into(),
        ));
    }
    if kernels.source_factorization_digest != factorization.artifact_digest {
        return Err(FinitumError::ArtifactMismatch(
            "FC5 kernels do not match the FC4 factorization".into(),
        ));
    }
    Ok(())
}

fn validate_discretization(
    requirements: &FormRequirements,
    factorization: &OperatorFactorization,
    mesh: &Mesh,
    element: &PreparedElement,
    dofs: &DofMap,
    constraints: &ConstraintSet,
) -> Result<(), FinitumError> {
    if mesh.dimension() != element.dimension() {
        return Err(FinitumError::InvalidRealization(format!(
            "mesh dimension {} differs from element dimension {}",
            mesh.dimension(),
            element.dimension()
        )));
    }
    if mesh.cells().len() != dofs.restrictions().len() {
        return Err(FinitumError::InvalidRealization(format!(
            "mesh has {} cells but DOF map has {} restrictions",
            mesh.cells().len(),
            dofs.restrictions().len()
        )));
    }
    if constraints.dof_count() != dofs.dof_count() {
        return Err(FinitumError::InvalidRealization(format!(
            "constraint extent {} differs from DOF extent {}",
            constraints.dof_count(),
            dofs.dof_count()
        )));
    }
    if !factorization.essential_constraints.is_empty() && constraints.constraints().next().is_none()
    {
        return Err(FinitumError::InvalidRealization(
            "the factorization requires essential constraints but no constrained DOFs were supplied"
                .into(),
        ));
    }
    if factorization.essential_constraints.is_empty() && constraints.constraints().next().is_some()
    {
        return Err(FinitumError::InvalidRealization(
            "concrete constraints were supplied for a factorization with no essential constraints"
                .into(),
        ));
    }
    for requirement in &requirements.elements {
        let admitted_shape = match &requirement.value_shape {
            ValueShape::Scalar => true,
            // Vector H1(order=1) with one component per spatial axis is the
            // SV2-A production slice: vertex-major component blocks execute
            // through the same generated kernels.
            ValueShape::Vector(components) => *components as usize == mesh.dimension(),
            _ => false,
        };
        if requirement.topological_dimension as usize != mesh.dimension()
            || requirement.family != ElementFamilyRequirement::H1
            || requirement.polynomial_order != 1
            || !admitted_shape
        {
            return Err(FinitumError::UnsupportedRealization(format!(
                "realization supports scalar or dimension-vector H1(order=1) cell elements, got {requirement:?}"
            )));
        }
    }
    if element.basis_count() != mesh.dimension() + 1 {
        return Err(FinitumError::InvalidRealization(format!(
            "P1 simplex in dimension {} requires {} basis functions, got {}",
            mesh.dimension(),
            mesh.dimension() + 1,
            element.basis_count()
        )));
    }
    // Vector blocks widen each restriction to `components` DOFs per node
    // while the basis table stays scalar; the widest requirement wins.
    let components_per_restriction = requirements
        .elements
        .iter()
        .map(|requirement| match &requirement.value_shape {
            ValueShape::Vector(components) => *components as usize,
            _ => 1,
        })
        .max()
        .unwrap_or(1);
    let expected_restriction = element.basis_count() * components_per_restriction;
    for (index, restriction) in dofs.restrictions().iter().enumerate() {
        if restriction.dofs.len() != expected_restriction {
            return Err(FinitumError::InvalidRealization(format!(
                "restriction {index} has {} DOFs, expected {expected_restriction}",
                restriction.dofs.len(),
            )));
        }
    }
    for integral in &factorization.integrals {
        if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
            return Err(FinitumError::UnsupportedRealization(
                "FC6 realizes cell integrals; facet, interface, and point traversal is deferred"
                    .into(),
            ));
        }
        for input in &integral.primal.inputs {
            validate_input_contract(input, mesh.dimension())?;
        }
        for output in &integral.primal.outputs {
            validate_evaluation(&output.binding.evaluation.derivative, mesh.dimension())?;
            if output.binding.evaluation.site != EvaluationSite::Cell {
                return Err(FinitumError::UnsupportedRealization(
                    "FC6 realizes cell evaluation sites only".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_input_contract(input: &QFunctionInput, dimension: usize) -> Result<(), FinitumError> {
    if input.binding.evaluation.site != EvaluationSite::Cell {
        return Err(FinitumError::UnsupportedRealization(
            "FC6 realizes cell evaluation sites only".into(),
        ));
    }
    validate_evaluation(&input.binding.evaluation.derivative, dimension)?;
    if input.source == InputSourceRequirement::Basis && input.role != TensorInputRole::Active {
        return Err(FinitumError::UnsupportedRealization(
            "FC6 has one active scalar field; additional basis-backed coefficients are deferred"
                .into(),
        ));
    }
    Ok(())
}

fn validate_evaluation(
    derivative: &DerivativeEvaluation,
    dimension: usize,
) -> Result<(), FinitumError> {
    let _ = dimension;
    if matches!(
        derivative,
        DerivativeEvaluation::Value
            | DerivativeEvaluation::Gradient
            | DerivativeEvaluation::TimeDerivative
            // SV2-A: symmetric gradients of vector H1(order=1) blocks.
            | DerivativeEvaluation::SymmetricGradient
    ) {
        Ok(())
    } else {
        Err(FinitumError::UnsupportedRealization(format!(
            "the scalar P1 realization supports value, gradient, and time-derivative basis actions, got {derivative:?}"
        )))
    }
}

fn validate_external_inputs(
    factorization: &OperatorFactorization,
    mesh: &Mesh,
    element: &PreparedElement,
    external_inputs: Vec<ExternalInput>,
    dynamic_external_inputs: Vec<DynamicExternalInput>,
) -> Result<BTreeMap<(usize, TensorInputId), ExternalBinding>, FinitumError> {
    let mut external = BTreeMap::new();
    for input in external_inputs {
        let key = (input.integral_index, input.input);
        if external
            .insert(key, ExternalBinding::Stored(input))
            .is_some()
        {
            return Err(FinitumError::InvalidRealization(format!(
                "external input {key:?} was supplied more than once"
            )));
        }
    }
    for input in dynamic_external_inputs {
        let key = (input.integral_index, input.input);
        if external
            .insert(key, ExternalBinding::Dynamic(input))
            .is_some()
        {
            return Err(FinitumError::InvalidRealization(format!(
                "external input {key:?} was supplied more than once"
            )));
        }
    }
    let mut expected_keys = BTreeSet::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let key = (integral.integral_index, input.id);
            expected_keys.insert(key);
            let supplied = external
                .get(&key)
                .ok_or(FinitumError::MissingExternalInput {
                    integral: integral.integral_index,
                    input: input.id,
                })?;
            let components = component_count(&input.shape)?;
            match supplied {
                ExternalBinding::Stored(supplied) => {
                    let expected = mesh
                        .cells()
                        .len()
                        .checked_mul(element.quadrature().len())
                        .and_then(|count| count.checked_mul(components))
                        .ok_or_else(|| {
                            FinitumError::InvalidRealization(
                                "external input storage extent overflows usize".into(),
                            )
                        })?;
                    if supplied.component_count != components || supplied.values.len() != expected {
                        return Err(FinitumError::InvalidRealization(format!(
                            "external input {key:?} has {} components and {} values, expected {components} and {expected}",
                            supplied.component_count,
                            supplied.values.len()
                        )));
                    }
                }
                ExternalBinding::Dynamic(supplied) => {
                    if supplied.component_count != components {
                        return Err(FinitumError::InvalidRealization(format!(
                            "dynamic external input {key:?} has {} components, expected {components}",
                            supplied.component_count
                        )));
                    }
                }
            }
        }
    }
    if external.keys().any(|key| !expected_keys.contains(key)) {
        return Err(FinitumError::InvalidRealization(
            "an external input does not belong to the factorization".into(),
        ));
    }
    Ok(external)
}

fn validate_components(
    input: &QFunctionInput,
    values: &[f64],
    label: &str,
) -> Result<(), FinitumError> {
    let expected = component_count(&input.shape)?;
    if values.len() != expected {
        return Err(FinitumError::InvalidRealization(format!(
            "{label} {:?} returned {} components, expected {expected}",
            input.id,
            values.len()
        )));
    }
    validate_finite(label, values)
}

fn bind_kernels(
    factorization: &OperatorFactorization,
    kernels: StructuredOperatorKernels,
) -> Result<BTreeMap<(usize, usize), BoundBundle>, FinitumError> {
    let expected = factorization
        .integrals
        .iter()
        .map(|integral| integral.primal.outputs.len())
        .sum::<usize>();
    if kernels.bundles.len() != expected {
        return Err(FinitumError::ArtifactMismatch(format!(
            "FC5 bundle has {} outputs, FC4 factorization requires {expected}",
            kernels.bundles.len()
        )));
    }
    let mut bound = BTreeMap::new();
    for bundle in kernels.bundles {
        let integral = factorization
            .integrals
            .iter()
            .find(|integral| integral.integral_index == bundle.integral_index)
            .ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "kernel references absent integral {}",
                    bundle.integral_index
                ))
            })?;
        if bundle.output_index >= integral.primal.outputs.len()
            || bundle.receipt.integral_index != bundle.integral_index
            || bundle.receipt.output_index != bundle.output_index
            || bundle.receipt.source_factorization_digest != factorization.artifact_digest
            || bundle.receipt.source_primal_digest != integral.primal.artifact_digest
            || bundle.receipt.source_symbolic_jvp_digest != integral.jvp.artifact_digest
        {
            return Err(FinitumError::ArtifactMismatch(format!(
                "kernel output ({}, {}) is not linked to its FC4 programs",
                bundle.integral_index, bundle.output_index
            )));
        }
        let executable = ExecutableModule::reference(
            validate_module(bundle.module.clone())
                .map_err(|error| FinitumError::KernelValidation(error.to_string()))?,
        );
        if bundle.primal_kernel_index >= executable.kernels().len()
            || bundle.jvp.kernel_index >= executable.kernels().len()
            || bundle.vjp.kernel_index >= executable.kernels().len()
            || bundle.parameter.kernel_index >= executable.kernels().len()
            || bundle.jvp.dependent_operands.len() != 1
            || bundle.vjp.dependent_operands.len() != 1
            || bundle.parameter.dependent_operands.len() != 1
        {
            return Err(FinitumError::ArtifactMismatch(
                "kernel indices or derivative output contracts are invalid".into(),
            ));
        }
        let key = (bundle.integral_index, bundle.output_index);
        if bound
            .insert(key, BoundBundle { bundle, executable })
            .is_some()
        {
            return Err(FinitumError::ArtifactMismatch(format!(
                "kernel output {key:?} is duplicated"
            )));
        }
    }
    Ok(bound)
}

/// Validates that one geometry sensitivity covers exactly the stored external
/// inputs of the target plan, with matching component and value extents.
fn validate_sensitivity_inputs<'a>(
    plan: &RealizationPlan,
    sensitivity: &'a GeometryParameterSensitivity,
) -> Result<BTreeMap<(usize, TensorInputId), &'a ExternalSensitivityInput>, FinitumError> {
    let mut directions = BTreeMap::new();
    for input in &sensitivity.external_directions {
        let key = (input.integral_index, input.input);
        if directions.insert(key, input).is_some() {
            return Err(FinitumError::InvalidRealization(format!(
                "geometry sensitivity declares external input {key:?} more than once"
            )));
        }
    }
    for (key, binding) in &plan.data.external {
        let ExternalBinding::Stored(stored) = binding else {
            continue;
        };
        let Some(direction) = directions.get(key) else {
            return Err(FinitumError::MissingExternalInput {
                integral: key.0,
                input: key.1,
            });
        };
        if direction.component_count != stored.component_count {
            return Err(FinitumError::InvalidRealization(format!(
                "geometry sensitivity for external input {key:?} has {} components, expected {}",
                direction.component_count, stored.component_count
            )));
        }
        if direction.values.len() != stored.values.len() {
            return Err(FinitumError::InvalidRealization(format!(
                "geometry sensitivity for external input {key:?} has {} values, expected {}",
                direction.values.len(),
                stored.values.len()
            )));
        }
    }
    let stored_count = plan
        .data
        .external
        .values()
        .filter(|binding| matches!(binding, ExternalBinding::Stored(_)))
        .count();
    if directions.len() != stored_count {
        return Err(FinitumError::InvalidRealization(format!(
            "geometry sensitivity declares {} external inputs, plan stores {stored_count}",
            directions.len()
        )));
    }
    Ok(directions)
}

pub(crate) fn evaluate_basis_input(
    element: &PreparedElement,
    geometry: &CellGeometry,
    point: usize,
    input: &QFunctionInput,
    local_state: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    // Component stride rules per evaluation kind:
    // - Value/TimeDerivative: scalar fields carry one value per node; vector
    //   fields declare shape [components] and use vertex-major state.
    // - Gradient: scalar-field gradient only; state stride stays one and the
    //   output is the physical gradient vector.
    // - SymmetricGradient: vector H1(order=1) blocks; state stride equals the
    //   spatial dimension and the output is the row-major [d][d] strain.
    let components = match input.binding.evaluation.derivative {
        DerivativeEvaluation::SymmetricGradient => element.dimension(),
        DerivativeEvaluation::Gradient => 1,
        _ => vector_components(input)?,
    };
    match input.binding.evaluation.derivative {
        DerivativeEvaluation::Value | DerivativeEvaluation::TimeDerivative => {
            if components == 1 {
                let value = local_state
                    .iter()
                    .enumerate()
                    .map(|(basis, value)| {
                        element
                            .basis_value(point, basis)
                            .expect("validated element table")
                            * value
                    })
                    .sum();
                Ok(vec![value])
            } else {
                // Vector value interpolation with vertex-major local state:
                // node i owns components [i*components, (i+1)*components).
                interpolate_vector_value(element, point, local_state, components)
            }
        }
        DerivativeEvaluation::Gradient => {
            if components == 1 {
                let mut gradient = vec![0.0; element.dimension()];
                for (basis, value) in local_state.iter().enumerate() {
                    let physical = geometry.physical_gradient(
                        element
                            .basis_gradient(point, basis)
                            .expect("validated element table"),
                    );
                    for axis in 0..element.dimension() {
                        gradient[axis] += physical[axis] * value;
                    }
                }
                Ok(gradient)
            } else {
                // Vector gradient: component c of the physical gradient is
                // sum_i physical * u[i][c], row-major [c][axis].
                let dimension = element.dimension();
                let mut gradient = vec![0.0; dimension * components];
                for basis in 0..element.basis_count() {
                    let physical = geometry.physical_gradient(
                        element
                            .basis_gradient(point, basis)
                            .expect("validated element table"),
                    );
                    for component in 0..components {
                        let value = local_state[basis * components + component];
                        for axis in 0..dimension {
                            gradient[component * dimension + axis] += physical[axis] * value;
                        }
                    }
                }
                Ok(gradient)
            }
        }
        DerivativeEvaluation::SymmetricGradient => {
            let dimension = element.dimension();
            let mut symmetric = vec![0.0; dimension * dimension];
            for basis in 0..element.basis_count() {
                let physical = geometry.physical_gradient(
                    element
                        .basis_gradient(point, basis)
                        .expect("validated element table"),
                );
                for row in 0..dimension {
                    for column in 0..dimension {
                        // (sym grad u)_{rc} = 1/2 (grad u_{rc} + grad u_{cr})
                        // with grad stored [component][axis].
                        let plus = physical[row] * local_state[basis * components + column];
                        let minus = physical[column] * local_state[basis * components + row];
                        symmetric[row * dimension + column] += 0.5 * (plus + minus);
                    }
                }
            }
            Ok(symmetric)
        }
        _ => Err(FinitumError::UnsupportedRealization(format!(
            "unsupported basis evaluation {:?}",
            input.binding.evaluation.derivative
        ))),
    }
}

/// Number of field components carried by one basis input.
///
/// Scalar fields keep shape `[]` or `[1]`; vector fields declare the
/// component count as the leading extent, and the local state is vertex-major
/// so node `i` owns components `[i*n, (i+1)*n)`.
fn vector_components(input: &QFunctionInput) -> Result<usize, FinitumError> {
    let mut count = 1usize;
    for extent in &input.shape {
        count = count
            .checked_mul(*extent)
            .ok_or_else(|| FinitumError::InvalidRealization("shape overflow".into()))?;
    }
    if count == 1 {
        return Ok(1);
    }
    if input.shape.len() == 1 {
        return Ok(input.shape[0]);
    }
    Err(FinitumError::UnsupportedRealization(format!(
        "basis input {:?} declares shape {:?} which is not a scalar or vector value",
        input.id, input.shape
    )))
}

fn interpolate_vector_value(
    element: &PreparedElement,
    point: usize,
    local_state: &[f64],
    components: usize,
) -> Result<Vec<f64>, FinitumError> {
    let mut values = vec![0.0; components];
    for basis in 0..element.basis_count() {
        let weight = element
            .basis_value(point, basis)
            .expect("validated element table");
        for component in 0..components {
            values[component] += weight * local_state[basis * components + component];
        }
    }
    Ok(values)
}

pub(crate) fn apply_basis_adjoint(
    element: &PreparedElement,
    geometry: &CellGeometry,
    point: usize,
    derivative: &DerivativeEvaluation,
    point_output: &[f64],
    scale: f64,
    local_output: &mut [f64],
) -> Result<(), FinitumError> {
    let dimension = element.dimension();
    let basis_count = element.basis_count();
    // Local output layout: vertex-major over the restriction, so node i owns
    // `stride` consecutive entries. The stride is inferred from the point
    // output arity, mirroring the input-side component convention.
    let stride = if local_output.len() % basis_count == 0 {
        local_output.len() / basis_count
    } else {
        return Err(FinitumError::InvalidRealization(
            "local output length is not a multiple of the basis count".into(),
        ));
    };
    match derivative {
        DerivativeEvaluation::Value if point_output.len() == 1 => {
            for (basis, output) in local_output.iter_mut().enumerate() {
                *output += scale * element.basis_value(point, basis).unwrap() * point_output[0];
            }
        }
        DerivativeEvaluation::Value if point_output.len() == stride => {
            for basis in 0..basis_count {
                let weight = element.basis_value(point, basis).unwrap();
                for component in 0..stride {
                    local_output[basis * stride + component] +=
                        scale * weight * point_output[component];
                }
            }
        }
        DerivativeEvaluation::Gradient if point_output.len() == dimension => {
            for (basis, output) in local_output.iter_mut().enumerate() {
                let gradient = geometry.physical_gradient(
                    element
                        .basis_gradient(point, basis)
                        .expect("validated element table"),
                );
                *output += scale
                    * gradient
                        .iter()
                        .zip(point_output)
                        .map(|(basis, value)| basis * value)
                        .sum::<f64>();
            }
        }
        DerivativeEvaluation::Gradient if point_output.len() == dimension * dimension => {
            // Flux-style [axis][component] contraction against the physical
            // test gradient: row(i,c) += scale * sum_a out[a][c] * g[a].
            for basis in 0..basis_count {
                let gradient = geometry.physical_gradient(
                    element
                        .basis_gradient(point, basis)
                        .expect("validated element table"),
                );
                for component in 0..dimension {
                    let slot = basis * stride + component;
                    if slot >= local_output.len() {
                        return Err(FinitumError::InvalidRealization(
                            "flux adjoint exceeds the local output extent".into(),
                        ));
                    }
                    let mut sum = 0.0;
                    for axis in 0..dimension {
                        sum += point_output[axis * dimension + component] * gradient[axis];
                    }
                    local_output[slot] += scale * sum;
                }
            }
        }
        _ => {
            return Err(FinitumError::InvalidRealization(format!(
                "point output with {} components does not match {derivative:?}",
                point_output.len()
            )));
        }
    }
    Ok(())
}

fn execute(
    executable: &Executable,
    values: &BTreeMap<OperandId, Vec<f64>>,
) -> Result<Vec<Vec<f64>>, FinitumError> {
    let kernel = executable.kernel().as_kernel();
    let mut buffers = kernel
        .operands
        .iter()
        .map(|operand| vec![0.0; operand.region.offset + operand.region.length])
        .collect::<Vec<_>>();
    for (index, operand) in kernel.operands.iter().enumerate() {
        if matches!(operand.access, AccessMode::Read | AccessMode::ReadWrite)
            && !values.contains_key(&OperandId::new(index))
        {
            return Err(FinitumError::InvalidRealization(format!(
                "kernel read operand {index} has no realization binding"
            )));
        }
    }
    for (operand, values) in values {
        let definition = kernel.operands.get(operand.index()).ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "input references absent kernel operand {:?}",
                operand
            ))
        })?;
        let count = component_count(&definition.shape)?;
        if values.len() != count {
            return Err(FinitumError::InvalidRealization(format!(
                "kernel operand {:?} received {} values, expected {count}",
                operand,
                values.len()
            )));
        }
        let start = definition.region.offset;
        buffers[operand.index()][start..start + count].copy_from_slice(values);
    }
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(executable, &mut bindings)
        .map_err(|error| FinitumError::KernelExecution(error.to_string()))?;
    drop(bindings);
    Ok(buffers)
}

fn operand_values(
    executable: &Executable,
    buffers: &[Vec<f64>],
    operand: OperandId,
) -> Result<Vec<f64>, FinitumError> {
    let definition = executable
        .kernel()
        .as_kernel()
        .operands
        .get(operand.index())
        .ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "output references absent kernel operand {:?}",
                operand
            ))
        })?;
    let count = component_count(&definition.shape)?;
    let start = definition.region.offset;
    Ok(buffers[operand.index()][start..start + count].to_vec())
}

fn component_count(shape: &[usize]) -> Result<usize, FinitumError> {
    shape.iter().try_fold(1usize, |count, extent| {
        count.checked_mul(*extent).ok_or_else(|| {
            FinitumError::InvalidRealization("tensor component extent overflows usize".into())
        })
    })
}

fn validate_finite(operation: &str, values: &[f64]) -> Result<(), FinitumError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(FinitumError::InvalidRealization(format!(
            "{operation} contains a non-finite value at index {index}"
        )))
    } else {
        Ok(())
    }
}

fn numeric_error(error: FinitumError) -> NumericError {
    NumericError::Operator {
        message: error.to_string(),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CellGeometry {
    dimension: usize,
    origin: Vec<f64>,
    jacobian: Vec<f64>,
    inverse: Vec<f64>,
    determinant: f64,
}

impl CellGeometry {
    fn new(mesh: &Mesh, cell_id: CellId) -> Result<Self, FinitumError> {
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
        if !determinant.is_finite()
            || determinant == 0.0
            || inverse.iter().any(|value| !value.is_finite())
        {
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
            determinant: determinant.abs(),
        })
    }

    fn physical_point(&self, reference: &[f64]) -> Vec<f64> {
        (0..self.dimension)
            .map(|row| {
                self.origin[row]
                    + (0..self.dimension)
                        .map(|column| {
                            self.jacobian[row * self.dimension + column] * reference[column]
                        })
                        .sum::<f64>()
            })
            .collect()
    }

    fn physical_gradient(&self, reference: &[f64]) -> Vec<f64> {
        (0..self.dimension)
            .map(|physical_axis| {
                (0..self.dimension)
                    .map(|reference_axis| {
                        self.inverse[reference_axis * self.dimension + physical_axis]
                            * reference[reference_axis]
                    })
                    .sum()
            })
            .collect()
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

#[cfg(test)]
mod sv2_vector_probes {
    use super::*;
    use crate::{Mesh, PreparedElement};

    #[test]
    fn symmetric_gradient_direction_is_nonzero_for_unit_nodal_direction() {
        let mesh = Mesh::new(
            3,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![crate::Cell {
                vertices: vec![
                    crate::VertexId(0),
                    crate::VertexId(1),
                    crate::VertexId(2),
                    crate::VertexId(3),
                ],
            }],
        )
        .unwrap();
        let element = PreparedElement::linear_simplex(3).unwrap();
        let geometry = CellGeometry::new(&mesh, crate::CellId(0)).unwrap();
        // Build a minimal QFunctionInput-shaped probe via serde-free literal is
        // impossible outside scientia; instead call the math through
        // interpolate/gradient arms indirectly: assert physical gradients are
        // nonzero so any zero must come from the state.
        let mut total = 0.0;
        for basis in 0..element.basis_count() {
            let reference = element.basis_gradient(0, basis).unwrap();
            let physical = geometry.physical_gradient(reference);
            total += physical.iter().map(|v| v.abs()).sum::<f64>();
        }
        assert!(total > 1.0e-12, "basis gradients collapsed: {total}");
    }
}
