use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use malleus::{
    AccessMode, BufferBinding, Executable, ExecutableModule, Interpreter, OperandId,
    validate_module,
};
use resolvent::scientific::ValueShape;
use resolvent::{
    DerivativeEvaluation, Digest, ElementFamilyRequirement, EvaluationSite, FormRequirements,
    InputSourceRequirement, IntegralOperatorFactorization, OperatorFactorization, QFunctionInput,
    SemanticMeasure, StructuredOperatorKernels, StructuredPointKernelBundle, TensorInputId,
    TensorInputRole,
};
use solverang::{CsrMatrix, EvaluationContext, LinearOperator, NumericError};

use crate::{CellId, ConstraintSet, DofMap, FinitumError, Mesh, PreparedElement};

/// Concrete quadrature-point values for one non-basis QFunction input.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalInput {
    pub integral_index: usize,
    pub input: TensorInputId,
    component_count: usize,
    values: Vec<f64>,
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

#[derive(Clone, Debug)]
struct BoundBundle {
    bundle: StructuredPointKernelBundle,
    executable: ExecutableModule,
}

#[derive(Clone, Debug)]
struct RealizationData {
    requirements: FormRequirements,
    factorization: OperatorFactorization,
    mesh: Mesh,
    element: PreparedElement,
    geometries: Vec<CellGeometry>,
    dofs: DofMap,
    constraints: ConstraintSet,
    external: BTreeMap<(usize, TensorInputId), ExternalInput>,
    bundles: BTreeMap<(usize, usize), BoundBundle>,
}

/// Digest-linked binding of FC3 requirements, an FC4 factorization, FC5 executables, and
/// concrete mesh/element/DOF/constraint/input data.
///
/// FC6 realizes a globally linear operator by evaluating generated JVPs at zero active input.
/// A nonlinear residual must not reuse this plan: its linearization point belongs in the future
/// stateful realization contract. Semantic boundary requirements are linked to the artifact
/// chain, but the caller currently supplies the concrete boundary-DOF membership.
#[derive(Clone, Debug)]
pub struct RealizationPlan {
    data: Arc<RealizationData>,
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
        validate_artifacts(&requirements, &factorization, &kernels)?;
        validate_discretization(
            &requirements,
            &factorization,
            &mesh,
            &element,
            &dofs,
            &constraints,
        )?;
        let external = validate_external_inputs(&factorization, &mesh, &element, external_inputs)?;
        let bundles = bind_kernels(&factorization, kernels)?;
        let geometries = (0..mesh.cells().len())
            .map(|cell| CellGeometry::new(&mesh, CellId(cell)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            data: Arc::new(RealizationData {
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

    pub fn source_factorization_digest(&self) -> &Digest {
        &self.data.factorization.artifact_digest
    }

    pub fn source_requirements_digest(&self) -> &Digest {
        &self.data.requirements.artifact_digest
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
        Ok(AssembledOperator {
            matrix,
            source_factorization_digest: self.source_factorization_digest().clone(),
        })
    }

    /// Build the affine right-hand side from the generated primal kernels and fixed essential
    /// values. No source term is duplicated in Finitum.
    pub fn load_vector(&self) -> Result<Vec<f64>, FinitumError> {
        let mut lifting = vec![0.0; self.dimension()];
        for constraint in self.data.constraints.constraints() {
            lifting[constraint.target.0] = constraint.offset;
        }
        let mut residual = vec![0.0; self.dimension()];
        self.apply_primal(&lifting, &mut residual)?;
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
        let mut homogeneous = input.to_vec();
        for constraint in self.data.constraints.constraints() {
            homogeneous[constraint.target.0] = 0.0;
        }
        output.fill(0.0);
        self.apply_cells(&homogeneous, output, Action::Jvp)?;
        for constraint in self.data.constraints.constraints() {
            output[constraint.target.0] = input[constraint.target.0];
        }
        validate_finite("matrix-free output", output)
    }

    fn apply_primal(&self, state: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.validate_action(state, output)?;
        output.fill(0.0);
        self.apply_cells(state, output, Action::Primal)?;
        validate_finite("primal residual", output)
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

    fn apply_cells(
        &self,
        state: &[f64],
        output: &mut [f64],
        action: Action,
    ) -> Result<(), FinitumError> {
        for (cell_index, restriction) in self.data.dofs.restrictions().iter().enumerate() {
            let geometry = &self.data.geometries[cell_index];
            let local_state = restriction
                .dofs
                .iter()
                .map(|dof| state[dof.0])
                .collect::<Vec<_>>();
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
                                &local_state,
                            )?,
                            Action::Jvp => self.execute_jvp(
                                bound,
                                integral,
                                cell_index,
                                point_index,
                                geometry,
                                &local_state,
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
        }
        Ok(())
    }

    fn execute_primal(
        &self,
        bound: &BoundBundle,
        integral: &IntegralOperatorFactorization,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        local_state: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let inputs = self.point_inputs(integral, cell, point, geometry, local_state)?;
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

    fn execute_jvp(
        &self,
        bound: &BoundBundle,
        integral: &IntegralOperatorFactorization,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        local_direction: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let directions = self.point_inputs(integral, cell, point, geometry, local_direction)?;
        let input_by_operand = bound
            .bundle
            .primal_inputs
            .iter()
            .map(|binding| (binding.operand, binding.input))
            .collect::<BTreeMap<_, _>>();
        let mut values = BTreeMap::new();
        for binding in &bound.bundle.primal_inputs {
            let input = integral
                .primal
                .inputs
                .iter()
                .find(|input| input.id == binding.input)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "bundle input {:?} is absent from integral {}",
                        binding.input, integral.integral_index
                    ))
                })?;
            let primal = if input.role == TensorInputRole::Active {
                vec![0.0; component_count(&input.shape)?]
            } else {
                directions[&binding.input].clone()
            };
            values.insert(binding.operand, primal);
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
        operand_values(
            executable,
            &buffers,
            bound.bundle.jvp.dependent_operands[0].derivative,
        )
    }

    fn point_inputs(
        &self,
        integral: &IntegralOperatorFactorization,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        local_state: &[f64],
    ) -> Result<BTreeMap<TensorInputId, Vec<f64>>, FinitumError> {
        integral
            .primal
            .inputs
            .iter()
            .map(|input| {
                let values = if input.source == InputSourceRequirement::Basis {
                    evaluate_basis_input(&self.data.element, geometry, point, input, local_state)?
                } else {
                    self.data.external[&(integral.integral_index, input.id)]
                        .point_values(cell, point, self.data.element.quadrature().len())
                        .to_vec()
                };
                Ok((input.id, values))
            })
            .collect()
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
#[derive(Clone, Debug)]
pub struct AssembledOperator {
    matrix: CsrMatrix,
    source_factorization_digest: Digest,
}

impl AssembledOperator {
    pub fn matrix(&self) -> &CsrMatrix {
        &self.matrix
    }

    pub fn source_factorization_digest(&self) -> &Digest {
        &self.source_factorization_digest
    }
}

impl LinearOperator for AssembledOperator {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }

    fn columns(&self) -> usize {
        self.matrix.columns()
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
    if constraints.has_affine_dependencies() {
        return Err(FinitumError::UnsupportedRealization(
            "FC6 essential realization supports fixed values; affine dependency elimination is deferred"
                .into(),
        ));
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
        if requirement.topological_dimension as usize != mesh.dimension()
            || requirement.family != ElementFamilyRequirement::H1
            || requirement.polynomial_order != 1
            || requirement.value_shape != ValueShape::Scalar
        {
            return Err(FinitumError::UnsupportedRealization(format!(
                "FC6 supports scalar H1(order=1) cell elements, got {requirement:?}"
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
    for (index, restriction) in dofs.restrictions().iter().enumerate() {
        if restriction.dofs.len() != element.basis_count() {
            return Err(FinitumError::InvalidRealization(format!(
                "restriction {index} has {} DOFs, expected {}",
                restriction.dofs.len(),
                element.basis_count()
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
    _dimension: usize,
) -> Result<(), FinitumError> {
    if matches!(
        derivative,
        DerivativeEvaluation::Value | DerivativeEvaluation::Gradient
    ) {
        Ok(())
    } else {
        Err(FinitumError::UnsupportedRealization(format!(
            "FC6 supports value and gradient basis actions, got {derivative:?}"
        )))
    }
}

fn validate_external_inputs(
    factorization: &OperatorFactorization,
    mesh: &Mesh,
    element: &PreparedElement,
    external_inputs: Vec<ExternalInput>,
) -> Result<BTreeMap<(usize, TensorInputId), ExternalInput>, FinitumError> {
    let mut external = BTreeMap::new();
    for input in external_inputs {
        let key = (input.integral_index, input.input);
        if external.insert(key, input).is_some() {
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
    }
    if external.keys().any(|key| !expected_keys.contains(key)) {
        return Err(FinitumError::InvalidRealization(
            "an external input does not belong to the factorization".into(),
        ));
    }
    Ok(external)
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

fn evaluate_basis_input(
    element: &PreparedElement,
    geometry: &CellGeometry,
    point: usize,
    input: &QFunctionInput,
    local_state: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    match input.binding.evaluation.derivative {
        DerivativeEvaluation::Value => {
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
        }
        DerivativeEvaluation::Gradient => {
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
        }
        _ => Err(FinitumError::UnsupportedRealization(format!(
            "unsupported basis evaluation {:?}",
            input.binding.evaluation.derivative
        ))),
    }
}

fn apply_basis_adjoint(
    element: &PreparedElement,
    geometry: &CellGeometry,
    point: usize,
    derivative: &DerivativeEvaluation,
    point_output: &[f64],
    scale: f64,
    local_output: &mut [f64],
) -> Result<(), FinitumError> {
    match derivative {
        DerivativeEvaluation::Value if point_output.len() == 1 => {
            for (basis, output) in local_output.iter_mut().enumerate() {
                *output += scale * element.basis_value(point, basis).unwrap() * point_output[0];
            }
        }
        DerivativeEvaluation::Gradient if point_output.len() == element.dimension() => {
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
struct CellGeometry {
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
