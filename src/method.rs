//! Concrete realizations for Resolvent sibling method-family programs.

use crate::{FinitumError, RealizationPlan};
use malleus::{
    BufferBinding, Executable, ExecutableModule, Interpreter, OperandId, validate_module,
};
use scientia::{MethodFamily, MethodProgram, MethodProgramKind};
use serde::Serialize;
use solverang::{DaeOperator, EvaluationContext, NumericError};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct FiniteVolumeFace {
    pub minus: usize,
    pub plus: usize,
}

#[derive(Clone, Debug)]
struct AffineExecutable {
    executable: Executable,
    input_count: usize,
    constant: f64,
}

impl AffineExecutable {
    fn from_program(program: &MethodProgram) -> Result<Self, FinitumError> {
        let kernel = program.local_kernel.as_ref().ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "{} program has no local Malleus kernel",
                program.family().as_str()
            ))
        })?;
        if program.receipt.local_kernel_digest.as_ref() != Some(&kernel.artifact_digest) {
            return Err(FinitumError::ArtifactMismatch(
                "method receipt does not identify its local kernel".into(),
            ));
        }
        let module = validate_module(kernel.module.clone())
            .map_err(|error| FinitumError::KernelValidation(error.to_string()))?;
        let executable = ExecutableModule::reference(module)
            .kernels()
            .first()
            .cloned()
            .ok_or_else(|| FinitumError::KernelValidation("method module is empty".into()))?;
        Ok(Self {
            executable,
            input_count: kernel.spec.inputs.len(),
            constant: kernel.spec.constant,
        })
    }

    fn run(&self, inputs: &[f64]) -> Result<f64, FinitumError> {
        if inputs.len() != self.input_count {
            return Err(FinitumError::InvalidRealization(format!(
                "local method kernel received {} inputs, expected {}",
                inputs.len(),
                self.input_count
            )));
        }
        let mut owned = inputs.to_vec();
        let mut output = [0.0];
        let mut bindings = owned
            .iter_mut()
            .enumerate()
            .map(|(index, value)| {
                BufferBinding::new(OperandId::new(index), std::slice::from_mut(value))
            })
            .collect::<Vec<_>>();
        bindings.push(BufferBinding::new(
            OperandId::new(self.input_count),
            &mut output,
        ));
        Interpreter::run(&self.executable, &mut bindings)
            .map_err(|error| FinitumError::KernelExecution(error.to_string()))?;
        Ok(output[0])
    }

    fn run_direction(&self, inputs: &[f64]) -> Result<f64, FinitumError> {
        Ok(self.run(inputs)? - self.constant)
    }
}

#[derive(Clone, Debug)]
pub struct FiniteVolumeRealization {
    program: Arc<MethodProgram>,
    cell_volumes: Vec<f64>,
    faces: Vec<FiniteVolumeFace>,
    flux: AffineExecutable,
    identity: String,
}

impl FiniteVolumeRealization {
    pub fn new(
        program: MethodProgram,
        cell_volumes: Vec<f64>,
        faces: Vec<FiniteVolumeFace>,
    ) -> Result<Self, FinitumError> {
        if !matches!(
            program.kind,
            MethodProgramKind::ConservationLawFiniteVolume(_)
        ) {
            return Err(FinitumError::ArtifactMismatch(
                "finite-volume realization requires a conservation-law program".into(),
            ));
        }
        validate_positive(&cell_volumes, "finite-volume cell volumes")?;
        if cell_volumes.is_empty() || faces.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "finite-volume realization requires cells and oriented faces".into(),
            ));
        }
        if faces
            .iter()
            .any(|face| face.minus >= cell_volumes.len() || face.plus >= cell_volumes.len())
        {
            return Err(FinitumError::InvalidRealization(
                "finite-volume face references a missing cell".into(),
            ));
        }
        let flux = AffineExecutable::from_program(&program)?;
        if flux.input_count != 2 {
            return Err(FinitumError::ArtifactMismatch(
                "reference finite-volume numerical flux must take minus and plus states".into(),
            ));
        }
        let identity = concrete_identity(
            "finitum-finite-volume/1",
            &program,
            &(&cell_volumes, &faces),
        );
        Ok(Self {
            program: Arc::new(program),
            cell_volumes,
            faces,
            flux,
            identity,
        })
    }

    pub fn program(&self) -> &MethodProgram {
        &self.program
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn dimension(&self) -> usize {
        self.cell_volumes.len()
    }

    fn action(&self, state: &[f64], rate: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state, rate, output)?;
        for (index, ((out, volume), state_rate)) in output
            .iter_mut()
            .zip(&self.cell_volumes)
            .zip(rate)
            .enumerate()
        {
            if !state[index].is_finite() || !state_rate.is_finite() {
                return Err(FinitumError::InvalidRealization(
                    "finite-volume state and rate must be finite".into(),
                ));
            }
            *out = volume * state_rate;
        }
        for face in &self.faces {
            let flux = self.flux.run(&[state[face.minus], state[face.plus]])?;
            output[face.minus] += flux;
            output[face.plus] -= flux;
        }
        Ok(())
    }

    fn derivative(
        &self,
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state_direction, rate_direction, output)?;
        for (out, (volume, rate)) in output
            .iter_mut()
            .zip(self.cell_volumes.iter().zip(rate_direction))
        {
            *out = volume * rate;
        }
        for face in &self.faces {
            let flux = self
                .flux
                .run_direction(&[state_direction[face.minus], state_direction[face.plus]])?;
            output[face.minus] += flux;
            output[face.plus] -= flux;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FiniteDifferenceRealization {
    program: Arc<MethodProgram>,
    row_inputs: Vec<Vec<usize>>,
    mass: Vec<f64>,
    stencil: AffineExecutable,
    identity: String,
}

impl FiniteDifferenceRealization {
    pub fn new(
        program: MethodProgram,
        row_inputs: Vec<Vec<usize>>,
        mass: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        let method = match &program.kind {
            MethodProgramKind::StructuredStencilFiniteDifference(method) => method,
            _ => {
                return Err(FinitumError::ArtifactMismatch(
                    "finite-difference realization requires a stencil program".into(),
                ));
            }
        };
        validate_positive(&mass, "finite-difference mass weights")?;
        if row_inputs.len() != mass.len() || row_inputs.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "finite-difference rows and mass weights must have one common nonzero extent"
                    .into(),
            ));
        }
        let stencil = AffineExecutable::from_program(&program)?;
        if stencil.input_count != method.offsets.len()
            || row_inputs.iter().any(|row| {
                row.len() != stencil.input_count || row.iter().any(|index| *index >= mass.len())
            })
        {
            return Err(FinitumError::InvalidRealization(
                "finite-difference neighbor rows do not match the compiled stencil".into(),
            ));
        }
        let identity = concrete_identity(
            "finitum-finite-difference/1",
            &program,
            &(&row_inputs, &mass),
        );
        Ok(Self {
            program: Arc::new(program),
            row_inputs,
            mass,
            stencil,
            identity,
        })
    }

    pub fn program(&self) -> &MethodProgram {
        &self.program
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn dimension(&self) -> usize {
        self.mass.len()
    }

    fn action(&self, state: &[f64], rate: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state, rate, output)?;
        for row in 0..self.dimension() {
            let inputs = self.row_inputs[row]
                .iter()
                .map(|index| state[*index])
                .collect::<Vec<_>>();
            output[row] = self.mass[row] * rate[row] + self.stencil.run(&inputs)?;
        }
        Ok(())
    }

    fn derivative(
        &self,
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state_direction, rate_direction, output)?;
        for row in 0..self.dimension() {
            let inputs = self.row_inputs[row]
                .iter()
                .map(|index| state_direction[*index])
                .collect::<Vec<_>>();
            output[row] =
                self.mass[row] * rate_direction[row] + self.stencil.run_direction(&inputs)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NetworkDaeRealization {
    program: Arc<MethodProgram>,
    mass: Vec<Vec<f64>>,
    stiffness: Vec<Vec<f64>>,
    source: Vec<f64>,
    identity: String,
}

impl NetworkDaeRealization {
    pub fn new(
        program: MethodProgram,
        mass: Vec<Vec<f64>>,
        stiffness: Vec<Vec<f64>>,
        source: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        let method = match &program.kind {
            MethodProgramKind::NetworkDae(method) => method,
            _ => {
                return Err(FinitumError::ArtifactMismatch(
                    "network realization requires a network DAE program".into(),
                ));
            }
        };
        let dimension = source.len();
        let semantic_dimension = method.state_components.iter().sum::<usize>();
        if dimension != semantic_dimension {
            return Err(FinitumError::ArtifactMismatch(format!(
                "network realization dimension {dimension} differs from the typed state extent {semantic_dimension}"
            )));
        }
        validate_matrix(&mass, dimension, "network mass matrix")?;
        validate_matrix(&stiffness, dimension, "network stiffness matrix")?;
        validate_finite(&source, "network source")?;
        if dimension == 0 {
            return Err(FinitumError::InvalidRealization(
                "network realization must have nonzero dimension".into(),
            ));
        }
        let identity = concrete_identity(
            "finitum-network-dae/1",
            &program,
            &(&mass, &stiffness, &source),
        );
        Ok(Self {
            program: Arc::new(program),
            mass,
            stiffness,
            source,
            identity,
        })
    }

    pub fn program(&self) -> &MethodProgram {
        &self.program
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn dimension(&self) -> usize {
        self.source.len()
    }

    fn action(&self, state: &[f64], rate: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state, rate, output)?;
        for (row, result) in output.iter_mut().enumerate() {
            *result =
                dot(&self.mass[row], rate) + dot(&self.stiffness[row], state) - self.source[row];
        }
        Ok(())
    }

    fn derivative(
        &self,
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state_direction, rate_direction, output)?;
        for (row, result) in output.iter_mut().enumerate() {
            *result =
                dot(&self.mass[row], rate_direction) + dot(&self.stiffness[row], state_direction);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ParticlePair {
    pub first: usize,
    pub second: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RadialPairPolynomial {
    /// Coefficients of `V(s) = sum c[k] s^k`, where `s` is squared pair distance.
    pub coefficients: Vec<f64>,
}

impl RadialPairPolynomial {
    fn derivatives(&self, squared_distance: f64) -> (f64, f64) {
        let mut first = 0.0;
        let mut second = 0.0;
        for (power, coefficient) in self.coefficients.iter().copied().enumerate().skip(1) {
            first += power as f64 * coefficient * squared_distance.powi(power as i32 - 1);
            if power >= 2 {
                second += (power * (power - 1)) as f64
                    * coefficient
                    * squared_distance.powi(power as i32 - 2);
            }
        }
        (first, second)
    }

    pub fn potential(&self, squared_distance: f64) -> f64 {
        self.coefficients
            .iter()
            .copied()
            .enumerate()
            .map(|(power, coefficient)| coefficient * squared_distance.powi(power as i32))
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct ParticleRealization {
    program: Arc<MethodProgram>,
    coordinate_dimension: usize,
    masses: Vec<f64>,
    pairs: Vec<ParticlePair>,
    potential: RadialPairPolynomial,
    identity: String,
}

impl ParticleRealization {
    pub fn new(
        program: MethodProgram,
        coordinate_dimension: usize,
        masses: Vec<f64>,
        pairs: Vec<ParticlePair>,
        potential: RadialPairPolynomial,
    ) -> Result<Self, FinitumError> {
        let method = match &program.kind {
            MethodProgramKind::Particle(method) => method,
            _ => {
                return Err(FinitumError::ArtifactMismatch(
                    "particle realization requires a particle method program".into(),
                ));
            }
        };
        if !(1..=3).contains(&coordinate_dimension) {
            return Err(FinitumError::InvalidRealization(
                "particle coordinate dimension must be in 1..=3".into(),
            ));
        }
        validate_positive(&masses, "particle masses")?;
        let concrete_components = masses.len() * coordinate_dimension;
        if concrete_components != method.position_components
            || concrete_components != method.velocity_components
        {
            return Err(FinitumError::ArtifactMismatch(format!(
                "particle realization has {concrete_components} components per state, typed position/velocity extents are {}/{}",
                method.position_components, method.velocity_components
            )));
        }
        validate_finite(&potential.coefficients, "radial pair coefficients")?;
        if masses.is_empty()
            || potential.coefficients.is_empty()
            || pairs.is_empty()
            || pairs.iter().any(|pair| {
                pair.first >= masses.len()
                    || pair.second >= masses.len()
                    || pair.first == pair.second
            })
        {
            return Err(FinitumError::InvalidRealization(
                "particle masses, distinct bounded pairs, and a radial potential are required"
                    .into(),
            ));
        }
        let identity = concrete_identity(
            "finitum-particle/1",
            &program,
            &(coordinate_dimension, &masses, &pairs, &potential),
        );
        Ok(Self {
            program: Arc::new(program),
            coordinate_dimension,
            masses,
            pairs,
            potential,
            identity,
        })
    }

    pub fn program(&self) -> &MethodProgram {
        &self.program
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn particle_count(&self) -> usize {
        self.masses.len()
    }

    pub fn dimension(&self) -> usize {
        2 * self.particle_count() * self.coordinate_dimension
    }

    pub fn potential_energy(&self, state: &[f64]) -> Result<f64, FinitumError> {
        require_one_length(self.dimension(), state, "particle state")?;
        let positions = &state[..self.position_extent()];
        Ok(self
            .pairs
            .iter()
            .map(|pair| {
                self.potential
                    .potential(self.squared_distance(positions, *pair))
            })
            .sum())
    }

    fn position_extent(&self) -> usize {
        self.particle_count() * self.coordinate_dimension
    }

    fn squared_distance(&self, positions: &[f64], pair: ParticlePair) -> f64 {
        (0..self.coordinate_dimension)
            .map(|axis| {
                let difference = positions[pair.first * self.coordinate_dimension + axis]
                    - positions[pair.second * self.coordinate_dimension + axis];
                difference * difference
            })
            .sum()
    }

    fn forces(&self, positions: &[f64]) -> Vec<f64> {
        let mut forces = vec![0.0; self.position_extent()];
        for pair in &self.pairs {
            let squared_distance = self.squared_distance(positions, *pair);
            let (first, _) = self.potential.derivatives(squared_distance);
            for axis in 0..self.coordinate_dimension {
                let first_index = pair.first * self.coordinate_dimension + axis;
                let second_index = pair.second * self.coordinate_dimension + axis;
                let contribution =
                    -2.0 * first * (positions[first_index] - positions[second_index]);
                forces[first_index] += contribution;
                forces[second_index] -= contribution;
            }
        }
        forces
    }

    fn force_direction(&self, positions: &[f64], direction: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.position_extent()];
        for pair in &self.pairs {
            let squared_distance = self.squared_distance(positions, *pair);
            let (first, second) = self.potential.derivatives(squared_distance);
            let radial_direction = (0..self.coordinate_dimension)
                .map(|axis| {
                    let a = pair.first * self.coordinate_dimension + axis;
                    let b = pair.second * self.coordinate_dimension + axis;
                    (positions[a] - positions[b]) * (direction[a] - direction[b])
                })
                .sum::<f64>();
            for axis in 0..self.coordinate_dimension {
                let a = pair.first * self.coordinate_dimension + axis;
                let b = pair.second * self.coordinate_dimension + axis;
                let delta = positions[a] - positions[b];
                let delta_direction = direction[a] - direction[b];
                let contribution =
                    -2.0 * (2.0 * second * radial_direction * delta + first * delta_direction);
                result[a] += contribution;
                result[b] -= contribution;
            }
        }
        result
    }

    fn action(&self, state: &[f64], rate: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state, rate, output)?;
        let extent = self.position_extent();
        let forces = self.forces(&state[..extent]);
        for index in 0..extent {
            output[index] = rate[index] - state[extent + index];
            let particle = index / self.coordinate_dimension;
            output[extent + index] = self.masses[particle] * rate[extent + index] - forces[index];
        }
        Ok(())
    }

    fn derivative(
        &self,
        state: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        require_lengths(self.dimension(), state_direction, rate_direction, output)?;
        require_one_length(self.dimension(), state, "particle state")?;
        let extent = self.position_extent();
        let force_direction = self.force_direction(&state[..extent], &state_direction[..extent]);
        for index in 0..extent {
            output[index] = rate_direction[index] - state_direction[extent + index];
            let particle = index / self.coordinate_dimension;
            output[extent + index] =
                self.masses[particle] * rate_direction[extent + index] - force_direction[index];
        }
        Ok(())
    }
}

/// Algebraic boundary-integral row in the common DAE interface.
///
/// Its residual is `A * state - right_hand_side`, its JVP is `A * state_direction`, and its DAE
/// mass contribution is zero. Consequently, state rates and rate directions are intentionally
/// ignored.
#[derive(Clone, Debug)]
pub struct BoundaryIntegralRealization {
    program: Arc<MethodProgram>,
    weights: Vec<f64>,
    kernel_values: Vec<Vec<f64>>,
    diagonal: Vec<f64>,
    right_hand_side: Vec<f64>,
    identity: String,
}

impl BoundaryIntegralRealization {
    pub fn new(
        program: MethodProgram,
        weights: Vec<f64>,
        kernel_values: Vec<Vec<f64>>,
        diagonal: Vec<f64>,
        right_hand_side: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if !matches!(program.kind, MethodProgramKind::BoundaryIntegral(_)) {
            return Err(FinitumError::ArtifactMismatch(
                "boundary-integral realization requires a boundary-integral program".into(),
            ));
        }
        let dimension = weights.len();
        validate_positive(&weights, "boundary quadrature weights")?;
        validate_matrix(&kernel_values, dimension, "boundary kernel table")?;
        validate_finite(&diagonal, "boundary diagonal")?;
        validate_finite(&right_hand_side, "boundary right-hand side")?;
        if dimension == 0 || diagonal.len() != dimension || right_hand_side.len() != dimension {
            return Err(FinitumError::InvalidRealization(
                "boundary-integral tables must have one common nonzero extent".into(),
            ));
        }
        let identity = concrete_identity(
            "finitum-boundary-integral/1",
            &program,
            &(&weights, &kernel_values, &diagonal, &right_hand_side),
        );
        Ok(Self {
            program: Arc::new(program),
            weights,
            kernel_values,
            diagonal,
            right_hand_side,
            identity,
        })
    }

    pub fn program(&self) -> &MethodProgram {
        &self.program
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn dimension(&self) -> usize {
        self.weights.len()
    }

    fn matrix_action(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        require_one_length(self.dimension(), input, "boundary density")?;
        require_one_length(self.dimension(), output, "boundary action")?;
        for (row, out) in output.iter_mut().enumerate() {
            *out = self.diagonal[row] * input[row]
                + self.kernel_values[row]
                    .iter()
                    .zip(&self.weights)
                    .zip(input)
                    .map(|((kernel, weight), value)| kernel * weight * value)
                    .sum::<f64>();
        }
        Ok(())
    }

    fn action(&self, state: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.matrix_action(state, output)?;
        for (value, right_hand_side) in output.iter_mut().zip(&self.right_hand_side) {
            *value -= right_hand_side;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum MethodRealization {
    FiniteVolume(FiniteVolumeRealization),
    FiniteDifference(FiniteDifferenceRealization),
    NetworkDae(NetworkDaeRealization),
    Particle(ParticleRealization),
    BoundaryIntegral(BoundaryIntegralRealization),
}

impl MethodRealization {
    pub const fn family(&self) -> MethodFamily {
        match self {
            Self::FiniteVolume(_) => MethodFamily::ConservationLawFiniteVolume,
            Self::FiniteDifference(_) => MethodFamily::StructuredStencilFiniteDifference,
            Self::NetworkDae(_) => MethodFamily::NetworkDae,
            Self::Particle(_) => MethodFamily::Particle,
            Self::BoundaryIntegral(_) => MethodFamily::BoundaryIntegral,
        }
    }

    pub fn identity(&self) -> &str {
        match self {
            Self::FiniteVolume(value) => value.identity(),
            Self::FiniteDifference(value) => value.identity(),
            Self::NetworkDae(value) => value.identity(),
            Self::Particle(value) => value.identity(),
            Self::BoundaryIntegral(value) => value.identity(),
        }
    }
}

impl DaeOperator for MethodRealization {
    fn dimension(&self) -> usize {
        match self {
            Self::FiniteVolume(value) => value.dimension(),
            Self::FiniteDifference(value) => value.dimension(),
            Self::NetworkDae(value) => value.dimension(),
            Self::Particle(value) => value.dimension(),
            Self::BoundaryIntegral(value) => value.dimension(),
        }
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        match self {
            Self::FiniteVolume(value) => value.action(state, state_rate, output),
            Self::FiniteDifference(value) => value.action(state, state_rate, output),
            Self::NetworkDae(value) => value.action(state, state_rate, output),
            Self::Particle(value) => value.action(state, state_rate, output),
            Self::BoundaryIntegral(value) => value.action(state, output),
        }
        .map_err(numeric_error)
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
        _state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        match self {
            Self::FiniteVolume(value) => value.derivative(state_direction, rate_direction, output),
            Self::FiniteDifference(value) => {
                value.derivative(state_direction, rate_direction, output)
            }
            Self::NetworkDae(value) => value.derivative(state_direction, rate_direction, output),
            Self::Particle(value) => {
                value.derivative(state, state_direction, rate_direction, output)
            }
            Self::BoundaryIntegral(value) => value.matrix_action(state_direction, output),
        }
        .map_err(numeric_error)
    }
}

/// A single Finitum-owned operator boundary for variational FEM and sibling methods.
#[derive(Clone, Debug)]
pub enum DiscreteOperator {
    VariationalFem(RealizationPlan),
    Sibling(Box<MethodRealization>),
}

impl DiscreteOperator {
    pub fn sibling(realization: MethodRealization) -> Self {
        Self::Sibling(Box::new(realization))
    }

    pub fn family_identity(&self) -> &str {
        match self {
            Self::VariationalFem(_) => "variational_fem",
            Self::Sibling(realization) => realization.family().as_str(),
        }
    }

    pub fn identity(&self) -> String {
        match self {
            Self::VariationalFem(plan) => format!("finitum-variational:{}", plan.digest()),
            Self::Sibling(realization) => realization.identity().to_owned(),
        }
    }
}

impl DaeOperator for DiscreteOperator {
    fn dimension(&self) -> usize {
        match self {
            Self::VariationalFem(plan) => plan.dimension(),
            Self::Sibling(realization) => realization.dimension(),
        }
    }

    fn residual(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        match self {
            Self::VariationalFem(plan) => plan
                .residual(time, state, state_rate, output)
                .map_err(numeric_error),
            Self::Sibling(realization) => {
                realization.residual(context, time, state, state_rate, output)
            }
        }
    }

    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        match self {
            Self::VariationalFem(plan) => plan
                .jacobian_vector_product(
                    time,
                    state,
                    state_rate,
                    state_direction,
                    rate_direction,
                    output,
                )
                .map_err(numeric_error),
            Self::Sibling(realization) => realization.jacobian_vector_product(
                context,
                time,
                state,
                state_rate,
                state_direction,
                rate_direction,
                output,
            ),
        }
    }
}

fn numeric_error(error: FinitumError) -> NumericError {
    NumericError::Operator {
        message: error.to_string(),
    }
}

fn require_lengths(
    expected: usize,
    state: &[f64],
    rate: &[f64],
    output: &[f64],
) -> Result<(), FinitumError> {
    require_one_length(expected, state, "method state")?;
    require_one_length(expected, rate, "method state rate")?;
    require_one_length(expected, output, "method residual")
}

fn require_one_length(expected: usize, values: &[f64], label: &str) -> Result<(), FinitumError> {
    if values.len() != expected {
        Err(FinitumError::InvalidRealization(format!(
            "{label} has length {}, expected {expected}",
            values.len()
        )))
    } else if values.iter().any(|value| !value.is_finite()) {
        Err(FinitumError::InvalidRealization(format!(
            "{label} must contain only finite values"
        )))
    } else {
        Ok(())
    }
}

fn validate_positive(values: &[f64], label: &str) -> Result<(), FinitumError> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        Err(FinitumError::InvalidRealization(format!(
            "{label} must be finite and positive"
        )))
    } else {
        Ok(())
    }
}

fn validate_finite(values: &[f64], label: &str) -> Result<(), FinitumError> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(FinitumError::InvalidRealization(format!(
            "{label} must contain only finite values"
        )))
    } else {
        Ok(())
    }
}

fn validate_matrix(matrix: &[Vec<f64>], dimension: usize, label: &str) -> Result<(), FinitumError> {
    if matrix.len() != dimension
        || matrix
            .iter()
            .any(|row| row.len() != dimension || row.iter().any(|value| !value.is_finite()))
    {
        Err(FinitumError::InvalidRealization(format!(
            "{label} must be a finite {dimension} by {dimension} matrix"
        )))
    } else {
        Ok(())
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn concrete_identity(
    config_schema: &str,
    program: &MethodProgram,
    config: &impl Serialize,
) -> String {
    #[derive(Serialize)]
    struct Payload<'a, T> {
        schema: &'a str,
        program: &'a scientia::Digest,
        config: &'a T,
    }
    let bytes = serde_json::to_vec(&Payload {
        schema: config_schema,
        program: &program.artifact_digest,
        config,
    })
    .expect("method realization identity is serializable");
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}
