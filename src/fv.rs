//! SV2-C2a first-order finite-volume realization (`finitum-fv-realization/1`, W12 lane L4).
//!
//! [`FvRealization`] binds the Scientia [`ConservationLawProgram`] kernels to an
//! [`FvTopology`] by [`FvOperandMeaning`]: minus = owner cell, plus = neighbour cell or the
//! region's `Ghost` state; a `PrescribedBoundaryFlux` kernel replaces the face flux on its
//! region. Each face block runs through one `PreparedKernel::run_batch` (state, normal and
//! output columns at stride 1, uniform data at stride 0) over reusable scratch, so a
//! steady-state evaluation allocates nothing. The residual is
//! `R_i = |Ω_i|·u̇_i + Σ_f s_if |f| F*_f − |Ω_i| S_i`, scattered in the fixed face order of
//! [`FvTopology`] (interior, periodic, boundary tags): the owner gains `|f|F*` and the
//! neighbour loses the same stored value, so interior fluxes telescope by construction.
//!
//! Reconstruction is constant (first order); MUSCL/limiters are SV2-C2b. Derivatives use the
//! Scientia [`FvKernel::differentiate`] refusal (decision D6): a kernel whose JVP crosses a
//! kink yields a typed `FV_NONSMOOTH_DERIVATIVE_REFUSED` evaluation failure from the JVP, never
//! a silently differentiated value; the primal operator is unaffected.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::Mutex;

use malleus::prepared::{ReadBinding, WriteBinding};
use malleus::{DerivativeMode, DerivativeRequest, Executable, FacetSide, OperandId, validate};
use methodus::{DaeOperator, EvaluationContext, ExplicitOperator, NumericError};
use scientia::{
    BoundaryFluxSemantics, ConservationLawProgram, ConservationLawSystem, Digest, FvDatumSource,
    FvKernel, FvKernelRole, FvOperandMeaning, PositivityIntent, ReconstructionIntent,
};
use serde::Serialize;

use crate::execution::{PreparedExecutable, run_batch};
use crate::{
    FinitumError, FvFaceBlock, FvTopology, InputEvaluationError, InputOrigin, RegionMap,
    RegionTagId,
};

pub const FV_REALIZATION_SCHEMA: &str = "finitum-fv-realization/1";
pub const FV_CONSERVATION_RECEIPT_SCHEMA: &str = "finitum-fv-conservation/1";

/// Operand count up to which a face call binds from the stack.
const FV_INLINE_OPERANDS: usize = 24;

fn fv_error(code: &str, message: impl std::fmt::Display) -> FinitumError {
    FinitumError::InvalidRealization(format!("{code}: {message}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Minus,
    Plus,
}

/// Where one kernel operand reads from (or that it is the single written output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    State(Side),
    Tangent(Side),
    Normal(usize),
    Datum(usize),
    Write,
}

/// Read columns of one batched call. Unused columns are empty.
struct Io<'a> {
    minus: &'a [f64],
    plus: &'a [f64],
    minus_tangent: &'a [f64],
    plus_tangent: &'a [f64],
    normal: &'a [Vec<f64>],
    data: &'a [f64],
}

/// A prepared kernel with every operand bound by meaning.
#[derive(Clone, Debug)]
struct BoundKernel {
    executable: PreparedExecutable,
    columns: Vec<Column>,
}

impl BoundKernel {
    fn new(kernel: malleus::StructuredKernel, columns: Vec<Column>) -> Result<Self, FinitumError> {
        if columns.len() > FV_INLINE_OPERANDS
            || columns
                .iter()
                .filter(|column| **column == Column::Write)
                .count()
                != 1
        {
            return Err(fv_error(
                "FV_KERNEL_SHAPE",
                format!(
                    "kernel `{}` needs one output and at most {FV_INLINE_OPERANDS} operands",
                    kernel.name
                ),
            ));
        }
        let validated = validate(kernel)
            .map_err(|error| FinitumError::KernelValidation(format!("{error:?}")))?;
        Ok(Self {
            executable: PreparedExecutable::new(Executable::reference(validated))?,
            columns,
        })
    }

    /// The primal kernel; `allowed` lists the column kinds its role may read.
    fn primal(
        kernel: &FvKernel,
        plus_allowed: bool,
        normal_allowed: bool,
    ) -> Result<Self, FinitumError> {
        let columns = kernel
            .operands
            .iter()
            .map(|meaning| match meaning {
                FvOperandMeaning::State {
                    side: None | Some(FacetSide::Minus),
                } => Ok(Column::State(Side::Minus)),
                FvOperandMeaning::State {
                    side: Some(FacetSide::Plus),
                } if plus_allowed => Ok(Column::State(Side::Plus)),
                FvOperandMeaning::Normal { axis } if normal_allowed => {
                    Ok(Column::Normal(usize::from(*axis)))
                }
                FvOperandMeaning::Datum { datum } => Ok(Column::Datum(*datum)),
                FvOperandMeaning::Output => Ok(Column::Write),
                other => Err(fv_error(
                    "FV_KERNEL_SHAPE",
                    format!(
                        "{:?} kernel operand {other:?} is not bound by a first-order realization",
                        kernel.role
                    ),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(kernel.kernel.clone(), columns)
    }

    fn run(&self, points: usize, io: &Io<'_>, output: &mut [f64]) -> Result<(), FinitumError> {
        if points == 0 {
            return Ok(());
        }
        let mut reads = [ReadBinding::new(OperandId::new(0), &[]); FV_INLINE_OPERANDS];
        let mut count = 0;
        let mut written = OperandId::new(0);
        for (index, column) in self.columns.iter().enumerate() {
            let operand = OperandId::new(index);
            let (values, stride): (&[f64], usize) = match *column {
                Column::State(Side::Minus) => (&io.minus[..points], 1),
                Column::State(Side::Plus) => (&io.plus[..points], 1),
                Column::Tangent(Side::Minus) => (&io.minus_tangent[..points], 1),
                Column::Tangent(Side::Plus) => (&io.plus_tangent[..points], 1),
                Column::Normal(axis) => (&io.normal[axis][..points], 1),
                Column::Datum(datum) => (&io.data[datum..datum + 1], 0),
                Column::Write => {
                    written = operand;
                    continue;
                }
            };
            reads[count] = ReadBinding::strided(operand, values, stride);
            count += 1;
        }
        let mut writes = [WriteBinding::strided(written, &mut output[..points], 1)];
        run_batch(
            self.executable.prepared(),
            points,
            &reads[..count],
            &mut writes,
        )
        .map_err(|error| FinitumError::KernelExecution(error.to_string()))
    }
}

/// The JVP of one kernel in its state operands.
#[derive(Clone, Debug)]
enum Tangent {
    /// The kernel reads no state; its state derivative is zero.
    Zero,
    Kernel(Box<BoundKernel>),
    /// Scientia refused the derivative (D6 kink crossing, or a failed differentiation).
    Refused {
        code: String,
        message: String,
    },
}

impl Tangent {
    fn of(kernel: &FvKernel, primal: &BoundKernel) -> Result<Self, FinitumError> {
        let states = primal
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| matches!(column, Column::State(_)))
            .map(|(index, _)| OperandId::new(index))
            .collect::<Vec<_>>();
        if states.is_empty() {
            return Ok(Self::Zero);
        }
        let output = kernel
            .operand(&FvOperandMeaning::Output)
            .ok_or_else(|| fv_error("FV_KERNEL_SHAPE", "kernel has no output"))?;
        let request = DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: states,
            dependent_operands: vec![output],
        };
        let product = match kernel.differentiate(&request) {
            Ok(product) => product,
            Err(error) => {
                return Ok(Self::Refused {
                    code: error.code.into(),
                    message: error.message,
                });
            }
        };
        let columns = (0..product.kernel.operands.len())
            .map(|index| {
                let operand = OperandId::new(index);
                if let Some(pair) = product
                    .primal_operands
                    .iter()
                    .find(|pair| pair.derivative == operand)
                {
                    return Ok(primal.columns[pair.primal.index()]);
                }
                if let Some(pair) = product
                    .independent_operands
                    .iter()
                    .find(|pair| pair.derivative == operand)
                {
                    let Column::State(side) = primal.columns[pair.primal.index()] else {
                        unreachable!("independents are state operands");
                    };
                    return Ok(Column::Tangent(side));
                }
                if product
                    .dependent_operands
                    .iter()
                    .any(|pair| pair.derivative == operand)
                {
                    return Ok(Column::Write);
                }
                Err(fv_error(
                    "FV_KERNEL_SHAPE",
                    format!("derivative operand {index} has no meaning"),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Kernel(Box::new(BoundKernel::new(
            product.kernel,
            columns,
        )?)))
    }

    /// The JVP kernel, `None` for a zero derivative, or the typed refusal located at `origin`
    /// (built only on refusal, so the hot path allocates nothing).
    fn kernel(
        &self,
        origin: impl FnOnce() -> String,
    ) -> Result<Option<&BoundKernel>, FinitumError> {
        match self {
            Self::Zero => Ok(None),
            Self::Kernel(kernel) => Ok(Some(kernel)),
            Self::Refused { code, message } => Err(FinitumError::InputEvaluation(Box::new(
                InputEvaluationError::new(
                    code.clone(),
                    InputOrigin::ExpressionPath(format!("fv/{}", origin())),
                    message.clone(),
                ),
            ))),
        }
    }

    fn disposition(&self) -> FvDerivativeDisposition {
        match self {
            Self::Zero => FvDerivativeDisposition::Zero,
            Self::Kernel(_) => FvDerivativeDisposition::Smooth,
            Self::Refused { code, .. } => FvDerivativeDisposition::Refused { code: code.clone() },
        }
    }
}

#[derive(Clone, Debug)]
enum BoundaryAction {
    Ghost,
    Prescribed,
}

#[derive(Clone, Debug)]
struct BoundaryBinding {
    tag: RegionTagId,
    /// Index into `ConservationLawSystem::boundaries` / `ConservationLawProgram::boundaries`.
    boundary: usize,
    action: BoundaryAction,
    kernel: BoundKernel,
    tangent: Tangent,
}

/// Reusable evaluation buffers, sized once at construction.
#[derive(Debug)]
struct Scratch {
    minus: Vec<f64>,
    plus: Vec<f64>,
    minus_tangent: Vec<f64>,
    plus_tangent: Vec<f64>,
    flux: Vec<f64>,
    flux_tangent: Vec<f64>,
    cell: Vec<f64>,
    data: Vec<f64>,
}

impl Scratch {
    fn new(faces: usize, cells: usize, data: &[f64]) -> Self {
        Self {
            minus: vec![0.0; faces],
            plus: vec![0.0; faces],
            minus_tangent: vec![0.0; faces],
            plus_tangent: vec![0.0; faces],
            flux: vec![0.0; faces],
            flux_tangent: vec![0.0; faces],
            cell: vec![0.0; cells],
            data: data.to_vec(),
        }
    }
}

/// How the state derivative of one kernel is realized.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FvDerivativeDisposition {
    Smooth,
    Zero,
    /// Not part of the residual (the CFL speed kernel), so never differentiated.
    NotDifferentiated,
    /// The JVP is a typed refusal carrying this Scientia code.
    Refused {
        code: String,
    },
}

/// One bound kernel in an [`FvRealizationReceipt`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvKernelReceipt {
    pub role: FvKernelRole,
    /// The topology boundary tag a boundary kernel is bound to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<RegionTagId>,
    pub kernel_digest: Digest,
    pub derivative: FvDerivativeDisposition,
}

/// What an [`FvRealization`] bound, for run evidence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvRealizationReceipt {
    pub schema: String,
    pub digest: Digest,
    pub program_digest: Digest,
    pub intent_digest: Digest,
    pub topology_digest: Digest,
    pub reconstruction: String,
    pub kernels: Vec<FvKernelReceipt>,
}

/// Numerical flux `F*` per face (per unit area) in [`FvTopology`] block order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvFaceFluxes {
    pub interior: Vec<f64>,
    pub periodic: Vec<f64>,
    /// Outward boundary flux per tag (the face flux against the ghost, or the prescribed one).
    pub boundary: BTreeMap<RegionTagId, Vec<f64>>,
}

/// The realized `DiscreteFluxBalance`: the flux residual summed over cells must equal the net
/// outward boundary flux, because every interior and periodic face adds and subtracts one
/// stored value `|f|F*`. Both sums are face/cell-ordered Neumaier sums; `balanced` holds when
/// `|imbalance| ≤ (faces_per_cell + 2)·ε·scale`, `scale = Σ_cells Σ_f |f||F*|`. This is a
/// round-off bound, not a bitwise claim.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvConservationReceipt {
    pub schema: String,
    pub cells: usize,
    pub interior_faces: usize,
    pub boundary_faces: usize,
    pub cell_flux_sum: f64,
    pub boundary_flux_sum: f64,
    pub imbalance: f64,
    pub scale: f64,
    pub relative_imbalance: f64,
    pub relative_bound: f64,
    pub balanced: bool,
}

/// The forward-Euler step bound `min_i |Ω_i| / Σ_f |f| λ_f` with `λ_f` the larger
/// `characteristic_speed` of the two face states.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FvCfl {
    /// `+∞` when every face speed is zero.
    pub stable_step: f64,
    /// The cell attaining the bound (`None` when unconstrained).
    pub limiting_cell: Option<usize>,
    pub max_face_speed: f64,
}

/// A first-order cell-centred finite-volume operator for one scalar conservation law.
#[derive(Debug)]
pub struct FvRealization {
    program: Arc<ConservationLawProgram>,
    topology: Arc<FvTopology>,
    face: BoundKernel,
    face_tangent: Tangent,
    speed: BoundKernel,
    boundaries: Vec<BoundaryBinding>,
    source: Option<(BoundKernel, Tangent)>,
    data: Vec<f64>,
    time_data: Vec<usize>,
    invariants: Vec<Vec<f64>>,
    scratch: Mutex<Scratch>,
    digest: Digest,
}

impl FvRealization {
    /// Bind `program` (lowered from `system`) to `topology`. `regions` maps each declared
    /// boundary region onto topology boundary tags; every tag must be claimed by exactly one
    /// declared boundary and every declared boundary must name existing tags. `data` gives a
    /// uniform value for every datum but model time, keyed by [`scientia::FvDatum::name`].
    pub fn new(
        system: &ConservationLawSystem,
        program: Arc<ConservationLawProgram>,
        topology: Arc<FvTopology>,
        regions: &RegionMap,
        data: &BTreeMap<String, f64>,
    ) -> Result<Self, FinitumError> {
        program
            .validate_identity()
            .map_err(|error| fv_error(error.code, error.message))?;
        if program.system_digest != system.artifact_digest
            || program.boundaries.len() != system.boundaries.len()
        {
            return Err(fv_error(
                "FV_IDENTITY_MISMATCH",
                "program was not lowered from this conservation-law system",
            ));
        }
        if usize::from(program.spatial_dimension) != topology.dimension() {
            return Err(fv_error(
                "FV_DIMENSION_MISMATCH",
                format!(
                    "law is {}-D, topology is {}-D",
                    program.spatial_dimension,
                    topology.dimension()
                ),
            ));
        }
        if !matches!(
            program.intent.reconstruction,
            ReconstructionIntent::Constant
        ) {
            return Err(fv_error(
                "FV_RECONSTRUCTION_UNSUPPORTED",
                "first-order realization takes constant reconstruction (MUSCL is SV2-C2b)",
            ));
        }
        if !matches!(program.intent.positivity, PositivityIntent::None) {
            return Err(fv_error(
                "FV_POSITIVITY_UNSUPPORTED",
                "positivity limiting is not realized",
            ));
        }
        let role = |kernel: &FvKernel, expected: FvKernelRole| {
            if kernel.role == expected {
                Ok(())
            } else {
                Err(fv_error(
                    "FV_KERNEL_SHAPE",
                    format!("expected a {expected:?} kernel, got {:?}", kernel.role),
                ))
            }
        };
        role(&program.face_flux, FvKernelRole::FaceFlux)?;
        role(
            &program.characteristic_speed,
            FvKernelRole::CharacteristicSpeed,
        )?;

        let mut values = Vec::with_capacity(program.data.len());
        let mut time_data = Vec::new();
        for (index, datum) in program.data.iter().enumerate() {
            if datum.source == FvDatumSource::Time {
                time_data.push(index);
                values.push(0.0);
                continue;
            }
            match data.get(&datum.name) {
                Some(value) if value.is_finite() => values.push(*value),
                _ => {
                    return Err(fv_error(
                        "FV_DATUM_MISSING",
                        format!("datum `{}` needs a finite uniform value", datum.name),
                    ));
                }
            }
        }
        let known = program
            .data
            .iter()
            .filter(|datum| datum.source != FvDatumSource::Time)
            .map(|datum| datum.name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(extra) = data.keys().find(|name| !known.contains(name.as_str())) {
            return Err(fv_error(
                "FV_DATUM_UNKNOWN",
                format!("datum `{extra}` is not read by the law"),
            ));
        }

        let face = BoundKernel::primal(&program.face_flux, true, true)?;
        let face_tangent = Tangent::of(&program.face_flux, &face)?;
        let speed = BoundKernel::primal(&program.characteristic_speed, false, true)?;

        let mut claimed = BTreeMap::<&RegionTagId, usize>::new();
        for (index, semantics) in system.boundaries.iter().enumerate() {
            let region = match semantics {
                BoundaryFluxSemantics::GhostState { region, .. }
                | BoundaryFluxSemantics::PrescribedFlux { region, .. } => *region,
            };
            let tags = regions
                .tags(region)
                .filter(|tags| !tags.is_empty())
                .ok_or_else(|| {
                    fv_error(
                        "FV_BOUNDARY_UNMAPPED",
                        format!(
                            "boundary `{}` maps to no topology tag",
                            semantics.region_name()
                        ),
                    )
                })?;
            for tag in tags {
                let Some((tag, _)) = topology.boundary().get_key_value(tag) else {
                    return Err(fv_error(
                        "FV_BOUNDARY_TAG_UNKNOWN",
                        format!(
                            "boundary `{}` maps to tag `{tag}`, which has no faces",
                            semantics.region_name()
                        ),
                    ));
                };
                if claimed.insert(tag, index).is_some() {
                    return Err(fv_error(
                        "FV_BOUNDARY_OVERLAP",
                        format!("tag `{tag}` is claimed by two boundaries"),
                    ));
                }
            }
        }
        let mut boundaries = Vec::with_capacity(topology.boundary().len());
        for tag in topology.boundary().keys() {
            let index = *claimed.get(tag).ok_or_else(|| {
                fv_error(
                    "FV_BOUNDARY_UNCLOSED",
                    format!("boundary tag `{tag}` has no flux semantics"),
                )
            })?;
            let kernel = &program.boundaries[index];
            let action = match &system.boundaries[index] {
                BoundaryFluxSemantics::GhostState { .. } => {
                    role(kernel, FvKernelRole::Ghost)?;
                    BoundaryAction::Ghost
                }
                BoundaryFluxSemantics::PrescribedFlux { .. } => {
                    role(kernel, FvKernelRole::PrescribedBoundaryFlux)?;
                    BoundaryAction::Prescribed
                }
            };
            let bound = BoundKernel::primal(kernel, false, true)?;
            let tangent = Tangent::of(kernel, &bound)?;
            boundaries.push(BoundaryBinding {
                tag: tag.clone(),
                boundary: index,
                action,
                kernel: bound,
                tangent,
            });
        }
        let source = match &program.source {
            Some(kernel) => {
                role(kernel, FvKernelRole::Source)?;
                let bound = BoundKernel::primal(kernel, false, false)?;
                let tangent = Tangent::of(kernel, &bound)?;
                Some((bound, tangent))
            }
            None => None,
        };

        let faces = topology.blocks().map(FvFaceBlock::len).max().unwrap_or(0);
        let scratch = Scratch::new(faces, topology.cell_count(), &values);
        let digest = Digest::blake3(
            &serde_json::to_vec(&(
                FV_REALIZATION_SCHEMA,
                &program.artifact_digest,
                topology.digest(),
                program
                    .data
                    .iter()
                    .zip(&values)
                    .map(|(datum, value)| (&datum.name, *value))
                    .collect::<Vec<_>>(),
                boundaries
                    .iter()
                    .map(|binding| (&binding.tag, binding.boundary))
                    .collect::<Vec<_>>(),
            ))
            .expect("FV realization identity is serializable"),
        );
        Ok(Self {
            invariants: vec![topology.cell_volumes().to_vec()],
            program,
            topology,
            face,
            face_tangent,
            speed,
            boundaries,
            source,
            data: values,
            time_data,
            scratch: Mutex::new(scratch),
            digest,
        })
    }

    pub fn program(&self) -> &ConservationLawProgram {
        &self.program
    }

    pub fn topology(&self) -> &FvTopology {
        &self.topology
    }

    /// Stable content identity over the program, topology, data values and boundary binding
    /// (`finitum-fv-realization/1`); Krasis's `OperatorIdentity` should format this.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    pub fn dimension(&self) -> usize {
        self.topology.cell_count()
    }

    pub fn receipt(&self) -> FvRealizationReceipt {
        let mut kernels = vec![
            FvKernelReceipt {
                role: FvKernelRole::FaceFlux,
                tag: None,
                kernel_digest: self.program.face_flux.digest.clone(),
                derivative: self.face_tangent.disposition(),
            },
            FvKernelReceipt {
                role: FvKernelRole::CharacteristicSpeed,
                tag: None,
                kernel_digest: self.program.characteristic_speed.digest.clone(),
                derivative: FvDerivativeDisposition::NotDifferentiated,
            },
        ];
        for binding in &self.boundaries {
            let kernel = &self.program.boundaries[binding.boundary];
            kernels.push(FvKernelReceipt {
                role: kernel.role,
                tag: Some(binding.tag.clone()),
                kernel_digest: kernel.digest.clone(),
                derivative: binding.tangent.disposition(),
            });
        }
        if let (Some(kernel), Some((_, tangent))) = (&self.program.source, &self.source) {
            kernels.push(FvKernelReceipt {
                role: FvKernelRole::Source,
                tag: None,
                kernel_digest: kernel.digest.clone(),
                derivative: tangent.disposition(),
            });
        }
        FvRealizationReceipt {
            schema: FV_REALIZATION_SCHEMA.into(),
            digest: self.digest.clone(),
            program_digest: self.program.artifact_digest.clone(),
            intent_digest: self.program.intent_digest.clone(),
            topology_digest: self.topology.digest().clone(),
            reconstruction: "constant".into(),
            kernels,
        }
    }

    fn with_scratch<T>(
        &self,
        time: f64,
        body: impl FnOnce(&mut Scratch) -> Result<T, FinitumError>,
    ) -> Result<T, FinitumError> {
        let mut fresh;
        let mut guard = self.scratch.try_lock().ok();
        let scratch = match guard.as_deref_mut() {
            Some(scratch) => scratch,
            None => {
                fresh = Scratch::new(self.scratch_faces(), self.dimension(), &self.data);
                &mut fresh
            }
        };
        for index in &self.time_data {
            scratch.data[*index] = time;
        }
        body(scratch)
    }

    fn scratch_faces(&self) -> usize {
        self.topology
            .blocks()
            .map(FvFaceBlock::len)
            .max()
            .unwrap_or(0)
    }

    fn require_state(&self, label: &str, values: &[f64]) -> Result<(), FinitumError> {
        if values.len() == self.dimension() {
            Ok(())
        } else {
            Err(fv_error(
                "FV_STATE_LENGTH",
                format!(
                    "{label} has length {}, expected {}",
                    values.len(),
                    self.dimension()
                ),
            ))
        }
    }

    /// Fill `scratch.flux` (and `scratch.plus` for ghost blocks) for one block.
    fn block_flux(
        &self,
        scratch: &mut Scratch,
        block: &FvFaceBlock,
        boundary: Option<&BoundaryBinding>,
        state: &[f64],
    ) -> Result<(), FinitumError> {
        let points = block.len();
        let Scratch {
            minus,
            plus,
            flux,
            data,
            ..
        } = scratch;
        for (slot, cell) in minus.iter_mut().zip(&block.minus) {
            *slot = state[*cell];
        }
        for (slot, cell) in plus.iter_mut().zip(&block.plus) {
            *slot = state[*cell];
        }
        match boundary.map(|binding| (&binding.action, &binding.kernel)) {
            Some((BoundaryAction::Prescribed, kernel)) => kernel.run(
                points,
                &Io {
                    minus,
                    plus: &[],
                    minus_tangent: &[],
                    plus_tangent: &[],
                    normal: &block.normal,
                    data,
                },
                flux,
            ),
            other => {
                if let Some((_, ghost)) = other {
                    ghost.run(
                        points,
                        &Io {
                            minus,
                            plus: &[],
                            minus_tangent: &[],
                            plus_tangent: &[],
                            normal: &block.normal,
                            data,
                        },
                        plus,
                    )?;
                }
                self.face.run(
                    points,
                    &Io {
                        minus,
                        plus,
                        minus_tangent: &[],
                        plus_tangent: &[],
                        normal: &block.normal,
                        data,
                    },
                    flux,
                )
            }
        }
    }

    fn source_values(&self, scratch: &mut Scratch, state: &[f64]) -> Result<bool, FinitumError> {
        let Some((source, _)) = &self.source else {
            return Ok(false);
        };
        source.run(
            state.len(),
            &Io {
                minus: state,
                plus: &[],
                minus_tangent: &[],
                plus_tangent: &[],
                normal: &[],
                data: &scratch.data,
            },
            &mut scratch.cell,
        )?;
        Ok(true)
    }

    /// `output += Σ_f s_if |f| F*_f − |Ω_i| S_i`.
    fn accumulate(
        &self,
        scratch: &mut Scratch,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        for block in [self.topology.interior(), self.topology.periodic()] {
            self.block_flux(scratch, block, None, state)?;
            scatter_block(block, &scratch.flux, output);
        }
        for (binding, block) in self
            .boundaries
            .iter()
            .zip(self.topology.boundary().values())
        {
            self.block_flux(scratch, block, Some(binding), state)?;
            scatter_block(block, &scratch.flux, output);
        }
        if self.source_values(scratch, state)? {
            let volumes = self.topology.cell_volumes();
            for ((out, volume), source) in output.iter_mut().zip(volumes).zip(&scratch.cell) {
                *out -= volume * source;
            }
        }
        Ok(())
    }

    /// `output += Σ_f s_if |f| δF*_f − |Ω_i| δS_i` for the state direction `direction`.
    fn accumulate_tangent(
        &self,
        scratch: &mut Scratch,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let face_tangent = self.face_tangent.kernel(|| "face_flux".into())?;
        let blocks = [self.topology.interior(), self.topology.periodic()]
            .into_iter()
            .map(|block| (block, None))
            .chain(
                self.topology
                    .boundary()
                    .values()
                    .zip(&self.boundaries)
                    .map(|(block, binding)| (block, Some(binding))),
            );
        for (block, boundary) in blocks {
            let points = block.len();
            if points == 0 {
                continue;
            }
            let Scratch {
                minus,
                plus,
                minus_tangent,
                plus_tangent,
                flux_tangent,
                data,
                ..
            } = &mut *scratch;
            for ((slot, tangent), cell) in minus
                .iter_mut()
                .zip(minus_tangent.iter_mut())
                .zip(&block.minus)
            {
                *slot = state[*cell];
                *tangent = direction[*cell];
            }
            for ((slot, tangent), cell) in plus
                .iter_mut()
                .zip(plus_tangent.iter_mut())
                .zip(&block.plus)
            {
                *slot = state[*cell];
                *tangent = direction[*cell];
            }
            if let Some(binding) = boundary {
                let side = Io {
                    minus,
                    plus: &[],
                    minus_tangent,
                    plus_tangent: &[],
                    normal: &block.normal,
                    data,
                };
                let tangent = binding
                    .tangent
                    .kernel(|| format!("boundary/{}", binding.tag))?;
                match binding.action {
                    BoundaryAction::Prescribed => {
                        match tangent {
                            Some(kernel) => kernel.run(points, &side, flux_tangent)?,
                            None => flux_tangent[..points].fill(0.0),
                        }
                        scatter_block(block, flux_tangent, output);
                        continue;
                    }
                    BoundaryAction::Ghost => {
                        binding.kernel.run(points, &side, plus)?;
                        match tangent {
                            Some(kernel) => kernel.run(points, &side, plus_tangent)?,
                            None => plus_tangent[..points].fill(0.0),
                        }
                    }
                }
            }
            if let Some(kernel) = face_tangent {
                kernel.run(
                    points,
                    &Io {
                        minus,
                        plus,
                        minus_tangent,
                        plus_tangent,
                        normal: &block.normal,
                        data,
                    },
                    flux_tangent,
                )?;
                scatter_block(block, flux_tangent, output);
            }
        }
        if let Some((_, tangent)) = &self.source
            && let Some(kernel) = tangent.kernel(|| "source".into())?
        {
            kernel.run(
                state.len(),
                &Io {
                    minus: state,
                    plus: &[],
                    minus_tangent: direction,
                    plus_tangent: &[],
                    normal: &[],
                    data: &scratch.data,
                },
                &mut scratch.cell,
            )?;
            let volumes = self.topology.cell_volumes();
            for ((out, volume), source) in output.iter_mut().zip(volumes).zip(&scratch.cell) {
                *out -= volume * source;
            }
        }
        Ok(())
    }

    /// The numerical flux of every face at `(time, state)` (diagnostic; allocates).
    pub fn face_fluxes(&self, time: f64, state: &[f64]) -> Result<FvFaceFluxes, FinitumError> {
        self.require_state("FV state", state)?;
        self.with_scratch(time, |scratch| {
            let mut fluxes = FvFaceFluxes {
                interior: Vec::new(),
                periodic: Vec::new(),
                boundary: BTreeMap::new(),
            };
            for (block, target) in [
                (self.topology.interior(), &mut fluxes.interior),
                (self.topology.periodic(), &mut fluxes.periodic),
            ] {
                self.block_flux(scratch, block, None, state)?;
                target.extend_from_slice(&scratch.flux[..block.len()]);
            }
            for (binding, block) in self
                .boundaries
                .iter()
                .zip(self.topology.boundary().values())
            {
                self.block_flux(scratch, block, Some(binding), state)?;
                fluxes
                    .boundary
                    .insert(binding.tag.clone(), scratch.flux[..block.len()].to_vec());
            }
            Ok(fluxes)
        })
    }

    /// The realized conservation receipt at `(time, state)`: the residual's own scatter of the
    /// face fluxes checked by [`discrete_flux_balance`].
    pub fn conservation_receipt(
        &self,
        time: f64,
        state: &[f64],
    ) -> Result<FvConservationReceipt, FinitumError> {
        let fluxes = self.face_fluxes(time, state)?;
        let mut residual = vec![0.0; self.dimension()];
        scatter_face_fluxes(&self.topology, &fluxes, &mut residual)?;
        discrete_flux_balance(&self.topology, &fluxes, &residual)
    }

    /// CFL metadata at `(time, state)`.
    pub fn cfl(&self, time: f64, state: &[f64]) -> Result<FvCfl, FinitumError> {
        self.require_state("FV state", state)?;
        self.with_scratch(time, |scratch| {
            let mut sums = std::mem::take(&mut scratch.cell);
            sums.fill(0.0);
            let result = self.speed_sums(scratch, state, &mut sums);
            let mut cfl = FvCfl {
                stable_step: f64::INFINITY,
                limiting_cell: None,
                max_face_speed: 0.0,
            };
            if let Ok(max_speed) = &result {
                cfl.max_face_speed = *max_speed;
                for (cell, (volume, sum)) in
                    self.topology.cell_volumes().iter().zip(&sums).enumerate()
                {
                    if *sum > 0.0 && volume / sum < cfl.stable_step {
                        cfl.stable_step = volume / sum;
                        cfl.limiting_cell = Some(cell);
                    }
                }
            }
            scratch.cell = sums;
            result.map(|_| cfl)
        })
    }

    /// `sums[i] = Σ_f |f| λ_f`; returns `max_f λ_f`.
    fn speed_sums(
        &self,
        scratch: &mut Scratch,
        state: &[f64],
        sums: &mut [f64],
    ) -> Result<f64, FinitumError> {
        let mut max_speed = 0.0_f64;
        let blocks = [self.topology.interior(), self.topology.periodic()]
            .into_iter()
            .map(|block| (block, None))
            .chain(
                self.topology
                    .boundary()
                    .values()
                    .zip(&self.boundaries)
                    .map(|(block, binding)| (block, Some(binding))),
            );
        for (block, boundary) in blocks {
            let points = block.len();
            if points == 0 {
                continue;
            }
            let Scratch {
                minus,
                plus,
                flux,
                flux_tangent,
                data,
                ..
            } = &mut *scratch;
            for (slot, cell) in minus.iter_mut().zip(&block.minus) {
                *slot = state[*cell];
            }
            for (slot, cell) in plus.iter_mut().zip(&block.plus) {
                *slot = state[*cell];
            }
            let one_side = |values: &[f64], output: &mut [f64]| {
                self.speed.run(
                    points,
                    &Io {
                        minus: values,
                        plus: &[],
                        minus_tangent: &[],
                        plus_tangent: &[],
                        normal: &block.normal,
                        data,
                    },
                    output,
                )
            };
            one_side(minus, flux)?;
            let two_sided = match boundary {
                None => true,
                Some(binding) => match binding.action {
                    BoundaryAction::Prescribed => false,
                    BoundaryAction::Ghost => {
                        binding.kernel.run(
                            points,
                            &Io {
                                minus,
                                plus: &[],
                                minus_tangent: &[],
                                plus_tangent: &[],
                                normal: &block.normal,
                                data,
                            },
                            plus,
                        )?;
                        true
                    }
                },
            };
            if two_sided {
                one_side(plus, flux_tangent)?;
                for (speed, other) in flux[..points].iter_mut().zip(&flux_tangent[..points]) {
                    *speed = speed.max(*other);
                }
            }
            for face in 0..points {
                let speed = flux[face];
                if !(speed.is_finite() && speed >= 0.0) {
                    return Err(fv_error(
                        "FV_SPEED_INVALID",
                        format!("characteristic speed {speed} at face {face}"),
                    ));
                }
                max_speed = max_speed.max(speed);
                let weighted = block.area[face] * speed;
                sums[block.minus[face]] += weighted;
                if let Some(plus) = block.plus.get(face) {
                    sums[*plus] += weighted;
                }
            }
        }
        Ok(max_speed)
    }

    /// `d(Σ_i |Ω_i| u_i)/dt = −Σ_b |f| F_b + Σ_i |Ω_i| S_i` (Neumaier sums).
    fn invariant_rate(&self, scratch: &mut Scratch, state: &[f64]) -> Result<f64, FinitumError> {
        let mut terms = Neumaier::default();
        for (binding, block) in self
            .boundaries
            .iter()
            .zip(self.topology.boundary().values())
        {
            self.block_flux(scratch, block, Some(binding), state)?;
            for (area, flux) in block.area.iter().zip(&scratch.flux) {
                terms.add(-(area * flux));
            }
        }
        if self.source_values(scratch, state)? {
            for (volume, source) in self.topology.cell_volumes().iter().zip(&scratch.cell) {
                terms.add(volume * source);
            }
        }
        Ok(terms.total())
    }

    fn residual_into(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.require_state("FV state", state)?;
        self.require_state("FV state rate", rate)?;
        self.require_state("FV residual", output)?;
        for ((out, volume), rate) in output
            .iter_mut()
            .zip(self.topology.cell_volumes())
            .zip(rate)
        {
            *out = volume * rate;
        }
        self.with_scratch(time, |scratch| self.accumulate(scratch, state, output))
    }

    fn jvp_into(
        &self,
        time: f64,
        state: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.require_state("FV state", state)?;
        self.require_state("FV state direction", state_direction)?;
        self.require_state("FV rate direction", rate_direction)?;
        self.require_state("FV residual direction", output)?;
        for ((out, volume), rate) in output
            .iter_mut()
            .zip(self.topology.cell_volumes())
            .zip(rate_direction)
        {
            *out = volume * rate;
        }
        self.with_scratch(time, |scratch| {
            self.accumulate_tangent(scratch, state, state_direction, output)
        })
    }

    fn rhs_into(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.require_state("FV state", state)?;
        self.require_state("FV right-hand side", output)?;
        output.fill(0.0);
        self.with_scratch(time, |scratch| self.accumulate(scratch, state, output))?;
        for (out, volume) in output.iter_mut().zip(self.topology.cell_volumes()) {
            *out = -*out / volume;
        }
        Ok(())
    }
}

/// `output[minus] += |f|F*`, `output[plus] −= |f|F*` in face order.
fn scatter_block(block: &FvFaceBlock, fluxes: &[f64], output: &mut [f64]) {
    for face in 0..block.len() {
        let term = block.area[face] * fluxes[face];
        output[block.minus[face]] += term;
        if let Some(plus) = block.plus.get(face) {
            output[*plus] -= term;
        }
    }
}

/// Scatter per-face fluxes onto cells exactly as the realization's residual does.
pub fn scatter_face_fluxes(
    topology: &FvTopology,
    fluxes: &FvFaceFluxes,
    output: &mut [f64],
) -> Result<(), FinitumError> {
    check_flux_shape(topology, fluxes, output)?;
    scatter_block(topology.interior(), &fluxes.interior, output);
    scatter_block(topology.periodic(), &fluxes.periodic, output);
    for (block, values) in topology.boundary().values().zip(fluxes.boundary.values()) {
        scatter_block(block, values, output);
    }
    Ok(())
}

fn check_flux_shape(
    topology: &FvTopology,
    fluxes: &FvFaceFluxes,
    cells: &[f64],
) -> Result<(), FinitumError> {
    let boundary_matches = topology.boundary().len() == fluxes.boundary.len()
        && topology
            .boundary()
            .iter()
            .zip(&fluxes.boundary)
            .all(|((tag, block), (other, values))| tag == other && block.len() == values.len());
    if cells.len() != topology.cell_count()
        || fluxes.interior.len() != topology.interior().len()
        || fluxes.periodic.len() != topology.periodic().len()
        || !boundary_matches
    {
        return Err(fv_error(
            "FV_FLUX_LAYOUT",
            "face fluxes or cell residual do not match the topology",
        ));
    }
    Ok(())
}

/// Check a cell flux residual (`Σ_f s_if |f| F*_f` per cell, no rate or source) against the
/// face fluxes it was scattered from; see [`FvConservationReceipt`].
pub fn discrete_flux_balance(
    topology: &FvTopology,
    fluxes: &FvFaceFluxes,
    cell_flux_residual: &[f64],
) -> Result<FvConservationReceipt, FinitumError> {
    check_flux_shape(topology, fluxes, cell_flux_residual)?;
    let mut scale = Neumaier::default();
    for (block, values) in [
        (topology.interior(), &fluxes.interior),
        (topology.periodic(), &fluxes.periodic),
    ] {
        for (area, flux) in block.area.iter().zip(values) {
            scale.add(2.0 * (area * flux).abs());
        }
    }
    let mut boundary = Neumaier::default();
    for (block, values) in topology.boundary().values().zip(fluxes.boundary.values()) {
        for (area, flux) in block.area.iter().zip(values) {
            boundary.add(area * flux);
            scale.add((area * flux).abs());
        }
    }
    let mut cells = Neumaier::default();
    for value in cell_flux_residual {
        cells.add(*value);
    }
    let (cell_flux_sum, boundary_flux_sum, scale) =
        (cells.total(), boundary.total(), scale.total());
    let imbalance = (cell_flux_sum - boundary_flux_sum).abs();
    let relative_imbalance = if scale > 0.0 {
        imbalance / scale
    } else {
        imbalance
    };
    let relative_bound = (topology.faces_per_cell() + 2) as f64 * f64::EPSILON;
    Ok(FvConservationReceipt {
        schema: FV_CONSERVATION_RECEIPT_SCHEMA.into(),
        cells: topology.cell_count(),
        interior_faces: topology.interior().len() + topology.periodic().len(),
        boundary_faces: fluxes.boundary.values().map(Vec::len).sum(),
        cell_flux_sum,
        boundary_flux_sum,
        imbalance,
        scale,
        relative_imbalance,
        relative_bound,
        balanced: relative_imbalance.is_finite() && relative_imbalance <= relative_bound,
    })
}

/// Neumaier compensated summation in insertion order.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    compensation: f64,
}

impl Neumaier {
    fn add(&mut self, value: f64) {
        let total = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - total) + value;
        } else {
            self.compensation += (value - total) + self.sum;
        }
        self.sum = total;
    }

    fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}

impl DaeOperator for FvRealization {
    fn dimension(&self) -> usize {
        self.topology.cell_count()
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.residual_into(time, state, state_rate, output)
            .map_err(NumericError::from)
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        _state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.jvp_into(time, state, state_direction, rate_direction, output)
            .map_err(NumericError::from)
    }
}

impl ExplicitOperator for FvRealization {
    fn dimension(&self) -> usize {
        self.topology.cell_count()
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.rhs_into(time, state, output)
            .map_err(NumericError::from)
    }

    fn stable_step(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        Ok(Some(self.cfl(time, state)?.stable_step))
    }

    fn linear_invariants(&self) -> &[Vec<f64>] {
        &self.invariants
    }

    fn invariant_rates(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<bool, NumericError> {
        self.require_state("FV state", state)?;
        let rate = self.with_scratch(time, |scratch| self.invariant_rate(scratch, state))?;
        output[0] = rate;
        Ok(true)
    }
}
