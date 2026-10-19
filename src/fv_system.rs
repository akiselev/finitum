//! W13 A7 (SV2C L8c): multi-component finite-volume realization
//! (`finitum-fv-system-realization/1`) of a Scientia [`MultiComponentProgram`] (compressible
//! Euler and other hyperbolic systems) on an [`FvTopology`].
//!
//! State layout is **component-major**: component `k` of cell `i` is `state[k·cells + i]`, so
//! each component is one contiguous scalar field (reconstruction runs per component) and the
//! Methodus linear invariants are the cell volumes of each component block.
//!
//! Every kernel operand is bound by its [`FvOperandMeaning`]: `StateComponent{component, side}`
//! reads component-major face (or cell) columns, `OutputComponent` writes them, normals and data
//! as in the scalar realization. Per face block one batched call each runs the ghost kernel
//! (`m` outputs: the plus state), or the prescribed boundary flux, and the facet-pair face flux
//! (`m` outputs), which are scattered in topology block order (owner `+= |f|F*_k`, neighbour
//! `-=` the same value), so interior fluxes telescope per component.
//!
//! Reconstruction is constant or MUSCL on conserved variables (component-wise least-squares /
//! axis gradients with the scalar limiters of [`FvReconstruction`]). With a Zhang–Shu intent the
//! slopes of cell `i` are scaled by the Scientia `theta` kernels before any flux is evaluated:
//! for each admissible quantity in declaration order, `q_min` is the minimum of `q_k` over the
//! cell's (already scaled) face points and `θ_k = theta_k(q_k(Ū_i), q_min)`; scaling the linear
//! reconstruction's slope by `θ_k` is exactly `U_f ← Ū + θ_k(U_f − Ū)` at every face point.
//! Cells whose average has `q_k(Ū) ≤ ε_k` refuse with `FV_STATE_INADMISSIBLE`, and every face
//! state (reconstructed, scaled or ghost) is checked against the admissible set before the face
//! flux runs, so an inadmissible face state is a typed refusal, never a NaN flux.
//!
//! Positivity of the cell averages is the integrator's concern: [`ExplicitOperator::admissibility`]
//! reports a stage state whose cell averages leave the admissible set (`q_k > ε_k` with
//! Zhang–Shu, `q_k > 0` otherwise), which Methodus/Krasis reject and retry at a smaller step.
//! The stable step `min_i |Ω_i| / Σ_f |f| λ_f` is the 1-D bound `h/(2λ)`; with a CFL number
//! `≤ 1` it meets the Zhang–Shu MUSCL condition `λΔt/h ≤ 1/2` (first-order LLF needs `≤ 1`).
//!
//! Derivatives (D6): the JVP of the residual is realized for constant and unlimited MUSCL
//! reconstruction without positivity scaling, from the Scientia JVP of each kernel; a kernel whose
//! JVP crosses an `abs/min/max` kink (LLF, HLL, HLLC faces), a limited reconstruction or a
//! Zhang–Shu scaling refuses with `FV_NONSMOOTH_DERIVATIVE_REFUSED`; the primal is unaffected.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use malleus::prepared::{ReadBinding, WriteBinding};
use malleus::{DerivativeMode, DerivativeRequest, Executable, FacetSide, OperandId, validate};
use methodus::{Admissibility, DaeOperator, EvaluationContext, ExplicitOperator, NumericError};
use scientia::{
    Digest, FV_STATE_INADMISSIBLE, FvDatumSource, FvKernel, FvKernelRole, FvOperandMeaning,
    GradientScheme, LimiterFamily, MultiComponentConservationLaw, MultiComponentProgram,
    PositivityIntent, ReconstructionIntent, ReconstructionVariables, SystemBoundaryKind,
};
use serde::Serialize;

use crate::execution::{ExecutionReceipt, PreparedExecutable, run_batch};
use crate::fv::{
    FvConservationReceipt, FvDerivativeDisposition, FvFaceFluxes, FvKernelReceipt, Neumaier, Side,
    discrete_flux_balance, fv_error,
};
use crate::fv_reconstruction::{FV_NONSMOOTH_DERIVATIVE_REFUSED, FvLimiter, FvReconstruction};
use crate::{
    FinitumError, FvCellShape, FvFaceBlock, FvTopology, InputEvaluationError, InputOrigin,
    RegionMap, RegionTagId,
};

pub const FV_SYSTEM_REALIZATION_SCHEMA: &str = "finitum-fv-system-realization/1";

/// Where one system-kernel operand reads from, or which output component it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SysColumn {
    State(usize, Side),
    Tangent(usize, Side),
    Normal(usize),
    Datum(usize),
    Mean,
    Min,
    Out(usize),
}

/// Read columns of one batched system call. `minus`/`plus` (and their tangents) are
/// component-major with `stride` values per component.
#[derive(Clone, Copy)]
struct SysIo<'a> {
    minus: &'a [f64],
    plus: &'a [f64],
    minus_tangent: &'a [f64],
    plus_tangent: &'a [f64],
    stride: usize,
    normal: &'a [Vec<f64>],
    data: &'a [f64],
    mean: &'a [f64],
    min: &'a [f64],
}

impl<'a> SysIo<'a> {
    fn new(minus: &'a [f64], stride: usize, data: &'a [f64]) -> Self {
        Self {
            minus,
            plus: &[],
            minus_tangent: &[],
            plus_tangent: &[],
            stride,
            normal: &[],
            data,
            mean: &[],
            min: &[],
        }
    }
}

/// A prepared multi-output kernel with every operand bound by meaning.
#[derive(Clone, Debug)]
struct SystemKernel {
    executable: PreparedExecutable,
    columns: Vec<SysColumn>,
    outputs: usize,
}

impl SystemKernel {
    fn new(
        kernel: malleus::StructuredKernel,
        columns: Vec<SysColumn>,
    ) -> Result<Self, FinitumError> {
        let outputs = columns
            .iter()
            .filter(|column| matches!(column, SysColumn::Out(_)))
            .count();
        if outputs == 0 {
            return Err(fv_error(
                "FV_KERNEL_SHAPE",
                format!("kernel `{}` writes no output", kernel.name),
            ));
        }
        let validated = validate(kernel)
            .map_err(|error| FinitumError::KernelValidation(format!("{error:?}")))?;
        Ok(Self {
            executable: PreparedExecutable::new(Executable::reference(validated))?,
            columns,
            outputs,
        })
    }

    /// Bind `kernel` by operand meaning; `plus`/`normal` say whether its role may read them.
    fn bind(
        kernel: &FvKernel,
        components: usize,
        plus: bool,
        normal: bool,
    ) -> Result<Self, FinitumError> {
        let shape = |what: String| {
            fv_error(
                "FV_KERNEL_SHAPE",
                format!("{:?} kernel operand {what}", kernel.role),
            )
        };
        let columns = kernel
            .operands
            .iter()
            .map(|meaning| match *meaning {
                FvOperandMeaning::StateComponent { component, side } => {
                    let component = usize::from(component);
                    if component >= components {
                        return Err(shape(format!(
                            "reads component {component} of {components}"
                        )));
                    }
                    match side {
                        None | Some(FacetSide::Minus) => {
                            Ok(SysColumn::State(component, Side::Minus))
                        }
                        Some(FacetSide::Plus) if plus => {
                            Ok(SysColumn::State(component, Side::Plus))
                        }
                        Some(FacetSide::Plus) => Err(shape("reads a plus state".into())),
                    }
                }
                FvOperandMeaning::Normal { axis } if normal => {
                    Ok(SysColumn::Normal(usize::from(axis)))
                }
                FvOperandMeaning::Datum { datum } => Ok(SysColumn::Datum(datum)),
                FvOperandMeaning::QuantityMean => Ok(SysColumn::Mean),
                FvOperandMeaning::QuantityMin => Ok(SysColumn::Min),
                FvOperandMeaning::Output => Ok(SysColumn::Out(0)),
                FvOperandMeaning::OutputComponent { component } => {
                    Ok(SysColumn::Out(usize::from(component)))
                }
                ref other => Err(shape(format!(
                    "{other:?} is not bound by a system realization"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(kernel.kernel.clone(), columns)
    }

    /// Run `points` calls; output component `k` is written to `out[k·out_stride..][..points]`.
    fn run(
        &self,
        points: usize,
        io: &SysIo<'_>,
        out: &mut [f64],
        out_stride: usize,
    ) -> Result<(), FinitumError> {
        if points == 0 {
            return Ok(());
        }
        let mut chunks = out
            .chunks_mut(out_stride)
            .take(self.outputs)
            .map(Some)
            .collect::<Vec<_>>();
        let mut reads = Vec::with_capacity(self.columns.len());
        let mut writes = Vec::with_capacity(self.outputs);
        fn column(values: &[f64], stride: usize, component: usize, points: usize) -> &[f64] {
            &values[component * stride..component * stride + points]
        }
        let stride = io.stride;
        for (index, slot) in self.columns.iter().enumerate() {
            let operand = OperandId::new(index);
            let values: &[f64] = match *slot {
                SysColumn::State(component, Side::Minus) => {
                    column(io.minus, stride, component, points)
                }
                SysColumn::State(component, Side::Plus) => {
                    column(io.plus, stride, component, points)
                }
                SysColumn::Tangent(component, Side::Minus) => {
                    column(io.minus_tangent, stride, component, points)
                }
                SysColumn::Tangent(component, Side::Plus) => {
                    column(io.plus_tangent, stride, component, points)
                }
                SysColumn::Normal(axis) => &io.normal[axis][..points],
                SysColumn::Mean => &io.mean[..points],
                SysColumn::Min => &io.min[..points],
                SysColumn::Datum(datum) => {
                    reads.push(ReadBinding::strided(operand, &io.data[datum..datum + 1], 0));
                    continue;
                }
                SysColumn::Out(component) => {
                    let chunk = chunks
                        .get_mut(component)
                        .and_then(Option::take)
                        .expect("one output column per component");
                    writes.push(WriteBinding::strided(operand, &mut chunk[..points], 1));
                    continue;
                }
            };
            reads.push(ReadBinding::strided(operand, values, 1));
        }
        run_batch(&self.executable, points, &reads, &mut writes)
            .map_err(|error| FinitumError::KernelExecution(error.to_string()))
    }
}

/// The JVP of one system kernel in its state operands.
#[derive(Clone, Debug)]
enum SysTangent {
    Zero,
    Kernel(Box<SystemKernel>),
    Refused { code: String, message: String },
}

impl SysTangent {
    fn of(kernel: &FvKernel, primal: &SystemKernel) -> Result<Self, FinitumError> {
        let indices = |keep: fn(&SysColumn) -> bool| {
            primal
                .columns
                .iter()
                .enumerate()
                .filter(|(_, column)| keep(column))
                .map(|(index, _)| OperandId::new(index))
                .collect::<Vec<_>>()
        };
        let states = indices(|column| matches!(column, SysColumn::State(..)));
        if states.is_empty() {
            return Ok(Self::Zero);
        }
        let request = DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: states,
            dependent_operands: indices(|column| matches!(column, SysColumn::Out(_))),
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
        let find = |pairs: &[malleus::DerivativeOperand], operand: OperandId| {
            pairs
                .iter()
                .find(|pair| pair.derivative == operand)
                .map(|pair| primal.columns[pair.primal.index()])
        };
        let columns = (0..product.kernel.operands.len())
            .map(|index| {
                let operand = OperandId::new(index);
                if let Some(column) = find(&product.primal_operands, operand) {
                    return Ok(column);
                }
                if let Some(SysColumn::State(component, side)) =
                    find(&product.independent_operands, operand)
                {
                    return Ok(SysColumn::Tangent(component, side));
                }
                if let Some(column @ SysColumn::Out(_)) = find(&product.dependent_operands, operand)
                {
                    return Ok(column);
                }
                Err(fv_error(
                    "FV_KERNEL_SHAPE",
                    format!("derivative operand {index} has no meaning"),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Kernel(Box::new(SystemKernel::new(
            product.kernel,
            columns,
        )?)))
    }

    fn kernel(&self, origin: &str) -> Result<Option<&SystemKernel>, FinitumError> {
        match self {
            Self::Zero => Ok(None),
            Self::Kernel(kernel) => Ok(Some(kernel)),
            Self::Refused { code, message } => Err(refusal(code, origin, message.clone())),
        }
    }

    fn disposition(&self) -> FvDerivativeDisposition {
        match self {
            Self::Zero => FvDerivativeDisposition::Zero,
            Self::Kernel(_) => FvDerivativeDisposition::Smooth,
            Self::Refused { code, .. } => FvDerivativeDisposition::Refused { code: code.clone() },
        }
    }

    fn executable(&self) -> Option<&PreparedExecutable> {
        match self {
            Self::Kernel(kernel) => Some(&kernel.executable),
            _ => None,
        }
    }
}

fn refusal(code: &str, origin: &str, message: impl Into<String>) -> FinitumError {
    FinitumError::InputEvaluation(Box::new(InputEvaluationError::new(
        code,
        InputOrigin::ExpressionPath(format!("fv/{origin}")),
        message,
    )))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryAction {
    Ghost,
    Prescribed,
}

#[derive(Clone, Debug)]
struct SysBoundary {
    tag: RegionTagId,
    boundary: usize,
    action: BoundaryAction,
    kernel: SystemKernel,
    tangent: SysTangent,
}

/// The bound admissible set: `q_k(U)` and, with Zhang–Shu, one θ kernel and `ε_k` per quantity.
#[derive(Clone, Debug)]
struct Admissible {
    names: Vec<String>,
    quantities: SystemKernel,
    theta: Vec<SystemKernel>,
    /// `ε_k` with Zhang–Shu scaling (cell averages need `q_k > ε_k`), else empty (`q_k > 0`).
    thresholds: Vec<f64>,
}

impl Admissible {
    fn floor(&self, quantity: usize) -> f64 {
        self.thresholds.get(quantity).copied().unwrap_or(0.0)
    }
}

/// Reusable evaluation buffers, component-major with `stride` faces per component.
#[derive(Debug)]
struct SysScratch {
    stride: usize,
    minus: Vec<f64>,
    plus: Vec<f64>,
    minus_tangent: Vec<f64>,
    plus_tangent: Vec<f64>,
    flux: Vec<f64>,
    quantity: Vec<f64>,
    speed_minus: Vec<f64>,
    speed_plus: Vec<f64>,
    /// `quantities × cells`: `q_k(Ū_i)`.
    cell_quantity: Vec<f64>,
    cell_min: Vec<f64>,
    theta: Vec<f64>,
    /// `components × cells × dimension` slopes (empty without reconstruction).
    slopes: Vec<f64>,
    slope_tangent: Vec<f64>,
    ratio: Vec<f64>,
    phi: Vec<f64>,
    data: Vec<f64>,
}

impl SysScratch {
    fn new(realization: &FvSystemRealization) -> Self {
        let m = realization.components;
        let cells = realization.topology.cell_count();
        let stride = realization
            .topology
            .blocks()
            .map(FvFaceBlock::len)
            .max()
            .unwrap_or(0)
            .max(cells);
        let quantities = realization
            .admissible
            .as_ref()
            .map_or(0, |admissible| admissible.names.len());
        let slopes = if realization.reconstruction.is_some() {
            cells * realization.topology.dimension()
        } else {
            0
        };
        Self {
            stride,
            minus: vec![0.0; m * stride],
            plus: vec![0.0; m * stride],
            minus_tangent: vec![0.0; m * stride],
            plus_tangent: vec![0.0; m * stride],
            flux: vec![0.0; m * stride],
            quantity: vec![0.0; quantities.max(1) * stride],
            speed_minus: vec![0.0; stride],
            speed_plus: vec![0.0; stride],
            cell_quantity: vec![0.0; quantities * cells],
            cell_min: vec![0.0; cells],
            theta: vec![0.0; cells],
            slopes: vec![0.0; m * slopes],
            slope_tangent: vec![0.0; m * slopes],
            ratio: vec![0.0; slopes],
            phi: vec![0.0; slopes],
            data: realization.data.clone(),
        }
    }
}

/// The realized MUSCL reconstruction of a system.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvSystemMusclReceipt {
    pub variables: ReconstructionVariables,
    pub gradient: GradientScheme,
    pub limiter: LimiterFamily,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limiter_kernel_digest: Option<Digest>,
    pub degenerate_gradient_cells: usize,
}

/// The realized Zhang–Shu positivity scaling.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvPositivityReceipt {
    pub scheme: String,
    pub quantities: Vec<String>,
    pub thresholds: Vec<f64>,
    pub theta_kernel_digests: Vec<Digest>,
}

/// What an [`FvSystemRealization`] bound, for run evidence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvSystemRealizationReceipt {
    pub schema: String,
    pub digest: Digest,
    pub law_digest: Digest,
    pub program_digest: Digest,
    pub intent_digest: Digest,
    pub topology_digest: Digest,
    pub components: usize,
    /// Always `component_major`: component `k` of cell `i` is `state[k·cells + i]`.
    pub layout: String,
    /// `constant` or `muscl`.
    pub reconstruction: String,
    pub kernels: Vec<FvKernelReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub muscl: Option<FvSystemMusclReceipt>,
    /// Admissible quantities checked at cell averages and face states (empty when none).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub admissible: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub positivity: Option<FvPositivityReceipt>,
    /// `Smooth` when the residual JVP is realized, else the D6 refusal code.
    pub residual_derivative: FvDerivativeDisposition,
}

/// The forward-Euler bound of a system: `min_i |Ω_i| / Σ_f |f| λ_f`, `λ_f` the larger
/// `characteristic_speed` of the two (reconstructed, scaled or ghost) face states.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FvSystemCfl {
    pub stable_step: f64,
    pub limiting_cell: Option<usize>,
    pub max_face_speed: f64,
}

/// A cell-centred finite-volume operator for one multi-component conservation law.
#[derive(Debug)]
pub struct FvSystemRealization {
    law_digest: Digest,
    program: Arc<MultiComponentProgram>,
    topology: Arc<FvTopology>,
    components: usize,
    face: SystemKernel,
    face_tangent: SysTangent,
    speed: SystemKernel,
    boundaries: Vec<SysBoundary>,
    source: Option<(SystemKernel, SysTangent)>,
    reconstruction: Option<FvReconstruction>,
    admissible: Option<Admissible>,
    data: Vec<f64>,
    time_data: Vec<usize>,
    invariants: Vec<Vec<f64>>,
    scratch: Mutex<Option<SysScratch>>,
    digest: Digest,
}

impl FvSystemRealization {
    /// Bind `program` (lowered from `law`) to `topology`. `regions` maps each law boundary
    /// region onto topology boundary tags (every tag claimed exactly once); `data` gives a
    /// finite uniform value for every datum but model time. Refuses a primitive-variable or
    /// Green–Gauss reconstruction, an axis gradient off tensor meshes, and identity, dimension
    /// or kernel-shape mismatches.
    pub fn new(
        law: &MultiComponentConservationLaw,
        program: Arc<MultiComponentProgram>,
        topology: Arc<FvTopology>,
        regions: &RegionMap,
        data: &BTreeMap<String, f64>,
    ) -> Result<Self, FinitumError> {
        program
            .validate_identity()
            .map_err(|error| fv_error(error.code, error.message))?;
        let components = program.components;
        if program.law_digest != law.artifact_digest
            || program.boundaries.len() != law.boundaries.len()
            || law.component_count() != components
        {
            return Err(fv_error(
                "FV_IDENTITY_MISMATCH",
                "program was not lowered from this multi-component law",
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
        let reconstruction = match &program.intent.reconstruction {
            ReconstructionIntent::Constant => None,
            ReconstructionIntent::Muscl {
                variables,
                gradient,
                limiter,
            } => {
                if *variables == ReconstructionVariables::Primitive {
                    return Err(fv_error(
                        "FV_PRIMITIVE_RECONSTRUCTION_UNSUPPORTED",
                        "the program declares no primitive map; systems reconstruct conserved variables",
                    ));
                }
                if *gradient == GradientScheme::GreenGauss
                    || (*gradient == GradientScheme::Axis1D
                        && topology.shape() != FvCellShape::Tensor)
                {
                    return Err(fv_error(
                        "FV_GRADIENT_UNSUPPORTED",
                        format!("{gradient:?} gradients are not realized on this mesh"),
                    ));
                }
                let limiter = match limiter {
                    LimiterFamily::None => FvLimiter::Unlimited,
                    LimiterFamily::BarthJespersen => FvLimiter::BarthJespersen,
                    LimiterFamily::Venkatakrishnan { k } => FvLimiter::Venkatakrishnan { k: *k },
                    _ => FvLimiter::Sweby(Box::new(program.limiter.clone().ok_or_else(|| {
                        fv_error(
                            "FV_KERNEL_SHAPE",
                            "a Sweby intent lowered no limiter kernel",
                        )
                    })?)),
                };
                Some(FvReconstruction::new(topology.clone(), limiter)?)
            }
        };
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

        let face = SystemKernel::bind(&program.face_flux, components, true, true)?;
        let face_tangent = SysTangent::of(&program.face_flux, &face)?;
        let speed = SystemKernel::bind(&program.characteristic_speed, components, false, true)?;
        if face.outputs != components || speed.outputs != 1 {
            return Err(fv_error(
                "FV_KERNEL_SHAPE",
                "face flux needs one output per component and the speed kernel one output",
            ));
        }

        let mut claimed = BTreeMap::<&RegionTagId, usize>::new();
        for (index, boundary) in law.boundaries.iter().enumerate() {
            let tags = regions
                .tags(boundary.region)
                .filter(|tags| !tags.is_empty())
                .ok_or_else(|| {
                    fv_error(
                        "FV_BOUNDARY_UNMAPPED",
                        format!(
                            "boundary `{}` maps to no topology tag",
                            boundary.region_name
                        ),
                    )
                })?;
            for tag in tags {
                let Some((tag, _)) = topology.boundary().get_key_value(tag) else {
                    return Err(fv_error(
                        "FV_BOUNDARY_TAG_UNKNOWN",
                        format!(
                            "boundary `{}` maps to tag `{tag}`, which has no faces",
                            boundary.region_name
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
            let action = match law.boundaries[index].kind {
                SystemBoundaryKind::GhostState => {
                    role(kernel, FvKernelRole::Ghost)?;
                    BoundaryAction::Ghost
                }
                SystemBoundaryKind::PrescribedFlux => {
                    role(kernel, FvKernelRole::PrescribedBoundaryFlux)?;
                    BoundaryAction::Prescribed
                }
            };
            let bound = SystemKernel::bind(kernel, components, false, true)?;
            if bound.outputs != components {
                return Err(fv_error(
                    "FV_KERNEL_SHAPE",
                    format!("boundary kernel of `{tag}` needs {components} outputs"),
                ));
            }
            let tangent = SysTangent::of(kernel, &bound)?;
            boundaries.push(SysBoundary {
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
                let bound = SystemKernel::bind(kernel, components, false, false)?;
                let tangent = SysTangent::of(kernel, &bound)?;
                Some((bound, tangent))
            }
            None => None,
        };
        let admissible = match &program.admissibility {
            None => {
                if !matches!(program.intent.positivity, PositivityIntent::None) {
                    return Err(fv_error(
                        "FV_POSITIVITY_UNSUPPORTED",
                        "Zhang-Shu scaling without an admissible set",
                    ));
                }
                None
            }
            Some(admissibility) => {
                role(
                    &admissibility.quantity_kernel,
                    FvKernelRole::AdmissibleQuantities,
                )?;
                let quantities =
                    SystemKernel::bind(&admissibility.quantity_kernel, components, false, false)?;
                let theta = admissibility
                    .theta
                    .iter()
                    .map(|kernel| {
                        role(kernel, FvKernelRole::PositivityTheta)?;
                        SystemKernel::bind(kernel, components, false, false)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let thresholds = admissibility
                    .quantities
                    .iter()
                    .filter_map(|quantity| quantity.threshold)
                    .collect::<Vec<_>>();
                let scaling = matches!(
                    program.intent.positivity,
                    PositivityIntent::ZhangShuScaling { .. }
                );
                let count = admissibility.quantities.len();
                if quantities.outputs != count
                    || (scaling && (theta.len() != count || thresholds.len() != count))
                    || (!scaling && !theta.is_empty())
                {
                    return Err(fv_error(
                        "FV_KERNEL_SHAPE",
                        "admissibility program does not match its quantities",
                    ));
                }
                Some(Admissible {
                    names: admissibility
                        .quantities
                        .iter()
                        .map(|quantity| quantity.name.clone())
                        .collect(),
                    quantities,
                    theta,
                    thresholds,
                })
            }
        };

        let cells = topology.cell_count();
        let invariants = (0..components)
            .map(|k| {
                let mut weights = vec![0.0; components * cells];
                weights[k * cells..(k + 1) * cells].copy_from_slice(topology.cell_volumes());
                weights
            })
            .collect();
        let digest = Digest::blake3(
            &serde_json::to_vec(&(
                FV_SYSTEM_REALIZATION_SCHEMA,
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
            .expect("FV system realization identity is serializable"),
        );
        let mut realization = Self {
            law_digest: law.artifact_digest.clone(),
            program,
            topology,
            components,
            face,
            face_tangent,
            speed,
            boundaries,
            source,
            reconstruction,
            admissible,
            data: values,
            time_data,
            invariants,
            scratch: Mutex::new(None),
            digest,
        };
        let scratch = SysScratch::new(&realization);
        realization.scratch = Mutex::new(Some(scratch));
        Ok(realization)
    }

    pub fn program(&self) -> &MultiComponentProgram {
        &self.program
    }

    pub fn topology(&self) -> &FvTopology {
        &self.topology
    }

    /// Identity over program, topology, data values and boundary binding (for Krasis
    /// `OperatorIdentity`).
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    pub fn components(&self) -> usize {
        self.components
    }

    /// `components × cells` (component-major).
    pub fn dimension(&self) -> usize {
        self.components * self.topology.cell_count()
    }

    pub fn execution_receipt(&self) -> ExecutionReceipt {
        let mut kernels = vec![&self.face.executable, &self.speed.executable];
        kernels.extend(self.face_tangent.executable());
        for boundary in &self.boundaries {
            kernels.push(&boundary.kernel.executable);
            kernels.extend(boundary.tangent.executable());
        }
        if let Some((source, tangent)) = &self.source {
            kernels.push(&source.executable);
            kernels.extend(tangent.executable());
        }
        if let Some(admissible) = &self.admissible {
            kernels.push(&admissible.quantities.executable);
            kernels.extend(admissible.theta.iter().map(|kernel| &kernel.executable));
        }
        ExecutionReceipt::collect(kernels)
    }

    /// The residual-JVP disposition: `Err` with the typed D6 refusal when any part of the
    /// residual is not smoothly differentiable.
    fn derivative_refusal(&self) -> Result<(), FinitumError> {
        if let Some(reconstruction) = &self.reconstruction
            && let Some((code, message)) = reconstruction.derivative_refusal()
        {
            return Err(refusal(code, "limiter", message));
        }
        if self
            .admissible
            .as_ref()
            .is_some_and(|admissible| !admissible.theta.is_empty())
        {
            return Err(refusal(
                FV_NONSMOOTH_DERIVATIVE_REFUSED,
                "positivity",
                "the Zhang-Shu scaling theta = min(1, ...) is not differentiable at its switch",
            ));
        }
        self.face_tangent.kernel("face_flux")?;
        for boundary in &self.boundaries {
            boundary
                .tangent
                .kernel(&format!("boundary/{}", boundary.tag))?;
        }
        if let Some((_, tangent)) = &self.source {
            tangent.kernel("source")?;
        }
        Ok(())
    }

    pub fn receipt(&self) -> FvSystemRealizationReceipt {
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
        for boundary in &self.boundaries {
            let kernel = &self.program.boundaries[boundary.boundary];
            kernels.push(FvKernelReceipt {
                role: kernel.role,
                tag: Some(boundary.tag.clone()),
                kernel_digest: kernel.digest.clone(),
                derivative: boundary.tangent.disposition(),
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
        let muscl = match (&self.program.intent.reconstruction, &self.reconstruction) {
            (
                ReconstructionIntent::Muscl {
                    variables,
                    gradient,
                    limiter,
                },
                Some(reconstruction),
            ) => Some(FvSystemMusclReceipt {
                variables: *variables,
                gradient: *gradient,
                limiter: limiter.clone(),
                limiter_kernel_digest: self
                    .program
                    .limiter
                    .as_ref()
                    .map(|kernel| kernel.digest.clone()),
                degenerate_gradient_cells: reconstruction.degenerate_cells(),
            }),
            _ => None,
        };
        let positivity = self.program.admissibility.as_ref().and_then(|program| {
            (!program.theta.is_empty()).then(|| FvPositivityReceipt {
                scheme: "zhang_shu_scaling".into(),
                quantities: program.quantities.iter().map(|q| q.name.clone()).collect(),
                thresholds: program
                    .quantities
                    .iter()
                    .filter_map(|q| q.threshold)
                    .collect(),
                theta_kernel_digests: program.theta.iter().map(|k| k.digest.clone()).collect(),
            })
        });
        FvSystemRealizationReceipt {
            schema: FV_SYSTEM_REALIZATION_SCHEMA.into(),
            digest: self.digest.clone(),
            law_digest: self.law_digest.clone(),
            program_digest: self.program.artifact_digest.clone(),
            intent_digest: self.program.intent_digest.clone(),
            topology_digest: self.topology.digest().clone(),
            components: self.components,
            layout: "component_major".into(),
            reconstruction: if self.reconstruction.is_some() {
                "muscl".into()
            } else {
                "constant".into()
            },
            kernels,
            muscl,
            admissible: self
                .admissible
                .as_ref()
                .map(|admissible| admissible.names.clone())
                .unwrap_or_default(),
            positivity,
            residual_derivative: match self.derivative_refusal() {
                Ok(()) => FvDerivativeDisposition::Smooth,
                Err(error) => FvDerivativeDisposition::Refused {
                    code: error
                        .code()
                        .unwrap_or(FV_NONSMOOTH_DERIVATIVE_REFUSED)
                        .into(),
                },
            },
        }
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

    fn with_scratch<T>(
        &self,
        time: f64,
        body: impl FnOnce(&mut SysScratch) -> Result<T, FinitumError>,
    ) -> Result<T, FinitumError> {
        let mut fresh;
        let mut guard = self.scratch.try_lock().ok();
        let scratch = match guard.as_deref_mut().and_then(Option::as_mut) {
            Some(scratch) => scratch,
            None => {
                fresh = SysScratch::new(self);
                &mut fresh
            }
        };
        for index in &self.time_data {
            scratch.data[*index] = time;
        }
        body(scratch)
    }

    fn blocks(&self) -> impl Iterator<Item = (usize, &FvFaceBlock, Option<&SysBoundary>)> {
        [self.topology.interior(), self.topology.periodic()]
            .into_iter()
            .map(|block| (block, None))
            .chain(
                self.topology
                    .boundary()
                    .values()
                    .zip(&self.boundaries)
                    .map(|(block, binding)| (block, Some(binding))),
            )
            .enumerate()
            .map(|(index, (block, binding))| (index, block, binding))
    }

    /// `q_k(Ū_i)` for every cell into `cell_quantity`; the first `(cell, quantity, value)` with
    /// `q_k ≤ floor_k` (or NaN), if any.
    fn first_inadmissible(
        &self,
        scratch: &mut SysScratch,
        state: &[f64],
    ) -> Result<Option<(usize, usize, f64)>, FinitumError> {
        let Some(admissible) = &self.admissible else {
            return Ok(None);
        };
        let cells = self.topology.cell_count();
        let SysScratch {
            cell_quantity,
            data,
            ..
        } = scratch;
        admissible
            .quantities
            .run(cells, &SysIo::new(state, cells, data), cell_quantity, cells)?;
        for quantity in 0..admissible.names.len() {
            let floor = admissible.floor(quantity);
            let values = &cell_quantity[quantity * cells..(quantity + 1) * cells];
            if let Some(cell) = values
                .iter()
                .position(|value| value.is_nan() || *value <= floor)
            {
                return Ok(Some((cell, quantity, values[cell])));
            }
        }
        Ok(None)
    }

    fn inadmissible_reason(&self, cell: usize, quantity: usize, value: f64) -> String {
        let admissible = self.admissible.as_ref().expect("an admissible set");
        format!(
            "cell {cell} average has {} = {value:e}, not above {:e}",
            admissible.names[quantity],
            admissible.floor(quantity)
        )
    }

    /// [`Self::first_inadmissible`] as a typed `FV_STATE_INADMISSIBLE` refusal.
    fn cell_quantities(&self, scratch: &mut SysScratch, state: &[f64]) -> Result<(), FinitumError> {
        match self.first_inadmissible(scratch, state)? {
            None => Ok(()),
            Some((cell, quantity, value)) => Err(refusal(
                FV_STATE_INADMISSIBLE,
                "admissibility",
                self.inadmissible_reason(cell, quantity, value),
            )),
        }
    }

    /// Reconstructed face states of block `index` into `minus`/`plus` (component-major).
    fn gather(
        &self,
        index: usize,
        block: &FvFaceBlock,
        state: &[f64],
        slopes: &[f64],
        stride: usize,
        (minus, plus): (&mut [f64], &mut [f64]),
    ) {
        let cells = self.topology.cell_count();
        let width = cells * self.topology.dimension();
        for k in 0..self.components {
            let values = &state[k * cells..(k + 1) * cells];
            let (minus, plus) = (
                &mut minus[k * stride..(k + 1) * stride],
                &mut plus[k * stride..(k + 1) * stride],
            );
            match &self.reconstruction {
                Some(reconstruction) => reconstruction.face_values(
                    index,
                    block,
                    values,
                    &slopes[k * width..(k + 1) * width],
                    minus,
                    plus,
                ),
                None => {
                    for (slot, cell) in minus.iter_mut().zip(&block.minus) {
                        *slot = values[*cell];
                    }
                    for (slot, cell) in plus.iter_mut().zip(&block.plus) {
                        *slot = values[*cell];
                    }
                }
            }
        }
    }

    /// Limited (and, with Zhang–Shu, θ-scaled) slopes of `state` into `scratch.slopes`.
    fn prepare(&self, scratch: &mut SysScratch, state: &[f64]) -> Result<(), FinitumError> {
        self.cell_quantities(scratch, state)?;
        let Some(reconstruction) = &self.reconstruction else {
            return Ok(());
        };
        let cells = self.topology.cell_count();
        let dimension = self.topology.dimension();
        let width = cells * dimension;
        for k in 0..self.components {
            let SysScratch {
                slopes, ratio, phi, ..
            } = &mut *scratch;
            reconstruction.slopes_into(
                &state[k * cells..(k + 1) * cells],
                &mut slopes[k * width..(k + 1) * width],
                ratio,
                phi,
            )?;
        }
        let Some(admissible) = &self.admissible else {
            return Ok(());
        };
        for (quantity, theta_kernel) in admissible.theta.iter().enumerate() {
            let SysScratch {
                stride,
                minus,
                plus,
                quantity: values,
                cell_quantity,
                cell_min,
                theta,
                slopes,
                data,
                ..
            } = &mut *scratch;
            let stride = *stride;
            cell_min.fill(f64::INFINITY);
            for (index, block, _) in self.blocks() {
                let points = block.len();
                if points == 0 {
                    continue;
                }
                self.gather(index, block, state, slopes, stride, (minus, plus));
                admissible.quantities.run(
                    points,
                    &SysIo::new(minus, stride, data),
                    values,
                    stride,
                )?;
                for (face, cell) in block.minus.iter().enumerate() {
                    let value = values[quantity * stride + face];
                    cell_min[*cell] = min_nan(cell_min[*cell], value);
                }
                if !block.plus.is_empty() {
                    admissible.quantities.run(
                        points,
                        &SysIo::new(plus, stride, data),
                        values,
                        stride,
                    )?;
                    for (face, cell) in block.plus.iter().enumerate() {
                        let value = values[quantity * stride + face];
                        cell_min[*cell] = min_nan(cell_min[*cell], value);
                    }
                }
            }
            let mut io = SysIo::new(&[], cells, data);
            io.mean = &cell_quantity[quantity * cells..(quantity + 1) * cells];
            io.min = &cell_min[..];
            theta_kernel.run(cells, &io, theta, cells)?;
            for (cell, factor) in theta.iter().enumerate() {
                if !(0.0..=1.0).contains(factor) {
                    return Err(refusal(
                        FV_STATE_INADMISSIBLE,
                        "positivity",
                        format!(
                            "cell {cell}: Zhang-Shu theta for {} is {factor:e} (face minimum {:e})",
                            admissible.names[quantity], cell_min[cell]
                        ),
                    ));
                }
                if *factor < 1.0 {
                    for k in 0..self.components {
                        for slope in &mut slopes
                            [k * width + cell * dimension..k * width + (cell + 1) * dimension]
                        {
                            *slope *= *factor;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Refuse face states (`points` component-major columns) outside the admissible set.
    fn check_faces(
        &self,
        states: &[f64],
        stride: usize,
        data: &[f64],
        values: &mut [f64],
        block: &FvFaceBlock,
        side: &str,
    ) -> Result<(), FinitumError> {
        let Some(admissible) = &self.admissible else {
            return Ok(());
        };
        let points = block.len();
        admissible
            .quantities
            .run(points, &SysIo::new(states, stride, data), values, stride)?;
        for (quantity, name) in admissible.names.iter().enumerate() {
            if let Some(face) = values[quantity * stride..quantity * stride + points]
                .iter()
                .position(|value| value.is_nan() || *value <= 0.0)
            {
                let cell = if side == "plus" {
                    block.plus.get(face)
                } else {
                    block.minus.get(face)
                };
                return Err(refusal(
                    FV_STATE_INADMISSIBLE,
                    "face_state",
                    format!(
                        "{side} face state of cell {cell:?} (face {face}) has {name} = {:e}",
                        values[quantity * stride + face]
                    ),
                ));
            }
        }
        Ok(())
    }

    /// `cell_flux[k·cells + i] = Σ_f s_if |f| F*_k` (overwritten). With `record`, also the
    /// per-component face fluxes.
    fn flux_residual(
        &self,
        scratch: &mut SysScratch,
        state: &[f64],
        cell_flux: &mut [f64],
        mut record: Option<&mut Vec<FvFaceFluxes>>,
    ) -> Result<(), FinitumError> {
        self.prepare(scratch, state)?;
        cell_flux.fill(0.0);
        let cells = self.topology.cell_count();
        let SysScratch {
            stride,
            minus,
            plus,
            flux,
            quantity,
            slopes,
            data,
            ..
        } = scratch;
        let stride = *stride;
        for (index, block, binding) in self.blocks() {
            let points = block.len();
            if points == 0 {
                if let Some(record) = record.as_deref_mut() {
                    push_record(record, self.components, index, block, binding, &[], stride);
                }
                continue;
            }
            self.gather(index, block, state, slopes, stride, (minus, plus));
            self.check_faces(minus, stride, data, quantity, block, "minus")?;
            let mut io = SysIo::new(minus, stride, data);
            io.normal = &block.normal[..];
            match binding.map(|binding| (binding.action, &binding.kernel)) {
                Some((BoundaryAction::Prescribed, kernel)) => {
                    kernel.run(points, &io, flux, stride)?;
                }
                ghost => {
                    if let Some((_, kernel)) = ghost {
                        kernel.run(points, &io, plus, stride)?;
                    }
                    self.check_faces(plus, stride, data, quantity, block, "plus")?;
                    let mut io = SysIo::new(minus, stride, data);
                    io.plus = &plus[..];
                    io.normal = &block.normal[..];
                    self.face.run(points, &io, flux, stride)?;
                }
            }
            scatter(block, flux, stride, cells, self.components, cell_flux);
            if let Some(record) = record.as_deref_mut() {
                push_record(record, self.components, index, block, binding, flux, stride);
            }
        }
        Ok(())
    }

    /// `|Ω_i| S_k(U_i)` into `output` (overwritten); zero without a source.
    fn source_terms(
        &self,
        scratch: &mut SysScratch,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let cells = self.topology.cell_count();
        match &self.source {
            None => output.fill(0.0),
            Some((kernel, _)) => {
                kernel.run(
                    cells,
                    &SysIo::new(state, cells, &scratch.data),
                    output,
                    cells,
                )?;
                for k in 0..self.components {
                    for (value, volume) in output[k * cells..(k + 1) * cells]
                        .iter_mut()
                        .zip(self.topology.cell_volumes())
                    {
                        *value *= volume;
                    }
                }
            }
        }
        Ok(())
    }

    fn rhs_into(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.require_state("FV system state", state)?;
        self.require_state("FV system output", output)?;
        let cells = self.topology.cell_count();
        self.with_scratch(time, |scratch| {
            let mut source = vec![0.0; output.len()];
            self.flux_residual(scratch, state, output, None)?;
            self.source_terms(scratch, state, &mut source)?;
            for k in 0..self.components {
                for (i, volume) in self.topology.cell_volumes().iter().enumerate() {
                    let slot = k * cells + i;
                    output[slot] = -(output[slot] - source[slot]) / volume;
                }
            }
            Ok(())
        })
    }

    fn residual_into(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.require_state("FV system state", state)?;
        self.require_state("FV system rate", rate)?;
        self.require_state("FV system output", output)?;
        let cells = self.topology.cell_count();
        self.with_scratch(time, |scratch| {
            let mut source = vec![0.0; output.len()];
            self.flux_residual(scratch, state, output, None)?;
            self.source_terms(scratch, state, &mut source)?;
            for k in 0..self.components {
                for (i, volume) in self.topology.cell_volumes().iter().enumerate() {
                    let slot = k * cells + i;
                    output[slot] += volume * rate[slot] - source[slot];
                }
            }
            Ok(())
        })
    }

    /// `(∂R/∂U)·direction + |Ω|·rate_direction` for constant or unlimited MUSCL
    /// reconstruction; the typed D6 refusal otherwise.
    fn jvp_into(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.require_state("FV system state", state)?;
        self.require_state("FV system direction", direction)?;
        self.require_state("FV system rate direction", rate_direction)?;
        self.require_state("FV system output", output)?;
        self.derivative_refusal()?;
        let cells = self.topology.cell_count();
        let width = cells * self.topology.dimension();
        self.with_scratch(time, |scratch| {
            self.prepare(scratch, state)?;
            if let Some(reconstruction) = &self.reconstruction {
                for k in 0..self.components {
                    reconstruction.gradient_into(
                        &direction[k * cells..(k + 1) * cells],
                        &mut scratch.slope_tangent[k * width..(k + 1) * width],
                    );
                }
            }
            output.fill(0.0);
            let SysScratch {
                stride,
                minus,
                plus,
                minus_tangent,
                plus_tangent,
                flux,
                slopes,
                slope_tangent,
                data,
                ..
            } = &mut *scratch;
            let stride = *stride;
            for (index, block, binding) in self.blocks() {
                let points = block.len();
                if points == 0 {
                    continue;
                }
                self.gather(index, block, state, slopes, stride, (minus, plus));
                self.gather(
                    index,
                    block,
                    direction,
                    slope_tangent,
                    stride,
                    (minus_tangent, plus_tangent),
                );
                let mut io = SysIo::new(minus, stride, data);
                io.minus_tangent = &minus_tangent[..];
                io.normal = &block.normal[..];
                let tangent = match binding {
                    Some(binding) => {
                        let tangent = binding.tangent.kernel("boundary")?;
                        if binding.action == BoundaryAction::Prescribed {
                            match tangent {
                                Some(kernel) => kernel.run(points, &io, flux, stride)?,
                                None => flux.fill(0.0),
                            }
                            scatter(block, flux, stride, cells, self.components, output);
                            continue;
                        }
                        binding.kernel.run(points, &io, plus, stride)?;
                        match tangent {
                            Some(kernel) => kernel.run(points, &io, plus_tangent, stride)?,
                            None => plus_tangent.fill(0.0),
                        }
                        self.face_tangent.kernel("face_flux")?
                    }
                    None => self.face_tangent.kernel("face_flux")?,
                };
                let Some(kernel) = tangent else {
                    continue;
                };
                let mut io = SysIo::new(minus, stride, data);
                io.plus = &plus[..];
                io.minus_tangent = &minus_tangent[..];
                io.plus_tangent = &plus_tangent[..];
                io.normal = &block.normal[..];
                kernel.run(points, &io, flux, stride)?;
                scatter(block, flux, stride, cells, self.components, output);
            }
            if let Some((_, tangent)) = &self.source
                && let Some(kernel) = tangent.kernel("source")?
            {
                let mut io = SysIo::new(state, cells, data);
                io.minus_tangent = direction;
                let mut source = vec![0.0; output.len()];
                kernel.run(cells, &io, &mut source, cells)?;
                for k in 0..self.components {
                    for (i, volume) in self.topology.cell_volumes().iter().enumerate() {
                        output[k * cells + i] -= volume * source[k * cells + i];
                    }
                }
            }
            for k in 0..self.components {
                for (i, volume) in self.topology.cell_volumes().iter().enumerate() {
                    output[k * cells + i] += volume * rate_direction[k * cells + i];
                }
            }
            Ok(())
        })
    }

    /// Per-component numerical face fluxes `F*_k` in topology block order (diagnostic).
    pub fn face_fluxes(&self, time: f64, state: &[f64]) -> Result<Vec<FvFaceFluxes>, FinitumError> {
        self.require_state("FV system state", state)?;
        let mut cell_flux = vec![0.0; state.len()];
        let mut record = Vec::new();
        self.with_scratch(time, |scratch| {
            self.flux_residual(scratch, state, &mut cell_flux, Some(&mut record))
        })?;
        Ok(record)
    }

    /// The `DiscreteFluxBalance` receipt of every component (see [`FvConservationReceipt`]).
    pub fn conservation_receipts(
        &self,
        time: f64,
        state: &[f64],
    ) -> Result<Vec<FvConservationReceipt>, FinitumError> {
        self.require_state("FV system state", state)?;
        let cells = self.topology.cell_count();
        let mut cell_flux = vec![0.0; state.len()];
        let mut record = Vec::new();
        self.with_scratch(time, |scratch| {
            self.flux_residual(scratch, state, &mut cell_flux, Some(&mut record))
        })?;
        record
            .iter()
            .enumerate()
            .map(|(k, fluxes)| {
                discrete_flux_balance(
                    &self.topology,
                    fluxes,
                    &cell_flux[k * cells..(k + 1) * cells],
                )
            })
            .collect()
    }

    /// The forward-Euler step bound at the (reconstructed, scaled, ghost) face states.
    pub fn cfl(&self, time: f64, state: &[f64]) -> Result<FvSystemCfl, FinitumError> {
        self.require_state("FV system state", state)?;
        let cells = self.topology.cell_count();
        self.with_scratch(time, |scratch| {
            self.prepare(scratch, state)?;
            let mut sums = vec![0.0; cells];
            let mut max_speed = 0.0_f64;
            let SysScratch {
                stride,
                minus,
                plus,
                speed_minus,
                speed_plus,
                slopes,
                data,
                ..
            } = &mut *scratch;
            let stride = *stride;
            for (index, block, binding) in self.blocks() {
                let points = block.len();
                if points == 0 {
                    continue;
                }
                self.gather(index, block, state, slopes, stride, (minus, plus));
                let mut io = SysIo::new(minus, stride, data);
                io.normal = &block.normal[..];
                self.speed.run(points, &io, speed_minus, stride)?;
                let two_sided = match binding {
                    None => true,
                    Some(binding) if binding.action == BoundaryAction::Ghost => {
                        binding.kernel.run(points, &io, plus, stride)?;
                        true
                    }
                    Some(_) => false,
                };
                if two_sided {
                    let mut io = SysIo::new(plus, stride, data);
                    io.normal = &block.normal[..];
                    self.speed.run(points, &io, speed_plus, stride)?;
                } else {
                    speed_plus[..points].fill(0.0);
                }
                for face in 0..points {
                    let lambda = speed_minus[face].max(speed_plus[face]);
                    if !lambda.is_finite() {
                        return Err(refusal(
                            FV_STATE_INADMISSIBLE,
                            "characteristic_speed",
                            format!("face {face} of block {index} has wave speed {lambda:e}"),
                        ));
                    }
                    max_speed = max_speed.max(lambda);
                    let term = block.area[face] * lambda;
                    sums[block.minus[face]] += term;
                    if let Some(plus) = block.plus.get(face) {
                        sums[*plus] += term;
                    }
                }
            }
            let mut cfl = FvSystemCfl {
                stable_step: f64::INFINITY,
                limiting_cell: None,
                max_face_speed: max_speed,
            };
            for (cell, (volume, sum)) in self.topology.cell_volumes().iter().zip(&sums).enumerate()
            {
                if *sum > 0.0 && volume / sum < cfl.stable_step {
                    cfl.stable_step = volume / sum;
                    cfl.limiting_cell = Some(cell);
                }
            }
            Ok(cfl)
        })
    }

    /// Whether every cell average lies in the admissible set (`q_k > ε_k` with Zhang–Shu,
    /// `q_k > 0` otherwise); `Inadmissible.index` is the state index of component 0 of the
    /// first offending cell.
    pub fn admissibility(&self, time: f64, state: &[f64]) -> Result<Admissibility, FinitumError> {
        self.require_state("FV system state", state)?;
        if self.admissible.is_none() {
            return Ok(Admissibility::Admissible);
        }
        self.with_scratch(time, |scratch| {
            Ok(match self.first_inadmissible(scratch, state)? {
                None => Admissibility::Admissible,
                Some((cell, quantity, value)) => Admissibility::Inadmissible {
                    index: Some(cell),
                    reason: format!(
                        "{FV_STATE_INADMISSIBLE}: {}",
                        self.inadmissible_reason(cell, quantity, value)
                    ),
                },
            })
        })
    }

    fn invariant_rates_into(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let cells = self.topology.cell_count();
        let fluxes = self.face_fluxes(time, state)?;
        let mut source = vec![0.0; state.len()];
        self.with_scratch(time, |scratch| {
            self.source_terms(scratch, state, &mut source)
        })?;
        for (k, fluxes) in fluxes.iter().enumerate() {
            let mut rate = Neumaier::default();
            for (block, values) in self
                .topology
                .boundary()
                .values()
                .zip(fluxes.boundary.values())
            {
                for (area, flux) in block.area.iter().zip(values) {
                    rate.add(-(area * flux));
                }
            }
            for value in &source[k * cells..(k + 1) * cells] {
                rate.add(*value);
            }
            output[k] = rate.total();
        }
        Ok(())
    }
}

/// `min` that propagates NaN (a NaN face quantity makes the cell minimum NaN).
fn min_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// `cell_flux[k·cells + minus] += |f|F*_k`, `[plus] −=` the same value, in face order.
fn scatter(
    block: &FvFaceBlock,
    flux: &[f64],
    stride: usize,
    cells: usize,
    components: usize,
    cell_flux: &mut [f64],
) {
    for k in 0..components {
        let output = &mut cell_flux[k * cells..(k + 1) * cells];
        for face in 0..block.len() {
            let term = block.area[face] * flux[k * stride + face];
            output[block.minus[face]] += term;
            if let Some(plus) = block.plus.get(face) {
                output[*plus] -= term;
            }
        }
    }
}

fn push_record(
    record: &mut Vec<FvFaceFluxes>,
    components: usize,
    index: usize,
    block: &FvFaceBlock,
    binding: Option<&SysBoundary>,
    flux: &[f64],
    stride: usize,
) {
    if record.is_empty() {
        record.resize_with(components, || FvFaceFluxes {
            interior: Vec::new(),
            periodic: Vec::new(),
            boundary: BTreeMap::new(),
        });
    }
    for (k, fluxes) in record.iter_mut().enumerate() {
        let values = if flux.is_empty() {
            Vec::new()
        } else {
            flux[k * stride..k * stride + block.len()].to_vec()
        };
        match (index, binding) {
            (0, _) => fluxes.interior = values,
            (1, _) => fluxes.periodic = values,
            (_, Some(binding)) => {
                fluxes.boundary.insert(binding.tag.clone(), values);
            }
            (_, None) => {}
        }
    }
}

impl DaeOperator for FvSystemRealization {
    fn dimension(&self) -> usize {
        FvSystemRealization::dimension(self)
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

impl ExplicitOperator for FvSystemRealization {
    fn dimension(&self) -> usize {
        FvSystemRealization::dimension(self)
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

    fn admissibility(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Admissibility, NumericError> {
        Ok(FvSystemRealization::admissibility(self, time, state)?)
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
        self.require_state("FV system state", state)?;
        self.invariant_rates_into(time, state, output)?;
        Ok(true)
    }
}
