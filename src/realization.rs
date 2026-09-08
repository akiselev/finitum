use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

use malleus::{
    AccessMode, BufferBinding, Executable, ExecutableModule, Interpreter, OperandId,
    validate_module,
};
use methodus::{
    CsrMatrix, EvaluationContext, LinearOperator, NumericError, OperatorSymmetry,
    TransposableOperator,
};
use scientia::scientific::ValueShape;
use scientia::{
    DerivativeEvaluation, Digest, ElementFamilyRequirement, EvaluationSite, FormRequirements,
    InputSourceRequirement, IntegralOperatorFactorization, OperatorFactorization, QFunctionInput,
    SemanticMeasure, StructuredOperatorKernels, StructuredPointKernelBundle, SymbolId,
    TensorInputId, TensorInputRole,
};
use serde::Serialize;

use crate::optimized::{ElementAssemblyOperator, PartialAssemblyOperator, PartialPointAction};
use crate::profile::{
    FieldSource, PartitionReport, RegionMap, RegionTags, evaluate_kernel_partial,
    evaluate_kernel_value, evaluate_table_slope, evaluate_table_value, named_coordinate_inputs,
    partition_report_for,
};
use crate::{
    CellId, ConstraintSet, DofMap, FacetId, FacetIncidence, FacetTopology, FinitumError,
    InputEvaluationError, Mesh, PreparedElement, QuadraturePoint, simplex_basis,
};

pub const REALIZATION_ARTIFACT_SCHEMA: &str = "finitum-realization-plan/2";

/// Concrete quadrature-point values for one non-basis QFunction input.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalInput {
    pub integral_index: usize,
    pub input: TensorInputId,
    component_count: usize,
    values: Vec<f64>,
}

/// SV1-C3: the caller-owned design space a stored external input's quadrature-point table is a
/// linear function of. Entries are entity-major with the input's component count per entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoefficientLayout {
    /// One entry per `(cell, quadrature point)`: the stored table itself, in its own order.
    QuadraturePoint,
    /// One entry per mesh cell: a piecewise-constant coefficient.
    Cell,
    /// One entry per mesh vertex, interpolated inside each cell with the P1 barycentric basis
    /// at the quadrature points: a nodal design field, independent of the plan's own element
    /// order.
    Vertex,
}

impl CoefficientLayout {
    /// Length of a design vector under this layout for `component_count` components per entry.
    pub fn dimension(
        self,
        mesh: &Mesh,
        element: &PreparedElement,
        component_count: usize,
    ) -> Result<usize, FinitumError> {
        self.dimension_at(mesh, element.quadrature().len(), component_count)
    }

    /// As [`Self::dimension`] for a realization integrating every cell with a `point_count`-
    /// point quadrature table that is not owned by one [`PreparedElement`] -- the system path
    /// (`SystemOperator::quadrature`), whose fields share one table (SC-W1 system-path parity).
    pub fn dimension_at(
        self,
        mesh: &Mesh,
        point_count: usize,
        component_count: usize,
    ) -> Result<usize, FinitumError> {
        let entities = match self {
            CoefficientLayout::QuadraturePoint => mesh.cells().len() * point_count,
            CoefficientLayout::Cell => mesh.cells().len(),
            CoefficientLayout::Vertex => mesh.vertices().len(),
        };
        entities.checked_mul(component_count).ok_or_else(|| {
            FinitumError::InvalidRealization("coefficient design extent overflows usize".into())
        })
    }

    /// The design entries (entity index, weight) whose linear combination is the coefficient's
    /// value at quadrature point `point` of `cell`.
    fn weights(
        self,
        mesh: &Mesh,
        element: &PreparedElement,
        cell: usize,
        point: usize,
    ) -> Result<Vec<(usize, f64)>, FinitumError> {
        self.weights_at(mesh, element.quadrature(), cell, point)
    }

    /// As [`Self::weights`] over an explicit cell quadrature table (the system path's shared
    /// table).
    pub(crate) fn weights_at(
        self,
        mesh: &Mesh,
        quadrature: &[QuadraturePoint],
        cell: usize,
        point: usize,
    ) -> Result<Vec<(usize, f64)>, FinitumError> {
        match self {
            CoefficientLayout::QuadraturePoint => Ok(vec![(cell * quadrature.len() + point, 1.0)]),
            CoefficientLayout::Cell => Ok(vec![(cell, 1.0)]),
            CoefficientLayout::Vertex => {
                let (values, _) =
                    simplex_basis(mesh.dimension(), 1, &quadrature[point].coordinates)?;
                Ok(mesh.cells()[cell]
                    .vertices
                    .iter()
                    .zip(values)
                    .map(|(vertex, weight)| (vertex.0, weight))
                    .collect())
            }
        }
    }
}

/// SV1-C3: one stored external input of a cell integral, viewed as a distributed coefficient
/// over a caller-owned design space (`layout`). The residual's derivatives with respect to
/// that design vector are [`RealizationPlan::coefficient_jacobian_vector_product`] and its
/// exact transpose [`RealizationPlan::coefficient_vector_jacobian_product`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedCoefficient {
    pub integral_index: usize,
    pub input: TensorInputId,
    pub layout: CoefficientLayout,
}

/// One basis-backed active input evaluated at a quadrature point.
#[derive(Clone, Debug, PartialEq)]
pub struct PointActiveInput {
    pub input: TensorInputId,
    pub derivative: DerivativeEvaluation,
    pub values: Vec<f64>,
}

/// One bound input of the evaluation point's instance on a composed plan (W8 lane F-MI,
/// `sinbad/ARCHITECTURE.md` §6): the producer output's value at this point on a same-mesh
/// `bind`, keyed by the consumer's local input-field symbol and the consumer slot it closes.
/// In a direction evaluation, `values` is the output's directional derivative along the
/// direction (its tangent to the producer's fields), so a consumer law such as `sigma(T)`
/// chains it exactly as it chains an active input.
#[derive(Clone, Debug, PartialEq)]
pub struct PointBoundInput {
    pub symbol: SymbolId,
    pub slot: String,
    pub values: Vec<f64>,
}

/// Runtime point data supplied to a state-dependent external input.
#[derive(Clone, Debug, PartialEq)]
pub struct PointEvaluation {
    pub time: f64,
    pub cell: CellId,
    pub coordinates: Vec<f64>,
    pub active: Vec<PointActiveInput>,
    /// The instance's bound inputs at this point (empty on every one-instance realization).
    pub bound: Vec<PointBoundInput>,
}

impl PointEvaluation {
    /// The bound input closing the consumer's input field `symbol`, by identity (W8 F-MI).
    pub fn bound_values(&self, symbol: SymbolId) -> Option<&[f64]> {
        self.bound
            .iter()
            .find(|input| input.symbol == symbol)
            .map(|input| input.values.as_slice())
    }

    /// The bound input closing the consumer `slot` (`<instance>/input/<name>`).
    pub fn bound_slot_values(&self, slot: &str) -> Option<&[f64]> {
        self.bound
            .iter()
            .find(|input| input.slot == slot)
            .map(|input| input.values.as_slice())
    }

    /// Return the first active binding with the requested evaluation kind.
    pub fn values(&self, derivative: DerivativeEvaluation) -> Option<&[f64]> {
        self.active
            .iter()
            .find(|input| input.derivative == derivative)
            .map(|input| input.values.as_slice())
    }

    /// Return the active binding of one QFunction input by identity -- unambiguous where several
    /// fields of a multi-field system share an evaluation kind (Batch P).
    pub fn input_values(&self, input: TensorInputId) -> Option<&[f64]> {
        self.active
            .iter()
            .find(|candidate| candidate.input == input)
            .map(|candidate| candidate.values.as_slice())
    }
}

type PointValueEvaluator =
    dyn Fn(&PointEvaluation) -> Result<Vec<f64>, InputEvaluationError> + Send + Sync;
type PointDirectionEvaluator = dyn Fn(&PointEvaluation, &PointEvaluation) -> Result<Vec<f64>, InputEvaluationError>
    + Send
    + Sync;

/// Locate a callback's failure where Finitum evaluated it: the evaluation's cell, physical
/// point and time replace whatever the callback set (W8 lane F2).
pub(crate) fn locate_failure(
    failure: InputEvaluationError,
    evaluation: &PointEvaluation,
) -> FinitumError {
    failure
        .at(
            Some(evaluation.cell),
            &evaluation.coordinates,
            Some(evaluation.time),
        )
        .into()
}

/// The refusal code of a bound property source ([`FieldSource::Kernel`] / [`FieldSource::Table`])
/// whose value cannot be evaluated at a runtime point: a table axis point outside its range, a
/// kernel execution failure, an evaluation point without the active input's value. Carried in
/// an [`InputEvaluationError`] whose origin is the expression path
/// `<model>[.<equation>][<integral>].<symbol>`, never as a non-finite placeholder (W8 lane F2).
pub const REALIZATION_PROPERTY_UNAVAILABLE: &str = "REALIZATION_PROPERTY_UNAVAILABLE";

/// The typed failure of one of Finitum's own bound property closures at a runtime point.
pub(crate) fn property_unavailable(origin: &str, error: FinitumError) -> InputEvaluationError {
    InputEvaluationError::new(
        REALIZATION_PROPERTY_UNAVAILABLE,
        crate::InputOrigin::ExpressionPath(origin.to_owned()),
        error.to_string(),
    )
}

/// The typed failure of a bound property whose tangent for `input` is not available at a
/// runtime point (the build-time check admitted the tangent; the kernel declined here).
pub(crate) fn tangent_unavailable(origin: &str, input: &str) -> InputEvaluationError {
    InputEvaluationError::new(
        "REALIZATION_TANGENT_UNAVAILABLE",
        crate::InputOrigin::ExpressionPath(origin.to_owned()),
        format!(
            "the bound property has no tangent for its state-dependent input {input:?} at this \
             point"
        ),
    )
}

/// The active input's values at `point` by identity, or the typed failure of a point that does
/// not carry them.
pub(crate) fn active_values<'a>(
    origin: &str,
    point: &'a PointEvaluation,
    input: TensorInputId,
) -> Result<&'a [f64], InputEvaluationError> {
    point.input_values(input).ok_or_else(|| {
        InputEvaluationError::new(
            REALIZATION_PROPERTY_UNAVAILABLE,
            crate::InputOrigin::ExpressionPath(origin.to_owned()),
            format!("the evaluation point carries no value for active input {input:?}"),
        )
    })
}

/// As [`active_values`], by evaluation kind (the single-model path's binding).
fn active_values_of<'a>(
    origin: &str,
    point: &'a PointEvaluation,
    derivative: DerivativeEvaluation,
) -> Result<&'a [f64], InputEvaluationError> {
    point.values(derivative).ok_or_else(|| {
        InputEvaluationError::new(
            REALIZATION_PROPERTY_UNAVAILABLE,
            crate::InputOrigin::ExpressionPath(origin.to_owned()),
            format!("the evaluation point carries no {derivative:?} active input"),
        )
    })
}

/// A property table's axis point from the named inputs, or the typed failure of an axis that
/// is neither a coordinate/time name nor the bound active input.
pub(crate) fn table_axis_point(
    origin: &str,
    table: &scientia::PropertyTable,
    named: &BTreeMap<String, f64>,
) -> Result<Vec<f64>, InputEvaluationError> {
    table
        .axes
        .iter()
        .map(|axis| {
            named.get(&axis.name).copied().ok_or_else(|| {
                InputEvaluationError::new(
                    REALIZATION_PROPERTY_UNAVAILABLE,
                    crate::InputOrigin::ExpressionPath(origin.to_owned()),
                    format!(
                        "property table axis {:?} is neither a coordinate/time name nor the \
                         bound active input",
                        axis.name
                    ),
                )
            })
        })
        .collect()
}

/// A non-basis QFunction input evaluated from the current point state.
///
/// The callbacks are supplied by the consuming product or material implementation. Finitum
/// only binds their values and directional derivatives into generated parameter-JVP kernels.
/// `identity` must change whenever either callback's semantics change. The direction callback is
/// trusted to return the exact directional derivative of the value callback; products should
/// retain centered-difference acceptance checks for every authored dynamic binding.
///
/// W8 lane F2: the callbacks are fallible ([`Self::try_new`]). A callback that meets a typed
/// refusal returns an [`InputEvaluationError`] (its own code, origin and message); Finitum
/// locates it (cell, point, time) and propagates it as [`FinitumError::InputEvaluation`] out of
/// every action -- residual, JVP, VJP, `linearize`, `assemble`, partial assembly, the
/// agreement checks and every Methodus trait entry point (as `NumericError::Evaluation`). The
/// first failure in cell / quadrature-point / input order wins; no action returns a non-finite
/// value in its place. [`Self::new`] is the infallible form: its closures cannot refuse.
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
    /// The infallible form: `value` and `direction` cannot refuse. A thin wrapper over
    /// [`Self::try_new`] (deleted by slice F3 once every consumer has migrated).
    pub fn new(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        identity: impl Into<String>,
        value: impl Fn(&PointEvaluation) -> Vec<f64> + Send + Sync + 'static,
        direction: impl Fn(&PointEvaluation, &PointEvaluation) -> Vec<f64> + Send + Sync + 'static,
    ) -> Result<Self, FinitumError> {
        Self::try_new(
            integral_index,
            input,
            component_count,
            identity,
            move |point| Ok(value(point)),
            move |point, direction_point| Ok(direction(point, direction_point)),
        )
    }

    /// The fallible form (W8 lane F2): `value` and `direction` return their own typed
    /// [`InputEvaluationError`] instead of a value when they cannot evaluate; see the type
    /// documentation for how it propagates.
    pub fn try_new(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        identity: impl Into<String>,
        value: impl Fn(&PointEvaluation) -> Result<Vec<f64>, InputEvaluationError>
        + Send
        + Sync
        + 'static,
        direction: impl Fn(&PointEvaluation, &PointEvaluation) -> Result<Vec<f64>, InputEvaluationError>
        + Send
        + Sync
        + 'static,
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

    /// The value callback at `evaluation`, a failure located there.
    pub(crate) fn evaluate_value(
        &self,
        evaluation: &PointEvaluation,
    ) -> Result<Vec<f64>, FinitumError> {
        (self.value)(evaluation).map_err(|failure| locate_failure(failure, evaluation))
    }

    /// The direction callback at `evaluation` along `direction`, a failure located there.
    pub(crate) fn evaluate_direction(
        &self,
        evaluation: &PointEvaluation,
        direction: &PointEvaluation,
    ) -> Result<Vec<f64>, FinitumError> {
        (self.direction)(evaluation, direction)
            .map_err(|failure| locate_failure(failure, evaluation))
    }
}

#[derive(Clone, Debug)]
enum ExternalBinding {
    Stored(ExternalInput),
    Dynamic(DynamicExternalInput),
}

/// A stored-table sampler's failure, located where the table was being sampled and re-labelled
/// [`crate::InputOrigin::Table`] (W8 lane F2): the refusal happened at bind time, not inside an
/// operator action, and the table is never built with a non-finite placeholder.
pub(crate) fn table_failure(
    failure: InputEvaluationError,
    cell: Option<CellId>,
    point: &[f64],
    time: Option<f64>,
) -> FinitumError {
    failure.at(cell, point, time).into_table().into()
}

fn validate_sampling_time(time: f64) -> Result<(), FinitumError> {
    if time.is_finite() {
        Ok(())
    } else {
        Err(FinitumError::InvalidRealization(
            "table sampling time must be finite".into(),
        ))
    }
}

/// The deterministic cell / quadrature-point / component sampling every stored cell table
/// shares: `sample` sees each cell's physical quadrature points in the element's own order,
/// and its first `FinitumError` (a located `InputEvaluation` included) ends the sampling.
fn sample_cell_table(
    mesh: &Mesh,
    element: &PreparedElement,
    component_count: usize,
    label: &str,
    mut sample: impl FnMut(CellId, &[f64]) -> Result<Vec<f64>, FinitumError>,
) -> Result<Vec<f64>, FinitumError> {
    if mesh.dimension() != element.dimension() {
        return Err(FinitumError::InvalidRealization(format!(
            "mesh dimension {} differs from element dimension {}",
            mesh.dimension(),
            element.dimension()
        )));
    }
    let mut values =
        Vec::with_capacity(mesh.cells().len() * element.quadrature().len() * component_count);
    for cell_index in 0..mesh.cells().len() {
        let cell = CellGeometry::new(mesh, CellId(cell_index))?;
        for point in element.quadrature() {
            let physical = cell.physical_point(&point.coordinates);
            let sampled = sample(CellId(cell_index), &physical)?;
            if sampled.len() != component_count {
                return Err(FinitumError::InvalidRealization(format!(
                    "{label} sampler returned {} components, expected {component_count}",
                    sampled.len()
                )));
            }
            values.extend(sampled);
        }
    }
    Ok(values)
}

/// The facet counterpart of [`sample_cell_table`]: one centroid point per facet of
/// `facet_ids`, in that order (`FacetGeometry`'s single-point rule).
fn sample_facet_table(
    mesh: &Mesh,
    facets: &FacetTopology,
    facet_ids: &[FacetId],
    component_count: usize,
    mut sample: impl FnMut(FacetId, &FacetGeometry) -> Result<Vec<f64>, FinitumError>,
) -> Result<Vec<f64>, FinitumError> {
    let mut values = Vec::with_capacity(facet_ids.len() * component_count);
    for &facet_id in facet_ids {
        let facet = facets.facets().get(facet_id.0).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("facet {} does not exist", facet_id.0))
        })?;
        if !facet.is_exterior() {
            return Err(FinitumError::UnsupportedRealization(format!(
                "facet {} is not exterior; exterior facet integrals refuse interior facets",
                facet_id.0
            )));
        }
        let geometry = FacetGeometry::compute(mesh, facet.minus())?;
        let sampled = sample(facet_id, &geometry)?;
        if sampled.len() != component_count {
            return Err(FinitumError::InvalidRealization(format!(
                "facet external input sampler returned {} components, expected {component_count}",
                sampled.len()
            )));
        }
        values.extend(sampled);
    }
    Ok(values)
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

    /// Sample and own input values in deterministic cell/quadrature/component order. The
    /// infallible form: `sample` cannot refuse (a thin wrapper over the same sampling as
    /// [`Self::try_sampled`]; deleted by slice F3).
    pub fn sampled(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        element: &PreparedElement,
        mut sample: impl FnMut(CellId, &[f64]) -> Vec<f64>,
    ) -> Result<Self, FinitumError> {
        let values = sample_cell_table(
            mesh,
            element,
            component_count,
            "external input",
            |cell, point| Ok(sample(cell, point)),
        )?;
        Self::new(integral_index, input, component_count, values)
    }

    /// W8 lane F2: the fallible form of [`Self::sampled`] for a steady table. A refusal
    /// `sample` returns is located at its cell and physical point (no time), re-labelled
    /// [`crate::InputOrigin::Table`] and returned as [`FinitumError::InputEvaluation`] -- the
    /// table is never built with a non-finite placeholder. The first refusal in cell, then
    /// quadrature-point order wins.
    pub fn try_sampled(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        element: &PreparedElement,
        mut sample: impl FnMut(CellId, &[f64]) -> Result<Vec<f64>, InputEvaluationError>,
    ) -> Result<Self, FinitumError> {
        let values = sample_cell_table(
            mesh,
            element,
            component_count,
            "external input",
            |cell, point| {
                sample(cell, point)
                    .map_err(|failure| table_failure(failure, Some(cell), point, None))
            },
        )?;
        Self::new(integral_index, input, component_count, values)
    }

    /// W8 lane F2: as [`Self::try_sampled`] at evaluation time `time`, which the sampler
    /// receives and a refusal records -- the transient path's tables (sources, coefficients)
    /// sampled at the step's time instead of frozen at `t = 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_sampled_at(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        element: &PreparedElement,
        time: f64,
        mut sample: impl FnMut(CellId, &[f64], f64) -> Result<Vec<f64>, InputEvaluationError>,
    ) -> Result<Self, FinitumError> {
        validate_sampling_time(time)?;
        let values = sample_cell_table(
            mesh,
            element,
            component_count,
            "external input",
            |cell, point| {
                sample(cell, point, time)
                    .map_err(|failure| table_failure(failure, Some(cell), point, Some(time)))
            },
        )?;
        Self::new(integral_index, input, component_count, values)
    }

    /// SV1-C3: own the quadrature-point values a caller-owned distributed coefficient `design`
    /// induces under `layout` (see [`CoefficientLayout`]), in the same deterministic
    /// cell/quadrature/component order as [`Self::sampled`]. The map from `design` to the
    /// stored table is linear; [`RealizationPlan::coefficient_jacobian_vector_product`] and
    /// [`RealizationPlan::coefficient_vector_jacobian_product`] differentiate through exactly
    /// this map, so a plan bound with this input and a design vector `p` is the realization
    /// `R(u; p)` those products are the derivatives of.
    pub fn from_coefficient(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        element: &PreparedElement,
        layout: CoefficientLayout,
        design: &[f64],
    ) -> Result<Self, FinitumError> {
        Self::from_coefficient_at(
            integral_index,
            input,
            component_count,
            mesh,
            element.quadrature(),
            layout,
            design,
        )
    }

    /// As [`Self::from_coefficient`] over an explicit cell quadrature table -- the table a
    /// system realization shares across its fields (`SystemOperator::quadrature`), so the same
    /// design vector parameterizes a `SystemOperator` integral input exactly as it does a
    /// `RealizationPlan` one (`SystemOperator::coefficient_jacobian_vector_product` and its
    /// transpose differentiate through this map).
    #[allow(clippy::too_many_arguments)]
    pub fn from_coefficient_at(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        quadrature: &[QuadraturePoint],
        layout: CoefficientLayout,
        design: &[f64],
    ) -> Result<Self, FinitumError> {
        if component_count == 0 {
            return Err(FinitumError::InvalidRealization(
                "coefficient component count must be non-zero".into(),
            ));
        }
        let expected = layout.dimension_at(mesh, quadrature.len(), component_count)?;
        if design.len() != expected {
            return Err(FinitumError::InvalidRealization(format!(
                "coefficient design vector has length {}, layout {layout:?} expects {expected}",
                design.len()
            )));
        }
        validate_finite("coefficient design vector", design)?;
        let point_count = quadrature.len();
        let mut values = Vec::with_capacity(mesh.cells().len() * point_count * component_count);
        for cell in 0..mesh.cells().len() {
            for point in 0..point_count {
                let weights = layout.weights_at(mesh, quadrature, cell, point)?;
                for component in 0..component_count {
                    values.push(
                        weights
                            .iter()
                            .map(|(entity, weight)| {
                                weight * design[entity * component_count + component]
                            })
                            .sum(),
                    );
                }
            }
        }
        Self::new(integral_index, input, component_count, values)
    }

    /// GX-C4: sample and own input values in the deterministic order of `facet_ids` (one
    /// centroid quadrature point per facet, matching `FacetGeometry`'s single-point rule).
    /// `facet_ids` must be exactly the facet list a `SemanticMeasure::ExteriorFacet { region }`
    /// integral resolves through its `RegionMap`/`RegionTags` binding, in the same order the
    /// realization itself will use -- passing a different order silently mismatches values to
    /// facets, since this stored array carries no facet identity of its own.
    pub fn sampled_on_facets(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        facets: &FacetTopology,
        facet_ids: &[FacetId],
        mut sample: impl FnMut(FacetId, &[f64]) -> Vec<f64>,
    ) -> Result<Self, FinitumError> {
        let values = sample_facet_table(
            mesh,
            facets,
            facet_ids,
            component_count,
            |facet, geometry| Ok(sample(facet, &geometry.physical_centroid)),
        )?;
        Self::new(integral_index, input, component_count, values)
    }

    /// W8 lane F2: the fallible form of [`Self::sampled_on_facets`] (steady); a refusal is
    /// located at the facet's owning cell and centroid, re-labelled
    /// [`crate::InputOrigin::Table`], and ends the sampling.
    pub fn try_sampled_on_facets(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        facets: &FacetTopology,
        facet_ids: &[FacetId],
        mut sample: impl FnMut(FacetId, &[f64]) -> Result<Vec<f64>, InputEvaluationError>,
    ) -> Result<Self, FinitumError> {
        let values = sample_facet_table(
            mesh,
            facets,
            facet_ids,
            component_count,
            |facet, geometry| {
                sample(facet, &geometry.physical_centroid).map_err(|failure| {
                    table_failure(
                        failure,
                        Some(geometry.cell),
                        &geometry.physical_centroid,
                        None,
                    )
                })
            },
        )?;
        Self::new(integral_index, input, component_count, values)
    }

    /// W8 lane F2: as [`Self::try_sampled_on_facets`] at evaluation time `time`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_sampled_on_facets_at(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        facets: &FacetTopology,
        facet_ids: &[FacetId],
        time: f64,
        mut sample: impl FnMut(FacetId, &[f64], f64) -> Result<Vec<f64>, InputEvaluationError>,
    ) -> Result<Self, FinitumError> {
        validate_sampling_time(time)?;
        let values = sample_facet_table(
            mesh,
            facets,
            facet_ids,
            component_count,
            |facet, geometry| {
                sample(facet, &geometry.physical_centroid, time).map_err(|failure| {
                    table_failure(
                        failure,
                        Some(geometry.cell),
                        &geometry.physical_centroid,
                        Some(time),
                    )
                })
            },
        )?;
        Self::new(integral_index, input, component_count, values)
    }

    /// Components per quadrature point of this table.
    pub fn component_count(&self) -> usize {
        self.component_count
    }

    /// The stored table in cell/quadrature/component order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub(crate) fn point_values(&self, cell: usize, point: usize, point_count: usize) -> &[f64] {
        let start = (cell * point_count + point) * self.component_count;
        &self.values[start..start + self.component_count]
    }

    fn facet_point_values(&self, facet_position: usize) -> &[f64] {
        let start = facet_position * self.component_count;
        &self.values[start..start + self.component_count]
    }
}

/// GX-C3: builds `ExternalInput`/`DynamicExternalInput` bindings for every non-basis QFunction
/// input of `factorization`'s **cell** integrals (`SemanticMeasure::Cell`) from a caller-supplied
/// `(SymbolId, FieldSource)` table, replacing hand-written per-input closures such as the ones
/// `fc7_runtime_state_rate_and_property_chain_rule_match_finite_differences` writes by hand.
/// GX-C4 facet integrals build their external inputs separately, since they sample at facet
/// points rather than cell quadrature points (see `ExternalInput::sampled_on_facets`).
///
/// A `Constant`/`Sampled`/`Fallible`/`Table` source, or a `Kernel` source whose declared inputs
/// are all resolvable as coordinates/time (names `"x"`/`"y"`/`"z"`/`"t"`/`"time"`), is sampled
/// once per cell quadrature point into a stored `ExternalInput` at `t = 0` -- coordinates only,
/// so it cannot vary with the runtime evaluation time or the active field; W8 lane F2's
/// [`external_inputs_from_at`] samples the same tables at a caller-chosen time instead. A
/// `Nodal` source is refused (it has no coordinate sampler). A `Fallible` source's refusal is
/// located, re-labelled [`crate::InputOrigin::Table`] and returned typed.
///
/// A `Kernel` or `Table` source whose declared inputs (or axes) include the name of exactly one
/// active, basis-sourced field of the same integral becomes a `DynamicExternalInput`: its value
/// closure evaluates the kernel/table at the point's coordinates and the active field's current
/// value; its direction closure evaluates the exact tangent/slope with respect to that one input
/// and multiplies by the supplied direction (the chain rule for every other declared input is
/// zero, since only coordinates/time and the one active field may appear). A source referencing
/// more than one active field by name is refused (ambiguous chain-rule combination, out of
/// bounded scope). A `Kernel` whose `DerivativeContract` supplied no tangent for that input, or a
/// `Table` with `TableDerivativePolicy::Unavailable`, is refused at build time
/// (`FinitumError::RealizationTangentUnavailable`) rather than silently trusted with a zero or
/// approximate direction.
pub fn external_inputs_from(
    factorization: &OperatorFactorization,
    model: &scientia::SemanticModel,
    mesh: &Mesh,
    element: &PreparedElement,
    sources: &[(SymbolId, FieldSource)],
) -> Result<(Vec<ExternalInput>, Vec<DynamicExternalInput>), FinitumError> {
    external_inputs_from_at(factorization, model, mesh, element, sources, 0.0)
}

/// W8 lane F2: [`external_inputs_from`] with the state-independent sources sampled at `time`
/// (their `"t"`/`"time"` inputs, table axes, and the `Fallible` closure's time argument), so a
/// transient realization's stored tables carry the step's data rather than `t = 0`'s. The
/// state-dependent (`DynamicExternalInput`) bindings are unaffected: they read the runtime
/// `PointEvaluation::time`.
pub fn external_inputs_from_at(
    factorization: &OperatorFactorization,
    model: &scientia::SemanticModel,
    mesh: &Mesh,
    element: &PreparedElement,
    sources: &[(SymbolId, FieldSource)],
    time: f64,
) -> Result<(Vec<ExternalInput>, Vec<DynamicExternalInput>), FinitumError> {
    validate_sampling_time(time)?;
    let sources_by_symbol = sources
        .iter()
        .map(|(symbol, source)| (*symbol, source))
        .collect::<BTreeMap<_, _>>();
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &factorization.integrals {
        if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
            continue;
        }
        let active_inputs = integral
            .primal
            .inputs
            .iter()
            .filter(|input| {
                input.source == InputSourceRequirement::Basis
                    && input.role == TensorInputRole::Active
            })
            .collect::<Vec<_>>();
        let active_names = active_inputs
            .iter()
            .map(|input| model.symbols[input.binding.symbol.index()].name.clone())
            .collect::<BTreeSet<_>>();
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let symbol = input.binding.symbol;
            let source = *sources_by_symbol.get(&symbol).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "external_inputs_from has no FieldSource for symbol {symbol:?}"
                ))
            })?;
            let components = component_count(&input.shape)?;
            let origin = format!(
                "{}[{}].{}",
                model.name,
                integral.integral_index,
                model.symbols[symbol.index()].name
            );
            match source {
                FieldSource::Kernel { kernel, executable } => {
                    let state_names = kernel
                        .inputs
                        .iter()
                        .filter(|slot| active_names.contains(&slot.name))
                        .map(|slot| slot.name.clone())
                        .collect::<Vec<_>>();
                    if state_names.is_empty() {
                        let values = sample_cell_table(
                            mesh,
                            element,
                            components,
                            "external input",
                            |_, point| {
                                let named = named_coordinate_inputs(point, time);
                                evaluate_kernel_value(kernel, executable, &named)
                                    .map(|value| vec![value])
                            },
                        )?;
                        stored.push(ExternalInput::new(
                            integral.integral_index,
                            input.id,
                            components,
                            values,
                        )?);
                        continue;
                    }
                    if state_names.len() > 1 || components != 1 {
                        return Err(FinitumError::UnsupportedRealization(
                            "state-dependent kernel external inputs support one scalar active \
                             field reference only"
                                .into(),
                        ));
                    }
                    let active_name = state_names[0].clone();
                    let active_input = *active_inputs
                        .iter()
                        .find(|candidate| {
                            model.symbols[candidate.binding.symbol.index()].name == active_name
                        })
                        .expect("active_name was derived from active_inputs");
                    if !kernel
                        .tangents
                        .iter()
                        .any(|tangent| tangent.input == active_name)
                    {
                        return Err(FinitumError::RealizationTangentUnavailable(format!(
                            "property kernel {:?} declares no tangent for its state-dependent \
                             input {active_name:?}",
                            kernel.identity
                        )));
                    }
                    let active_derivative = active_input.binding.evaluation.derivative;
                    let identity = format!("finitum.field-source-kernel/1:{}", kernel.identity);
                    let value_kernel = kernel.clone();
                    let value_executable = executable.clone();
                    let value_active_name = active_name.clone();
                    let direction_kernel = kernel.clone();
                    let direction_executable = executable.clone();
                    let direction_active_name = active_name.clone();
                    let value_origin = origin.clone();
                    let direction_origin = origin.clone();
                    dynamic.push(DynamicExternalInput::try_new(
                        integral.integral_index,
                        input.id,
                        1,
                        identity,
                        move |point: &PointEvaluation| {
                            let mut named = named_coordinate_inputs(&point.coordinates, point.time);
                            let active_value =
                                active_values_of(&value_origin, point, active_derivative)?[0];
                            named.insert(value_active_name.clone(), active_value);
                            evaluate_kernel_value(&value_kernel, &value_executable, &named)
                                .map(|value| vec![value])
                                .map_err(|error| property_unavailable(&value_origin, error))
                        },
                        move |point: &PointEvaluation, direction: &PointEvaluation| {
                            let mut named = named_coordinate_inputs(&point.coordinates, point.time);
                            let active_value =
                                active_values_of(&direction_origin, point, active_derivative)?[0];
                            named.insert(direction_active_name.clone(), active_value);
                            let partial = evaluate_kernel_partial(
                                &direction_kernel,
                                &direction_executable,
                                &named,
                                &direction_active_name,
                            )
                            .map_err(|error| property_unavailable(&direction_origin, error))?
                            .ok_or_else(|| {
                                tangent_unavailable(&direction_origin, &direction_active_name)
                            })?;
                            let active_direction =
                                active_values_of(&direction_origin, direction, active_derivative)?
                                    [0];
                            Ok(vec![partial * active_direction])
                        },
                    )?);
                }
                FieldSource::Table(table) => {
                    let state_names = table
                        .axes
                        .iter()
                        .filter(|axis| active_names.contains(&axis.name))
                        .map(|axis| axis.name.clone())
                        .collect::<Vec<_>>();
                    if state_names.is_empty() {
                        let values = sample_cell_table(
                            mesh,
                            element,
                            components,
                            "external input",
                            |_, point| {
                                let named = named_coordinate_inputs(point, time);
                                let axis_point = table
                                    .axes
                                    .iter()
                                    .map(|axis| {
                                        named.get(&axis.name).copied().ok_or_else(|| {
                                            FinitumError::InvalidRealization(format!(
                                                "property table axis {:?} is not a coordinate/time name",
                                                axis.name
                                            ))
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                evaluate_table_value(table, &axis_point).map(|value| vec![value])
                            },
                        )?;
                        stored.push(ExternalInput::new(
                            integral.integral_index,
                            input.id,
                            components,
                            values,
                        )?);
                        continue;
                    }
                    if state_names.len() > 1 || components != 1 {
                        return Err(FinitumError::UnsupportedRealization(
                            "state-dependent table external inputs support one scalar active \
                             field axis only"
                                .into(),
                        ));
                    }
                    if table.derivative_policy == scientia::TableDerivativePolicy::Unavailable {
                        return Err(FinitumError::RealizationTangentUnavailable(format!(
                            "property table axis {:?} has no derivative policy",
                            state_names[0]
                        )));
                    }
                    let active_name = state_names[0].clone();
                    let axis_index = table
                        .axes
                        .iter()
                        .position(|axis| axis.name == active_name)
                        .expect("active_name was derived from table.axes");
                    let active_input = *active_inputs
                        .iter()
                        .find(|candidate| {
                            model.symbols[candidate.binding.symbol.index()].name == active_name
                        })
                        .expect("active_name was derived from active_inputs");
                    let active_derivative = active_input.binding.evaluation.derivative;
                    let identity = format!(
                        "finitum.field-source-table/1:{}",
                        field_source_table_digest(table)
                    );
                    let value_table = table.clone();
                    let value_active_name = active_name.clone();
                    let slope_table = table.clone();
                    let slope_active_name = active_name.clone();
                    let value_origin = origin.clone();
                    let slope_origin = origin.clone();
                    dynamic.push(DynamicExternalInput::try_new(
                        integral.integral_index,
                        input.id,
                        1,
                        identity,
                        move |point: &PointEvaluation| {
                            let mut named = named_coordinate_inputs(&point.coordinates, point.time);
                            let active_value =
                                active_values_of(&value_origin, point, active_derivative)?[0];
                            named.insert(value_active_name.clone(), active_value);
                            let axis_point = table_axis_point(&value_origin, &value_table, &named)?;
                            evaluate_table_value(&value_table, &axis_point)
                                .map(|value| vec![value])
                                .map_err(|error| property_unavailable(&value_origin, error))
                        },
                        move |point: &PointEvaluation, direction: &PointEvaluation| {
                            let mut named = named_coordinate_inputs(&point.coordinates, point.time);
                            let active_value =
                                active_values_of(&slope_origin, point, active_derivative)?[0];
                            named.insert(slope_active_name.clone(), active_value);
                            let axis_point = table_axis_point(&slope_origin, &slope_table, &named)?;
                            let slope = evaluate_table_slope(&slope_table, &axis_point, axis_index)
                                .map_err(|error| property_unavailable(&slope_origin, error))?;
                            let active_direction =
                                active_values_of(&slope_origin, direction, active_derivative)?[0];
                            Ok(vec![slope * active_direction])
                        },
                    )?);
                }
                FieldSource::Nodal(_) => {
                    return Err(FinitumError::UnsupportedRealization(
                        "a Nodal field source has no coordinate sampler; supply its \
                         already-projected quadrature-point values directly as an \
                         ExternalInput instead of through external_inputs_from"
                            .into(),
                    ));
                }
                FieldSource::Constant(values) => {
                    stored.push(ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        components,
                        mesh,
                        element,
                        |_, _| values.clone(),
                    )?);
                }
                FieldSource::Sampled(sampler) => {
                    stored.push(ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        components,
                        mesh,
                        element,
                        |_, point| sampler(point),
                    )?);
                }
                FieldSource::Fallible(sampler) => {
                    stored.push(ExternalInput::try_sampled_at(
                        integral.integral_index,
                        input.id,
                        components,
                        mesh,
                        element,
                        time,
                        |_, point, time| sampler(point, time),
                    )?);
                }
            }
        }
    }
    Ok((stored, dynamic))
}

fn field_source_table_digest(table: &scientia::PropertyTable) -> Digest {
    FieldSource::Table(table.clone()).identity()
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
        let values = sample_cell_table(
            mesh,
            element,
            component_count,
            "external sensitivity",
            |cell, point| Ok(sample(cell, point)),
        )?;
        Self::new(integral_index, input, component_count, values)
    }

    /// W8 lane F2: the fallible form of [`Self::sampled`]; a refusal is located at its cell
    /// and baseline point, re-labelled [`crate::InputOrigin::Table`], and ends the sampling
    /// (design derivatives are steady, so there is no `_at` form).
    pub fn try_sampled(
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        mesh: &Mesh,
        element: &PreparedElement,
        mut sample: impl FnMut(CellId, &[f64]) -> Result<Vec<f64>, InputEvaluationError>,
    ) -> Result<Self, FinitumError> {
        let values = sample_cell_table(
            mesh,
            element,
            component_count,
            "external sensitivity",
            |cell, point| {
                sample(cell, point)
                    .map_err(|failure| table_failure(failure, Some(cell), point, None))
            },
        )?;
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
pub(crate) struct BoundBundle {
    pub(crate) bundle: StructuredPointKernelBundle,
    pub(crate) executable: ExecutableModule,
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
    /// GX-C4: exterior facets in scope for each `SemanticMeasure::ExteriorFacet { region }`
    /// integral, resolved by the caller through a `RegionMap`/`RegionTags` binding before
    /// construction. Empty for a realization with no facet integrals.
    facet_regions: BTreeMap<scientia::RegionId, Vec<FacetId>>,
    /// Precomputed geometry for every facet referenced by `facet_regions`, keyed by [`FacetId`].
    facet_geometries: BTreeMap<FacetId, FacetGeometry>,
    /// Lazily proven, then cached, symmetry declaration of the matrix-free action. The proof
    /// assembles the action once per realization; every later `symmetry()` query is a read.
    symmetry_proof: OnceLock<OperatorSymmetry>,
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

pub const REALIZATION_CAPABILITY_SCHEMA: &str = "finitum-realization-capability/1";

/// One admitted Scientia element requirement, as reported by [`RealizationPlan::capability`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapabilityElement {
    pub symbol: SymbolId,
    pub topological_dimension: u8,
    pub family: ElementFamilyRequirement,
    pub polynomial_order: u8,
    pub value_shape: ValueShape,
}

/// The kinds of essential-constraint rows a [`RealizationPlan`] admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    /// A fixed row with no weighted dependencies.
    Fixed,
    /// A row expressed as an affine combination of other degrees of freedom.
    AffineDependency,
}

/// A global operator representation this plan can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationKind {
    MatrixFree,
    Assembled,
    ElementAssembly,
    PartialAssembly,
}

/// A derivative product this plan can execute through its public API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeProduct {
    Primal,
    Jvp,
    /// [`RealizationPlan::vector_jacobian_product`]: the exact transpose action of the state
    /// JVP at rate direction zero, executing the bound Malleus VJP kernels (GX-F7).
    Vjp,
    /// [`RealizationPlan::coefficient_jacobian_vector_product`]: the residual's directional
    /// derivative with respect to a stored distributed coefficient (SV1-C3).
    CoefficientJvp,
    /// [`RealizationPlan::coefficient_vector_jacobian_product`]: its exact transpose,
    /// accumulated into the caller-owned coefficient space (SV1-C3).
    CoefficientVjp,
}

/// Source-artifact provenance for one [`RealizationCapability`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RealizationReceipt {
    pub source_requirements_digest: Digest,
    pub source_factorization_digest: Digest,
    pub source_kernels_digest: Digest,
    pub realization_digest: Digest,
}

/// Versioned, serializable description of what one [`RealizationPlan`] admitted (SV2-A2): the
/// topology dimension, admitted element family/order/value shapes, measures realized,
/// constraint kinds present, representation kinds and derivative products this plan exposes,
/// and its declared symmetry, bound to a canonical digest. Purely descriptive: it carries no
/// admissibility policy of its own.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RealizationCapability {
    pub schema: String,
    pub topology_dimension: usize,
    pub elements: Vec<CapabilityElement>,
    pub measures: Vec<SemanticMeasure>,
    pub constraint_kinds: Vec<ConstraintKind>,
    pub representation_kinds: Vec<RepresentationKind>,
    pub derivative_products: Vec<DerivativeProduct>,
    pub symmetry: OperatorSymmetry,
    pub receipt: RealizationReceipt,
    pub digest: Digest,
}

#[derive(Serialize)]
struct CapabilityDigestPayload<'a> {
    schema: &'static str,
    topology_dimension: usize,
    elements: &'a [CapabilityElement],
    measures: &'a [SemanticMeasure],
    constraint_kinds: &'a [ConstraintKind],
    representation_kinds: &'a [RepresentationKind],
    derivative_products: &'a [DerivativeProduct],
    symmetry: OperatorSymmetry,
    receipt: &'a RealizationReceipt,
}

/// Seals one [`RealizationCapability`] under the canonical `finitum-realization-capability/1`
/// digest -- the one place that payload is spelled, shared by `RealizationPlan::capability`
/// and the system path's `ReducedSystemOperator::capability`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_capability(
    topology_dimension: usize,
    elements: Vec<CapabilityElement>,
    measures: Vec<SemanticMeasure>,
    constraint_kinds: Vec<ConstraintKind>,
    representation_kinds: Vec<RepresentationKind>,
    derivative_products: Vec<DerivativeProduct>,
    symmetry: OperatorSymmetry,
    receipt: RealizationReceipt,
) -> RealizationCapability {
    let payload = CapabilityDigestPayload {
        schema: REALIZATION_CAPABILITY_SCHEMA,
        topology_dimension,
        elements: &elements,
        measures: &measures,
        constraint_kinds: &constraint_kinds,
        representation_kinds: &representation_kinds,
        derivative_products: &derivative_products,
        symmetry,
        receipt: &receipt,
    };
    let digest =
        Digest::blake3(&serde_json::to_vec(&payload).expect("capability payload is serializable"));
    RealizationCapability {
        schema: REALIZATION_CAPABILITY_SCHEMA.into(),
        topology_dimension,
        elements,
        measures,
        constraint_kinds,
        representation_kinds,
        derivative_products,
        symmetry,
        receipt,
        digest,
    }
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
        Self::new_with_facets(
            requirements,
            factorization,
            kernels,
            mesh,
            element,
            dofs,
            constraints,
            external_inputs,
            dynamic_external_inputs,
            BTreeMap::new(),
        )
    }

    /// GX-C4: as [`Self::new_stateful`], additionally admitting `SemanticMeasure::ExteriorFacet`
    /// integrals. `facet_regions` maps each such integral's `region` to the concrete exterior
    /// [`FacetId`]s it integrates over -- resolved by the caller through a `RegionMap`/
    /// `RegionTags` binding (mirroring [`crate::essential_constraints_from`]) before calling
    /// this constructor, since a bare [`Mesh`] carries no region tags of its own. A
    /// `SemanticMeasure::ExteriorFacet { region }` integral whose `region` has no entry (or an
    /// empty entry) here is refused (`FinitumError::RealizationRegionUnmapped`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_facets(
        requirements: FormRequirements,
        factorization: OperatorFactorization,
        kernels: StructuredOperatorKernels,
        mesh: Mesh,
        element: PreparedElement,
        dofs: DofMap,
        constraints: ConstraintSet,
        external_inputs: Vec<ExternalInput>,
        dynamic_external_inputs: Vec<DynamicExternalInput>,
        facet_regions: BTreeMap<scientia::RegionId, Vec<FacetId>>,
    ) -> Result<Self, FinitumError> {
        validate_artifacts(&requirements, &factorization, &kernels)?;
        validate_discretization(
            &requirements,
            &factorization,
            &mesh,
            &element,
            &dofs,
            &constraints,
            &facet_regions,
        )?;
        let external = validate_external_inputs(
            &factorization,
            &mesh,
            &element,
            external_inputs,
            dynamic_external_inputs,
            &facet_regions,
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
        let facet_geometries = if facet_regions.is_empty() {
            BTreeMap::new()
        } else {
            let facet_topology = FacetTopology::from_mesh(&mesh)?;
            let mut referenced = BTreeSet::new();
            for facet_ids in facet_regions.values() {
                referenced.extend(facet_ids.iter().copied());
            }
            let mut geometries = BTreeMap::new();
            for facet_id in referenced {
                let facet = facet_topology.facets().get(facet_id.0).ok_or_else(|| {
                    FinitumError::InvalidRealization(format!("facet {} does not exist", facet_id.0))
                })?;
                if !facet.is_exterior() {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "facet {} is not exterior; GX-C4 refuses interior facet integrals",
                        facet_id.0
                    )));
                }
                geometries.insert(facet_id, FacetGeometry::compute(&mesh, facet.minus())?);
            }
            geometries
        };
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
                facet_regions,
                facet_geometries,
                symmetry_proof: OnceLock::new(),
            }),
        })
    }

    pub fn dimension(&self) -> usize {
        self.data.dofs.dof_count()
    }

    pub fn mesh(&self) -> &Mesh {
        &self.data.mesh
    }

    /// The prepared reference element (basis tables and the cell quadrature rule) this plan
    /// integrates with -- what [`crate::FieldSampler::from_realization_plan`] and
    /// [`crate::QuadratureView::of_realization_plan`] read (W8 lane F1; additive accessor).
    pub fn element(&self) -> &PreparedElement {
        &self.data.element
    }

    /// The global degree-of-freedom map this plan gathers and scatters through: local DOF `k`
    /// of a cell's restriction is basis function `k` of [`Self::element`] (node-major with the
    /// field's component stride for a vector block). Read by
    /// [`crate::FieldSampler::from_realization_plan`] (W8 lane F1; additive accessor).
    pub fn dofs(&self) -> &DofMap {
        &self.data.dofs
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

    /// Describes what this plan admitted: topology dimension, element family/order/value
    /// shapes, measures realized, constraint kinds, the representation and derivative products
    /// this plan exposes, and its declared symmetry (SV2-A2). This is purely descriptive: it
    /// reports what the plan already does, adding no policy of its own.
    pub fn capability(&self) -> RealizationCapability {
        let elements = self
            .data
            .requirements
            .elements
            .iter()
            .map(|element| CapabilityElement {
                symbol: element.symbol,
                topological_dimension: element.topological_dimension,
                family: element.family,
                polynomial_order: element.polynomial_order,
                value_shape: element.value_shape.clone(),
            })
            .collect::<Vec<_>>();
        let measures = self
            .data
            .factorization
            .integrals
            .iter()
            .map(|integral| integral.measure.clone())
            .collect::<Vec<_>>();
        let mut constraint_kinds = BTreeSet::new();
        for constraint in self.data.constraints.constraints() {
            if constraint.dependencies.is_empty() {
                constraint_kinds.insert(ConstraintKind::Fixed);
            } else {
                constraint_kinds.insert(ConstraintKind::AffineDependency);
            }
        }
        let constraint_kinds = constraint_kinds.into_iter().collect::<Vec<_>>();
        let representation_kinds = vec![
            RepresentationKind::MatrixFree,
            RepresentationKind::Assembled,
            RepresentationKind::ElementAssembly,
            RepresentationKind::PartialAssembly,
        ];
        // GX-F7: `vector_jacobian_product` executes the bound Malleus VJP kernels, but it
        // refuses affine dependency constraints (their transpose lands with SV1-C2), so `Vjp`
        // is reported only when this plan has none.
        let mut derivative_products = vec![DerivativeProduct::Primal, DerivativeProduct::Jvp];
        if !self.data.constraints.has_affine_dependencies() {
            derivative_products.push(DerivativeProduct::Vjp);
        }
        // SV1-C3: a distributed coefficient is a stored external input of a cell integral; the
        // coefficient products exist exactly when such an input exists (the JVP unconditionally,
        // the VJP under the same affine-dependency rule as `Vjp`).
        let has_stored_cell_input = self.data.factorization.integrals.iter().any(|integral| {
            matches!(integral.measure, SemanticMeasure::Cell { .. })
                && integral.primal.inputs.iter().any(|input| {
                    matches!(
                        self.data.external.get(&(integral.integral_index, input.id)),
                        Some(ExternalBinding::Stored(_))
                    )
                })
        });
        if has_stored_cell_input {
            derivative_products.push(DerivativeProduct::CoefficientJvp);
            if !self.data.constraints.has_affine_dependencies() {
                derivative_products.push(DerivativeProduct::CoefficientVjp);
            }
        }
        let symmetry = self.matrix_free().symmetry();
        let receipt = RealizationReceipt {
            source_requirements_digest: self.data.requirements.artifact_digest.clone(),
            source_factorization_digest: self.data.factorization.artifact_digest.clone(),
            source_kernels_digest: self.data.kernels_digest.clone(),
            realization_digest: self.data.digest.clone(),
        };
        build_capability(
            self.data.mesh.dimension(),
            elements,
            measures,
            constraint_kinds,
            representation_kinds,
            derivative_products,
            symmetry,
            receipt,
        )
    }

    /// Checks every `BoundaryPartitionRequirement` this plan's source requirements declared
    /// against `tags`/`region_map`, without requiring a [`crate::TaggedMesh`] wrapper around
    /// this plan's own mesh. Additive: existing callers that never call this are unaffected.
    pub fn validate_boundary_partition(
        &self,
        tags: &RegionTags,
        region_map: &RegionMap,
    ) -> Result<Vec<PartitionReport>, FinitumError> {
        self.data
            .requirements
            .boundary_partitions
            .iter()
            .map(|requirement| partition_report_for(&self.data.mesh, tags, requirement, region_map))
            .collect()
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
        self.apply_facets(
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
        self.apply_facets(
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

    /// Evaluate the exact transpose action of [`Self::jacobian_vector_product`] at the same
    /// linearization point, with rate direction held at zero (GX-F7).
    ///
    /// This executes the bound Malleus VJP kernel for every integral output: the test-side
    /// scatter's transpose becomes a forward gather of `adjoint` through the test basis, the VJP
    /// kernel replaces the JVP kernel, and the result is scattered back through the transpose of
    /// the trial-side gather (the same basis-adjoint machinery the forward JVP's test-side
    /// scatter uses). A dynamic external input's own forward `direction` closure is reused,
    /// probed with unit basis directions, to invert its chain-rule contribution exactly under
    /// the same trusted-linearity contract the forward JVP relies on; a stored external input
    /// contributes nothing further, matching its always-zero forward direction. Constraint rows
    /// are transposed exactly like [`AssembledOperator::transpose`] would: a fixed row is its own
    /// transpose, so its contribution is added back in directly. Affine dependency constraints
    /// are refused with a typed error because their transpose is not yet implemented (SV1-C2).
    ///
    /// Cost: one VJP kernel execution per integral output per quadrature point, matching the
    /// JVP's per-point kernel cost; a dynamic external input adds a bounded number of additional
    /// point-local probes (proportional to that input's own and the active inputs' component
    /// counts, never to the global degree-of-freedom count). No global assembly and no
    /// dimension-many operator applications are performed.
    pub fn vector_jacobian_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.vector_jacobian_product_shifted(time, state, state_rate, adjoint, 0.0, output)
    }

    /// SV1-C1: the exact transpose of the *shifted* Jacobian action
    /// `x -> jacobian_vector_product(x, rate_shift * x)`, i.e. of `dR/du + rate_shift * dR/du_t`
    /// -- the linearization an implicit time integrator (BDF: `rate_shift = alpha / dt`) or a
    /// steady Newton step (`rate_shift = 0`) solves with. `rate_shift = 0` is exactly
    /// [`Self::vector_jacobian_product`].
    ///
    /// The rate half of the transpose reuses the state half's machinery: a `TimeDerivative`
    /// active input is gathered through the same value basis as a `Value` input (see
    /// `evaluate_basis_input`), so its cotangent is scattered through that same basis scaled
    /// by `rate_shift`; a dynamic external input's chain rule through a time-derivative active
    /// input is probed exactly as its state chain rule is, and scaled the same way.
    pub fn vector_jacobian_product_shifted(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_time_action(time, state, state_rate, output)?;
        self.validate_action(adjoint, output)?;
        if !rate_shift.is_finite() {
            return Err(FinitumError::InvalidRealization(
                "rate shift must be finite".into(),
            ));
        }
        if self.data.constraints.has_affine_dependencies() {
            return Err(FinitumError::UnsupportedRealization(
                "vector_jacobian_product refuses affine dependency constraints; their exact \
                 transpose is not yet implemented (SV1-C2)"
                    .into(),
            ));
        }
        let physical_state = self.data.constraints.expand(state)?;
        let physical_rate = self.data.constraints.expand_homogeneous(state_rate)?;
        // Only unconstrained (non-target) rows of `adjoint` flow into the cell-transpose action;
        // every fixed constraint row contributes only through the constraint's own transpose
        // below, so it is masked to zero here (mirrors the row/column split of `A =
        // R*(E^T A_cell E) + C` used to derive this transpose).
        let mut restricted_adjoint = adjoint.to_vec();
        for constraint in self.data.constraints.constraints() {
            restricted_adjoint[constraint.target.0] = 0.0;
        }
        let physical_adjoint = self
            .data
            .constraints
            .expand_homogeneous(&restricted_adjoint)?;
        let mut physical_output = vec![0.0; self.dimension()];
        self.apply_cells_transpose(
            time,
            &physical_state,
            &physical_rate,
            &physical_adjoint,
            rate_shift,
            &mut physical_output,
        )?;
        self.apply_facets_transpose(
            time,
            &physical_state,
            &physical_rate,
            &physical_adjoint,
            rate_shift,
            &mut physical_output,
        )?;
        output.copy_from_slice(&self.data.constraints.restrict_transpose(&physical_output)?);
        // A fixed constraint row `C[t,:] = e_t^T` is its own transpose, so `C^T adjoint`
        // contributes `adjoint[t]` back at row `t`; `restrict_transpose` always drops row `t`
        // (no affine dependencies exist here), so this is additive, not an overwrite.
        for constraint in self.data.constraints.constraints() {
            output[constraint.target.0] += adjoint[constraint.target.0];
        }
        validate_finite("stateful VJP", output)
    }

    /// SV1-C1: the Jacobian `J = dR/du + rate_shift * dR/du_t` of this realization at one
    /// linearization point, as a Methodus operator whose primal action is
    /// [`Self::jacobian_vector_product`] with rate direction `rate_shift * x` and whose
    /// transpose action ([`TransposableOperator`]) is
    /// [`Self::vector_jacobian_product_shifted`] -- the operator pair an adjoint solve at a
    /// converged state (or inside an implicit time step) consumes.
    pub fn linearize(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        rate_shift: f64,
    ) -> Result<LinearizedOperator, FinitumError> {
        let probe = vec![0.0; self.dimension()];
        self.validate_time_action(time, state, state_rate, &probe)?;
        if !rate_shift.is_finite() {
            return Err(FinitumError::InvalidRealization(
                "rate shift must be finite".into(),
            ));
        }
        Ok(LinearizedOperator {
            plan: self.clone(),
            time,
            state: state.to_vec(),
            state_rate: state_rate.to_vec(),
            rate_shift,
        })
    }

    /// Length of the design vector `coefficient` ranges over (SV1-C3).
    pub fn coefficient_dimension(
        &self,
        coefficient: &DistributedCoefficient,
    ) -> Result<usize, FinitumError> {
        let (_, stored) = self.coefficient_binding(coefficient)?;
        coefficient
            .layout
            .dimension(&self.data.mesh, &self.data.element, stored.component_count)
    }

    /// SV1-C3: `dR/dp * direction` at a fixed state -- the residual's directional derivative
    /// with respect to the design vector of `coefficient`, executing each integral output's
    /// bound frozen-input (parameter) Malleus JVP kernel with the direction routed to the
    /// coefficient's input only. Constraint rows carry zero coefficient derivative, because
    /// essential values are frozen inputs of the realization.
    pub fn coefficient_jacobian_vector_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        coefficient: &DistributedCoefficient,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_time_action(time, state, state_rate, output)?;
        let (integral, stored) = self.coefficient_binding(coefficient)?;
        let components = stored.component_count;
        let expected = self.coefficient_dimension(coefficient)?;
        if direction.len() != expected {
            return Err(FinitumError::InvalidRealization(format!(
                "coefficient direction has length {}, expected {expected}",
                direction.len()
            )));
        }
        validate_finite("coefficient direction", direction)?;
        let physical_state = self.data.constraints.expand(state)?;
        let physical_rate = self.data.constraints.expand_homogeneous(state_rate)?;
        let mut physical_output = vec![0.0; self.dimension()];
        for cell_index in 0..self.data.dofs.restrictions().len() {
            let restriction = &self.data.dofs.restrictions()[cell_index];
            let geometry = &self.data.geometries[cell_index];
            let local_state = restriction
                .dofs
                .iter()
                .map(|dof| physical_state[dof.0])
                .collect::<Vec<_>>();
            let local_rate = restriction
                .dofs
                .iter()
                .map(|dof| physical_rate[dof.0])
                .collect::<Vec<_>>();
            let mut local_output = vec![0.0; restriction.dofs.len()];
            for (point_index, point) in self.data.element.quadrature().iter().enumerate() {
                let scale = point.weight * geometry.determinant;
                let weights = coefficient.layout.weights(
                    &self.data.mesh,
                    &self.data.element,
                    cell_index,
                    point_index,
                )?;
                let point_direction = (0..components)
                    .map(|component| {
                        weights
                            .iter()
                            .map(|(entity, weight)| {
                                weight * direction[entity * components + component]
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let (inputs, _) = self.point_inputs(
                    integral,
                    cell_index,
                    point_index,
                    geometry,
                    time,
                    &local_state,
                    &local_rate,
                )?;
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = &self.data.bundles[&(integral.integral_index, output_index)];
                    let Some(point_output) = execute_parameter_jvp_values(
                        bound,
                        &inputs,
                        coefficient.input,
                        &point_direction,
                    )?
                    else {
                        continue;
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
            for (local, dof) in restriction.dofs.iter().enumerate() {
                physical_output[dof.0] += local_output[local];
            }
        }
        output.copy_from_slice(&self.data.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.data.constraints.constraints() {
            output[constraint.target.0] = 0.0;
        }
        validate_finite("coefficient JVP", output)
    }

    /// SV1-C3: `(dR/dp)^T * adjoint` -- the exact transpose of
    /// [`Self::coefficient_jacobian_vector_product`], accumulating each quadrature point's
    /// parameter cotangent (the bound parameter kernel's exact point-local Jacobian contracted
    /// against the test-side adjoint seed, as GX-F7's VJP already computes for dynamic inputs)
    /// into the caller-owned design space through the transpose of the layout's interpolation.
    /// Affine dependency constraints refuse exactly as [`Self::vector_jacobian_product`] does.
    pub fn coefficient_vector_jacobian_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        coefficient: &DistributedCoefficient,
        adjoint: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let probe = vec![0.0; self.dimension()];
        self.validate_time_action(time, state, state_rate, &probe)?;
        self.validate_action(adjoint, &probe)?;
        if self.data.constraints.has_affine_dependencies() {
            return Err(FinitumError::UnsupportedRealization(
                "coefficient_vector_jacobian_product refuses affine dependency constraints; \
                 their exact transpose is not yet implemented (SV1-C2)"
                    .into(),
            ));
        }
        let (integral, stored) = self.coefficient_binding(coefficient)?;
        let components = stored.component_count;
        let expected = self.coefficient_dimension(coefficient)?;
        if output.len() != expected {
            return Err(FinitumError::InvalidRealization(format!(
                "coefficient VJP output has length {}, expected {expected}",
                output.len()
            )));
        }
        let physical_state = self.data.constraints.expand(state)?;
        let physical_rate = self.data.constraints.expand_homogeneous(state_rate)?;
        // Constraint rows carry no coefficient derivative (see the JVP), so their adjoint
        // entries are masked out before the homogeneous expansion, mirroring
        // `vector_jacobian_product_shifted`.
        let mut restricted_adjoint = adjoint.to_vec();
        for constraint in self.data.constraints.constraints() {
            restricted_adjoint[constraint.target.0] = 0.0;
        }
        let physical_adjoint = self
            .data
            .constraints
            .expand_homogeneous(&restricted_adjoint)?;
        output.fill(0.0);
        for cell_index in 0..self.data.dofs.restrictions().len() {
            let restriction = &self.data.dofs.restrictions()[cell_index];
            let geometry = &self.data.geometries[cell_index];
            let local_state = restriction
                .dofs
                .iter()
                .map(|dof| physical_state[dof.0])
                .collect::<Vec<_>>();
            let local_rate = restriction
                .dofs
                .iter()
                .map(|dof| physical_rate[dof.0])
                .collect::<Vec<_>>();
            let local_adjoint = restriction
                .dofs
                .iter()
                .map(|dof| physical_adjoint[dof.0])
                .collect::<Vec<_>>();
            for (point_index, point) in self.data.element.quadrature().iter().enumerate() {
                let scale = point.weight * geometry.determinant;
                let weights = coefficient.layout.weights(
                    &self.data.mesh,
                    &self.data.element,
                    cell_index,
                    point_index,
                )?;
                let (inputs, _) = self.point_inputs(
                    integral,
                    cell_index,
                    point_index,
                    geometry,
                    time,
                    &local_state,
                    &local_rate,
                )?;
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = &self.data.bundles[&(integral.integral_index, output_index)];
                    let output_components = component_count(&qoutput.shape)?;
                    let seed = gather_test_adjoint(
                        &self.data.element,
                        geometry,
                        point_index,
                        &qoutput.binding.evaluation.derivative,
                        output_components,
                        &local_adjoint,
                    )?;
                    let cotangents = point_parameter_cotangents(bound, &inputs, &seed)?;
                    let Some(cotangent) = cotangents.get(&coefficient.input) else {
                        continue;
                    };
                    if cotangent.len() != components {
                        return Err(FinitumError::InvalidRealization(format!(
                            "coefficient cotangent has {} components, expected {components}",
                            cotangent.len()
                        )));
                    }
                    for (entity, weight) in &weights {
                        for component in 0..components {
                            output[entity * components + component] +=
                                scale * weight * cotangent[component];
                        }
                    }
                }
            }
        }
        validate_finite("coefficient VJP", output)
    }

    /// Resolves `coefficient` to its cell integral and stored binding, refusing a dynamic
    /// binding (its values are not a design vector), a facet integral (facet tables are
    /// sampled per facet, not per cell quadrature point), or an unknown key.
    fn coefficient_binding(
        &self,
        coefficient: &DistributedCoefficient,
    ) -> Result<(&IntegralOperatorFactorization, &ExternalInput), FinitumError> {
        let key = (coefficient.integral_index, coefficient.input);
        let integral = self
            .data
            .factorization
            .integrals
            .iter()
            .find(|integral| integral.integral_index == coefficient.integral_index)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "distributed coefficient names absent integral {}",
                    coefficient.integral_index
                ))
            })?;
        if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
            return Err(FinitumError::UnsupportedRealization(format!(
                "distributed coefficients are realized on cell integrals only; integral {} has \
                 measure {:?}",
                coefficient.integral_index, integral.measure
            )));
        }
        match self.data.external.get(&key) {
            Some(ExternalBinding::Stored(stored)) => Ok((integral, stored)),
            Some(ExternalBinding::Dynamic(_)) => {
                Err(FinitumError::UnsupportedRealization(format!(
                    "external input {key:?} is a dynamic binding, not a stored distributed \
                     coefficient"
                )))
            }
            None => Err(FinitumError::MissingExternalInput {
                integral: key.0,
                input: key.1,
            }),
        }
    }

    /// Return the zero-active-state JVP realization for this globally linear FC6 plan.
    pub fn matrix_free(&self) -> MatrixFreeOperator {
        MatrixFreeOperator { plan: self.clone() }
    }

    /// Establish, once per realization, whether the matrix-free action is self-adjoint, and
    /// record the answer so every later [`MatrixFreeOperator::symmetry`] query on this
    /// realization (and its clones) reports it.
    ///
    /// The proof assembles the action through the same generated JVP kernels `apply` executes
    /// and compares every entry against its transpose with the relative `tolerance`
    /// (`|a_ij - a_ji| <= tolerance * max(|a_ij|, |a_ji|, 1)`), so floating-point reassociation
    /// across cells does not masquerade as nonsymmetry. It is explicit and bounded: affine
    /// dependency constraints short-circuit to `Nonsymmetric`; realizations larger than
    /// [`SYMMETRY_PROOF_DIMENSION_CAP`] are refused rather than probed; a non-finite or
    /// negative tolerance is refused. Repeated calls return the recorded proof without
    /// reassembling, regardless of the tolerance they pass.
    pub fn prove_symmetry(&self, tolerance: f64) -> Result<OperatorSymmetry, FinitumError> {
        if self.data.constraints.has_affine_dependencies() {
            return Ok(OperatorSymmetry::Nonsymmetric);
        }
        if let Some(proof) = self.data.symmetry_proof.get() {
            return Ok(*proof);
        }
        if !(tolerance.is_finite() && tolerance >= 0.0) {
            return Err(FinitumError::UnsupportedRealization(format!(
                "symmetry proof tolerance must be finite and nonnegative, got {tolerance}"
            )));
        }
        if self.dimension() > SYMMETRY_PROOF_DIMENSION_CAP {
            return Err(FinitumError::UnsupportedRealization(format!(
                "symmetry proof by assembly is refused above {SYMMETRY_PROOF_DIMENSION_CAP} \
                 degrees of freedom (realization has {})",
                self.dimension()
            )));
        }
        let assembled = self.assemble()?;
        let proof = if csr_is_symmetric_within(&assembled.matrix, tolerance) {
            OperatorSymmetry::Symmetric
        } else {
            OperatorSymmetry::Nonsymmetric
        };
        Ok(*self.data.symmetry_proof.get_or_init(|| proof))
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
        if !self.data.facet_regions.is_empty() {
            return Err(FinitumError::UnsupportedRealization(
                "element assembly does not yet cover exterior facet integrals (GX-C4)".into(),
            ));
        }
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
        if !self.data.facet_regions.is_empty() {
            return Err(FinitumError::UnsupportedRealization(
                "partial assembly does not yet cover exterior facet integrals (GX-C4)".into(),
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
                            columns.push(execute_jvp_values(bound, &inputs, &directions)?);
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

    pub(crate) fn apply_direction(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
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
        self.apply_facets(
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
        self.apply_facets(0.0, state, &zero, None, None, output, Action::Primal)?;
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
        if !self.data.facet_regions.is_empty() {
            return Err(FinitumError::UnsupportedRealization(
                "geometry sensitivity does not yet cover exterior facet integrals (GX-C4)".into(),
            ));
        }
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
                if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                    continue;
                }
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
                        execute_jvp_values(bound, &inputs, &point_directions)?;
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
                if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                    continue;
                }
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

    /// GX-C4: exterior facet analog of [`Self::apply_cells`]. Every `SemanticMeasure::
    /// ExteriorFacet { region }` integral is resolved through `self.data.facet_regions` (already
    /// validated non-empty at construction) into concrete facets, each contributing through its
    /// single incident cell's restriction.
    #[allow(clippy::too_many_arguments)]
    fn apply_facets(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: Option<&[f64]>,
        rate_direction: Option<&[f64]>,
        output: &mut [f64],
        action: Action,
    ) -> Result<(), FinitumError> {
        for integral in &self.data.factorization.integrals {
            let region = match &integral.measure {
                SemanticMeasure::ExteriorFacet { region } => *region,
                _ => continue,
            };
            let facet_ids = self
                .data
                .facet_regions
                .get(&region)
                .expect("validated non-empty by validate_discretization");
            for (facet_position, &facet_id) in facet_ids.iter().enumerate() {
                self.apply_facet(
                    integral,
                    facet_position,
                    facet_id,
                    time,
                    state,
                    state_rate,
                    state_direction,
                    rate_direction,
                    output,
                    action,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_facet(
        &self,
        integral: &IntegralOperatorFactorization,
        facet_position: usize,
        facet_id: FacetId,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: Option<&[f64]>,
        rate_direction: Option<&[f64]>,
        output: &mut [f64],
        action: Action,
    ) -> Result<(), FinitumError> {
        let geometry = self
            .data
            .facet_geometries
            .get(&facet_id)
            .expect("facet geometry was precomputed at construction");
        let restriction = &self.data.dofs.restrictions()[geometry.cell.0];
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
        let basis_values = p1_trace_basis_values(&geometry.reference_centroid);
        let scale = geometry.scale(self.data.mesh.dimension());
        let mut local_output = vec![0.0; restriction.dofs.len()];
        for (output_index, _qoutput) in integral.primal.outputs.iter().enumerate() {
            let bound = &self.data.bundles[&(integral.integral_index, output_index)];
            let point_output = match action {
                Action::Primal => self.execute_facet_primal(
                    bound,
                    integral,
                    facet_position,
                    geometry,
                    &basis_values,
                    time,
                    &local_state,
                    &local_rate,
                )?,
                Action::Jvp => self.execute_facet_jvp(
                    bound,
                    integral,
                    facet_position,
                    geometry,
                    &basis_values,
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
            apply_trace_basis_adjoint(&basis_values, &point_output, scale, &mut local_output)?;
        }
        for (local, dof) in restriction.dofs.iter().enumerate() {
            output[dof.0] += local_output[local];
        }
        Ok(())
    }

    /// Facet analog of [`Self::point_inputs`]: basis-sourced inputs are gathered through the
    /// cell's trace basis at the facet centroid (`Value`/`TimeDerivative` only, enforced by
    /// `validate_facet_evaluation`); external inputs are looked up by `facet_position` in
    /// `self.data.facet_regions`' order.
    #[allow(clippy::too_many_arguments)]
    fn point_inputs_facet(
        &self,
        integral: &IntegralOperatorFactorization,
        facet_position: usize,
        geometry: &FacetGeometry,
        basis_values: &[f64],
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
            let values = evaluate_trace_basis_input(basis_values, input, dofs)?;
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
            cell: geometry.cell,
            coordinates: geometry.physical_centroid.clone(),
            active,
            bound: Vec::new(),
        };
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let binding = &self.data.external[&(integral.integral_index, input.id)];
            let values = match binding {
                ExternalBinding::Stored(stored) => {
                    stored.facet_point_values(facet_position).to_vec()
                }
                ExternalBinding::Dynamic(dynamic) => dynamic.evaluate_value(&evaluation)?,
            };
            validate_components(input, &values, "facet external input")?;
            inputs.insert(input.id, values);
        }
        Ok((inputs, evaluation))
    }

    /// Facet analog of [`Self::point_directions`].
    #[allow(clippy::too_many_arguments)]
    fn point_directions_facet(
        &self,
        integral: &IntegralOperatorFactorization,
        geometry: &FacetGeometry,
        basis_values: &[f64],
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
            let values = evaluate_trace_basis_input(basis_values, input, dofs)?;
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
            cell: geometry.cell,
            coordinates: evaluation.coordinates.clone(),
            active,
            bound: Vec::new(),
        };
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let binding = &self.data.external[&(integral.integral_index, input.id)];
            let values = match binding {
                ExternalBinding::Stored(stored) => vec![0.0; stored.component_count],
                ExternalBinding::Dynamic(dynamic) => {
                    dynamic.evaluate_direction(evaluation, &direction_evaluation)?
                }
            };
            validate_components(input, &values, "facet external input direction")?;
            directions.insert(input.id, values);
        }
        Ok(directions)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_facet_primal(
        &self,
        bound: &BoundBundle,
        integral: &IntegralOperatorFactorization,
        facet_position: usize,
        geometry: &FacetGeometry,
        basis_values: &[f64],
        time: f64,
        local_state: &[f64],
        local_rate: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let (inputs, _) = self.point_inputs_facet(
            integral,
            facet_position,
            geometry,
            basis_values,
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
    fn execute_facet_jvp(
        &self,
        bound: &BoundBundle,
        integral: &IntegralOperatorFactorization,
        facet_position: usize,
        geometry: &FacetGeometry,
        basis_values: &[f64],
        time: f64,
        local_state: &[f64],
        local_rate: &[f64],
        local_state_direction: &[f64],
        local_rate_direction: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let (inputs, evaluation) = self.point_inputs_facet(
            integral,
            facet_position,
            geometry,
            basis_values,
            time,
            local_state,
            local_rate,
        )?;
        let directions = self.point_directions_facet(
            integral,
            geometry,
            basis_values,
            time,
            local_state_direction,
            local_rate_direction,
            &evaluation,
        )?;
        execute_jvp_values(bound, &inputs, &directions)
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
        execute_primal_values(bound, &inputs)
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
        execute_jvp_values(bound, &inputs, &directions)
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
            bound: Vec::new(),
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
                ExternalBinding::Dynamic(dynamic) => dynamic.evaluate_value(&evaluation)?,
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
            bound: Vec::new(),
        };
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let binding = &self.data.external[&(integral.integral_index, input.id)];
            let values = match binding {
                ExternalBinding::Stored(stored) => vec![0.0; stored.component_count],
                ExternalBinding::Dynamic(dynamic) => {
                    dynamic.evaluate_direction(evaluation, &direction_evaluation)?
                }
            };
            validate_components(input, &values, "external input direction")?;
            directions.insert(input.id, values);
        }
        Ok(directions)
    }

    fn apply_cells_transpose(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        for cell_index in 0..self.data.dofs.restrictions().len() {
            self.apply_cell_transpose(
                cell_index, time, state, state_rate, adjoint, rate_shift, output,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_cell_transpose(
        &self,
        cell_index: usize,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
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
        let local_adjoint = restriction
            .dofs
            .iter()
            .map(|dof| adjoint[dof.0])
            .collect::<Vec<_>>();
        let mut local_output = vec![0.0; restriction.dofs.len()];
        for (point_index, point) in self.data.element.quadrature().iter().enumerate() {
            let scale = point.weight * geometry.determinant;
            for integral in &self.data.factorization.integrals {
                if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                    continue;
                }
                let (inputs, evaluation) = self.point_inputs(
                    integral,
                    cell_index,
                    point_index,
                    geometry,
                    time,
                    &local_state,
                    &local_rate,
                )?;
                let active_inputs = active_probe_inputs(integral, rate_shift);
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = &self.data.bundles[&(integral.integral_index, output_index)];
                    let output_components = component_count(&qoutput.shape)?;
                    let seed = gather_test_adjoint(
                        &self.data.element,
                        geometry,
                        point_index,
                        &qoutput.binding.evaluation.derivative,
                        output_components,
                        &local_adjoint,
                    )?;
                    let mut cotangents = execute_vjp_values(bound, &inputs, seed.clone())?;
                    if !bound.bundle.parameter.independent_operands.is_empty() {
                        let parameter_cotangents =
                            point_parameter_cotangents(bound, &inputs, &seed)?;
                        self.accumulate_parameter_cotangents(
                            integral,
                            point_index,
                            geometry,
                            scale,
                            rate_shift,
                            &evaluation,
                            &active_inputs,
                            &parameter_cotangents,
                            &mut cotangents,
                            &mut local_output,
                        )?;
                    }
                    for input in &integral.primal.inputs {
                        if input.source != InputSourceRequirement::Basis
                            || input.role != TensorInputRole::Active
                        {
                            continue;
                        }
                        let Some((derivative, factor)) = transpose_scatter_shape(input, rate_shift)
                        else {
                            continue;
                        };
                        let Some(cotangent) = cotangents.get(&input.id) else {
                            continue;
                        };
                        apply_basis_adjoint(
                            &self.data.element,
                            geometry,
                            point_index,
                            &derivative,
                            cotangent,
                            factor * scale,
                            &mut local_output,
                        )?;
                    }
                }
            }
        }
        for (local, dof) in restriction.dofs.iter().enumerate() {
            output[dof.0] += local_output[local];
        }
        Ok(())
    }

    /// GX-C4: exterior facet analog of [`Self::apply_cells_transpose`].
    fn apply_facets_transpose(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        for integral in &self.data.factorization.integrals {
            let region = match &integral.measure {
                SemanticMeasure::ExteriorFacet { region } => *region,
                _ => continue,
            };
            let facet_ids = self
                .data
                .facet_regions
                .get(&region)
                .expect("validated non-empty by validate_discretization");
            for (facet_position, &facet_id) in facet_ids.iter().enumerate() {
                self.apply_facet_transpose(
                    integral,
                    facet_position,
                    facet_id,
                    time,
                    state,
                    state_rate,
                    adjoint,
                    rate_shift,
                    output,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_facet_transpose(
        &self,
        integral: &IntegralOperatorFactorization,
        facet_position: usize,
        facet_id: FacetId,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let geometry = self
            .data
            .facet_geometries
            .get(&facet_id)
            .expect("facet geometry was precomputed at construction");
        let restriction = &self.data.dofs.restrictions()[geometry.cell.0];
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
        let local_adjoint = restriction
            .dofs
            .iter()
            .map(|dof| adjoint[dof.0])
            .collect::<Vec<_>>();
        let basis_values = p1_trace_basis_values(&geometry.reference_centroid);
        let scale = geometry.scale(self.data.mesh.dimension());
        let mut local_output = vec![0.0; restriction.dofs.len()];
        let (inputs, evaluation) = self.point_inputs_facet(
            integral,
            facet_position,
            geometry,
            &basis_values,
            time,
            &local_state,
            &local_rate,
        )?;
        let active_inputs = active_probe_inputs(integral, rate_shift);
        for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
            let bound = &self.data.bundles[&(integral.integral_index, output_index)];
            let output_components = component_count(&qoutput.shape)?;
            let seed = gather_trace_test_adjoint(&basis_values, output_components, &local_adjoint)?;
            let mut cotangents = execute_vjp_values(bound, &inputs, seed.clone())?;
            if !bound.bundle.parameter.independent_operands.is_empty() {
                let parameter_cotangents = point_parameter_cotangents(bound, &inputs, &seed)?;
                self.accumulate_parameter_cotangents_facet(
                    integral,
                    &basis_values,
                    scale,
                    rate_shift,
                    &evaluation,
                    &active_inputs,
                    &parameter_cotangents,
                    &mut cotangents,
                    &mut local_output,
                )?;
            }
            for input in &integral.primal.inputs {
                if input.source != InputSourceRequirement::Basis
                    || input.role != TensorInputRole::Active
                {
                    continue;
                }
                let Some((_, factor)) = transpose_scatter_shape(input, rate_shift) else {
                    continue;
                };
                let Some(cotangent) = cotangents.get(&input.id) else {
                    continue;
                };
                apply_trace_basis_adjoint(
                    &basis_values,
                    cotangent,
                    factor * scale,
                    &mut local_output,
                )?;
            }
        }
        for (local, dof) in restriction.dofs.iter().enumerate() {
            output[dof.0] += local_output[local];
        }
        Ok(())
    }

    /// Facet analog of [`Self::accumulate_parameter_cotangents`].
    #[allow(clippy::too_many_arguments)]
    fn accumulate_parameter_cotangents_facet(
        &self,
        integral: &IntegralOperatorFactorization,
        basis_values: &[f64],
        scale: f64,
        rate_shift: f64,
        evaluation: &PointEvaluation,
        active_inputs: &[&QFunctionInput],
        parameter_cotangents: &BTreeMap<TensorInputId, Vec<f64>>,
        cotangents: &mut BTreeMap<TensorInputId, Vec<f64>>,
        local_output: &mut [f64],
    ) -> Result<(), FinitumError> {
        for (input_id, grad) in parameter_cotangents {
            let input = integral
                .primal
                .inputs
                .iter()
                .find(|candidate| candidate.id == *input_id)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "parameter cotangent references undeclared input {input_id:?}"
                    ))
                })?;
            if input.source == InputSourceRequirement::Basis {
                let Some((_, factor)) = transpose_scatter_shape(input, rate_shift) else {
                    continue;
                };
                apply_trace_basis_adjoint(basis_values, grad, factor * scale, local_output)?;
                continue;
            }
            let binding = self
                .data
                .external
                .get(&(integral.integral_index, *input_id))
                .ok_or(FinitumError::MissingExternalInput {
                    integral: integral.integral_index,
                    input: *input_id,
                })?;
            let dynamic = match binding {
                ExternalBinding::Stored(_) => continue,
                ExternalBinding::Dynamic(dynamic) => dynamic,
            };
            for probe_input in active_inputs {
                let count = component_count(&probe_input.shape)?;
                for component in 0..count {
                    let probe = probe_direction_evaluation(
                        evaluation,
                        active_inputs,
                        probe_input.id,
                        component,
                    )?;
                    let response = dynamic.evaluate_direction(evaluation, &probe)?;
                    if response.len() != grad.len() {
                        return Err(FinitumError::InvalidRealization(format!(
                            "dynamic external input {input_id:?} direction returned {} \
                             components, expected {}",
                            response.len(),
                            grad.len()
                        )));
                    }
                    let contribution = response
                        .iter()
                        .zip(grad.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    let entry = cotangents
                        .entry(probe_input.id)
                        .or_insert_with(|| vec![0.0; count]);
                    entry[component] += contribution;
                }
            }
        }
        Ok(())
    }
}

/// Execute one integral output's bound VJP kernel: feed `seed` into the kernel's cotangent
/// seed operand and read back one cotangent per active input operand, summing contributions
/// when the same [`TensorInputId`] is read through more than one kernel operand access. A free
/// function (like [`execute_jvp_values`]) so the single-field and system realizations share it.
pub(crate) fn execute_vjp_values(
    bound: &BoundBundle,
    inputs: &BTreeMap<TensorInputId, Vec<f64>>,
    seed: Vec<f64>,
) -> Result<BTreeMap<TensorInputId, Vec<f64>>, FinitumError> {
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
    values.insert(bound.bundle.vjp.dependent_operands[0].derivative, seed);
    let executable = &bound.executable.kernels()[bound.bundle.vjp.kernel_index];
    let buffers = execute(executable, &values)?;
    let mut cotangents: BTreeMap<TensorInputId, Vec<f64>> = BTreeMap::new();
    for pair in &bound.bundle.vjp.independent_operands {
        let input = input_by_operand.get(&pair.primal).copied().ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "VJP operand {:?} has no QFunction input binding",
                pair.primal
            ))
        })?;
        let contribution = operand_values(executable, &buffers, pair.derivative)?;
        match cotangents.get_mut(&input) {
            Some(existing) => {
                if existing.len() != contribution.len() {
                    return Err(FinitumError::InvalidRealization(
                        "VJP cotangent has inconsistent extents across accesses".into(),
                    ));
                }
                for (total, value) in existing.iter_mut().zip(&contribution) {
                    *total += value;
                }
            }
            None => {
                cotangents.insert(input, contribution);
            }
        }
    }
    Ok(cotangents)
}

/// Extract the exact point-local Jacobian of the bound "parameter" (frozen-input) JVP
/// kernel with respect to every one of its independent (non-active) operands, by probing it
/// with unit direction columns -- exact because that kernel is Malleus's own linear tangent
/// map -- then contract each column against `seed` to produce one cotangent per parameter
/// input, summed across repeated accesses of the same [`TensorInputId`].
pub(crate) fn point_parameter_cotangents(
    bound: &BoundBundle,
    inputs: &BTreeMap<TensorInputId, Vec<f64>>,
    seed: &[f64],
) -> Result<BTreeMap<TensorInputId, Vec<f64>>, FinitumError> {
    let input_by_operand = bound
        .bundle
        .primal_inputs
        .iter()
        .map(|binding| (binding.operand, binding.input))
        .collect::<BTreeMap<_, _>>();
    let mut base_values = BTreeMap::new();
    for binding in &bound.bundle.primal_inputs {
        base_values.insert(binding.operand, inputs[&binding.input].clone());
    }
    let executable = &bound.executable.kernels()[bound.bundle.parameter.kernel_index];
    let output_operand = bound.bundle.parameter.dependent_operands[0].derivative;
    let mut grad: BTreeMap<TensorInputId, Vec<f64>> = BTreeMap::new();
    for pair in &bound.bundle.parameter.independent_operands {
        let component_count = base_values
            .get(&pair.primal)
            .map(|values| values.len())
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "parameter-JVP operand {:?} has no base value binding",
                    pair.primal
                ))
            })?;
        let input = input_by_operand.get(&pair.primal).copied().ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "parameter-JVP operand {:?} has no QFunction input binding",
                pair.primal
            ))
        })?;
        let mut cotangent = vec![0.0; component_count];
        for component in 0..component_count {
            let mut probe_values = base_values.clone();
            for other in &bound.bundle.parameter.independent_operands {
                let length = base_values[&other.primal].len();
                probe_values.insert(other.derivative, vec![0.0; length]);
            }
            let mut direction = vec![0.0; component_count];
            direction[component] = 1.0;
            probe_values.insert(pair.derivative, direction);
            let buffers = execute(executable, &probe_values)?;
            let column = operand_values(executable, &buffers, output_operand)?;
            if column.len() != seed.len() {
                return Err(FinitumError::InvalidRealization(
                    "parameter-JVP output extent does not match the VJP seed".into(),
                ));
            }
            cotangent[component] = column
                .iter()
                .zip(seed)
                .map(|(value, seed)| value * seed)
                .sum();
        }
        match grad.get_mut(&input) {
            Some(existing) => {
                if existing.len() != cotangent.len() {
                    return Err(FinitumError::InvalidRealization(
                        "parameter cotangent has inconsistent extents across accesses".into(),
                    ));
                }
                for (total, value) in existing.iter_mut().zip(&cotangent) {
                    *total += value;
                }
            }
            None => {
                grad.insert(input, cotangent);
            }
        }
    }
    Ok(grad)
}

impl RealizationPlan {
    /// Route each parameter cotangent to its exact destination: a passive basis-sourced input
    /// (state-dependent but not part of the active JVP/VJP contract) is scattered directly
    /// through its own trial-side basis, like an active input; a stored external input is a dead
    /// end, matching its always-zero forward direction; a dynamic external input's cotangent is
    /// pushed back into the active-gather cotangents by probing its own trusted forward
    /// `direction` closure with unit active-basis perturbations -- exact because that closure is
    /// contracted to return the exact (hence linear and homogeneous) directional derivative of
    /// its value closure.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_parameter_cotangents(
        &self,
        integral: &IntegralOperatorFactorization,
        point: usize,
        geometry: &CellGeometry,
        scale: f64,
        rate_shift: f64,
        evaluation: &PointEvaluation,
        active_inputs: &[&QFunctionInput],
        parameter_cotangents: &BTreeMap<TensorInputId, Vec<f64>>,
        cotangents: &mut BTreeMap<TensorInputId, Vec<f64>>,
        local_output: &mut [f64],
    ) -> Result<(), FinitumError> {
        for (input_id, grad) in parameter_cotangents {
            let input = integral
                .primal
                .inputs
                .iter()
                .find(|candidate| candidate.id == *input_id)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "parameter cotangent references undeclared input {input_id:?}"
                    ))
                })?;
            if input.source == InputSourceRequirement::Basis {
                let Some((derivative, factor)) = transpose_scatter_shape(input, rate_shift) else {
                    continue;
                };
                apply_basis_adjoint(
                    &self.data.element,
                    geometry,
                    point,
                    &derivative,
                    grad,
                    factor * scale,
                    local_output,
                )?;
                continue;
            }
            let binding = self
                .data
                .external
                .get(&(integral.integral_index, *input_id))
                .ok_or(FinitumError::MissingExternalInput {
                    integral: integral.integral_index,
                    input: *input_id,
                })?;
            let dynamic = match binding {
                ExternalBinding::Stored(_) => continue,
                ExternalBinding::Dynamic(dynamic) => dynamic,
            };
            for probe_input in active_inputs {
                let count = component_count(&probe_input.shape)?;
                for component in 0..count {
                    let probe = probe_direction_evaluation(
                        evaluation,
                        active_inputs,
                        probe_input.id,
                        component,
                    )?;
                    let response = dynamic.evaluate_direction(evaluation, &probe)?;
                    if response.len() != grad.len() {
                        return Err(FinitumError::InvalidRealization(format!(
                            "dynamic external input {input_id:?} direction returned {} \
                             components, expected {}",
                            response.len(),
                            grad.len()
                        )));
                    }
                    let contribution = response
                        .iter()
                        .zip(grad.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    let entry = cotangents
                        .entry(probe_input.id)
                        .or_insert_with(|| vec![0.0; count]);
                    entry[component] += contribution;
                }
            }
        }
        Ok(())
    }
}

/// Every active, basis-sourced input of one integral whose cotangent the shifted transpose
/// scatters: the exact probe basis for a dynamic external input's chain rule back into the
/// gather space. A time-derivative-kind active input participates only under a nonzero
/// `rate_shift` (at `rate_shift = 0`, [`RealizationPlan::vector_jacobian_product`]'s domain,
/// the rate direction is fixed at zero, so such inputs never contribute and are excluded).
pub(crate) fn active_probe_inputs(
    integral: &IntegralOperatorFactorization,
    rate_shift: f64,
) -> Vec<&QFunctionInput> {
    integral
        .primal
        .inputs
        .iter()
        .filter(|input| {
            input.source == InputSourceRequirement::Basis
                && input.role == TensorInputRole::Active
                && transpose_scatter_shape(input, rate_shift).is_some()
        })
        .collect()
}

/// How a basis-sourced input's cotangent is scattered by the shifted transpose: the basis
/// evaluation to scatter through and the scalar factor to apply. A `TimeDerivative` input is
/// gathered through the value basis of the rate vector (see `evaluate_basis_input`), so its
/// transpose scatters through the `Value` basis scaled by `rate_shift`, and is absent entirely
/// when `rate_shift == 0`. Every other evaluation scatters through its own basis with factor
/// one.
pub(crate) fn transpose_scatter_shape(
    input: &QFunctionInput,
    rate_shift: f64,
) -> Option<(DerivativeEvaluation, f64)> {
    match input.binding.evaluation.derivative {
        DerivativeEvaluation::TimeDerivative => {
            (rate_shift != 0.0).then_some((DerivativeEvaluation::Value, rate_shift))
        }
        other => Some((other, 1.0)),
    }
}

/// Build a synthetic direction [`PointEvaluation`] that is zero everywhere except a single unit
/// component of `hot_input`, for probing a [`DynamicExternalInput`]'s trusted `direction`
/// closure at the real (unperturbed) `evaluation`.
pub(crate) fn probe_direction_evaluation(
    evaluation: &PointEvaluation,
    active_inputs: &[&QFunctionInput],
    hot_input: TensorInputId,
    hot_component: usize,
) -> Result<PointEvaluation, FinitumError> {
    let mut active = Vec::with_capacity(active_inputs.len());
    for input in active_inputs {
        let count = component_count(&input.shape)?;
        let mut values = vec![0.0; count];
        if input.id == hot_input {
            if hot_component >= count {
                return Err(FinitumError::InvalidRealization(format!(
                    "probe component {hot_component} is outside input {:?} extent {count}",
                    input.id
                )));
            }
            values[hot_component] = 1.0;
        }
        active.push(PointActiveInput {
            input: input.id,
            derivative: input.binding.evaluation.derivative,
            values,
        });
    }
    Ok(PointEvaluation {
        time: evaluation.time,
        cell: evaluation.cell,
        coordinates: evaluation.coordinates.clone(),
        active,
        bound: Vec::new(),
    })
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

/// Largest `dimension()` for which [`RealizationPlan::prove_symmetry`] is willing to assemble the
/// action (`dimension()` operator applications) to establish a symmetry proof. Above it the
/// proof is refused rather than attempted; a structural proof from the factorization is the
/// intended replacement (GX-C5 follow-up).
pub const SYMMETRY_PROOF_DIMENSION_CAP: usize = 4_096;

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

    /// Declared symmetry: `Nonsymmetric` whenever an affine dependency constraint replaces a
    /// target row with a constraint residual (the existing rule); otherwise the proof recorded
    /// by an explicit [`RealizationPlan::prove_symmetry`] call on this realization, or `Unknown`
    /// when no proof has been established. This method never assembles or probes the action
    /// itself, so it is cheap to call from solver loops.
    fn symmetry(&self) -> OperatorSymmetry {
        if self.plan.data.constraints.has_affine_dependencies() {
            return OperatorSymmetry::Nonsymmetric;
        }
        self.plan
            .data
            .symmetry_proof
            .get()
            .copied()
            .unwrap_or(OperatorSymmetry::Unknown)
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

impl MatrixFreeOperator {
    pub(crate) fn plan(&self) -> &RealizationPlan {
        &self.plan
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

/// Entrywise transpose comparison of a square CSR matrix under a relative tolerance. Missing
/// transposed entries count as zero, so a structurally one-sided but numerically negligible
/// entry still passes; any pair that differs beyond `tolerance * max(|a|, |b|, 1)` fails.
pub(crate) fn csr_is_symmetric_within(matrix: &CsrMatrix, tolerance: f64) -> bool {
    let rows = matrix.rows();
    if rows != matrix.columns() {
        return false;
    }
    let offsets = matrix.row_offsets();
    let columns = matrix.column_indices();
    let values = matrix.values();
    let entry = |row: usize, column: usize| -> f64 {
        let range = offsets[row]..offsets[row + 1];
        columns[range.clone()]
            .binary_search(&column)
            .map_or(0.0, |offset| values[range.start + offset])
    };
    (0..rows).all(|row| {
        (offsets[row]..offsets[row + 1]).all(|index| {
            let column = columns[index];
            let value = values[index];
            let mirrored = entry(column, row);
            let scale = value.abs().max(mirrored.abs()).max(1.0);
            (value - mirrored).abs() <= tolerance * scale
        })
    })
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

impl TransposableOperator for AssembledOperator {
    /// SV1-C1: the canonical CSR transpose action, without materializing
    /// [`Self::transpose`]'s matrix.
    fn apply_transpose(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.matrix.apply_transpose(context, input, output)
    }
}

impl TransposableOperator for MatrixFreeOperator {
    /// SV1-C1: the exact transpose of [`LinearOperator::apply`] at the same zero linearization
    /// point, executing the bound Malleus VJP kernels through
    /// [`RealizationPlan::vector_jacobian_product`]. Affine dependency constraints refuse
    /// exactly as that method does (SV1-C2).
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; self.plan.dimension()];
        self.plan
            .vector_jacobian_product(0.0, &zero, &zero, input, output)
            .map_err(numeric_error)
    }
}

/// SV1-C1: the Jacobian `J = dR/du + rate_shift * dR/du_t` of one [`RealizationPlan`] at a
/// fixed linearization point `(time, state, state_rate)`, as a Methodus operator pair: the
/// primal action executes the bound JVP kernels
/// ([`RealizationPlan::jacobian_vector_product`] with rate direction `rate_shift * x`) and the
/// transpose action executes the bound VJP kernels
/// ([`RealizationPlan::vector_jacobian_product_shifted`]). `rate_shift = 0` is the steady
/// (Newton / adjoint-at-converged-state) Jacobian; a BDF step's Jacobian is `rate_shift =
/// alpha / dt`. Constructed by [`RealizationPlan::linearize`].
///
/// Symmetry is declared `Nonsymmetric` under affine dependency constraints (their rows replace
/// the operator's rows) and `Unknown` otherwise: a nonlinear form's Jacobian at a nonzero state
/// is not certified by the zero-state proof [`RealizationPlan::prove_symmetry`] records, so
/// nothing is claimed. Adjoint consumers use [`methodus::TransposeOperator::explicit`].
#[derive(Clone, Debug)]
pub struct LinearizedOperator {
    plan: RealizationPlan,
    time: f64,
    state: Vec<f64>,
    state_rate: Vec<f64>,
    rate_shift: f64,
}

impl LinearizedOperator {
    pub fn plan(&self) -> &RealizationPlan {
        &self.plan
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn state(&self) -> &[f64] {
        &self.state
    }

    pub fn state_rate(&self) -> &[f64] {
        &self.state_rate
    }

    pub fn rate_shift(&self) -> f64 {
        self.rate_shift
    }
}

impl LinearOperator for LinearizedOperator {
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
        let rate_direction = input
            .iter()
            .map(|value| self.rate_shift * value)
            .collect::<Vec<_>>();
        self.plan
            .jacobian_vector_product(
                self.time,
                &self.state,
                &self.state_rate,
                input,
                &rate_direction,
                output,
            )
            .map_err(numeric_error)
    }
}

impl TransposableOperator for LinearizedOperator {
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.plan
            .vector_jacobian_product_shifted(
                self.time,
                &self.state,
                &self.state_rate,
                input,
                self.rate_shift,
                output,
            )
            .map_err(numeric_error)
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
    facet_regions: &BTreeMap<scientia::RegionId, Vec<FacetId>>,
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
            // Vector H1 with one component per spatial axis is the SV2-A/SV2-B1 production
            // slice: vertex-major (P1) or vertex-then-edge-major (P2) component blocks execute
            // through the same generated kernels.
            ValueShape::Vector(components) => *components as usize == mesh.dimension(),
            _ => false,
        };
        if requirement.topological_dimension as usize != mesh.dimension()
            || requirement.family != ElementFamilyRequirement::H1
            || !matches!(requirement.polynomial_order, 1 | 2)
            || !admitted_shape
        {
            return Err(FinitumError::UnsupportedRealization(format!(
                "realization supports scalar or dimension-vector H1(order=1|2) cell elements, got {requirement:?}"
            )));
        }
    }
    // A single `RealizationPlan` binds one `PreparedElement`/`DofMap` pair, so every admitted
    // element requirement must agree on one polynomial order (SV2-B1: P1 and P2 are each
    // supported, but not mixed within one plan -- a genuine mixed-order product space is
    // `crate::mixed::MixedSpace`, not this single-field plan).
    let polynomial_order = requirements
        .elements
        .first()
        .map(|first| first.polynomial_order)
        .unwrap_or(1);
    if requirements
        .elements
        .iter()
        .any(|requirement| requirement.polynomial_order != polynomial_order)
    {
        return Err(FinitumError::UnsupportedRealization(
            "realization requires every admitted element requirement to share one polynomial \
             order; a mixed-order product space is realized through `mixed::MixedSpace` instead"
                .into(),
        ));
    }
    let expected_basis_count = match polynomial_order {
        1 => mesh.dimension() + 1,
        2 => (mesh.dimension() + 1) * (mesh.dimension() + 2) / 2,
        other => {
            return Err(FinitumError::UnsupportedRealization(format!(
                "realization supports H1(order=1|2) cell elements, got order {other}"
            )));
        }
    };
    if element.basis_count() != expected_basis_count {
        return Err(FinitumError::InvalidRealization(format!(
            "P{polynomial_order} simplex in dimension {} requires {expected_basis_count} basis \
             functions, got {}",
            mesh.dimension(),
            element.basis_count()
        )));
    }
    if polynomial_order != 1
        && factorization
            .integrals
            .iter()
            .any(|integral| matches!(integral.measure, SemanticMeasure::ExteriorFacet { .. }))
    {
        return Err(FinitumError::UnsupportedRealization(
            "GX-C4 exterior facet trace evaluation is realized with a hardcoded P1 trace basis; \
             a P2 (or higher) realization with a facet integral is refused rather than silently \
             using the wrong trace basis"
                .into(),
        ));
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
        match &integral.measure {
            SemanticMeasure::Cell { .. } => {
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
            SemanticMeasure::ExteriorFacet { region } => {
                // GX-C4 (SV2-B2 pulled forward): bounded to mesh dimension 2/3, Value-only
                // trace evaluation, and a caller-resolved, non-empty facet list per region.
                reference_facet_weight(mesh.dimension())?;
                let facets = facet_regions
                    .get(region)
                    .filter(|facets| !facets.is_empty());
                if facets.is_none() {
                    return Err(FinitumError::RealizationRegionUnmapped(format!(
                        "{region:?}"
                    )));
                }
                for input in &integral.primal.inputs {
                    validate_facet_input_contract(input)?;
                }
                for output in &integral.primal.outputs {
                    validate_facet_evaluation(&output.binding.evaluation.derivative)?;
                    if output.binding.evaluation.site != EvaluationSite::ExteriorTrace {
                        return Err(FinitumError::UnsupportedRealization(
                            "GX-C4 realizes exterior facet trace evaluation sites only".into(),
                        ));
                    }
                }
            }
            SemanticMeasure::InteriorFacet { .. }
            | SemanticMeasure::Interface { .. }
            | SemanticMeasure::Point { .. } => {
                return Err(FinitumError::UnsupportedRealization(
                    "interior facet, interface, and point traversal is deferred; only cell and \
                     exterior facet integrals (GX-C4) are realized"
                        .into(),
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

/// GX-C4: as [`validate_input_contract`], for an exterior facet trace input. Every basis-sourced
/// input must be `Active`, exactly like the cell path; only `Value`/`TimeDerivative` trace
/// evaluation is admitted (Gradient traces are refused per the bounded contract).
fn validate_facet_input_contract(input: &QFunctionInput) -> Result<(), FinitumError> {
    if input.binding.evaluation.site != EvaluationSite::ExteriorTrace {
        return Err(FinitumError::UnsupportedRealization(
            "GX-C4 realizes exterior facet trace evaluation sites only".into(),
        ));
    }
    validate_facet_evaluation(&input.binding.evaluation.derivative)?;
    if input.source == InputSourceRequirement::Basis && input.role != TensorInputRole::Active {
        return Err(FinitumError::UnsupportedRealization(
            "GX-C4 has one active field trace; additional basis-backed coefficients are deferred"
                .into(),
        ));
    }
    Ok(())
}

fn validate_facet_evaluation(derivative: &DerivativeEvaluation) -> Result<(), FinitumError> {
    if matches!(
        derivative,
        DerivativeEvaluation::Value | DerivativeEvaluation::TimeDerivative
    ) {
        Ok(())
    } else {
        Err(FinitumError::UnsupportedRealization(format!(
            "GX-C4 exterior facet integrals support Value and time-derivative trace evaluation \
             only (Gradient traces are refused), got {derivative:?}"
        )))
    }
}

fn validate_external_inputs(
    factorization: &OperatorFactorization,
    mesh: &Mesh,
    element: &PreparedElement,
    external_inputs: Vec<ExternalInput>,
    dynamic_external_inputs: Vec<DynamicExternalInput>,
    facet_regions: &BTreeMap<scientia::RegionId, Vec<FacetId>>,
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
                    let expected = match &integral.measure {
                        SemanticMeasure::ExteriorFacet { region } => facet_regions
                            .get(region)
                            .map(Vec::len)
                            .unwrap_or(0)
                            .checked_mul(components)
                            .ok_or_else(|| {
                                FinitumError::InvalidRealization(
                                    "facet external input storage extent overflows usize".into(),
                                )
                            })?,
                        _ => mesh
                            .cells()
                            .len()
                            .checked_mul(element.quadrature().len())
                            .and_then(|count| count.checked_mul(components))
                            .ok_or_else(|| {
                                FinitumError::InvalidRealization(
                                    "external input storage extent overflows usize".into(),
                                )
                            })?,
                    };
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

pub(crate) fn bind_kernels(
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
    // - Divergence: dimension-vector field input; state stride equals the spatial dimension and
    //   the output is the single scalar div(u) (GX-... mixed-system divergence coupling).
    let components = match input.binding.evaluation.derivative {
        DerivativeEvaluation::SymmetricGradient | DerivativeEvaluation::Divergence => {
            element.dimension()
        }
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
        DerivativeEvaluation::Divergence => {
            let dimension = element.dimension();
            let mut divergence = 0.0;
            for basis in 0..element.basis_count() {
                let physical = geometry.physical_gradient(
                    element
                        .basis_gradient(point, basis)
                        .expect("validated element table"),
                );
                for axis in 0..dimension {
                    divergence += physical[axis] * local_state[basis * components + axis];
                }
            }
            Ok(vec![divergence])
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
        DerivativeEvaluation::SymmetricGradient if point_output.len() == dimension * dimension => {
            // Transpose of the symmetric-gradient gather `evaluate_basis_input` builds:
            // sym[r][c] = sum_basis 0.5 * (g[r]*u[basis][c] + g[c]*u[basis][r]). Differentiating
            // with respect to state[basis][component] and contracting against `point_output`
            // gives the row/column pair below (GX-F7, used by the VJP trial-side scatter of a
            // symmetric-gradient active input, e.g. vector H1 elasticity strain).
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
                            "symmetric-gradient adjoint exceeds the local output extent".into(),
                        ));
                    }
                    let mut sum = 0.0;
                    for axis in 0..dimension {
                        sum += point_output[axis * dimension + component] * gradient[axis];
                        sum += point_output[component * dimension + axis] * gradient[axis];
                    }
                    local_output[slot] += scale * 0.5 * sum;
                }
            }
        }
        DerivativeEvaluation::Divergence if point_output.len() == 1 => {
            // div(v) = sum_c d(v_c)/dx_c: component c of the test basis at node `basis`
            // contributes its physical gradient's own c-th axis, scaled by the scalar point
            // output (GX-... mixed-system divergence coupling, the adjoint of the `Divergence`
            // case `evaluate_basis_input` gathers).
            for basis in 0..basis_count {
                let gradient = geometry.physical_gradient(
                    element
                        .basis_gradient(point, basis)
                        .expect("validated element table"),
                );
                for (component, gradient_component) in gradient.iter().enumerate().take(dimension) {
                    let slot = basis * stride + component;
                    if slot >= local_output.len() {
                        return Err(FinitumError::InvalidRealization(
                            "divergence adjoint exceeds the local output extent".into(),
                        ));
                    }
                    local_output[slot] += scale * gradient_component * point_output[0];
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

/// Exact transpose of [`apply_basis_adjoint`]: gather a point-space cotangent from a restricted
/// adjoint vector through the test basis of `derivative`, mirroring the same four evaluation
/// shapes (GX-F7, used by the VJP's test-side gather of the adjoint before the bound VJP kernel
/// runs). Unlike `apply_basis_adjoint`, this never scales by the quadrature weight: scale is
/// applied once, at the trial-side scatter that follows the kernel, matching where
/// `evaluate_basis_input`'s own forward gather (unscaled) sits in the JVP pipeline.
pub(crate) fn gather_test_adjoint(
    element: &PreparedElement,
    geometry: &CellGeometry,
    point: usize,
    derivative: &DerivativeEvaluation,
    output_components: usize,
    local_adjoint: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    let dimension = element.dimension();
    let basis_count = element.basis_count();
    let stride = if local_adjoint.len() % basis_count == 0 {
        local_adjoint.len() / basis_count
    } else {
        return Err(FinitumError::InvalidRealization(
            "local adjoint length is not a multiple of the basis count".into(),
        ));
    };
    match derivative {
        DerivativeEvaluation::Value if output_components == 1 => {
            let mut value = 0.0;
            for (basis, coefficient) in local_adjoint.iter().enumerate().take(basis_count) {
                value += element.basis_value(point, basis).unwrap() * coefficient;
            }
            Ok(vec![value])
        }
        DerivativeEvaluation::Value if output_components == stride => {
            let mut value = vec![0.0; stride];
            for basis in 0..basis_count {
                let weight = element.basis_value(point, basis).unwrap();
                for component in 0..stride {
                    value[component] += weight * local_adjoint[basis * stride + component];
                }
            }
            Ok(value)
        }
        DerivativeEvaluation::Gradient if output_components == dimension => {
            let mut value = vec![0.0; dimension];
            for (basis, coefficient) in local_adjoint.iter().enumerate().take(basis_count) {
                let gradient = geometry.physical_gradient(
                    element
                        .basis_gradient(point, basis)
                        .expect("validated element table"),
                );
                for axis in 0..dimension {
                    value[axis] += gradient[axis] * coefficient;
                }
            }
            Ok(value)
        }
        DerivativeEvaluation::Gradient if output_components == dimension * dimension => {
            let mut value = vec![0.0; dimension * dimension];
            for basis in 0..basis_count {
                let gradient = geometry.physical_gradient(
                    element
                        .basis_gradient(point, basis)
                        .expect("validated element table"),
                );
                for component in 0..dimension {
                    let slot = basis * stride + component;
                    if slot >= local_adjoint.len() {
                        return Err(FinitumError::InvalidRealization(
                            "flux adjoint gather exceeds the local adjoint extent".into(),
                        ));
                    }
                    let coefficient = local_adjoint[slot];
                    for axis in 0..dimension {
                        value[axis * dimension + component] += gradient[axis] * coefficient;
                    }
                }
            }
            Ok(value)
        }
        _ => Err(FinitumError::InvalidRealization(format!(
            "adjoint output with {output_components} components does not match {derivative:?}"
        ))),
    }
}

/// Execute one bound bundle's JVP (and, for a frozen-parameter contribution, its parameter-JVP)
/// kernel at one quadrature point from already-gathered primal `inputs` and direction `values`.
///
/// This is the exact per-point JVP evaluation `RealizationPlan::execute_jvp`/`execute_jvp_facet`
/// use; it is a free function (not a `RealizationPlan` method) because it touches neither
/// `self.data.element` nor `self.data.dofs` -- `inputs`/`directions` are already resolved into
/// [`TensorInputId`]-keyed value maps by the caller, so the same function serves any caller that
/// can build those maps, single-field or multi-field ([`crate::system`]'s system realization
/// reuses it directly rather than duplicating this dispatch).
pub(crate) fn execute_jvp_values(
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

/// SV1-C3: execute one bound bundle's frozen-input (parameter) JVP kernel with `direction`
/// routed to `hot_input` only and every other frozen operand's direction zero -- the exact
/// point-local `d(output)/d(hot_input) * direction`. Returns `None` when `hot_input` is not an
/// operand of this output's kernels at all (the output does not depend on it); refuses typed
/// when the kernel reads the input but its parameter program carries no tangent for it.
pub(crate) fn execute_parameter_jvp_values(
    bound: &BoundBundle,
    inputs: &BTreeMap<TensorInputId, Vec<f64>>,
    hot_input: TensorInputId,
    direction: &[f64],
) -> Result<Option<Vec<f64>>, FinitumError> {
    let input_by_operand = bound
        .bundle
        .primal_inputs
        .iter()
        .map(|binding| (binding.operand, binding.input))
        .collect::<BTreeMap<_, _>>();
    if !input_by_operand.values().any(|input| *input == hot_input) {
        return Ok(None);
    }
    let mut values = bound
        .bundle
        .primal_inputs
        .iter()
        .map(|binding| (binding.operand, inputs[&binding.input].clone()))
        .collect::<BTreeMap<_, _>>();
    let mut routed = false;
    for pair in &bound.bundle.parameter.independent_operands {
        let input = input_by_operand.get(&pair.primal).copied().ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "parameter-JVP operand {:?} has no QFunction input binding",
                pair.primal
            ))
        })?;
        if input == hot_input {
            if direction.len() != inputs[&input].len() {
                return Err(FinitumError::InvalidRealization(format!(
                    "coefficient direction has {} components, input {:?} has {}",
                    direction.len(),
                    input,
                    inputs[&input].len()
                )));
            }
            values.insert(pair.derivative, direction.to_vec());
            routed = true;
        } else {
            values.insert(pair.derivative, vec![0.0; inputs[&input].len()]);
        }
    }
    if !routed {
        return Err(FinitumError::RealizationTangentUnavailable(format!(
            "kernel output ({}, {}) reads input {hot_input:?} but its parameter program \
             carries no tangent for it",
            bound.bundle.integral_index, bound.bundle.output_index
        )));
    }
    let executable = &bound.executable.kernels()[bound.bundle.parameter.kernel_index];
    let buffers = execute(executable, &values)?;
    operand_values(
        executable,
        &buffers,
        bound.bundle.parameter.dependent_operands[0].derivative,
    )
    .map(Some)
}

/// Execute one bound bundle's PRIMAL kernel against already-gathered point `inputs`, returning
/// its declared primal output -- the shared kernel-execution core [`RealizationPlan::
/// execute_primal`] and [`crate::system::SystemOperator::load_vector`] both drive, promoted here
/// (mirroring [`execute_jvp_values`]'s own promotion) so the single-field and system realization
/// paths share one PRIMAL-kernel-execution implementation rather than each repeating it.
pub(crate) fn execute_primal_values(
    bound: &BoundBundle,
    inputs: &BTreeMap<TensorInputId, Vec<f64>>,
) -> Result<Vec<f64>, FinitumError> {
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
                        "bundle input {:?} is absent from the gathered point inputs",
                        binding.input
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

pub(crate) fn execute(
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

pub(crate) fn operand_values(
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

pub(crate) fn component_count(shape: &[usize]) -> Result<usize, FinitumError> {
    shape.iter().try_fold(1usize, |count, extent| {
        count.checked_mul(*extent).ok_or_else(|| {
            FinitumError::InvalidRealization("tensor component extent overflows usize".into())
        })
    })
}

pub(crate) fn validate_finite(operation: &str, values: &[f64]) -> Result<(), FinitumError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(FinitumError::InvalidRealization(format!(
            "{operation} contains a non-finite value at index {index}"
        )))
    } else {
        Ok(())
    }
}

fn numeric_error(error: FinitumError) -> NumericError {
    NumericError::from(error)
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
    pub(crate) fn new(mesh: &Mesh, cell_id: CellId) -> Result<Self, FinitumError> {
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

    pub(crate) fn physical_point(&self, reference: &[f64]) -> Vec<f64> {
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

    pub(crate) fn physical_gradient(&self, reference: &[f64]) -> Vec<f64> {
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

    /// Absolute Jacobian determinant of this cell's affine reference-to-physical map, i.e. the
    /// quadrature weight scale factor `crate::system` needs for its own per-cell integration
    /// loop (mirroring how `apply_cell` uses this field internally within this module).
    pub(crate) fn determinant(&self) -> f64 {
        self.determinant
    }
}

/// GX-C4: exterior facet geometry for one facet's single incident cell. Scope is bounded to mesh
/// dimension 2 (segment facets of triangles) and 3 (triangular facets of tetrahedra), per the
/// SV2-B2 pulled-forward contract. Quadrature is a single reference-facet centroid point,
/// matching the same degree-1-exact convention `PreparedElement::linear_simplex` uses for cells
/// (the boundary integrand of a P1 form is itself affine, so one point is exact).
#[derive(Clone, Debug)]
pub(crate) struct FacetGeometry {
    pub(crate) cell: CellId,
    #[allow(dead_code)]
    pub(crate) local_facet: usize,
    /// Physical coordinates of the facet centroid quadrature point.
    pub(crate) physical_centroid: Vec<f64>,
    /// The centroid in the owning cell's reference coordinates (same convention as
    /// [`CellGeometry`]: reference vertex 0 is the origin, reference vertex `i` is the `i`-th
    /// standard basis vector), used to evaluate the cell's own trace basis functions.
    reference_centroid: Vec<f64>,
    /// Physical facet measure divided by the reference facet's own measure (`1` for a segment,
    /// `0.5` for a triangle) -- i.e. the exact analog of [`CellGeometry::determinant`] for the
    /// facet's own affine embedding.
    jacobian_determinant: f64,
    /// Unit outward normal in physical coordinates.
    #[allow(dead_code)]
    pub(crate) normal: Vec<f64>,
}

/// Reference measure of the (mesh_dimension - 1)-simplex facet, matching
/// `PreparedElement::linear_simplex`'s weight formula (`1/(d-1)!`).
fn reference_facet_weight(mesh_dimension: usize) -> Result<f64, FinitumError> {
    match mesh_dimension {
        2 => Ok(1.0),
        3 => Ok(0.5),
        other => Err(FinitumError::UnsupportedRealization(format!(
            "exterior facet integrals are supported for mesh dimension 2 or 3, got {other}"
        ))),
    }
}

impl FacetGeometry {
    pub(crate) fn compute(mesh: &Mesh, incidence: FacetIncidence) -> Result<Self, FinitumError> {
        let dimension = mesh.dimension();
        reference_facet_weight(dimension)?;
        let cell = mesh.cell(incidence.cell).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("mesh has no cell {}", incidence.cell.0))
        })?;
        if incidence.local_facet >= cell.vertices.len() {
            return Err(FinitumError::InvalidRealization(
                "facet incidence local_facet index is out of range".into(),
            ));
        }
        let omitted = incidence.local_facet;
        let retained_slots = (0..cell.vertices.len())
            .filter(|&slot| slot != omitted)
            .collect::<Vec<_>>();
        let physical_vertices = retained_slots
            .iter()
            .map(|&slot| &mesh.vertices()[cell.vertices[slot].0])
            .collect::<Vec<_>>();
        let physical_centroid = mean_point(&physical_vertices, dimension);
        let reference_vertices = retained_slots
            .iter()
            .map(|&slot| reference_vertex(slot, dimension))
            .collect::<Vec<_>>();
        let reference_centroid =
            mean_point(&reference_vertices.iter().collect::<Vec<_>>(), dimension);
        let origin = physical_vertices[0];
        let tangents = physical_vertices[1..]
            .iter()
            .map(|vertex| {
                (0..dimension)
                    .map(|axis| vertex[axis] - origin[axis])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (jacobian_determinant, mut normal) = match dimension {
            2 => {
                let t = &tangents[0];
                let measure = (t[0] * t[0] + t[1] * t[1]).sqrt();
                (measure, vec![t[1], -t[0]])
            }
            3 => {
                let t1 = &tangents[0];
                let t2 = &tangents[1];
                let cross = [
                    t1[1] * t2[2] - t1[2] * t2[1],
                    t1[2] * t2[0] - t1[0] * t2[2],
                    t1[0] * t2[1] - t1[1] * t2[0],
                ];
                let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                (norm, cross.to_vec())
            }
            _ => unreachable!("reference_facet_weight already validated the dimension"),
        };
        if !jacobian_determinant.is_finite() || jacobian_determinant <= 0.0 {
            return Err(FinitumError::InvalidRealization(format!(
                "facet {} of cell {} has a degenerate geometry",
                incidence.local_facet, incidence.cell.0
            )));
        }
        let normal_norm = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
        for value in &mut normal {
            *value /= normal_norm;
        }
        let opposite = &mesh.vertices()[cell.vertices[omitted].0];
        let to_opposite = (0..dimension)
            .map(|axis| opposite[axis] - origin[axis])
            .collect::<Vec<_>>();
        let dot = normal
            .iter()
            .zip(&to_opposite)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        if dot > 0.0 {
            for value in &mut normal {
                *value = -*value;
            }
        }
        if normal.iter().any(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(format!(
                "facet {} of cell {} has a non-finite normal",
                incidence.local_facet, incidence.cell.0
            )));
        }
        Ok(Self {
            cell: incidence.cell,
            local_facet: omitted,
            physical_centroid,
            reference_centroid,
            jacobian_determinant,
            normal,
        })
    }

    /// `PreparedElement::linear_simplex`-style scale: reference facet weight times the physical
    /// Jacobian determinant, the exact analog of `quadrature.weight * geometry.determinant` for
    /// the single-point facet rule.
    pub(crate) fn scale(&self, mesh_dimension: usize) -> f64 {
        reference_facet_weight(mesh_dimension).expect("validated at construction")
            * self.jacobian_determinant
    }
}

fn reference_vertex(slot: usize, dimension: usize) -> Vec<f64> {
    let mut vertex = vec![0.0; dimension];
    if slot > 0 {
        vertex[slot - 1] = 1.0;
    }
    vertex
}

fn mean_point(points: &[&Vec<f64>], dimension: usize) -> Vec<f64> {
    let mut mean = vec![0.0; dimension];
    for point in points {
        for axis in 0..dimension {
            mean[axis] += point[axis];
        }
    }
    for value in &mut mean {
        *value /= points.len() as f64;
    }
    mean
}

/// P1 barycentric basis values at an arbitrary cell reference point (not restricted to the
/// cell's own stored quadrature table), for facet trace evaluation: `basis[0] = 1 - sum(x)`,
/// `basis[i] = x[i - 1]` for `i = 1..=dimension`, matching `CellGeometry`'s reference-vertex
/// convention.
fn p1_trace_basis_values(reference: &[f64]) -> Vec<f64> {
    let mut values = Vec::with_capacity(reference.len() + 1);
    values.push(1.0 - reference.iter().sum::<f64>());
    values.extend_from_slice(reference);
    values
}

/// Value-only analog of [`evaluate_basis_input`] at a facet trace point: gathers the local state
/// through the cell's P1 basis evaluated at `basis_values` (already computed at the facet
/// centroid's cell-reference coordinates). Refuses non-`Value`/`TimeDerivative` evaluations,
/// since GX-C4's bounded scope refuses Gradient traces at `validate_discretization` time.
pub(crate) fn evaluate_trace_basis_input(
    basis_values: &[f64],
    input: &QFunctionInput,
    local_state: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    if !matches!(
        input.binding.evaluation.derivative,
        DerivativeEvaluation::Value | DerivativeEvaluation::TimeDerivative
    ) {
        return Err(FinitumError::UnsupportedRealization(format!(
            "exterior facet trace evaluation supports Value only, got {:?}",
            input.binding.evaluation.derivative
        )));
    }
    let components = vector_components(input)?;
    if components == 1 {
        let value = local_state
            .iter()
            .zip(basis_values)
            .map(|(state, basis)| state * basis)
            .sum();
        Ok(vec![value])
    } else {
        let mut values = vec![0.0; components];
        for (basis_index, basis) in basis_values.iter().enumerate() {
            for component in 0..components {
                values[component] += basis * local_state[basis_index * components + component];
            }
        }
        Ok(values)
    }
}

/// Value-only analog of [`apply_basis_adjoint`] for a facet trace point.
pub(crate) fn apply_trace_basis_adjoint(
    basis_values: &[f64],
    point_output: &[f64],
    scale: f64,
    local_output: &mut [f64],
) -> Result<(), FinitumError> {
    let basis_count = basis_values.len();
    let stride = if local_output.len() % basis_count == 0 {
        local_output.len() / basis_count
    } else {
        return Err(FinitumError::InvalidRealization(
            "local output length is not a multiple of the basis count".into(),
        ));
    };
    if point_output.len() == 1 {
        for (basis, output) in local_output.iter_mut().enumerate() {
            *output += scale * basis_values[basis] * point_output[0];
        }
        Ok(())
    } else if point_output.len() == stride {
        for basis in 0..basis_count {
            for component in 0..stride {
                local_output[basis * stride + component] +=
                    scale * basis_values[basis] * point_output[component];
            }
        }
        Ok(())
    } else {
        Err(FinitumError::InvalidRealization(format!(
            "point output with {} components does not match a facet trace Value evaluation",
            point_output.len()
        )))
    }
}

/// Value-only analog of [`gather_test_adjoint`] for a facet trace point.
pub(crate) fn gather_trace_test_adjoint(
    basis_values: &[f64],
    output_components: usize,
    local_adjoint: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    let basis_count = basis_values.len();
    let stride = if local_adjoint.len() % basis_count == 0 {
        local_adjoint.len() / basis_count
    } else {
        return Err(FinitumError::InvalidRealization(
            "local adjoint length is not a multiple of the basis count".into(),
        ));
    };
    if output_components == 1 {
        let mut value = 0.0;
        for (basis, coefficient) in local_adjoint.iter().enumerate().take(basis_count) {
            value += basis_values[basis] * coefficient;
        }
        Ok(vec![value])
    } else if output_components == stride {
        let mut value = vec![0.0; stride];
        for basis in 0..basis_count {
            for component in 0..stride {
                value[component] += basis_values[basis] * local_adjoint[basis * stride + component];
            }
        }
        Ok(value)
    } else {
        Err(FinitumError::InvalidRealization(format!(
            "adjoint output with {output_components} components does not match a facet trace \
             Value evaluation"
        )))
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

#[cfg(test)]
mod gx_c4_facets {
    use super::*;
    use crate::{Cell, Mesh, VertexId};

    #[test]
    fn facet_evaluation_refuses_gradient_traces_but_admits_value_and_time_derivative() {
        assert!(validate_facet_evaluation(&DerivativeEvaluation::Value).is_ok());
        assert!(validate_facet_evaluation(&DerivativeEvaluation::TimeDerivative).is_ok());
        assert!(validate_facet_evaluation(&DerivativeEvaluation::Gradient).is_err());
        assert!(validate_facet_evaluation(&DerivativeEvaluation::SymmetricGradient).is_err());
    }

    #[test]
    fn reference_facet_weight_admits_only_dimension_two_and_three() {
        assert_eq!(reference_facet_weight(2).unwrap(), 1.0);
        assert_eq!(reference_facet_weight(3).unwrap(), 0.5);
        assert!(reference_facet_weight(1).is_err());
        assert!(reference_facet_weight(4).is_err());
    }

    /// Unit right triangle (0,0)-(1,0)-(0,1): the facet opposite vertex 0 (the hypotenuse,
    /// local_facet = 0) has physical length `sqrt(2)`, centroid `(0.5, 0.5)`, and outward normal
    /// `(1/sqrt(2), 1/sqrt(2))` (pointing away from the origin).
    #[test]
    fn facet_geometry_matches_hand_computation_on_a_unit_right_triangle() {
        let mesh = Mesh::new(
            2,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![Cell {
                vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
            }],
        )
        .unwrap();
        let incidence = FacetIncidence {
            cell: CellId(0),
            local_facet: 0,
            orientation: 1,
        };
        let geometry = FacetGeometry::compute(&mesh, incidence).unwrap();
        assert!((geometry.physical_centroid[0] - 0.5).abs() <= 1.0e-12);
        assert!((geometry.physical_centroid[1] - 0.5).abs() <= 1.0e-12);
        let expected_length = std::f64::consts::SQRT_2;
        assert!((geometry.jacobian_determinant - expected_length).abs() <= 1.0e-12);
        let expected_normal = 1.0 / std::f64::consts::SQRT_2;
        assert!((geometry.normal[0] - expected_normal).abs() <= 1.0e-12);
        assert!((geometry.normal[1] - expected_normal).abs() <= 1.0e-12);
        // scale() for a dim-2 mesh is reference weight 1.0 times the Jacobian determinant.
        assert!((geometry.scale(2) - expected_length).abs() <= 1.0e-12);
    }

    /// The facet opposite vertex 1 (the segment from (0,0) to (0,1), local_facet = 1) has
    /// outward normal `(-1, 0)` and length `1`.
    #[test]
    fn facet_geometry_normal_points_away_from_the_opposite_vertex() {
        let mesh = Mesh::new(
            2,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![Cell {
                vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
            }],
        )
        .unwrap();
        let incidence = FacetIncidence {
            cell: CellId(0),
            local_facet: 1,
            orientation: 1,
        };
        let geometry = FacetGeometry::compute(&mesh, incidence).unwrap();
        assert!((geometry.jacobian_determinant - 1.0).abs() <= 1.0e-12);
        assert!((geometry.normal[0] - (-1.0)).abs() <= 1.0e-12);
        assert!(geometry.normal[1].abs() <= 1.0e-12);
    }

    #[test]
    fn p1_trace_basis_values_vanish_at_the_omitted_vertex_reference() {
        // Facet opposite vertex 0: cell reference centroid of vertices 1 and 2 is (0.5, 0.5).
        let values = p1_trace_basis_values(&[0.5, 0.5]);
        assert_eq!(values.len(), 3);
        assert!(
            values[0].abs() <= 1.0e-12,
            "basis 0 should vanish: {values:?}"
        );
        assert!((values[1] - 0.5).abs() <= 1.0e-12);
        assert!((values[2] - 0.5).abs() <= 1.0e-12);
    }
}
