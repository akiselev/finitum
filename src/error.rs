use std::fmt;

use methodus::NumericError;
use thiserror::Error;

use crate::CellId;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum FinitumError {
    #[error("mesh dimension must be in 1..=3, got {0}")]
    InvalidDimension(usize),
    #[error("vertex {vertex} has coordinate dimension {actual}, expected {expected}")]
    CoordinateDimension {
        vertex: usize,
        actual: usize,
        expected: usize,
    },
    #[error("vertex {vertex} coordinate axis {axis} is not finite")]
    NonFiniteCoordinate { vertex: usize, axis: usize },
    #[error("cell {cell} references missing vertex {vertex}")]
    MissingVertex { cell: usize, vertex: usize },
    #[error("cell {cell} has {actual} vertices, expected {expected} for a simplex")]
    CellArity {
        cell: usize,
        actual: usize,
        expected: usize,
    },
    #[error("cell {cell} repeats vertex {vertex}; simplex vertices must be distinct")]
    DuplicateCellVertex { cell: usize, vertex: usize },
    #[error("element restriction references missing degree of freedom {0}")]
    MissingDof(usize),
    #[error("element restriction {restriction} contains no degrees of freedom")]
    EmptyRestriction { restriction: usize },
    #[error("element restriction {restriction} repeats degree of freedom {dof}")]
    DuplicateRestrictionDof { restriction: usize, dof: usize },
    #[error("constraint target {0} is outside the degree-of-freedom map")]
    InvalidConstraintTarget(usize),
    #[error("constraint dependency {0} is outside the degree-of-freedom map")]
    InvalidConstraintDependency(usize),
    #[error("degree of freedom {0} has more than one affine constraint")]
    DuplicateConstraintTarget(usize),
    #[error("constraint for degree of freedom {target} has a non-finite coefficient")]
    InvalidConstraintCoefficient { target: usize },
    #[error("constraint for degree of freedom {target} repeats dependency {dependency}")]
    DuplicateConstraintDependency { target: usize, dependency: usize },
    #[error("constraint input has length {actual}, expected {expected}")]
    ConstraintInputLength { actual: usize, expected: usize },
    #[error("constraint input for degree of freedom {0} is not finite")]
    NonFiniteConstraintInput(usize),
    #[error("constraint expansion for degree of freedom {0} produced a non-finite value")]
    NonFiniteConstraintResult(usize),
    #[error("constraint graph contains a cycle involving degree of freedom {0}")]
    ConstraintCycle(usize),
    #[error("prepared element table shape is inconsistent: {0}")]
    InvalidElementShape(String),
    #[error("prepared element contains non-finite data at {location}")]
    NonFiniteElementData { location: String },
    #[error("realization artifact mismatch: {0}")]
    ArtifactMismatch(String),
    #[error("unsupported realization contract: {0}")]
    UnsupportedRealization(String),
    #[error("invalid realization data: {0}")]
    InvalidRealization(String),
    #[error("realization is missing external input {input:?} for integral {integral}")]
    MissingExternalInput {
        integral: usize,
        input: scientia::TensorInputId,
    },
    #[error("Malleus kernel validation failed: {0}")]
    KernelValidation(String),
    #[error("Malleus kernel execution failed: {0}")]
    KernelExecution(String),
    #[error("assembled realization failed: {0}")]
    Assembly(String),
    #[error("stale CAD geometry revision: expected {expected}, got {actual}")]
    StaleGeometryRevision { expected: u64, actual: u64 },
    #[error("CAD provider source identity does not match the realized geometry")]
    CadGeometrySourceMismatch,
    #[error("CAD boundary association is missing for {0}")]
    MissingCadBoundary(String),
    #[error("CAD boundary association is ambiguous for {0}")]
    AmbiguousCadBoundary(String),
    #[error("invalid CAD geometry realization: {0}")]
    InvalidCadGeometry(String),
    #[error("CAD family {family} has no R3D realization path: {reason}")]
    UnsupportedCadFamily {
        family: String,
        reason: &'static str,
    },
    #[error("mesh profile is unsupported: {0}")]
    MeshProfileUnsupported(String),
    #[error("region {0} has no entry in the caller-supplied region map")]
    RealizationRegionUnmapped(String),
    #[error(
        "boundary partition failed: {uncovered} exterior facets uncovered, \
         {overlapping} facets covered by more than one mapped region"
    )]
    RealizationPartitionFailed {
        uncovered: usize,
        overlapping: usize,
    },
    #[error(
        "region tags {left} and {right} assign conflicting essential values at vertex {vertex}"
    )]
    ConflictingRegionValue {
        left: String,
        right: String,
        vertex: usize,
    },
    /// A global representation ([`crate::RepresentationKind`]) this realization cannot take,
    /// named down to the equation, the integral, and (when one is the cause) the bound input.
    #[error(
        "REPRESENTATION_UNSUPPORTED: {representation:?} is refused at equation `{equation}` \
         integral {integral} (input {input:?}): {reason}"
    )]
    RepresentationUnsupported {
        representation: crate::RepresentationKind,
        equation: String,
        integral: usize,
        input: Option<scientia::TensorInputId>,
        reason: String,
    },
    #[error("REALIZATION_TANGENT_UNAVAILABLE: {0}")]
    RealizationTangentUnavailable(String),
    /// A field family (or element table shape) the public field sampler
    /// ([`crate::FieldSampler`], W8 lane F1) does not reconstruct, named.
    #[error("SAMPLING_UNSUPPORTED: field family {family} is not sampled: {reason}")]
    SamplingUnsupported { family: String, reason: String },
    #[error("INF_SUP_UNSTABLE: {0}")]
    InfSupUnstable(String),
    /// A typed failure raised inside a fallible input callback (W8 lane F2, PLAN §6 W8
    /// decision 3): the producer's own refusal code and origin, located by Finitum at the cell,
    /// point and time it evaluated. [`FinitumError::code`] returns that original code, never a
    /// Finitum one, and every Methodus boundary maps it to
    /// [`NumericError::Evaluation`] unchanged (see the `From` impl below).
    #[error(transparent)]
    InputEvaluation(Box<InputEvaluationError>),
}

impl From<InputEvaluationError> for FinitumError {
    fn from(failure: InputEvaluationError) -> Self {
        Self::InputEvaluation(Box::new(failure))
    }
}

impl FinitumError {
    /// The refusal code a consumer reports for this error, when it carries one: the original
    /// producer code of an [`FinitumError::InputEvaluation`] (never a generic Finitum code), or
    /// the static code of the typed refusals whose messages begin with one. Structural errors
    /// (`InvalidRealization`, `ArtifactMismatch`, ...) carry none.
    pub fn code(&self) -> Option<&str> {
        match self {
            FinitumError::InputEvaluation(error) => Some(error.code.as_str()),
            FinitumError::RepresentationUnsupported { .. } => Some("REPRESENTATION_UNSUPPORTED"),
            FinitumError::RealizationTangentUnavailable(_) => {
                Some("REALIZATION_TANGENT_UNAVAILABLE")
            }
            FinitumError::SamplingUnsupported { .. } => Some("SAMPLING_UNSUPPORTED"),
            FinitumError::InfSupUnstable(_) => Some("INF_SUP_UNSTABLE"),
            _ => None,
        }
    }
}

/// Every Methodus operator boundary (`LinearOperator::apply`, `TransposableOperator`,
/// `NonlinearOperator`, `DaeOperator`) maps a Finitum failure through this: a typed
/// [`FinitumError::InputEvaluation`] becomes [`NumericError::Evaluation`] with the producer's
/// code and origin verbatim (and the located point/time/cell ahead of the message), so a
/// callback failure propagates through a Methodus solve as itself; every other Finitum error
/// stays the flat [`NumericError::Operator`] message it always was.
impl From<FinitumError> for NumericError {
    fn from(error: FinitumError) -> Self {
        match error {
            FinitumError::InputEvaluation(failure) => {
                let failure = *failure;
                let location = failure.location_text();
                NumericError::Evaluation {
                    code: failure.code,
                    origin: failure.origin.to_string(),
                    message: if location.is_empty() {
                        failure.message
                    } else {
                        format!("{location}: {}", failure.message)
                    },
                }
            }
            other => NumericError::Operator {
                message: other.to_string(),
            },
        }
    }
}

/// Where a failed input evaluation came from, as the producing callback names it. Finitum
/// never invents one: a callback returns its own origin, and only a stored-table builder
/// re-labels it as [`InputOrigin::Table`] (the failure happened while the table was sampled at
/// bind time, not during an operator action).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputOrigin {
    /// A consumer case slot (`provider/diffusivity`, `boundary/walls`).
    Slot(String),
    /// A model expression path (`<model>.<equation>[<integral>].<symbol>`).
    ExpressionPath(String),
    /// A named provider; displayed as `provider/<name>`.
    Provider(String),
    /// A stored quadrature-point table sampled at bind time; the payload is the display of the
    /// origin the table was built for.
    Table(String),
}

impl InputOrigin {
    /// The inverse of [`fmt::Display`] as far as it goes: `provider/<p>` is a `Provider`,
    /// `<t> (stored table)` a `Table`, anything else a `Slot` (a `Slot` and an `ExpressionPath`
    /// display identically, so a round trip through Methodus's string origin yields `Slot`).
    pub(crate) fn from_display(origin: &str) -> Self {
        if let Some(provider) = origin.strip_prefix("provider/") {
            InputOrigin::Provider(provider.to_owned())
        } else if let Some(table) = origin.strip_suffix(" (stored table)") {
            InputOrigin::Table(table.to_owned())
        } else {
            InputOrigin::Slot(origin.to_owned())
        }
    }
}

impl fmt::Display for InputOrigin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputOrigin::Slot(slot) => formatter.write_str(slot),
            InputOrigin::ExpressionPath(path) => formatter.write_str(path),
            InputOrigin::Provider(provider) => write!(formatter, "provider/{provider}"),
            InputOrigin::Table(origin) => write!(formatter, "{origin} (stored table)"),
        }
    }
}

/// Where Finitum evaluated the callback that failed (W8 lane F2).
#[derive(Clone, Debug, PartialEq)]
pub struct InputLocation {
    /// The cell whose evaluation failed (`None` for a nodal datum, which belongs to no cell).
    pub cell: Option<CellId>,
    /// Physical coordinates of the evaluation point.
    pub point: Vec<f64>,
    /// The evaluation time (`None` on a steady sampling, whose points carry no time).
    pub time: Option<f64>,
}

/// One typed failure of a fallible input callback (a constitutive law, an external input, a
/// sampled datum), as the callback raised it and as Finitum located it. The callback fills
/// `code`, `origin` and `message`; Finitum fills `location` from the evaluation it was
/// performing (overwriting whatever the callback set, since Finitum is the authority on where
/// it evaluated), so the consumer reports
/// `RUN_TANGENT_UNAVAILABLE at provider/diffusivity, point (0.25, 0.5), t = 0.1, cell 3: ..`
/// and never "non-finite value". The location is boxed so the error stays small in every
/// callback's `Result`; read it through [`Self::cell`] / [`Self::point`] / [`Self::time`].
#[derive(Clone, Debug, PartialEq)]
pub struct InputEvaluationError {
    /// The producer's own refusal code (`RUN_TANGENT_UNAVAILABLE`, `RUN_PROPERTY_UNSUPPORTED`,
    /// ...), carried verbatim.
    pub code: String,
    pub origin: InputOrigin,
    /// The producer's own message.
    pub message: String,
    /// Where Finitum evaluated the failing callback; `None` until Finitum locates it.
    pub location: Option<Box<InputLocation>>,
}

impl InputEvaluationError {
    /// A failure as a callback raises it: code, origin and message, not yet located.
    pub fn new(code: impl Into<String>, origin: InputOrigin, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            origin,
            message: message.into(),
            location: None,
        }
    }

    /// The cell Finitum evaluated in, when located and cell-bound.
    pub fn cell(&self) -> Option<CellId> {
        self.location.as_ref().and_then(|location| location.cell)
    }

    /// The physical point Finitum evaluated at, when located.
    pub fn point(&self) -> Option<&[f64]> {
        self.location
            .as_ref()
            .map(|location| location.point.as_slice())
    }

    /// The time Finitum evaluated at, when located on a transient path.
    pub fn time(&self) -> Option<f64> {
        self.location.as_ref().and_then(|location| location.time)
    }

    /// The located part of the failure as text, `point (x, y), t = 0.1, cell 3` (each part
    /// present only when known; empty when unlocated).
    pub fn location_text(&self) -> String {
        let Some(location) = &self.location else {
            return String::new();
        };
        let coordinates = location
            .point
            .iter()
            .map(|value| format!("{value}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut parts = vec![format!("point ({coordinates})")];
        if let Some(time) = location.time {
            parts.push(format!("t = {time}"));
        }
        if let Some(cell) = location.cell {
            parts.push(format!("cell {}", cell.0));
        }
        parts.join(", ")
    }

    /// Locate this failure where Finitum evaluated it (Finitum's values replace the callback's).
    pub(crate) fn at(mut self, cell: Option<CellId>, point: &[f64], time: Option<f64>) -> Self {
        self.location = Some(Box::new(InputLocation {
            cell,
            point: point.to_vec(),
            time,
        }));
        self
    }
}

impl InputEvaluationError {
    /// The typed failure a Methodus [`NumericError::Evaluation`] carries, when a Finitum check
    /// drives a caller-supplied Methodus operator and gets one back: code and origin are exact,
    /// the location stays inside `message` as text (`point (..), t = .., cell N: ..`) rather
    /// than as fields. Finitum's own operators never take this path (see
    /// `crate::verification`); it exists so `FinitumError::code` still answers.
    pub(crate) fn from_numeric(code: String, origin: &str, message: String) -> Self {
        Self {
            code,
            origin: InputOrigin::from_display(origin),
            message,
            location: None,
        }
    }
}

impl fmt::Display for InputEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} at {}", self.code, self.origin)?;
        let location = self.location_text();
        if !location.is_empty() {
            write!(formatter, ", {location}")?;
        }
        write!(formatter, ": {}", self.message)
    }
}

impl std::error::Error for InputEvaluationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure() -> InputEvaluationError {
        InputEvaluationError::new(
            "RUN_TANGENT_UNAVAILABLE",
            InputOrigin::Provider("diffusivity".into()),
            "no tangent for an analytic_provided law",
        )
    }

    #[test]
    fn a_located_failure_displays_its_code_origin_point_time_and_cell_in_that_order() {
        let located = failure().at(Some(CellId(3)), &[0.25, 0.5], Some(0.1));
        assert_eq!(
            located.to_string(),
            "RUN_TANGENT_UNAVAILABLE at provider/diffusivity, point (0.25, 0.5), t = 0.1, \
             cell 3: no tangent for an analytic_provided law"
        );
        let steady = failure().at(None, &[1.0, 2.0, 3.0], None);
        assert_eq!(
            steady.to_string(),
            "RUN_TANGENT_UNAVAILABLE at provider/diffusivity, point (1, 2, 3): no tangent for \
             an analytic_provided law"
        );
        assert_eq!(
            failure().to_string(),
            "RUN_TANGENT_UNAVAILABLE at provider/diffusivity: no tangent for an \
             analytic_provided law"
        );
    }

    #[test]
    fn finitum_error_code_is_the_original_code_and_the_methodus_mapping_keeps_it() {
        let located = failure().at(Some(CellId(3)), &[0.25, 0.5], Some(0.1));
        let error = FinitumError::from(located.clone());
        assert_eq!(error.code(), Some("RUN_TANGENT_UNAVAILABLE"));
        assert_eq!(error.to_string(), located.to_string());
        assert_eq!(FinitumError::InvalidRealization("x".into()).code(), None);
        assert_eq!(
            FinitumError::InfSupUnstable("x".into()).code(),
            Some("INF_SUP_UNSTABLE")
        );
        let numeric = NumericError::from(error);
        assert_eq!(numeric.evaluation_code(), Some("RUN_TANGENT_UNAVAILABLE"));
        assert_eq!(
            numeric,
            NumericError::Evaluation {
                code: "RUN_TANGENT_UNAVAILABLE".into(),
                origin: "provider/diffusivity".into(),
                message: "point (0.25, 0.5), t = 0.1, cell 3: no tangent for an \
                          analytic_provided law"
                    .into(),
            }
        );
        let flat = NumericError::from(FinitumError::InvalidRealization("bad".into()));
        assert_eq!(
            flat,
            NumericError::Operator {
                message: "invalid realization data: bad".into()
            }
        );
    }

    #[test]
    fn a_methodus_evaluation_error_round_trips_its_code_and_origin() {
        let numeric = NumericError::from(FinitumError::from(failure().at(
            Some(CellId(3)),
            &[0.25, 0.5],
            Some(0.1),
        )));
        let NumericError::Evaluation {
            code,
            origin,
            message,
        } = numeric
        else {
            panic!("expected an Evaluation error");
        };
        let back = InputEvaluationError::from_numeric(code, &origin, message);
        assert_eq!(back.code, "RUN_TANGENT_UNAVAILABLE");
        assert_eq!(back.origin, InputOrigin::Provider("diffusivity".into()));
        assert_eq!(back.location, None);
        assert_eq!(
            back.to_string(),
            "RUN_TANGENT_UNAVAILABLE at provider/diffusivity: point (0.25, 0.5), t = 0.1, \
             cell 3: no tangent for an analytic_provided law"
        );
        assert_eq!(
            InputOrigin::from_display("Heat.energy[0].k (stored table)"),
            InputOrigin::Table("Heat.energy[0].k".into())
        );
        assert_eq!(
            InputOrigin::from_display("boundary/walls"),
            InputOrigin::Slot("boundary/walls".into())
        );
    }

    #[test]
    fn every_origin_kind_displays_as_the_consumer_names_it() {
        assert_eq!(
            InputOrigin::Table("provider/diffusivity".into()).to_string(),
            "provider/diffusivity (stored table)"
        );
        assert_eq!(
            InputOrigin::ExpressionPath("Heat.energy[0].k".into()).to_string(),
            "Heat.energy[0].k"
        );
        assert_eq!(
            InputOrigin::Slot("boundary/walls".into()).to_string(),
            "boundary/walls"
        );
    }
}
