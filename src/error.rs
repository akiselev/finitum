use thiserror::Error;

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
}
