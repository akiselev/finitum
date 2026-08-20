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
    #[error("cell {cell} references missing vertex {vertex}")]
    MissingVertex { cell: usize, vertex: usize },
    #[error("cell {cell} has {actual} vertices, expected {expected} for a simplex")]
    CellArity {
        cell: usize,
        actual: usize,
        expected: usize,
    },
    #[error("element restriction references missing degree of freedom {0}")]
    MissingDof(usize),
    #[error("constraint target {0} is outside the degree-of-freedom map")]
    InvalidConstraintTarget(usize),
    #[error("constraint dependency {0} is outside the degree-of-freedom map")]
    InvalidConstraintDependency(usize),
    #[error("constraint graph contains a cycle involving degree of freedom {0}")]
    ConstraintCycle(usize),
    #[error("prepared element table shape is inconsistent: {0}")]
    InvalidElementShape(String),
    #[error("CSR row offsets are inconsistent with matrix dimensions")]
    InvalidRowOffsets,
    #[error("CSR column/value storage lengths differ")]
    InvalidCsrStorage,
    #[error("CSR column {column} is outside a matrix with {columns} columns")]
    InvalidColumn { column: usize, columns: usize },
    #[error("operator input has length {actual}, expected {expected}")]
    InputLength { actual: usize, expected: usize },
}
