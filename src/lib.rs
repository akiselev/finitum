//! Concrete discretization and global operator realization.

mod constraint;
mod element;
mod error;
mod mesh;
mod operator;
mod space;

pub use constraint::{AffineConstraint, ConstraintSet, WeightedDof};
pub use element::{PreparedElement, QuadraturePoint};
pub use error::FinitumError;
pub use mesh::{Cell, CellId, Mesh, VertexId};
pub use operator::{AssembledOperator, CsrMatrix, RealizationKind};
pub use space::{DofId, DofMap, ElementRestriction};
