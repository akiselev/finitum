//! Concrete discretization and global operator realization.

mod block;
mod condensation;
mod constraint;
mod element;
mod error;
mod mapping;
mod mesh;
mod realization;
mod space;
mod system;
mod topology;

pub use block::{BlockLayout, FieldBlock};
pub use condensation::{CondensedLocalSystem, static_condense};
pub use constraint::{AffineConstraint, ConstraintSet, WeightedDof};
pub use element::{PreparedElement, QuadraturePoint};
pub use error::FinitumError;
pub use mapping::AffineMap;
pub use mesh::{Cell, CellId, Mesh, VertexId};
pub use realization::{
    AssembledOperator, DynamicExternalInput, ExternalInput, MatrixFreeOperator, PointActiveInput,
    PointEvaluation, RealizationPlan,
};
pub use space::{DofId, DofMap, ElementRestriction};
pub use system::SystemRealizationPlan;
pub use topology::{
    CompatibleDofMaps, ExactSequence, FacetId, FacetIncidence, FacetTopology, MeshFacet,
    OrientedFacetPair, OrientedRestriction, SignedIncidence,
};
