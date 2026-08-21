//! Concrete discretization and global operator realization.

mod adaptivity;
mod block;
mod condensation;
mod constraint;
mod element;
mod embedded;
mod error;
mod mapping;
mod mesh;
mod optimized;
mod realization;
mod space;
mod system;
mod topology;
mod transfer;

pub use adaptivity::{HangingNodeConstraint, VariableOrderSegmentElements};
pub use block::{BlockLayout, FieldBlock};
pub use condensation::{CondensedLocalSystem, static_condense};
pub use constraint::{AffineConstraint, ConstraintSet, WeightedDof};
pub use element::{PreparedElement, QuadraturePoint};
pub use embedded::{EmbeddedQuadraturePolicy, EmbeddedSegmentQuadrature};
pub use error::FinitumError;
pub use mapping::AffineMap;
pub use mesh::{Cell, CellId, Mesh, VertexId};
pub use optimized::{
    AcceleratorLayout, CellBatchLayout, ElementAssemblyOperator, PartialAssemblyOperator,
    TensorProductBasis, TensorProductEvaluation,
};
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
pub use transfer::{MortarInterface, NonmatchingTransfer};
