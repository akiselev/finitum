//! Concrete discretization and global operator realization.

mod adaptivity;
mod block;
mod cad_geometry;
mod condensation;
mod constraint;
mod element;
mod embedded;
mod error;
mod mapping;
mod mesh;
mod method;
mod optimized;
mod realization;
mod space;
mod system;
mod topology;
mod transfer;
mod verification;

pub use adaptivity::{HangingNodeConstraint, VariableOrderSegmentElements};
pub use block::{BlockLayout, FieldBlock};
pub use cad_geometry::{
    CadBoundaryAssociation, CadBoundaryCondition, CadCellAssociation, CadGeometryRealization,
    CadGeometrySource, CadNodeAssociation, CadParameterCoordinate, CadPrimalPlan,
};
pub use condensation::{CondensedLocalSystem, static_condense};
pub use constraint::{AffineConstraint, ConstraintSet, WeightedDof};
pub use element::{PreparedElement, QuadraturePoint};
pub use embedded::{EmbeddedQuadraturePolicy, EmbeddedSegmentQuadrature};
pub use error::FinitumError;
pub use mapping::AffineMap;
pub use mesh::{Cell, CellId, Mesh, VertexId};
pub use method::{
    BoundaryIntegralRealization, DiscreteOperator, FiniteDifferenceRealization, FiniteVolumeFace,
    FiniteVolumeRealization, MethodRealization, NetworkDaeRealization, ParticlePair,
    ParticleRealization, RadialPairPolynomial,
};
pub use optimized::{
    AcceleratorLayout, CellBatchLayout, ElementAssemblyOperator, PartialAssemblyOperator,
    TensorProductBasis, TensorProductEvaluation,
};
pub use realization::{
    AssembledOperator, DynamicExternalInput, ExternalInput, MatrixFreeOperator, PointActiveInput,
    PointEvaluation, REALIZATION_ARTIFACT_SCHEMA, RealizationArtifact, RealizationExternalInput,
    RealizationPlan,
};
pub use space::{DofId, DofMap, ElementRestriction};
pub use system::SystemRealizationPlan;
pub use topology::{
    CompatibleDofMaps, ExactSequence, FacetId, FacetIncidence, FacetTopology, MeshFacet,
    OrientedFacetPair, OrientedRestriction, SignedIncidence,
};
pub use transfer::{MortarInterface, NonmatchingTransfer};
pub use verification::{
    ConstraintWorkBody, ConstraintWorkReport, ExactSequenceCheckBody, ExactSequenceCheckReport,
    GlobalTransposeWorkBody, GlobalTransposeWorkReport, MeshRefinementCheckBody,
    MeshRefinementCheckReport, MeshRefinementLevel, MeshRefinementSample, PatchCheckBody,
    PatchCheckReport, RealizationAgreementBody, RealizationAgreementReport,
    TransferConservationBody, TransferConservationReport, VERIFICATION_REPORT_SCHEMA,
    ValidatedVerification, VerificationCheckKind, VerificationReportHeader, VerificationSubject,
    check_constraint_work, check_exact_sequence, check_global_transpose, check_mesh_refinement,
    check_nodal_patch, check_realization_agreement, check_transfer_conservation,
};
