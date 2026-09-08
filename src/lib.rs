//! Concrete discretization and global operator realization.

mod adaptivity;
mod block;
mod cad_geometry;
mod condensation;
mod constraint;
mod element;
mod embedded;
mod error;
mod infsup;
mod interface;
mod mapping;
mod mesh;
mod method;
mod mixed;
mod optimized;
mod profile;
mod realization;
mod sampler;
mod space;
mod system;
mod system_ids;
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
pub use element::{PreparedElement, QuadraturePoint, simplex_basis};
pub use embedded::{EmbeddedQuadraturePolicy, EmbeddedSegmentQuadrature};
pub use error::{FinitumError, InputEvaluationError, InputLocation, InputOrigin};
pub use infsup::{
    INF_SUP_DIMENSION_CAP, INF_SUP_REPORT_SCHEMA, InfSupConfig, InfSupEstimate, InfSupInstability,
    InfSupNorm, InfSupPairing, InfSupVerdict, estimate_inf_sup, require_inf_sup_stable,
};
pub use interface::{
    INTERFACE_REALIZATION_SCHEMA, InterfaceKernel, InterfaceMeasure, InterfaceOperand,
    InterfaceOperator, InterfaceSpace, InterfaceSwapKernelReceipt, InterfaceSwapReceipt,
    TraceEvaluation, TraceFieldSpec,
};
pub use mapping::AffineMap;
pub use mesh::{Cell, CellId, Mesh, VertexId};
pub use method::{
    BoundaryIntegralRealization, DiscreteOperator, FiniteDifferenceRealization, FiniteVolumeFace,
    FiniteVolumeRealization, MethodRealization, NetworkDaeRealization, ParticlePair,
    ParticleRealization, RadialPairPolynomial,
};
pub use mixed::{
    BlockCoupling, BlockEssentialValue, BlockNullspaceCandidate, BlockNullspaceMode, CouplingKind,
    FieldSpec, MixedOperator, MixedSpace, NullspaceModeKind, ReducedMixedOperator,
    essential_constraints_for_blocks,
};
pub use optimized::{
    AcceleratorLayout, CellBatchLayout, ElementAssemblyOperator, PartialAssemblyOperator,
    TensorProductBasis, TensorProductEvaluation,
};
pub use profile::{
    ComponentSelection, FieldSource, MeshProfile, MeshProvenance, PartitionReport, RegionMap,
    RegionTagId, RegionTags, TaggedMesh, check_boundary_partition, essential_constraints_from,
    essential_constraints_from_at, essential_constraints_from_selected,
    essential_constraints_from_selected_at, facet_membership_from, realize, refine_uniform,
};
pub use realization::{
    AssembledOperator, CapabilityElement, CoefficientLayout, ConstraintKind, DerivativeProduct,
    DistributedCoefficient, DynamicExternalInput, ExternalInput, ExternalSensitivityInput,
    GeometryParameterSensitivity, LinearizedOperator, MatrixFreeOperator, PointActiveInput,
    PointEvaluation, REALIZATION_ARTIFACT_SCHEMA, REALIZATION_CAPABILITY_SCHEMA,
    REALIZATION_PROPERTY_UNAVAILABLE, RealizationArtifact, RealizationCapability,
    RealizationExternalInput, RealizationPlan, RealizationReceipt, RepresentationKind,
    SYMMETRY_PROOF_DIMENSION_CAP, external_inputs_from, external_inputs_from_at,
};
pub use sampler::{
    ExteriorFacet, FIELD_SAMPLER_SCHEMA, FacetTrace, FieldSample, FieldSampler,
    FieldSamplerConventions, PhysicalQuadraturePoint, QUADRATURE_RULE_SCHEMA, QuadratureRule,
    QuadratureView, SampledFamily, cell_centroid, cell_measure, exterior_facet,
    simplex_monomial_moment,
};
pub use space::{
    DofId, DofMap, ElementRestriction, quadratic_simplex_dof_map, quadratic_simplex_node_points,
    vector_nodal_dof_map,
};
pub use system::{
    LinearizedSystemOperator, ReducedSystemOperator, SYSTEM_OPERATOR_DIGEST_SCHEMA,
    SYSTEM_REALIZATION_ARTIFACT_SCHEMA, SystemBlockOperator, SystemBlockReceipt,
    SystemConstitutiveInput, SystemDistributedCoefficient, SystemEssentialConstraintRequirement,
    SystemExternalInput, SystemFieldArtifact, SystemOperator, SystemPartialAssemblyOperator,
    SystemQuadrature, SystemRealizationArtifact, SystemRealizationExternalInput,
    SystemRealizationPlan, essential_constraints_from_system, essential_constraints_from_system_at,
    system_constitutive_from_sources,
};
pub use system_ids::{
    InstanceId, InstanceRecord, SYSTEM_ID_MAP_SCHEMA, SysRes, SysResId, SysVar, SysVarId,
    SystemIdMap,
};
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
    check_nodal_patch, check_realization_agreement, check_system_realization_agreement,
    check_transfer_conservation,
};
