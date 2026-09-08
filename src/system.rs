use crate::CellBatchLayout;
use crate::InputEvaluationError;
use crate::element::{
    barycenter_quadrature, rt0_basis_count, rt0_reference_basis, simplex_basis,
    simplex_basis_count, simplex_quadrature,
};
use crate::mesh::CellId;
use crate::mixed::{
    BlockNullspaceCandidate, BlockVariableEssentialValue, essential_constraints_for_variables,
    solver_block_layout,
};
use crate::optimized::ElementAssemblyOperator;
use crate::profile::{
    RegionMap, TaggedMesh, evaluate_kernel_partial, evaluate_kernel_value, evaluate_table_slope,
    evaluate_table_value, named_coordinate_inputs,
};
use crate::realization::{
    BoundBundle, CapabilityElement, CellGeometry, ConstraintKind, DerivativeProduct,
    DistributedCoefficient, ExternalInput, FacetGeometry, PointActiveInput, PointBoundInput,
    PointEvaluation, RealizationCapability, RealizationExternalInput, RealizationReceipt,
    RepresentationKind, active_probe_inputs, active_values, apply_basis_adjoint, bind_kernels,
    build_capability, component_count, evaluate_basis_input, execute_jvp_values,
    execute_parameter_jvp_values, execute_primal_values, execute_vjp_values, gather_test_adjoint,
    locate_failure, point_parameter_cotangents, probe_direction_evaluation, property_unavailable,
    table_axis_point, tangent_unavailable, transpose_scatter_shape, validate_finite,
};
use crate::space::{
    DofMap, ElementRestriction, cell_constant_dof_map, quadratic_simplex_dof_map,
    vector_nodal_dof_map,
};
use crate::system_ids::{InstanceId, SysResId, SysVarId, SystemIdMap};
use crate::{
    AffineMap, BlockLayout, CompatibleDofMaps, ConstraintSet, ExactSequence, FacetId,
    FacetTopology, FieldBlock, FieldSource, FinitumError, Mesh, PreparedElement, QuadraturePoint,
};
use crate::{DofId, InputOrigin};
use methodus::{
    BlockLinearOperator, CsrMatrix, DaeOperator, Definiteness, EvaluationContext, LinearOperator,
    NonlinearOperator, NumericError, OperatorProperties, OperatorStructureHint, OperatorSymmetry,
    TransposableOperator,
};
use scientia::scientific::ValueShape;
use scientia::{
    DerivativeEvaluation, Digest, ElementFamilyRequirement, EssentialConstraintRequirement,
    EvaluationSite, FormSymmetry, InputSourceRequirement, IntegralOperatorFactorization, Linearity,
    NullspaceKind, OperatorStructure, OperatorSystem, OperatorSystemBlock, QFunctionInput,
    RegionId, SemanticMeasure, SemanticModel, SymbolId, TensorInputId, TensorInputRole,
    TraceMapping, derive_operator_structure_for_system,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// The shared cell quadrature rule a [`SystemRealizationPlan`] integrates every field with
/// (W7 package 7c, single compile path): chosen per plan at
/// [`SystemRealizationPlan::with_quadrature`], reported by [`SystemRealizationPlan::quadrature`]
/// and [`SystemOperator::quadrature`] so stored tables ([`SystemExternalInput`]) are sized per
/// point, and part of the plan's identity ([`SystemRealizationPlan::artifact_digest`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemQuadrature {
    /// One point per cell at the barycenter, exact for polynomial degree 1: the rule
    /// [`PreparedElement::linear_simplex`] (the single-model P1 default, C11.8) tabulates on, so
    /// a one-instance P1 system on it reproduces the single-model default plan bitwise.
    /// Admitted for order-0/1 Lagrange (H1/L2) fields only -- one point cannot integrate a P2
    /// stiffness or an RT0 mass -- and, exactly like the single-model default, it
    /// under-integrates the P1 mass matrix to a rank-one local block (C11.8).
    Barycenter,
    /// The richest rule this crate has for the mesh dimension: degree 4 on triangles (6
    /// points), degree 2 on tetrahedra (4 points), 3-point Gauss on segments -- what every
    /// system plan integrated with before the rule became a choice, and what a Taylor-Hood
    /// (P2/P1) or RT0/P0 system needs.
    Richest,
}

impl SystemQuadrature {
    /// The reference-simplex table of this rule in `dimension` (1..=3).
    pub fn table(self, dimension: usize) -> Result<Vec<QuadraturePoint>, FinitumError> {
        match self {
            Self::Barycenter => barycenter_quadrature(dimension),
            Self::Richest => simplex_quadrature(dimension),
        }
    }
}

/// One realized equation row of a realization group (W8 lane F-MI): the instance it belongs
/// to and its per-model `OperatorSystemBlock`, by index into that instance's `/1` artifact.
/// Rows are ordered by [`SysResId`]; for a one-instance plan the row index is the block index
/// of the model's `OperatorSystem`, which keeps every `(block index, integral, input)` key of
/// the one-instance operator (and its digest payload) exactly as before.
#[derive(Clone, Debug)]
struct RealizedRow {
    residual: SysResId,
    instance: InstanceId,
    /// Index into `instances[instance].blocks`.
    block: usize,
    /// Display path (§2.4): the bare equation name on a one-instance plan, `<instance>.<equation>`
    /// otherwise.
    path: String,
}

/// Digest-bound concrete ownership plan for an FC8 mixed operator system -- one realization
/// group (`sinbad/ARCHITECTURE.md` §8): one instance of one model ([`Self::with_quadrature`]),
/// or several instances of one or more models on one mesh composed through their same-mesh
/// binds ([`Self::composed`], W8 lane F-MI).
#[derive(Clone, Debug)]
pub struct SystemRealizationPlan {
    /// The first instance's per-model `/1` artifact (the only one of a one-instance plan).
    system: Arc<OperatorSystem>,
    /// Every instance's per-model `/1` artifact, by [`InstanceId`] index (§2.2: referenced
    /// verbatim, two instances of one model share one artifact).
    instances: Vec<Arc<OperatorSystem>>,
    /// Every equation row in [`SysResId`] order.
    rows: Vec<RealizedRow>,
    /// The `/2` operator, output kernels and bind compositions of a composed plan; `None` on
    /// a one-instance plan built by [`Self::with_quadrature`].
    composed: Option<Arc<ComposedSystem>>,
    mesh: Mesh,
    layout: BlockLayout,
    facets: FacetTopology,
    compatible_dofs: Option<CompatibleDofMaps>,
    exact_sequence: Option<ExactSequence>,
    /// SC-W1: the system-level ids of this realization group and their per-model origins;
    /// the layout's blocks are keyed by the same `SysVarId`s.
    system_ids: SystemIdMap,
    /// The shared cell quadrature rule every field is integrated with (part of
    /// `artifact_digest`).
    quadrature: SystemQuadrature,
    artifact_digest: Digest,
}

impl SystemRealizationPlan {
    /// [`Self::with_quadrature`] on [`SystemQuadrature::Richest`] -- the rule every system plan
    /// integrated with before the rule became a per-plan choice.
    pub fn new(
        system: OperatorSystem,
        mesh: Mesh,
        layout: BlockLayout,
    ) -> Result<Self, FinitumError> {
        Self::with_quadrature(system, mesh, layout, SystemQuadrature::Richest)
    }

    /// Validates the shape of the (one-instance) `system` over `mesh`/`layout` and fixes the
    /// shared cell quadrature rule every field is integrated with ([`Self::quadrature`]).
    /// [`SystemQuadrature::Barycenter`] is refused typed (`UnsupportedRealization`) when any
    /// block requires a field that is not an order-0/1 Lagrange (H1/L2) field.
    pub fn with_quadrature(
        system: OperatorSystem,
        mesh: Mesh,
        layout: BlockLayout,
        quadrature: SystemQuadrature,
    ) -> Result<Self, FinitumError> {
        for symbol in &system.field_order {
            if layout.block(*symbol).is_none() {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "mixed layout has no block for system field {symbol}"
                )));
            }
        }
        for block in &system.blocks {
            if block.form.source_semantic_digest != system.source_semantic_digest
                || block.factorization.receipt.source_form_digest != block.form.artifact_digest
                || block.kernels.source_factorization_digest != block.factorization.artifact_digest
            {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "system equation `{}` has a broken form/factorization/kernel receipt chain",
                    block.equation
                )));
            }
            for coordinate in &block.coordinates {
                if layout.block(coordinate.row).is_none()
                    || layout.block(coordinate.column).is_none()
                {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "system coordinate ({}, {}) is absent from the concrete layout",
                        coordinate.row, coordinate.column
                    )));
                }
            }
        }
        if quadrature == SystemQuadrature::Barycenter {
            for block in &system.blocks {
                for element in &block.requirements.elements {
                    let lagrange = matches!(
                        element.family,
                        ElementFamilyRequirement::H1 | ElementFamilyRequirement::L2
                    );
                    if !lagrange || element.polynomial_order > 1 {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "the barycenter quadrature rule is admitted for order-0/1 Lagrange \
                             (H1/L2) fields only; equation `{}` requires {:?}(order={}) for \
                             field {}",
                            block.equation,
                            element.family,
                            element.polynomial_order,
                            element.symbol
                        )));
                    }
                }
            }
        }
        validate_components(&system, &layout)?;
        let system_ids = SystemIdMap::one_instance(&system)?;
        for variable in system_ids.variables() {
            match layout.block_by_variable(variable.id) {
                Some(block) if block.symbol == variable.local => {}
                _ => {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "layout block for field {} is not keyed by its system variable {}",
                        variable.local, variable.id
                    )));
                }
            }
        }
        let facets = FacetTopology::from_mesh(&mesh)?;
        let uses_facets = system.blocks.iter().any(|block| {
            block
                .factorization
                .integrals
                .iter()
                .any(|integral| !matches!(integral.measure, SemanticMeasure::Cell { .. }))
        });
        if uses_facets && facets.facets().is_empty() {
            return Err(FinitumError::InvalidRealization(
                "facet operator system requires a nonempty facet topology".into(),
            ));
        }
        let uses_compatible = system.blocks.iter().any(|block| {
            block.requirements.elements.iter().any(|element| {
                matches!(
                    element.family,
                    ElementFamilyRequirement::Hcurl | ElementFamilyRequirement::Hdiv
                )
            })
        });
        let (compatible_dofs, exact_sequence) = if uses_compatible {
            (
                Some(CompatibleDofMaps::simplex(&mesh, &facets)?),
                Some(ExactSequence::simplex(&mesh, &facets)?),
            )
        } else {
            (None, None)
        };
        let artifact_digest = digest_plan(&system, &mesh, &layout, &facets, quadrature);
        let rows = system
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| RealizedRow {
                residual: SysResId(u32::try_from(index).expect("block count fits u32")),
                instance: InstanceId(0),
                block: index,
                path: block.equation.clone(),
            })
            .collect();
        let system = Arc::new(system);
        Ok(Self {
            instances: vec![system.clone()],
            system,
            rows,
            composed: None,
            mesh,
            layout,
            facets,
            compatible_dofs,
            exact_sequence,
            system_ids,
            quadrature,
            artifact_digest,
        })
    }

    /// W8 lane F-MI: the multi-instance realization group of a declared system
    /// (`sinbad/ARCHITECTURE.md` §2.6, §6, §8) over one mesh -- Scientia's
    /// `scientia-operator-system/2` `SystemOperator` with its per-instance `/1` artifacts,
    /// output kernels and bind compositions, exactly as `compile_system_operator` produced
    /// them. Rows are keyed by [`SysResId`] and fields by [`SysVarId`] from
    /// [`SystemIdMap::from_scientia`], so two instances of one model (colliding per-model
    /// `SymbolId`s) realize as distinct blocks; `layout` must be keyed by the same variables
    /// ([`BlockLayout::new_keyed`]). Every same-mesh `bind` (`BoundChain::Composed`) is
    /// realized at bind time ([`Self::bind_kernels_with_inputs`]): on the `kernel_input` path
    /// the producer's output kernel feeds the consumer kernel's operand through the Malleus
    /// [`scientia::BindComposition`], on the `provider_input` path the output's value (and
    /// tangent) reaches the consumer's constitutive closures as
    /// [`crate::PointEvaluation::bound`]. Cross-mesh binds (`Transferred`) are Krasis's and are
    /// not represented here. An algebraic loop among bound outputs (an output reading a bound
    /// input whose producer reads the first output) is refused typed. The plan's identity is
    /// `finitum-system-realization/3` (it covers the `/2` identity, the per-instance `/1`
    /// artifacts and the composition digests); one-instance plans built by
    /// [`Self::with_quadrature`] keep `/2` bitwise.
    pub fn composed(
        compilation: &scientia::SystemOperatorCompilation,
        mesh: Mesh,
        layout: BlockLayout,
        quadrature: SystemQuadrature,
    ) -> Result<Self, FinitumError> {
        let operator = &compilation.operator;
        if operator.schema != scientia::SYSTEM_OPERATOR_SCHEMA {
            return Err(FinitumError::ArtifactMismatch(format!(
                "composed realization expects a `{}` artifact, got `{}`",
                scientia::SYSTEM_OPERATOR_SCHEMA,
                operator.schema
            )));
        }
        let system_ids = SystemIdMap::from_scientia(operator)?;
        // Per-instance `/1` artifacts, by instance index, checked against the `/2` receipts.
        let mut instances = Vec::with_capacity(operator.instances.len());
        for (index, record) in operator.instances.iter().enumerate() {
            if record.instance.index() != index {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "system operator instance records are not dense (record {index} is instance \
                     {})",
                    record.instance
                )));
            }
            let (_, model_system) = compilation
                .model_systems
                .iter()
                .find(|(instance, _)| *instance == record.instance)
                .ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "system operator compilation carries no `/1` artifact for instance {} \
                         (`{}`)",
                        record.instance, record.model_name
                    ))
                })?;
            let expected = &system_ids.instances()[index].artifact_digest;
            if &model_system.artifact_digest != expected {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "instance {} (`{}`) `/1` artifact digest {} does not match the `/2` receipt {}",
                    record.instance,
                    record.model_name,
                    model_system.artifact_digest.hex,
                    expected.hex
                )));
            }
            if model_system.source_semantic_digest != record.semantic_digest {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "instance {} (`{}`) `/1` artifact was compiled from semantic digest {}, the \
                     `/2` record names {}",
                    record.instance,
                    record.model_name,
                    model_system.source_semantic_digest.hex,
                    record.semantic_digest.hex
                )));
            }
            instances.push(Arc::new(model_system.clone()));
        }
        if instances.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "a composed realization needs at least one instance".into(),
            ));
        }
        // Rows in `SysResId` order, each the per-model block its `/2` residual references.
        let mut rows = Vec::with_capacity(operator.residuals.len());
        for (index, residual) in operator.residuals.iter().enumerate() {
            if residual.id.index() != index {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "system operator residuals are not dense (entry {index} is {})",
                    residual.id
                )));
            }
            let scientia::ResidualOrigin::Equation { instance, name, .. } = &residual.origin;
            let instance_id = InstanceId(instance.0);
            let system = instances.get(instance.index()).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "residual {} names instance {instance} which the operator does not declare",
                    residual.id
                ))
            })?;
            let block = system
                .blocks
                .iter()
                .position(|block| &block.equation == name)
                .ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "residual {} names equation `{name}` which instance {instance}'s `/1` \
                         artifact does not carry",
                        residual.id
                    ))
                })?;
            let block_digest = scientia::block_digest(&system.blocks[block]);
            if block_digest != residual.block || system.artifact_digest != residual.model_system {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "residual {} (`{name}`) references block {} of artifact {}, the instance's \
                     `/1` artifact carries block {} of {}",
                    residual.id,
                    residual.block.hex,
                    residual.model_system.hex,
                    block_digest.hex,
                    system.artifact_digest.hex
                )));
            }
            if system.blocks[block].row != residual.row {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "residual {} row symbol {} disagrees with its block's row {}",
                    residual.id, residual.row, system.blocks[block].row
                )));
            }
            let residual_id = SysResId(residual.id.0);
            let path = system_ids
                .residual_path(residual_id)
                .expect("from_scientia recorded every residual");
            rows.push(RealizedRow {
                residual: residual_id,
                instance: instance_id,
                block,
                path,
            });
        }
        if rows.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "a composed realization needs at least one residual row".into(),
            ));
        }
        // The layout is keyed by the group's variables, each carrying its per-model symbol.
        for variable in system_ids.variables() {
            match layout.block_by_variable(variable.id) {
                Some(block) if block.symbol == variable.local => {}
                _ => {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "layout block for field {} of {} is not keyed by its system variable {}",
                        variable.local, variable.instance, variable.id
                    )));
                }
            }
        }
        if layout.blocks().len() != system_ids.variables().len() {
            return Err(FinitumError::ArtifactMismatch(format!(
                "layout has {} blocks, the system has {} variables",
                layout.blocks().len(),
                system_ids.variables().len()
            )));
        }
        // Per-instance shape checks, exactly `with_quadrature`'s, through the instance's ids.
        for (instance_index, system) in instances.iter().enumerate() {
            let instance = InstanceId(u32::try_from(instance_index).expect("fits u32"));
            let variable_of = |symbol: SymbolId| system_ids.variable(instance, symbol);
            for symbol in &system.field_order {
                if variable_of(*symbol).is_none() {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "system has no variable for field {symbol} of {instance}"
                    )));
                }
            }
            for block in &system.blocks {
                if block.form.source_semantic_digest != system.source_semantic_digest
                    || block.factorization.receipt.source_form_digest != block.form.artifact_digest
                    || block.kernels.source_factorization_digest
                        != block.factorization.artifact_digest
                {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "equation `{}` of {instance} has a broken form/factorization/kernel \
                         receipt chain",
                        block.equation
                    )));
                }
                for coordinate in &block.coordinates {
                    if variable_of(coordinate.row).is_none()
                        || variable_of(coordinate.column).is_none()
                    {
                        return Err(FinitumError::ArtifactMismatch(format!(
                            "coordinate ({}, {}) of {instance} is absent from the system's \
                             variables",
                            coordinate.row, coordinate.column
                        )));
                    }
                }
            }
            if quadrature == SystemQuadrature::Barycenter {
                for block in &system.blocks {
                    for element in &block.requirements.elements {
                        let lagrange = matches!(
                            element.family,
                            ElementFamilyRequirement::H1 | ElementFamilyRequirement::L2
                        );
                        if !lagrange || element.polynomial_order > 1 {
                            return Err(FinitumError::UnsupportedRealization(format!(
                                "the barycenter quadrature rule is admitted for order-0/1 \
                                 Lagrange (H1/L2) fields only; equation `{}` of {instance} \
                                 requires {:?}(order={}) for field {}",
                                block.equation,
                                element.family,
                                element.polynomial_order,
                                element.symbol
                            )));
                        }
                    }
                }
            }
            validate_components_keyed(system, &layout, &variable_of)?;
        }
        let facets = FacetTopology::from_mesh(&mesh)?;
        let all_blocks = || instances.iter().flat_map(|system| system.blocks.iter());
        let uses_facets = all_blocks().any(|block| {
            block
                .factorization
                .integrals
                .iter()
                .any(|integral| !matches!(integral.measure, SemanticMeasure::Cell { .. }))
        });
        if uses_facets && facets.facets().is_empty() {
            return Err(FinitumError::InvalidRealization(
                "facet operator system requires a nonempty facet topology".into(),
            ));
        }
        let uses_compatible = all_blocks().any(|block| {
            block.requirements.elements.iter().any(|element| {
                matches!(
                    element.family,
                    ElementFamilyRequirement::Hcurl | ElementFamilyRequirement::Hdiv
                )
            })
        });
        let (compatible_dofs, exact_sequence) = if uses_compatible {
            (
                Some(CompatibleDofMaps::simplex(&mesh, &facets)?),
                Some(ExactSequence::simplex(&mesh, &facets)?),
            )
        } else {
            (None, None)
        };
        let composed = ComposedSystem::new(compilation, &system_ids, &instances, &rows)?;
        let artifact_digest =
            digest_plan_composed(&composed, &system_ids, &mesh, &layout, &facets, quadrature);
        Ok(Self {
            system: instances[0].clone(),
            instances,
            rows,
            composed: Some(Arc::new(composed)),
            mesh,
            layout,
            facets,
            compatible_dofs,
            exact_sequence,
            system_ids,
            quadrature,
            artifact_digest,
        })
    }

    /// The first instance's per-model `scientia-operator-system/1` artifact -- the whole
    /// system of a one-instance plan; on a composed plan use [`Self::instance_system`] for the
    /// others and [`Self::system_ids`] for the rows.
    pub fn system(&self) -> &OperatorSystem {
        &self.system
    }

    /// The per-model `/1` artifact an instance of this group reuses verbatim (§2.2).
    pub fn instance_system(&self, instance: InstanceId) -> Option<&OperatorSystem> {
        self.instances.get(instance.0 as usize).map(Arc::as_ref)
    }

    /// The `scientia-operator-system/2` artifact a composed plan was built from
    /// ([`Self::composed`]); `None` on a one-instance plan.
    pub fn system_operator(&self) -> Option<&scientia::SystemOperator> {
        self.composed.as_ref().map(|composed| &composed.operator)
    }

    pub(crate) fn has_bound_input(&self, instance: InstanceId, symbol: SymbolId) -> bool {
        self.composed.as_ref().is_some_and(|composed| {
            composed
                .binds
                .iter()
                .any(|bind| bind.consumer == instance && bind.consumer_symbol == symbol)
        })
    }

    /// The realized rows in `SysResId` order as `(row index, row, per-model block)`.
    fn rows(&self) -> impl Iterator<Item = (usize, &RealizedRow, &OperatorSystemBlock)> + '_ {
        self.rows.iter().enumerate().map(move |(index, row)| {
            (
                index,
                row,
                &self.instances[row.instance.0 as usize].blocks[row.block],
            )
        })
    }

    /// One realized row by index.
    fn row(&self, index: usize) -> (&RealizedRow, &OperatorSystemBlock) {
        let row = &self.rows[index];
        (
            row,
            &self.instances[row.instance.0 as usize].blocks[row.block],
        )
    }

    /// The row index of a system residual.
    fn row_index(&self, residual: SysResId) -> Option<usize> {
        self.rows.iter().position(|row| row.residual == residual)
    }

    /// SC-W1: the system-level ids (`SysVarId`/`SysResId`) of this realization group with
    /// their per-model origins.
    pub fn system_ids(&self) -> &SystemIdMap {
        &self.system_ids
    }

    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    pub fn facets(&self) -> &FacetTopology {
        &self.facets
    }

    pub fn compatible_dofs(&self) -> Option<&CompatibleDofMaps> {
        self.compatible_dofs.as_ref()
    }

    pub fn exact_sequence(&self) -> Option<&ExactSequence> {
        self.exact_sequence.as_ref()
    }

    pub fn artifact_digest(&self) -> &Digest {
        &self.artifact_digest
    }

    /// The shared cell quadrature table [`SystemRealizationPlan::bind_kernels`] integrates every
    /// field with (see [`SystemOperator::quadrature`]) -- available before binding so stored
    /// tables ([`SystemExternalInput`]) can be laid out over it.
    pub fn quadrature(&self) -> Result<Vec<QuadraturePoint>, FinitumError> {
        self.quadrature.table(self.mesh.dimension())
    }

    /// The rule [`Self::quadrature`] tabulates, fixed at [`Self::with_quadrature`].
    pub fn quadrature_rule(&self) -> SystemQuadrature {
        self.quadrature
    }
}

fn validate_components(system: &OperatorSystem, layout: &BlockLayout) -> Result<(), FinitumError> {
    for block in &system.blocks {
        for space in &block.requirements.spaces {
            let Some(concrete) = layout.block(space.symbol) else {
                continue;
            };
            let expected = match space.space.family {
                // A compatible-element (Hdiv/Hcurl) space owns exactly one scalar DOF per
                // topological entity (one facet/edge orientation flux/circulation value); the
                // field's vector-ness lives in the Piola-mapped basis function, never in a
                // per-component DOF split the way nodal Lagrange spaces use -- see
                // `build_field_elements`'s Hdiv(order=0) branch.
                scientia::scientific::SpaceFamily::HDiv
                | scientia::scientific::SpaceFamily::HCurl => 1,
                _ => match space.value_shape {
                    ValueShape::Scalar => 1,
                    ValueShape::Vector(extent) => usize::from(extent),
                    ValueShape::Tensor { rows, cols } => usize::from(rows) * usize::from(cols),
                    ValueShape::SymmetricTensor(extent) => {
                        usize::from(extent) * usize::from(extent)
                    }
                },
            };
            if concrete.component_count != expected {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "field {} has {} concrete components, typed space requires {expected}",
                    space.symbol, concrete.component_count
                )));
            }
        }
    }
    Ok(())
}

fn digest_plan(
    system: &OperatorSystem,
    mesh: &Mesh,
    layout: &BlockLayout,
    facets: &FacetTopology,
    quadrature: SystemQuadrature,
) -> Digest {
    #[derive(Serialize)]
    struct BlockIdentity {
        symbol: u32,
        entity_count: usize,
        component_count: usize,
        offset: usize,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        system: &'a Digest,
        dimension: usize,
        vertices: &'a [Vec<f64>],
        cells: Vec<Vec<usize>>,
        blocks: Vec<BlockIdentity>,
        facet_count: usize,
        quadrature: SystemQuadrature,
    }
    let bytes = serde_json::to_vec(&Payload {
        schema: "finitum-system-realization/2",
        system: &system.artifact_digest,
        dimension: mesh.dimension(),
        vertices: mesh.vertices(),
        cells: mesh
            .cells()
            .iter()
            .map(|cell| cell.vertices.iter().map(|vertex| vertex.0).collect())
            .collect(),
        blocks: layout
            .blocks()
            .iter()
            .map(|block| BlockIdentity {
                symbol: block.symbol.0,
                entity_count: block.entity_count,
                component_count: block.component_count,
                offset: block.offset,
            })
            .collect(),
        facet_count: facets.facets().len(),
        quadrature,
    })
    .expect("system realization identity is serializable");
    Digest {
        algorithm: "blake3".into(),
        hex: blake3::hash(&bytes).to_hex().to_string(),
    }
}

/// [`validate_components`] through an instance's `symbol -> SysVarId` map (composed plans).
fn validate_components_keyed(
    system: &OperatorSystem,
    layout: &BlockLayout,
    variable_of: &dyn Fn(SymbolId) -> Option<SysVarId>,
) -> Result<(), FinitumError> {
    for block in &system.blocks {
        for space in &block.requirements.spaces {
            let Some(concrete) =
                variable_of(space.symbol).and_then(|v| layout.block_by_variable(v))
            else {
                continue;
            };
            let expected = match space.space.family {
                scientia::scientific::SpaceFamily::HDiv
                | scientia::scientific::SpaceFamily::HCurl => 1,
                _ => match space.value_shape {
                    ValueShape::Scalar => 1,
                    ValueShape::Vector(extent) => usize::from(extent),
                    ValueShape::Tensor { rows, cols } => usize::from(rows) * usize::from(cols),
                    ValueShape::SymmetricTensor(extent) => {
                        usize::from(extent) * usize::from(extent)
                    }
                },
            };
            if concrete.component_count != expected {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "field {} ({}) has {} concrete components, typed space requires {expected}",
                    space.symbol, concrete.variable, concrete.component_count
                )));
            }
        }
    }
    Ok(())
}

/// Which way a same-mesh bind reaches its consumer (`scientia::ComposedPath`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindPath {
    /// The consumer's residual kernels read the bound input as an external operand: the
    /// producer's output kernel feeds it through the Malleus `BindComposition`s.
    KernelInput,
    /// The consumer reads the bound input only inside its properties / constitutive laws: the
    /// producer's output value and tangent reach the consumer's closures as
    /// [`crate::PointEvaluation::bound`].
    ProviderInput,
}

/// One same-mesh bind of a composed plan, resolved to the group's ids.
#[derive(Clone, Debug)]
struct RealizedBind {
    /// Index into `ScientificSystem::binds`.
    index: usize,
    consumer: InstanceId,
    /// The consumer's local input-field symbol the bind closes.
    consumer_symbol: SymbolId,
    consumer_slot: String,
    producer: InstanceId,
    output: scientia::OutputId,
    /// Display path `<producer>.<output>`.
    output_path: String,
    path: BindPath,
    /// Indices into `ComposedSystem::compositions` (kernel-input path).
    compositions: Vec<usize>,
}

/// The `/2` operator artifact and the per-bind payloads a composed plan realizes (W8 lane
/// F-MI): Scientia's `SystemOperator`, every output kernel chain, every Malleus
/// `BindComposition`, and the binds in dependency order (a producer output that reads a
/// bound input of its own instance comes after the bind supplying it).
#[derive(Debug)]
struct ComposedSystem {
    operator: scientia::SystemOperator,
    /// By `OutputId` index.
    outputs: Vec<scientia::OutputKernels>,
    compositions: Vec<scientia::BindComposition>,
    /// Dependency order: every bind after the binds its producer's output kernel reads.
    binds: Vec<RealizedBind>,
}

impl ComposedSystem {
    fn new(
        compilation: &scientia::SystemOperatorCompilation,
        system_ids: &SystemIdMap,
        instances: &[Arc<OperatorSystem>],
        rows: &[RealizedRow],
    ) -> Result<Self, FinitumError> {
        let system = &compilation.system;
        let operator = &compilation.operator;
        let mut outputs = Vec::with_capacity(compilation.output_kernels.len());
        for (index, kernels) in compilation.output_kernels.iter().enumerate() {
            let declared = system.outputs.get(index).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "output kernels entry {index} has no declared system output"
                ))
            })?;
            if kernels.output.index() != index || declared.id.index() != index {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "system outputs are not dense (entry {index} is output {})",
                    kernels.output
                )));
            }
            let record = operator
                .outputs
                .iter()
                .find(|record| record.output == kernels.output)
                .ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "the `/2` operator records no kernel chain for output {}",
                        kernels.output
                    ))
                })?;
            if record.form != kernels.form.artifact_digest
                || record.requirements != kernels.requirements.artifact_digest
                || record.factorization != kernels.factorization.artifact_digest
                || record.kernels != kernels.kernels.artifact_digest
                || kernels.factorization.receipt.source_form_digest != kernels.form.artifact_digest
                || kernels.kernels.source_factorization_digest
                    != kernels.factorization.artifact_digest
            {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "output `{}` has a broken form/factorization/kernel receipt chain",
                    declared.name
                )));
            }
            let [integral] = kernels.factorization.integrals.as_slice() else {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "output `{}` factorizes into {} integrals; an output is one point function",
                    declared.name,
                    kernels.factorization.integrals.len()
                )));
            };
            if !matches!(integral.measure, SemanticMeasure::Cell { .. })
                || integral.primal.outputs.len() != 1
                || kernels.kernels.bundles.len() != 1
            {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "output `{}` is not one cell-measure point function with one kernel bundle",
                    declared.name
                )));
            }
            outputs.push(kernels.clone());
        }
        let compositions = compilation.compositions.clone();
        let mut binds = Vec::with_capacity(system.binds.len());
        for (index, bind) in system.binds.iter().enumerate() {
            let scientia::BoundChain::Composed = bind.chain;
            let declared = system.outputs.get(bind.producer.index()).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "bind `{}` names output {} which the system does not declare",
                    bind.consumer_slot, bind.producer
                ))
            })?;
            if outputs.len() <= bind.producer.index() {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "bind `{}` names output {} which has no kernel chain",
                    bind.consumer_slot, bind.producer
                )));
            }
            let consumer = InstanceId(bind.consumer.0);
            let producer = InstanceId(declared.instance.0);
            if instances.get(consumer.0 as usize).is_none()
                || instances.get(producer.0 as usize).is_none()
            {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "bind `{}` names an instance the operator does not declare",
                    bind.consumer_slot
                )));
            }
            let bind_compositions = compositions
                .iter()
                .enumerate()
                .filter(|(_, composition)| composition.bind == index)
                .map(|(position, composition)| {
                    let row = rows
                        .iter()
                        .position(|row| row.residual == SysResId(composition.row.0))
                        .ok_or_else(|| {
                            FinitumError::ArtifactMismatch(format!(
                                "bind `{}` composition names residual {} which the plan has no \
                                 row for",
                                bind.consumer_slot, composition.row
                            ))
                        })?;
                    if rows[row].instance != consumer {
                        return Err(FinitumError::ArtifactMismatch(format!(
                            "bind `{}` composition names residual {} of another instance",
                            bind.consumer_slot, composition.row
                        )));
                    }
                    Ok(position)
                })
                .collect::<Result<Vec<_>, FinitumError>>()?;
            let path = if bind_compositions.is_empty() {
                BindPath::ProviderInput
            } else {
                BindPath::KernelInput
            };
            for block in &operator.blocks {
                let scientia::BlockConstruction::Composed {
                    bind: block_bind,
                    path: block_path,
                    ..
                } = &block.construction
                else {
                    continue;
                };
                if *block_bind != index {
                    continue;
                }
                let agrees = match block_path {
                    scientia::ComposedPath::KernelInput { compositions, .. } => {
                        path == BindPath::KernelInput
                            && compositions.len() == bind_compositions.len()
                            && bind_compositions.iter().all(|&i| {
                                compositions.contains(&compilation.compositions[i].digest)
                            })
                    }
                    scientia::ComposedPath::ProviderInput { .. } => path == BindPath::ProviderInput,
                };
                if !agrees {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "bind `{}` is recorded as a {block_path:?} block but the compilation \
                         carries {} compositions for it",
                        bind.consumer_slot,
                        bind_compositions.len()
                    )));
                }
            }
            let producer_name = &system_ids.instances()[producer.0 as usize].name;
            binds.push(RealizedBind {
                index,
                consumer,
                consumer_symbol: bind.consumer_symbol,
                consumer_slot: bind.consumer_slot.clone(),
                producer,
                output: bind.producer,
                output_path: format!("{producer_name}.{}", declared.name),
                path,
                compositions: bind_compositions,
            });
        }
        // Dependency order: bind `i`'s producer output kernel reads every provider-path bound
        // input of the producer instance (inside its properties) and every kernel-path bound
        // input its own integral names as a non-basis operand.
        let reads = |i: usize, j: usize| -> bool {
            let (this, other) = (&binds[i], &binds[j]);
            if other.consumer != this.producer {
                return false;
            }
            match other.path {
                BindPath::ProviderInput => outputs[this.output.index()].factorization.integrals[0]
                    .primal
                    .inputs
                    .iter()
                    .any(|input| input.source != InputSourceRequirement::Basis),
                BindPath::KernelInput => outputs[this.output.index()].factorization.integrals[0]
                    .primal
                    .inputs
                    .iter()
                    .any(|input| {
                        input.source != InputSourceRequirement::Basis
                            && input.binding.symbol == other.consumer_symbol
                    }),
            }
        };
        let mut order = Vec::with_capacity(binds.len());
        let mut placed = vec![false; binds.len()];
        while order.len() < binds.len() {
            let next = (0..binds.len())
                .find(|&i| !placed[i] && (0..binds.len()).all(|j| placed[j] || !reads(i, j)));
            match next {
                Some(i) => {
                    placed[i] = true;
                    order.push(i);
                }
                None => {
                    let cycle = binds
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !placed[*i])
                        .map(|(_, bind)| {
                            format!("`{}` <- {}", bind.consumer_slot, bind.output_path)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "the bound outputs form an algebraic loop at the quadrature point \
                         ({cycle}); a same-mesh bind chain must be acyclic among outputs"
                    )));
                }
            }
        }
        let binds = order.into_iter().map(|i| binds[i].clone()).collect();
        Ok(Self {
            operator: operator.clone(),
            outputs,
            compositions,
            binds,
        })
    }
}

/// Schema of a composed plan's [`SystemRealizationPlan::artifact_digest`] payload (W8 lane
/// F-MI): the `/2` operator identity, every instance's `/1` artifact, the layout keyed by
/// system variable, and every bind's composition digests. One-instance plans built by
/// [`SystemRealizationPlan::with_quadrature`] keep `finitum-system-realization/2` bitwise.
pub const SYSTEM_REALIZATION_COMPOSED_DIGEST_SCHEMA: &str = "finitum-system-realization/3";

fn digest_plan_composed(
    composed: &ComposedSystem,
    system_ids: &SystemIdMap,
    mesh: &Mesh,
    layout: &BlockLayout,
    facets: &FacetTopology,
    quadrature: SystemQuadrature,
) -> Digest {
    #[derive(Serialize)]
    struct InstanceIdentity<'a> {
        name: &'a str,
        model: &'a str,
        artifact: &'a Digest,
    }
    #[derive(Serialize)]
    struct BlockIdentity {
        variable: u32,
        symbol: u32,
        entity_count: usize,
        component_count: usize,
        offset: usize,
    }
    #[derive(Serialize)]
    struct BindIdentity<'a> {
        consumer_slot: &'a str,
        output: &'a str,
        path: BindPath,
        compositions: Vec<&'a Digest>,
        jvp_compositions: Vec<&'a Digest>,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        system: &'a Digest,
        instances: Vec<InstanceIdentity<'a>>,
        dimension: usize,
        vertices: &'a [Vec<f64>],
        cells: Vec<Vec<usize>>,
        blocks: Vec<BlockIdentity>,
        facet_count: usize,
        quadrature: SystemQuadrature,
        binds: Vec<BindIdentity<'a>>,
    }
    let bytes = serde_json::to_vec(&Payload {
        schema: SYSTEM_REALIZATION_COMPOSED_DIGEST_SCHEMA,
        system: &composed.operator.identity,
        instances: system_ids
            .instances()
            .iter()
            .map(|record| InstanceIdentity {
                name: &record.name,
                model: &record.model,
                artifact: &record.artifact_digest,
            })
            .collect(),
        dimension: mesh.dimension(),
        vertices: mesh.vertices(),
        cells: mesh
            .cells()
            .iter()
            .map(|cell| cell.vertices.iter().map(|vertex| vertex.0).collect())
            .collect(),
        blocks: layout
            .blocks()
            .iter()
            .map(|block| BlockIdentity {
                variable: block.variable.0,
                symbol: block.symbol.0,
                entity_count: block.entity_count,
                component_count: block.component_count,
                offset: block.offset,
            })
            .collect(),
        facet_count: facets.facets().len(),
        quadrature,
        binds: composed
            .binds
            .iter()
            .map(|bind| BindIdentity {
                consumer_slot: &bind.consumer_slot,
                output: &bind.output_path,
                path: bind.path,
                compositions: bind
                    .compositions
                    .iter()
                    .map(|&i| &composed.compositions[i].digest)
                    .collect(),
                jvp_compositions: bind
                    .compositions
                    .iter()
                    .map(|&i| &composed.compositions[i].jvp_digest)
                    .collect(),
            })
            .collect(),
    })
    .expect("composed system realization identity is serializable");
    Digest::blake3(&bytes)
}

// ---------------------------------------------------------------------------------------------
// SV2-B4 continuation: executable Scientia-form/Malleus-kernel-driven mixed realization.
//
// `SystemRealizationPlan::bind_kernels` turns a shape-validated plan into a [`SystemOperator`]:
// every system field gets a shared-quadrature Lagrange discretization (LAGRANGE/Taylor-Hood
// scope: H1 or L2, order 1 or 2, scalar or dimension-vector -- see [`build_field_elements`] for
// why an L2-typed field is admitted), every block's FC4/FC5 artifact triple is bound into real
// executable kernels via [`bind_kernels`] (reused unchanged from [`crate::realization`]), and
// Scientia's C5.4 `OperatorStructure` is derived once and cross-checked against the realized
// block coordinates. [`SystemOperator::apply_action`] composes every block's present `(row,
// column)` coordinates cell-by-cell into one global matrix-free action, gathering/scattering
// through `BlockLayout` -- the executable analogue of `MixedOperator::apply_action`, driven by
// the real bound kernels instead of hand-written local-matrix builders.
// ---------------------------------------------------------------------------------------------

/// One system field's concrete discretization: either a shared-quadrature Lagrange table
/// (H1/L2 order 1/2, and L2(order=0) piecewise-constant), or an RT0 (Hdiv(order=0))
/// compatible-element field, which has no single reference basis table shared across cells --
/// its physical basis is Piola-pushed per cell (see [`FieldKind::Hdiv0`]).
#[derive(Clone, Debug)]
enum FieldKind {
    Lagrange(PreparedElement),
    /// RT0 (lowest-order Raviart-Thomas): one scalar (oriented facet-flux) DOF per mesh facet,
    /// realized through [`crate::element::rt0_reference_basis`] and
    /// `crate::mapping::AffineMap`'s contravariant Piola map (both reused unchanged from FC8).
    /// `orientations[cell][local_facet]` mirrors [`CompatibleDofMaps::hdiv`]'s own per-cell sign
    /// table exactly (in fact copied from it at construction) -- needed alongside `dofs` for
    /// every per-cell gather/scatter, since (unlike a nodal Lagrange restriction) a physical RT0
    /// coefficient is `orientation * global_dof_value`, not the raw global value.
    Hdiv0 {
        orientations: Vec<Vec<i8>>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct FieldElement {
    kind: FieldKind,
    pub(crate) dofs: DofMap,
}

/// The concrete DOF-layout component count a system field's typed requirement implies:
/// `1` for a compatible-element (Hdiv/Hcurl) field regardless of its vector-valued
/// `value_shape` (one scalar DOF per topological entity; the vector-ness lives in the
/// Piola-mapped basis function -- see [`FieldKind::Hdiv0`]), otherwise the scalar/dimension-
/// vector extent nodal Lagrange fields already use. Shared by [`validate_components`] (which
/// runs at [`SystemRealizationPlan::new`] time, before any element is built) and
/// [`build_field_elements`], so both apply the same rule.
fn expected_field_components(
    family: ElementFamilyRequirement,
    value_shape: &ValueShape,
    dimension: usize,
) -> Result<usize, FinitumError> {
    if matches!(
        family,
        ElementFamilyRequirement::Hdiv | ElementFamilyRequirement::Hcurl
    ) {
        return Ok(1);
    }
    match value_shape {
        ValueShape::Scalar => Ok(1),
        ValueShape::Vector(extent) if *extent as usize == dimension => Ok(dimension),
        other => Err(FinitumError::UnsupportedRealization(format!(
            "system field must be scalar or dimension-{dimension}-vector valued, got {other:?}"
        ))),
    }
}

/// Builds one [`FieldElement`] per system field.
///
/// Every Lagrange field's basis table is tabulated at the *same* shared quadrature points
/// (`simplex_quadrature`, mirroring `crate::mixed`'s own cross-block evaluation convention), so
/// a quadrature-point index means the same physical point for every field -- required for
/// [`point_inputs_system`]/[`point_directions_system`] to gather several fields' basis inputs
/// within one integral. An RT0 field has no such table (its physical basis is cell-specific,
/// Piola-mapped -- see [`FieldKind::Hdiv0`]), but shares the same quadrature-point *index*
/// convention: `point_inputs_system`/`point_directions_system` resolve `quadrature[point]`'s own
/// reference coordinates directly rather than through a per-field table.
///
/// Admits `ElementFamilyRequirement::H1` and `ElementFamilyRequirement::L2` fields of order 1 or
/// 2, scalar or dimension-vector valued -- both realized through the *same* continuous nodal
/// Lagrange basis. This is a deliberate widening relative to the single-field
/// `RealizationPlan`'s H1-only admission: Scientia's `PullbackRequirement` (`H1Composition` vs.
/// `L2Density`) is never consulted anywhere downstream of FC4 in this crate (grep confirms
/// neither `crate::realization` nor `crate::mixed` reads it), so the concrete pullback that
/// actually executes is always plain composition -- correct for a continuous nodal basis
/// regardless of the field's own *minimum* required regularity. A field typed `L2(order=k)` by
/// its variational form (its weak-form usage requires no more than L2, e.g. a Taylor-Hood
/// pressure after integration by parts removes its derivative from the momentum equation) is
/// mathematically well realized by a strictly smoother continuous-P`k` choice, since H1 subset
/// L2.
///
/// Additionally admits `ElementFamilyRequirement::L2` order 0 (piecewise-constant P0, one DOF
/// per cell, [`cell_constant_dof_map`]) and `ElementFamilyRequirement::Hdiv` order 0 (RT0, one
/// oriented flux DOF per facet, reusing FC8's already-computed `compatible_dofs.hdiv` --
/// `compatible` is `None` only when no block required a compatible element, in which case no
/// field can legitimately reach the `Hdiv` arm below). `ElementFamilyRequirement::Hcurl`/
/// `DiscontinuousGalerkin`, any other polynomial order, and any Hdiv order other than 0, remain
/// refused typed.
pub(crate) fn build_field_elements(
    plan: &SystemRealizationPlan,
    quadrature: &[QuadraturePoint],
) -> Result<BTreeMap<SysVarId, FieldElement>, FinitumError> {
    let mesh = &plan.mesh;
    let layout = &plan.layout;
    let compatible = plan.compatible_dofs.as_ref();
    let ids = &plan.system_ids;
    let mut fields = BTreeMap::new();
    for (instance_index, system) in plan.instances.iter().enumerate() {
        let instance = InstanceId(u32::try_from(instance_index).expect("fits u32"));
        for &local in &system.field_order {
            let variable = ids.variable(instance, local).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "system has no variable for field {local} of {instance}"
                ))
            })?;
            // Display: the bare symbol on a one-instance plan (every existing message), the
            // system variable with its instance otherwise.
            let symbol = if plan.instances.len() == 1 {
                local.to_string()
            } else {
                format!("{local} ({variable} of {instance})")
            };
            let mut found: Option<&scientia::ElementRequirement> = None;
            for block in &system.blocks {
                for requirement in &block.requirements.elements {
                    if requirement.symbol != local {
                        continue;
                    }
                    match found {
                        None => found = Some(requirement),
                        Some(existing) => {
                            if existing != requirement {
                                return Err(FinitumError::ArtifactMismatch(format!(
                                    "system field {symbol} has inconsistent element requirements \
                                 across blocks"
                                )));
                            }
                        }
                    }
                }
            }
            let requirement = found.ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "system field {symbol} has no element requirement in any block"
                ))
            })?;
            if requirement.topological_dimension as usize != mesh.dimension() {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "system field {symbol} requires topological dimension {}, mesh has dimension {}",
                    requirement.topological_dimension,
                    mesh.dimension()
                )));
            }
            let components = expected_field_components(
                requirement.family,
                &requirement.value_shape,
                mesh.dimension(),
            )?;
            let block = layout.block_by_variable(variable).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "layout has no block for system field {symbol}"
                ))
            })?;
            if block.component_count != components {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "system field {symbol} has {} concrete layout components, typed requirement \
                 needs {components}",
                    block.component_count
                )));
            }
            let field = match (requirement.family, requirement.polynomial_order) {
                (ElementFamilyRequirement::H1 | ElementFamilyRequirement::L2, order @ (1 | 2)) => {
                    let basis_count = simplex_basis_count(mesh.dimension(), order);
                    let mut basis_values = Vec::with_capacity(quadrature.len() * basis_count);
                    let mut basis_gradients =
                        Vec::with_capacity(quadrature.len() * basis_count * mesh.dimension());
                    for point in quadrature {
                        let (values, gradients) =
                            simplex_basis(mesh.dimension(), order, &point.coordinates)?;
                        basis_values.extend(values);
                        for gradient in gradients {
                            basis_gradients.extend(gradient);
                        }
                    }
                    let element = PreparedElement::new(
                        mesh.dimension(),
                        basis_count,
                        quadrature.to_vec(),
                        basis_values,
                        basis_gradients,
                    )?;
                    let dofs = match order {
                        1 => vector_nodal_dof_map(mesh, components)?,
                        2 => quadratic_simplex_dof_map(mesh, components)?,
                        _ => unreachable!("polynomial order matched above"),
                    };
                    FieldElement {
                        kind: FieldKind::Lagrange(element),
                        dofs,
                    }
                }
                (ElementFamilyRequirement::L2, 0) => {
                    let dofs = cell_constant_dof_map(mesh)?;
                    let basis_values = vec![1.0; quadrature.len()];
                    let basis_gradients = vec![0.0; quadrature.len() * mesh.dimension()];
                    let element = PreparedElement::new(
                        mesh.dimension(),
                        1,
                        quadrature.to_vec(),
                        basis_values,
                        basis_gradients,
                    )?;
                    FieldElement {
                        kind: FieldKind::Lagrange(element),
                        dofs,
                    }
                }
                (ElementFamilyRequirement::Hdiv, 0) => {
                    let compatible = compatible.ok_or_else(|| {
                        FinitumError::ArtifactMismatch(format!(
                            "system field {symbol} requires Hdiv(order=0) but no compatible DOF \
                         maps were computed by SystemRealizationPlan::new"
                        ))
                    })?;
                    if block.entity_count != compatible.hdiv_dof_count {
                        return Err(FinitumError::ArtifactMismatch(format!(
                            "system field {symbol} has {} concrete layout entities, RT0's compatible \
                         DOF map needs {} (one per mesh facet)",
                            block.entity_count, compatible.hdiv_dof_count
                        )));
                    }
                    if compatible.hdiv.len() != mesh.cells().len() {
                        return Err(FinitumError::ArtifactMismatch(
                            "RT0 compatible DOF map has a different cell count than the mesh"
                                .into(),
                        ));
                    }
                    let mut dof_restrictions = Vec::with_capacity(mesh.cells().len());
                    let mut orientations = Vec::with_capacity(mesh.cells().len());
                    for restriction in &compatible.hdiv {
                        if restriction.dofs.len() != rt0_basis_count(mesh.dimension()) {
                            return Err(FinitumError::ArtifactMismatch(
                                "RT0 compatible DOF map cell restriction has the wrong facet count"
                                    .into(),
                            ));
                        }
                        dof_restrictions.push(ElementRestriction {
                            dofs: restriction.dofs.clone(),
                        });
                        orientations.push(restriction.orientations.clone());
                    }
                    let dofs = DofMap::new(compatible.hdiv_dof_count, dof_restrictions)?;
                    FieldElement {
                        kind: FieldKind::Hdiv0 { orientations },
                        dofs,
                    }
                }
                _ => {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "system realization admits scalar or dimension-vector Lagrange fields of \
                     order 1 or 2 (H1 or L2), piecewise-constant fields (L2(order=0)), or RT0 \
                     fields (Hdiv(order=0)) in the mesh's own topological dimension only; field \
                     {symbol} requires {requirement:?}"
                    )));
                }
            };
            fields.insert(variable, field);
        }
    }
    Ok(fields)
}

// ---------------------------------------------------------------------------------------------
// RT0 (Hdiv(order=0)) per-cell evaluation and adjoint scatter.
//
// Unlike a nodal Lagrange field, RT0's physical basis functions depend on the cell's own affine
// map (contravariant Piola pushforward), so there is no single shared reference table the way
// `crate::realization::evaluate_basis_input`/`apply_basis_adjoint` assume -- these are genuinely
// new, RT0-specific evaluate/adjoint pairs (not a generalization of those functions), reusing
// only `crate::element::rt0_reference_basis` (the pure reference-space math) and
// `crate::mapping::AffineMap`'s already-landed Piola maps (FC8). Each physical coefficient is
// `orientation[i] * local_state[i]` (the DOF map's raw stored value is unsigned; the sign lives
// alongside it in `FieldKind::Hdiv0::orientations`, exactly [`CompatibleDofMaps::hdiv`]'s own
// per-cell table), and both the forward evaluation and its adjoint are LINEAR in that
// coefficient, so each adjoint below is the exact algebraic transpose of its evaluate
// counterpart (documented per-function).
// ---------------------------------------------------------------------------------------------

/// RT0 field value at one reference point: `Piola(sum_i orientation_i * local_state_i *
/// phi_i(reference_point))`. `local_state`/`orientations` must both have length
/// `rt0_basis_count(affine.dimension())`.
fn evaluate_rt0_value(
    affine: &AffineMap,
    orientations: &[i8],
    reference_point: &[f64],
    local_state: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    let dimension = affine.dimension();
    let (basis_values, _divergence) = rt0_reference_basis(dimension, reference_point)?;
    if orientations.len() != basis_values.len() || local_state.len() != basis_values.len() {
        return Err(FinitumError::InvalidRealization(format!(
            "RT0 value evaluation needs {} oriented coefficients, got {} orientations and {} \
             state values",
            basis_values.len(),
            orientations.len(),
            local_state.len()
        )));
    }
    let mut reference_value = vec![0.0; dimension];
    for ((sign, coefficient), basis) in orientations.iter().zip(local_state).zip(&basis_values) {
        let signed = f64::from(*sign) * coefficient;
        for (axis, component) in basis.iter().enumerate() {
            reference_value[axis] += signed * component;
        }
    }
    affine.contravariant_piola(&reference_value)
}

/// RT0 field divergence at any point (the reference divergence is the same constant everywhere
/// on the cell -- see [`rt0_reference_basis`]'s own doc comment): `map_hdiv_divergence(dimension
/// * sum_i orientation_i * local_state_i)`.
fn evaluate_rt0_divergence(
    affine: &AffineMap,
    orientations: &[i8],
    local_state: &[f64],
) -> Result<f64, FinitumError> {
    if orientations.len() != local_state.len() {
        return Err(FinitumError::InvalidRealization(format!(
            "RT0 divergence evaluation needs matching orientation/state lengths, got {} and {}",
            orientations.len(),
            local_state.len()
        )));
    }
    let dimension = affine.dimension();
    let reference_divergence: f64 = orientations
        .iter()
        .zip(local_state)
        .map(|(sign, value)| f64::from(*sign) * value)
        .sum::<f64>()
        * dimension as f64;
    Ok(affine.map_hdiv_divergence(reference_divergence))
}

/// Exact transpose of [`evaluate_rt0_value`]: `d(value)/d(local_state_i) = orientation_i *
/// Piola(phi_i(reference_point))`, contracted against `point_output` and accumulated (scaled by
/// `scale`) into `local_output[i]`.
fn apply_rt0_value_adjoint(
    affine: &AffineMap,
    orientations: &[i8],
    reference_point: &[f64],
    point_output: &[f64],
    scale: f64,
    local_output: &mut [f64],
) -> Result<(), FinitumError> {
    let dimension = affine.dimension();
    if point_output.len() != dimension {
        return Err(FinitumError::InvalidRealization(format!(
            "RT0 value adjoint expects a {dimension}-component point output, got {}",
            point_output.len()
        )));
    }
    let (basis_values, _divergence) = rt0_reference_basis(dimension, reference_point)?;
    if orientations.len() != basis_values.len() || local_output.len() != basis_values.len() {
        return Err(FinitumError::InvalidRealization(
            "RT0 value adjoint local output does not match the RT0 basis count".into(),
        ));
    }
    for ((sign, basis), output) in orientations
        .iter()
        .zip(&basis_values)
        .zip(local_output.iter_mut())
    {
        let physical_basis = affine.contravariant_piola(basis)?;
        let dot: f64 = physical_basis
            .iter()
            .zip(point_output)
            .map(|(component, value)| component * value)
            .sum();
        *output += scale * f64::from(*sign) * dot;
    }
    Ok(())
}

/// Exact transpose of [`evaluate_rt0_divergence`]: `d(value)/d(local_state_i) = orientation_i *
/// map_hdiv_divergence(dimension)` (the same constant for every `i`), scaled by `point_output`
/// and `scale`, accumulated into `local_output[i]`.
fn apply_rt0_divergence_adjoint(
    affine: &AffineMap,
    orientations: &[i8],
    point_output: f64,
    scale: f64,
    local_output: &mut [f64],
) -> Result<(), FinitumError> {
    if orientations.len() != local_output.len() {
        return Err(FinitumError::InvalidRealization(
            "RT0 divergence adjoint local output does not match the orientation count".into(),
        ));
    }
    let dimension = affine.dimension();
    let physical_basis_divergence = affine.map_hdiv_divergence(dimension as f64);
    for (sign, output) in orientations.iter().zip(local_output.iter_mut()) {
        *output += scale * f64::from(*sign) * physical_basis_divergence * point_output;
    }
    Ok(())
}

/// Dispatching gather: evaluates one basis-sourced input's value through whichever
/// [`FieldKind`] `field` is. `reference_point` is `quadrature[point].coordinates` (the shared
/// index convention [`build_field_elements`] documents); `cell` selects the RT0 orientation row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_field_basis_input(
    field: &FieldElement,
    geometry: &CellGeometry,
    affine: &AffineMap,
    cell: usize,
    point: usize,
    reference_point: &[f64],
    input: &scientia::QFunctionInput,
    local_state: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    match &field.kind {
        FieldKind::Lagrange(element) => {
            evaluate_basis_input(element, geometry, point, input, local_state)
        }
        FieldKind::Hdiv0 { orientations } => {
            let cell_orientations = orientations.get(cell).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "RT0 field has no orientation row for cell {cell}"
                ))
            })?;
            match input.binding.evaluation.derivative {
                DerivativeEvaluation::Value => {
                    evaluate_rt0_value(affine, cell_orientations, reference_point, local_state)
                }
                DerivativeEvaluation::Divergence => {
                    evaluate_rt0_divergence(affine, cell_orientations, local_state)
                        .map(|value| vec![value])
                }
                other => Err(FinitumError::UnsupportedRealization(format!(
                    "RT0 (Hdiv(order=0)) fields support Value/Divergence basis evaluation only, \
                     got {other:?}"
                ))),
            }
        }
    }
}

/// Dispatching adjoint: exact transpose of [`evaluate_field_basis_input`] for a *row* (test)
/// field's cell-integral output scatter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_field_basis_adjoint(
    field: &FieldElement,
    geometry: &CellGeometry,
    affine: &AffineMap,
    cell: usize,
    point: usize,
    reference_point: &[f64],
    derivative: &DerivativeEvaluation,
    point_output: &[f64],
    scale: f64,
    local_output: &mut [f64],
) -> Result<(), FinitumError> {
    match &field.kind {
        FieldKind::Lagrange(element) => apply_basis_adjoint(
            element,
            geometry,
            point,
            derivative,
            point_output,
            scale,
            local_output,
        ),
        FieldKind::Hdiv0 { orientations } => {
            let cell_orientations = orientations.get(cell).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "RT0 field has no orientation row for cell {cell}"
                ))
            })?;
            match derivative {
                DerivativeEvaluation::Value => apply_rt0_value_adjoint(
                    affine,
                    cell_orientations,
                    reference_point,
                    point_output,
                    scale,
                    local_output,
                ),
                DerivativeEvaluation::Divergence if point_output.len() == 1 => {
                    apply_rt0_divergence_adjoint(
                        affine,
                        cell_orientations,
                        point_output[0],
                        scale,
                        local_output,
                    )
                }
                other => Err(FinitumError::UnsupportedRealization(format!(
                    "RT0 (Hdiv(order=0)) fields support Value/Divergence basis adjoints only, \
                     got {other:?}"
                ))),
            }
        }
    }
}

/// Adjoint of the RT0 exterior-facet normal-trace evaluation (mission item 2's narrowly-scoped
/// `SemanticMeasure::ExteriorFacet` support -- see [`SystemRealizationPlan::
/// bind_kernels_with_facets`]'s doc comment): scatters a single already-quadrature-integrated
/// `point_output` scalar into the one local coefficient it multiplies.
///
/// Derivation: RT0's normal trace on any one of its own cell's facets is *constant* along that
/// facet (the defining "order zero" property of the space) and equals exactly `orientation *
/// coefficient / area(F)` (Piola pushforward preserves reference flux exactly -- see
/// `crate::mapping::AffineMap::contravariant_piola`'s own doc comment and the FC8
/// `darcy_hdiv_mapping_preserves_flux_and_oriented_shared_facets` test -- so the physical flux
/// through the local facet the coefficient owns is exactly `orientation * coefficient`, spread
/// uniformly over `area(F)`). The single-centroid-point facet rule this crate uses throughout
/// (matching `crate::realization`'s own GX-C4 convention) therefore evaluates the integral as
/// `area(F) * point_output * (orientation * coefficient / area(F)) = point_output * orientation *
/// coefficient` -- the `area(F)` factor cancels exactly, so this adjoint needs neither the
/// facet's geometry nor a quadrature scale.
fn apply_rt0_normal_trace_adjoint(
    orientations: &[i8],
    local_facet: usize,
    point_output: &[f64],
    local_output: &mut [f64],
) -> Result<(), FinitumError> {
    if point_output.len() != 1 {
        return Err(FinitumError::InvalidRealization(format!(
            "RT0 exterior-facet normal trace output must be scalar, got {} components",
            point_output.len()
        )));
    }
    let orientation = orientations.get(local_facet).copied().ok_or_else(|| {
        FinitumError::InvalidRealization(
            "facet local index out of range for the RT0 restriction".into(),
        )
    })?;
    let slot = local_output.get_mut(local_facet).ok_or_else(|| {
        FinitumError::InvalidRealization(
            "facet local index out of range for the RT0 restriction".into(),
        )
    })?;
    *slot += f64::from(orientation) * point_output[0];
    Ok(())
}

type SystemPointValueEvaluator =
    dyn Fn(&PointEvaluation) -> Result<Vec<f64>, InputEvaluationError> + Send + Sync;
type SystemPointDirectionEvaluator = dyn Fn(&PointEvaluation, &PointEvaluation) -> Result<Vec<f64>, InputEvaluationError>
    + Send
    + Sync;

/// Caller-supplied closure resolution of one block's non-`Basis`-sourced primal input.
///
/// This is the bounded slice of the realization inventory's item 6 ("per-block external inputs")
/// this lane implements: a value callback and its exact directional derivative, evaluated from
/// that integral's own basis-backed active inputs at each quadrature point -- mirroring
/// [`crate::realization::DynamicExternalInput`]'s contract exactly, just keyed additionally by
/// `equation` since one [`SystemOperator`] binds several blocks' factorizations (each with its
/// own locally-indexed integrals) at once. It does *not* cover item 6's full generality: a
/// `Stored`/regional material-property tensor table bound per block (the analogue of
/// `RealizationPlan`'s `ExternalInput`) remains deferred future work; only closure-based
/// resolution is admitted here.
///
/// W8 lane F2: the callbacks are fallible ([`Self::try_new`]), exactly as
/// [`crate::realization::DynamicExternalInput`]'s: a typed [`InputEvaluationError`] returned by
/// either callback is located (cell, point, time) and propagated as
/// [`FinitumError::InputEvaluation`] out of every `SystemOperator` / [`ReducedSystemOperator`]
/// action and, as `NumericError::Evaluation`, out of every Methodus trait entry point. The
/// first failure in cell / quadrature-point / input order wins. [`Self::new`] is the
/// infallible form.
#[derive(Clone)]
pub struct SystemConstitutiveInput {
    /// The equation the closure binds to (display only for a keyed target -- see `target`).
    pub equation: String,
    pub integral_index: usize,
    pub input: TensorInputId,
    component_count: usize,
    identity: String,
    target: ConstitutiveTarget,
    value: Arc<SystemPointValueEvaluator>,
    direction: Arc<SystemPointDirectionEvaluator>,
}

/// What a [`SystemConstitutiveInput`] binds to (W8 lane F-MI): an equation by name (the
/// one-instance form; on a composed plan the name must be unique across instances), a
/// residual row by [`SysResId`], or a producer output kernel of a same-mesh bind.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ConstitutiveTarget {
    Equation,
    Residual(SysResId),
    Output {
        instance: InstanceId,
        output: scientia::OutputId,
    },
}

impl std::fmt::Debug for SystemConstitutiveInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SystemConstitutiveInput")
            .field("equation", &self.equation)
            .field("integral_index", &self.integral_index)
            .field("input", &self.input)
            .field("component_count", &self.component_count)
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl SystemConstitutiveInput {
    /// The infallible form: `value` and `direction` cannot refuse. A thin wrapper over
    /// [`Self::try_new`] (deleted by slice F3 once every consumer has migrated).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        equation: impl Into<String>,
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        identity: impl Into<String>,
        value: impl Fn(&PointEvaluation) -> Vec<f64> + Send + Sync + 'static,
        direction: impl Fn(&PointEvaluation, &PointEvaluation) -> Vec<f64> + Send + Sync + 'static,
    ) -> Result<Self, FinitumError> {
        Self::try_new(
            equation,
            integral_index,
            input,
            component_count,
            identity,
            move |point| Ok(value(point)),
            move |point, direction_point| Ok(direction(point, direction_point)),
        )
    }

    /// The fallible form (W8 lane F2): `value` and `direction` return their own typed
    /// [`InputEvaluationError`] instead of a value when they cannot evaluate.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        equation: impl Into<String>,
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
        let equation = equation.into();
        if equation.trim().is_empty() {
            return Err(FinitumError::InvalidRealization(
                "constitutive input equation name must not be empty".into(),
            ));
        }
        if component_count == 0 {
            return Err(FinitumError::InvalidRealization(
                "constitutive input component count must be non-zero".into(),
            ));
        }
        let identity = identity.into();
        if identity.trim().is_empty() {
            return Err(FinitumError::InvalidRealization(
                "constitutive input identity must not be empty".into(),
            ));
        }
        Ok(Self {
            equation,
            integral_index,
            input,
            component_count,
            identity,
            target: ConstitutiveTarget::Equation,
            value: Arc::new(value),
            direction: Arc::new(direction),
        })
    }

    /// W8 lane F-MI: [`Self::try_new`] keyed by a residual row of the plan's
    /// [`SystemIdMap`] instead of an equation name -- unambiguous on a composed plan where two
    /// instances of one model carry the same equation names. `equation` displays as the
    /// residual id.
    pub fn try_new_for_residual(
        residual: SysResId,
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
        let mut built = Self::try_new(
            residual.to_string(),
            integral_index,
            input,
            component_count,
            identity,
            value,
            direction,
        )?;
        built.target = ConstitutiveTarget::Residual(residual);
        Ok(built)
    }

    /// W8 lane F-MI: a closure for a non-basis input of a producer **output** kernel of a
    /// composed plan (`integral_index` is that output's single integral, `input` one of its
    /// non-basis `QFunction` inputs), evaluated at every consumer quadrature point the bind
    /// reads the output at; its [`PointEvaluation`] carries the producer instance's own
    /// active inputs and bound inputs. `equation` displays as `output#<id>`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_for_output(
        instance: InstanceId,
        output: scientia::OutputId,
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
        let mut built = Self::try_new(
            format!("{instance}/output#{output}"),
            integral_index,
            input,
            component_count,
            identity,
            value,
            direction,
        )?;
        built.target = ConstitutiveTarget::Output { instance, output };
        Ok(built)
    }

    /// The caller-declared identity of this closure (covered by the operator digest).
    pub fn identity(&self) -> &str {
        &self.identity
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

/// SC-W1 system-path parity (a): a stored quadrature-point table bound to one non-basis input
/// of a cell integral of the residual `residual` -- the system counterpart of the stored
/// [`ExternalInput`] a `RealizationPlan` carries, in the same cell/quadrature/component order
/// over [`SystemOperator::quadrature`] (build one with [`ExternalInput::from_coefficient_at`] to
/// parameterize it by a design vector, or [`ExternalInput::new`]). Bound through
/// [`SystemRealizationPlan::bind_kernels_with_inputs`]; state-independent (its direction is
/// zero, exactly as a stored single-model input), and covered value-by-value by the operator
/// digest.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemExternalInput {
    pub residual: SysResId,
    pub input: ExternalInput,
}

/// SC-W1 system-path parity (a): one stored input of residual `residual`'s cell integral viewed
/// as a distributed coefficient over a caller-owned design space -- the system counterpart of
/// [`DistributedCoefficient`], consumed by [`SystemOperator::coefficient_dimension`],
/// [`SystemOperator::coefficient_jacobian_vector_product`] and its exact transpose
/// [`SystemOperator::coefficient_vector_jacobian_product`] (and their
/// [`ReducedSystemOperator`] forms).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemDistributedCoefficient {
    pub residual: SysResId,
    pub coefficient: DistributedCoefficient,
}

/// The producer output's value at one quadrature point for one same-mesh bind (W8 lane
/// F-MI), with everything the transpose needs to chain back through the producer kernel.
#[derive(Clone, Debug)]
struct BoundPointValue {
    /// Index into `SystemOperatorData::binds`.
    bind: usize,
    values: Vec<f64>,
    /// The output's directional derivative along the current direction (JVP actions only).
    direction: Option<Vec<f64>>,
    /// The producer output kernel's gathered point inputs and its evaluation point.
    producer_inputs: BTreeMap<TensorInputId, Vec<f64>>,
    producer_evaluation: PointEvaluation,
}

/// The bound inputs visible to one instance at one quadrature point.
#[derive(Clone, Copy)]
struct BoundTable<'a> {
    binds: &'a [BoundBind],
    values: &'a [BoundPointValue],
    instance: InstanceId,
}

impl<'a> BoundTable<'a> {
    fn entries(&self) -> impl Iterator<Item = (&'a RealizedBind, &'a BoundPointValue)> + 'a {
        let (binds, instance) = (self.binds, self.instance);
        self.values
            .iter()
            .map(move |value| (&binds[value.bind].bind, value))
            .filter(move |(bind, _)| bind.consumer == instance)
    }

    fn get(&self, symbol: SymbolId) -> Option<&'a BoundPointValue> {
        self.entries()
            .find(|(bind, _)| bind.consumer_symbol == symbol)
            .map(|(_, value)| value)
    }

    /// The point's bound inputs as the consumer's closures see them (values).
    fn point_inputs(&self) -> Vec<PointBoundInput> {
        self.entries()
            .map(|(bind, value)| PointBoundInput {
                symbol: bind.consumer_symbol,
                slot: bind.consumer_slot.clone(),
                values: value.values.clone(),
            })
            .collect()
    }

    /// The point's bound inputs' directions (zero where no direction was evaluated).
    fn point_directions(&self) -> Vec<PointBoundInput> {
        self.entries()
            .map(|(bind, value)| PointBoundInput {
                symbol: bind.consumer_symbol,
                slot: bind.consumer_slot.clone(),
                values: value
                    .direction
                    .clone()
                    .unwrap_or_else(|| vec![0.0; value.values.len()]),
            })
            .collect()
    }
}

/// One resolved non-basis input binding of a system operator.
enum SystemInputBinding<'a> {
    Stored(&'a ExternalInput),
    Constitutive(&'a SystemConstitutiveInput),
    /// A same-mesh bind's producer output (kernel-input path).
    Bound(&'a BoundPointValue),
}

/// Whose closures and tables a point evaluation resolves against: one residual row of the
/// operator, or one producer output kernel of a bind.
#[derive(Clone, Copy)]
enum InputScope<'a> {
    Row {
        row: usize,
        constitutive: &'a BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
        stored: &'a BTreeMap<(usize, usize, TensorInputId), ExternalInput>,
    },
    Output {
        constitutive: &'a BTreeMap<TensorInputId, SystemConstitutiveInput>,
    },
}

/// The non-basis input bindings every point evaluation of a system operator resolves against:
/// the scope's closure-based constitutive inputs and stored tables, the instance's bound
/// inputs at the point, plus the shared quadrature's point count the stored tables are
/// indexed with.
#[derive(Clone, Copy)]
struct SystemInputBindings<'a> {
    scope: InputScope<'a>,
    bound: BoundTable<'a>,
    point_count: usize,
}

impl<'a> SystemInputBindings<'a> {
    fn resolve(
        &self,
        integral_index: usize,
        input: &QFunctionInput,
    ) -> Result<SystemInputBinding<'a>, FinitumError> {
        if let Some(bound) = self.bound.get(input.binding.symbol) {
            return Ok(SystemInputBinding::Bound(bound));
        }
        match self.scope {
            InputScope::Row {
                row,
                constitutive,
                stored,
            } => {
                let key = (row, integral_index, input.id);
                if let Some(stored) = stored.get(&key) {
                    return Ok(SystemInputBinding::Stored(stored));
                }
                constitutive
                    .get(&key)
                    .map(SystemInputBinding::Constitutive)
                    .ok_or_else(|| {
                        FinitumError::ArtifactMismatch(format!(
                            "integral {integral_index} input {:?} has no bound constitutive or \
                             stored input",
                            input.id
                        ))
                    })
            }
            InputScope::Output { constitutive } => constitutive
                .get(&input.id)
                .map(SystemInputBinding::Constitutive)
                .ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "output integral {integral_index} input {:?} has no bound constitutive \
                         input",
                        input.id
                    ))
                }),
        }
    }
}

/// One instance's view of the operator's realized fields: every field by system variable,
/// and the instance's own `symbol -> variable` map (W8 lane F-MI: two instances of one model
/// share symbols, so a kernel input's `binding.symbol` resolves through its instance).
#[derive(Clone, Copy)]
struct InstanceFields<'a> {
    fields: &'a BTreeMap<SysVarId, FieldElement>,
    variables: &'a BTreeMap<SymbolId, SysVarId>,
}

impl<'a> InstanceFields<'a> {
    fn variable(&self, symbol: SymbolId) -> Result<SysVarId, FinitumError> {
        self.variables.get(&symbol).copied().ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "field {symbol} is not a variable of this instance"
            ))
        })
    }

    fn field(&self, symbol: SymbolId) -> Result<(SysVarId, &'a FieldElement), FinitumError> {
        let variable = self.variable(symbol)?;
        let field = self.fields.get(&variable).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "field {symbol} ({variable}) has not been realized by the system operator"
            ))
        })?;
        Ok((variable, field))
    }
}

/// Maps Scientia's structural `FormSymmetry` (C5.4) directly onto a declared
/// `methodus::OperatorSymmetry` (C5.5) -- item 8's "thread `OperatorStructure` into the realized
/// operator's declared `symmetry()`".
fn map_form_symmetry(symmetry: FormSymmetry) -> OperatorSymmetry {
    match symmetry {
        FormSymmetry::Symmetric => OperatorSymmetry::Symmetric,
        FormSymmetry::Nonsymmetric => OperatorSymmetry::Nonsymmetric,
        FormSymmetry::Unknown => OperatorSymmetry::Unknown,
    }
}

/// Cross-checks Scientia's derived `OperatorStructure` block presence against the realized
/// system's own `(row, column)` coordinates -- item 8's "refusing typed when structure and
/// realization disagree". `OperatorStructure` is derived read-only from the FC3/FC4 artifact
/// chain (never touching mesh/DOF data), so this is a structural consistency check, not a
/// numerical one.
fn validate_structure_matches_system(
    system: &OperatorSystem,
    structure: &OperatorStructure,
) -> Result<(), FinitumError> {
    let mut present = BTreeSet::new();
    for block in &system.blocks {
        for coordinate in &block.coordinates {
            present.insert((coordinate.row, coordinate.column));
        }
    }
    for block_structure in &structure.blocks {
        let is_present = present.contains(&(block_structure.row, block_structure.column));
        if block_structure.present != is_present {
            return Err(FinitumError::ArtifactMismatch(format!(
                "derived operator structure disagrees with the realized system at block ({}, \
                 {}): structure claims present={}, the system's own coordinates say {is_present}",
                block_structure.row, block_structure.column, block_structure.present,
            )));
        }
    }
    Ok(())
}

/// Auto-derives [`BlockNullspaceCandidate`] declarations from Scientia's structural
/// `nullspace_candidates` (C5.4) -- item 9, no hand-written candidate. Only `NullspaceKind::
/// Constant` is representable by [`BlockNullspaceCandidate`] today; a structural `RigidBody`
/// candidate is refused typed rather than silently dropped.
fn derive_nullspace_candidates(
    structure: &OperatorStructure,
) -> Result<Vec<BlockNullspaceCandidate>, FinitumError> {
    structure
        .nullspace_candidates
        .iter()
        .map(|candidate| match candidate.kind {
            NullspaceKind::Constant => Ok(BlockNullspaceCandidate::constant(
                candidate.field,
                candidate.reason.clone(),
            )),
            NullspaceKind::RigidBody { .. } => Err(FinitumError::UnsupportedRealization(format!(
                "structural nullspace candidate for field {} is RigidBody, which \
                 finitum::mixed::BlockNullspaceCandidate does not yet represent (Constant-only \
                 scope)",
                candidate.field
            ))),
        })
        .collect()
}

/// Schema of [`SystemOperator::digest`]'s payload: `/2` adds every stored table's values
/// (SC-W1 system-path parity), so two operators differing only in a design vector differ in
/// identity exactly as two `RealizationPlan`s do.
pub const SYSTEM_OPERATOR_DIGEST_SCHEMA: &str = "finitum-system-operator/2";

fn system_operator_digest(
    plan: &SystemRealizationPlan,
    constitutive: &BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
    stored: &BTreeMap<(usize, usize, TensorInputId), ExternalInput>,
    equation_sign: &BTreeMap<usize, f64>,
    facet_regions: &BTreeMap<RegionId, Vec<FacetId>>,
    binds: &[BoundBind],
) -> Digest {
    #[derive(Serialize)]
    struct ConstitutiveIdentity<'a> {
        block_index: usize,
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        identity: &'a str,
    }
    #[derive(Serialize)]
    struct StoredIdentity<'a> {
        block_index: usize,
        integral_index: usize,
        input: TensorInputId,
        component_count: usize,
        values: &'a [f64],
    }
    /// A closure bound to a producer output kernel of a composed plan (W8 F-MI); absent from
    /// the payload of every plan without binds, so one-instance digests are unchanged.
    #[derive(Serialize)]
    struct OutputConstitutiveIdentity<'a> {
        consumer_slot: &'a str,
        output: &'a str,
        input: TensorInputId,
        component_count: usize,
        identity: &'a str,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        plan_digest: &'a Digest,
        constitutive: Vec<ConstitutiveIdentity<'a>>,
        stored: Vec<StoredIdentity<'a>>,
        equation_sign: &'a BTreeMap<usize, f64>,
        facet_regions: BTreeMap<u32, Vec<usize>>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        outputs: Vec<OutputConstitutiveIdentity<'a>>,
    }
    let payload = Payload {
        schema: SYSTEM_OPERATOR_DIGEST_SCHEMA,
        plan_digest: plan.artifact_digest(),
        stored: stored
            .iter()
            .map(
                |(&(block_index, integral_index, input), table)| StoredIdentity {
                    block_index,
                    integral_index,
                    input,
                    component_count: table.component_count(),
                    values: table.values(),
                },
            )
            .collect(),
        constitutive: constitutive
            .iter()
            .map(
                |(&(block_index, integral_index, input), binding)| ConstitutiveIdentity {
                    block_index,
                    integral_index,
                    input,
                    component_count: binding.component_count,
                    identity: &binding.identity,
                },
            )
            .collect(),
        equation_sign,
        facet_regions: facet_regions
            .iter()
            .map(|(region, facets)| (region.0, facets.iter().map(|facet| facet.0).collect()))
            .collect(),
        outputs: binds
            .iter()
            .flat_map(|bind| {
                bind.constitutive
                    .iter()
                    .map(move |(&input, binding)| OutputConstitutiveIdentity {
                        consumer_slot: &bind.bind.consumer_slot,
                        output: &bind.bind.output_path,
                        input,
                        component_count: binding.component_count,
                        identity: &binding.identity,
                    })
            })
            .collect(),
    };
    let bytes =
        serde_json::to_vec(&payload).expect("system operator digest payload is serializable");
    Digest {
        algorithm: "blake3".into(),
        hex: blake3::hash(&bytes).to_hex().to_string(),
    }
}

/// The Methodus block layout of a system operator: `field_<symbol>` per block on a
/// one-instance plan (unchanged), `<instance>/field_<symbol>` on a composed plan (the
/// partition names Sinbad's transient runner records per instance).
fn solver_block_layout_keyed(
    layout: &BlockLayout,
    ids: &SystemIdMap,
) -> Result<methodus::BlockLayout, FinitumError> {
    if ids.instances().len() == 1 {
        return solver_block_layout(layout);
    }
    let specifications = layout
        .blocks()
        .iter()
        .map(|block| {
            let instance = ids
                .variable_origin(block.variable)
                .and_then(|origin| {
                    ids.instances()
                        .iter()
                        .find(|record| record.instance == origin.instance)
                })
                .map(|record| record.name.as_str())
                .unwrap_or("?");
            methodus::BlockSpec {
                name: format!("{instance}/field_{}", block.symbol.0),
                length: block.extent,
                residual_scale: 1.0,
            }
        })
        .collect();
    methodus::BlockLayout::new(specifications)
        .map_err(|error| FinitumError::InvalidRealization(error.to_string()))
}

impl SystemRealizationPlan {
    /// Binds every present `(row, column)` block coordinate to real, executable Malleus kernels
    /// (mission items 1/2), derives Scientia's structural `OperatorStructure` and cross-checks it
    /// against the realized coordinates (item 8), and auto-derives `BlockNullspaceCandidate`
    /// declarations from it (item 9) -- turning this shape-validated plan into an executable
    /// [`SystemOperator`].
    ///
    /// LAGRANGE (Taylor-Hood P2/P1) scope only: refuses typed (`FinitumError::
    /// UnsupportedRealization`) when a field needs a compatible-element (Hcurl/Hdiv) or
    /// discontinuous-Galerkin DOF map (see `build_field_elements`), when any block integrates
    /// anything but a `SemanticMeasure::Cell` measure or realizes an output at anything but a
    /// cell evaluation site (facet bindings, item 5), or when a block references a non-`Basis`
    /// primal input `constitutive` supplies no [`SystemConstitutiveInput`] for (item 6's bounded
    /// closure-based slice; a `Stored`/regional external tensor table is refused typed).
    ///
    /// `equation_sign` is a solution-preserving orientation correction: multiplying an equation
    /// row by a nonzero constant never changes its solution set (`c * (row) = c * rhs` has the
    /// same solutions as `row = rhs` for any `c != 0`), so this is restricted to exactly `1.0`
    /// or `-1.0` -- refused typed otherwise -- and applied only to *whole* equation rows, keyed
    /// by `equation` name (an unknown equation name is refused typed). It exists because a model
    /// may author its equations without a shared sign convention across a coupled system (e.g.
    /// one equation's residual already carries a global minus sign the other's does not): when
    /// that happens, Scientia's own structural `form_symmetry` correctly reports `Unknown`
    /// (never `Symmetric`) for the *unsigned* system, and this lane found exactly that case in
    /// the 25-stokes.res corpus's literal `momentum`/`incompressibility` equation pair (see the
    /// lane report). A caller who supplies a nontrivial `equation_sign` therefore gets a
    /// declared `symmetry()` of `Unknown` until they explicitly call
    /// [`SystemOperator::prove_symmetry`] to establish it by assembly, rather than this method
    /// silently reusing Scientia's now-inapplicable structural claim for the resigned system.
    pub fn bind_kernels(
        &self,
        constitutive: Vec<SystemConstitutiveInput>,
        equation_sign: BTreeMap<String, f64>,
    ) -> Result<SystemOperator, FinitumError> {
        self.bind_kernels_with_facets(constitutive, equation_sign, BTreeMap::new())
    }

    /// As [`Self::bind_kernels`], additionally admitting a narrowly-scoped
    /// `SemanticMeasure::ExteriorFacet` integral (mission item 2's extension): an RT0
    /// (Hdiv(order=0)) row field's own exterior normal trace (`EvaluationSite::ExteriorTrace`,
    /// `DerivativeEvaluation::Value`, `TraceMapping::Normal`), with **no** `Basis`-sourced
    /// (trial) input at all -- `13-mixed-darcy.res`'s own `darcy_law` Neumann boundary integral
    /// is exactly this shape (`primal.inputs` is empty; the compiled expression is a
    /// caller-independent constant). External/constitutive facet inputs, active-input facet
    /// integrals, interior/interface/point measures, and non-RT0 facet traces remain refused
    /// typed -- this is real, narrow machinery (RT0's own normal trace has an exact closed form,
    /// see this module's own `apply_rt0_normal_trace_adjoint` doc comment), not a blanket facet
    /// realization.
    ///
    /// `facet_regions` maps each such integral's `region` to the concrete exterior [`FacetId`]s
    /// it integrates over -- resolved by the caller through a `RegionMap`/`RegionTags` binding
    /// exactly as [`crate::RealizationPlan::new_with_facets`] documents (a bare [`Mesh`] carries
    /// no region tags of its own). A region with no entry (or an empty entry) here is refused
    /// (`FinitumError::RealizationRegionUnmapped`).
    pub fn bind_kernels_with_facets(
        &self,
        constitutive: Vec<SystemConstitutiveInput>,
        equation_sign: BTreeMap<String, f64>,
        facet_regions: BTreeMap<RegionId, Vec<FacetId>>,
    ) -> Result<SystemOperator, FinitumError> {
        self.bind_kernels_with_inputs(constitutive, Vec::new(), equation_sign, facet_regions)
    }

    /// As [`Self::bind_kernels_with_facets`], additionally binding stored quadrature-point
    /// tables ([`SystemExternalInput`], SC-W1 system-path parity) to non-basis inputs of cell
    /// integrals: each names a residual of this plan's [`SystemIdMap`], an existing non-`Basis`
    /// input of one of that residual's `SemanticMeasure::Cell` integrals, carries that input's
    /// component count, and has exactly `cells * quadrature points * components` values over
    /// [`SystemOperator::quadrature`]. A key bound both as a closure and as a table, a facet
    /// integral, a basis input, or a shape mismatch is refused typed. Every non-basis cell
    /// input must be bound one way or the other.
    pub fn bind_kernels_with_inputs(
        &self,
        constitutive: Vec<SystemConstitutiveInput>,
        stored: Vec<SystemExternalInput>,
        equation_sign: BTreeMap<String, f64>,
        facet_regions: BTreeMap<RegionId, Vec<FacetId>>,
    ) -> Result<SystemOperator, FinitumError> {
        for (_, row, block) in self.rows() {
            for integral in &block.factorization.integrals {
                match &integral.measure {
                    SemanticMeasure::Cell { .. } => {
                        for output in &integral.primal.outputs {
                            if output.binding.evaluation.site != EvaluationSite::Cell {
                                return Err(FinitumError::UnsupportedRealization(format!(
                                    "system realization realizes cell evaluation sites only; \
                                     equation `{}` integral {} has an output at site {:?}",
                                    row.path,
                                    integral.integral_index,
                                    output.binding.evaluation.site
                                )));
                            }
                        }
                    }
                    SemanticMeasure::ExteriorFacet { region } => {
                        if facet_regions
                            .get(region)
                            .is_none_or(|facets| facets.is_empty())
                        {
                            return Err(FinitumError::RealizationRegionUnmapped(format!(
                                "{region:?}"
                            )));
                        }
                        if !integral.primal.inputs.is_empty() {
                            return Err(FinitumError::UnsupportedRealization(format!(
                                "system realization admits exterior-facet integrals with no \
                                 primal input at all (RT0's own normal-trace closed form only); \
                                 equation `{}` integral {} declares {} input(s)",
                                row.path,
                                integral.integral_index,
                                integral.primal.inputs.len()
                            )));
                        }
                        for output in &integral.primal.outputs {
                            let evaluation = &output.binding.evaluation;
                            if evaluation.site != EvaluationSite::ExteriorTrace
                                || evaluation.derivative != DerivativeEvaluation::Value
                                || evaluation.trace_mapping != Some(TraceMapping::Normal)
                            {
                                return Err(FinitumError::UnsupportedRealization(format!(
                                    "system realization admits exterior-facet integrals only for \
                                     an Hdiv(order=0) row field's Value/ExteriorTrace/Normal \
                                     trace; equation `{}` integral {} has evaluation {evaluation:?}",
                                    row.path, integral.integral_index
                                )));
                            }
                        }
                    }
                    other => {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "system realization admits SemanticMeasure::Cell integrals, or a \
                             narrowly-scoped Hdiv(order=0) SemanticMeasure::ExteriorFacet normal \
                             trace, only; equation `{}` integral {} has measure {other:?}",
                            row.path, integral.integral_index
                        )));
                    }
                }
            }
        }
        let quadrature = self.quadrature()?;
        let fields = build_field_elements(self, &quadrature)?;
        let instance_variables = (0..self.instances.len())
            .map(|index| {
                let instance = InstanceId(u32::try_from(index).expect("fits u32"));
                self.system_ids
                    .variables()
                    .iter()
                    .filter(|variable| variable.instance == instance)
                    .map(|variable| (variable.local, variable.id))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        for (_, row, block) in self.rows() {
            let has_exterior_facet =
                block.factorization.integrals.iter().any(|integral| {
                    matches!(integral.measure, SemanticMeasure::ExteriorFacet { .. })
                });
            if has_exterior_facet {
                let row_field = instance_variables[row.instance.0 as usize]
                    .get(&block.row)
                    .and_then(|variable| fields.get(variable))
                    .ok_or_else(|| {
                        FinitumError::ArtifactMismatch(format!(
                            "equation `{}` row field {} was not realized",
                            row.path, block.row
                        ))
                    })?;
                if !matches!(row_field.kind, FieldKind::Hdiv0 { .. }) {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "equation `{}` has an exterior-facet integral but row field {} is not \
                         Hdiv(order=0)",
                        row.path, block.row
                    )));
                }
            }
        }
        let mut facet_geometries: BTreeMap<FacetId, FacetGeometry> = BTreeMap::new();
        if !facet_regions.is_empty() {
            let mut referenced = BTreeSet::new();
            for ids in facet_regions.values() {
                referenced.extend(ids.iter().copied());
            }
            for facet_id in referenced {
                let facet = self.facets.facets().get(facet_id.0).ok_or_else(|| {
                    FinitumError::InvalidRealization(format!("facet {} does not exist", facet_id.0))
                })?;
                if !facet.is_exterior() {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "facet {} is not exterior; system realization refuses interior facet \
                         integrals",
                        facet_id.0
                    )));
                }
                facet_geometries
                    .insert(facet_id, FacetGeometry::compute(&self.mesh, facet.minus())?);
            }
        }
        let mut bindings = BTreeMap::new();
        for (index, _, block) in self.rows() {
            bindings.insert(
                index,
                bind_kernels(&block.factorization, block.kernels.clone())?,
            );
        }
        // The consumer-side bound symbols of every instance (same-mesh binds).
        let bound_symbols = |instance: InstanceId| -> Vec<(SymbolId, &str)> {
            self.composed
                .as_ref()
                .map(|composed| {
                    composed
                        .binds
                        .iter()
                        .filter(|bind| bind.consumer == instance)
                        .map(|bind| (bind.consumer_symbol, bind.consumer_slot.as_str()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut constitutive_by_key = BTreeMap::new();
        let mut output_constitutive: BTreeMap<
            (InstanceId, scientia::OutputId),
            BTreeMap<TensorInputId, SystemConstitutiveInput>,
        > = BTreeMap::new();
        for input in constitutive {
            let row_index = match input.target.clone() {
                ConstitutiveTarget::Equation => {
                    let matches = self
                        .rows()
                        .filter(|(_, row, block)| {
                            block.equation == input.equation || row.path == input.equation
                        })
                        .map(|(index, _, _)| index)
                        .collect::<Vec<_>>();
                    match matches.as_slice() {
                        [] => {
                            return Err(FinitumError::InvalidRealization(format!(
                                "constitutive input names equation `{}` which is absent from \
                                 the system",
                                input.equation
                            )));
                        }
                        [index] => *index,
                        _ => {
                            return Err(FinitumError::InvalidRealization(format!(
                                "constitutive input names equation `{}` which {} instances of \
                                 this composed plan carry; key it by residual \
                                 (`SystemConstitutiveInput::try_new_for_residual`) or by its \
                                 display path",
                                input.equation,
                                matches.len()
                            )));
                        }
                    }
                }
                ConstitutiveTarget::Residual(residual) => {
                    self.row_index(residual).ok_or_else(|| {
                        FinitumError::InvalidRealization(format!(
                            "constitutive input names residual {residual} which the system does \
                             not carry"
                        ))
                    })?
                }
                ConstitutiveTarget::Output { instance, output } => {
                    let composed = self.composed.as_ref().ok_or_else(|| {
                        FinitumError::InvalidRealization(format!(
                            "constitutive input `{}` names output {output} of {instance}, but \
                             this plan has no same-mesh binds (build it with \
                             `SystemRealizationPlan::composed`)",
                            input.identity
                        ))
                    })?;
                    if !composed
                        .binds
                        .iter()
                        .any(|bind| bind.producer == instance && bind.output == output)
                    {
                        return Err(FinitumError::InvalidRealization(format!(
                            "constitutive input `{}` names output {output} of {instance}, which \
                             no bind of this plan reads",
                            input.identity
                        )));
                    }
                    let inputs = output_constitutive.entry((instance, output)).or_default();
                    let id = input.input;
                    if inputs.insert(id, input).is_some() {
                        return Err(FinitumError::InvalidRealization(format!(
                            "constitutive input for output {output} of {instance} input {id:?} \
                             is bound more than once"
                        )));
                    }
                    continue;
                }
            };
            let key = (row_index, input.integral_index, input.input);
            if constitutive_by_key.insert(key, input).is_some() {
                return Err(FinitumError::InvalidRealization(format!(
                    "constitutive input for equation index {}, integral {}, input {:?} is bound \
                     more than once",
                    key.0, key.1, key.2
                )));
            }
        }
        let mut stored_by_key = BTreeMap::new();
        for table in stored {
            let row_index = self.row_index(table.residual).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "stored input names residual {} which the system does not carry",
                    table.residual
                ))
            })?;
            let (row, block) = self.row(row_index);
            let integral = block
                .factorization
                .integrals
                .iter()
                .find(|integral| integral.integral_index == table.input.integral_index)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "stored input names absent integral {} of equation `{}`",
                        table.input.integral_index, row.path
                    ))
                })?;
            if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "stored system inputs are realized on cell integrals only; equation `{}` \
                     integral {} has measure {:?}",
                    row.path, integral.integral_index, integral.measure
                )));
            }
            let input = integral
                .primal
                .inputs
                .iter()
                .find(|input| input.id == table.input.input)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "stored input names undeclared input {:?} of equation `{}` integral {}",
                        table.input.input, row.path, integral.integral_index
                    ))
                })?;
            if input.source == InputSourceRequirement::Basis {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{}` integral {} input {:?} is a basis input, not an external one",
                    row.path, integral.integral_index, input.id
                )));
            }
            let components = component_count(&input.shape)?;
            let expected = self.mesh.cells().len() * quadrature.len() * components;
            if table.input.component_count() != components || table.input.values().len() != expected
            {
                return Err(FinitumError::InvalidRealization(format!(
                    "stored input for equation `{}` integral {} input {:?} has {} components \
                     and {} values; the realization expects {components} components over {} \
                     cells x {} quadrature points ({expected} values)",
                    row.path,
                    integral.integral_index,
                    input.id,
                    table.input.component_count(),
                    table.input.values().len(),
                    self.mesh.cells().len(),
                    quadrature.len()
                )));
            }
            let key = (row_index, integral.integral_index, input.id);
            if constitutive_by_key.contains_key(&key) {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{}` integral {} input {:?} is bound both as a constitutive \
                     closure and as a stored table",
                    row.path, integral.integral_index, input.id
                )));
            }
            if stored_by_key.insert(key, table.input).is_some() {
                return Err(FinitumError::InvalidRealization(format!(
                    "stored input for equation `{}` integral {} input {:?} is bound more than \
                     once",
                    row.path, integral.integral_index, input.id
                )));
            }
        }
        for (row_index, row, block) in self.rows() {
            let bound = bound_symbols(row.instance);
            for integral in &block.factorization.integrals {
                for input in &integral.primal.inputs {
                    if input.source == InputSourceRequirement::Basis {
                        continue;
                    }
                    let key = (row_index, integral.integral_index, input.id);
                    let closed =
                        constitutive_by_key.contains_key(&key) || stored_by_key.contains_key(&key);
                    if let Some((_, slot)) = bound
                        .iter()
                        .find(|(symbol, _)| *symbol == input.binding.symbol)
                    {
                        if closed {
                            return Err(FinitumError::InvalidRealization(format!(
                                "equation `{}` integral {} input {:?} is closed by the bind on \
                                 `{slot}`; it must not also be bound as a closure or a stored \
                                 table",
                                row.path, integral.integral_index, input.id
                            )));
                        }
                        continue;
                    }
                    if !closed {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "equation `{}` integral {} input {:?} requires a caller-supplied \
                             SystemConstitutiveInput closure or a stored SystemExternalInput \
                             table (regional external tensor tables remain future work)",
                            row.path, integral.integral_index, input.id
                        )));
                    }
                }
            }
        }
        let mut equation_sign_by_block = BTreeMap::new();
        for (equation, sign) in &equation_sign {
            let matches = self
                .rows()
                .filter(|(_, row, block)| &block.equation == equation || &row.path == equation)
                .map(|(index, _, _)| index)
                .collect::<Vec<_>>();
            let block_index = match matches.as_slice() {
                [] => {
                    return Err(FinitumError::InvalidRealization(format!(
                        "equation sign names equation `{equation}` which is absent from the \
                         system"
                    )));
                }
                [index] => *index,
                _ => {
                    return Err(FinitumError::InvalidRealization(format!(
                        "equation sign names equation `{equation}` which {} instances of this \
                         composed plan carry; name it by its display path `<instance>.<equation>`",
                        matches.len()
                    )));
                }
            };
            if *sign != 1.0 && *sign != -1.0 {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{equation}` sign must be exactly 1.0 or -1.0 (a solution-\
                     preserving orientation correction only), got {sign}"
                )));
            }
            equation_sign_by_block.insert(block_index, *sign);
        }
        let mut structures = Vec::with_capacity(self.instances.len());
        let mut nullspace_candidates = Vec::new();
        for system in &self.instances {
            let structure = derive_operator_structure_for_system(system, None)
                .map_err(|error| FinitumError::ArtifactMismatch(error.to_string()))?;
            validate_structure_matches_system(system, &structure)?;
            for candidate in derive_nullspace_candidates(&structure)? {
                if self.layout.block(candidate.block).is_none() {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "structural nullspace candidate for field {} names a symbol several \
                         instances of this composed plan carry; `BlockNullspaceCandidate` is \
                         keyed by per-model symbol",
                        candidate.block
                    )));
                }
                nullspace_candidates.push(candidate);
            }
            structures.push(structure);
        }
        let structure = structures[0].clone();
        let binds = self.bind_outputs(&fields, &instance_variables, &mut output_constitutive)?;
        if let Some(((instance, output), _)) = output_constitutive.iter().next() {
            return Err(FinitumError::InvalidRealization(format!(
                "constitutive inputs are bound to output {output} of {instance}, which no bind \
                 of this plan reads"
            )));
        }
        let solver_layout = solver_block_layout_keyed(&self.layout, &self.system_ids)?;
        let digest = system_operator_digest(
            self,
            &constitutive_by_key,
            &stored_by_key,
            &equation_sign_by_block,
            &facet_regions,
            &binds,
        );
        Ok(SystemOperator {
            data: Arc::new(SystemOperatorData {
                plan: self.clone(),
                fields,
                instance_variables,
                quadrature,
                bindings,
                constitutive: constitutive_by_key,
                stored: stored_by_key,
                equation_sign: equation_sign_by_block,
                facet_regions,
                facet_geometries,
                structure,
                structures,
                nullspace_candidates,
                solver_layout,
                digest,
                symmetry_proof: std::sync::OnceLock::new(),
                binds,
            }),
        })
    }

    /// W8 lane F-MI: binds every same-mesh bind's producer output kernel (its single bundle,
    /// its non-basis inputs' closures from `output_constitutive`, and -- on the kernel-input
    /// path -- every Malleus composition Scientia emitted for it, validated and
    /// digest-checked, with its JVP composition rebuilt and checked against the recorded
    /// `jvp_digest`).
    fn bind_outputs(
        &self,
        fields: &BTreeMap<SysVarId, FieldElement>,
        instance_variables: &[BTreeMap<SymbolId, SysVarId>],
        output_constitutive: &mut BTreeMap<
            (InstanceId, scientia::OutputId),
            BTreeMap<TensorInputId, SystemConstitutiveInput>,
        >,
    ) -> Result<Vec<BoundBind>, FinitumError> {
        let Some(composed) = self.composed.as_ref() else {
            return Ok(Vec::new());
        };
        let mut binds = Vec::with_capacity(composed.binds.len());
        for bind in &composed.binds {
            let kernels = &composed.outputs[bind.output.index()];
            let integral = kernels.factorization.integrals[0].clone();
            let mut bound = bind_kernels(&kernels.factorization, kernels.kernels.clone())?;
            let bundle = bound.remove(&(integral.integral_index, 0)).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "output `{}` has no bound kernel for its integral",
                    bind.output_path
                ))
            })?;
            let producer_bound = composed
                .binds
                .iter()
                .filter(|other| other.consumer == bind.producer)
                .map(|other| (other.consumer_symbol, other.consumer_slot.as_str()))
                .collect::<Vec<_>>();
            let constitutive = output_constitutive
                .get(&(bind.producer, bind.output))
                .cloned()
                .unwrap_or_default();
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    if input.binding.evaluation.site != EvaluationSite::Cell {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "output `{}` reads its field {} at site {:?}; a bound output is a \
                             cell point function",
                            bind.output_path, input.binding.symbol, input.binding.evaluation.site
                        )));
                    }
                    let realized = instance_variables[bind.producer.0 as usize]
                        .get(&input.binding.symbol)
                        .is_some_and(|variable| fields.contains_key(variable));
                    if !realized {
                        return Err(FinitumError::ArtifactMismatch(format!(
                            "output `{}` reads field {} which its instance {} has not realized",
                            bind.output_path, input.binding.symbol, bind.producer
                        )));
                    }
                    continue;
                }
                let closed = constitutive.contains_key(&input.id);
                if let Some((_, slot)) = producer_bound
                    .iter()
                    .find(|(symbol, _)| *symbol == input.binding.symbol)
                {
                    if closed {
                        return Err(FinitumError::InvalidRealization(format!(
                            "output `{}` input {:?} is closed by the bind on `{slot}`; it must \
                             not also be bound as a closure",
                            bind.output_path, input.id
                        )));
                    }
                    continue;
                }
                if !closed {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "output `{}` (read by the bind on `{}`) input {:?} requires a \
                         caller-supplied SystemConstitutiveInput closure \
                         (`SystemConstitutiveInput::try_new_for_output`)",
                        bind.output_path, bind.consumer_slot, input.id
                    )));
                }
            }
            for (id, closure) in &constitutive {
                let declared = integral.primal.inputs.iter().find(|input| input.id == *id);
                match declared {
                    Some(input)
                        if input.source != InputSourceRequirement::Basis
                            && closure.integral_index == integral.integral_index => {}
                    _ => {
                        return Err(FinitumError::InvalidRealization(format!(
                            "constitutive input `{}` names integral {} input {id:?}, which \
                             output `{}` does not declare as a non-basis input",
                            closure.identity, closure.integral_index, bind.output_path
                        )));
                    }
                }
            }
            let mut compositions = Vec::with_capacity(bind.compositions.len());
            for &index in &bind.compositions {
                let composition = &composed.compositions[index];
                let row = self
                    .row_index(SysResId(composition.row.0))
                    .expect("checked by ComposedSystem::new");
                let (row_record, block) = self.row(row);
                let consumer_bundle = block
                    .kernels
                    .bundles
                    .iter()
                    .find(|bundle| {
                        bundle.integral_index == composition.consumer_integral_index
                            && bundle.output_index == composition.consumer_output_index
                    })
                    .ok_or_else(|| {
                        FinitumError::ArtifactMismatch(format!(
                            "bind `{}` composition names integral {} output {} which equation \
                             `{}` has no kernel bundle for",
                            bind.consumer_slot,
                            composition.consumer_integral_index,
                            composition.consumer_output_index,
                            row_record.path
                        ))
                    })?;
                let digest = malleus::composition_digest(&composition.composition);
                let recorded = &composition.digest;
                if digest.hex != recorded.hex {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "bind `{}` composition digest {} does not match its recorded digest {}",
                        bind.consumer_slot, digest.hex, recorded.hex
                    )));
                }
                let validated = malleus::validate_composition(composition.composition.clone())
                    .map_err(|error| FinitumError::KernelValidation(error.to_string()))?;
                // The recorded JVP composition (Scientia's request: the producer's active basis
                // operands -> the consumer output), rebuilt and checked; see
                // `SystemOperator::apply_block_cell` for why it is not the executed tangent.
                let producer_program = &kernels.factorization.integrals[0].primal;
                let independent = bundle
                    .bundle
                    .primal_inputs
                    .iter()
                    .filter(|binding| {
                        producer_program.inputs.iter().any(|input| {
                            input.id == binding.input && input.role == TensorInputRole::Active
                        })
                    })
                    .map(|binding| malleus::StageOperand::new(0, binding.operand))
                    .collect::<Vec<_>>();
                let jvp = malleus::differentiate_composition(
                    &composition.composition,
                    &malleus::CompositionDerivativeRequest {
                        mode: malleus::DerivativeMode::Jvp,
                        independent_operands: independent,
                        dependent_operands: vec![malleus::StageOperand::new(
                            1,
                            consumer_bundle.primal_output,
                        )],
                    },
                )
                .map_err(|error| FinitumError::KernelValidation(error.to_string()))?;
                let jvp_digest = malleus::composition_digest(&jvp.composition);
                if jvp_digest.hex != composition.jvp_digest.hex {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "bind `{}` JVP composition digest {} does not match its recorded digest \
                         {}",
                        bind.consumer_slot, jvp_digest.hex, composition.jvp_digest.hex
                    )));
                }
                if composition.producer_operand
                    != malleus::StageOperand::new(0, bundle.bundle.primal_output)
                {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "bind `{}` composition shares stage-0 operand {:?}, the producer output \
                         is operand {:?}",
                        bind.consumer_slot,
                        composition.producer_operand,
                        bundle.bundle.primal_output
                    )));
                }
                compositions.push(BoundComposition {
                    row,
                    integral_index: composition.consumer_integral_index,
                    output_index: composition.consumer_output_index,
                    composition: composition.clone(),
                    executable: malleus::ExecutableComposition::reference(validated),
                });
            }
            binds.push(BoundBind {
                bind: bind.clone(),
                integral,
                bound: bundle,
                constitutive,
                compositions,
            });
        }
        // An output can feed several consumers. Its closures must remain available while
        // every outgoing bind is built, then be removed once from the admission map.
        for bind in &composed.binds {
            output_constitutive.remove(&(bind.producer, bind.output));
        }
        Ok(binds)
    }
}

/// One same-mesh bind bound to executable kernels (W8 lane F-MI): the producer output's
/// single integral and bundle, its closures, and its kernel-input compositions.
#[derive(Debug)]
struct BoundBind {
    bind: RealizedBind,
    integral: IntegralOperatorFactorization,
    bound: BoundBundle,
    constitutive: BTreeMap<TensorInputId, SystemConstitutiveInput>,
    compositions: Vec<BoundComposition>,
}

/// One validated Malleus `BindComposition`, keyed by the consumer row/integral/output it
/// realizes.
#[derive(Debug)]
struct BoundComposition {
    row: usize,
    integral_index: usize,
    output_index: usize,
    composition: scientia::BindComposition,
    executable: malleus::ExecutableComposition,
}

#[derive(Debug)]
struct SystemOperatorData {
    plan: SystemRealizationPlan,
    /// Every realized field by system variable (W8 F-MI); `SysVarId(symbol.0)` on a
    /// one-instance plan, so the iteration order is the per-model symbol order it always was.
    fields: BTreeMap<SysVarId, FieldElement>,
    /// Per instance, its `symbol -> variable` map.
    instance_variables: Vec<BTreeMap<SymbolId, SysVarId>>,
    /// Shared quadrature table every field's basis is tabulated at (or, for an RT0 field,
    /// evaluated at directly -- see [`build_field_elements`]'s doc comment); the per-cell
    /// per-block integration loop indexes into this rather than any one field's own table, since
    /// an RT0 [`FieldElement`] carries no [`PreparedElement`] of its own.
    quadrature: Vec<QuadraturePoint>,
    /// Keyed by row index (the block index of a one-instance plan).
    bindings: BTreeMap<usize, BTreeMap<(usize, usize), BoundBundle>>,
    constitutive: BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
    /// Stored quadrature-point tables (SC-W1 system-path parity), keyed like `constitutive`.
    stored: BTreeMap<(usize, usize, TensorInputId), ExternalInput>,
    /// Per-row equation orientation (`1.0` or `-1.0`, absent means `1.0`); see
    /// `SystemRealizationPlan::bind_kernels`'s `equation_sign` parameter.
    equation_sign: BTreeMap<usize, f64>,
    /// Mission item 2's exterior-facet extension: caller-resolved region -> facet-id lists (see
    /// `SystemRealizationPlan::bind_kernels_with_facets`), and their precomputed
    /// `crate::realization::FacetGeometry` (cell + local facet index; reused from GX-C4
    /// unchanged).
    facet_regions: BTreeMap<RegionId, Vec<FacetId>>,
    facet_geometries: BTreeMap<FacetId, FacetGeometry>,
    /// The first instance's structure (the whole structure of a one-instance plan).
    structure: OperatorStructure,
    /// Every instance's structure, by instance index.
    structures: Vec<OperatorStructure>,
    nullspace_candidates: Vec<BlockNullspaceCandidate>,
    solver_layout: methodus::BlockLayout,
    digest: Digest,
    /// Lazily proven, then cached, symmetry declaration used only when `equation_sign` is
    /// nontrivial (mirrors `RealizationPlan`'s own `symmetry_proof` cache).
    symmetry_proof: std::sync::OnceLock<OperatorSymmetry>,
    /// The same-mesh binds in dependency order (empty on a one-instance plan).
    binds: Vec<BoundBind>,
}

/// Executable, Scientia-form/Malleus-kernel-driven realization of an entire
/// [`OperatorSystem`] (SV2-B4 continuation of `crate::mixed::MixedOperator`'s structural
/// composition, but driven by real bound kernels instead of hand-written local-matrix builders).
///
/// Composes every block's present `(row, column)` coordinates cell-by-cell into one monolithic
/// action over the plan's [`BlockLayout`]. Batch P: the residual, JVP, and VJP are evaluated at
/// an actual `(t, u, u_t)` linearization point ([`Self::residual`],
/// [`Self::jacobian_vector_product`], [`Self::vector_jacobian_product_shifted`],
/// [`Self::linearize`]) with the GX-A3 chain rule through every constitutive closure; the
/// `LinearOperator` view ([`Self::apply_action`]) is the JVP at the zero point -- the same
/// "globally linear FC6 scope" convention `RealizationPlan::MatrixFreeOperator` and
/// `MixedOperator::apply_action` document, kept for the steady runner. Declares `symmetry()`/
/// `properties()` from Scientia's structural `OperatorStructure` (C5.4/C5.5) rather than an
/// unconditional or proof-by-assembly claim.
#[derive(Clone, Debug)]
pub struct SystemOperator {
    data: Arc<SystemOperatorData>,
}

impl SystemOperator {
    pub fn plan(&self) -> &SystemRealizationPlan {
        &self.data.plan
    }

    pub fn layout(&self) -> &BlockLayout {
        self.data.plan.layout()
    }

    pub fn dimension(&self) -> usize {
        self.layout().extent()
    }

    /// Content-addressed identity over this operator's realized content: the plan's own shape
    /// digest plus every bound [`SystemConstitutiveInput`]'s identity (mission item 7, mirroring
    /// `RealizationPlan::digest()`). The bound Malleus kernels and per-field DOF maps add no new
    /// information beyond the plan's digest, since both are pure deterministic functions of
    /// `system`/`mesh`/`layout`, which the plan's own digest already covers.
    pub fn digest(&self) -> &Digest {
        &self.data.digest
    }

    /// Scientia's structural `OperatorStructure` (C5.4) this operator's `symmetry()`/
    /// `properties()`/[`Self::nullspace_candidates`] are derived from.
    pub fn structure(&self) -> &OperatorStructure {
        &self.data.structure
    }

    /// W8 F-MI: the structural `OperatorStructure` of one instance of a composed plan (the
    /// per-model structure the instance's `/1` artifact derives); [`Self::structure`] is
    /// instance 0's.
    pub fn instance_structure(&self, instance: InstanceId) -> Option<&OperatorStructure> {
        self.data.structures.get(instance.0 as usize)
    }

    /// Auto-derived nullspace candidates (mission item 9): representation-only declarations, not
    /// yet resolved against a concrete essential-constraint set. Resolve one against
    /// [`Self::layout`] via [`BlockNullspaceCandidate::resolve`].
    pub fn nullspace_candidates(&self) -> &[BlockNullspaceCandidate] {
        &self.data.nullspace_candidates
    }

    /// The DOF map of a realized field by per-model symbol; `None` on a composed plan when
    /// several instances carry the symbol (use [`Self::dof_map_by_variable`]).
    pub fn dof_map(&self, field: SymbolId) -> Option<&DofMap> {
        let variable = self.layout().block(field)?.variable;
        self.dof_map_by_variable(variable)
    }

    /// The DOF map of a realized field by system variable (W8 F-MI).
    pub fn dof_map_by_variable(&self, variable: SysVarId) -> Option<&DofMap> {
        self.data.fields.get(&variable).map(|field| &field.dofs)
    }

    /// W8 F-MI: every same-mesh bind this operator realizes, in dependency order (empty on a
    /// one-instance plan).
    pub fn binds(&self) -> Vec<SystemBindReceipt> {
        let Some(composed) = self.data.plan.composed.as_ref() else {
            return Vec::new();
        };
        self.data
            .binds
            .iter()
            .map(|bound| {
                let bind = &bound.bind;
                let mut rows = BTreeSet::new();
                let mut columns = BTreeSet::new();
                for block in &composed.operator.blocks {
                    if let scientia::BlockConstruction::Composed { bind: index, .. } =
                        &block.construction
                    {
                        if *index == bind.index {
                            rows.insert(SysResId(block.row.0));
                            columns.insert(SysVarId(block.column.0));
                        }
                    }
                }
                SystemBindReceipt {
                    consumer_slot: bind.consumer_slot.clone(),
                    consumer: bind.consumer,
                    producer: bind.producer,
                    output: bind.output_path.clone(),
                    path: bind.path,
                    rows: rows.into_iter().collect(),
                    columns: columns.into_iter().collect(),
                    compositions: bound
                        .compositions
                        .iter()
                        .map(|composition| composition.composition.digest.clone())
                        .collect(),
                    jvp_compositions: bound
                        .compositions
                        .iter()
                        .map(|composition| composition.composition.jvp_digest.clone())
                        .collect(),
                }
            })
            .collect()
    }

    /// Matrix-free monolithic action `output = A * input` at the zero linearization point
    /// (`t = 0`, zero state and rate, zero rate direction): exactly
    /// [`Self::jacobian_vector_product`] at that point, the linear view the steady system
    /// runner solves with. A nonlinear system's operator at another state is [`Self::linearize`].
    pub fn apply_action(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        let zero = vec![0.0; self.dimension()];
        self.jacobian_vector_product(0.0, &zero, &zero, input, &zero, output)
    }

    /// Batch P: the residual `R(t, u, u_t)` of the whole system in the layout's physical
    /// coordinates (no constraint rows; see [`ReducedSystemOperator::residual`]), executing
    /// every block's bound PRIMAL kernel at the actual state and rate: every field's basis
    /// inputs are gathered from `state` (from `state_rate` for `TimeDerivative` inputs) and
    /// every constitutive closure sees the point's actual active values and time. Admitted
    /// exterior-facet integrals contribute their input-free PRIMAL value. Each block's row is
    /// scaled by its `equation_sign`.
    pub fn residual(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_point(time, state, state_rate, output)?;
        output.fill(0.0);
        self.apply_cells(time, state, state_rate, SystemAction::Primal, output)?;
        self.apply_facets(output, FacetAction::Primal)?;
        validate_finite("system residual", output)
    }

    /// Batch P: the JVP `dR/du * state_direction + dR/du_t * rate_direction` at `(t, u, u_t)`,
    /// executing every block's bound JVP kernel plus the GX-A3 chain rule through every
    /// constitutive closure's exact `direction` (composed by the generated parameter-JVP
    /// kernel) -- so a property tangent through another field's value (`ka = ka(b)` inside the
    /// `a` equation) lands in the off-diagonal block without any expression rewriting.
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
        self.validate_point(time, state, state_rate, output)?;
        self.validate_vector("system JVP state direction", state_direction)?;
        self.validate_vector("system JVP rate direction", rate_direction)?;
        output.fill(0.0);
        self.apply_cells(
            time,
            state,
            state_rate,
            SystemAction::Jvp {
                state_direction,
                rate_direction,
            },
            output,
        )?;
        // Every admitted exterior-facet integral has no basis input, so its directional
        // derivative is identically zero; executed anyway for uniformity (see `apply_facets`).
        self.apply_facets(output, FacetAction::Jvp)?;
        validate_finite("system JVP", output)
    }

    /// SV1-C1 over the system: the exact transpose of [`Self::jacobian_vector_product`] with the
    /// rate direction held at zero; see [`Self::vector_jacobian_product_shifted`].
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

    /// SV1-C1 over the system: the exact transpose of `x -> jacobian_vector_product(x,
    /// rate_shift * x)`, executing every block's bound VJP kernel with the row field's adjoint
    /// (scaled by the block's `equation_sign`) as the seed and scattering each cotangent
    /// through its own column field's basis -- the multi-field analogue of
    /// `RealizationPlan::vector_jacobian_product_shifted`, sharing its kernel-level VJP
    /// execution, parameter-cotangent probing (a constitutive closure's chain rule is inverted
    /// by probing its `direction` with unit active perturbations, exactly as a dynamic external
    /// input's is), and rate-shift scatter rule. Admitted exterior-facet integrals carry no
    /// active input, so their transpose contribution is identically zero (validated at bind
    /// time).
    #[allow(clippy::too_many_arguments)]
    pub fn vector_jacobian_product_shifted(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_point(time, state, state_rate, output)?;
        self.validate_vector("system VJP adjoint", adjoint)?;
        if !rate_shift.is_finite() {
            return Err(FinitumError::InvalidRealization(
                "rate shift must be finite".into(),
            ));
        }
        output.fill(0.0);
        self.apply_cells(
            time,
            state,
            state_rate,
            SystemAction::Vjp {
                adjoint,
                rate_shift,
            },
            output,
        )?;
        validate_finite("system VJP", output)
    }

    /// The affine forcing contribution the zero-point linear view ([`Self::apply_action`])
    /// cannot see: `-R(0, 0, 0)`, the negated PRIMAL residual at zero state and rate, so that
    /// for a globally linear system `apply_action(u) == load_vector()` is the zero-Dirichlet
    /// weak-form equation (`PRIMAL(0) = a(0, v) - L(v) = -L(v)`, the same convention
    /// `RealizationPlan::load_vector` uses). Each block's contribution carries its own
    /// `equation_sign`, consistent with `apply_action`. Full [`Self::dimension`]-length and
    /// unconstrained; [`ReducedSystemOperator::load_vector`] composes the Dirichlet lifting.
    /// A system with no bound source produces the exact zero vector.
    pub fn load_vector(&self) -> Result<Vec<f64>, FinitumError> {
        let dimension = self.dimension();
        let zero = vec![0.0; dimension];
        let mut output = vec![0.0; dimension];
        self.residual(0.0, &zero, &zero, &mut output)?;
        for value in &mut output {
            *value = -*value;
        }
        validate_finite("system operator load vector", &output)?;
        Ok(output)
    }

    /// SV1-C1: the Jacobian `dR/du + rate_shift * dR/du_t` at one linearization point in the
    /// layout's physical coordinates as a Methodus `LinearOperator + TransposableOperator +
    /// BlockLinearOperator`; [`ReducedSystemOperator::linearize`] is the essential-
    /// constraint-eliminated one.
    pub fn linearize(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        rate_shift: f64,
    ) -> Result<LinearizedSystemOperator, FinitumError> {
        let probe = vec![0.0; self.dimension()];
        self.validate_point(time, state, state_rate, &probe)?;
        validate_rate_shift(rate_shift)?;
        Ok(LinearizedSystemOperator {
            operator: self.clone(),
            reduced: None,
            time,
            state: state.to_vec(),
            state_rate: state_rate.to_vec(),
            rate_shift,
        })
    }

    fn validate_point(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &[f64],
    ) -> Result<(), FinitumError> {
        if !time.is_finite() {
            return Err(FinitumError::InvalidRealization(
                "system operator evaluation time must be finite".into(),
            ));
        }
        self.validate_vector("system operator state", state)?;
        self.validate_vector("system operator state rate", state_rate)?;
        if output.len() != self.dimension() {
            return Err(FinitumError::InvalidRealization(format!(
                "system operator output expects length {}, got {}",
                self.dimension(),
                output.len()
            )));
        }
        Ok(())
    }

    fn validate_vector(&self, label: &str, vector: &[f64]) -> Result<(), FinitumError> {
        if vector.len() != self.dimension() {
            return Err(FinitumError::InvalidRealization(format!(
                "{label} expects length {}, got {}",
                self.dimension(),
                vector.len()
            )));
        }
        validate_finite(label, vector)
    }

    fn apply_cells(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        action: SystemAction<'_>,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.apply_cells_of(time, state, state_rate, action, output, None)
    }

    /// As [`Self::apply_cells`], restricted to one equation block (by index in
    /// `system.blocks`) when `only_block` is set -- the per-row half of a public block action.
    fn apply_cells_of(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        action: SystemAction<'_>,
        output: &mut [f64],
        only_block: Option<usize>,
    ) -> Result<(), FinitumError> {
        for cell in 0..self.data.plan.mesh().cells().len() {
            self.apply_cell_blocks(cell, time, state, state_rate, action, output, only_block)?;
        }
        Ok(())
    }

    /// One cell's contribution of every block (or of `only_block`) to `output`.
    #[allow(clippy::too_many_arguments)]
    fn apply_cell_blocks(
        &self,
        cell: usize,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        action: SystemAction<'_>,
        output: &mut [f64],
        only_block: Option<usize>,
    ) -> Result<(), FinitumError> {
        let geometry = CellGeometry::new(self.data.plan.mesh(), CellId(cell))?;
        let affine = AffineMap::from_cell(self.data.plan.mesh(), CellId(cell))?;
        for (row_index, row, block) in self.data.plan.rows() {
            if only_block.is_some_and(|only| only != row_index) {
                continue;
            }
            self.apply_block_cell(
                row_index, row, block, cell, &geometry, &affine, time, state, state_rate, action,
                output,
            )?;
        }
        Ok(())
    }

    /// The shared cell quadrature table every field of this operator is integrated with
    /// (degree-4-exact on triangles, degree-2 on tetrahedra); stored tables
    /// ([`SystemExternalInput`]) and distributed coefficients ([`crate::CoefficientLayout`]) are
    /// laid out over it.
    pub fn quadrature(&self) -> &[QuadraturePoint] {
        &self.data.quadrature
    }

    /// The layout-wide DOFs each cell touches: every realized field's cell restriction, offset
    /// into its block, in field (`SymbolId`) order -- the element restriction a cell-local
    /// matrix of the whole system is indexed by.
    fn cell_restrictions(&self) -> Vec<ElementRestriction> {
        let layout = self.layout();
        (0..self.data.plan.mesh().cells().len())
            .map(|cell| ElementRestriction {
                dofs: self
                    .data
                    .fields
                    .iter()
                    .flat_map(|(&variable, field)| {
                        let offset = layout
                            .block_by_variable(variable)
                            .expect("realized field implies a layout block")
                            .offset;
                        field.dofs.restrictions()[cell]
                            .dofs
                            .iter()
                            .map(move |dof| crate::DofId(offset + dof.0))
                    })
                    .collect(),
            })
            .collect()
    }

    /// Resolves `coefficient` to its block index, integral, and stored table, refusing an
    /// unknown residual/integral/input, a facet integral, or a closure-bound (constitutive)
    /// input, whose values are not a design vector.
    fn coefficient_binding(
        &self,
        coefficient: &SystemDistributedCoefficient,
    ) -> Result<(usize, &IntegralOperatorFactorization, &ExternalInput), FinitumError> {
        let block_index = self
            .data
            .plan
            .row_index(coefficient.residual)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "distributed coefficient names residual {} which the system does not carry",
                    coefficient.residual
                ))
            })?;
        let (_, block) = self.data.plan.row(block_index);
        let integral = block
            .factorization
            .integrals
            .iter()
            .find(|integral| integral.integral_index == coefficient.coefficient.integral_index)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "distributed coefficient names absent integral {} of residual {}",
                    coefficient.coefficient.integral_index, coefficient.residual
                ))
            })?;
        if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
            return Err(FinitumError::UnsupportedRealization(format!(
                "distributed coefficients are realized on cell integrals only; integral {} has \
                 measure {:?}",
                integral.integral_index, integral.measure
            )));
        }
        let key = (
            block_index,
            integral.integral_index,
            coefficient.coefficient.input,
        );
        match self.data.stored.get(&key) {
            Some(stored) => Ok((block_index, integral, stored)),
            None if self.data.constitutive.contains_key(&key) => {
                Err(FinitumError::UnsupportedRealization(format!(
                    "residual {} integral {} input {:?} is a constitutive closure, not a stored \
                     distributed coefficient",
                    coefficient.residual, key.1, key.2
                )))
            }
            None => Err(FinitumError::MissingExternalInput {
                integral: key.1,
                input: key.2,
            }),
        }
    }

    /// SC-W1 system-path parity (a): the design-space extent of `coefficient` under its
    /// layout over this operator's shared quadrature (`RealizationPlan::coefficient_dimension`).
    pub fn coefficient_dimension(
        &self,
        coefficient: &SystemDistributedCoefficient,
    ) -> Result<usize, FinitumError> {
        let (_, _, stored) = self.coefficient_binding(coefficient)?;
        coefficient.coefficient.layout.dimension_at(
            self.data.plan.mesh(),
            self.data.quadrature.len(),
            stored.component_count(),
        )
    }

    /// SC-W1 system-path parity (a): `dR/dp * direction` at `(t, u, u_t)` in the layout's
    /// physical coordinates -- the coefficient's residual's bound parameter (frozen-input) JVP
    /// kernels with the direction routed to that input only, scaled by the row's
    /// `equation_sign`; the exact system counterpart of
    /// `RealizationPlan::coefficient_jacobian_vector_product`.
    #[allow(clippy::too_many_arguments)]
    pub fn coefficient_jacobian_vector_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        coefficient: &SystemDistributedCoefficient,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_point(time, state, state_rate, output)?;
        let (block_index, integral, stored) = self.coefficient_binding(coefficient)?;
        let components = stored.component_count();
        let expected = self.coefficient_dimension(coefficient)?;
        if direction.len() != expected {
            return Err(FinitumError::InvalidRealization(format!(
                "coefficient direction has length {}, expected {expected}",
                direction.len()
            )));
        }
        validate_finite("coefficient direction", direction)?;
        let (row, block) = self.data.plan.row(block_index);
        let fields = self.instance_fields(row.instance);
        let (row_variable, row_field) = fields.field(block.row)?;
        let row_block = self
            .layout()
            .block_by_variable(row_variable)
            .expect("row field implies a layout block");
        let sign = self
            .data
            .equation_sign
            .get(&block_index)
            .copied()
            .unwrap_or(1.0);
        let mesh = self.data.plan.mesh();
        let bindings = &self.data.bindings[&block_index];
        output.fill(0.0);
        for cell in 0..mesh.cells().len() {
            let geometry = CellGeometry::new(mesh, CellId(cell))?;
            let affine = AffineMap::from_cell(mesh, CellId(cell))?;
            let local_state = self.gather_cell(cell, state);
            let local_rate = self.gather_cell(cell, state_rate);
            let row_restriction = &row_field.dofs.restrictions()[cell];
            let mut local_output = vec![0.0; row_restriction.dofs.len()];
            for (point, quadrature_point) in self.data.quadrature.iter().enumerate() {
                let reference_point = &quadrature_point.coordinates;
                let scale = quadrature_point.weight * geometry.determinant();
                let weights = coefficient.coefficient.layout.weights_at(
                    mesh,
                    &self.data.quadrature,
                    cell,
                    point,
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
                let bound_values = self.bound_point_values(
                    cell,
                    point,
                    &geometry,
                    &affine,
                    reference_point,
                    time,
                    &local_state,
                    &local_rate,
                    None,
                )?;
                let point_bindings = self.row_bindings(block_index, row.instance, &bound_values);
                let (inputs, _) = point_inputs_system(
                    fields,
                    integral,
                    cell,
                    point,
                    &geometry,
                    &affine,
                    reference_point,
                    time,
                    &local_state,
                    &local_rate,
                    &point_bindings,
                )?;
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = bound_kernel(bindings, &row.path, integral, output_index)?;
                    let Some(point_output) = execute_parameter_jvp_values(
                        bound,
                        &inputs,
                        coefficient.coefficient.input,
                        &point_direction,
                    )?
                    else {
                        continue;
                    };
                    apply_field_basis_adjoint(
                        row_field,
                        &geometry,
                        &affine,
                        cell,
                        point,
                        reference_point,
                        &qoutput.binding.evaluation.derivative,
                        &point_output,
                        scale,
                        &mut local_output,
                    )?;
                }
            }
            for (local_index, dof) in row_restriction.dofs.iter().enumerate() {
                output[row_block.offset + dof.0] += sign * local_output[local_index];
            }
        }
        validate_finite("system coefficient JVP", output)
    }

    /// SC-W1 system-path parity (a): `(dR/dp)^T * adjoint` -- the exact transpose of
    /// [`Self::coefficient_jacobian_vector_product`], each quadrature point's parameter
    /// cotangent (the bound parameter kernel's point-local Jacobian contracted against the
    /// row field's sign-scaled test adjoint) accumulated into the caller-owned design space
    /// through the transpose of the layout's interpolation.
    #[allow(clippy::too_many_arguments)]
    pub fn coefficient_vector_jacobian_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        coefficient: &SystemDistributedCoefficient,
        adjoint: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let probe = vec![0.0; self.dimension()];
        self.validate_point(time, state, state_rate, &probe)?;
        self.validate_vector("system coefficient VJP adjoint", adjoint)?;
        let (block_index, integral, stored) = self.coefficient_binding(coefficient)?;
        let components = stored.component_count();
        let expected = self.coefficient_dimension(coefficient)?;
        if output.len() != expected {
            return Err(FinitumError::InvalidRealization(format!(
                "coefficient VJP output has length {}, expected {expected}",
                output.len()
            )));
        }
        let (row, block) = self.data.plan.row(block_index);
        let fields = self.instance_fields(row.instance);
        let (row_variable, row_field) = fields.field(block.row)?;
        let row_block = self
            .layout()
            .block_by_variable(row_variable)
            .expect("row field implies a layout block");
        let sign = self
            .data
            .equation_sign
            .get(&block_index)
            .copied()
            .unwrap_or(1.0);
        let mesh = self.data.plan.mesh();
        let bindings = &self.data.bindings[&block_index];
        output.fill(0.0);
        for cell in 0..mesh.cells().len() {
            let geometry = CellGeometry::new(mesh, CellId(cell))?;
            let affine = AffineMap::from_cell(mesh, CellId(cell))?;
            let local_state = self.gather_cell(cell, state);
            let local_rate = self.gather_cell(cell, state_rate);
            let row_restriction = &row_field.dofs.restrictions()[cell];
            let local_adjoint = row_restriction
                .dofs
                .iter()
                .map(|dof| sign * adjoint[row_block.offset + dof.0])
                .collect::<Vec<_>>();
            for (point, quadrature_point) in self.data.quadrature.iter().enumerate() {
                let reference_point = &quadrature_point.coordinates;
                let scale = quadrature_point.weight * geometry.determinant();
                let weights = coefficient.coefficient.layout.weights_at(
                    mesh,
                    &self.data.quadrature,
                    cell,
                    point,
                )?;
                let bound_values = self.bound_point_values(
                    cell,
                    point,
                    &geometry,
                    &affine,
                    reference_point,
                    time,
                    &local_state,
                    &local_rate,
                    None,
                )?;
                let point_bindings = self.row_bindings(block_index, row.instance, &bound_values);
                let (inputs, _) = point_inputs_system(
                    fields,
                    integral,
                    cell,
                    point,
                    &geometry,
                    &affine,
                    reference_point,
                    time,
                    &local_state,
                    &local_rate,
                    &point_bindings,
                )?;
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = bound_kernel(bindings, &row.path, integral, output_index)?;
                    let output_components = component_count(&qoutput.shape)?;
                    let seed = gather_field_test_adjoint(
                        row_field,
                        &geometry,
                        &affine,
                        cell,
                        point,
                        reference_point,
                        &qoutput.binding.evaluation.derivative,
                        output_components,
                        &local_adjoint,
                    )?;
                    let cotangents = point_parameter_cotangents(bound, &inputs, &seed)?;
                    let Some(cotangent) = cotangents.get(&coefficient.coefficient.input) else {
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
        validate_finite("system coefficient VJP", output)
    }

    /// SC-W1 system-path parity (b): the zero-point linear view as one cell-local matrix per
    /// cell over [`Self::cell_restrictions`] (unit-column probing of the cell's own JVP), the
    /// counterpart of `RealizationPlan::element_assembly`; refuses admitted exterior-facet
    /// integrals exactly as the single-model path does. `constraints` are the essential rows
    /// the operator is reduced by (empty for the physical operator).
    fn element_assembly_with(
        &self,
        constraints: ConstraintSet,
        lane_width: usize,
    ) -> Result<ElementAssemblyOperator, FinitumError> {
        if !self.data.facet_regions.is_empty() {
            return Err(FinitumError::UnsupportedRealization(
                "element assembly does not yet cover exterior facet integrals (GX-C4)".into(),
            ));
        }
        let dimension = self.dimension();
        let zero = vec![0.0; dimension];
        let restrictions = self.cell_restrictions();
        let mut local_matrices = Vec::with_capacity(restrictions.len());
        for (cell, restriction) in restrictions.iter().enumerate() {
            let local_dimension = restriction.dofs.len();
            let mut matrix = vec![0.0; local_dimension * local_dimension];
            for (column, dof) in restriction.dofs.iter().enumerate() {
                let mut direction = vec![0.0; dimension];
                direction[dof.0] = 1.0;
                let mut output = vec![0.0; dimension];
                self.apply_cell_blocks(
                    cell,
                    0.0,
                    &zero,
                    &zero,
                    SystemAction::Jvp {
                        state_direction: &direction,
                        rate_direction: &zero,
                    },
                    &mut output,
                    None,
                )?;
                for (row, row_dof) in restriction.dofs.iter().enumerate() {
                    matrix[row * local_dimension + column] = output[row_dof.0];
                }
            }
            local_matrices.push(matrix);
        }
        ElementAssemblyOperator::new(
            dimension,
            self.data.plan.artifact_digest().clone(),
            restrictions,
            constraints,
            local_matrices,
            lane_width,
        )
    }

    /// Element assembly of the physical (unreduced) operator; see [`ReducedSystemOperator::element_assembly`] for the constrained one.
    pub fn element_assembly(
        &self,
        lane_width: usize,
    ) -> Result<ElementAssemblyOperator, FinitumError> {
        self.element_assembly_with(ConstraintSet::new(self.dimension(), [])?, lane_width)
    }

    fn partial_assembly_with(
        &self,
        constraints: ConstraintSet,
        lane_width: usize,
    ) -> Result<SystemPartialAssemblyOperator, FinitumError> {
        let mesh = self.data.plan.mesh();
        if let Some(bound) = self.data.binds.first() {
            let consumer_row = self
                .data
                .plan
                .rows()
                .find(|(_, row, _)| row.instance == bound.bind.consumer)
                .map(|(_, row, _)| row.path.clone())
                .unwrap_or_default();
            return Err(FinitumError::RepresentationUnsupported {
                representation: RepresentationKind::PartialAssembly,
                equation: consumer_row,
                integral: 0,
                input: None,
                reason: format!(
                    "the same-mesh bind on `{}` ({} <- {}) is a state-dependent point chain; \
                     partial assembly stores state-independent point Jacobians",
                    bound.bind.consumer_slot, bound.bind.consumer_slot, bound.bind.output_path
                ),
            });
        }
        if let Some((&(block_index, integral_index, input), binding)) =
            self.data.constitutive.iter().next()
        {
            return Err(FinitumError::RepresentationUnsupported {
                representation: RepresentationKind::PartialAssembly,
                equation: self.data.plan.row(block_index).0.path.clone(),
                integral: integral_index,
                input: Some(input),
                reason: format!(
                    "the constitutive closure `{}` is bound to this input; partial assembly \
                     stores state-independent point Jacobians, so it requires every non-basis \
                     input to be a stored table (the single-model path refuses its dynamic \
                     inputs the same way)",
                    binding.identity
                ),
            });
        }
        for (_, row, block) in self.data.plan.rows() {
            if let Some(integral) = block
                .factorization
                .integrals
                .iter()
                .find(|integral| !matches!(integral.measure, SemanticMeasure::Cell { .. }))
            {
                return Err(FinitumError::RepresentationUnsupported {
                    representation: RepresentationKind::PartialAssembly,
                    equation: row.path.clone(),
                    integral: integral.integral_index,
                    input: None,
                    reason: format!(
                        "partial assembly does not yet cover {:?} integrals (GX-C4)",
                        integral.measure
                    ),
                });
            }
        }
        let zero = vec![0.0; self.dimension()];
        let mut point_actions = Vec::with_capacity(mesh.cells().len());
        for cell in 0..mesh.cells().len() {
            let geometry = CellGeometry::new(mesh, CellId(cell))?;
            let affine = AffineMap::from_cell(mesh, CellId(cell))?;
            let local_zero = self.gather_cell(cell, &zero);
            let mut cell_actions = Vec::new();
            let no_bound: Vec<BoundPointValue> = Vec::new();
            for (block_index, row, block) in self.data.plan.rows() {
                let fields = self.instance_fields(row.instance);
                let row_variable = fields.variable(block.row)?;
                let sign = self
                    .data
                    .equation_sign
                    .get(&block_index)
                    .copied()
                    .unwrap_or(1.0);
                let bindings = &self.data.bindings[&block_index];
                for integral in &block.factorization.integrals {
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
                        .cloned()
                        .collect::<Vec<_>>();
                    let input_components = active_inputs
                        .iter()
                        .map(|input| component_count(&input.shape))
                        .sum::<Result<usize, _>>()?;
                    if input_components == 0 {
                        continue;
                    }
                    for (point, quadrature_point) in self.data.quadrature.iter().enumerate() {
                        let reference_point = &quadrature_point.coordinates;
                        let scale = sign * quadrature_point.weight * geometry.determinant();
                        let point_bindings =
                            self.row_bindings(block_index, row.instance, &no_bound);
                        let (inputs, _) = point_inputs_system(
                            fields,
                            integral,
                            cell,
                            point,
                            &geometry,
                            &affine,
                            reference_point,
                            0.0,
                            &local_zero,
                            &local_zero,
                            &point_bindings,
                        )?;
                        for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                            let bound = bound_kernel(bindings, &row.path, integral, output_index)?;
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
                                        directions
                                            .get_mut(&input.id)
                                            .expect("input was inserted")[selected - offset] = 1.0;
                                        break;
                                    }
                                    offset += count;
                                }
                                columns.push(execute_jvp_values(bound, &inputs, &directions)?);
                            }
                            let output_components = component_count(&qoutput.shape)?;
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
                            cell_actions.push(SystemPartialPointAction {
                                row: row_variable,
                                instance: row.instance,
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
            }
            point_actions.push(cell_actions);
        }
        let batches = CellBatchLayout::new(mesh.cells().len(), lane_width)?;
        Ok(SystemPartialAssemblyOperator {
            operator: self.clone(),
            constraints,
            point_actions,
            batches,
        })
    }

    /// SC-W1 system-path parity (b): the zero-point linear view as stored per-quadrature-point
    /// Jacobians (from the concatenated active field evaluations of each block integral output
    /// to that output) applied through the fields' own basis actions -- the counterpart of
    /// `RealizationPlan::partial_assembly`, exact for a system whose inputs are all stored
    /// tables or basis fields; refuses a bound constitutive closure and exterior-facet integrals
    /// exactly as the single-model path refuses dynamic inputs and facets.
    pub fn partial_assembly(
        &self,
        lane_width: usize,
    ) -> Result<SystemPartialAssemblyOperator, FinitumError> {
        self.partial_assembly_with(ConstraintSet::new(self.dimension(), [])?, lane_width)
    }

    /// SC-W1: the system-level ids of this realization group (`plan().system_ids()`).
    pub fn system_ids(&self) -> &SystemIdMap {
        self.data.plan.system_ids()
    }

    fn block_coordinates(
        &self,
        row: SysResId,
        column: SysVarId,
    ) -> Result<(usize, &FieldBlock, &FieldBlock), FinitumError> {
        let block_index = self.data.plan.row_index(row).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("system has no residual {row}"))
        })?;
        let (record, block) = self.data.plan.row(block_index);
        let layout = self.layout();
        let row_variable = self.instance_fields(record.instance).variable(block.row)?;
        let row_block = layout.block_by_variable(row_variable).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "residual {row} row field {} has no layout block",
                block.row
            ))
        })?;
        let column_block = layout.block_by_variable(column).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("layout has no block for {column}"))
        })?;
        Ok((block_index, row_block, column_block))
    }

    /// SC-W1 public per-`(row, column)` block action (`sinbad/ARCHITECTURE.md` §8): the
    /// `(row, column)` block of the Jacobian `dR/du + rate_shift * dR/du_t` at `(t, u, u_t)`
    /// applied to a column-block direction, i.e. the row equation's JVP with the direction
    /// nonzero on `column`'s block only -- so a cross-field chain-rule tangent (`ka = ka(b)`
    /// inside the `a` equation) is exactly the `(ea, b)` block, no expression rewriting.
    /// `direction` has the column block's extent, `output` the row block's extent (its
    /// `equation_sign` included). Admitted exterior-facet integrals carry no active input and
    /// contribute nothing to any block.
    #[allow(clippy::too_many_arguments)]
    pub fn block_jacobian_vector_product(
        &self,
        row: SysResId,
        column: SysVarId,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        direction: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let (block_index, row_block, column_block) = self.block_coordinates(row, column)?;
        let dimension = self.dimension();
        self.validate_point(time, state, state_rate, &vec![0.0; dimension])?;
        validate_rate_shift(rate_shift)?;
        if direction.len() != column_block.extent || output.len() != row_block.extent {
            return Err(FinitumError::InvalidRealization(format!(
                "block ({row}, {column}) JVP expects a direction of length {} and an output of \
                 length {}, got {} and {}",
                column_block.extent,
                row_block.extent,
                direction.len(),
                output.len()
            )));
        }
        validate_finite("block JVP direction", direction)?;
        let mut state_direction = vec![0.0; dimension];
        state_direction[column_block.offset..column_block.offset + column_block.extent]
            .copy_from_slice(direction);
        let rate_direction = state_direction
            .iter()
            .map(|value| rate_shift * value)
            .collect::<Vec<_>>();
        let mut full = vec![0.0; dimension];
        self.apply_cells_of(
            time,
            state,
            state_rate,
            SystemAction::Jvp {
                state_direction: &state_direction,
                rate_direction: &rate_direction,
            },
            &mut full,
            Some(block_index),
        )?;
        output.copy_from_slice(&full[row_block.offset..row_block.offset + row_block.extent]);
        validate_finite("block JVP", output)
    }

    /// SC-W1 public per-`(row, column)` block transpose: the exact transpose of
    /// [`Self::block_jacobian_vector_product`] (row adjoint in, column cotangent out),
    /// executing the row equation's bound VJP kernels with every other row's adjoint at zero.
    #[allow(clippy::too_many_arguments)]
    pub fn block_vector_jacobian_product(
        &self,
        row: SysResId,
        column: SysVarId,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let (block_index, row_block, column_block) = self.block_coordinates(row, column)?;
        let dimension = self.dimension();
        self.validate_point(time, state, state_rate, &vec![0.0; dimension])?;
        validate_rate_shift(rate_shift)?;
        if adjoint.len() != row_block.extent || output.len() != column_block.extent {
            return Err(FinitumError::InvalidRealization(format!(
                "block ({row}, {column}) VJP expects an adjoint of length {} and an output of \
                 length {}, got {} and {}",
                row_block.extent,
                column_block.extent,
                adjoint.len(),
                output.len()
            )));
        }
        validate_finite("block VJP adjoint", adjoint)?;
        let mut full_adjoint = vec![0.0; dimension];
        full_adjoint[row_block.offset..row_block.offset + row_block.extent]
            .copy_from_slice(adjoint);
        let mut full = vec![0.0; dimension];
        self.apply_cells_of(
            time,
            state,
            state_rate,
            SystemAction::Vjp {
                adjoint: &full_adjoint,
                rate_shift,
            },
            &mut full,
            Some(block_index),
        )?;
        output
            .copy_from_slice(&full[column_block.offset..column_block.offset + column_block.extent]);
        validate_finite("block VJP", output)
    }

    /// The zero-point (`t = 0`, zero state and rate, no rate shift) `(row, column)` block
    /// action -- the block view of [`Self::apply_action`].
    pub fn block_action(
        &self,
        row: SysResId,
        column: SysVarId,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let zero = vec![0.0; self.dimension()];
        self.block_jacobian_vector_product(row, column, 0.0, &zero, &zero, input, 0.0, output)
    }

    /// The zero-point `(row, column)` block transpose action.
    pub fn block_transpose_action(
        &self,
        row: SysResId,
        column: SysVarId,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let zero = vec![0.0; self.dimension()];
        self.block_vector_jacobian_product(row, column, 0.0, &zero, &zero, input, 0.0, output)
    }

    /// One `(row, column)` block of the Jacobian at a linearization point as a rectangular
    /// Methodus `LinearOperator + TransposableOperator` (rows = the row block's extent,
    /// columns = the column block's extent), for Krasis/Methodus block compositions.
    pub fn block_operator(
        &self,
        row: SysResId,
        column: SysVarId,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        rate_shift: f64,
    ) -> Result<SystemBlockOperator, FinitumError> {
        let (_, row_block, column_block) = self.block_coordinates(row, column)?;
        self.validate_point(time, state, state_rate, &vec![0.0; self.dimension()])?;
        validate_rate_shift(rate_shift)?;
        Ok(SystemBlockOperator {
            operator: self.clone(),
            row,
            column,
            rows: row_block.extent,
            columns: column_block.extent,
            time,
            state: state.to_vec(),
            state_rate: state_rate.to_vec(),
            rate_shift,
        })
    }

    /// Essential-constraint-eliminated action, mirroring `MixedOperator::apply_reduced_action`
    /// (reusing the same shared `crate::constraint::apply_constrained_action` transform).
    pub fn apply_reduced_action(
        &self,
        constraints: &ConstraintSet,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.dimension();
        if constraints.dof_count() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "essential constraint set has {} degrees of freedom, system operator has \
                 {dimension}",
                constraints.dof_count()
            )));
        }
        if output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "system operator reduced action expects output length {dimension}, got {}",
                output.len()
            )));
        }
        crate::constraint::apply_constrained_action(constraints, input, output, |input, output| {
            self.apply_action(input, output)
        })
    }

    /// Binds `constraints` into a [`ReducedSystemOperator`] (mirroring
    /// `MixedOperator::reduced`). Refuses a constraint set whose `dof_count()` does not match
    /// [`Self::dimension`].
    pub fn reduced(
        &self,
        constraints: ConstraintSet,
    ) -> Result<ReducedSystemOperator, FinitumError> {
        if constraints.dof_count() != self.dimension() {
            return Err(FinitumError::InvalidRealization(format!(
                "essential constraint set has {} degrees of freedom, system operator has {}",
                constraints.dof_count(),
                self.dimension()
            )));
        }
        Ok(ReducedSystemOperator {
            operator: self.clone(),
            constraints,
            motion: None,
        })
    }

    /// Canonical CSR assembly by unit-column probing of [`Self::apply_action`], mirroring
    /// `RealizationPlan::assemble`/`MixedOperator::assemble`.
    pub fn assemble(&self) -> Result<methodus::CsrMatrix, FinitumError> {
        let dimension = self.dimension();
        let mut entries = Vec::new();
        let mut direction = vec![0.0; dimension];
        let mut output = vec![0.0; dimension];
        for column in 0..dimension {
            direction[column] = 1.0;
            self.apply_action(&direction, &mut output)?;
            for (row, value) in output.iter().copied().enumerate() {
                if value != 0.0 {
                    entries.push((row, column, value));
                }
            }
            direction[column] = 0.0;
        }
        methodus::CsrMatrix::from_triplets(dimension, dimension, entries)
            .map_err(|error| FinitumError::Assembly(error.to_string()))
    }

    /// The L2 mass (Gram) matrix `integral(phi_i phi_j)` of one realized Lagrange field
    /// (P0, P1, or P2; a vector field is block-diagonal across its components), dense row-major
    /// over the field's own [`BlockLayout`] block extent, integrated with the operator's shared
    /// quadrature (the plan's [`SystemQuadrature`]: on `Richest`, degree-4 exact on triangles,
    /// degree-2 on tetrahedra -- the 3-D P2 mass is under-integrated, the same limit `STATUS.md`
    /// records for the cell quadrature; on `Barycenter` the P1 mass is rank one per cell,
    /// C11.8). This is
    /// the multiplier norm [`crate::estimate_inf_sup`] takes as [`crate::InfSupNorm::Gram`];
    /// an RT0 field is refused typed (its L2 mass needs the Piola-mapped basis, which no
    /// multiplier field of this crate's admitted pairings uses).
    pub fn mass_matrix(&self, field: SymbolId) -> Result<Vec<f64>, FinitumError> {
        let block = self.layout().block(field).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("field {field} was not realized"))
        })?;
        let element_field = self.data.fields.get(&block.variable).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("field {field} was not realized"))
        })?;
        let element = match &element_field.kind {
            FieldKind::Lagrange(element) => element,
            FieldKind::Hdiv0 { .. } => {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "mass matrix of the Hdiv(order=0) field {field} is not realized"
                )));
            }
        };
        let extent = block.extent;
        let components = block.component_count;
        let basis_count = element.basis_count();
        let mut matrix = vec![0.0; extent * extent];
        let mesh = self.data.plan.mesh();
        for (cell, restriction) in element_field.dofs.restrictions().iter().enumerate() {
            if restriction.dofs.len() != basis_count * components {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "field {field} cell {cell} restriction has {} DOFs, expected {}",
                    restriction.dofs.len(),
                    basis_count * components
                )));
            }
            let geometry = CellGeometry::new(mesh, CellId(cell))?;
            for (point, quadrature_point) in self.data.quadrature.iter().enumerate() {
                let scale = quadrature_point.weight * geometry.determinant();
                for i in 0..basis_count {
                    let phi_i = element
                        .basis_value(point, i)
                        .expect("validated element table");
                    for j in 0..basis_count {
                        let phi_j = element
                            .basis_value(point, j)
                            .expect("validated element table");
                        let weight = scale * phi_i * phi_j;
                        for component in 0..components {
                            let row = restriction.dofs[i * components + component].0;
                            let column = restriction.dofs[j * components + component].0;
                            matrix[row * extent + column] += weight;
                        }
                    }
                }
            }
        }
        validate_finite("field mass matrix", &matrix)?;
        Ok(matrix)
    }

    /// Establishes, once, whether this operator's zero-point action is self-adjoint by
    /// assembling it, and records the answer for every later [`Self::symmetry`] query on this
    /// operator, its clones, and every [`ReducedSystemOperator`] / [`LinearizedSystemOperator`]
    /// / capability derived from it -- mirroring `RealizationPlan::prove_symmetry` exactly,
    /// including its dimension cap. A taken proof outranks Scientia's structural claim in both
    /// directions: a passed proof upgrades a structural `Unknown`/`Nonsymmetric` claim to
    /// `Symmetric` (so a Methodus conjugate-gradient or MINRES solve admits the operator with no
    /// caller-side assumption), a failed proof reports `Nonsymmetric` and never upgrades
    /// anything. A refused proof (non-finite tolerance, dimension cap) records nothing and
    /// leaves the claim as it was.
    pub fn prove_symmetry(&self, tolerance: f64) -> Result<OperatorSymmetry, FinitumError> {
        if let Some(proof) = self.data.symmetry_proof.get() {
            return Ok(*proof);
        }
        if !(tolerance.is_finite() && tolerance >= 0.0) {
            return Err(FinitumError::UnsupportedRealization(format!(
                "symmetry proof tolerance must be finite and nonnegative, got {tolerance}"
            )));
        }
        if self.dimension() > crate::realization::SYMMETRY_PROOF_DIMENSION_CAP {
            return Err(FinitumError::UnsupportedRealization(format!(
                "symmetry proof by assembly is refused above \
                 {} degrees of freedom (operator has {})",
                crate::realization::SYMMETRY_PROOF_DIMENSION_CAP,
                self.dimension()
            )));
        }
        let assembled = self.assemble()?;
        let proof = if crate::realization::csr_is_symmetric_within(&assembled, tolerance) {
            OperatorSymmetry::Symmetric
        } else {
            OperatorSymmetry::Nonsymmetric
        };
        Ok(*self.data.symmetry_proof.get_or_init(|| proof))
    }

    /// The per-cell gather of every realized field's local DOFs of `vector`, by variable.
    fn gather_cell(&self, cell: usize, vector: &[f64]) -> BTreeMap<SysVarId, Vec<f64>> {
        let layout = self.layout();
        self.data
            .fields
            .iter()
            .map(|(&variable, field)| {
                let field_block = layout
                    .block_by_variable(variable)
                    .expect("realized field implies a layout block");
                let restriction = &field.dofs.restrictions()[cell];
                (
                    variable,
                    restriction
                        .dofs
                        .iter()
                        .map(|dof| vector[field_block.offset + dof.0])
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    /// One instance's view of the realized fields.
    fn instance_fields(&self, instance: InstanceId) -> InstanceFields<'_> {
        InstanceFields {
            fields: &self.data.fields,
            variables: &self.data.instance_variables[instance.0 as usize],
        }
    }

    /// The input bindings of one residual row at a point, over the row instance's bound
    /// inputs.
    fn row_bindings<'a>(
        &'a self,
        row_index: usize,
        instance: InstanceId,
        bound: &'a [BoundPointValue],
    ) -> SystemInputBindings<'a> {
        SystemInputBindings {
            scope: InputScope::Row {
                row: row_index,
                constitutive: &self.data.constitutive,
                stored: &self.data.stored,
            },
            bound: BoundTable {
                binds: &self.data.binds,
                values: bound,
                instance,
            },
            point_count: self.data.quadrature.len(),
        }
    }

    /// W8 F-MI: every same-mesh bind's producer output at one quadrature point, in the plan's
    /// dependency order (each output kernel sees the bound inputs of its own instance that
    /// precede it), with its directional derivative along `directions` when given. Empty on
    /// a one-instance plan.
    #[allow(clippy::too_many_arguments)]
    fn bound_point_values(
        &self,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        affine: &AffineMap,
        reference_point: &[f64],
        time: f64,
        local_state: &BTreeMap<SysVarId, Vec<f64>>,
        local_rate: &BTreeMap<SysVarId, Vec<f64>>,
        directions: LocalDirections<'_>,
    ) -> Result<Vec<BoundPointValue>, FinitumError> {
        let mut values: Vec<BoundPointValue> = Vec::with_capacity(self.data.binds.len());
        for (index, bound) in self.data.binds.iter().enumerate() {
            let producer = bound.bind.producer;
            let fields = self.instance_fields(producer);
            let bindings = SystemInputBindings {
                scope: InputScope::Output {
                    constitutive: &bound.constitutive,
                },
                bound: BoundTable {
                    binds: &self.data.binds,
                    values: &values,
                    instance: producer,
                },
                point_count: self.data.quadrature.len(),
            };
            let (inputs, evaluation) = point_inputs_system(
                fields,
                &bound.integral,
                cell,
                point,
                geometry,
                affine,
                reference_point,
                time,
                local_state,
                local_rate,
                &bindings,
            )?;
            let output = execute_primal_values(&bound.bound, &inputs)?;
            validate_finite("bound output", &output)?;
            let direction = match directions {
                None => None,
                Some((state_direction, rate_direction)) => {
                    let directions = point_directions_system(
                        fields,
                        &bound.integral,
                        cell,
                        point,
                        geometry,
                        affine,
                        reference_point,
                        time,
                        state_direction,
                        rate_direction,
                        &bindings,
                        &evaluation,
                    )?;
                    let tangent = execute_jvp_values(&bound.bound, &inputs, &directions)?;
                    validate_finite("bound output direction", &tangent)?;
                    Some(tangent)
                }
            };
            values.push(BoundPointValue {
                bind: index,
                values: output,
                direction,
                producer_inputs: inputs,
                producer_evaluation: evaluation,
            });
        }
        Ok(values)
    }

    /// W8 F-MI: the transpose of the bind chain at one quadrature point. Walks the binds in
    /// reverse dependency order: a bind's accumulated seed (the cotangent of its output) is
    /// pushed through the producer output kernel's VJP into the producer's fields (scattered
    /// through their bases, scaled like any column field's cotangent) and, through its
    /// parameter cotangents, into the producer's closures (probed with unit active and bound
    /// perturbations) and into the bound inputs the producer reads -- whose binds precede it
    /// and are processed next.
    #[allow(clippy::too_many_arguments)]
    fn push_bound_cotangents(
        &self,
        values: &[BoundPointValue],
        seeds: &mut BoundSeeds,
        cell: usize,
        point: usize,
        geometry: &CellGeometry,
        affine: &AffineMap,
        reference_point: &[f64],
        scale: f64,
        rate_shift: f64,
        local_outputs: &mut BTreeMap<SysVarId, Vec<f64>>,
    ) -> Result<(), FinitumError> {
        for index in (0..self.data.binds.len()).rev() {
            let Some(seed) = seeds[index].take() else {
                continue;
            };
            let bound = &self.data.binds[index];
            let value = &values[index];
            let producer = bound.bind.producer;
            let fields = self.instance_fields(producer);
            let bindings = SystemInputBindings {
                scope: InputScope::Output {
                    constitutive: &bound.constitutive,
                },
                bound: BoundTable {
                    binds: &self.data.binds,
                    values: &values[..index],
                    instance: producer,
                },
                point_count: self.data.quadrature.len(),
            };
            let active_inputs = active_probe_inputs(&bound.integral, rate_shift);
            let mut cotangents =
                execute_vjp_values(&bound.bound, &value.producer_inputs, seed.clone())?;
            if !bound.bound.bundle.parameter.independent_operands.is_empty() {
                let parameter_cotangents =
                    point_parameter_cotangents(&bound.bound, &value.producer_inputs, &seed)?;
                accumulate_parameter_cotangents_system(
                    fields,
                    &bound.integral,
                    cell,
                    point,
                    geometry,
                    affine,
                    reference_point,
                    scale,
                    rate_shift,
                    &value.producer_evaluation,
                    &active_inputs,
                    &bindings,
                    &parameter_cotangents,
                    &mut cotangents,
                    local_outputs,
                    seeds,
                )?;
            }
            for input in &bound.integral.primal.inputs {
                if input.source != InputSourceRequirement::Basis
                    || input.role != TensorInputRole::Active
                {
                    continue;
                }
                let Some((derivative, factor)) = transpose_scatter_shape(input, rate_shift) else {
                    continue;
                };
                let Some(cotangent) = cotangents.get(&input.id) else {
                    continue;
                };
                scatter_field_cotangent(
                    fields,
                    input.binding.symbol,
                    geometry,
                    affine,
                    cell,
                    point,
                    reference_point,
                    &derivative,
                    cotangent,
                    factor * scale,
                    local_outputs,
                )?;
            }
        }
        Ok(())
    }

    /// One row's contribution on one cell for one [`SystemAction`]: gathers every field's
    /// local state/rate (and directions) through its own DOF restriction, drives each cell
    /// integral output's bound kernel at every shared quadrature point, and scatters through
    /// the row field's basis (PRIMAL/JVP) or through every active column field's basis (VJP).
    ///
    /// On a composed plan (W8 F-MI) every bind's producer output is evaluated at the point
    /// first ([`Self::bound_point_values`]). PRIMAL: a consumer bundle on the kernel-input path
    /// runs its Malleus `BindComposition` (`execute_bind_composition`); every other bundle
    /// reads its bound inputs' values as external operands. JVP: the tangent is the §6
    /// chain-rule product of local point kernels -- the producer output bundle's full JVP
    /// (its active basis directions *and* its frozen-input tangents, which is how `sigma(T)`
    /// inside `joule_heat` reaches `dR_thermal/dT`) fed as the bound operand's direction into
    /// the consumer's parameter JVP, and as [`PointEvaluation::bound`] directions into the
    /// consumer's closures. The recorded `jvp_compositions` are digest-checked at bind time
    /// but not executed: their independent set is the producer's active basis operands only,
    /// so executing them alone would drop the producer's provider-input tangent. VJP: the
    /// exact transpose, [`Self::push_bound_cotangents`].
    #[allow(clippy::too_many_arguments)]
    fn apply_block_cell(
        &self,
        row_index: usize,
        row: &RealizedRow,
        block: &OperatorSystemBlock,
        cell: usize,
        geometry: &CellGeometry,
        affine: &AffineMap,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        action: SystemAction<'_>,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let layout = self.layout();
        let fields = self.instance_fields(row.instance);
        let (row_variable, row_field) = fields.field(block.row).map_err(|_| {
            FinitumError::ArtifactMismatch(format!(
                "equation `{}` row field {} was not realized",
                row.path, block.row
            ))
        })?;
        let row_block = layout
            .block_by_variable(row_variable)
            .expect("row field implies a layout block");
        let row_restriction = &row_field.dofs.restrictions()[cell];
        let sign = self
            .data
            .equation_sign
            .get(&row_index)
            .copied()
            .unwrap_or(1.0);

        let local_state = self.gather_cell(cell, state);
        let local_rate = self.gather_cell(cell, state_rate);
        let bindings = &self.data.bindings[&row_index];
        let quadrature = &self.data.quadrature;

        match action {
            SystemAction::Primal | SystemAction::Jvp { .. } => {
                let directions = match action {
                    SystemAction::Jvp {
                        state_direction,
                        rate_direction,
                    } => Some((
                        self.gather_cell(cell, state_direction),
                        self.gather_cell(cell, rate_direction),
                    )),
                    _ => None,
                };
                let mut local_output = vec![0.0; row_restriction.dofs.len()];
                for integral in &block.factorization.integrals {
                    if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                        // Exterior-facet integrals are processed by `Self::apply_facets`.
                        continue;
                    }
                    for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                        let bound = bound_kernel(bindings, &row.path, integral, output_index)?;
                        let composition = self.data.binds.iter().find_map(|bind| {
                            bind.compositions
                                .iter()
                                .find(|composition| {
                                    composition.row == row_index
                                        && composition.integral_index == integral.integral_index
                                        && composition.output_index == output_index
                                })
                                .map(|composition| (bind, composition))
                        });
                        for (point, quadrature_point) in quadrature.iter().enumerate() {
                            let reference_point = &quadrature_point.coordinates;
                            let scale = quadrature_point.weight * geometry.determinant();
                            let bound_values = self.bound_point_values(
                                cell,
                                point,
                                geometry,
                                affine,
                                reference_point,
                                time,
                                &local_state,
                                &local_rate,
                                directions.as_ref().map(|(state, rate)| (state, rate)),
                            )?;
                            let point_bindings =
                                self.row_bindings(row_index, row.instance, &bound_values);
                            let (inputs, evaluation) = point_inputs_system(
                                fields,
                                integral,
                                cell,
                                point,
                                geometry,
                                affine,
                                reference_point,
                                time,
                                &local_state,
                                &local_rate,
                                &point_bindings,
                            )?;
                            let point_output = match &directions {
                                None => match composition {
                                    Some((bind, composition)) => {
                                        let producer = bound_values
                                            .iter()
                                            .find(|value| {
                                                std::ptr::eq(&self.data.binds[value.bind], bind)
                                            })
                                            .expect("every bind has a point value");
                                        execute_bind_composition(
                                            composition,
                                            &bind.bound,
                                            &producer.producer_inputs,
                                            bound,
                                            &inputs,
                                        )?
                                    }
                                    None => execute_primal_values(bound, &inputs)?,
                                },
                                Some((local_state_direction, local_rate_direction)) => {
                                    let directions = point_directions_system(
                                        fields,
                                        integral,
                                        cell,
                                        point,
                                        geometry,
                                        affine,
                                        reference_point,
                                        time,
                                        local_state_direction,
                                        local_rate_direction,
                                        &point_bindings,
                                        &evaluation,
                                    )?;
                                    execute_jvp_values(bound, &inputs, &directions)?
                                }
                            };
                            apply_field_basis_adjoint(
                                row_field,
                                geometry,
                                affine,
                                cell,
                                point,
                                reference_point,
                                &qoutput.binding.evaluation.derivative,
                                &point_output,
                                scale,
                                &mut local_output,
                            )?;
                        }
                    }
                }
                for (local_index, dof) in row_restriction.dofs.iter().enumerate() {
                    output[row_block.offset + dof.0] += sign * local_output[local_index];
                }
            }
            SystemAction::Vjp {
                adjoint,
                rate_shift,
            } => {
                // `(sign * A)^T = sign * A^T`: the row sign scales the adjoint seed.
                let local_adjoint = row_restriction
                    .dofs
                    .iter()
                    .map(|dof| sign * adjoint[row_block.offset + dof.0])
                    .collect::<Vec<_>>();
                let mut local_outputs = self
                    .data
                    .fields
                    .iter()
                    .map(|(&variable, field)| {
                        (
                            variable,
                            vec![0.0; field.dofs.restrictions()[cell].dofs.len()],
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                for integral in &block.factorization.integrals {
                    if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                        continue;
                    }
                    let active_inputs = active_probe_inputs(integral, rate_shift);
                    for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                        let bound = bound_kernel(bindings, &row.path, integral, output_index)?;
                        let output_components = component_count(&qoutput.shape)?;
                        for (point, quadrature_point) in quadrature.iter().enumerate() {
                            let reference_point = &quadrature_point.coordinates;
                            let scale = quadrature_point.weight * geometry.determinant();
                            let bound_values = self.bound_point_values(
                                cell,
                                point,
                                geometry,
                                affine,
                                reference_point,
                                time,
                                &local_state,
                                &local_rate,
                                None,
                            )?;
                            let point_bindings =
                                self.row_bindings(row_index, row.instance, &bound_values);
                            let (inputs, evaluation) = point_inputs_system(
                                fields,
                                integral,
                                cell,
                                point,
                                geometry,
                                affine,
                                reference_point,
                                time,
                                &local_state,
                                &local_rate,
                                &point_bindings,
                            )?;
                            let seed = gather_field_test_adjoint(
                                row_field,
                                geometry,
                                affine,
                                cell,
                                point,
                                reference_point,
                                &qoutput.binding.evaluation.derivative,
                                output_components,
                                &local_adjoint,
                            )?;
                            let mut cotangents = execute_vjp_values(bound, &inputs, seed.clone())?;
                            let mut bound_seeds: BoundSeeds = vec![None; self.data.binds.len()];
                            if !bound.bundle.parameter.independent_operands.is_empty() {
                                let parameter_cotangents =
                                    point_parameter_cotangents(bound, &inputs, &seed)?;
                                accumulate_parameter_cotangents_system(
                                    fields,
                                    integral,
                                    cell,
                                    point,
                                    geometry,
                                    affine,
                                    reference_point,
                                    scale,
                                    rate_shift,
                                    &evaluation,
                                    &active_inputs,
                                    &point_bindings,
                                    &parameter_cotangents,
                                    &mut cotangents,
                                    &mut local_outputs,
                                    &mut bound_seeds,
                                )?;
                            }
                            for input in &integral.primal.inputs {
                                if input.source != InputSourceRequirement::Basis
                                    || input.role != TensorInputRole::Active
                                {
                                    continue;
                                }
                                let Some((derivative, factor)) =
                                    transpose_scatter_shape(input, rate_shift)
                                else {
                                    continue;
                                };
                                let Some(cotangent) = cotangents.get(&input.id) else {
                                    continue;
                                };
                                scatter_field_cotangent(
                                    fields,
                                    input.binding.symbol,
                                    geometry,
                                    affine,
                                    cell,
                                    point,
                                    reference_point,
                                    &derivative,
                                    cotangent,
                                    factor * scale,
                                    &mut local_outputs,
                                )?;
                            }
                            if bound_seeds.iter().any(Option::is_some) {
                                self.push_bound_cotangents(
                                    &bound_values,
                                    &mut bound_seeds,
                                    cell,
                                    point,
                                    geometry,
                                    affine,
                                    reference_point,
                                    scale,
                                    rate_shift,
                                    &mut local_outputs,
                                )?;
                            }
                        }
                    }
                }
                for (variable, local_output) in &local_outputs {
                    let field_block = layout
                        .block_by_variable(*variable)
                        .expect("realized field implies a layout block");
                    let restriction = &self.data.fields[variable].dofs.restrictions()[cell];
                    for (local_index, dof) in restriction.dofs.iter().enumerate() {
                        output[field_block.offset + dof.0] += local_output[local_index];
                    }
                }
            }
        }
        Ok(())
    }

    /// Mission item 2's exterior-facet extension: iterates every block's `SemanticMeasure::
    /// ExteriorFacet` integral over its resolved facet list, scattering each into the row
    /// field's global DOFs. Every admitted facet integral has no primal input at all (see
    /// `SystemRealizationPlan::bind_kernels_with_facets`'s doc comment), so `action` selects only
    /// which of `execute_primal_values`/`execute_jvp_values` runs the (input-free) bound kernel.
    fn apply_facets(&self, output: &mut [f64], action: FacetAction) -> Result<(), FinitumError> {
        for (row_index, row, block) in self.data.plan.rows() {
            for integral in &block.factorization.integrals {
                let region = match &integral.measure {
                    SemanticMeasure::ExteriorFacet { region } => *region,
                    _ => continue,
                };
                let facet_ids = self
                    .data
                    .facet_regions
                    .get(&region)
                    .expect("validated non-empty at bind_kernels_with_facets");
                for &facet_id in facet_ids {
                    self.apply_block_facet(
                        row_index, row, block, integral, facet_id, output, action,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_block_facet(
        &self,
        row_index: usize,
        row: &RealizedRow,
        block: &OperatorSystemBlock,
        integral: &IntegralOperatorFactorization,
        facet_id: FacetId,
        output: &mut [f64],
        action: FacetAction,
    ) -> Result<(), FinitumError> {
        let layout = self.layout();
        let block_index = row_index;
        let (row_variable, row_field) = self
            .instance_fields(row.instance)
            .field(block.row)
            .map_err(|_| {
                FinitumError::ArtifactMismatch(format!(
                    "equation `{}` row field {} was not realized",
                    row.path, block.row
                ))
            })?;
        let orientations = match &row_field.kind {
            FieldKind::Hdiv0 { orientations } => orientations,
            FieldKind::Lagrange(_) => {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "equation `{}` has an exterior-facet integral but row field {} is not \
                     Hdiv(order=0) (should have been refused at bind_kernels_with_facets time)",
                    row.path, block.row
                )));
            }
        };
        let geometry = self
            .data
            .facet_geometries
            .get(&facet_id)
            .expect("facet geometry was precomputed at bind_kernels_with_facets time");
        let cell = geometry.cell.0;
        let cell_orientations = orientations.get(cell).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "RT0 field has no orientation row for cell {cell}"
            ))
        })?;
        let row_block = layout
            .block_by_variable(row_variable)
            .expect("row field implies a layout block");
        let restriction = &row_field.dofs.restrictions()[cell];
        let mut local_output = vec![0.0; restriction.dofs.len()];
        let bindings = &self.data.bindings[&block_index];
        let inputs = BTreeMap::new();
        for (output_index, _qoutput) in integral.primal.outputs.iter().enumerate() {
            let bound = bound_kernel(bindings, &row.path, integral, output_index)?;
            let point_output = match action {
                FacetAction::Primal => execute_primal_values(bound, &inputs)?,
                FacetAction::Jvp => {
                    let directions = BTreeMap::new();
                    execute_jvp_values(bound, &inputs, &directions)?
                }
            };
            apply_rt0_normal_trace_adjoint(
                cell_orientations,
                geometry.local_facet,
                &point_output,
                &mut local_output,
            )?;
        }
        let sign = self
            .data
            .equation_sign
            .get(&block_index)
            .copied()
            .unwrap_or(1.0);
        for (local_index, dof) in restriction.dofs.iter().enumerate() {
            output[row_block.offset + dof.0] += sign * local_output[local_index];
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum FacetAction {
    Primal,
    Jvp,
}

/// Which cell-integral action [`SystemOperator::apply_block_cell`] executes at a linearization
/// point.
#[derive(Clone, Copy)]
enum SystemAction<'a> {
    /// The bound PRIMAL kernels: the residual.
    Primal,
    /// The bound JVP (and parameter-JVP) kernels along a state/rate direction pair.
    Jvp {
        state_direction: &'a [f64],
        rate_direction: &'a [f64],
    },
    /// The bound VJP kernels seeded by the row field's adjoint (SV1-C1), transposing the
    /// rate-shifted Jacobian `dR/du + rate_shift * dR/du_t`.
    Vjp { adjoint: &'a [f64], rate_shift: f64 },
}

fn validate_rate_shift(rate_shift: f64) -> Result<(), FinitumError> {
    if rate_shift.is_finite() {
        Ok(())
    } else {
        Err(FinitumError::InvalidRealization(
            "rate shift must be finite".into(),
        ))
    }
}

fn numeric_error(error: FinitumError) -> NumericError {
    NumericError::from(error)
}

fn bound_kernel<'a>(
    bindings: &'a BTreeMap<(usize, usize), BoundBundle>,
    equation: &str,
    integral: &IntegralOperatorFactorization,
    output_index: usize,
) -> Result<&'a BoundBundle, FinitumError> {
    bindings
        .get(&(integral.integral_index, output_index))
        .ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "equation `{equation}` integral {} output {output_index} has no bound kernel",
                integral.integral_index
            ))
        })
}

impl LinearOperator for SystemOperator {
    fn rows(&self) -> usize {
        self.dimension()
    }

    fn columns(&self) -> usize {
        self.dimension()
    }

    /// The proof recorded by a taken [`Self::prove_symmetry`] call when there is one (proof
    /// outranks declaration, in both directions); otherwise Scientia's structural
    /// `form_symmetry` (C5.4/C5.5, item 8) when no `equation_sign` orientation correction was
    /// applied at [`SystemRealizationPlan::bind_kernels`] time (`structure.form_symmetry`
    /// describes exactly this, unsigned, system), or `Unknown` for a resigned system -- a
    /// resigned system's symmetry is not implied by the unsigned system's structural claim, so
    /// it is never reused silently.
    fn symmetry(&self) -> OperatorSymmetry {
        if let Some(proof) = self.data.symmetry_proof.get() {
            return *proof;
        }
        if self.data.equation_sign.values().any(|&sign| sign != 1.0) {
            return OperatorSymmetry::Unknown;
        }
        // A same-mesh bind adds cross blocks no per-model structure describes; several
        // instances without binds are block-diagonal, symmetric iff every instance is.
        if !self.data.binds.is_empty() {
            return OperatorSymmetry::Unknown;
        }
        let claims = self
            .data
            .structures
            .iter()
            .map(|structure| map_form_symmetry(structure.form_symmetry))
            .collect::<Vec<_>>();
        if claims
            .iter()
            .all(|claim| *claim == OperatorSymmetry::Symmetric)
        {
            OperatorSymmetry::Symmetric
        } else if claims.contains(&OperatorSymmetry::Nonsymmetric) {
            OperatorSymmetry::Nonsymmetric
        } else {
            OperatorSymmetry::Unknown
        }
    }

    fn properties(&self) -> OperatorProperties {
        OperatorProperties::new(
            self.symmetry(),
            Definiteness::Unknown,
            None,
            OperatorStructureHint::Block {
                layout: self.data.solver_layout.clone(),
                saddle_point: self.data.structure.saddle_point,
            },
        )
        .expect("system operator properties are internally consistent")
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.apply_action(input, output).map_err(NumericError::from)
    }
}

impl BlockLinearOperator for SystemOperator {
    fn block_layout(&self) -> &methodus::BlockLayout {
        &self.data.solver_layout
    }
}

/// One `(row, column)` Jacobian block of a [`SystemOperator`] at a fixed linearization point
/// (see [`SystemOperator::block_operator`]): a rectangular Methodus operator whose action is
/// [`SystemOperator::block_jacobian_vector_product`] and whose transpose is
/// [`SystemOperator::block_vector_jacobian_product`].
#[derive(Clone, Debug)]
pub struct SystemBlockOperator {
    operator: SystemOperator,
    row: SysResId,
    column: SysVarId,
    rows: usize,
    columns: usize,
    time: f64,
    state: Vec<f64>,
    state_rate: Vec<f64>,
    rate_shift: f64,
}

impl SystemBlockOperator {
    pub fn row(&self) -> SysResId {
        self.row
    }

    pub fn column(&self) -> SysVarId {
        self.column
    }

    pub fn rate_shift(&self) -> f64 {
        self.rate_shift
    }
}

impl LinearOperator for SystemBlockOperator {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn symmetry(&self) -> OperatorSymmetry {
        OperatorSymmetry::Unknown
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.operator
            .block_jacobian_vector_product(
                self.row,
                self.column,
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

impl TransposableOperator for SystemBlockOperator {
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.operator
            .block_vector_jacobian_product(
                self.row,
                self.column,
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

/// One prescribed essential target's value and analytic time derivative. Both are required;
/// an unavailable rate must be a typed callback failure, never an implicit zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrescribedValueAndRate {
    pub value: f64,
    pub rate: f64,
}

type PrescribedEvaluator =
    dyn Fn(f64) -> Result<PrescribedValueAndRate, InputEvaluationError> + Send + Sync;

/// A time-dependent prescribed value on an existing fixed essential target. The coordinates
/// locate callback failures; the caller's motion identity authenticates the scientific data.
#[derive(Clone)]
pub struct PrescribedEssentialValue {
    target: DofId,
    coordinates: Vec<f64>,
    origin: InputOrigin,
    evaluator: Arc<PrescribedEvaluator>,
}

impl std::fmt::Debug for PrescribedEssentialValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrescribedEssentialValue")
            .field("target", &self.target)
            .field("coordinates", &self.coordinates)
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

impl PrescribedEssentialValue {
    /// Binds a pure evaluator of prescribed value and analytic rate at the requested time.
    /// Finitum validates the target and coordinates when attaching it to a reduced operator.
    pub fn new(
        target: DofId,
        coordinates: Vec<f64>,
        origin: InputOrigin,
        evaluator: impl Fn(f64) -> Result<PrescribedValueAndRate, InputEvaluationError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            target,
            coordinates,
            origin,
            evaluator: Arc::new(evaluator),
        }
    }
}

#[derive(Clone, Debug)]
struct PrescribedMotion {
    identity: String,
    values: Vec<PrescribedEssentialValue>,
}

/// A [`SystemOperator`] with essential (Dirichlet) constraints eliminated through
/// [`SystemOperator::apply_reduced_action`] -- mirroring `crate::mixed::ReducedMixedOperator`.
#[derive(Clone, Debug)]
pub struct ReducedSystemOperator {
    operator: SystemOperator,
    constraints: ConstraintSet,
    motion: Option<PrescribedMotion>,
}

impl ReducedSystemOperator {
    pub fn operator(&self) -> &SystemOperator {
        &self.operator
    }

    pub fn constraints(&self) -> &ConstraintSet {
        &self.constraints
    }

    /// Adds prescribed time-dependent offsets/rates on fixed essential targets. Targets not
    /// listed retain their static offsets and zero prescribed rates. Affine constraints and
    /// duplicate/missing targets are refused. The identity must describe both value and rate
    /// data; callbacks must be pure because solver retries may revisit times.
    pub fn with_prescribed_values(
        mut self,
        identity: impl Into<String>,
        values: Vec<PrescribedEssentialValue>,
    ) -> Result<Self, FinitumError> {
        let identity = identity.into();
        if identity.trim().is_empty() || values.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "prescribed motion requires identity and targets".into(),
            ));
        }
        if self.constraints.has_affine_dependencies() {
            return Err(FinitumError::UnsupportedRealization(
                "prescribed motion supports fixed essential targets only".into(),
            ));
        }
        let targets = self
            .constraints
            .constraints()
            .map(|c| c.target)
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        for value in &values {
            if !targets.contains(&value.target) || !seen.insert(value.target) {
                return Err(FinitumError::InvalidRealization(
                    "prescribed motion has missing or duplicate essential target".into(),
                ));
            }
            if value.coordinates.len() != self.operator.data.plan.mesh().dimension()
                || value.coordinates.iter().any(|x| !x.is_finite())
            {
                return Err(FinitumError::InvalidRealization(
                    "prescribed motion coordinates do not match mesh".into(),
                ));
            }
        }
        self.motion = Some(PrescribedMotion { identity, values });
        Ok(self)
    }

    /// Whether this operator has prescribed runtime essential data.
    pub fn has_prescribed_motion(&self) -> bool {
        self.motion.is_some()
    }

    /// The essential values at `time`, evaluated together with their required analytic rates.
    /// `constraints()` remains the topology/reference snapshot for legacy static consumers.
    pub fn constraints_at(&self, time: f64) -> Result<ConstraintSet, FinitumError> {
        Ok(self.prescribed_at(time)?.0)
    }

    fn prescribed_at(&self, time: f64) -> Result<(ConstraintSet, Vec<f64>), FinitumError> {
        if !time.is_finite() {
            return Err(FinitumError::InvalidRealization(
                "prescribed evaluation time must be finite".into(),
            ));
        }
        if self.motion.is_none() {
            return Ok((
                self.constraints.clone(),
                vec![0.0; self.operator.dimension()],
            ));
        }
        let mut constraints = self.constraints.constraints().cloned().collect::<Vec<_>>();
        let indices = constraints
            .iter()
            .enumerate()
            .map(|(i, c)| (c.target, i))
            .collect::<BTreeMap<_, _>>();
        let mut rates = vec![0.0; self.operator.dimension()];
        if let Some(motion) = &self.motion {
            for value in &motion.values {
                let evaluated = (value.evaluator)(time).map_err(|error| {
                    FinitumError::from(error.at(None, &value.coordinates, Some(time)))
                })?;
                if !evaluated.value.is_finite() || !evaluated.rate.is_finite() {
                    return Err(InputEvaluationError::new(
                        "PRESCRIBED_VALUE_NONFINITE",
                        value.origin.clone(),
                        "prescribed value or analytic rate is nonfinite",
                    )
                    .at(None, &value.coordinates, Some(time))
                    .into());
                }
                constraints[indices[&value.target]].offset = evaluated.value;
                rates[value.target.0] = evaluated.rate;
            }
        }
        Ok((
            ConstraintSet::new(self.operator.dimension(), constraints)?,
            rates,
        ))
    }

    /// Expands the state and rate into physical coordinates at actual time. Prescribed rates
    /// contribute to interior mass terms. Use this for initialization and field/observable
    /// sampling instead of independently expanding the static constraint snapshot.
    pub fn physical_state_and_rate(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), FinitumError> {
        let (constraints, lifting_rate) = self.prescribed_at(time)?;
        let physical_state = constraints.expand(state)?;
        let mut physical_rate = constraints.expand_homogeneous(state_rate)?;
        for (rate, lifting) in physical_rate.iter_mut().zip(lifting_rate) {
            *rate += lifting;
        }
        validate_finite("prescribed physical rate", &physical_rate)?;
        Ok((physical_state, physical_rate))
    }

    fn require_static_view(&self) -> Result<(), FinitumError> {
        if self.motion.is_some() {
            Err(FinitumError::UnsupportedRealization(
                "prescribed motion requires an explicit-time residual or linearization".into(),
            ))
        } else {
            Ok(())
        }
    }

    /// Motion-aware realization identity. Static operators return their underlying operator
    /// digest unchanged; consumers must use this when authenticating a reduced trajectory.
    pub fn realization_digest(&self) -> Digest {
        match &self.motion {
            None => self.operator.data.digest.clone(),
            Some(motion) => Digest::blake3(
                &serde_json::to_vec(&(
                    "finitum-prescribed-motion/1",
                    &self.operator.data.digest,
                    &self.constraints,
                    &motion.identity,
                    motion
                        .values
                        .iter()
                        .map(|v| (v.target, &v.coordinates, v.origin.to_string()))
                        .collect::<Vec<_>>(),
                ))
                .expect("motion receipt serializes"),
            ),
        }
    }

    /// The elimination-ready right-hand side for `self` (mission item 2): the exact multi-block
    /// analogue of `RealizationPlan::load_vector`'s own no-argument contract (this is why
    /// composition lives here rather than on [`SystemOperator`] -- [`SystemOperator`] does not
    /// itself own a [`ConstraintSet`]).
    ///
    /// # Composition contract
    ///
    /// `reduced RHS = load + Dirichlet lifting`, matching `RealizationPlan::load_vector`'s
    /// convention exactly:
    ///
    /// - `load = self.operator().load_vector()` -- the pure, state-independent forcing
    ///   contribution (mission item 1), full [`SystemOperator::dimension`]-length, unconstrained.
    /// - `lifting = self.constraints().expand(&vec![0.0; dimension])` -- the physical-space
    ///   vector that is zero at every free coordinate and the constrained value at every
    ///   constrained coordinate.
    /// - Because this system is globally linear, `self.operator().apply_action(&lifting, ..)`
    ///   computes exactly the bilinear form's action on the lifted state (`jacobian_vector_product`'s own
    ///   doc comment: "The JVP of a linear map is the map itself"), so no separate PRIMAL
    ///   evaluation at the lifted state is needed for this term.
    /// - `combined[dof] = load[dof] - (A * lifting)[dof]` in full physical space, then
    ///   `constraints().restrict_transpose(&combined)` folds it down to
    ///   [`SystemOperator::dimension`] free/constrained coordinates.
    /// - Every constrained row is finally overwritten with its own `AffineConstraint::offset`
    ///   (the constrained value itself), so the returned vector is ready to use as-is on the
    ///   right-hand side of `self.apply(..)`/`methodus::solve_minres`/`solve_cg` et al: solving
    ///   `self * x = self.load_vector()` returns `x` with the constrained rows honoring their
    ///   Dirichlet data automatically, exactly as `RealizationPlan`'s reduced system already
    ///   does.
    ///
    /// This is a drop-in replacement for any caller that today hand-builds only the Dirichlet-
    /// lifting half of this (i.e. calls `self.operator().apply_action` on the lifting and negates
    /// it, without a `load` term): that caller's existing computation is exactly this method's
    /// `combined` term with `load` fixed at all-zero, so switching to this method changes nothing
    /// when no source is bound, and adds the previously-unrepresentable forcing contribution when
    /// one is.
    pub fn load_vector(&self) -> Result<Vec<f64>, FinitumError> {
        let dimension = self.operator.dimension();
        let load = self.operator.load_vector()?;
        self.require_static_view()?;
        let lifting = self.constraints.expand(&vec![0.0; dimension])?;
        let mut lifted_action = vec![0.0; dimension];
        self.operator.apply_action(&lifting, &mut lifted_action)?;
        let mut combined = vec![0.0; dimension];
        for index in 0..dimension {
            combined[index] = load[index] - lifted_action[index];
        }
        let mut rhs = self.constraints.restrict_transpose(&combined)?;
        for constraint in self.constraints.constraints() {
            rhs[constraint.target.0] = constraint.offset;
        }
        validate_finite("reduced system operator load vector", &rhs)?;
        Ok(rhs)
    }
}

impl LinearOperator for ReducedSystemOperator {
    fn rows(&self) -> usize {
        self.operator.dimension()
    }

    fn columns(&self) -> usize {
        self.operator.dimension()
    }

    /// `Nonsymmetric` whenever `constraints` carries an affine dependency (matching
    /// `RealizationPlan`/`ReducedMixedOperator`'s own convention); otherwise the unconstrained
    /// operator's own structurally-derived symmetry (a Fixed-only constraint set's identity-row/
    /// zero-column elimination preserves symmetry exactly as `ReducedMixedOperator`'s doc comment
    /// argues, so this inherits `self.operator.symmetry()` rather than assuming `Symmetric`
    /// unconditionally).
    fn symmetry(&self) -> OperatorSymmetry {
        if self.constraints.has_affine_dependencies() {
            OperatorSymmetry::Nonsymmetric
        } else {
            self.operator.symmetry()
        }
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.require_static_view().map_err(NumericError::from)?;
        self.operator
            .apply_reduced_action(&self.constraints, input, output)
            .map_err(NumericError::from)
    }
}

impl BlockLinearOperator for ReducedSystemOperator {
    fn block_layout(&self) -> &methodus::BlockLayout {
        self.operator.block_layout()
    }
}

impl ReducedSystemOperator {
    /// Batch P: the essential-constraint-eliminated residual at `(t, u, u_t)`, mirroring
    /// `RealizationPlan::residual` exactly: the state is expanded through the constraints (a
    /// constrained coordinate takes its Dirichlet value), the rate homogeneously plus any
    /// prescribed analytic rate lifting. The physical residual is restricted back, and every
    /// constrained row becomes `state[target] - prescribed_value(time)` -- the
    /// `F(t, y, y') = 0` shape a Krasis transaction or a
    /// Methodus BDF step consumes (this type implements [`DaeOperator`] and
    /// [`NonlinearOperator`] over these actions).
    pub fn residual(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.operator.dimension();
        if output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "reduced system residual expects output length {dimension}, got {}",
                output.len()
            )));
        }
        let (constraints, lifting_rate) = self.prescribed_at(time)?;
        let physical_state = constraints.expand(state)?;
        let mut physical_rate = constraints.expand_homogeneous(state_rate)?;
        for (rate, lifting) in physical_rate.iter_mut().zip(lifting_rate) {
            *rate += lifting;
        }
        validate_finite("prescribed physical rate", &physical_rate)?;
        let mut physical_output = vec![0.0; dimension];
        self.operator
            .residual(time, &physical_state, &physical_rate, &mut physical_output)?;
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] =
                constraints.equation_residual(state, constraint.target)?;
        }
        validate_finite("reduced system residual", output)
    }

    /// Batch P: the constraint-eliminated JVP at `(t, u, u_t)`; constrained rows carry the
    /// direction's own constraint residual, exactly as `RealizationPlan::jacobian_vector_product`.
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
        let dimension = self.operator.dimension();
        if output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "reduced system JVP expects output length {dimension}, got {}",
                output.len()
            )));
        }
        let (constraints, lifting_rate) = self.prescribed_at(time)?;
        let physical_state = constraints.expand(state)?;
        let mut physical_rate = constraints.expand_homogeneous(state_rate)?;
        for (rate, lifting) in physical_rate.iter_mut().zip(lifting_rate) {
            *rate += lifting;
        }
        validate_finite("prescribed physical rate", &physical_rate)?;
        let physical_state_direction = self.constraints.expand_homogeneous(state_direction)?;
        let physical_rate_direction = self.constraints.expand_homogeneous(rate_direction)?;
        let mut physical_output = vec![0.0; dimension];
        self.operator.jacobian_vector_product(
            time,
            &physical_state,
            &physical_rate,
            &physical_state_direction,
            &physical_rate_direction,
            &mut physical_output,
        )?;
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] = self
                .constraints
                .direction_residual(state_direction, constraint.target)?;
        }
        validate_finite("reduced system JVP", output)
    }

    /// SV1-C1: the exact transpose of [`Self::jacobian_vector_product`] with the rate direction
    /// held at zero; see [`Self::vector_jacobian_product_shifted`].
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

    /// SV1-C1: the exact transpose of `x -> jacobian_vector_product(x, rate_shift * x)` under
    /// the constraint elimination, mirroring `RealizationPlan::vector_jacobian_product_shifted`
    /// row for row: fixed constraint rows are their own transpose (their adjoint entries are
    /// masked out of the physical transpose and added back directly); affine dependency
    /// constraints are refused typed until SV1-C2.
    #[allow(clippy::too_many_arguments)]
    pub fn vector_jacobian_product_shifted(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        adjoint: &[f64],
        rate_shift: f64,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.operator.dimension();
        if output.len() != dimension || adjoint.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "reduced system VJP expects adjoint/output length {dimension}, got {}/{}",
                adjoint.len(),
                output.len()
            )));
        }
        if self.constraints.has_affine_dependencies() {
            return Err(FinitumError::UnsupportedRealization(
                "vector_jacobian_product refuses affine dependency constraints; their exact \
                 transpose is not yet implemented (SV1-C2)"
                    .into(),
            ));
        }
        let (constraints, lifting_rate) = self.prescribed_at(time)?;
        let physical_state = constraints.expand(state)?;
        let mut physical_rate = constraints.expand_homogeneous(state_rate)?;
        for (rate, lifting) in physical_rate.iter_mut().zip(lifting_rate) {
            *rate += lifting;
        }
        validate_finite("prescribed physical rate", &physical_rate)?;
        let mut restricted_adjoint = adjoint.to_vec();
        for constraint in self.constraints.constraints() {
            restricted_adjoint[constraint.target.0] = 0.0;
        }
        let physical_adjoint = self.constraints.expand_homogeneous(&restricted_adjoint)?;
        let mut physical_output = vec![0.0; dimension];
        self.operator.vector_jacobian_product_shifted(
            time,
            &physical_state,
            &physical_rate,
            &physical_adjoint,
            rate_shift,
            &mut physical_output,
        )?;
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] += adjoint[constraint.target.0];
        }
        validate_finite("reduced system VJP", output)
    }

    /// SV1-C1: the constraint-eliminated Jacobian `dR/du + rate_shift * dR/du_t` at one
    /// linearization point as a Methodus operator pair (see [`LinearizedSystemOperator`]).
    pub fn linearize(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        rate_shift: f64,
    ) -> Result<LinearizedSystemOperator, FinitumError> {
        validate_rate_shift(rate_shift)?;
        let dimension = self.operator.dimension();
        if state.len() != dimension || state_rate.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "reduced system linearization expects state/rate length {dimension}, got {}/{}",
                state.len(),
                state_rate.len()
            )));
        }
        validate_finite("reduced system linearization state", state)?;
        validate_finite("reduced system linearization rate", state_rate)?;
        Ok(LinearizedSystemOperator {
            operator: self.operator.clone(),
            reduced: Some(self.clone()),
            time,
            state: state.to_vec(),
            state_rate: state_rate.to_vec(),
            rate_shift,
        })
    }

    /// Declared properties of every Jacobian this operator's nonlinear views form. For a system
    /// whose forms are structurally linear in every trial field and carry no time derivative,
    /// the Jacobian at every state is the zero-point linear view, so its declared
    /// [`LinearOperator::properties`] apply; otherwise nothing beyond the block partition is
    /// claimed (a nonlinear form's Jacobian at a nonzero state, or a rate-shifted Jacobian, is
    /// not certified by the zero-point structure or proof).
    fn linearized_properties(&self) -> OperatorProperties {
        let structure = self.operator.structure();
        if structure.trial_linearity == Linearity::Linear && !structure.time.transient {
            LinearOperator::properties(self)
        } else {
            OperatorProperties::new(
                if self.constraints.has_affine_dependencies() {
                    OperatorSymmetry::Nonsymmetric
                } else {
                    OperatorSymmetry::Unknown
                },
                Definiteness::Unknown,
                None,
                OperatorStructureHint::Block {
                    layout: self.operator.data.solver_layout.clone(),
                    saddle_point: structure.saddle_point,
                },
            )
            .expect("reduced system Jacobian properties are internally consistent")
        }
    }
}

impl ReducedSystemOperator {
    /// [`SystemOperator::coefficient_dimension`] of the underlying physical operator.
    pub fn coefficient_dimension(
        &self,
        coefficient: &SystemDistributedCoefficient,
    ) -> Result<usize, FinitumError> {
        self.operator.coefficient_dimension(coefficient)
    }

    /// SC-W1 system-path parity (a): the essential-constraint-eliminated form of
    /// [`SystemOperator::coefficient_jacobian_vector_product`], row for row the
    /// `RealizationPlan` one -- constraint rows carry zero coefficient derivative, because
    /// essential values are frozen inputs of the realization.
    #[allow(clippy::too_many_arguments)]
    pub fn coefficient_jacobian_vector_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        coefficient: &SystemDistributedCoefficient,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.operator.dimension();
        if output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "reduced system coefficient JVP expects output length {dimension}, got {}",
                output.len()
            )));
        }
        let (constraints, lifting_rate) = self.prescribed_at(time)?;
        let physical_state = constraints.expand(state)?;
        let mut physical_rate = constraints.expand_homogeneous(state_rate)?;
        for (rate, lifting) in physical_rate.iter_mut().zip(lifting_rate) {
            *rate += lifting;
        }
        validate_finite("prescribed physical rate", &physical_rate)?;
        let mut physical_output = vec![0.0; dimension];
        self.operator.coefficient_jacobian_vector_product(
            time,
            &physical_state,
            &physical_rate,
            coefficient,
            direction,
            &mut physical_output,
        )?;
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] = 0.0;
        }
        validate_finite("reduced system coefficient JVP", output)
    }

    /// SC-W1 system-path parity (a): the exact transpose of
    /// [`Self::coefficient_jacobian_vector_product`] (constraint rows of the adjoint masked
    /// out before the homogeneous expansion); affine dependency constraints refuse exactly as
    /// [`Self::vector_jacobian_product`] does.
    #[allow(clippy::too_many_arguments)]
    pub fn coefficient_vector_jacobian_product(
        &self,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        coefficient: &SystemDistributedCoefficient,
        adjoint: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.operator.dimension();
        if adjoint.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "reduced system coefficient VJP expects adjoint length {dimension}, got {}",
                adjoint.len()
            )));
        }
        if self.constraints.has_affine_dependencies() {
            return Err(FinitumError::UnsupportedRealization(
                "coefficient_vector_jacobian_product refuses affine dependency constraints; \
                 their exact transpose is not yet implemented (SV1-C2)"
                    .into(),
            ));
        }
        let (constraints, lifting_rate) = self.prescribed_at(time)?;
        let physical_state = constraints.expand(state)?;
        let mut physical_rate = constraints.expand_homogeneous(state_rate)?;
        for (rate, lifting) in physical_rate.iter_mut().zip(lifting_rate) {
            *rate += lifting;
        }
        validate_finite("prescribed physical rate", &physical_rate)?;
        let mut restricted_adjoint = adjoint.to_vec();
        for constraint in self.constraints.constraints() {
            restricted_adjoint[constraint.target.0] = 0.0;
        }
        let physical_adjoint = self.constraints.expand_homogeneous(&restricted_adjoint)?;
        self.operator.coefficient_vector_jacobian_product(
            time,
            &physical_state,
            &physical_rate,
            coefficient,
            &physical_adjoint,
            output,
        )
    }

    /// Canonical CSR assembly of the reduced zero-point linear view by unit-column probing of
    /// [`LinearOperator::apply`] (constraint rows included), the counterpart of
    /// `RealizationPlan::assemble`; Methodus's `CsrMatrix` is `TransposableOperator`, so the
    /// assembled transpose is available without a symmetry declaration.
    pub fn assemble(&self) -> Result<CsrMatrix, FinitumError> {
        assemble_by_probing(self)
    }

    /// [`SystemOperator::element_assembly`] reduced by this operator's constraint rows.
    pub fn element_assembly(
        &self,
        lane_width: usize,
    ) -> Result<ElementAssemblyOperator, FinitumError> {
        self.require_static_view()?;
        self.operator
            .element_assembly_with(self.constraints.clone(), lane_width)
    }

    /// [`SystemOperator::partial_assembly`] reduced by this operator's constraint rows.
    pub fn partial_assembly(
        &self,
        lane_width: usize,
    ) -> Result<SystemPartialAssemblyOperator, FinitumError> {
        self.require_static_view()?;
        self.operator
            .partial_assembly_with(self.constraints.clone(), lane_width)
    }

    /// SC-W1 system-path parity (b): what this reduced system realization admitted, in the
    /// single-model [`RealizationCapability`] shape (`finitum-realization-capability/1`):
    /// every block's admitted element requirements (deduplicated by field), every block
    /// integral's measure, the constraint kinds present, the representation kinds and
    /// derivative products this operator exposes (element/partial assembly only without facet
    /// integrals; `Vjp` and `CoefficientVjp` under the affine-dependency rule; the coefficient
    /// products exactly when a stored cell input is bound), and its declared symmetry. The
    /// receipt's source digests are the per-block requirement/factorization/kernel digests for
    /// a one-block system and the blake3 of their ordered lists otherwise; the realization
    /// digest is [`SystemOperator::digest`].
    pub fn capability(&self) -> RealizationCapability {
        let data = &self.operator.data;
        let mut elements = Vec::new();
        let mut seen_elements = BTreeSet::new();
        for (_, row, block) in data.plan.rows() {
            for element in &block.requirements.elements {
                if !seen_elements.insert((row.instance, element.symbol)) {
                    continue;
                }
                elements.push(CapabilityElement {
                    symbol: element.symbol,
                    topological_dimension: element.topological_dimension,
                    family: element.family,
                    polynomial_order: element.polynomial_order,
                    value_shape: element.value_shape.clone(),
                });
            }
        }
        let measures = data
            .plan
            .rows()
            .flat_map(|(_, _, block)| block.factorization.integrals.iter())
            .map(|integral| integral.measure.clone())
            .collect::<Vec<_>>();
        let mut constraint_kinds = BTreeSet::new();
        for constraint in self.constraints.constraints() {
            if constraint.dependencies.is_empty() {
                constraint_kinds.insert(ConstraintKind::Fixed);
            } else {
                constraint_kinds.insert(ConstraintKind::AffineDependency);
            }
        }
        let mut representation_kinds = vec![
            RepresentationKind::MatrixFree,
            RepresentationKind::Assembled,
        ];
        if self.motion.is_some() {
            representation_kinds = vec![RepresentationKind::MatrixFree];
        }
        if self.motion.is_none() && data.facet_regions.is_empty() {
            representation_kinds.push(RepresentationKind::ElementAssembly);
            if data.constitutive.is_empty() && data.binds.is_empty() {
                representation_kinds.push(RepresentationKind::PartialAssembly);
            }
        }
        let affine = self.constraints.has_affine_dependencies();
        let mut derivative_products = vec![DerivativeProduct::Primal, DerivativeProduct::Jvp];
        if !affine {
            derivative_products.push(DerivativeProduct::Vjp);
        }
        if !data.stored.is_empty() {
            derivative_products.push(DerivativeProduct::CoefficientJvp);
            if !affine {
                derivative_products.push(DerivativeProduct::CoefficientVjp);
            }
        }
        let receipt = RealizationReceipt {
            source_requirements_digest: combined_digest(
                data.plan
                    .rows()
                    .map(|(_, _, block)| &block.requirements.artifact_digest),
            ),
            source_factorization_digest: combined_digest(
                data.plan
                    .rows()
                    .map(|(_, _, block)| &block.factorization.artifact_digest),
            ),
            source_kernels_digest: combined_digest(
                data.plan
                    .rows()
                    .map(|(_, _, block)| &block.kernels.artifact_digest),
            ),
            realization_digest: self.realization_digest(),
        };
        build_capability(
            data.plan.mesh().dimension(),
            elements,
            measures,
            constraint_kinds.into_iter().collect(),
            representation_kinds,
            derivative_products,
            self.symmetry(),
            receipt,
        )
    }

    /// SC-W1 system-path parity (b): the inspectable projection of this reduced system
    /// realization ([`SystemRealizationArtifact`]).
    pub fn artifact(&self) -> SystemRealizationArtifact {
        let data = &self.operator.data;
        let ids = data.plan.system_ids();
        let blocks = data
            .plan
            .rows()
            .map(|(_, row, block)| SystemBlockReceipt {
                equation: row.path.clone(),
                residual: Some(row.residual),
                source_requirements_digest: block.requirements.artifact_digest.clone(),
                source_factorization_digest: block.factorization.artifact_digest.clone(),
                source_kernels_digest: block.kernels.artifact_digest.clone(),
            })
            .collect();
        let fields = data
            .fields
            .iter()
            .map(|(&variable, field)| SystemFieldArtifact {
                symbol: data
                    .plan
                    .layout()
                    .block_by_variable(variable)
                    .expect("realized field implies a layout block")
                    .symbol,
                variable,
                dofs: field.dofs.clone(),
            })
            .collect();
        let residual_of = |block_index: usize| Some(data.plan.rows[block_index].residual);
        let mut external_inputs = Vec::new();
        for (&(block_index, integral_index, input), table) in &data.stored {
            external_inputs.push(SystemRealizationExternalInput {
                residual: residual_of(block_index),
                input: RealizationExternalInput::Stored {
                    integral_index,
                    input,
                    component_count: table.component_count(),
                    values: table.values().to_vec(),
                },
            });
        }
        for (&(block_index, integral_index, input), closure) in &data.constitutive {
            external_inputs.push(SystemRealizationExternalInput {
                residual: residual_of(block_index),
                input: RealizationExternalInput::Dynamic {
                    integral_index,
                    input,
                    component_count: closure.component_count,
                    identity: closure.identity.clone(),
                },
            });
        }
        let instances = if data.plan.composed.is_some() {
            ids.instances().to_vec()
        } else {
            Vec::new()
        };
        SystemRealizationArtifact {
            schema: SYSTEM_REALIZATION_ARTIFACT_SCHEMA.into(),
            artifact_digest: self.realization_digest(),
            plan_digest: data.plan.artifact_digest().clone(),
            system_ids_identity: ids.identity().clone(),
            blocks,
            mesh: data.plan.mesh().clone(),
            fields,
            constraints: self.constraints.clone(),
            external_inputs,
            instances,
            binds: self.operator.binds(),
            prescribed_motion_identity: self.motion.as_ref().map(|m| m.identity.clone()),
        }
    }
}

/// The per-block digest itself for one block, the blake3 of the ordered list otherwise.
fn combined_digest<'a>(digests: impl Iterator<Item = &'a Digest>) -> Digest {
    let digests = digests.collect::<Vec<_>>();
    match digests.as_slice() {
        [single] => (*single).clone(),
        many => Digest::blake3(&serde_json::to_vec(many).expect("digests are serializable")),
    }
}

fn assemble_by_probing(operator: &dyn LinearOperator) -> Result<CsrMatrix, FinitumError> {
    let rows = operator.rows();
    let columns = operator.columns();
    let context = EvaluationContext::reproducible();
    let mut entries = Vec::new();
    let mut direction = vec![0.0; columns];
    let mut output = vec![0.0; rows];
    for column in 0..columns {
        direction[column] = 1.0;
        operator
            .apply(&context, &direction, &mut output)
            .map_err(|error| FinitumError::Assembly(error.to_string()))?;
        for (row, value) in output.iter().copied().enumerate() {
            if value != 0.0 {
                entries.push((row, column, value));
            }
        }
        direction[column] = 0.0;
    }
    CsrMatrix::from_triplets(rows, columns, entries)
        .map_err(|error| FinitumError::Assembly(error.to_string()))
}

pub const SYSTEM_REALIZATION_ARTIFACT_SCHEMA: &str = "finitum-system-realization-artifact/1";

/// Receipt of one system block's Scientia artifact chain in a [`SystemRealizationArtifact`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemBlockReceipt {
    pub equation: String,
    pub residual: Option<SysResId>,
    pub source_requirements_digest: Digest,
    pub source_factorization_digest: Digest,
    pub source_kernels_digest: Digest,
}

/// One realized field of a [`SystemRealizationArtifact`]: its per-model symbol, its system
/// variable, and its DOF map (the layout block it occupies is `symbol`'s in the layout).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemFieldArtifact {
    pub symbol: SymbolId,
    pub variable: SysVarId,
    pub dofs: DofMap,
}

/// One bound non-basis input of a [`SystemRealizationArtifact`], in the single-model
/// [`RealizationExternalInput`] shape plus the residual it belongs to.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemRealizationExternalInput {
    pub residual: Option<SysResId>,
    pub input: RealizationExternalInput,
}

/// SC-W1 system-path parity (b): the stable, inspectable projection of one reduced system
/// realization -- the counterpart of [`crate::RealizationArtifact`] with one entry per block,
/// field, and bound input. Like it, this is not a reconstruction API: bound kernels are
/// absent and constitutive closures appear only by their digest-covered identity.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemRealizationArtifact {
    pub schema: String,
    /// [`SystemOperator::digest`] for static data; includes prescribed motion identity and
    /// target descriptors when runtime essential values are attached.
    pub artifact_digest: Digest,
    pub plan_digest: Digest,
    /// [`SystemIdMap::identity`] (`finitum-system-ids/1`).
    pub system_ids_identity: Digest,
    pub blocks: Vec<SystemBlockReceipt>,
    pub mesh: Mesh,
    pub fields: Vec<SystemFieldArtifact>,
    pub constraints: ConstraintSet,
    pub external_inputs: Vec<SystemRealizationExternalInput>,
    /// W8 F-MI: the instances of a composed plan (empty on a one-instance plan, whose
    /// serialized form is therefore unchanged).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub instances: Vec<crate::system_ids::InstanceRecord>,
    /// W8 F-MI: every same-mesh bind the operator realizes (empty on a one-instance plan).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub binds: Vec<SystemBindReceipt>,
    /// Optional prescribed value/rate identity; absent on static artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prescribed_motion_identity: Option<String>,
}

/// One same-mesh bind of a composed [`SystemOperator`] (W8 lane F-MI): which slot it closes,
/// the producer output, the path it takes, the `(row, column)` blocks it occupies in the
/// `scientia-operator-system/2` artifact, and the Malleus composition digests (kernel-input
/// path) the operator validated and, for the primal, executes.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemBindReceipt {
    pub consumer_slot: String,
    pub consumer: InstanceId,
    pub producer: InstanceId,
    /// `<producer instance>.<output>`.
    pub output: String,
    pub path: BindPath,
    pub rows: Vec<SysResId>,
    pub columns: Vec<SysVarId>,
    pub compositions: Vec<Digest>,
    pub jvp_compositions: Vec<Digest>,
}

/// One stored quadrature-point Jacobian of a [`SystemPartialAssemblyOperator`].
#[derive(Clone, Debug)]
struct SystemPartialPointAction {
    row: SysVarId,
    instance: InstanceId,
    point: usize,
    /// Quadrature weight x cell determinant x the row's `equation_sign`.
    scale: f64,
    active_inputs: Vec<QFunctionInput>,
    output_derivative: DerivativeEvaluation,
    output_components: usize,
    /// Row-major point Jacobian from the concatenated active field evaluations to one output.
    matrix: Vec<f64>,
}

/// SC-W1 system-path parity (b): the quadrature-data (`E^T B^T D B E`) realization of a
/// system's zero-point linear view -- per-cell, per-point stored Jacobians applied through
/// each field's own basis action and the row field's basis transpose, never a cell or global
/// matrix (see [`SystemOperator::partial_assembly`]). Constraint rows are handled exactly as
/// [`crate::PartialAssemblyOperator`]'s.
#[derive(Clone, Debug)]
pub struct SystemPartialAssemblyOperator {
    operator: SystemOperator,
    constraints: ConstraintSet,
    point_actions: Vec<Vec<SystemPartialPointAction>>,
    batches: CellBatchLayout,
}

impl SystemPartialAssemblyOperator {
    pub fn batches(&self) -> &CellBatchLayout {
        &self.batches
    }

    pub fn stored_point_action_count(&self) -> usize {
        self.point_actions.iter().map(Vec::len).sum()
    }

    pub(crate) fn apply_inner(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.operator.dimension();
        if input.len() != dimension || output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "system partial-assembly input/output must contain {dimension} values"
            )));
        }
        validate_finite("system partial-assembly input", input)?;
        let physical = self.constraints.expand_homogeneous(input)?;
        let data = &self.operator.data;
        let mesh = data.plan.mesh();
        let layout = data.plan.layout();
        let mut physical_output = vec![0.0; dimension];
        for batch in 0..self.batches.batch_count() {
            for cell in self.batches.batch(batch).expect("batch index is bounded") {
                let Some(cell) = *cell else { continue };
                let geometry = CellGeometry::new(mesh, CellId(cell))?;
                let affine = AffineMap::from_cell(mesh, CellId(cell))?;
                let local = self.operator.gather_cell(cell, &physical);
                let mut local_outputs = local
                    .iter()
                    .map(|(&variable, values)| (variable, vec![0.0; values.len()]))
                    .collect::<BTreeMap<_, _>>();
                for action in &self.point_actions[cell] {
                    let reference_point = &data.quadrature[action.point].coordinates;
                    let fields = self.operator.instance_fields(action.instance);
                    let mut point_input = Vec::new();
                    for qinput in &action.active_inputs {
                        if qinput.binding.evaluation.derivative
                            == DerivativeEvaluation::TimeDerivative
                        {
                            point_input.extend(vec![0.0; component_count(&qinput.shape)?]);
                            continue;
                        }
                        let (variable, field) = fields.field(qinput.binding.symbol)?;
                        point_input.extend(evaluate_field_basis_input(
                            field,
                            &geometry,
                            &affine,
                            cell,
                            action.point,
                            reference_point,
                            qinput,
                            &local[&variable],
                        )?);
                    }
                    let input_components = point_input.len();
                    let mut point_output = vec![0.0; action.output_components];
                    for (row, value) in point_output.iter_mut().enumerate() {
                        *value = (0..input_components)
                            .map(|column| {
                                action.matrix[row * input_components + column] * point_input[column]
                            })
                            .sum();
                    }
                    apply_field_basis_adjoint(
                        &data.fields[&action.row],
                        &geometry,
                        &affine,
                        cell,
                        action.point,
                        reference_point,
                        &action.output_derivative,
                        &point_output,
                        action.scale,
                        local_outputs
                            .get_mut(&action.row)
                            .expect("row field is realized"),
                    )?;
                }
                for (variable, local_output) in &local_outputs {
                    let offset = layout
                        .block_by_variable(*variable)
                        .expect("realized field implies a layout block")
                        .offset;
                    let restriction = &data.fields[variable].dofs.restrictions()[cell];
                    for (local_index, dof) in restriction.dofs.iter().enumerate() {
                        physical_output[offset + dof.0] += local_output[local_index];
                    }
                }
            }
        }
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] = self
                .constraints
                .direction_residual(input, constraint.target)?;
        }
        validate_finite("system partial-assembly output", output)
    }
}

impl LinearOperator for SystemPartialAssemblyOperator {
    fn rows(&self) -> usize {
        self.operator.dimension()
    }

    fn columns(&self) -> usize {
        self.operator.dimension()
    }

    fn symmetry(&self) -> OperatorSymmetry {
        if self.constraints.has_affine_dependencies() {
            OperatorSymmetry::Nonsymmetric
        } else {
            self.operator.symmetry()
        }
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.apply_inner(input, output).map_err(numeric_error)
    }
}

impl DaeOperator for ReducedSystemOperator {
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    fn jacobian_properties(&self) -> OperatorProperties {
        self.linearized_properties()
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        ReducedSystemOperator::residual(self, time, state, state_rate, output)
            .map_err(numeric_error)
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        ReducedSystemOperator::jacobian_vector_product(
            self,
            time,
            state,
            state_rate,
            state_direction,
            rate_direction,
            output,
        )
        .map_err(numeric_error)
    }
}

/// The steady view at `t = 0`, `u_t = 0` (mirroring Krasis's `CoupledOperator` convention).
impl NonlinearOperator for ReducedSystemOperator {
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    fn jacobian_properties(&self) -> OperatorProperties {
        self.linearized_properties()
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.require_static_view().map_err(numeric_error)?;
        let zero = vec![0.0; self.operator.dimension()];
        ReducedSystemOperator::residual(self, 0.0, state, &zero, output).map_err(numeric_error)
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.require_static_view().map_err(numeric_error)?;
        let zero = vec![0.0; self.operator.dimension()];
        ReducedSystemOperator::jacobian_vector_product(
            self, 0.0, state, &zero, direction, &zero, output,
        )
        .map_err(numeric_error)
    }
}

impl TransposableOperator for ReducedSystemOperator {
    /// SV1-C1: the exact transpose of the zero-point reduced action ([`LinearOperator::apply`]).
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.require_static_view().map_err(numeric_error)?;
        let zero = vec![0.0; self.operator.dimension()];
        self.vector_jacobian_product(0.0, &zero, &zero, input, output)
            .map_err(numeric_error)
    }
}

impl TransposableOperator for SystemOperator {
    /// SV1-C1: the exact transpose of the zero-point action ([`Self::apply_action`]).
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; self.dimension()];
        self.vector_jacobian_product(0.0, &zero, &zero, input, output)
            .map_err(numeric_error)
    }
}

/// SV1-C1 over a system: the Jacobian `dR/du + rate_shift * dR/du_t` of one [`SystemOperator`]
/// (or of its essential-constraint-eliminated [`ReducedSystemOperator`]) at a fixed
/// linearization point, as a Methodus `LinearOperator` (bound JVP kernels) whose
/// `TransposableOperator` action executes the bound VJP kernels, partitioned by the system's
/// block layout. Constructed by [`SystemOperator::linearize`] /
/// [`ReducedSystemOperator::linearize`].
///
/// Symmetry: `Nonsymmetric` under affine dependency constraints; the zero-point operator's
/// declared symmetry when the system is structurally linear in every trial field and
/// `rate_shift == 0` (the Jacobian is then that same operator at every state); `Unknown`
/// otherwise. Adjoint consumers use `methodus::TransposeOperator::explicit`.
#[derive(Clone, Debug)]
pub struct LinearizedSystemOperator {
    operator: SystemOperator,
    reduced: Option<ReducedSystemOperator>,
    time: f64,
    state: Vec<f64>,
    state_rate: Vec<f64>,
    rate_shift: f64,
}

impl LinearizedSystemOperator {
    pub fn operator(&self) -> &SystemOperator {
        &self.operator
    }

    /// The constraint set eliminated by this linearization, if any.
    pub fn constraints(&self) -> Option<&ConstraintSet> {
        self.reduced
            .as_ref()
            .map(ReducedSystemOperator::constraints)
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

    /// Canonical CSR assembly of this Jacobian at its linearization point by unit-column
    /// probing of [`LinearOperator::apply`] -- the system counterpart of
    /// `RealizationPlan::assemble` for a nonlinear state; Methodus's `CsrMatrix` is
    /// `TransposableOperator`, so the assembled transpose action is available alongside
    /// [`TransposableOperator::apply_transpose`]'s kernel-executed one.
    pub fn assemble(&self) -> Result<CsrMatrix, FinitumError> {
        assemble_by_probing(self)
    }
}

impl LinearOperator for LinearizedSystemOperator {
    fn rows(&self) -> usize {
        self.operator.dimension()
    }

    fn columns(&self) -> usize {
        self.operator.dimension()
    }

    fn symmetry(&self) -> OperatorSymmetry {
        if self
            .reduced
            .as_ref()
            .is_some_and(|reduced| reduced.constraints.has_affine_dependencies())
        {
            return OperatorSymmetry::Nonsymmetric;
        }
        if self.operator.structure().trial_linearity == Linearity::Linear && self.rate_shift == 0.0
        {
            match &self.reduced {
                Some(reduced) => reduced.symmetry(),
                None => self.operator.symmetry(),
            }
        } else {
            OperatorSymmetry::Unknown
        }
    }

    fn properties(&self) -> OperatorProperties {
        OperatorProperties::new(
            self.symmetry(),
            Definiteness::Unknown,
            None,
            OperatorStructureHint::Block {
                layout: self.operator.data.solver_layout.clone(),
                saddle_point: self.operator.structure().saddle_point,
            },
        )
        .expect("linearized system operator properties are internally consistent")
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
        match &self.reduced {
            Some(reduced) => reduced.jacobian_vector_product(
                self.time,
                &self.state,
                &self.state_rate,
                input,
                &rate_direction,
                output,
            ),
            None => self.operator.jacobian_vector_product(
                self.time,
                &self.state,
                &self.state_rate,
                input,
                &rate_direction,
                output,
            ),
        }
        .map_err(numeric_error)
    }
}

impl TransposableOperator for LinearizedSystemOperator {
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        match &self.reduced {
            Some(reduced) => reduced.vector_jacobian_product_shifted(
                self.time,
                &self.state,
                &self.state_rate,
                input,
                self.rate_shift,
                output,
            ),
            None => self.operator.vector_jacobian_product_shifted(
                self.time,
                &self.state,
                &self.state_rate,
                input,
                self.rate_shift,
                output,
            ),
        }
        .map_err(numeric_error)
    }
}

impl BlockLinearOperator for LinearizedSystemOperator {
    fn block_layout(&self) -> &methodus::BlockLayout {
        &self.operator.data.solver_layout
    }
}

/// Batch P / GX-A3 in the system path: builds every block's [`SystemConstitutiveInput`] from a
/// caller-supplied `(SymbolId, FieldSource)` table, mirroring
/// [`crate::external_inputs_from`]'s resolution rules field-for-field, so a case's bound
/// property kernels and tables carry their exact tangents into the system JVP/VJP without a
/// hand-written closure per input:
///
/// - a `Kernel`/`Table` whose declared inputs (axes) name exactly one active field of the same
///   integral becomes a state-dependent closure pair: the value evaluates the kernel/table at
///   the point's coordinates, time, and that field's *value* (looked up by the active input's
///   own [`TensorInputId`], so several fields sharing an evaluation kind stay distinct); the
///   direction is the exact kernel tangent / table slope times that field's direction. A
///   kernel with no tangent for that input, or a table with `TableDerivativePolicy::Unavailable`,
///   is refused (`FinitumError::RealizationTangentUnavailable`); a source naming more than one
///   active field, or one whose field has no `Value`-kind active input on that integral, is
///   refused typed;
/// - a coordinate/time-only `Kernel`/`Table`, a `Constant`, or a `Sampled` source is a state-
///   independent closure with an identically zero direction;
/// - a `Nodal` source has no coordinate sampler and is refused typed.
///
/// Every cell integral's non-`Basis` input must have a source entry (`FinitumError::
/// InvalidRealization` otherwise); admitted exterior-facet integrals have no such input.
pub fn system_constitutive_from_sources(
    system: &OperatorSystem,
    model: &SemanticModel,
    sources: &[(SymbolId, FieldSource)],
) -> Result<Vec<SystemConstitutiveInput>, FinitumError> {
    let sources_by_symbol = sources
        .iter()
        .map(|(symbol, source)| (*symbol, source))
        .collect::<BTreeMap<_, _>>();
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
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
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let symbol = input.binding.symbol;
                let source = *sources_by_symbol.get(&symbol).ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "system_constitutive_from_sources has no FieldSource for symbol {symbol} \
                         (equation `{}` integral {})",
                        block.equation, integral.integral_index
                    ))
                })?;
                let components = component_count(&input.shape)?;
                let origin = format!(
                    "{}.{}[{}].{}",
                    model.name,
                    block.equation,
                    integral.integral_index,
                    model.symbols[symbol.index()].name
                );
                let built = match source {
                    FieldSource::Kernel { kernel, executable } => {
                        let names = kernel
                            .inputs
                            .iter()
                            .map(|slot| slot.name.as_str())
                            .collect::<Vec<_>>();
                        match resolve_state_dependence(model, &active_inputs, &names)? {
                            None => {
                                if components != 1 {
                                    return Err(FinitumError::UnsupportedRealization(format!(
                                        "kernel field sources evaluate to one scalar; equation \
                                         `{}` integral {} input {:?} has {components} components",
                                        block.equation, integral.integral_index, input.id
                                    )));
                                }
                                let kernel = kernel.clone();
                                let executable = executable.clone();
                                let origin = origin.clone();
                                SystemConstitutiveInput::try_new(
                                    block.equation.clone(),
                                    integral.integral_index,
                                    input.id,
                                    1,
                                    format!(
                                        "finitum.system-field-source-kernel/1:{}",
                                        kernel.identity
                                    ),
                                    move |point: &PointEvaluation| {
                                        let named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        evaluate_kernel_value(&kernel, &executable, &named)
                                            .map(|value| vec![value])
                                            .map_err(|error| property_unavailable(&origin, error))
                                    },
                                    |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
                                )?
                            }
                            Some((name, active_id)) => {
                                if components != 1 {
                                    return Err(FinitumError::UnsupportedRealization(
                                        "state-dependent kernel field sources support one scalar \
                                         active field reference only"
                                            .into(),
                                    ));
                                }
                                if !kernel.tangents.iter().any(|tangent| tangent.input == name) {
                                    return Err(FinitumError::RealizationTangentUnavailable(
                                        format!(
                                            "property kernel {:?} declares no tangent for its \
                                             state-dependent input {name:?}",
                                            kernel.identity
                                        ),
                                    ));
                                }
                                let value_kernel = kernel.clone();
                                let value_executable = executable.clone();
                                let value_name = name.clone();
                                let direction_kernel = kernel.clone();
                                let direction_executable = executable.clone();
                                let direction_name = name.clone();
                                let value_origin = origin.clone();
                                let direction_origin = origin.clone();
                                SystemConstitutiveInput::try_new(
                                    block.equation.clone(),
                                    integral.integral_index,
                                    input.id,
                                    1,
                                    format!(
                                        "finitum.system-field-source-kernel/1:{}",
                                        kernel.identity
                                    ),
                                    move |point: &PointEvaluation| {
                                        let mut named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let active =
                                            active_values(&value_origin, point, active_id)?;
                                        named.insert(value_name.clone(), active[0]);
                                        evaluate_kernel_value(
                                            &value_kernel,
                                            &value_executable,
                                            &named,
                                        )
                                        .map(|value| vec![value])
                                        .map_err(|error| property_unavailable(&value_origin, error))
                                    },
                                    move |point: &PointEvaluation, direction: &PointEvaluation| {
                                        let mut named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let active =
                                            active_values(&direction_origin, point, active_id)?;
                                        let active_direction =
                                            active_values(&direction_origin, direction, active_id)?;
                                        named.insert(direction_name.clone(), active[0]);
                                        let partial = evaluate_kernel_partial(
                                            &direction_kernel,
                                            &direction_executable,
                                            &named,
                                            &direction_name,
                                        )
                                        .map_err(|error| {
                                            property_unavailable(&direction_origin, error)
                                        })?
                                        .ok_or_else(|| {
                                            tangent_unavailable(&direction_origin, &direction_name)
                                        })?;
                                        Ok(vec![partial * active_direction[0]])
                                    },
                                )?
                            }
                        }
                    }
                    FieldSource::Table(table) => {
                        let names = table
                            .axes
                            .iter()
                            .map(|axis| axis.name.as_str())
                            .collect::<Vec<_>>();
                        if components != 1 {
                            return Err(FinitumError::UnsupportedRealization(format!(
                                "table field sources evaluate to one scalar; equation `{}` \
                                 integral {} input {:?} has {components} components",
                                block.equation, integral.integral_index, input.id
                            )));
                        }
                        let identity = format!(
                            "finitum.system-field-source-table/1:{}",
                            source.identity().hex
                        );
                        match resolve_state_dependence(model, &active_inputs, &names)? {
                            None => {
                                let table = table.clone();
                                let origin = origin.clone();
                                SystemConstitutiveInput::try_new(
                                    block.equation.clone(),
                                    integral.integral_index,
                                    input.id,
                                    1,
                                    identity,
                                    move |point: &PointEvaluation| {
                                        let named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let axis_point = table_axis_point(&origin, &table, &named)?;
                                        evaluate_table_value(&table, &axis_point)
                                            .map(|value| vec![value])
                                            .map_err(|error| property_unavailable(&origin, error))
                                    },
                                    |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
                                )?
                            }
                            Some((name, active_id)) => {
                                if table.derivative_policy
                                    == scientia::TableDerivativePolicy::Unavailable
                                {
                                    return Err(FinitumError::RealizationTangentUnavailable(
                                        format!(
                                            "property table axis {name:?} has no derivative policy"
                                        ),
                                    ));
                                }
                                let axis_index = table
                                    .axes
                                    .iter()
                                    .position(|axis| axis.name == name)
                                    .expect("name was derived from table.axes");
                                let value_table = table.clone();
                                let value_name = name.clone();
                                let slope_table = table.clone();
                                let slope_name = name.clone();
                                let value_origin = origin.clone();
                                let slope_origin = origin.clone();
                                SystemConstitutiveInput::try_new(
                                    block.equation.clone(),
                                    integral.integral_index,
                                    input.id,
                                    1,
                                    identity,
                                    move |point: &PointEvaluation| {
                                        let mut named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let active =
                                            active_values(&value_origin, point, active_id)?;
                                        named.insert(value_name.clone(), active[0]);
                                        let axis_point =
                                            table_axis_point(&value_origin, &value_table, &named)?;
                                        evaluate_table_value(&value_table, &axis_point)
                                            .map(|value| vec![value])
                                            .map_err(|error| {
                                                property_unavailable(&value_origin, error)
                                            })
                                    },
                                    move |point: &PointEvaluation, direction: &PointEvaluation| {
                                        let mut named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let active =
                                            active_values(&slope_origin, point, active_id)?;
                                        let active_direction =
                                            active_values(&slope_origin, direction, active_id)?;
                                        named.insert(slope_name.clone(), active[0]);
                                        let axis_point =
                                            table_axis_point(&slope_origin, &slope_table, &named)?;
                                        let slope = evaluate_table_slope(
                                            &slope_table,
                                            &axis_point,
                                            axis_index,
                                        )
                                        .map_err(|error| {
                                            property_unavailable(&slope_origin, error)
                                        })?;
                                        Ok(vec![slope * active_direction[0]])
                                    },
                                )?
                            }
                        }
                    }
                    FieldSource::Constant(values) => {
                        if values.len() != components {
                            return Err(FinitumError::InvalidRealization(format!(
                                "constant field source for symbol {symbol} has {} components, \
                                 equation `{}` integral {} input {:?} expects {components}",
                                values.len(),
                                block.equation,
                                integral.integral_index,
                                input.id
                            )));
                        }
                        let values = values.clone();
                        SystemConstitutiveInput::new(
                            block.equation.clone(),
                            integral.integral_index,
                            input.id,
                            components,
                            format!("finitum.system-field-source/1:{}", source.identity().hex),
                            move |_: &PointEvaluation| values.clone(),
                            move |_: &PointEvaluation, _: &PointEvaluation| vec![0.0; components],
                        )?
                    }
                    FieldSource::Sampled(sampler) => {
                        let sampler = sampler.clone();
                        SystemConstitutiveInput::new(
                            block.equation.clone(),
                            integral.integral_index,
                            input.id,
                            components,
                            format!("finitum.system-field-source/1:{}", source.identity().hex),
                            move |point: &PointEvaluation| sampler(&point.coordinates),
                            move |_: &PointEvaluation, _: &PointEvaluation| vec![0.0; components],
                        )?
                    }
                    FieldSource::Fallible(sampler) => {
                        let sampler = sampler.clone();
                        SystemConstitutiveInput::try_new(
                            block.equation.clone(),
                            integral.integral_index,
                            input.id,
                            components,
                            format!("finitum.system-field-source/1:{}", source.identity().hex),
                            move |point: &PointEvaluation| sampler(&point.coordinates, point.time),
                            move |_: &PointEvaluation, _: &PointEvaluation| {
                                Ok(vec![0.0; components])
                            },
                        )?
                    }
                    FieldSource::Nodal(_) => {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "a Nodal field source has no coordinate sampler; symbol {symbol} \
                             (equation `{}` integral {}) needs a Constant, Sampled, Kernel, or \
                             Table source in the system path",
                            block.equation, integral.integral_index
                        )));
                    }
                };
                constitutive.push(built);
            }
        }
    }
    Ok(constitutive)
}

/// Which declared input `names` of a kernel/table name an active field of one integral, and
/// the [`TensorInputId`] of that field's `Value`-kind active input. `None` when no name is an
/// active field (a coordinate/time-only source); refuses more than one such name (ambiguous
/// chain-rule combination, out of bounded scope) or a field that appears in the integral only
/// through a derivative evaluation (the source needs its value).
fn resolve_state_dependence(
    model: &SemanticModel,
    active_inputs: &[&QFunctionInput],
    names: &[&str],
) -> Result<Option<(String, TensorInputId)>, FinitumError> {
    let active_names = active_inputs
        .iter()
        .map(|input| model.symbols[input.binding.symbol.index()].name.as_str())
        .collect::<BTreeSet<_>>();
    let state_names = names
        .iter()
        .filter(|name| active_names.contains(*name))
        .collect::<Vec<_>>();
    match state_names.as_slice() {
        [] => Ok(None),
        [name] => {
            let value_input = active_inputs.iter().find(|input| {
                model.symbols[input.binding.symbol.index()].name == **name
                    && input.binding.evaluation.derivative == DerivativeEvaluation::Value
            });
            match value_input {
                Some(input) => Ok(Some(((*name).to_string(), input.id))),
                None => Err(FinitumError::UnsupportedRealization(format!(
                    "field source depends on active field {name:?}, which this integral \
                     evaluates only through a derivative; a Value-kind active input is required"
                ))),
            }
        }
        _ => Err(FinitumError::UnsupportedRealization(
            "state-dependent field sources support one active field reference only".into(),
        )),
    }
}

/// Exact transpose of [`apply_field_basis_adjoint`]'s scatter: gathers the row field's local
/// adjoint at one quadrature point into the point-space seed of the bound VJP kernel,
/// unscaled (the quadrature scale is applied once, at the trial-side scatter that follows).
#[allow(clippy::too_many_arguments)]
fn gather_field_test_adjoint(
    field: &FieldElement,
    geometry: &CellGeometry,
    affine: &AffineMap,
    cell: usize,
    point: usize,
    reference_point: &[f64],
    derivative: &DerivativeEvaluation,
    output_components: usize,
    local_adjoint: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    match &field.kind {
        FieldKind::Lagrange(element) => gather_test_adjoint(
            element,
            geometry,
            point,
            derivative,
            output_components,
            local_adjoint,
        ),
        FieldKind::Hdiv0 { orientations } => {
            let cell_orientations = orientations.get(cell).ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "RT0 field has no orientation row for cell {cell}"
                ))
            })?;
            // `apply_rt0_value_adjoint`/`apply_rt0_divergence_adjoint` are the exact scaled
            // transposes of `evaluate_rt0_value`/`evaluate_rt0_divergence`, so the unscaled
            // gather is the evaluation itself applied to the adjoint coefficients.
            match derivative {
                DerivativeEvaluation::Value => {
                    evaluate_rt0_value(affine, cell_orientations, reference_point, local_adjoint)
                }
                DerivativeEvaluation::Divergence if output_components == 1 => {
                    evaluate_rt0_divergence(affine, cell_orientations, local_adjoint)
                        .map(|value| vec![value])
                }
                other => Err(FinitumError::UnsupportedRealization(format!(
                    "RT0 (Hdiv(order=0)) fields support Value/Divergence adjoint gathers only, \
                     got {other:?}"
                ))),
            }
        }
    }
}

/// Scatters one column field's point cotangent through that field's own basis into its local
/// output (the trial-side scatter of the system VJP).
#[allow(clippy::too_many_arguments)]
fn scatter_field_cotangent(
    fields: InstanceFields<'_>,
    symbol: SymbolId,
    geometry: &CellGeometry,
    affine: &AffineMap,
    cell: usize,
    point: usize,
    reference_point: &[f64],
    derivative: &DerivativeEvaluation,
    cotangent: &[f64],
    scale: f64,
    local_outputs: &mut BTreeMap<SysVarId, Vec<f64>>,
) -> Result<(), FinitumError> {
    let (variable, field) = fields.field(symbol).map_err(|_| {
        FinitumError::ArtifactMismatch(format!(
            "VJP cotangent references field {symbol} which the system operator has not realized"
        ))
    })?;
    let local_output = local_outputs
        .get_mut(&variable)
        .expect("every realized field has a local output");
    apply_field_basis_adjoint(
        field,
        geometry,
        affine,
        cell,
        point,
        reference_point,
        derivative,
        cotangent,
        scale,
        local_output,
    )
}

/// A cell's local state and rate directions by variable, when a JVP is being evaluated.
type LocalDirections<'a> = Option<(
    &'a BTreeMap<SysVarId, Vec<f64>>,
    &'a BTreeMap<SysVarId, Vec<f64>>,
)>;

/// The per-point cotangent seeds of every bind's producer output (W8 F-MI), indexed like
/// `SystemOperatorData::binds`; filled by [`accumulate_parameter_cotangents_system`] and
/// pushed back through the producer kernels by [`SystemOperator::push_bound_cotangents`].
type BoundSeeds = Vec<Option<Vec<f64>>>;

fn add_bound_seed(seeds: &mut BoundSeeds, bind: usize, contribution: &[f64]) {
    let entry = seeds[bind].get_or_insert_with(|| vec![0.0; contribution.len()]);
    for (total, value) in entry.iter_mut().zip(contribution) {
        *total += value;
    }
}

/// A direction [`PointEvaluation`] that is zero on every active input and every bound input
/// except one unit component of the bound input closing `symbol` -- the probe of a
/// constitutive closure's chain rule through a same-mesh bind (W8 F-MI).
fn probe_bound_direction_evaluation(
    evaluation: &PointEvaluation,
    active_inputs: &[&QFunctionInput],
    symbol: SymbolId,
    component: usize,
) -> Result<PointEvaluation, FinitumError> {
    let mut active = Vec::with_capacity(active_inputs.len());
    for input in active_inputs {
        active.push(PointActiveInput {
            input: input.id,
            derivative: input.binding.evaluation.derivative,
            values: vec![0.0; component_count(&input.shape)?],
        });
    }
    let bound = evaluation
        .bound
        .iter()
        .map(|input| {
            let mut values = vec![0.0; input.values.len()];
            if input.symbol == symbol {
                values[component] = 1.0;
            }
            PointBoundInput {
                symbol: input.symbol,
                slot: input.slot.clone(),
                values,
            }
        })
        .collect();
    Ok(PointEvaluation {
        time: evaluation.time,
        cell: evaluation.cell,
        coordinates: evaluation.coordinates.clone(),
        active,
        bound,
    })
}

/// System analogue of `RealizationPlan::accumulate_parameter_cotangents`: routes each frozen-
/// input cotangent of the bound parameter kernel to its exact destination. A passive basis-
/// sourced input scatters directly through its own field's basis; a constitutive closure's
/// cotangent is pushed back into the active cotangents by probing its trusted `direction`
/// closure with unit active perturbations (exact because that closure is contracted to return
/// the exact, hence linear and homogeneous, directional derivative of its value closure), and
/// -- on a composed plan -- into the bound inputs' seeds by probing it with unit bound
/// perturbations; a bound input's own cotangent (kernel-input path) seeds its producer output.
#[allow(clippy::too_many_arguments)]
fn accumulate_parameter_cotangents_system(
    fields: InstanceFields<'_>,
    integral: &IntegralOperatorFactorization,
    cell: usize,
    point: usize,
    geometry: &CellGeometry,
    affine: &AffineMap,
    reference_point: &[f64],
    scale: f64,
    rate_shift: f64,
    evaluation: &PointEvaluation,
    active_inputs: &[&QFunctionInput],
    bindings: &SystemInputBindings<'_>,
    parameter_cotangents: &BTreeMap<TensorInputId, Vec<f64>>,
    cotangents: &mut BTreeMap<TensorInputId, Vec<f64>>,
    local_outputs: &mut BTreeMap<SysVarId, Vec<f64>>,
    bound_seeds: &mut BoundSeeds,
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
            scatter_field_cotangent(
                fields,
                input.binding.symbol,
                geometry,
                affine,
                cell,
                point,
                reference_point,
                &derivative,
                grad,
                factor * scale,
                local_outputs,
            )?;
            continue;
        }
        let binding = match bindings.resolve(integral.integral_index, input)? {
            // A stored table is state-independent: no chain rule through it.
            SystemInputBinding::Stored(_) => continue,
            SystemInputBinding::Bound(bound) => {
                // The kernel-input path: the cotangent of the bound operand (unscaled: the
                // quadrature scale is applied once, when the producer chain scatters).
                add_bound_seed(bound_seeds, bound.bind, grad);
                continue;
            }
            SystemInputBinding::Constitutive(binding) => binding,
        };
        for probe_input in active_inputs {
            let count = component_count(&probe_input.shape)?;
            for component in 0..count {
                let mut probe = probe_direction_evaluation(
                    evaluation,
                    active_inputs,
                    probe_input.id,
                    component,
                )?;
                // An active probe holds every bound input's direction at zero.
                probe.bound = evaluation
                    .bound
                    .iter()
                    .map(|input| PointBoundInput {
                        symbol: input.symbol,
                        slot: input.slot.clone(),
                        values: vec![0.0; input.values.len()],
                    })
                    .collect();
                let response = binding.evaluate_direction(evaluation, &probe)?;
                if response.len() != grad.len() {
                    return Err(FinitumError::InvalidRealization(format!(
                        "constitutive input {input_id:?} direction returned {} components, \
                         expected {}",
                        response.len(),
                        grad.len()
                    )));
                }
                validate_finite("constitutive input direction probe", &response)?;
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
        // The provider-input path: the closure's chain rule through every bound input of its
        // instance, probed with unit bound perturbations.
        for (bind, value) in bindings.bound.entries() {
            let mut seed = vec![0.0; value.values.len()];
            for (component, seed_component) in seed.iter_mut().enumerate() {
                let probe = probe_bound_direction_evaluation(
                    evaluation,
                    active_inputs,
                    bind.consumer_symbol,
                    component,
                )?;
                let response = binding.evaluate_direction(evaluation, &probe)?;
                if response.len() != grad.len() {
                    return Err(FinitumError::InvalidRealization(format!(
                        "constitutive input {input_id:?} direction returned {} components, \
                         expected {}",
                        response.len(),
                        grad.len()
                    )));
                }
                validate_finite("constitutive input bound direction probe", &response)?;
                *seed_component = response
                    .iter()
                    .zip(grad.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            }
            if seed.iter().any(|component| *component != 0.0) {
                add_bound_seed(bound_seeds, value.bind, &seed);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn point_inputs_system(
    fields: InstanceFields<'_>,
    integral: &IntegralOperatorFactorization,
    cell: usize,
    point: usize,
    geometry: &CellGeometry,
    affine: &AffineMap,
    reference_point: &[f64],
    time: f64,
    local_state: &BTreeMap<SysVarId, Vec<f64>>,
    local_rate: &BTreeMap<SysVarId, Vec<f64>>,
    bindings: &SystemInputBindings<'_>,
) -> Result<(BTreeMap<TensorInputId, Vec<f64>>, PointEvaluation), FinitumError> {
    let mut inputs = BTreeMap::new();
    let mut active = Vec::new();
    for input in &integral.primal.inputs {
        if input.source != InputSourceRequirement::Basis {
            continue;
        }
        let (variable, field) = fields.field(input.binding.symbol).map_err(|_| {
            FinitumError::ArtifactMismatch(format!(
                "integral {} input {:?} references field {} which the system operator has not \
                 realized",
                integral.integral_index, input.id, input.binding.symbol
            ))
        })?;
        let dofs = if input.binding.evaluation.derivative == DerivativeEvaluation::TimeDerivative {
            &local_rate[&variable]
        } else {
            &local_state[&variable]
        };
        let values = evaluate_field_basis_input(
            field,
            geometry,
            affine,
            cell,
            point,
            reference_point,
            input,
            dofs,
        )?;
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
        coordinates: geometry.physical_point(reference_point),
        active,
        bound: bindings.bound.point_inputs(),
    };
    for input in &integral.primal.inputs {
        if input.source == InputSourceRequirement::Basis {
            continue;
        }
        let values = match bindings.resolve(integral.integral_index, input)? {
            SystemInputBinding::Stored(stored) => stored
                .point_values(cell, point, bindings.point_count)
                .to_vec(),
            SystemInputBinding::Constitutive(binding) => binding.evaluate_value(&evaluation)?,
            SystemInputBinding::Bound(bound) => bound.values.clone(),
        };
        if values.len() != component_count(&input.shape)? {
            return Err(FinitumError::InvalidRealization(format!(
                "constitutive input {:?} returned {} components, expected {}",
                input.id,
                values.len(),
                component_count(&input.shape)?
            )));
        }
        validate_finite("constitutive input", &values)?;
        inputs.insert(input.id, values);
    }
    Ok((inputs, evaluation))
}

#[allow(clippy::too_many_arguments)]
fn point_directions_system(
    fields: InstanceFields<'_>,
    integral: &IntegralOperatorFactorization,
    cell: usize,
    point: usize,
    geometry: &CellGeometry,
    affine: &AffineMap,
    reference_point: &[f64],
    time: f64,
    local_state_direction: &BTreeMap<SysVarId, Vec<f64>>,
    local_rate_direction: &BTreeMap<SysVarId, Vec<f64>>,
    bindings: &SystemInputBindings<'_>,
    evaluation: &PointEvaluation,
) -> Result<BTreeMap<TensorInputId, Vec<f64>>, FinitumError> {
    let mut directions = BTreeMap::new();
    let mut active = Vec::new();
    for input in &integral.primal.inputs {
        if input.source != InputSourceRequirement::Basis {
            continue;
        }
        let (variable, field) = fields.field(input.binding.symbol).map_err(|_| {
            FinitumError::ArtifactMismatch(format!(
                "integral {} input {:?} references field {} which the system operator has not \
                 realized",
                integral.integral_index, input.id, input.binding.symbol
            ))
        })?;
        let dofs = if input.binding.evaluation.derivative == DerivativeEvaluation::TimeDerivative {
            &local_rate_direction[&variable]
        } else {
            &local_state_direction[&variable]
        };
        let values = evaluate_field_basis_input(
            field,
            geometry,
            affine,
            cell,
            point,
            reference_point,
            input,
            dofs,
        )?;
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
        bound: bindings.bound.point_directions(),
    };
    for input in &integral.primal.inputs {
        if input.source == InputSourceRequirement::Basis {
            continue;
        }
        let values = match bindings.resolve(integral.integral_index, input)? {
            SystemInputBinding::Stored(stored) => vec![0.0; stored.component_count()],
            SystemInputBinding::Constitutive(binding) => {
                binding.evaluate_direction(evaluation, &direction_evaluation)?
            }
            SystemInputBinding::Bound(bound) => bound
                .direction
                .clone()
                .unwrap_or_else(|| vec![0.0; bound.values.len()]),
        };
        if values.len() != component_count(&input.shape)? {
            return Err(FinitumError::InvalidRealization(format!(
                "constitutive input direction {:?} returned {} components, expected {}",
                input.id,
                values.len(),
                component_count(&input.shape)?
            )));
        }
        validate_finite("constitutive input direction", &values)?;
        directions.insert(input.id, values);
    }
    Ok(directions)
}

/// Runs one kernel-input bind composition at a quadrature point (W8 F-MI, the Malleus
/// contract for `BoundChain::Composed`): stage 0 is the producer output's primal kernel over
/// `producer_inputs`, stage 1 the consumer's primal kernel over `consumer_inputs`; the shared
/// buffer carries the output into the consumer's bound operand(s). Returns the consumer's
/// primal output at the point.
fn execute_bind_composition(
    composition: &BoundComposition,
    producer: &BoundBundle,
    producer_inputs: &BTreeMap<TensorInputId, Vec<f64>>,
    consumer: &BoundBundle,
    consumer_inputs: &BTreeMap<TensorInputId, Vec<f64>>,
) -> Result<Vec<f64>, FinitumError> {
    let stages = [(producer, producer_inputs), (consumer, consumer_inputs)];
    let mut buffers: Vec<(malleus::StageOperand, Vec<f64>)> = Vec::new();
    for (stage_index, (bound, inputs)) in stages.iter().enumerate() {
        let kernel = &composition.composition.composition.stages[stage_index];
        let by_operand = bound
            .bundle
            .primal_inputs
            .iter()
            .map(|binding| (binding.operand, binding.input))
            .collect::<BTreeMap<_, _>>();
        for (index, operand) in kernel.operands.iter().enumerate() {
            let target = malleus::StageOperand::new(stage_index, malleus::OperandId::new(index));
            if composition.executable.shared_buffer_of(target).is_some() {
                continue;
            }
            let mut values = vec![0.0; operand.region.offset + operand.region.length];
            if let Some(input) = by_operand.get(&target.operand) {
                let point_values = inputs.get(input).ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "bundle input {input:?} is absent from the gathered point inputs"
                    ))
                })?;
                let count = component_count(&operand.shape)?;
                if point_values.len() != count {
                    return Err(FinitumError::InvalidRealization(format!(
                        "composition stage {stage_index} operand {index} received {} values, \
                         expected {count}",
                        point_values.len()
                    )));
                }
                let start = operand.region.offset;
                values[start..start + count].copy_from_slice(point_values);
            }
            buffers.push((target, values));
        }
    }
    let mut bindings = buffers
        .iter_mut()
        .map(|(target, values)| malleus::CompositionBinding::operand(*target, values))
        .collect::<Vec<_>>();
    malleus::Interpreter::run_composition(&composition.executable, &mut bindings)
        .map_err(|error| FinitumError::KernelExecution(error.to_string()))?;
    drop(bindings);
    let output = malleus::StageOperand::new(1, consumer.bundle.primal_output);
    let definition =
        &composition.composition.composition.stages[1].operands[output.operand.index()];
    let count = component_count(&definition.shape)?;
    let start = definition.region.offset;
    buffers
        .iter()
        .find(|(target, _)| *target == output)
        .map(|(_, values)| values[start..start + count].to_vec())
        .ok_or_else(|| {
            FinitumError::InvalidRealization(
                "the composition's consumer output operand was not bound".into(),
            )
        })
}

/// One essential (Dirichlet) constraint requirement for a single field within a
/// [`SystemOperator`]'s [`BlockLayout`].
#[derive(Clone, Debug)]
pub struct SystemEssentialConstraintRequirement {
    pub field: SymbolId,
    pub requirement: EssentialConstraintRequirement,
    pub value: FieldSource,
}

/// Region-tag-driven multi-field essential constraints (mission item 3): resolves several
/// fields' boundary DOFs from a [`TaggedMesh`]/[`RegionMap`] into `operator`'s [`BlockLayout`]
/// [`ConstraintSet`], reusing the same [`crate::FacetTopology`]/[`RegionMap`] region-tag
/// resolution [`crate::essential_constraints_from`] uses and composing it with
/// [`crate::essential_constraints_for_variables`] for the elimination-facing `ConstraintSet`
/// construction -- rather than duplicating *that* DOF-indexing machinery.
///
/// This does not delegate to [`crate::essential_constraints_from`] itself, because that function is
/// documented and typed as vertex-major only (P1 Lagrange): a P2 (Taylor-Hood velocity) field's
/// boundary also includes the *edge* node on every tagged facet, which a vertex-only walk misses
/// entirely (leaving a P2 boundary only partially constrained -- wrong, not merely incomplete).
/// Node order is auto-detected from `operator.dof_map(field)`'s node count against the mesh's
/// vertex count (P1: `dof_count() == vertex_count * components`; P2: `vertex_count + edge_count`
/// nodes, matching [`crate::quadratic_simplex_dof_map`]/[`crate::quadratic_simplex_node_points`]'s
/// shared vertex-then-edge convention), so this one function serves either order.
/// [`FieldSource::Table`]/[`FieldSource::Kernel`] are refused typed (unsupported here; only
/// `Constant`/`Nodal`/`Sampled`/`Fallible` are admitted; `Fallible` values are evaluated at
/// `t = 0` here and at the caller's time by [`essential_constraints_from_system_at`]).
pub fn essential_constraints_from_system(
    operator: &SystemOperator,
    mesh: &TaggedMesh,
    region_map: &RegionMap,
    requirements: &[SystemEssentialConstraintRequirement],
) -> Result<ConstraintSet, FinitumError> {
    essential_constraints_from_system_at(operator, mesh, region_map, requirements, 0.0)
}

/// W8 lane F2: [`essential_constraints_from_system`] with every value evaluated at `time` (a
/// [`FieldSource::Fallible`] closure's time argument; `Constant` / `Nodal` / `Sampled` values do
/// not depend on it). This is a value snapshot, not a transient lifting: attach
/// [`ReducedSystemOperator::with_prescribed_values`] for runtime essential values and their
/// analytic rate contributions. Rebuilding this set alone omits prescribed mass-rate terms.
/// A `Fallible` refusal is
/// located at the node's coordinates (or the RT0 facet centroid) and `time`, without a cell,
/// and returned as [`FinitumError::InputEvaluation`].
pub fn essential_constraints_from_system_at(
    operator: &SystemOperator,
    mesh: &TaggedMesh,
    region_map: &RegionMap,
    requirements: &[SystemEssentialConstraintRequirement],
    time: f64,
) -> Result<ConstraintSet, FinitumError> {
    let layout = operator.layout();
    let keyed = requirements
        .iter()
        .map(|requirement| {
            let variable = layout
                .block(requirement.field)
                .map(|block| block.variable)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "essential constraint requirement names field {} which the system \
                         operator has not realized",
                        requirement.field
                    ))
                })?;
            Ok(SystemVariableEssentialConstraint {
                variable,
                requirement: requirement.requirement.clone(),
                value: requirement.value.clone(),
            })
        })
        .collect::<Result<Vec<_>, FinitumError>>()?;
    let region_maps = operator
        .system_ids()
        .instances()
        .iter()
        .map(|record| (record.instance, region_map))
        .collect::<Vec<_>>();
    essential_constraints_from_system_by_variable_at(operator, mesh, &region_maps, &keyed, time)
}

/// One essential (Dirichlet) constraint requirement keyed by system variable (W8 lane F-MI):
/// the per-model requirement of the variable's instance (its `region` is that instance's
/// per-model `RegionId`) and the value source.
#[derive(Clone, Debug)]
pub struct SystemVariableEssentialConstraint {
    pub variable: SysVarId,
    pub requirement: EssentialConstraintRequirement,
    pub value: FieldSource,
}

/// Prescribed value and analytic rate sources for one system essential requirement. Both
/// sources use the same component layout and physical sampling point; unavailable analytic
/// rates must be a fallible source returning a typed error.
#[derive(Clone, Debug)]
pub struct SystemVariablePrescribedValue {
    pub variable: SysVarId,
    pub requirement: EssentialConstraintRequirement,
    pub value: FieldSource,
    pub rate: FieldSource,
    pub origin: InputOrigin,
}

/// Projects paired prescribed sources onto the owner's exact essential target selection.
/// P1/P2 scalar and vector node coordinates/components are owned here; callers do not infer
/// DOF conventions. Requirements sharing a target must agree on both value and rate at every
/// runtime evaluation. Include static requirements sharing targets with moving ones, using
/// a zero rate source, so a later conflict at a shared corner cannot be hidden. Passing all
/// essential requirements is sufficient. RT0 normal-flux motion is refused until its rate/orientation contract
/// is implemented. Kernel/Table sources use the same refusal boundary as system essential
/// sampling; compiled consumer callbacks can be supplied as `FieldSource::Fallible`.
pub fn prescribed_values_from_system_by_variable(
    operator: &SystemOperator,
    mesh: &TaggedMesh,
    region_maps: &[(InstanceId, &RegionMap)],
    requirements: &[SystemVariablePrescribedValue],
) -> Result<Vec<PrescribedEssentialValue>, FinitumError> {
    if &mesh.mesh != operator.plan().mesh() {
        return Err(FinitumError::InvalidRealization(
            "prescribed source mesh differs from the realized mesh".into(),
        ));
    }
    #[derive(Clone)]
    struct Contribution {
        value: FieldSource,
        rate: FieldSource,
        origin: InputOrigin,
        node: usize,
        component: usize,
        components: usize,
    }
    let mut targets = BTreeMap::<DofId, (Vec<f64>, Vec<Contribution>)>::new();
    for requirement in requirements {
        let field = operator
            .data
            .fields
            .get(&requirement.variable)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(
                    "prescribed source variable is not realized".into(),
                )
            })?;
        if matches!(field.kind, FieldKind::Hdiv0 { .. }) {
            return Err(FinitumError::UnsupportedRealization(
                "prescribed RT0 normal-flux motion is not implemented".into(),
            ));
        }
        let block = operator
            .layout()
            .block_by_variable(requirement.variable)
            .expect("realized field block");
        let components = block.component_count;
        for source in [&requirement.value, &requirement.rate] {
            if matches!(source, FieldSource::Kernel { .. } | FieldSource::Table(_)) {
                return Err(FinitumError::UnsupportedRealization(
                    "prescribed system sources require Constant/Nodal/Fallible sampling".into(),
                ));
            }
        }
        // The existing owner helper remains the authority on which region/edge/component
        // targets are essential. Zero data select topology without evaluating user sources.
        let selected = essential_constraints_from_system_by_variable_at(
            operator,
            mesh,
            region_maps,
            &[SystemVariableEssentialConstraint {
                variable: requirement.variable,
                requirement: requirement.requirement.clone(),
                value: FieldSource::constant(vec![0.0; components]),
            }],
            0.0,
        )?;
        let count = block.extent / components;
        let points = if count == mesh.mesh.vertices().len() {
            mesh.mesh.vertices().to_vec()
        } else {
            crate::quadratic_simplex_node_points(&mesh.mesh)
        };
        if count != points.len() {
            return Err(FinitumError::UnsupportedRealization(
                "prescribed motion requires a P1/P2 nodal field".into(),
            ));
        }
        for constraint in selected.constraints() {
            let local = constraint
                .target
                .0
                .checked_sub(block.offset)
                .filter(|local| *local < block.extent)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(
                        "essential target is outside the selected variable".into(),
                    )
                })?;
            let node = local / components;
            let entry = targets
                .entry(constraint.target)
                .or_insert_with(|| (points[node].clone(), Vec::new()));
            entry.1.push(Contribution {
                value: requirement.value.clone(),
                rate: requirement.rate.clone(),
                origin: requirement.origin.clone(),
                node,
                component: local % components,
                components,
            });
        }
    }
    fn sample(
        source: &FieldSource,
        point: &[f64],
        time: f64,
        node: usize,
        components: usize,
        origin: &InputOrigin,
    ) -> Result<Vec<f64>, InputEvaluationError> {
        let values = match source {
            FieldSource::Constant(values) => values.clone(),
            FieldSource::Nodal(values) => {
                let start = node * components;
                values
                    .get(start..start + components)
                    .ok_or_else(|| {
                        InputEvaluationError::new(
                            "PRESCRIBED_SOURCE_SHAPE",
                            origin.clone(),
                            "nodal source does not cover the essential node",
                        )
                    })?
                    .to_vec()
            }
            FieldSource::Sampled(callback) => callback(point),
            FieldSource::Fallible(callback) => callback(point, time)?,
            FieldSource::Kernel { .. } | FieldSource::Table(_) => {
                unreachable!("validated prescribed source")
            }
        };
        if values.len() != components || values.iter().any(|value| !value.is_finite()) {
            return Err(InputEvaluationError::new(
                "PRESCRIBED_SOURCE_SHAPE",
                origin.clone(),
                "prescribed source shape mismatch or nonfinite component",
            ));
        }
        Ok(values)
    }
    Ok(targets
        .into_iter()
        .map(|(target, (point, contributions))| {
            let origin = contributions[0].origin.clone();
            let sampling_point = point.clone();
            PrescribedEssentialValue::new(target, point, origin, move |time| {
                let mut result: Option<PrescribedValueAndRate> = None;
                for contribution in &contributions {
                    let value = sample(
                        &contribution.value,
                        &sampling_point,
                        time,
                        contribution.node,
                        contribution.components,
                        &contribution.origin,
                    )?[contribution.component];
                    let rate = sample(
                        &contribution.rate,
                        &sampling_point,
                        time,
                        contribution.node,
                        contribution.components,
                        &contribution.origin,
                    )?[contribution.component];
                    let next = PrescribedValueAndRate { value, rate };
                    if result.is_some_and(|old| old != next) {
                        return Err(InputEvaluationError::new(
                            "PRESCRIBED_SOURCE_CONFLICT",
                            contribution.origin.clone(),
                            "overlapping essential requirements disagree on value or analytic rate",
                        ));
                    }
                    result = Some(next);
                }
                Ok(result.expect("target has contributors"))
            })
        })
        .collect())
}

/// [`essential_constraints_from_system_by_variable_at`] at `t = 0`.
pub fn essential_constraints_from_system_by_variable(
    operator: &SystemOperator,
    mesh: &TaggedMesh,
    region_maps: &[(InstanceId, &RegionMap)],
    requirements: &[SystemVariableEssentialConstraint],
) -> Result<ConstraintSet, FinitumError> {
    essential_constraints_from_system_by_variable_at(operator, mesh, region_maps, requirements, 0.0)
}

/// W8 lane F-MI: [`essential_constraints_from_system_at`] over a composed (multi-instance)
/// layout -- every requirement names its field by [`SysVarId`], and its per-model region is
/// resolved through the [`RegionMap`] of the variable's instance (`region_maps`, one entry per
/// instance that has constrained fields; an instance without an entry is refused
/// `RealizationRegionUnmapped`). Constraints touch only the named variable's block. On a
/// one-instance plan this is exactly [`essential_constraints_from_system_at`].
pub fn essential_constraints_from_system_by_variable_at(
    operator: &SystemOperator,
    mesh: &TaggedMesh,
    region_maps: &[(InstanceId, &RegionMap)],
    requirements: &[SystemVariableEssentialConstraint],
    time: f64,
) -> Result<ConstraintSet, FinitumError> {
    if !time.is_finite() {
        return Err(FinitumError::InvalidRealization(
            "essential constraint evaluation time must be finite".into(),
        ));
    }
    if mesh.mesh.vertices().len() != operator.plan().mesh().vertices().len()
        || mesh.mesh.cells().len() != operator.plan().mesh().cells().len()
    {
        return Err(FinitumError::InvalidRealization(
            "tagged mesh does not match the system operator's own realized mesh".into(),
        ));
    }
    let layout = operator.layout();
    let vertex_count = mesh.mesh.vertices().len();
    let facets = crate::FacetTopology::from_mesh(&mesh.mesh)?;
    let facet_vertices = facets
        .facets()
        .iter()
        .map(|facet| (facet.id, facet.vertices.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let edges = crate::topology::mesh_edges(&mesh.mesh);
    let edge_index = edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let node_points_p1 = mesh.mesh.vertices();
    let node_points_p2 = crate::quadratic_simplex_node_points(&mesh.mesh);

    let mut values = Vec::new();
    for requirement in requirements {
        let dof_map = operator
            .dof_map_by_variable(requirement.variable)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "essential constraint requirement names variable {} which the system \
                     operator has not realized",
                    requirement.variable
                ))
            })?;
        let block = layout
            .block_by_variable(requirement.variable)
            .expect("a realized field's DOF map implies a layout block");
        let instance = operator
            .system_ids()
            .variable_origin(requirement.variable)
            .expect("a realized variable has an origin")
            .instance;
        let region_map = region_maps
            .iter()
            .find(|(candidate, _)| *candidate == instance)
            .map(|(_, region_map)| *region_map)
            .ok_or_else(|| {
                FinitumError::RealizationRegionUnmapped(format!(
                    "{:?} of {instance} (no region map for the instance)",
                    requirement.requirement.region
                ))
            })?;
        let components = block.component_count;
        if matches!(
            operator.data.fields[&requirement.variable].kind,
            FieldKind::Hdiv0 { .. }
        ) {
            rt0_essential_values(
                &facets,
                &mesh.mesh,
                mesh,
                region_map,
                requirement,
                block.symbol,
                time,
                &mut values,
            )?;
            continue;
        }
        let node_count = dof_map.dof_count() / components;
        let quadratic = node_count != vertex_count;
        let node_points = if quadratic {
            &node_points_p2
        } else {
            node_points_p1
        };

        let tags = region_map
            .tags(requirement.requirement.region)
            .filter(|tags| !tags.is_empty())
            .ok_or_else(|| {
                FinitumError::RealizationRegionUnmapped(format!(
                    "{:?}",
                    requirement.requirement.region
                ))
            })?;
        let mut nodes = BTreeSet::new();
        for tag in tags {
            let Some(facet_ids) = mesh.tags.facet_regions.get(tag) else {
                continue;
            };
            for facet_id in facet_ids {
                let Some(vertices) = facet_vertices.get(facet_id) else {
                    continue;
                };
                for vertex in *vertices {
                    nodes.insert(vertex.0);
                }
                if quadratic {
                    for (left_index, left) in vertices.iter().enumerate() {
                        for right in &vertices[left_index + 1..] {
                            let key = if left.0 < right.0 {
                                vec![left.0, right.0]
                            } else {
                                vec![right.0, left.0]
                            };
                            if let Some(&edge) = edge_index.get(&key) {
                                nodes.insert(vertex_count + edge);
                            }
                        }
                    }
                }
            }
        }
        for node in nodes {
            let coordinates = &node_points[node];
            let evaluated = match &requirement.value {
                FieldSource::Constant(constant) => constant.clone(),
                FieldSource::Nodal(nodal) => {
                    let start = node * components;
                    if nodal.len() < start + components {
                        return Err(FinitumError::InvalidRealization(
                            "nodal field source does not cover every tagged node".into(),
                        ));
                    }
                    nodal[start..start + components].to_vec()
                }
                FieldSource::Sampled(sampler) => sampler(coordinates),
                FieldSource::Fallible(sampler) => {
                    sampler(coordinates, time).map_err(|failure| {
                        FinitumError::from(failure.at(None, coordinates, Some(time)))
                    })?
                }
                FieldSource::Table(_) | FieldSource::Kernel { .. } => {
                    return Err(FinitumError::UnsupportedRealization(
                        "essential_constraints_from_system admits Constant/Nodal/Sampled/Fallible field \
                         sources only"
                            .into(),
                    ));
                }
            };
            if evaluated.len() != components {
                return Err(FinitumError::InvalidRealization(format!(
                    "essential value has {} components, expected {components}",
                    evaluated.len()
                )));
            }
            for (component, value) in evaluated.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(FinitumError::InvalidRealization(
                        "essential value is not finite".into(),
                    ));
                }
                values.push(BlockVariableEssentialValue {
                    variable: requirement.variable,
                    entity: node,
                    component,
                    value,
                });
            }
        }
    }
    essential_constraints_for_variables(layout, values)
}

/// GX-CONTRACTS C11.22: essential normal-trace data on an RT0 (`Hdiv(order=0)`) field. The
/// reference basis `phi_i = x - p_i` (`rt0_reference_basis`) carries the outward flux
/// `integral_{F_i} phi_i . n_i = d |K_ref| = 1 / (d - 1)!` through its own facet (`1` on the
/// triangle, `1/2` on the tetrahedron), preserved by the contravariant Piola map, so a global
/// DOF `c_F` on facet `F` is `(d - 1)!` times the flux through `F` in the canonical
/// (sorted-vertex) facet orientation; the boundary cell's `FacetIncidence::orientation`
/// relates that to the cell's outward normal, which on an exterior facet is the domain's
/// outward normal. A datum `flux . n = g` (scalar, the `FieldSource` evaluated at the facet
/// centroid) therefore fixes the DOF to `orientation * g * |F| * (d - 1)!` -- verified against
/// the divergence theorem through the `mass_balance` block action in
/// `tests/w7_rt0_essential.rs`. Interior facets in the region and nodal sources are refused
/// typed (RT0 has no nodes; an interior facet has no outward normal).
#[allow(clippy::too_many_arguments)]
fn rt0_essential_values(
    facets: &crate::FacetTopology,
    mesh: &Mesh,
    tagged: &TaggedMesh,
    region_map: &RegionMap,
    requirement: &SystemVariableEssentialConstraint,
    field: SymbolId,
    time: f64,
    values: &mut Vec<BlockVariableEssentialValue>,
) -> Result<(), FinitumError> {
    let tags = region_map
        .tags(requirement.requirement.region)
        .filter(|tags| !tags.is_empty())
        .ok_or_else(|| {
            FinitumError::RealizationRegionUnmapped(format!("{:?}", requirement.requirement.region))
        })?;
    let dimension = mesh.dimension();
    let mut facet_ids = BTreeSet::new();
    for tag in tags {
        if let Some(ids) = tagged.tags.facet_regions.get(tag) {
            facet_ids.extend(ids.iter().copied());
        }
    }
    for facet_id in facet_ids {
        let facet = facets.facets().get(facet_id.0).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("facet {} does not exist", facet_id.0))
        })?;
        if !facet.is_exterior() {
            return Err(FinitumError::UnsupportedRealization(format!(
                "essential normal-trace data on RT0 field {} names interior facet {}; only \
                 exterior facets carry an outward normal",
                field, facet_id.0
            )));
        }
        let vertices = facet
            .vertices
            .iter()
            .map(|vertex| &mesh.vertices()[vertex.0])
            .collect::<Vec<_>>();
        let centroid = (0..dimension)
            .map(|axis| {
                vertices.iter().map(|vertex| vertex[axis]).sum::<f64>() / vertices.len() as f64
            })
            .collect::<Vec<_>>();
        let measure = match dimension {
            2 => {
                let t = [
                    vertices[1][0] - vertices[0][0],
                    vertices[1][1] - vertices[0][1],
                ];
                (t[0] * t[0] + t[1] * t[1]).sqrt()
            }
            3 => {
                let t1 = (0..3)
                    .map(|axis| vertices[1][axis] - vertices[0][axis])
                    .collect::<Vec<_>>();
                let t2 = (0..3)
                    .map(|axis| vertices[2][axis] - vertices[0][axis])
                    .collect::<Vec<_>>();
                let cross = [
                    t1[1] * t2[2] - t1[2] * t2[1],
                    t1[2] * t2[0] - t1[0] * t2[2],
                    t1[0] * t2[1] - t1[1] * t2[0],
                ];
                0.5 * cross.iter().map(|value| value * value).sum::<f64>().sqrt()
            }
            other => {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "RT0 essential normal-trace data is realized for mesh dimension 2 or 3, \
                     got {other}"
                )));
            }
        };
        let datum = match &requirement.value {
            FieldSource::Constant(constant) => constant.clone(),
            FieldSource::Sampled(sampler) => sampler(&centroid),
            FieldSource::Fallible(sampler) => sampler(&centroid, time)
                .map_err(|failure| FinitumError::from(failure.at(None, &centroid, Some(time))))?,
            FieldSource::Nodal(_) => {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "RT0 field {} has no nodes; essential normal-trace data must be a \
                     Constant or Sampled scalar",
                    field
                )));
            }
            FieldSource::Table(_) | FieldSource::Kernel { .. } => {
                return Err(FinitumError::UnsupportedRealization(
                    "essential_constraints_from_system admits Constant/Nodal/Sampled/Fallible field \
                     sources only"
                        .into(),
                ));
            }
        };
        let [g] = datum[..] else {
            return Err(FinitumError::InvalidRealization(format!(
                "RT0 essential normal-trace datum must be one scalar (`flux . n`), got {} \
                 components",
                datum.len()
            )));
        };
        if !g.is_finite() {
            return Err(FinitumError::InvalidRealization(
                "essential value is not finite".into(),
            ));
        }
        let orientation = f64::from(facet.minus().orientation);
        let basis_flux = 1.0 / (1..dimension).product::<usize>() as f64;
        values.push(BlockVariableEssentialValue {
            variable: requirement.variable,
            entity: facet_id.0,
            component: 0,
            value: orientation * g * measure / basis_flux,
        });
    }
    Ok(())
}

#[cfg(test)]
mod rt0_tests {
    use super::*;
    use crate::{Cell, VertexId};

    /// Two triangles sharing the diagonal of the unit square: cell A = `[0,1,2]`
    /// `(0,0),(1,0),(0,1)` (whose reference map is the identity: `physical_point(ref) == ref`),
    /// cell B = `[1,3,2]` `(1,0),(1,1),(0,1)`. The shared facet is the diagonal `{1,2}`
    /// (physical midpoint `(0.5,0.5)`), which is cell A's local facet 0 (omits local vertex 0)
    /// and cell B's local facet 1 (omits local vertex 1) -- reference coordinates for that
    /// physical midpoint are hand-derived in each cell's own affine map (see the module's own
    /// lane report for the derivation) rather than computed by inverting `AffineMap`, which
    /// exposes no inverse-point method.
    fn two_triangle_mesh() -> Mesh {
        Mesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
            vec![
                Cell {
                    vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
                },
                Cell {
                    vertices: vec![VertexId(1), VertexId(3), VertexId(2)],
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn rt0_normal_flux_is_continuous_across_a_shared_interior_facet_using_a_single_fixed_normal() {
        let mesh = two_triangle_mesh();
        let facets = FacetTopology::from_mesh(&mesh).unwrap();
        let compatible = CompatibleDofMaps::simplex(&mesh, &facets).unwrap();
        let shared = facets.interior().next().expect("one shared interior facet");
        assert_eq!(
            shared.vertices,
            vec![crate::VertexId(1), crate::VertexId(2)]
        );

        let coefficient = 3.7_f64;
        let mut global_state = vec![0.0; compatible.hdiv_dof_count];
        global_state[shared.id.0] = coefficient;

        // Cell A: physical_point(ref) == ref (identity jacobian, origin (0,0)); the shared
        // facet's physical midpoint (0.5, 0.5) is therefore its own reference coordinate too.
        let affine_a = AffineMap::from_cell(&mesh, CellId(0)).unwrap();
        let restriction_a = &compatible.hdiv[0];
        let local_state_a: Vec<f64> = restriction_a
            .dofs
            .iter()
            .map(|dof| global_state[dof.0])
            .collect();
        let reference_point_a = [0.5, 0.5];
        let value_a = evaluate_rt0_value(
            &affine_a,
            &restriction_a.orientations,
            &reference_point_a,
            &local_state_a,
        )
        .unwrap();

        // Cell B: origin (1,0), jacobian columns (0,1) and (-1,1) (local vertices 1,3,2 minus
        // local vertex 1); solving physical_point(ref) == (0.5, 0.5) gives ref = (0.0, 0.5).
        let affine_b = AffineMap::from_cell(&mesh, CellId(1)).unwrap();
        let restriction_b = &compatible.hdiv[1];
        let local_state_b: Vec<f64> = restriction_b
            .dofs
            .iter()
            .map(|dof| global_state[dof.0])
            .collect();
        let reference_point_b = [0.0, 0.5];
        let value_b = evaluate_rt0_value(
            &affine_b,
            &restriction_b.orientations,
            &reference_point_b,
            &local_state_b,
        )
        .unwrap();

        // Sanity: `physical_point` really does map both reference points to the shared
        // midpoint, confirming the hand-derived reference coordinates above.
        assert!(
            affine_a
                .physical_point(&reference_point_a)
                .unwrap()
                .iter()
                .zip([0.5, 0.5])
                .all(|(actual, expected): (&f64, f64)| (actual - expected).abs() < 1e-13)
        );
        assert!(
            affine_b
                .physical_point(&reference_point_b)
                .unwrap()
                .iter()
                .zip([0.5, 0.5])
                .all(|(actual, expected): (&f64, f64)| (actual - expected).abs() < 1e-13)
        );

        // A single, fixed physical normal direction (cell A's own outward normal at the shared
        // facet, computed independently of `evaluate_rt0_value`): the diagonal from (1,0) to
        // (0,1) has tangent (-1,1); an outward-from-A normal pointing away from A's own
        // remaining vertex (0,0) is (1,1)/sqrt(2).
        let normal = [
            1.0 / std::f64::consts::SQRT_2,
            1.0 / std::f64::consts::SQRT_2,
        ];
        let flux_a: f64 = value_a.iter().zip(normal).map(|(v, n)| v * n).sum();
        let flux_b: f64 = value_b.iter().zip(normal).map(|(v, n)| v * n).sum();
        assert!(
            (flux_a - flux_b).abs() < 1e-12,
            "RT0 normal flux is not continuous across the shared facet: {flux_a} != {flux_b}"
        );

        // The magnitude itself matches the defining flux property: physical flux through the
        // owning facet equals exactly `orientation * coefficient` (Piola preserves reference
        // flux exactly), spread over the facet's own physical length (`sqrt(2)` here).
        let facet_length = std::f64::consts::SQRT_2;
        let local_index_a = restriction_a
            .dofs
            .iter()
            .position(|dof| dof.0 == shared.id.0)
            .unwrap();
        let expected_flux =
            f64::from(restriction_a.orientations[local_index_a]) * coefficient / facet_length;
        assert!(
            (flux_a - expected_flux).abs() < 1e-12,
            "flux magnitude {flux_a} does not match the closed-form {expected_flux}"
        );
    }

    #[test]
    fn rt0_value_and_divergence_adjoints_are_the_exact_algebraic_transpose_of_their_evaluators() {
        let mesh = two_triangle_mesh();
        let facets = FacetTopology::from_mesh(&mesh).unwrap();
        let compatible = CompatibleDofMaps::simplex(&mesh, &facets).unwrap();
        let affine = AffineMap::from_cell(&mesh, CellId(0)).unwrap();
        let restriction = &compatible.hdiv[0];
        let reference_point = [0.3, 0.2];

        // Value adjoint: for random `direction` (a perturbation of the local coefficients) and
        // random `point_output` (a cotangent), `dot(evaluate(direction), point_output)` must
        // equal `dot(direction, adjoint(point_output))` -- the standard linear-map/adjoint
        // dot-product identity, checked directly (not merely asserted from the derivation).
        let direction = [0.6, -1.2, 2.5];
        let point_output = [0.9, -0.4];
        let value = evaluate_rt0_value(
            &affine,
            &restriction.orientations,
            &reference_point,
            &direction,
        )
        .unwrap();
        let lhs: f64 = value.iter().zip(point_output).map(|(v, p)| v * p).sum();
        let mut adjoint = vec![0.0; 3];
        apply_rt0_value_adjoint(
            &affine,
            &restriction.orientations,
            &reference_point,
            &point_output,
            1.0,
            &mut adjoint,
        )
        .unwrap();
        let rhs: f64 = direction.iter().zip(&adjoint).map(|(d, a)| d * a).sum();
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "value adjoint mismatch: {lhs} != {rhs}"
        );

        // Divergence adjoint: same identity, scalar point output.
        let divergence_point_output = -2.3_f64;
        let divergence_value =
            evaluate_rt0_divergence(&affine, &restriction.orientations, &direction).unwrap();
        let divergence_lhs = divergence_value * divergence_point_output;
        let mut divergence_adjoint = vec![0.0; 3];
        apply_rt0_divergence_adjoint(
            &affine,
            &restriction.orientations,
            divergence_point_output,
            1.0,
            &mut divergence_adjoint,
        )
        .unwrap();
        let divergence_rhs: f64 = direction
            .iter()
            .zip(&divergence_adjoint)
            .map(|(d, a)| d * a)
            .sum();
        assert!(
            (divergence_lhs - divergence_rhs).abs() < 1e-12,
            "divergence adjoint mismatch: {divergence_lhs} != {divergence_rhs}"
        );
    }

    #[test]
    fn rt0_normal_trace_adjoint_matches_the_orientation_times_output_closed_form() {
        let orientations = [1_i8, -1, 1];
        let mut local_output = vec![0.0; 3];
        apply_rt0_normal_trace_adjoint(&orientations, 1, &[2.5], &mut local_output).unwrap();
        assert_eq!(local_output, vec![0.0, -2.5, 0.0]);
        assert!(
            apply_rt0_normal_trace_adjoint(&orientations, 1, &[1.0, 2.0], &mut [0.0; 3]).is_err()
        );
        assert!(apply_rt0_normal_trace_adjoint(&orientations, 5, &[1.0], &mut [0.0; 3]).is_err());
    }
}
