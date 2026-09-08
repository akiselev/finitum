use crate::CellBatchLayout;
use crate::InputEvaluationError;
use crate::element::{
    barycenter_quadrature, rt0_basis_count, rt0_reference_basis, simplex_basis,
    simplex_basis_count, simplex_quadrature,
};
use crate::mesh::CellId;
use crate::mixed::{
    BlockEssentialValue, BlockNullspaceCandidate, essential_constraints_for_blocks,
    solver_block_layout,
};
use crate::optimized::ElementAssemblyOperator;
use crate::profile::{
    RegionMap, TaggedMesh, evaluate_kernel_partial, evaluate_kernel_value, evaluate_table_slope,
    evaluate_table_value, named_coordinate_inputs,
};
use crate::realization::{
    BoundBundle, CapabilityElement, CellGeometry, ConstraintKind, DerivativeProduct,
    DistributedCoefficient, ExternalInput, FacetGeometry, PointActiveInput, PointEvaluation,
    RealizationCapability, RealizationExternalInput, RealizationReceipt, RepresentationKind,
    active_probe_inputs, apply_basis_adjoint, bind_kernels, build_capability, component_count,
    evaluate_basis_input, execute_jvp_values, execute_parameter_jvp_values, execute_primal_values,
    execute_vjp_values, gather_test_adjoint, locate_failure, point_parameter_cotangents,
    probe_direction_evaluation, transpose_scatter_shape, validate_finite,
};
use crate::space::{
    DofMap, ElementRestriction, cell_constant_dof_map, quadratic_simplex_dof_map,
    vector_nodal_dof_map,
};
use crate::system_ids::{SysResId, SysVarId, SystemIdMap};
use crate::{
    AffineMap, BlockLayout, CompatibleDofMaps, ConstraintSet, ExactSequence, FacetId,
    FacetTopology, FieldBlock, FieldSource, FinitumError, Mesh, PreparedElement, QuadraturePoint,
};
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

/// Digest-bound concrete ownership plan for an FC8 mixed operator system.
#[derive(Clone, Debug)]
pub struct SystemRealizationPlan {
    system: Arc<OperatorSystem>,
    mesh: Mesh,
    layout: BlockLayout,
    facets: FacetTopology,
    compatible_dofs: Option<CompatibleDofMaps>,
    exact_sequence: Option<ExactSequence>,
    /// SC-W1: the system-level ids of this (one-instance) realization group and their
    /// per-model origins; the layout's blocks are keyed by the same `SysVarId`s.
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
        Ok(Self {
            system: Arc::new(system),
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

    pub fn system(&self) -> &OperatorSystem {
        &self.system
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
struct FieldElement {
    kind: FieldKind,
    dofs: DofMap,
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
fn build_field_elements(
    system: &OperatorSystem,
    mesh: &Mesh,
    layout: &BlockLayout,
    quadrature: &[QuadraturePoint],
    compatible: Option<&CompatibleDofMaps>,
) -> Result<BTreeMap<SymbolId, FieldElement>, FinitumError> {
    let mut fields = BTreeMap::new();
    for &symbol in &system.field_order {
        let mut found: Option<&scientia::ElementRequirement> = None;
        for block in &system.blocks {
            for requirement in &block.requirements.elements {
                if requirement.symbol != symbol {
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
        let block = layout.block(symbol).ok_or_else(|| {
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
                        "RT0 compatible DOF map has a different cell count than the mesh".into(),
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
        fields.insert(symbol, field);
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
fn evaluate_field_basis_input(
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
fn apply_field_basis_adjoint(
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
    pub equation: String,
    pub integral_index: usize,
    pub input: TensorInputId,
    component_count: usize,
    identity: String,
    value: Arc<SystemPointValueEvaluator>,
    direction: Arc<SystemPointDirectionEvaluator>,
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

/// One resolved non-basis input binding of a system operator.
enum SystemInputBinding<'a> {
    Stored(&'a ExternalInput),
    Constitutive(&'a SystemConstitutiveInput),
}

/// The non-basis input bindings every point evaluation of a system operator resolves against:
/// the closure-based constitutive inputs and the stored tables, keyed by
/// `(block index, integral index, input)`, plus the shared quadrature's point count the stored
/// tables are indexed with.
struct SystemInputBindings<'a> {
    constitutive: &'a BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
    stored: &'a BTreeMap<(usize, usize, TensorInputId), ExternalInput>,
    point_count: usize,
}

impl SystemInputBindings<'_> {
    fn resolve(
        &self,
        key: (usize, usize, TensorInputId),
    ) -> Result<SystemInputBinding<'_>, FinitumError> {
        if let Some(stored) = self.stored.get(&key) {
            return Ok(SystemInputBinding::Stored(stored));
        }
        self.constitutive
            .get(&key)
            .map(SystemInputBinding::Constitutive)
            .ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "integral {} input {:?} has no bound constitutive or stored input",
                    key.1, key.2
                ))
            })
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
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        plan_digest: &'a Digest,
        constitutive: Vec<ConstitutiveIdentity<'a>>,
        stored: Vec<StoredIdentity<'a>>,
        equation_sign: &'a BTreeMap<usize, f64>,
        facet_regions: BTreeMap<u32, Vec<usize>>,
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
    };
    let bytes =
        serde_json::to_vec(&payload).expect("system operator digest payload is serializable");
    Digest {
        algorithm: "blake3".into(),
        hex: blake3::hash(&bytes).to_hex().to_string(),
    }
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
        for block in &self.system.blocks {
            for integral in &block.factorization.integrals {
                match &integral.measure {
                    SemanticMeasure::Cell { .. } => {
                        for output in &integral.primal.outputs {
                            if output.binding.evaluation.site != EvaluationSite::Cell {
                                return Err(FinitumError::UnsupportedRealization(format!(
                                    "system realization realizes cell evaluation sites only; \
                                     equation `{}` integral {} has an output at site {:?}",
                                    block.equation,
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
                                block.equation,
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
                                    block.equation, integral.integral_index
                                )));
                            }
                        }
                    }
                    other => {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "system realization admits SemanticMeasure::Cell integrals, or a \
                             narrowly-scoped Hdiv(order=0) SemanticMeasure::ExteriorFacet normal \
                             trace, only; equation `{}` integral {} has measure {other:?}",
                            block.equation, integral.integral_index
                        )));
                    }
                }
            }
        }
        let quadrature = self.quadrature()?;
        let fields = build_field_elements(
            &self.system,
            &self.mesh,
            &self.layout,
            &quadrature,
            self.compatible_dofs.as_ref(),
        )?;
        for block in &self.system.blocks {
            let has_exterior_facet =
                block.factorization.integrals.iter().any(|integral| {
                    matches!(integral.measure, SemanticMeasure::ExteriorFacet { .. })
                });
            if has_exterior_facet {
                let row_field = fields.get(&block.row).ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "equation `{}` row field {} was not realized",
                        block.equation, block.row
                    ))
                })?;
                if !matches!(row_field.kind, FieldKind::Hdiv0 { .. }) {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "equation `{}` has an exterior-facet integral but row field {} is not \
                         Hdiv(order=0)",
                        block.equation, block.row
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
        for (index, block) in self.system.blocks.iter().enumerate() {
            bindings.insert(
                index,
                bind_kernels(&block.factorization, block.kernels.clone())?,
            );
        }
        let mut constitutive_by_key = BTreeMap::new();
        for input in constitutive {
            let block_index = self
                .system
                .blocks
                .iter()
                .position(|block| block.equation == input.equation)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "constitutive input names equation `{}` which is absent from the system",
                        input.equation
                    ))
                })?;
            let key = (block_index, input.integral_index, input.input);
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
            let origin = self
                .system_ids
                .residual_origin(table.residual)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "stored input names residual {} which the system does not carry",
                        table.residual
                    ))
                })?;
            let block_index = self
                .system
                .blocks
                .iter()
                .position(|block| block.equation == origin.equation)
                .ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "residual {} names equation `{}` which the system does not carry",
                        table.residual, origin.equation
                    ))
                })?;
            let block = &self.system.blocks[block_index];
            let integral = block
                .factorization
                .integrals
                .iter()
                .find(|integral| integral.integral_index == table.input.integral_index)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "stored input names absent integral {} of equation `{}`",
                        table.input.integral_index, block.equation
                    ))
                })?;
            if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "stored system inputs are realized on cell integrals only; equation `{}` \
                     integral {} has measure {:?}",
                    block.equation, integral.integral_index, integral.measure
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
                        table.input.input, block.equation, integral.integral_index
                    ))
                })?;
            if input.source == InputSourceRequirement::Basis {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{}` integral {} input {:?} is a basis input, not an external one",
                    block.equation, integral.integral_index, input.id
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
                    block.equation,
                    integral.integral_index,
                    input.id,
                    table.input.component_count(),
                    table.input.values().len(),
                    self.mesh.cells().len(),
                    quadrature.len()
                )));
            }
            let key = (block_index, integral.integral_index, input.id);
            if constitutive_by_key.contains_key(&key) {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{}` integral {} input {:?} is bound both as a constitutive \
                     closure and as a stored table",
                    block.equation, integral.integral_index, input.id
                )));
            }
            if stored_by_key.insert(key, table.input).is_some() {
                return Err(FinitumError::InvalidRealization(format!(
                    "stored input for equation `{}` integral {} input {:?} is bound more than \
                     once",
                    block.equation, integral.integral_index, input.id
                )));
            }
        }
        for (block_index, block) in self.system.blocks.iter().enumerate() {
            for integral in &block.factorization.integrals {
                for input in &integral.primal.inputs {
                    if input.source == InputSourceRequirement::Basis {
                        continue;
                    }
                    let key = (block_index, integral.integral_index, input.id);
                    if !constitutive_by_key.contains_key(&key) && !stored_by_key.contains_key(&key)
                    {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "equation `{}` integral {} input {:?} requires a caller-supplied \
                             SystemConstitutiveInput closure or a stored SystemExternalInput \
                             table (regional external tensor tables remain future work)",
                            block.equation, integral.integral_index, input.id
                        )));
                    }
                }
            }
        }
        let mut equation_sign_by_block = BTreeMap::new();
        for (equation, sign) in &equation_sign {
            let block_index = self
                .system
                .blocks
                .iter()
                .position(|block| &block.equation == equation)
                .ok_or_else(|| {
                    FinitumError::InvalidRealization(format!(
                        "equation sign names equation `{equation}` which is absent from the \
                         system"
                    ))
                })?;
            if *sign != 1.0 && *sign != -1.0 {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{equation}` sign must be exactly 1.0 or -1.0 (a solution-\
                     preserving orientation correction only), got {sign}"
                )));
            }
            equation_sign_by_block.insert(block_index, *sign);
        }
        let structure = derive_operator_structure_for_system(&self.system, None)
            .map_err(|error| FinitumError::ArtifactMismatch(error.to_string()))?;
        validate_structure_matches_system(&self.system, &structure)?;
        let nullspace_candidates = derive_nullspace_candidates(&structure)?;
        let solver_layout = solver_block_layout(&self.layout)?;
        let digest = system_operator_digest(
            self,
            &constitutive_by_key,
            &stored_by_key,
            &equation_sign_by_block,
            &facet_regions,
        );
        Ok(SystemOperator {
            data: Arc::new(SystemOperatorData {
                plan: self.clone(),
                fields,
                quadrature,
                bindings,
                constitutive: constitutive_by_key,
                stored: stored_by_key,
                equation_sign: equation_sign_by_block,
                facet_regions,
                facet_geometries,
                structure,
                nullspace_candidates,
                solver_layout,
                digest,
                symmetry_proof: std::sync::OnceLock::new(),
            }),
        })
    }
}

#[derive(Debug)]
struct SystemOperatorData {
    plan: SystemRealizationPlan,
    fields: BTreeMap<SymbolId, FieldElement>,
    /// Shared quadrature table every field's basis is tabulated at (or, for an RT0 field,
    /// evaluated at directly -- see [`build_field_elements`]'s doc comment); the per-cell
    /// per-block integration loop indexes into this rather than any one field's own table, since
    /// an RT0 [`FieldElement`] carries no [`PreparedElement`] of its own.
    quadrature: Vec<QuadraturePoint>,
    bindings: BTreeMap<usize, BTreeMap<(usize, usize), BoundBundle>>,
    constitutive: BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
    /// Stored quadrature-point tables (SC-W1 system-path parity), keyed like `constitutive`.
    stored: BTreeMap<(usize, usize, TensorInputId), ExternalInput>,
    /// Per-block-index equation orientation (`1.0` or `-1.0`, absent means `1.0`); see
    /// `SystemRealizationPlan::bind_kernels`'s `equation_sign` parameter.
    equation_sign: BTreeMap<usize, f64>,
    /// Mission item 2's exterior-facet extension: caller-resolved region -> facet-id lists (see
    /// `SystemRealizationPlan::bind_kernels_with_facets`), and their precomputed
    /// `crate::realization::FacetGeometry` (cell + local facet index; reused from GX-C4
    /// unchanged).
    facet_regions: BTreeMap<RegionId, Vec<FacetId>>,
    facet_geometries: BTreeMap<FacetId, FacetGeometry>,
    structure: OperatorStructure,
    nullspace_candidates: Vec<BlockNullspaceCandidate>,
    solver_layout: methodus::BlockLayout,
    digest: Digest,
    /// Lazily proven, then cached, symmetry declaration used only when `equation_sign` is
    /// nontrivial (mirrors `RealizationPlan`'s own `symmetry_proof` cache).
    symmetry_proof: std::sync::OnceLock<OperatorSymmetry>,
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

    /// Auto-derived nullspace candidates (mission item 9): representation-only declarations, not
    /// yet resolved against a concrete essential-constraint set. Resolve one against
    /// [`Self::layout`] via [`BlockNullspaceCandidate::resolve`].
    pub fn nullspace_candidates(&self) -> &[BlockNullspaceCandidate] {
        &self.data.nullspace_candidates
    }

    pub fn dof_map(&self, field: SymbolId) -> Option<&DofMap> {
        self.data.fields.get(&field).map(|field| &field.dofs)
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
        for (block_index, block) in self.data.plan.system().blocks.iter().enumerate() {
            if only_block.is_some_and(|only| only != block_index) {
                continue;
            }
            self.apply_block_cell(
                block_index,
                block,
                cell,
                &geometry,
                &affine,
                time,
                state,
                state_rate,
                action,
                output,
            )?;
        }
        Ok(())
    }

    fn input_bindings(&self) -> SystemInputBindings<'_> {
        SystemInputBindings {
            constitutive: &self.data.constitutive,
            stored: &self.data.stored,
            point_count: self.data.quadrature.len(),
        }
    }

    /// The shared cell quadrature table every field of this operator is integrated with
    /// (degree-4-exact on triangles, degree-2 on tetrahedra); stored tables
    /// ([`SystemExternalInput`]) and distributed coefficients ([`crate::CoefficientLayout`]) are
    /// laid out over it.
    pub fn quadrature(&self) -> &[QuadraturePoint] {
        &self.data.quadrature
    }

    /// Per-cell gather of every realized field's local DOF values from a layout-wide vector.
    fn gather_local(&self, cell: usize, vector: &[f64]) -> BTreeMap<SymbolId, Vec<f64>> {
        let layout = self.layout();
        self.data
            .fields
            .iter()
            .map(|(&symbol, field)| {
                let field_block = layout
                    .block(symbol)
                    .expect("realized field implies a layout block");
                let restriction = &field.dofs.restrictions()[cell];
                (
                    symbol,
                    restriction
                        .dofs
                        .iter()
                        .map(|dof| vector[field_block.offset + dof.0])
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
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
                    .flat_map(|(&symbol, field)| {
                        let offset = layout
                            .block(symbol)
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
        let origin = self
            .system_ids()
            .residual_origin(coefficient.residual)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "distributed coefficient names residual {} which the system does not carry",
                    coefficient.residual
                ))
            })?;
        let system = self.data.plan.system();
        let block_index = system
            .blocks
            .iter()
            .position(|block| block.equation == origin.equation)
            .ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "residual {} names equation `{}` which the system does not carry",
                    coefficient.residual, origin.equation
                ))
            })?;
        let integral = system.blocks[block_index]
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
        let block = &self.data.plan.system().blocks[block_index];
        let row_field = &self.data.fields[&block.row];
        let row_block = self
            .layout()
            .block(block.row)
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
            let local_state = self.gather_local(cell, state);
            let local_rate = self.gather_local(cell, state_rate);
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
                let (inputs, _) = point_inputs_system(
                    &self.data.fields,
                    integral,
                    cell,
                    point,
                    &geometry,
                    &affine,
                    reference_point,
                    time,
                    &local_state,
                    &local_rate,
                    &self.input_bindings(),
                    block_index,
                )?;
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = bound_kernel(bindings, block, integral, output_index)?;
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
        let block = &self.data.plan.system().blocks[block_index];
        let row_field = &self.data.fields[&block.row];
        let row_block = self
            .layout()
            .block(block.row)
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
            let local_state = self.gather_local(cell, state);
            let local_rate = self.gather_local(cell, state_rate);
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
                let (inputs, _) = point_inputs_system(
                    &self.data.fields,
                    integral,
                    cell,
                    point,
                    &geometry,
                    &affine,
                    reference_point,
                    time,
                    &local_state,
                    &local_rate,
                    &self.input_bindings(),
                    block_index,
                )?;
                for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                    let bound = bound_kernel(bindings, block, integral, output_index)?;
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
        let system = self.data.plan.system();
        if let Some((&(block_index, integral_index, input), binding)) =
            self.data.constitutive.iter().next()
        {
            return Err(FinitumError::RepresentationUnsupported {
                representation: RepresentationKind::PartialAssembly,
                equation: system.blocks[block_index].equation.clone(),
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
        for block in &system.blocks {
            if let Some(integral) = block
                .factorization
                .integrals
                .iter()
                .find(|integral| !matches!(integral.measure, SemanticMeasure::Cell { .. }))
            {
                return Err(FinitumError::RepresentationUnsupported {
                    representation: RepresentationKind::PartialAssembly,
                    equation: block.equation.clone(),
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
            let local_zero = self.gather_local(cell, &zero);
            let mut cell_actions = Vec::new();
            for (block_index, block) in system.blocks.iter().enumerate() {
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
                        let (inputs, _) = point_inputs_system(
                            &self.data.fields,
                            integral,
                            cell,
                            point,
                            &geometry,
                            &affine,
                            reference_point,
                            0.0,
                            &local_zero,
                            &local_zero,
                            &self.input_bindings(),
                            block_index,
                        )?;
                        for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                            let bound = bound_kernel(bindings, block, integral, output_index)?;
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
                                row: block.row,
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
        let ids = self.system_ids();
        let residual = ids.residual_origin(row).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("system has no residual {row}"))
        })?;
        let block_index = self
            .data
            .plan
            .system()
            .blocks
            .iter()
            .position(|block| block.equation == residual.equation)
            .ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "residual {row} names equation `{}` which the system does not carry",
                    residual.equation
                ))
            })?;
        let layout = self.layout();
        let row_block = layout.block(residual.row).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "residual {row} row field {} has no layout block",
                residual.row
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
        let element_field = self.data.fields.get(&field).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("field {field} was not realized"))
        })?;
        let block = self
            .layout()
            .block(field)
            .expect("realized field implies a layout block");
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

    /// One block's contribution on one cell for one [`SystemAction`]: gathers every field's
    /// local state/rate (and directions) through its own DOF restriction, drives each cell
    /// integral output's bound kernel at every shared quadrature point, and scatters through
    /// the row field's basis (PRIMAL/JVP) or through every active column field's basis (VJP).
    #[allow(clippy::too_many_arguments)]
    fn apply_block_cell(
        &self,
        block_index: usize,
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
        let row_field = self.data.fields.get(&block.row).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "equation `{}` row field {} was not realized",
                block.equation, block.row
            ))
        })?;
        let row_block = layout
            .block(block.row)
            .expect("row field implies a layout block");
        let row_restriction = &row_field.dofs.restrictions()[cell];
        let sign = self
            .data
            .equation_sign
            .get(&block_index)
            .copied()
            .unwrap_or(1.0);

        let gather = |vector: &[f64]| -> BTreeMap<SymbolId, Vec<f64>> {
            self.data
                .fields
                .iter()
                .map(|(&symbol, field)| {
                    let field_block = layout
                        .block(symbol)
                        .expect("realized field implies a layout block");
                    let restriction = &field.dofs.restrictions()[cell];
                    (
                        symbol,
                        restriction
                            .dofs
                            .iter()
                            .map(|dof| vector[field_block.offset + dof.0])
                            .collect::<Vec<_>>(),
                    )
                })
                .collect()
        };
        let local_state = gather(state);
        let local_rate = gather(state_rate);
        let bindings = &self.data.bindings[&block_index];
        let quadrature = &self.data.quadrature;

        match action {
            SystemAction::Primal | SystemAction::Jvp { .. } => {
                let directions = match action {
                    SystemAction::Jvp {
                        state_direction,
                        rate_direction,
                    } => Some((gather(state_direction), gather(rate_direction))),
                    _ => None,
                };
                let mut local_output = vec![0.0; row_restriction.dofs.len()];
                for integral in &block.factorization.integrals {
                    if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                        // Exterior-facet integrals are processed by `Self::apply_facets`.
                        continue;
                    }
                    for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                        let bound = bound_kernel(bindings, block, integral, output_index)?;
                        for (point, quadrature_point) in quadrature.iter().enumerate() {
                            let reference_point = &quadrature_point.coordinates;
                            let scale = quadrature_point.weight * geometry.determinant();
                            let (inputs, evaluation) = point_inputs_system(
                                &self.data.fields,
                                integral,
                                cell,
                                point,
                                geometry,
                                affine,
                                reference_point,
                                time,
                                &local_state,
                                &local_rate,
                                &self.input_bindings(),
                                block_index,
                            )?;
                            let point_output = match &directions {
                                None => execute_primal_values(bound, &inputs)?,
                                Some((local_state_direction, local_rate_direction)) => {
                                    let directions = point_directions_system(
                                        &self.data.fields,
                                        integral,
                                        cell,
                                        point,
                                        geometry,
                                        affine,
                                        reference_point,
                                        time,
                                        local_state_direction,
                                        local_rate_direction,
                                        &self.input_bindings(),
                                        block_index,
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
                    .map(|(&symbol, field)| {
                        (
                            symbol,
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
                        let bound = bound_kernel(bindings, block, integral, output_index)?;
                        let output_components = component_count(&qoutput.shape)?;
                        for (point, quadrature_point) in quadrature.iter().enumerate() {
                            let reference_point = &quadrature_point.coordinates;
                            let scale = quadrature_point.weight * geometry.determinant();
                            let (inputs, evaluation) = point_inputs_system(
                                &self.data.fields,
                                integral,
                                cell,
                                point,
                                geometry,
                                affine,
                                reference_point,
                                time,
                                &local_state,
                                &local_rate,
                                &self.input_bindings(),
                                block_index,
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
                            if !bound.bundle.parameter.independent_operands.is_empty() {
                                let parameter_cotangents =
                                    point_parameter_cotangents(bound, &inputs, &seed)?;
                                accumulate_parameter_cotangents_system(
                                    &self.data.fields,
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
                                    &self.input_bindings(),
                                    block_index,
                                    &parameter_cotangents,
                                    &mut cotangents,
                                    &mut local_outputs,
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
                                    &self.data.fields,
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
                        }
                    }
                }
                for (symbol, local_output) in &local_outputs {
                    let field_block = layout
                        .block(*symbol)
                        .expect("realized field implies a layout block");
                    let restriction = &self.data.fields[symbol].dofs.restrictions()[cell];
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
        for (block_index, block) in self.data.plan.system().blocks.iter().enumerate() {
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
                    self.apply_block_facet(block_index, block, integral, facet_id, output, action)?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_block_facet(
        &self,
        block_index: usize,
        block: &OperatorSystemBlock,
        integral: &IntegralOperatorFactorization,
        facet_id: FacetId,
        output: &mut [f64],
        action: FacetAction,
    ) -> Result<(), FinitumError> {
        let layout = self.layout();
        let row_field = self.data.fields.get(&block.row).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "equation `{}` row field {} was not realized",
                block.equation, block.row
            ))
        })?;
        let orientations = match &row_field.kind {
            FieldKind::Hdiv0 { orientations } => orientations,
            FieldKind::Lagrange(_) => {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "equation `{}` has an exterior-facet integral but row field {} is not \
                     Hdiv(order=0) (should have been refused at bind_kernels_with_facets time)",
                    block.equation, block.row
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
            .block(block.row)
            .expect("row field implies a layout block");
        let restriction = &row_field.dofs.restrictions()[cell];
        let mut local_output = vec![0.0; restriction.dofs.len()];
        let bindings = &self.data.bindings[&block_index];
        let inputs = BTreeMap::new();
        for (output_index, _qoutput) in integral.primal.outputs.iter().enumerate() {
            let bound = bindings
                .get(&(integral.integral_index, output_index))
                .ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "equation `{}` integral {} output {output_index} has no bound kernel",
                        block.equation, integral.integral_index
                    ))
                })?;
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
    block: &OperatorSystemBlock,
    integral: &IntegralOperatorFactorization,
    output_index: usize,
) -> Result<&'a BoundBundle, FinitumError> {
    bindings
        .get(&(integral.integral_index, output_index))
        .ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "equation `{}` integral {} output {output_index} has no bound kernel",
                block.equation, integral.integral_index
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
            OperatorSymmetry::Unknown
        } else {
            map_form_symmetry(self.data.structure.form_symmetry)
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

/// A [`SystemOperator`] with essential (Dirichlet) constraints eliminated through
/// [`SystemOperator::apply_reduced_action`] -- mirroring `crate::mixed::ReducedMixedOperator`.
#[derive(Clone, Debug)]
pub struct ReducedSystemOperator {
    operator: SystemOperator,
    constraints: ConstraintSet,
}

impl ReducedSystemOperator {
    pub fn operator(&self) -> &SystemOperator {
        &self.operator
    }

    pub fn constraints(&self) -> &ConstraintSet {
        &self.constraints
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
    /// constrained coordinate takes its Dirichlet value), the rate homogeneously, the physical
    /// residual is restricted back, and every constrained row becomes its own constraint
    /// residual `u_t - value` -- the `F(t, y, y') = 0` shape a Krasis transaction or a
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
        let physical_state = self.constraints.expand(state)?;
        let physical_rate = self.constraints.expand_homogeneous(state_rate)?;
        let mut physical_output = vec![0.0; dimension];
        self.operator
            .residual(time, &physical_state, &physical_rate, &mut physical_output)?;
        output.copy_from_slice(&self.constraints.restrict_transpose(&physical_output)?);
        for constraint in self.constraints.constraints() {
            output[constraint.target.0] = self
                .constraints
                .equation_residual(state, constraint.target)?;
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
        let physical_state = self.constraints.expand(state)?;
        let physical_rate = self.constraints.expand_homogeneous(state_rate)?;
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
        let physical_state = self.constraints.expand(state)?;
        let physical_rate = self.constraints.expand_homogeneous(state_rate)?;
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
        let physical_state = self.constraints.expand(state)?;
        let physical_rate = self.constraints.expand_homogeneous(state_rate)?;
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
        let physical_state = self.constraints.expand(state)?;
        let physical_rate = self.constraints.expand_homogeneous(state_rate)?;
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
        self.operator
            .element_assembly_with(self.constraints.clone(), lane_width)
    }

    /// [`SystemOperator::partial_assembly`] reduced by this operator's constraint rows.
    pub fn partial_assembly(
        &self,
        lane_width: usize,
    ) -> Result<SystemPartialAssemblyOperator, FinitumError> {
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
        let system = data.plan.system();
        let mut elements = Vec::new();
        for block in &system.blocks {
            for element in &block.requirements.elements {
                if elements
                    .iter()
                    .any(|seen: &CapabilityElement| seen.symbol == element.symbol)
                {
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
        let measures = system
            .blocks
            .iter()
            .flat_map(|block| block.factorization.integrals.iter())
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
        if data.facet_regions.is_empty() {
            representation_kinds.push(RepresentationKind::ElementAssembly);
            if data.constitutive.is_empty() {
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
                system
                    .blocks
                    .iter()
                    .map(|block| &block.requirements.artifact_digest),
            ),
            source_factorization_digest: combined_digest(
                system
                    .blocks
                    .iter()
                    .map(|block| &block.factorization.artifact_digest),
            ),
            source_kernels_digest: combined_digest(
                system
                    .blocks
                    .iter()
                    .map(|block| &block.kernels.artifact_digest),
            ),
            realization_digest: data.digest.clone(),
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
        let system = data.plan.system();
        let ids = data.plan.system_ids();
        let blocks = system
            .blocks
            .iter()
            .map(|block| SystemBlockReceipt {
                equation: block.equation.clone(),
                residual: ids
                    .residuals()
                    .iter()
                    .find(|residual| residual.equation == block.equation)
                    .map(|residual| residual.id),
                source_requirements_digest: block.requirements.artifact_digest.clone(),
                source_factorization_digest: block.factorization.artifact_digest.clone(),
                source_kernels_digest: block.kernels.artifact_digest.clone(),
            })
            .collect();
        let fields = data
            .fields
            .iter()
            .map(|(&symbol, field)| SystemFieldArtifact {
                symbol,
                variable: data
                    .plan
                    .layout()
                    .block(symbol)
                    .expect("realized field implies a layout block")
                    .variable,
                dofs: field.dofs.clone(),
            })
            .collect();
        let residual_of = |block_index: usize| {
            ids.residuals()
                .iter()
                .find(|residual| residual.equation == system.blocks[block_index].equation)
                .map(|residual| residual.id)
        };
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
        SystemRealizationArtifact {
            schema: SYSTEM_REALIZATION_ARTIFACT_SCHEMA.into(),
            artifact_digest: data.digest.clone(),
            plan_digest: data.plan.artifact_digest().clone(),
            system_ids_identity: ids.identity().clone(),
            blocks,
            mesh: data.plan.mesh().clone(),
            fields,
            constraints: self.constraints.clone(),
            external_inputs,
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
    /// [`SystemOperator::digest`] (`finitum-system-operator/2`).
    pub artifact_digest: Digest,
    pub plan_digest: Digest,
    /// [`SystemIdMap::identity`] (`finitum-system-ids/1`).
    pub system_ids_identity: Digest,
    pub blocks: Vec<SystemBlockReceipt>,
    pub mesh: Mesh,
    pub fields: Vec<SystemFieldArtifact>,
    pub constraints: ConstraintSet,
    pub external_inputs: Vec<SystemRealizationExternalInput>,
}

/// One stored quadrature-point Jacobian of a [`SystemPartialAssemblyOperator`].
#[derive(Clone, Debug)]
struct SystemPartialPointAction {
    row: SymbolId,
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
                let local = self.operator.gather_local(cell, &physical);
                let mut local_outputs = local
                    .iter()
                    .map(|(&symbol, values)| (symbol, vec![0.0; values.len()]))
                    .collect::<BTreeMap<_, _>>();
                for action in &self.point_actions[cell] {
                    let reference_point = &data.quadrature[action.point].coordinates;
                    let mut point_input = Vec::new();
                    for qinput in &action.active_inputs {
                        if qinput.binding.evaluation.derivative
                            == DerivativeEvaluation::TimeDerivative
                        {
                            point_input.extend(vec![0.0; component_count(&qinput.shape)?]);
                            continue;
                        }
                        let field = &data.fields[&qinput.binding.symbol];
                        point_input.extend(evaluate_field_basis_input(
                            field,
                            &geometry,
                            &affine,
                            cell,
                            action.point,
                            reference_point,
                            qinput,
                            &local[&qinput.binding.symbol],
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
                for (symbol, local_output) in &local_outputs {
                    let offset = layout
                        .block(*symbol)
                        .expect("realized field implies a layout block")
                        .offset;
                    let restriction = &data.fields[symbol].dofs.restrictions()[cell];
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
                                SystemConstitutiveInput::new(
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
                                        vec![
                                            evaluate_kernel_value(&kernel, &executable, &named)
                                                .unwrap_or(f64::NAN),
                                        ]
                                    },
                                    |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
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
                                SystemConstitutiveInput::new(
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
                                        let Some(active) = point.input_values(active_id) else {
                                            return vec![f64::NAN];
                                        };
                                        named.insert(value_name.clone(), active[0]);
                                        vec![
                                            evaluate_kernel_value(
                                                &value_kernel,
                                                &value_executable,
                                                &named,
                                            )
                                            .unwrap_or(f64::NAN),
                                        ]
                                    },
                                    move |point: &PointEvaluation, direction: &PointEvaluation| {
                                        let mut named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let (Some(active), Some(active_direction)) = (
                                            point.input_values(active_id),
                                            direction.input_values(active_id),
                                        ) else {
                                            return vec![f64::NAN];
                                        };
                                        named.insert(direction_name.clone(), active[0]);
                                        let partial = evaluate_kernel_partial(
                                            &direction_kernel,
                                            &direction_executable,
                                            &named,
                                            &direction_name,
                                        );
                                        match partial {
                                            Ok(Some(partial)) => {
                                                vec![partial * active_direction[0]]
                                            }
                                            _ => vec![f64::NAN],
                                        }
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
                                SystemConstitutiveInput::new(
                                    block.equation.clone(),
                                    integral.integral_index,
                                    input.id,
                                    1,
                                    identity,
                                    move |point: &PointEvaluation| {
                                        let named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let axis_point = table
                                            .axes
                                            .iter()
                                            .map(|axis| named.get(&axis.name).copied())
                                            .collect::<Option<Vec<_>>>();
                                        vec![
                                            axis_point
                                                .and_then(|point| {
                                                    evaluate_table_value(&table, &point).ok()
                                                })
                                                .unwrap_or(f64::NAN),
                                        ]
                                    },
                                    |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
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
                                SystemConstitutiveInput::new(
                                    block.equation.clone(),
                                    integral.integral_index,
                                    input.id,
                                    1,
                                    identity,
                                    move |point: &PointEvaluation| {
                                        let mut named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let Some(active) = point.input_values(active_id) else {
                                            return vec![f64::NAN];
                                        };
                                        named.insert(value_name.clone(), active[0]);
                                        let axis_point = value_table
                                            .axes
                                            .iter()
                                            .map(|axis| named.get(&axis.name).copied())
                                            .collect::<Option<Vec<_>>>();
                                        vec![
                                            axis_point
                                                .and_then(|point| {
                                                    evaluate_table_value(&value_table, &point).ok()
                                                })
                                                .unwrap_or(f64::NAN),
                                        ]
                                    },
                                    move |point: &PointEvaluation, direction: &PointEvaluation| {
                                        let mut named =
                                            named_coordinate_inputs(&point.coordinates, point.time);
                                        let (Some(active), Some(active_direction)) = (
                                            point.input_values(active_id),
                                            direction.input_values(active_id),
                                        ) else {
                                            return vec![f64::NAN];
                                        };
                                        named.insert(slope_name.clone(), active[0]);
                                        let axis_point = slope_table
                                            .axes
                                            .iter()
                                            .map(|axis| named.get(&axis.name).copied())
                                            .collect::<Option<Vec<_>>>();
                                        vec![
                                            axis_point
                                                .and_then(|point| {
                                                    evaluate_table_slope(
                                                        &slope_table,
                                                        &point,
                                                        axis_index,
                                                    )
                                                    .ok()
                                                })
                                                .map_or(f64::NAN, |slope| {
                                                    slope * active_direction[0]
                                                }),
                                        ]
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
    fields: &BTreeMap<SymbolId, FieldElement>,
    symbol: SymbolId,
    geometry: &CellGeometry,
    affine: &AffineMap,
    cell: usize,
    point: usize,
    reference_point: &[f64],
    derivative: &DerivativeEvaluation,
    cotangent: &[f64],
    scale: f64,
    local_outputs: &mut BTreeMap<SymbolId, Vec<f64>>,
) -> Result<(), FinitumError> {
    let field = fields.get(&symbol).ok_or_else(|| {
        FinitumError::ArtifactMismatch(format!(
            "VJP cotangent references field {symbol} which the system operator has not realized"
        ))
    })?;
    let local_output = local_outputs
        .get_mut(&symbol)
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

/// System analogue of `RealizationPlan::accumulate_parameter_cotangents`: routes each frozen-
/// input cotangent of the bound parameter kernel to its exact destination. A passive basis-
/// sourced input scatters directly through its own field's basis; a constitutive closure's
/// cotangent is pushed back into the active cotangents by probing its trusted `direction`
/// closure with unit active perturbations (exact because that closure is contracted to return
/// the exact, hence linear and homogeneous, directional derivative of its value closure).
#[allow(clippy::too_many_arguments)]
fn accumulate_parameter_cotangents_system(
    fields: &BTreeMap<SymbolId, FieldElement>,
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
    block_index: usize,
    parameter_cotangents: &BTreeMap<TensorInputId, Vec<f64>>,
    cotangents: &mut BTreeMap<TensorInputId, Vec<f64>>,
    local_outputs: &mut BTreeMap<SymbolId, Vec<f64>>,
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
        let key = (block_index, integral.integral_index, *input_id);
        let binding = match bindings.resolve(key)? {
            // A stored table is state-independent: no chain rule through it.
            SystemInputBinding::Stored(_) => continue,
            SystemInputBinding::Constitutive(binding) => binding,
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
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn point_inputs_system(
    fields: &BTreeMap<SymbolId, FieldElement>,
    integral: &IntegralOperatorFactorization,
    cell: usize,
    point: usize,
    geometry: &CellGeometry,
    affine: &AffineMap,
    reference_point: &[f64],
    time: f64,
    local_state: &BTreeMap<SymbolId, Vec<f64>>,
    local_rate: &BTreeMap<SymbolId, Vec<f64>>,
    bindings: &SystemInputBindings<'_>,
    block_index: usize,
) -> Result<(BTreeMap<TensorInputId, Vec<f64>>, PointEvaluation), FinitumError> {
    let mut inputs = BTreeMap::new();
    let mut active = Vec::new();
    for input in &integral.primal.inputs {
        if input.source != InputSourceRequirement::Basis {
            continue;
        }
        let field = fields.get(&input.binding.symbol).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "integral {} input {:?} references field {} which the system operator has not \
                 realized",
                integral.integral_index, input.id, input.binding.symbol
            ))
        })?;
        let dofs = if input.binding.evaluation.derivative == DerivativeEvaluation::TimeDerivative {
            &local_rate[&input.binding.symbol]
        } else {
            &local_state[&input.binding.symbol]
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
    };
    for input in &integral.primal.inputs {
        if input.source == InputSourceRequirement::Basis {
            continue;
        }
        let key = (block_index, integral.integral_index, input.id);
        let values = match bindings.resolve(key)? {
            SystemInputBinding::Stored(stored) => stored
                .point_values(cell, point, bindings.point_count)
                .to_vec(),
            SystemInputBinding::Constitutive(binding) => binding.evaluate_value(&evaluation)?,
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
    fields: &BTreeMap<SymbolId, FieldElement>,
    integral: &IntegralOperatorFactorization,
    cell: usize,
    point: usize,
    geometry: &CellGeometry,
    affine: &AffineMap,
    reference_point: &[f64],
    time: f64,
    local_state_direction: &BTreeMap<SymbolId, Vec<f64>>,
    local_rate_direction: &BTreeMap<SymbolId, Vec<f64>>,
    bindings: &SystemInputBindings<'_>,
    block_index: usize,
    evaluation: &PointEvaluation,
) -> Result<BTreeMap<TensorInputId, Vec<f64>>, FinitumError> {
    let mut directions = BTreeMap::new();
    let mut active = Vec::new();
    for input in &integral.primal.inputs {
        if input.source != InputSourceRequirement::Basis {
            continue;
        }
        let field = fields.get(&input.binding.symbol).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "integral {} input {:?} references field {} which the system operator has not \
                 realized",
                integral.integral_index, input.id, input.binding.symbol
            ))
        })?;
        let dofs = if input.binding.evaluation.derivative == DerivativeEvaluation::TimeDerivative {
            &local_rate_direction[&input.binding.symbol]
        } else {
            &local_state_direction[&input.binding.symbol]
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
    };
    for input in &integral.primal.inputs {
        if input.source == InputSourceRequirement::Basis {
            continue;
        }
        let key = (block_index, integral.integral_index, input.id);
        let values = match bindings.resolve(key)? {
            SystemInputBinding::Stored(stored) => vec![0.0; stored.component_count()],
            SystemInputBinding::Constitutive(binding) => {
                binding.evaluate_direction(evaluation, &direction_evaluation)?
            }
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
/// [`essential_constraints_for_blocks`] (unchanged) for the elimination-facing `ConstraintSet`
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
/// `Constant`/`Nodal`/`Sampled` are admitted).
pub fn essential_constraints_from_system(
    operator: &SystemOperator,
    mesh: &TaggedMesh,
    region_map: &RegionMap,
    requirements: &[SystemEssentialConstraintRequirement],
) -> Result<ConstraintSet, FinitumError> {
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
        let dof_map = operator.dof_map(requirement.field).ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "essential constraint requirement names field {} which the system operator has \
                 not realized",
                requirement.field
            ))
        })?;
        let block = layout
            .block(requirement.field)
            .expect("a realized field's DOF map implies a layout block");
        let components = block.component_count;
        if matches!(
            operator.data.fields[&requirement.field].kind,
            FieldKind::Hdiv0 { .. }
        ) {
            rt0_essential_values(
                &facets,
                &mesh.mesh,
                mesh,
                region_map,
                requirement,
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
                FieldSource::Table(_) | FieldSource::Kernel { .. } => {
                    return Err(FinitumError::UnsupportedRealization(
                        "essential_constraints_from_system admits Constant/Nodal/Sampled field \
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
                values.push(BlockEssentialValue {
                    block: requirement.field,
                    entity: node,
                    component,
                    value,
                });
            }
        }
    }
    essential_constraints_for_blocks(layout, values)
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
fn rt0_essential_values(
    facets: &crate::FacetTopology,
    mesh: &Mesh,
    tagged: &TaggedMesh,
    region_map: &RegionMap,
    requirement: &SystemEssentialConstraintRequirement,
    values: &mut Vec<BlockEssentialValue>,
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
                requirement.field, facet_id.0
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
            FieldSource::Nodal(_) => {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "RT0 field {} has no nodes; essential normal-trace data must be a \
                     Constant or Sampled scalar",
                    requirement.field
                )));
            }
            FieldSource::Table(_) | FieldSource::Kernel { .. } => {
                return Err(FinitumError::UnsupportedRealization(
                    "essential_constraints_from_system admits Constant/Nodal/Sampled field \
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
        values.push(BlockEssentialValue {
            block: requirement.field,
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
