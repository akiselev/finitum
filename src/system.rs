use crate::element::{
    rt0_basis_count, rt0_reference_basis, simplex_basis, simplex_basis_count, simplex_quadrature,
};
use crate::mesh::CellId;
use crate::mixed::{
    BlockEssentialValue, BlockNullspaceCandidate, essential_constraints_for_blocks,
    solver_block_layout,
};
use crate::profile::{RegionMap, TaggedMesh};
use crate::realization::{
    BoundBundle, CellGeometry, FacetGeometry, PointActiveInput, PointEvaluation,
    apply_basis_adjoint, bind_kernels, component_count, evaluate_basis_input, execute_jvp_values,
    execute_primal_values, validate_finite,
};
use crate::space::{
    DofMap, ElementRestriction, cell_constant_dof_map, quadratic_simplex_dof_map,
    vector_nodal_dof_map,
};
use crate::{
    AffineMap, BlockLayout, CompatibleDofMaps, ConstraintSet, ExactSequence, FacetId,
    FacetTopology, FieldSource, FinitumError, Mesh, PreparedElement, QuadraturePoint,
};
use methodus::{
    BlockLinearOperator, Definiteness, EvaluationContext, LinearOperator, NumericError,
    OperatorProperties, OperatorStructureHint, OperatorSymmetry,
};
use scientia::scientific::ValueShape;
use scientia::{
    DerivativeEvaluation, Digest, ElementFamilyRequirement, EssentialConstraintRequirement,
    EvaluationSite, FormSymmetry, InputSourceRequirement, IntegralOperatorFactorization,
    NullspaceKind, OperatorStructure, OperatorSystem, OperatorSystemBlock, RegionId,
    SemanticMeasure, SymbolId, TensorInputId, TensorInputRole, TraceMapping,
    derive_operator_structure_for_system,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Digest-bound concrete ownership plan for an FC8 mixed operator system.
#[derive(Clone, Debug)]
pub struct SystemRealizationPlan {
    system: Arc<OperatorSystem>,
    mesh: Mesh,
    layout: BlockLayout,
    facets: FacetTopology,
    compatible_dofs: Option<CompatibleDofMaps>,
    exact_sequence: Option<ExactSequence>,
    artifact_digest: Digest,
}

impl SystemRealizationPlan {
    pub fn new(
        system: OperatorSystem,
        mesh: Mesh,
        layout: BlockLayout,
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
        validate_components(&system, &layout)?;
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
        let artifact_digest = digest_plan(&system, &mesh, &layout, &facets);
        Ok(Self {
            system: Arc::new(system),
            mesh,
            layout,
            facets,
            compatible_dofs,
            exact_sequence,
            artifact_digest,
        })
    }

    pub fn system(&self) -> &OperatorSystem {
        &self.system
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
    }
    let bytes = serde_json::to_vec(&Payload {
        schema: "finitum-system-realization/1",
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

type SystemPointValueEvaluator = dyn Fn(&PointEvaluation) -> Vec<f64> + Send + Sync;
type SystemPointDirectionEvaluator =
    dyn Fn(&PointEvaluation, &PointEvaluation) -> Vec<f64> + Send + Sync;

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

fn system_operator_digest(
    plan: &SystemRealizationPlan,
    constitutive: &BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
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
    struct Payload<'a> {
        schema: &'static str,
        plan_digest: &'a Digest,
        constitutive: Vec<ConstitutiveIdentity<'a>>,
        equation_sign: &'a BTreeMap<usize, f64>,
        facet_regions: BTreeMap<u32, Vec<usize>>,
    }
    let payload = Payload {
        schema: "finitum-system-operator/1",
        plan_digest: plan.artifact_digest(),
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
        let quadrature = simplex_quadrature(self.mesh.dimension())?;
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
        for (block_index, block) in self.system.blocks.iter().enumerate() {
            for integral in &block.factorization.integrals {
                for input in &integral.primal.inputs {
                    if input.source == InputSourceRequirement::Basis {
                        continue;
                    }
                    let key = (block_index, integral.integral_index, input.id);
                    if !constitutive_by_key.contains_key(&key) {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "equation `{}` integral {} input {:?} requires a caller-supplied \
                             SystemConstitutiveInput (system realization admits closure-based \
                             ModelDefinedConstitutive/Value/Property/ExternalValue resolution \
                             only -- Stored/regional external tensor tables remain future work)",
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
/// matrix-free action over the plan's [`BlockLayout`], evaluated as the generated JVP at zero
/// active state -- the same "globally linear FC6 scope" convention `RealizationPlan::
/// MatrixFreeOperator` and `MixedOperator::apply_action` already document. Declares `symmetry()`/
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

    /// Matrix-free monolithic action `output = A * input`, composed cell-by-cell from every
    /// block's bound kernels (mission item 2).
    pub fn apply_action(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        let dimension = self.dimension();
        if input.len() != dimension || output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "system operator action expects length {dimension}, got input={} output={}",
                input.len(),
                output.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(
                "system operator input must be finite".into(),
            ));
        }
        output.iter_mut().for_each(|value| *value = 0.0);
        for cell in 0..self.data.plan.mesh().cells().len() {
            let geometry = CellGeometry::new(self.data.plan.mesh(), CellId(cell))?;
            let affine = AffineMap::from_cell(self.data.plan.mesh(), CellId(cell))?;
            for (block_index, block) in self.data.plan.system().blocks.iter().enumerate() {
                self.apply_block_cell(block_index, block, cell, &geometry, &affine, input, output)?;
            }
        }
        // Every exterior-facet integral admitted by `bind_kernels_with_facets` has no `Basis`-
        // sourced input at all, so its JVP contribution is exactly zero for any direction (a
        // directional derivative of a state-independent expression) -- this is included for
        // architectural uniformity with the cell-integral treatment (and so a future
        // active-input facet integral, refused typed at bind time, is never silently skipped
        // here rather than refused), not because it changes `output`.
        self.apply_facets(output, FacetAction::Jvp)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(
                "system operator output is not finite".into(),
            ));
        }
        Ok(())
    }

    /// The zero-forcing linear residual `A * state` (mirroring `MixedOperator::residual`: no
    /// forcing/load term is represented by the pure operator action).
    pub fn residual(&self, state: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.apply_action(state, output)
    }

    /// The JVP of a linear map is the map itself.
    pub fn jacobian_vector_product(
        &self,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.apply_action(direction, output)
    }

    /// The affine forcing contribution `apply_action`/`residual`/`jacobian_vector_product`
    /// cannot see: those three evaluate only the generated JVP, and the JVP of a state-
    /// independent (no active `Basis` input) primal output -- exactly the shape a bound source
    /// term like a body-force closure has -- is always exactly zero, by construction (a
    /// directional derivative with nothing to differentiate). This method instead executes every
    /// block's bound PRIMAL Malleus kernel at zero active state (mirroring `RealizationPlan::
    /// execute_primal`, sharing its kernel-execution core via the promoted `crate::realization::
    /// execute_primal_values` rather than duplicating it), scattering the result per-field
    /// through [`Self::layout`] exactly as [`Self::apply_action`] does, then negates it so that,
    /// for a globally linear system, `apply_action(u) == load_vector()` is the correct
    /// zero-Dirichlet weak-form equation (the same sign convention `RealizationPlan::
    /// load_vector` reaches for the single-field case: `PRIMAL(0) = a(0, v) - L(v) = -L(v)`
    /// since `a` is bilinear, so `-PRIMAL(0) = L(v)`, the positive forcing functional).
    ///
    /// Each block's contribution is scaled by its own `equation_sign` exactly as
    /// [`Self::apply_action`] scales its own per-block contribution, so a flipped equation's row
    /// stays consistent between the operator and this load vector (a system solved as
    /// `A * x = load_vector()` remains the same solved system after any subset of rows is
    /// flipped by [`SystemRealizationPlan::bind_kernels`]'s `equation_sign`).
    ///
    /// Full [`Self::dimension`]-length, unconstrained -- this is the multi-block analogue of
    /// `RealizationPlan::load_vector`'s own PRIMAL-kernel execution, but stops short of that
    /// method's Dirichlet-lifting composition (which needs a [`ConstraintSet`] this operator
    /// does not itself own). See [`ReducedSystemOperator::load_vector`] for the composed,
    /// elimination-ready right-hand side.
    ///
    /// A system with no bound source (every non-`Basis` input's closure returns zero, or no
    /// block has one) produces the exact zero vector: every PRIMAL kernel evaluated at zero
    /// active state with an all-zero-valued external/constitutive input returns zero (the
    /// generated kernel is a pure function of its inputs), so there is nothing for the adjoint
    /// scatter to accumulate -- matching `SystemOperator`'s documented all-zero-RHS behavior
    /// today when no source is representable.
    pub fn load_vector(&self) -> Result<Vec<f64>, FinitumError> {
        let dimension = self.dimension();
        let mut output = vec![0.0; dimension];
        for cell in 0..self.data.plan.mesh().cells().len() {
            let geometry = CellGeometry::new(self.data.plan.mesh(), CellId(cell))?;
            let affine = AffineMap::from_cell(self.data.plan.mesh(), CellId(cell))?;
            for (block_index, block) in self.data.plan.system().blocks.iter().enumerate() {
                self.apply_block_cell_load(
                    block_index,
                    block,
                    cell,
                    &geometry,
                    &affine,
                    &mut output,
                )?;
            }
        }
        // Mission item 2's exterior-facet extension: unlike a cell integral's contribution,
        // this is the PRIMAL evaluation itself (not a JVP), so a facet integral with genuinely
        // nonzero data (a future extension beyond `13-mixed-darcy.res`'s own literal
        // `Constant{0.0}`) would contribute here for real.
        self.apply_facets(&mut output, FacetAction::Primal)?;
        for value in &mut output {
            *value = -*value;
        }
        validate_finite("system operator load vector", &output)?;
        Ok(output)
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

    /// Establishes, once, whether this operator's action is self-adjoint, and records the
    /// answer for every later [`Self::symmetry`] query on this operator (and its clones) --
    /// mirroring `RealizationPlan::prove_symmetry` exactly, including its dimension cap. Only
    /// meaningful (and only consulted by [`Self::symmetry`]) when
    /// `SystemRealizationPlan::bind_kernels`'s `equation_sign` was nontrivial; for the default
    /// (unsigned) case `symmetry()` already reports Scientia's structural claim without needing
    /// a proof.
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

    #[allow(clippy::too_many_arguments)]
    fn apply_block_cell(
        &self,
        block_index: usize,
        block: &OperatorSystemBlock,
        cell: usize,
        geometry: &CellGeometry,
        affine: &AffineMap,
        direction: &[f64],
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
        let mut local_output = vec![0.0; row_restriction.dofs.len()];

        let mut local_zero = BTreeMap::new();
        let mut local_direction = BTreeMap::new();
        for (&symbol, field) in &self.data.fields {
            let field_block = layout
                .block(symbol)
                .expect("realized field implies a layout block");
            let restriction = &field.dofs.restrictions()[cell];
            local_zero.insert(symbol, vec![0.0; restriction.dofs.len()]);
            local_direction.insert(
                symbol,
                restriction
                    .dofs
                    .iter()
                    .map(|dof| direction[field_block.offset + dof.0])
                    .collect::<Vec<_>>(),
            );
        }

        let bindings = &self.data.bindings[&block_index];
        for integral in &block.factorization.integrals {
            if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                // Exterior-facet integrals are processed separately by `Self::apply_facets`
                // (mission item 2); this per-cell loop only ever handles `SemanticMeasure::Cell`.
                continue;
            }
            for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                let bound = bindings
                    .get(&(integral.integral_index, output_index))
                    .ok_or_else(|| {
                        FinitumError::ArtifactMismatch(format!(
                            "equation `{}` integral {} output {output_index} has no bound kernel",
                            block.equation, integral.integral_index
                        ))
                    })?;
                for point in 0..self.data.quadrature.len() {
                    let reference_point = &self.data.quadrature[point].coordinates;
                    let scale = self.data.quadrature[point].weight * geometry.determinant();
                    let (inputs, evaluation) = point_inputs_system(
                        &self.data.fields,
                        integral,
                        cell,
                        point,
                        geometry,
                        affine,
                        reference_point,
                        &local_zero,
                        &self.data.constitutive,
                        block_index,
                    )?;
                    let directions = point_directions_system(
                        &self.data.fields,
                        integral,
                        cell,
                        point,
                        geometry,
                        affine,
                        reference_point,
                        &local_direction,
                        &self.data.constitutive,
                        block_index,
                        &evaluation,
                    )?;
                    let point_output = execute_jvp_values(bound, &inputs, &directions)?;
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
        let sign = self
            .data
            .equation_sign
            .get(&block_index)
            .copied()
            .unwrap_or(1.0);
        for (local_index, dof) in row_restriction.dofs.iter().enumerate() {
            output[row_block.offset + dof.0] += sign * local_output[local_index];
        }
        Ok(())
    }

    /// [`Self::load_vector`]'s per-cell, per-block core: the PRIMAL analogue of
    /// [`Self::apply_block_cell`], evaluated at zero active state for every field (no direction
    /// to gather -- PRIMAL takes no direction argument), executed through the promoted
    /// `execute_primal_values` rather than `execute_jvp_values`. Structurally identical to
    /// `apply_block_cell` otherwise, including the same per-block `equation_sign` scaling.
    fn apply_block_cell_load(
        &self,
        block_index: usize,
        block: &OperatorSystemBlock,
        cell: usize,
        geometry: &CellGeometry,
        affine: &AffineMap,
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
        let mut local_output = vec![0.0; row_restriction.dofs.len()];

        let mut local_zero = BTreeMap::new();
        for (&symbol, field) in &self.data.fields {
            let restriction = &field.dofs.restrictions()[cell];
            local_zero.insert(symbol, vec![0.0; restriction.dofs.len()]);
        }

        let bindings = &self.data.bindings[&block_index];
        for integral in &block.factorization.integrals {
            if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                // Exterior-facet integrals are processed separately by `Self::apply_facets`
                // (mission item 2); this per-cell loop only ever handles `SemanticMeasure::Cell`.
                continue;
            }
            for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                let bound = bindings
                    .get(&(integral.integral_index, output_index))
                    .ok_or_else(|| {
                        FinitumError::ArtifactMismatch(format!(
                            "equation `{}` integral {} output {output_index} has no bound kernel",
                            block.equation, integral.integral_index
                        ))
                    })?;
                for point in 0..self.data.quadrature.len() {
                    let reference_point = &self.data.quadrature[point].coordinates;
                    let scale = self.data.quadrature[point].weight * geometry.determinant();
                    let (inputs, _evaluation) = point_inputs_system(
                        &self.data.fields,
                        integral,
                        cell,
                        point,
                        geometry,
                        affine,
                        reference_point,
                        &local_zero,
                        &self.data.constitutive,
                        block_index,
                    )?;
                    let point_output = execute_primal_values(bound, &inputs)?;
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
        let sign = self
            .data
            .equation_sign
            .get(&block_index)
            .copied()
            .unwrap_or(1.0);
        for (local_index, dof) in row_restriction.dofs.iter().enumerate() {
            output[row_block.offset + dof.0] += sign * local_output[local_index];
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

impl LinearOperator for SystemOperator {
    fn rows(&self) -> usize {
        self.dimension()
    }

    fn columns(&self) -> usize {
        self.dimension()
    }

    /// Scientia's structural `form_symmetry` (C5.4/C5.5, item 8) when no `equation_sign`
    /// orientation correction was applied at [`SystemRealizationPlan::bind_kernels`] time
    /// (`structure.form_symmetry` describes exactly this, unsigned, system); otherwise the
    /// proof recorded by an explicit [`Self::prove_symmetry`] call, or `Unknown` when no proof
    /// has been established yet -- a resigned system's symmetry is not implied by the unsigned
    /// system's structural claim, so it is never reused silently.
    fn symmetry(&self) -> OperatorSymmetry {
        if self.data.equation_sign.values().any(|&sign| sign != 1.0) {
            self.data
                .symmetry_proof
                .get()
                .copied()
                .unwrap_or(OperatorSymmetry::Unknown)
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
        self.apply_action(input, output)
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

impl BlockLinearOperator for SystemOperator {
    fn block_layout(&self) -> &methodus::BlockLayout {
        &self.data.solver_layout
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
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

impl BlockLinearOperator for ReducedSystemOperator {
    fn block_layout(&self) -> &methodus::BlockLayout {
        self.operator.block_layout()
    }
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
    local_state: &BTreeMap<SymbolId, Vec<f64>>,
    constitutive: &BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
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
        let dofs = &local_state[&input.binding.symbol];
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
        time: 0.0,
        cell: CellId(cell),
        coordinates: geometry.physical_point(reference_point),
        active,
    };
    for input in &integral.primal.inputs {
        if input.source == InputSourceRequirement::Basis {
            continue;
        }
        let key = (block_index, integral.integral_index, input.id);
        let binding = constitutive.get(&key).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "integral {} input {:?} has no bound constitutive input",
                integral.integral_index, input.id
            ))
        })?;
        let values = (binding.value)(&evaluation);
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
    local_direction: &BTreeMap<SymbolId, Vec<f64>>,
    constitutive: &BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
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
        let dofs = &local_direction[&input.binding.symbol];
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
        time: 0.0,
        cell: CellId(cell),
        coordinates: evaluation.coordinates.clone(),
        active,
    };
    for input in &integral.primal.inputs {
        if input.source == InputSourceRequirement::Basis {
            continue;
        }
        let key = (block_index, integral.integral_index, input.id);
        let binding = constitutive.get(&key).ok_or_else(|| {
            FinitumError::ArtifactMismatch(format!(
                "integral {} input {:?} has no bound constitutive input",
                integral.integral_index, input.id
            ))
        })?;
        let values = (binding.direction)(evaluation, &direction_evaluation);
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
