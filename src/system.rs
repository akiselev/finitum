use crate::element::{simplex_basis, simplex_basis_count, simplex_quadrature};
use crate::mesh::CellId;
use crate::mixed::{
    BlockEssentialValue, BlockNullspaceCandidate, essential_constraints_for_blocks,
    solver_block_layout,
};
use crate::profile::{RegionMap, TaggedMesh};
use crate::realization::{
    BoundBundle, CellGeometry, PointActiveInput, PointEvaluation, apply_basis_adjoint,
    bind_kernels, component_count, evaluate_basis_input, execute_jvp_values, validate_finite,
};
use crate::space::{DofMap, quadratic_simplex_dof_map, vector_nodal_dof_map};
use crate::{
    BlockLayout, CompatibleDofMaps, ConstraintSet, ExactSequence, FacetTopology, FieldSource,
    FinitumError, Mesh, PreparedElement, QuadraturePoint,
};
use methodus::{
    BlockLinearOperator, Definiteness, EvaluationContext, LinearOperator, NumericError,
    OperatorProperties, OperatorStructureHint, OperatorSymmetry,
};
use scientia::scientific::ValueShape;
use scientia::{
    Digest, ElementFamilyRequirement, EssentialConstraintRequirement, EvaluationSite, FormSymmetry,
    InputSourceRequirement, IntegralOperatorFactorization, NullspaceKind, OperatorStructure,
    OperatorSystem, OperatorSystemBlock, SemanticMeasure, SymbolId, TensorInputId, TensorInputRole,
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
            let expected = match space.value_shape {
                ValueShape::Scalar => 1,
                ValueShape::Vector(extent) => usize::from(extent),
                ValueShape::Tensor { rows, cols } => usize::from(rows) * usize::from(cols),
                ValueShape::SymmetricTensor(extent) => usize::from(extent) * usize::from(extent),
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

/// One system field's concrete shared-quadrature Lagrange discretization.
#[derive(Clone, Debug)]
struct FieldElement {
    element: PreparedElement,
    dofs: DofMap,
}

/// Builds one [`FieldElement`] per system field (LAGRANGE/Taylor-Hood scope only).
///
/// Every field's basis table is tabulated at the *same* shared quadrature points
/// (`simplex_quadrature`, mirroring `crate::mixed`'s own cross-block evaluation convention), so
/// a quadrature-point index means the same physical point for every field -- required for
/// [`point_inputs_system`]/[`point_directions_system`] to gather several fields' basis inputs
/// within one integral.
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
/// L2. `ElementFamilyRequirement::Hcurl`/`Hdiv`/`DiscontinuousGalerkin` are refused typed
/// (compatible-element/DG DOF maps are the realization inventory's item 1, out of this lane's
/// scope).
fn build_field_elements(
    system: &OperatorSystem,
    mesh: &Mesh,
    layout: &BlockLayout,
    quadrature: &[QuadraturePoint],
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
        if !matches!(
            requirement.family,
            ElementFamilyRequirement::H1 | ElementFamilyRequirement::L2
        ) || !matches!(requirement.polynomial_order, 1 | 2)
            || requirement.topological_dimension as usize != mesh.dimension()
        {
            return Err(FinitumError::UnsupportedRealization(format!(
                "system realization admits scalar or dimension-vector Lagrange fields of order \
                 1 or 2 (H1 or L2) in the mesh's own topological dimension only; field {symbol} \
                 requires {requirement:?}"
            )));
        }
        let components = match &requirement.value_shape {
            ValueShape::Scalar => 1,
            ValueShape::Vector(extent) if *extent as usize == mesh.dimension() => mesh.dimension(),
            other => {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "system field {symbol} must be scalar or dimension-{}-vector valued, got \
                     {other:?}",
                    mesh.dimension()
                )));
            }
        };
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
        let order = requirement.polynomial_order;
        let basis_count = simplex_basis_count(mesh.dimension(), order);
        let mut basis_values = Vec::with_capacity(quadrature.len() * basis_count);
        let mut basis_gradients =
            Vec::with_capacity(quadrature.len() * basis_count * mesh.dimension());
        for point in quadrature {
            let (values, gradients) = simplex_basis(mesh.dimension(), order, &point.coordinates)?;
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
            _ => unreachable!("polynomial order validated above"),
        };
        fields.insert(symbol, FieldElement { element, dofs });
    }
    Ok(fields)
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
        for block in &self.system.blocks {
            for integral in &block.factorization.integrals {
                if !matches!(integral.measure, SemanticMeasure::Cell { .. }) {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "system realization admits SemanticMeasure::Cell integrals only; \
                         equation `{}` integral {} has measure {:?} (facet bindings are out of \
                         this lane's scope)",
                        block.equation, integral.integral_index, integral.measure
                    )));
                }
                for output in &integral.primal.outputs {
                    if output.binding.evaluation.site != EvaluationSite::Cell {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "system realization realizes cell evaluation sites only; equation \
                             `{}` integral {} has an output at site {:?}",
                            block.equation, integral.integral_index, output.binding.evaluation.site
                        )));
                    }
                }
            }
        }
        let quadrature = simplex_quadrature(self.mesh.dimension())?;
        let fields = build_field_elements(&self.system, &self.mesh, &self.layout, &quadrature)?;
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
        let digest = system_operator_digest(self, &constitutive_by_key, &equation_sign_by_block);
        Ok(SystemOperator {
            data: Arc::new(SystemOperatorData {
                plan: self.clone(),
                fields,
                bindings,
                constitutive: constitutive_by_key,
                equation_sign: equation_sign_by_block,
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
    bindings: BTreeMap<usize, BTreeMap<(usize, usize), BoundBundle>>,
    constitutive: BTreeMap<(usize, usize, TensorInputId), SystemConstitutiveInput>,
    /// Per-block-index equation orientation (`1.0` or `-1.0`, absent means `1.0`); see
    /// `SystemRealizationPlan::bind_kernels`'s `equation_sign` parameter.
    equation_sign: BTreeMap<usize, f64>,
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
            for (block_index, block) in self.data.plan.system().blocks.iter().enumerate() {
                self.apply_block_cell(block_index, block, cell, &geometry, input, output)?;
            }
        }
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
            for (output_index, qoutput) in integral.primal.outputs.iter().enumerate() {
                let bound = bindings
                    .get(&(integral.integral_index, output_index))
                    .ok_or_else(|| {
                        FinitumError::ArtifactMismatch(format!(
                            "equation `{}` integral {} output {output_index} has no bound kernel",
                            block.equation, integral.integral_index
                        ))
                    })?;
                for point in 0..row_field.element.quadrature().len() {
                    let scale =
                        row_field.element.quadrature()[point].weight * geometry.determinant();
                    let (inputs, evaluation) = point_inputs_system(
                        &self.data.fields,
                        integral,
                        cell,
                        point,
                        geometry,
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
                        &local_direction,
                        &self.data.constitutive,
                        block_index,
                        &evaluation,
                    )?;
                    let point_output = execute_jvp_values(bound, &inputs, &directions)?;
                    apply_basis_adjoint(
                        &row_field.element,
                        geometry,
                        point,
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
        let values = evaluate_basis_input(&field.element, geometry, point, input, dofs)?;
        if input.role == TensorInputRole::Active {
            active.push(PointActiveInput {
                input: input.id,
                derivative: input.binding.evaluation.derivative,
                values: values.clone(),
            });
        }
        inputs.insert(input.id, values);
    }
    let any_field = fields
        .values()
        .next()
        .expect("a system realizes at least one field");
    let evaluation = PointEvaluation {
        time: 0.0,
        cell: CellId(cell),
        coordinates: geometry.physical_point(&any_field.element.quadrature()[point].coordinates),
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
        let values = evaluate_basis_input(&field.element, geometry, point, input, dofs)?;
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
