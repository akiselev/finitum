//! SV2-B1/B4: executable vector/mixed product layouts, and a start on block operator
//! composition.
//!
//! [`MixedSpace`] is a product of per-field blocks -- each with its own polynomial order and
//! component count -- laid out end-to-end by an explicit [`BlockLayout`] (offsets/sizes).
//! [`MixedOperator`] composes per-block (diagonal) and coupling (off-diagonal) bilinear actions
//! into one monolithic matrix-free action over that layout (SV2-B4 start).
//!
//! This module names no physics: every bilinear pairing is a generic structural contraction
//! (see [`CouplingKind`]), independent of the single-field, generated-kernel `RealizationPlan`.
//! It does not depend on Scientia forms or execute Malleus kernels; its local element math is
//! the same simplex Lagrange basis machinery [`crate::element`] uses for P1/P2, evaluated at one
//! shared quadrature rule per cell so fields of different order can be paired in one integral.
//!
//! [`BlockNullspaceCandidate`] is a typed, representation-only declaration that one block of a
//! [`BlockLayout`] carries a constant nullspace mode (the pure-Dirichlet pressure mode of a
//! saddle-point system, named generically). It resolves into the exact vector and a
//! `methodus::ConstantModeProjector` a downstream MINRES-family solver (SV2-B6, not implemented
//! here) would consume; no solver algorithm is implemented in this module.

use crate::block::BlockLayout;
use crate::element::{simplex_basis, simplex_basis_count, simplex_quadrature};
use crate::mapping::AffineMap;
use crate::mesh::{CellId, Mesh};
use crate::space::{DofId, DofMap, quadratic_simplex_dof_map, vector_nodal_dof_map};
use crate::{AffineConstraint, ConstraintSet, FinitumError, QuadraturePoint};
use methodus::{
    BlockLinearOperator, ConstantModeProjector, CsrMatrix, EvaluationContext, LinearOperator,
};
use scientia::{Digest, SymbolId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One field's concrete order/shape within a [`MixedSpace`] product layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldSpec {
    pub symbol: SymbolId,
    /// `1` (P1, vertex nodes) or `2` (P2, vertex-then-edge nodes); any other value is a typed
    /// refusal at [`MixedSpace::new`].
    pub order: u8,
    /// `1` (scalar) or the mesh dimension (dimension-vector); no other shape is admitted.
    pub components: usize,
}

/// A product-space simplex discretization: one shared mesh, several fields each with their own
/// order/DOF map, laid out end-to-end by an explicit [`BlockLayout`].
///
/// Reuses [`vector_nodal_dof_map`] (order 1) and [`quadratic_simplex_dof_map`] (order 2) for
/// each field's own block-local DOF numbering (`0..field_dof_count`); [`BlockLayout`] then
/// records each field's `offset`/`extent` in the monolithic vector, with `entity_count` the
/// field's node count and `component_count` its true component count (`1` scalar, or the mesh
/// dimension), so a consumer of the layout alone (e.g. [`BlockNullspaceCandidate::resolve`]) can
/// tell a scalar block from a vector one without consulting [`MixedSpace::fields`].
#[derive(Clone, Debug)]
pub struct MixedSpace {
    mesh: Mesh,
    fields: Vec<FieldSpec>,
    dof_maps: BTreeMap<SymbolId, DofMap>,
    layout: BlockLayout,
}

impl MixedSpace {
    pub fn new(mesh: Mesh, fields: Vec<FieldSpec>) -> Result<Self, FinitumError> {
        if fields.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "a mixed space requires at least one field".into(),
            ));
        }
        let mut dof_maps = BTreeMap::new();
        let mut specifications = Vec::with_capacity(fields.len());
        for field in &fields {
            if field.components == 0 {
                return Err(FinitumError::InvalidRealization(format!(
                    "field {} must declare at least one component",
                    field.symbol
                )));
            }
            if field.components != 1 && field.components != mesh.dimension() {
                return Err(FinitumError::UnsupportedRealization(format!(
                    "field {} declares {} components; a mixed-space field must be scalar (1) or \
                     dimension-vector ({}) valued",
                    field.symbol,
                    field.components,
                    mesh.dimension()
                )));
            }
            let dof_map = match field.order {
                1 => vector_nodal_dof_map(&mesh, field.components)?,
                2 => quadratic_simplex_dof_map(&mesh, field.components)?,
                other => {
                    return Err(FinitumError::UnsupportedRealization(format!(
                        "field {} declares polynomial order {other}; a mixed space admits order \
                         1 or 2",
                        field.symbol
                    )));
                }
            };
            // `dof_map.dof_count()` is exactly `node_count * field.components` (both DOF-map
            // constructors multiply every node by `components`), so `entity_count` here is the
            // node count and `component_count` is the field's true component count -- unlike
            // gather/scatter through `apply_gradient_gradient`/`apply_divergence_value`, which
            // only ever use `block.offset`/`block.extent` (unaffected by this split), this lets
            // `BlockNullspaceCandidate::resolve` tell a genuinely scalar block from a vector one.
            specifications.push((
                field.symbol,
                dof_map.dof_count() / field.components,
                field.components,
            ));
            dof_maps.insert(field.symbol, dof_map);
        }
        let layout = BlockLayout::new(specifications)?;
        Ok(Self {
            mesh,
            fields,
            dof_maps,
            layout,
        })
    }

    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub fn fields(&self) -> &[FieldSpec] {
        &self.fields
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    pub fn dof_map(&self, symbol: SymbolId) -> Option<&DofMap> {
        self.dof_maps.get(&symbol)
    }

    pub fn field(&self, symbol: SymbolId) -> Option<&FieldSpec> {
        self.fields.iter().find(|field| field.symbol == symbol)
    }

    fn restriction(
        &self,
        symbol: SymbolId,
        cell: usize,
    ) -> Result<&crate::space::ElementRestriction, FinitumError> {
        self.dof_maps
            .get(&symbol)
            .and_then(|map| map.restrictions().get(cell))
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "no DOF restriction for field {symbol} at cell {cell}"
                ))
            })
    }
}

/// A generic structural bilinear pairing between two blocks of a [`MixedSpace`], evaluated at a
/// shared quadrature rule. No named physics: these are textbook bilinear-form shapes, reused
/// wherever a scientific model needs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CouplingKind {
    /// `integral(grad(test) . grad(trial))`, decoupled across components (component `c` of the
    /// test field pairs only with component `c` of the trial field). A diagonal (self) block:
    /// `test` and `trial` must name the same field.
    GradientGradient,
    /// `integral(sym_grad(test) : sym_grad(trial))` -- a diagonal (self) block over a
    /// dimension-vector field, `test == trial`. Not a rescaling of [`Self::GradientGradient`]:
    /// pointwise `2 * sym_grad(u):sym_grad(v) = grad(u):grad(v) + sum_ij d(u_i)/d(x_j) *
    /// d(v_j)/d(x_i)`, and that second term only reduces to `div(u)*div(v)` after an
    /// integration by parts whose boundary contribution vanishes solely for compactly
    /// supported (or matching-Dirichlet-eliminated) `u`/`v` -- so the two coupling kinds
    /// realize genuinely different bilinear forms on the raw (pre-elimination) action.
    SymmetricGradientGradient,
    /// `integral(div(test_vector) * trial_scalar)`, contributed together with its exact
    /// transpose `integral(test_scalar . div(trial_vector))` into the mirrored block, so a
    /// [`MixedOperator`] built only from `DivergenceValue` and `GradientGradient` couplings is
    /// symmetric by construction. `test` must be a dimension-vector field; `trial` a scalar
    /// field; `test != trial`.
    DivergenceValue,
}

/// One declared coupling contribution to a [`MixedOperator`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BlockCoupling {
    pub test: SymbolId,
    pub trial: SymbolId,
    pub kind: CouplingKind,
    pub scale: f64,
}

/// One essential (Dirichlet) value on one field's block-local degree of freedom: `entity` is a
/// node index in the field's own numbering (the same convention [`BlockLayout::gather`] and
/// [`BlockLayout::scatter_add`] use for their `entities` argument), `component` is a zero-based
/// component index (`< field.components`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockEssentialValue {
    pub block: SymbolId,
    pub entity: usize,
    pub component: usize,
    pub value: f64,
}

/// Builds a global [`ConstraintSet`] of Fixed (no-dependency) essential constraints from
/// block-local declarations, lifting each `(block, entity, component)` into `layout`'s
/// monolithic degree-of-freedom numbering via each block's `offset` and the same
/// `entity * component_count + component` convention [`BlockLayout::gather`]/
/// [`BlockLayout::scatter_add`] use. Mirrors [`crate::essential_constraints_from`]'s
/// DOF-indexing convention at the block level; unlike it, this function has no region-tag or
/// mesh dependency, since [`MixedSpace`] carries no `TaggedMesh` wiring today (see the SV2-B4
/// realization inventory). Refuses an unknown block, an out-of-range entity or component, a
/// non-finite value, or -- via [`ConstraintSet::new`] -- two declarations targeting the same
/// degree of freedom.
pub fn essential_constraints_for_blocks(
    layout: &BlockLayout,
    values: impl IntoIterator<Item = BlockEssentialValue>,
) -> Result<ConstraintSet, FinitumError> {
    let mut constraints = Vec::new();
    for value in values {
        let block = layout.block(value.block).ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "essential value names block {} which is absent from the layout",
                value.block
            ))
        })?;
        if value.entity >= block.entity_count {
            return Err(FinitumError::InvalidRealization(format!(
                "essential value entity {} is outside block {}'s {} entities",
                value.entity, value.block, block.entity_count
            )));
        }
        if value.component >= block.component_count {
            return Err(FinitumError::InvalidRealization(format!(
                "essential value component {} is outside block {}'s {} components",
                value.component, value.block, block.component_count
            )));
        }
        if !value.value.is_finite() {
            return Err(FinitumError::InvalidRealization(format!(
                "essential value for block {} entity {} component {} is not finite",
                value.block, value.entity, value.component
            )));
        }
        let target = DofId(block.offset + value.entity * block.component_count + value.component);
        constraints.push(AffineConstraint {
            target,
            dependencies: Vec::new(),
            offset: value.value,
        });
    }
    ConstraintSet::new(layout.extent(), constraints)
}

/// Block-and-coupling-operator composition over one [`MixedSpace`], into a single monolithic
/// matrix-free action (SV2-B4 start).
///
/// Every declared [`CouplingKind::GradientGradient`] contributes its own diagonal block;
/// [`CouplingKind::DivergenceValue`] contributes both an off-diagonal block and its exact
/// transpose. The action is linear (no forcing term is represented -- this module is structural
/// machinery, not a named equation), so `residual` and `jacobian_vector_product` coincide with
/// [`Self::apply_action`], matching this crate's existing FC6 convention that a globally linear
/// operator evaluates its own JVP directly.
#[derive(Clone, Debug)]
pub struct MixedOperator {
    space: Arc<MixedSpace>,
    couplings: Vec<BlockCoupling>,
    quadrature: Arc<Vec<QuadraturePoint>>,
    solver_layout: Arc<methodus::BlockLayout>,
    digest: Digest,
}

impl MixedOperator {
    pub fn new(space: MixedSpace, couplings: Vec<BlockCoupling>) -> Result<Self, FinitumError> {
        if couplings.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "a mixed operator requires at least one declared coupling".into(),
            ));
        }
        for coupling in &couplings {
            if !coupling.scale.is_finite() {
                return Err(FinitumError::InvalidRealization(format!(
                    "coupling ({}, {}) has a non-finite scale",
                    coupling.test, coupling.trial
                )));
            }
            let test = space.field(coupling.test).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "coupling names test field {} which is absent from the space",
                    coupling.test
                ))
            })?;
            let trial = space.field(coupling.trial).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "coupling names trial field {} which is absent from the space",
                    coupling.trial
                ))
            })?;
            match coupling.kind {
                CouplingKind::GradientGradient => {
                    if coupling.test != coupling.trial || test.components != trial.components {
                        return Err(FinitumError::UnsupportedRealization(
                            "GradientGradient requires a diagonal coupling (test == trial, \
                             matching component count)"
                                .into(),
                        ));
                    }
                }
                CouplingKind::SymmetricGradientGradient => {
                    if coupling.test != coupling.trial
                        || test.components != trial.components
                        || test.components != space.mesh().dimension()
                    {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "SymmetricGradientGradient requires a diagonal coupling (test == \
                             trial) over a dimension-{}-vector field",
                            space.mesh().dimension()
                        )));
                    }
                }
                CouplingKind::DivergenceValue => {
                    if coupling.test == coupling.trial {
                        return Err(FinitumError::UnsupportedRealization(
                            "DivergenceValue requires two distinct fields".into(),
                        ));
                    }
                    if test.components != space.mesh().dimension() || trial.components != 1 {
                        return Err(FinitumError::UnsupportedRealization(format!(
                            "DivergenceValue requires a dimension-{}-vector test field and a \
                             scalar trial field, got test components {} and trial components {}",
                            space.mesh().dimension(),
                            test.components,
                            trial.components
                        )));
                    }
                }
            }
        }
        let quadrature = simplex_quadrature(space.mesh().dimension())?;
        let solver_layout = solver_block_layout(space.layout())?;
        let digest = mixed_operator_digest(&space, &couplings);
        Ok(Self {
            space: Arc::new(space),
            couplings,
            quadrature: Arc::new(quadrature),
            solver_layout: Arc::new(solver_layout),
            digest,
        })
    }

    pub fn space(&self) -> &MixedSpace {
        &self.space
    }

    pub fn couplings(&self) -> &[BlockCoupling] {
        &self.couplings
    }

    pub fn dimension(&self) -> usize {
        self.space.layout().extent()
    }

    /// Content-addressed identity of this operator's declared structure -- its space (mesh and
    /// field specs) and couplings -- mirroring `RealizationPlan::digest()`'s content-addressing
    /// (GX-D1 91dfd25 recorded this as a proposed follow-up so a `krasis::BlockLinearCheckpoint`
    /// can be bound to a `MixedOperator`'s exact numerical identity rather than only its shape:
    /// `krasis::block_solve::operator_identity` cannot distinguish two `MixedOperator`s that
    /// share a block layout but differ in their coupling scales, since it hashes only `rows`,
    /// `columns`, declared `OperatorProperties`, and `block_layout()`). Computed once at
    /// construction.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// Matrix-free monolithic action `output = A * input`, composed cell-by-cell from every
    /// declared diagonal and coupling block.
    pub fn apply_action(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        let dimension = self.dimension();
        if input.len() != dimension || output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "mixed operator action expects length {dimension}, got input={} output={}",
                input.len(),
                output.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(
                "mixed operator input must be finite".into(),
            ));
        }
        output.iter_mut().for_each(|value| *value = 0.0);
        for cell in 0..self.space.mesh().cells().len() {
            for coupling in &self.couplings {
                match coupling.kind {
                    CouplingKind::GradientGradient => {
                        self.apply_gradient_gradient(cell, coupling, input, output)?;
                    }
                    CouplingKind::SymmetricGradientGradient => {
                        self.apply_symmetric_gradient_gradient(cell, coupling, input, output)?;
                    }
                    CouplingKind::DivergenceValue => {
                        self.apply_divergence_value(cell, coupling, input, output)?;
                    }
                }
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(
                "mixed operator output is not finite".into(),
            ));
        }
        Ok(())
    }

    /// The zero-forcing linear residual `A * state` (SV2-B4: no named forcing term is
    /// represented by this generic structural composition).
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

    /// Canonical CSR assembly by unit-column probing of [`Self::apply_action`] (mirrors this
    /// crate's existing `RealizationPlan::assemble` convention): the same composed action,
    /// materialized. This is *not* an independent oracle; it shares the exact code path with
    /// [`Self::apply_action`] by construction, so it agrees with it trivially. Acceptance tests
    /// compare against a genuinely independent, separately derived monolithic reference.
    pub fn assemble(&self) -> Result<CsrMatrix, FinitumError> {
        let dimension = self.dimension();
        let mut entries = Vec::new();
        let mut unit = vec![0.0; dimension];
        let mut column = vec![0.0; dimension];
        for index in 0..dimension {
            unit[index] = 1.0;
            self.apply_action(&unit, &mut column)?;
            for (row, value) in column.iter().enumerate() {
                if *value != 0.0 {
                    entries.push((row, index, *value));
                }
            }
            unit[index] = 0.0;
        }
        CsrMatrix::from_triplets(dimension, dimension, entries)
            .map_err(|error| FinitumError::Assembly(error.to_string()))
    }

    /// Essential-constraint-eliminated ("reduced") matrix-free action, mirroring
    /// `RealizationPlan::apply_direction`'s identity-row/zero-column treatment exactly, over a
    /// caller-supplied [`ConstraintSet`] instead of a per-realization one. Vectors stay the full
    /// [`Self::dimension`] length: a constrained degree of freedom's column contribution is
    /// removed by zeroing it in the input before [`Self::apply_action`] runs
    /// (`constraints.expand_homogeneous`), and its row becomes an identity row
    /// (`output[t] = input[t]`, via `constraints.direction_residual`) rather than the raw
    /// structural residual [`Self::apply_action`] alone would leave there -- the same
    /// `pressure_like_nullspace_candidate...` fixture that shows a boundary `field_a` row is
    /// genuinely nonzero against the unconstrained action shows it vanish here once that row is
    /// declared constrained (SV2-B4).
    pub fn apply_reduced_action(
        &self,
        constraints: &ConstraintSet,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.dimension();
        if constraints.dof_count() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "essential constraint set has {} degrees of freedom, mixed operator has \
                 {dimension}",
                constraints.dof_count()
            )));
        }
        if output.len() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "mixed operator reduced action expects output length {dimension}, got {}",
                output.len()
            )));
        }
        crate::constraint::apply_constrained_action(constraints, input, output, |input, output| {
            self.apply_action(input, output)
        })
    }

    /// Binds `constraints` (a [`ConstraintSet`] over this operator's monolithic numbering, e.g.
    /// built by [`essential_constraints_for_blocks`]) into a [`ReducedMixedOperator`]: the
    /// essential-constraint-eliminated linear operator a MINRES-family solver actually consumes.
    /// Refuses a constraint set whose `dof_count()` does not match [`Self::dimension`].
    pub fn reduced(
        &self,
        constraints: ConstraintSet,
    ) -> Result<ReducedMixedOperator, FinitumError> {
        if constraints.dof_count() != self.dimension() {
            return Err(FinitumError::InvalidRealization(format!(
                "essential constraint set has {} degrees of freedom, mixed operator has {}",
                constraints.dof_count(),
                self.dimension()
            )));
        }
        Ok(ReducedMixedOperator {
            operator: self.clone(),
            constraints,
        })
    }

    fn apply_gradient_gradient(
        &self,
        cell: usize,
        coupling: &BlockCoupling,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let field = self
            .space
            .field(coupling.test)
            .expect("validated at construction");
        let restriction = self.space.restriction(coupling.test, cell)?;
        let block = self
            .space
            .layout()
            .block(coupling.test)
            .expect("validated at construction");
        let local = local_gradient_gradient(
            self.space.mesh(),
            cell,
            field.order,
            field.components,
            &self.quadrature,
        )?;
        let size = restriction.dofs.len();
        let local_input = restriction
            .dofs
            .iter()
            .map(|dof| input[block.offset + dof.0])
            .collect::<Vec<_>>();
        let mut local_output = vec![0.0; size];
        for row in 0..size {
            let mut accumulator = 0.0;
            for column in 0..size {
                accumulator += local[row * size + column] * local_input[column];
            }
            local_output[row] = coupling.scale * accumulator;
        }
        for (local_index, dof) in restriction.dofs.iter().enumerate() {
            output[block.offset + dof.0] += local_output[local_index];
        }
        Ok(())
    }

    fn apply_symmetric_gradient_gradient(
        &self,
        cell: usize,
        coupling: &BlockCoupling,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let field = self
            .space
            .field(coupling.test)
            .expect("validated at construction");
        let restriction = self.space.restriction(coupling.test, cell)?;
        let block = self
            .space
            .layout()
            .block(coupling.test)
            .expect("validated at construction");
        let local = local_symmetric_gradient_gradient(
            self.space.mesh(),
            cell,
            field.order,
            field.components,
            &self.quadrature,
        )?;
        let size = restriction.dofs.len();
        let local_input = restriction
            .dofs
            .iter()
            .map(|dof| input[block.offset + dof.0])
            .collect::<Vec<_>>();
        let mut local_output = vec![0.0; size];
        for row in 0..size {
            let mut accumulator = 0.0;
            for column in 0..size {
                accumulator += local[row * size + column] * local_input[column];
            }
            local_output[row] = coupling.scale * accumulator;
        }
        for (local_index, dof) in restriction.dofs.iter().enumerate() {
            output[block.offset + dof.0] += local_output[local_index];
        }
        Ok(())
    }

    fn apply_divergence_value(
        &self,
        cell: usize,
        coupling: &BlockCoupling,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let test_field = self
            .space
            .field(coupling.test)
            .expect("validated at construction");
        let trial_field = self
            .space
            .field(coupling.trial)
            .expect("validated at construction");
        let test_restriction = self.space.restriction(coupling.test, cell)?;
        let trial_restriction = self.space.restriction(coupling.trial, cell)?;
        let test_block = self
            .space
            .layout()
            .block(coupling.test)
            .expect("validated at construction");
        let trial_block = self
            .space
            .layout()
            .block(coupling.trial)
            .expect("validated at construction");
        let local = local_divergence_value(
            self.space.mesh(),
            cell,
            test_field.order,
            trial_field.order,
            &self.quadrature,
        )?;
        let rows = test_restriction.dofs.len();
        let columns = trial_restriction.dofs.len();
        let local_trial_input = trial_restriction
            .dofs
            .iter()
            .map(|dof| input[trial_block.offset + dof.0])
            .collect::<Vec<_>>();
        let local_test_input = test_restriction
            .dofs
            .iter()
            .map(|dof| input[test_block.offset + dof.0])
            .collect::<Vec<_>>();
        let mut test_output = vec![0.0; rows];
        let mut trial_output = vec![0.0; columns];
        for row in 0..rows {
            for column in 0..columns {
                let value = coupling.scale * local[row * columns + column];
                test_output[row] += value * local_trial_input[column];
                trial_output[column] += value * local_test_input[row];
            }
        }
        for (local_index, dof) in test_restriction.dofs.iter().enumerate() {
            output[test_block.offset + dof.0] += test_output[local_index];
        }
        for (local_index, dof) in trial_restriction.dofs.iter().enumerate() {
            output[trial_block.offset + dof.0] += trial_output[local_index];
        }
        Ok(())
    }
}

impl LinearOperator for MixedOperator {
    fn rows(&self) -> usize {
        self.dimension()
    }

    fn columns(&self) -> usize {
        self.dimension()
    }

    /// Symmetric by construction: every diagonal `GradientGradient` block is its own transpose
    /// and every `DivergenceValue` coupling contributes its mirrored block as the exact
    /// transpose of the same local matrix (see `Self::apply_divergence_value`).
    fn symmetry(&self) -> methodus::OperatorSymmetry {
        methodus::OperatorSymmetry::Symmetric
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), methodus::NumericError> {
        self.apply_action(input, output)
            .map_err(|error| methodus::NumericError::Operator {
                message: error.to_string(),
            })
    }
}

impl methodus::BlockLinearOperator for MixedOperator {
    fn block_layout(&self) -> &methodus::BlockLayout {
        &self.solver_layout
    }
}

/// A [`MixedOperator`] with essential (Dirichlet) constraints eliminated through
/// [`MixedOperator::apply_reduced_action`] -- SV2-B4's completion of the representation-only
/// [`BlockNullspaceCandidate`] machinery into an operator a MINRES-family solver can actually
/// run against.
///
/// Declared [`methodus::OperatorSymmetry::Symmetric`] whenever `constraints` carries no affine
/// dependency (the only kind [`essential_constraints_for_blocks`] produces): for a Fixed-only
/// constraint set, row `t` of the reduced action is the identity row `output[t] = input[t]`
/// (independent of every other index), and the homogeneous expansion zeroes column `t`'s
/// contribution to every other row before [`MixedOperator::apply_action`] runs -- so the reduced
/// matrix is block-diagonal between the identity on constrained indices and the free/free
/// submatrix of the already-symmetric unconstrained action, and is therefore symmetric whenever
/// that submatrix is (which it is here, `MixedOperator`'s own symmetry being unconditional by
/// construction). This is an analytic consequence of the elimination, not a proof by assembly,
/// so no separate `prove_symmetry`-style cache is needed the way `RealizationPlan` needs one for
/// its more general FC6 kernels. An affine dependency constraint (not producible by
/// [`essential_constraints_for_blocks`] today) would instead replace a target row with a
/// genuinely asymmetric constraint residual, matching `RealizationPlan`'s own convention, so
/// `symmetry()` reports `Nonsymmetric` in that case.
#[derive(Clone, Debug)]
pub struct ReducedMixedOperator {
    operator: MixedOperator,
    constraints: ConstraintSet,
}

impl ReducedMixedOperator {
    pub fn operator(&self) -> &MixedOperator {
        &self.operator
    }

    pub fn constraints(&self) -> &ConstraintSet {
        &self.constraints
    }
}

impl LinearOperator for ReducedMixedOperator {
    fn rows(&self) -> usize {
        self.operator.dimension()
    }

    fn columns(&self) -> usize {
        self.operator.dimension()
    }

    fn symmetry(&self) -> methodus::OperatorSymmetry {
        if self.constraints.has_affine_dependencies() {
            methodus::OperatorSymmetry::Nonsymmetric
        } else {
            methodus::OperatorSymmetry::Symmetric
        }
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), methodus::NumericError> {
        self.operator
            .apply_reduced_action(&self.constraints, input, output)
            .map_err(|error| methodus::NumericError::Operator {
                message: error.to_string(),
            })
    }
}

impl BlockLinearOperator for ReducedMixedOperator {
    fn block_layout(&self) -> &methodus::BlockLayout {
        self.operator.block_layout()
    }
}

fn mixed_operator_digest(space: &MixedSpace, couplings: &[BlockCoupling]) -> Digest {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        mesh: &'a Mesh,
        fields: &'a [FieldSpec],
        couplings: &'a [BlockCoupling],
    }
    let payload = Payload {
        schema: "finitum-mixed-operator/1",
        mesh: space.mesh(),
        fields: space.fields(),
        couplings,
    };
    Digest::blake3(
        &serde_json::to_vec(&payload).expect("mixed operator digest payload is serializable"),
    )
}

pub(crate) fn solver_block_layout(
    layout: &BlockLayout,
) -> Result<methodus::BlockLayout, FinitumError> {
    let specifications = layout
        .blocks()
        .iter()
        .map(|block| methodus::BlockSpec {
            name: format!("field_{}", block.symbol.0),
            length: block.extent,
            residual_scale: 1.0,
        })
        .collect();
    methodus::BlockLayout::new(specifications)
        .map_err(|error| FinitumError::InvalidRealization(error.to_string()))
}

fn local_gradient_gradient(
    mesh: &Mesh,
    cell: usize,
    order: u8,
    components: usize,
    quadrature: &[QuadraturePoint],
) -> Result<Vec<f64>, FinitumError> {
    let dimension = mesh.dimension();
    let map = AffineMap::from_cell(mesh, CellId(cell))?;
    let basis_count = simplex_basis_count(dimension, order);
    let size = basis_count * components;
    let mut local = vec![0.0; size * size];
    for point in quadrature {
        let (_, gradients) = simplex_basis(dimension, order, &point.coordinates)?;
        let mut physical = Vec::with_capacity(gradients.len());
        for gradient in &gradients {
            physical.push(map.covariant_piola(gradient)?);
        }
        let scale = point.weight * map.determinant();
        for i in 0..basis_count {
            for j in 0..basis_count {
                let dot = (0..dimension)
                    .map(|axis| physical[i][axis] * physical[j][axis])
                    .sum::<f64>();
                for component in 0..components {
                    let row = i * components + component;
                    let column = j * components + component;
                    local[row * size + column] += scale * dot;
                }
            }
        }
    }
    Ok(local)
}

/// Local `integral(sym_grad(test) : sym_grad(trial))` matrix over one dimension-vector field's
/// element, sharing [`local_gradient_gradient`]'s cell-quadrature-basis structure. Basis function
/// `(node i, component ci)` is `N_i * e_ci`, whose physical symmetric gradient is `0.5 * (g_i
/// (x) e_ci + e_ci (x) g_i)` (`g_i` the physical gradient of `N_i`, `(x)` outer product); the
/// contraction of two such tensors reduces to the closed form used below.
fn local_symmetric_gradient_gradient(
    mesh: &Mesh,
    cell: usize,
    order: u8,
    components: usize,
    quadrature: &[QuadraturePoint],
) -> Result<Vec<f64>, FinitumError> {
    let dimension = mesh.dimension();
    let map = AffineMap::from_cell(mesh, CellId(cell))?;
    let basis_count = simplex_basis_count(dimension, order);
    let size = basis_count * components;
    let mut local = vec![0.0; size * size];
    for point in quadrature {
        let (_, gradients) = simplex_basis(dimension, order, &point.coordinates)?;
        let mut physical = Vec::with_capacity(gradients.len());
        for gradient in &gradients {
            physical.push(map.covariant_piola(gradient)?);
        }
        let scale = point.weight * map.determinant();
        let dot = |left: &[f64], right: &[f64]| -> f64 {
            left.iter().zip(right).map(|(a, b)| a * b).sum()
        };
        for i in 0..basis_count {
            for ci in 0..components {
                for j in 0..basis_count {
                    for cj in 0..components {
                        // sym_grad(N_i e_ci) : sym_grad(N_j e_cj)
                        //   = 0.5 * (delta(ci,cj) * g_i.g_j + g_i[cj] * g_j[ci])
                        let contraction = 0.5
                            * (if ci == cj {
                                dot(&physical[i], &physical[j])
                            } else {
                                0.0
                            } + physical[i][cj] * physical[j][ci]);
                        let row = i * components + ci;
                        let column = j * components + cj;
                        local[row * size + column] += scale * contraction;
                    }
                }
            }
        }
    }
    Ok(local)
}

fn local_divergence_value(
    mesh: &Mesh,
    cell: usize,
    test_order: u8,
    trial_order: u8,
    quadrature: &[QuadraturePoint],
) -> Result<Vec<f64>, FinitumError> {
    let dimension = mesh.dimension();
    let map = AffineMap::from_cell(mesh, CellId(cell))?;
    let test_basis_count = simplex_basis_count(dimension, test_order);
    let trial_basis_count = simplex_basis_count(dimension, trial_order);
    let rows = test_basis_count * dimension;
    let mut local = vec![0.0; rows * trial_basis_count];
    for point in quadrature {
        let (_, test_gradients) = simplex_basis(dimension, test_order, &point.coordinates)?;
        let (trial_values, _) = simplex_basis(dimension, trial_order, &point.coordinates)?;
        let mut physical_test_gradients = Vec::with_capacity(test_gradients.len());
        for gradient in &test_gradients {
            physical_test_gradients.push(map.covariant_piola(gradient)?);
        }
        let scale = point.weight * map.determinant();
        for (i, gradient) in physical_test_gradients
            .iter()
            .enumerate()
            .take(test_basis_count)
        {
            for (component, &divergence_contribution) in gradient.iter().enumerate() {
                let row = i * dimension + component;
                for (j, &trial_value) in trial_values.iter().enumerate().take(trial_basis_count) {
                    local[row * trial_basis_count + j] +=
                        scale * divergence_contribution * trial_value;
                }
            }
        }
    }
    Ok(local)
}

/// Kinds of block nullspace mode this crate can represent. `Constant` mirrors Scientia's
/// structural `NullspaceCandidate::Constant` (GX-CONTRACTS C5.4) at the concrete realization
/// level: a scalar block determined only up to one additive constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullspaceModeKind {
    Constant,
}

/// Typed, representation-only declaration that one block of a [`BlockLayout`] carries a
/// nullspace mode -- e.g. the pure-Dirichlet pressure mode of a saddle-point system, kept
/// generic here (SV2-B1: representation only, no solver algorithm). Round-trips through
/// [`serde`]; [`Self::resolve`] reconstructs the concrete vector deterministically from a
/// [`BlockLayout`] rather than carrying it as redundant serialized data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockNullspaceCandidate {
    pub block: SymbolId,
    pub kind: NullspaceModeKind,
    pub reason: String,
}

impl BlockNullspaceCandidate {
    pub fn constant(block: SymbolId, reason: impl Into<String>) -> Self {
        Self {
            block,
            kind: NullspaceModeKind::Constant,
            reason: reason.into(),
        }
    }

    /// Resolve this declaration against a concrete [`BlockLayout`] into the exact unit-norm
    /// constant-mode vector and a `methodus::ConstantModeProjector` a downstream MINRES-family
    /// solver would consume (SV2-B6; no solve is performed here). Refuses a block absent from
    /// `layout`, or one with more than one component per entity (a `Constant` mode is
    /// scalar-per-block).
    pub fn resolve(&self, layout: &BlockLayout) -> Result<BlockNullspaceMode, FinitumError> {
        let field = layout.block(self.block).ok_or_else(|| {
            FinitumError::InvalidRealization(format!(
                "nullspace candidate names block {} which is absent from the layout",
                self.block
            ))
        })?;
        let NullspaceModeKind::Constant = self.kind;
        if field.component_count != 1 {
            return Err(FinitumError::UnsupportedRealization(format!(
                "a constant nullspace mode requires a scalar (single-component) block; block \
                 {} has {} components",
                self.block, field.component_count
            )));
        }
        let projector = ConstantModeProjector::new(layout.extent(), field.offset, field.extent)
            .map_err(|error| FinitumError::InvalidRealization(error.to_string()))?;
        let value = 1.0 / (field.extent as f64).sqrt();
        let mut vector = vec![0.0; layout.extent()];
        vector[field.offset..field.offset + field.extent].fill(value);
        Ok(BlockNullspaceMode {
            candidate: self.clone(),
            projector,
            vector,
        })
    }
}

/// A [`BlockNullspaceCandidate`] resolved against a concrete [`BlockLayout`]: the exact
/// unit-Euclidean-norm constant-mode vector, and the `methodus::ConstantModeProjector` a
/// downstream solver would use to remove it.
#[derive(Clone, Debug)]
pub struct BlockNullspaceMode {
    candidate: BlockNullspaceCandidate,
    projector: ConstantModeProjector,
    vector: Vec<f64>,
}

impl BlockNullspaceMode {
    pub fn candidate(&self) -> &BlockNullspaceCandidate {
        &self.candidate
    }

    pub fn projector(&self) -> &ConstantModeProjector {
        &self.projector
    }

    pub fn vector(&self) -> &[f64] {
        &self.vector
    }

    /// Representation-level evidence, not a solver action: verify this candidate's vector lies
    /// in the kernel of `operator` within `tolerance` (`||operator * vector|| <= tolerance`).
    pub fn verify_in_kernel(
        &self,
        operator: &impl LinearOperator,
        tolerance: f64,
    ) -> Result<bool, FinitumError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(FinitumError::InvalidRealization(
                "nullspace verification tolerance must be finite and non-negative".into(),
            ));
        }
        if operator.columns() != self.vector.len() || operator.rows() != self.vector.len() {
            return Err(FinitumError::InvalidRealization(format!(
                "nullspace vector has length {}, operator is {} by {}",
                self.vector.len(),
                operator.rows(),
                operator.columns()
            )));
        }
        let mut output = vec![0.0; operator.rows()];
        operator
            .apply(&EvaluationContext::default(), &self.vector, &mut output)
            .map_err(|error| FinitumError::InvalidRealization(error.to_string()))?;
        let norm = output.iter().map(|value| value * value).sum::<f64>().sqrt();
        Ok(norm <= tolerance)
    }
}
