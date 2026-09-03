//! Interface / interior-facet measure realization over shared facets (SV2-B2, pulled forward
//! as SC-W1 item 3 of `sinbad/ARCHITECTURE.md` §8/§12), binding Malleus **facet-pair kernels**
//! through their side roles.
//!
//! A [`InterfaceOperator`] integrates one or more [`InterfaceKernel`]s over an
//! [`InterfaceMeasure`] -- a set of two-sided facets with an explicit minus/plus orientation --
//! for fields living in one realization plan's [`BlockLayout`] ([`InterfaceSpace`]). Every
//! kernel operand is bound by role: a `Cell { side }` operand is a field's trace (value or
//! gradient) on that side, or a side-owned residual contribution scattered through that side's
//! trace basis; a `Facet { Even }` operand is facet-native even data (coordinates, measure, a
//! constant); the `Facet { Odd }` operand is the unit normal oriented **from the minus cell to
//! the plus cell**, Malleus's `FACET_NORMAL_CONVENTION`. Finitum decides which cell is minus
//! ([`InterfaceMeasure::interior`], [`InterfaceMeasure::between`], [`InterfaceMeasure::flipped`]);
//! a well-formed kernel is covariant under that choice, and [`InterfaceOperator::
//! swap_symmetry_receipt`] proves it at the realized traces by executing Malleus's own
//! `check_facet_swap_symmetry`.
//!
//! Residual, JVP, and VJP execute the kernel and its `differentiate_facet_pair` products at
//! every facet quadrature point (3-point Gauss on segments, degree-4 rule on triangles); the
//! JVP/VJP pair is an exact transpose by construction (linear gather, Malleus adjoint kernels,
//! linear scatter). Time-derivative traces, curved facets, and mesh dimension 1 are refused
//! typed. This module composes *within* one realization plan (the same `BlockLayout` as a
//! `SystemOperator`); composition across plans is Krasis's.

use crate::element::{gauss_legendre_unit_interval, simplex_basis, triangle_degree4_quadrature};
use crate::realization::{component_count, execute, validate_finite};
use crate::space::{DofMap, ElementRestriction, cell_constant_dof_map};
use crate::{
    AffineMap, BlockLayout, CellId, DofId, FacetTopology, FinitumError, Mesh, OrientedFacetPair,
    quadratic_simplex_dof_map, vector_nodal_dof_map,
};
use malleus::{
    DerivativeMode, DerivativeRequest, Executable, FacetOperandRole, FacetPairDerivativeProduct,
    FacetPairKernel, FacetSide, FacetSwapError, OperandId, SwapParity, ValidatedFacetPairKernel,
    check_facet_swap_symmetry, differentiate_facet_pair, facet_pair_digest, validate,
    validate_facet_pair,
};
use methodus::{
    EvaluationContext, LinearOperator, NumericError, OperatorSymmetry, TransposableOperator,
};
use scientia::{Digest, SymbolId};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Schema of [`InterfaceOperator::digest`].
pub const INTERFACE_REALIZATION_SCHEMA: &str = "finitum-interface-realization/1";

/// One field whose traces an interface kernel may read or write: Lagrange order 0, 1, or 2,
/// scalar or `components`-vector, continuous (shared nodes) or discontinuous (cell-owned
/// nodes; order 0 is always discontinuous, order 2 discontinuous is not realized).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TraceFieldSpec {
    pub symbol: SymbolId,
    pub order: u8,
    pub components: usize,
    pub continuous: bool,
}

impl TraceFieldSpec {
    fn validate(&self) -> Result<(), FinitumError> {
        if self.components == 0 {
            return Err(FinitumError::InvalidRealization(format!(
                "trace field {} needs at least one component",
                self.symbol
            )));
        }
        match (self.order, self.continuous) {
            (0, false) | (1, _) | (2, true) => Ok(()),
            (0, true) => Err(FinitumError::UnsupportedRealization(format!(
                "trace field {} of order 0 is cell-constant and cannot be continuous",
                self.symbol
            ))),
            (order, continuous) => Err(FinitumError::UnsupportedRealization(format!(
                "trace field {} requires order {order} (continuous = {continuous}); interface \
                 realization admits P0, P1 (continuous or discontinuous), and continuous P2",
                self.symbol
            ))),
        }
    }

    fn dof_map(&self, mesh: &Mesh) -> Result<DofMap, FinitumError> {
        match (self.order, self.continuous) {
            (0, _) => {
                let cells = cell_constant_dof_map(mesh)?;
                if self.components == 1 {
                    Ok(cells)
                } else {
                    discontinuous_dof_map(mesh, 1, self.components)
                }
            }
            (1, true) => vector_nodal_dof_map(mesh, self.components),
            (2, true) => quadratic_simplex_dof_map(mesh, self.components),
            (1, false) => discontinuous_dof_map(mesh, mesh.dimension() + 1, self.components),
            _ => unreachable!("validated by TraceFieldSpec::validate"),
        }
    }
}

/// Cell-owned nodes: cell `c` owns `nodes_per_cell` consecutive nodes, node-major over
/// `components` (the same local layout `vector_nodal_dof_map` uses).
fn discontinuous_dof_map(
    mesh: &Mesh,
    nodes_per_cell: usize,
    components: usize,
) -> Result<DofMap, FinitumError> {
    let cells = mesh.cells().len();
    let restrictions = (0..cells)
        .map(|cell| ElementRestriction {
            dofs: (0..nodes_per_cell)
                .flat_map(|node| {
                    (0..components).map(move |component| {
                        DofId((cell * nodes_per_cell + node) * components + component)
                    })
                })
                .collect(),
        })
        .collect();
    DofMap::new(cells * nodes_per_cell * components, restrictions)
}

/// The fields an interface realization acts on, laid out end-to-end by one [`BlockLayout`].
#[derive(Clone, Debug)]
pub struct InterfaceSpace {
    mesh: Mesh,
    fields: Vec<TraceFieldSpec>,
    layout: BlockLayout,
    dof_maps: BTreeMap<SymbolId, DofMap>,
}

impl InterfaceSpace {
    /// Builds the layout from the fields (one block per field in declaration order).
    pub fn new(mesh: Mesh, fields: Vec<TraceFieldSpec>) -> Result<Self, FinitumError> {
        let mut specifications = Vec::with_capacity(fields.len());
        let mut dof_maps = BTreeMap::new();
        for field in &fields {
            field.validate()?;
            let dofs = field.dof_map(&mesh)?;
            specifications.push((
                field.symbol,
                dofs.dof_count() / field.components,
                field.components,
            ));
            dof_maps.insert(field.symbol, dofs);
        }
        let layout = BlockLayout::new(specifications)?;
        Ok(Self {
            mesh,
            fields,
            layout,
            dof_maps,
        })
    }

    /// Binds the fields to an existing layout (a `SystemOperator`'s, so interface and cell
    /// actions add on the same vector); every field must own a block of the matching
    /// entity/component counts, and the layout may carry further blocks the interface leaves
    /// untouched.
    pub fn over_layout(
        mesh: Mesh,
        fields: Vec<TraceFieldSpec>,
        layout: BlockLayout,
    ) -> Result<Self, FinitumError> {
        let mut dof_maps = BTreeMap::new();
        for field in &fields {
            field.validate()?;
            let dofs = field.dof_map(&mesh)?;
            let block = layout.block(field.symbol).ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "layout has no (unambiguous) block for trace field {}",
                    field.symbol
                ))
            })?;
            if block.extent != dofs.dof_count() || block.component_count != field.components {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "trace field {} needs {} DOFs in {} components, layout block has {} in {}",
                    field.symbol,
                    dofs.dof_count(),
                    field.components,
                    block.extent,
                    block.component_count
                )));
            }
            if dof_maps.insert(field.symbol, dofs).is_some() {
                return Err(FinitumError::InvalidRealization(format!(
                    "trace field {} is declared more than once",
                    field.symbol
                )));
            }
        }
        Ok(Self {
            mesh,
            fields,
            layout,
            dof_maps,
        })
    }

    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub fn fields(&self) -> &[TraceFieldSpec] {
        &self.fields
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    pub fn dof_map(&self, symbol: SymbolId) -> Option<&DofMap> {
        self.dof_maps.get(&symbol)
    }

    fn field(&self, symbol: SymbolId) -> Result<&TraceFieldSpec, FinitumError> {
        self.fields
            .iter()
            .find(|field| field.symbol == symbol)
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "interface space has no trace field {symbol}"
                ))
            })
    }
}

/// A set of two-sided facets with an explicit minus/plus orientation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterfaceMeasure {
    pairs: Vec<OrientedFacetPair>,
}

impl InterfaceMeasure {
    /// Every interior facet, with the topology's canonical first incidence as the minus cell.
    pub fn interior(topology: &FacetTopology) -> Result<Self, FinitumError> {
        let pairs = topology
            .interior()
            .map(|facet| topology.oriented_pair(facet.id, facet.minus().cell))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_pairs(pairs)
    }

    /// The interface between `minus_cells` and the rest of the mesh: every interior facet
    /// with exactly one incident cell in `minus_cells`, oriented from that cell outward.
    pub fn between(
        topology: &FacetTopology,
        minus_cells: &BTreeSet<CellId>,
    ) -> Result<Self, FinitumError> {
        let mut pairs = Vec::new();
        for facet in topology.interior() {
            let minus = facet.minus();
            let plus = facet.plus().expect("interior facet");
            match (
                minus_cells.contains(&minus.cell),
                minus_cells.contains(&plus.cell),
            ) {
                (true, false) => pairs.push(topology.oriented_pair(facet.id, minus.cell)?),
                (false, true) => pairs.push(topology.oriented_pair(facet.id, plus.cell)?),
                _ => {}
            }
        }
        Self::from_pairs(pairs)
    }

    /// An explicit facet list; every facet at most once.
    pub fn from_pairs(pairs: Vec<OrientedFacetPair>) -> Result<Self, FinitumError> {
        if pairs.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "interface measure must contain at least one facet".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for pair in &pairs {
            if !seen.insert(pair.facet) {
                return Err(FinitumError::InvalidRealization(format!(
                    "facet {} appears more than once in the interface measure",
                    pair.facet.0
                )));
            }
            if pair.minus.cell == pair.plus.cell {
                return Err(FinitumError::InvalidRealization(format!(
                    "facet {} pairs cell {} with itself",
                    pair.facet.0, pair.minus.cell.0
                )));
            }
        }
        Ok(Self { pairs })
    }

    /// The same facets with minus and plus exchanged.
    pub fn flipped(&self) -> Self {
        Self {
            pairs: self
                .pairs
                .iter()
                .map(|pair| OrientedFacetPair {
                    facet: pair.facet,
                    minus: pair.plus,
                    plus: pair.minus,
                })
                .collect(),
        }
    }

    pub fn pairs(&self) -> &[OrientedFacetPair] {
        &self.pairs
    }
}

/// Which trace of a field an operand reads or a residual contribution scatters through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceEvaluation {
    /// The field value on the side: `components` entries.
    Value,
    /// The physical gradient on the side: `components x dimension` entries, row-major.
    Gradient,
}

/// How one operand of a facet-pair kernel is bound to the realization.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterfaceOperand {
    /// A field's trace on one side (role must be `Cell { side }`, readable).
    Trace {
        field: SymbolId,
        side: FacetSide,
        evaluation: TraceEvaluation,
    },
    /// The unit normal from the minus cell to the plus cell (role `Facet { Odd }`).
    Normal,
    /// Physical coordinates of the quadrature point (role `Facet { Even }`).
    Coordinates,
    /// The physical measure of the facet (role `Facet { Even }`, scalar).
    FacetMeasure,
    /// A caller-supplied constant (role `Facet { Even }`).
    Constant { values: Vec<f64> },
    /// A side-owned residual contribution scattered through that side's trace basis (role
    /// `Cell { side }`, writable).
    Residual {
        field: SymbolId,
        side: FacetSide,
        evaluation: TraceEvaluation,
    },
    /// A writable operand the realization does not scatter (an auxiliary output).
    Ignored,
}

/// A validated facet-pair kernel with every operand bound.
#[derive(Clone, Debug)]
pub struct InterfaceKernel {
    kernel: FacetPairKernel,
    validated: ValidatedFacetPairKernel,
    operands: Vec<InterfaceOperand>,
    digest: Digest,
}

impl InterfaceKernel {
    /// Validates the kernel as a Malleus facet-pair kernel and every binding against its
    /// operand's role, access, and shape (shapes are checked at realization time against
    /// the space's dimension and field components).
    pub fn new(
        kernel: FacetPairKernel,
        operands: Vec<InterfaceOperand>,
    ) -> Result<Self, FinitumError> {
        let validated = validate_facet_pair(kernel.clone())
            .map_err(|error| FinitumError::KernelValidation(error.to_string()))?;
        let definition = validated.kernel().as_kernel();
        if operands.len() != definition.operands.len() {
            return Err(FinitumError::InvalidRealization(format!(
                "interface kernel `{}` has {} operands, {} bindings supplied",
                definition.name,
                definition.operands.len(),
                operands.len()
            )));
        }
        let mut residuals = 0usize;
        for (index, (binding, role)) in operands.iter().zip(validated.roles()).enumerate() {
            let access = definition.operands[index].access;
            let expect_side = |side: FacetSide| match role {
                FacetOperandRole::Cell { side: own, .. } if *own == side => Ok(()),
                other => Err(FinitumError::InvalidRealization(format!(
                    "interface kernel `{}` operand {index} is bound to the {side:?} side but \
                     carries role {other:?}",
                    definition.name
                ))),
            };
            let expect_parity = |parity: SwapParity| match role {
                FacetOperandRole::Facet { parity: own } if *own == parity => Ok(()),
                other => Err(FinitumError::InvalidRealization(format!(
                    "interface kernel `{}` operand {index} is bound to facet data of parity \
                     {parity:?} but carries role {other:?}",
                    definition.name
                ))),
            };
            let expect_readable = || {
                if access.can_read() && !access.can_write() {
                    Ok(())
                } else {
                    Err(FinitumError::InvalidRealization(format!(
                        "interface kernel `{}` operand {index} is bound as an input but has \
                         access {access:?}",
                        definition.name
                    )))
                }
            };
            let expect_writable = || {
                if access.can_write() {
                    Ok(())
                } else {
                    Err(FinitumError::InvalidRealization(format!(
                        "interface kernel `{}` operand {index} is bound as an output but has \
                         access {access:?}",
                        definition.name
                    )))
                }
            };
            match binding {
                InterfaceOperand::Trace { side, .. } => {
                    expect_side(*side)?;
                    expect_readable()?;
                }
                InterfaceOperand::Normal => {
                    expect_parity(SwapParity::Odd)?;
                    expect_readable()?;
                }
                InterfaceOperand::Coordinates
                | InterfaceOperand::FacetMeasure
                | InterfaceOperand::Constant { .. } => {
                    expect_parity(SwapParity::Even)?;
                    expect_readable()?;
                }
                InterfaceOperand::Residual { side, .. } => {
                    expect_side(*side)?;
                    expect_writable()?;
                    residuals += 1;
                }
                InterfaceOperand::Ignored => expect_writable()?,
            }
        }
        if residuals == 0 {
            return Err(FinitumError::InvalidRealization(format!(
                "interface kernel `{}` binds no residual output",
                definition.name
            )));
        }
        let malleus_digest = facet_pair_digest(&kernel);
        let digest = Digest {
            algorithm: malleus_digest.algorithm,
            hex: malleus_digest.hex,
        };
        Ok(Self {
            kernel,
            validated,
            operands,
            digest,
        })
    }

    pub fn kernel(&self) -> &FacetPairKernel {
        &self.kernel
    }

    pub fn operands(&self) -> &[InterfaceOperand] {
        &self.operands
    }

    /// Malleus's `facet_pair_digest` of the bound kernel.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }
}

#[derive(Debug)]
struct BoundKernel {
    kernel: InterfaceKernel,
    primal: Executable,
    /// JVP/VJP products over every `Trace` input and every `Residual` output; `None` when
    /// the kernel reads no trace (its derivative is identically zero).
    jvp: Option<(FacetPairDerivativeProduct, Executable)>,
    vjp: Option<(FacetPairDerivativeProduct, Executable)>,
}

#[derive(Clone, Debug)]
struct FacetPoint {
    physical: Vec<f64>,
    minus_reference: Vec<f64>,
    plus_reference: Vec<f64>,
    scale: f64,
}

#[derive(Clone, Debug)]
struct FacetQuadrature {
    pair: OrientedFacetPair,
    points: Vec<FacetPoint>,
    normal: Vec<f64>,
    measure: f64,
    minus_map: AffineMap,
    plus_map: AffineMap,
}

/// Interface-measure operator: interface kernels integrated over an interface measure for
/// the traces of an [`InterfaceSpace`]'s fields.
#[derive(Debug)]
pub struct InterfaceOperator {
    space: InterfaceSpace,
    measure: InterfaceMeasure,
    kernels: Vec<BoundKernel>,
    facets: Vec<FacetQuadrature>,
    digest: Digest,
}

/// Outcome of [`InterfaceOperator::swap_symmetry_receipt`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InterfaceSwapReceipt {
    pub schema: &'static str,
    pub kernels: Vec<InterfaceSwapKernelReceipt>,
    pub facets_checked: usize,
    pub max_absolute: f64,
    pub within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InterfaceSwapKernelReceipt {
    pub kernel: Digest,
    pub max_absolute: f64,
    pub within_tolerance: bool,
}

#[derive(Clone, Copy)]
enum Action<'a> {
    Primal,
    Jvp { direction: &'a [f64] },
    Vjp { adjoint: &'a [f64] },
}

impl InterfaceOperator {
    pub fn new(
        space: InterfaceSpace,
        measure: InterfaceMeasure,
        kernels: Vec<InterfaceKernel>,
    ) -> Result<Self, FinitumError> {
        let dimension = space.mesh.dimension();
        if !(2..=3).contains(&dimension) {
            return Err(FinitumError::UnsupportedRealization(format!(
                "interface realization is supported for mesh dimension 2 or 3, got {dimension}"
            )));
        }
        if kernels.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "interface operator needs at least one kernel".into(),
            ));
        }
        let topology = FacetTopology::from_mesh(&space.mesh)?;
        let facets = measure
            .pairs
            .iter()
            .map(|pair| facet_quadrature(&space.mesh, &topology, *pair))
            .collect::<Result<Vec<_>, _>>()?;
        let mut bound = Vec::with_capacity(kernels.len());
        for kernel in kernels {
            validate_operand_shapes(&space, &kernel)?;
            let primal = Executable::reference(kernel.validated.kernel().clone());
            let independent = kernel
                .operands
                .iter()
                .enumerate()
                .filter(|(_, operand)| matches!(operand, InterfaceOperand::Trace { .. }))
                .map(|(index, _)| OperandId::new(index))
                .collect::<Vec<_>>();
            let dependent = kernel
                .operands
                .iter()
                .enumerate()
                .filter(|(_, operand)| matches!(operand, InterfaceOperand::Residual { .. }))
                .map(|(index, _)| OperandId::new(index))
                .collect::<Vec<_>>();
            let derive = |mode: DerivativeMode| -> Result<
                Option<(FacetPairDerivativeProduct, Executable)>,
                FinitumError,
            > {
                if independent.is_empty() {
                    return Ok(None);
                }
                let product = differentiate_facet_pair(
                    &kernel.kernel,
                    &DerivativeRequest {
                        mode,
                        independent_operands: independent.clone(),
                        dependent_operands: dependent.clone(),
                    },
                )
                .map_err(|error| FinitumError::KernelValidation(error.to_string()))?;
                let validated = validate(product.product.kernel.clone())
                    .map_err(|error| FinitumError::KernelValidation(error.to_string()))?;
                Ok(Some((product, Executable::reference(validated))))
            };
            let jvp = derive(DerivativeMode::Jvp)?;
            let vjp = derive(DerivativeMode::Vjp)?;
            bound.push(BoundKernel {
                kernel,
                primal,
                jvp,
                vjp,
            });
        }
        let digest = interface_digest(&space, &measure, &bound);
        Ok(Self {
            space,
            measure,
            kernels: bound,
            facets,
            digest,
        })
    }

    pub fn space(&self) -> &InterfaceSpace {
        &self.space
    }

    pub fn measure(&self) -> &InterfaceMeasure {
        &self.measure
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.space.layout
    }

    pub fn dimension(&self) -> usize {
        self.space.layout.extent()
    }

    pub fn kernels(&self) -> impl Iterator<Item = &InterfaceKernel> {
        self.kernels.iter().map(|bound| &bound.kernel)
    }

    /// Content-addressed identity over the space, measure (facets and orientation), and
    /// kernels with their bindings.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// The residual `R(u)` of every kernel over the measure, overwriting `output`.
    pub fn residual(&self, state: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.validate_vector("interface state", state)?;
        self.validate_vector("interface output", output)?;
        output.fill(0.0);
        self.apply(state, Action::Primal, output)?;
        validate_finite("interface residual", output)
    }

    /// `dR/du * direction` at `state`, overwriting `output`.
    pub fn jacobian_vector_product(
        &self,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_vector("interface state", state)?;
        self.validate_vector("interface direction", direction)?;
        self.validate_vector("interface output", output)?;
        output.fill(0.0);
        self.apply(state, Action::Jvp { direction }, output)?;
        validate_finite("interface JVP", output)
    }

    /// `(dR/du)^T * adjoint` at `state`, overwriting `output`: the exact transpose of
    /// [`Self::jacobian_vector_product`].
    pub fn vector_jacobian_product(
        &self,
        state: &[f64],
        adjoint: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.validate_vector("interface state", state)?;
        self.validate_vector("interface adjoint", adjoint)?;
        self.validate_vector("interface output", output)?;
        output.fill(0.0);
        self.apply(state, Action::Vjp { adjoint }, output)?;
        validate_finite("interface VJP", output)
    }

    /// The zero-state linear view: `jacobian_vector_product(0, input)`.
    pub fn apply_action(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        let zero = vec![0.0; self.dimension()];
        self.jacobian_vector_product(&zero, input, output)
    }

    /// Canonical CSR assembly of the zero-state linear view by unit-column probing.
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

    /// Proves, by executing Malleus's `check_facet_swap_symmetry` at the first quadrature
    /// point of every facet with the traces realized from `state`, that every kernel is
    /// covariant under exchanging the minus and plus sides -- the receipt that the global
    /// action does not depend on Finitum's orientation choice. Refused typed when a kernel
    /// carries a one-sided (partnerless) cell operand, which Malleus cannot exchange.
    pub fn swap_symmetry_receipt(
        &self,
        state: &[f64],
        tolerance: f64,
    ) -> Result<InterfaceSwapReceipt, FinitumError> {
        self.validate_vector("interface state", state)?;
        let mut kernels = Vec::with_capacity(self.kernels.len());
        let mut max_absolute: f64 = 0.0;
        for bound in &self.kernels {
            let mut kernel_max: f64 = 0.0;
            for facet in &self.facets {
                let point = &facet.points[0];
                let buffers = self.primal_buffers(bound, facet, point, state)?;
                let report =
                    check_facet_swap_symmetry(&bound.kernel.validated, &buffers, tolerance)
                        .map_err(|error| match error {
                            FacetSwapError::UnpairedOperand(index) => {
                                FinitumError::UnsupportedRealization(format!(
                                    "interface kernel `{}` operand {index} is one-sided (no \
                                 partner); the swap receipt cannot exchange its sides",
                                    bound.kernel.kernel.kernel.name
                                ))
                            }
                            other => FinitumError::KernelExecution(other.to_string()),
                        })?;
                kernel_max = kernel_max.max(report.max_absolute);
            }
            max_absolute = max_absolute.max(kernel_max);
            kernels.push(InterfaceSwapKernelReceipt {
                kernel: bound.kernel.digest.clone(),
                max_absolute: kernel_max,
                within_tolerance: kernel_max.is_finite() && kernel_max <= tolerance,
            });
        }
        Ok(InterfaceSwapReceipt {
            schema: "finitum-interface-swap-receipt/1",
            facets_checked: self.facets.len(),
            within_tolerance: kernels.iter().all(|kernel| kernel.within_tolerance),
            max_absolute,
            kernels,
        })
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

    fn apply(
        &self,
        state: &[f64],
        action: Action<'_>,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        for bound in &self.kernels {
            for facet in &self.facets {
                for point in &facet.points {
                    self.apply_point(bound, facet, point, state, action, output)?;
                }
            }
        }
        Ok(())
    }

    /// Gathers every side's local values of `vector` for `field`.
    fn local_values(
        &self,
        field: SymbolId,
        cell: CellId,
        vector: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let block = self.space.layout.block(field).expect("validated field");
        let restriction = &self.space.dof_maps[&field].restrictions()[cell.0];
        Ok(restriction
            .dofs
            .iter()
            .map(|dof| vector[block.offset + dof.0])
            .collect())
    }

    fn scatter_local(&self, field: SymbolId, cell: CellId, local: &[f64], output: &mut [f64]) {
        let block = self.space.layout.block(field).expect("validated field");
        let restriction = &self.space.dof_maps[&field].restrictions()[cell.0];
        for (index, dof) in restriction.dofs.iter().enumerate() {
            output[block.offset + dof.0] += local[index];
        }
    }

    /// Basis values and physical gradients of `field` on `side` at the point.
    fn side_basis(
        &self,
        field: &TraceFieldSpec,
        facet: &FacetQuadrature,
        point: &FacetPoint,
        side: FacetSide,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), FinitumError> {
        let dimension = self.space.mesh.dimension();
        let (reference, map) = match side {
            FacetSide::Minus => (&point.minus_reference, &facet.minus_map),
            FacetSide::Plus => (&point.plus_reference, &facet.plus_map),
        };
        if field.order == 0 {
            return Ok((vec![1.0], vec![vec![0.0; dimension]]));
        }
        let (values, reference_gradients) = simplex_basis(dimension, field.order, reference)?;
        let gradients = reference_gradients
            .iter()
            .map(|gradient| map.covariant_piola(gradient))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((values, gradients))
    }

    fn evaluate_trace(
        &self,
        field: &TraceFieldSpec,
        evaluation: TraceEvaluation,
        basis: &(Vec<f64>, Vec<Vec<f64>>),
        local: &[f64],
    ) -> Vec<f64> {
        let components = field.components;
        let dimension = self.space.mesh.dimension();
        match evaluation {
            TraceEvaluation::Value => (0..components)
                .map(|component| {
                    basis
                        .0
                        .iter()
                        .enumerate()
                        .map(|(node, phi)| phi * local[node * components + component])
                        .sum()
                })
                .collect(),
            TraceEvaluation::Gradient => {
                let mut values = vec![0.0; components * dimension];
                for (node, gradient) in basis.1.iter().enumerate() {
                    for component in 0..components {
                        let coefficient = local[node * components + component];
                        for axis in 0..dimension {
                            values[component * dimension + axis] += gradient[axis] * coefficient;
                        }
                    }
                }
                values
            }
        }
    }

    /// Transpose of [`Self::evaluate_trace`]: accumulates `scale * basis^T cotangent` into a
    /// node-major local vector.
    fn scatter_trace(
        &self,
        field: &TraceFieldSpec,
        evaluation: TraceEvaluation,
        basis: &(Vec<f64>, Vec<Vec<f64>>),
        cotangent: &[f64],
        scale: f64,
        local: &mut [f64],
    ) {
        let components = field.components;
        let dimension = self.space.mesh.dimension();
        match evaluation {
            TraceEvaluation::Value => {
                for (node, phi) in basis.0.iter().enumerate() {
                    for component in 0..components {
                        local[node * components + component] += scale * phi * cotangent[component];
                    }
                }
            }
            TraceEvaluation::Gradient => {
                for (node, gradient) in basis.1.iter().enumerate() {
                    for component in 0..components {
                        let mut sum = 0.0;
                        for axis in 0..dimension {
                            sum += gradient[axis] * cotangent[component * dimension + axis];
                        }
                        local[node * components + component] += scale * sum;
                    }
                }
            }
        }
    }

    fn side_cell(facet: &FacetQuadrature, side: FacetSide) -> CellId {
        match side {
            FacetSide::Minus => facet.pair.minus.cell,
            FacetSide::Plus => facet.pair.plus.cell,
        }
    }

    /// The read-operand values of the primal kernel at one point (and zero for every
    /// writable operand), in operand order -- the buffers Malleus's swap check consumes.
    fn primal_buffers(
        &self,
        bound: &BoundKernel,
        facet: &FacetQuadrature,
        point: &FacetPoint,
        state: &[f64],
    ) -> Result<Vec<Vec<f64>>, FinitumError> {
        let definition = bound.kernel.validated.kernel().as_kernel();
        let mut buffers = Vec::with_capacity(definition.operands.len());
        for (index, operand) in bound.kernel.operands.iter().enumerate() {
            let values = match operand {
                InterfaceOperand::Trace {
                    field,
                    side,
                    evaluation,
                } => {
                    let spec = self.space.field(*field)?;
                    let cell = Self::side_cell(facet, *side);
                    let basis = self.side_basis(spec, facet, point, *side)?;
                    let local = self.local_values(*field, cell, state)?;
                    self.evaluate_trace(spec, *evaluation, &basis, &local)
                }
                InterfaceOperand::Normal => facet.normal.clone(),
                InterfaceOperand::Coordinates => point.physical.clone(),
                InterfaceOperand::FacetMeasure => vec![facet.measure],
                InterfaceOperand::Constant { values } => values.clone(),
                InterfaceOperand::Residual { .. } | InterfaceOperand::Ignored => {
                    vec![0.0; component_count(&definition.operands[index].shape)?]
                }
            };
            buffers.push(values);
        }
        Ok(buffers)
    }

    fn apply_point(
        &self,
        bound: &BoundKernel,
        facet: &FacetQuadrature,
        point: &FacetPoint,
        state: &[f64],
        action: Action<'_>,
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let primal = self.primal_buffers(bound, facet, point, state)?;
        let definition = bound.kernel.validated.kernel().as_kernel();
        let readable = |index: usize| definition.operands[index].access.can_read();
        match action {
            Action::Primal => {
                let values = primal
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| readable(*index))
                    .map(|(index, values)| (OperandId::new(index), values.clone()))
                    .collect::<BTreeMap<_, _>>();
                let buffers = execute(&bound.primal, &values)?;
                for (index, operand) in bound.kernel.operands.iter().enumerate() {
                    if let InterfaceOperand::Residual {
                        field,
                        side,
                        evaluation,
                    } = operand
                    {
                        self.scatter_output(
                            facet,
                            point,
                            *field,
                            *side,
                            *evaluation,
                            &buffers[index][..component_count(&definition.operands[index].shape)?],
                            output,
                        )?;
                    }
                }
            }
            Action::Jvp { direction } => {
                let Some((product, executable)) = &bound.jvp else {
                    return Ok(());
                };
                let mut values = BTreeMap::new();
                for pair in &product.product.primal_operands {
                    if readable(pair.primal.index()) {
                        values.insert(pair.derivative, primal[pair.primal.index()].clone());
                    }
                }
                for pair in &product.product.independent_operands {
                    let InterfaceOperand::Trace {
                        field,
                        side,
                        evaluation,
                    } = &bound.kernel.operands[pair.primal.index()]
                    else {
                        unreachable!("independent operands are traces");
                    };
                    let spec = self.space.field(*field)?;
                    let cell = Self::side_cell(facet, *side);
                    let basis = self.side_basis(spec, facet, point, *side)?;
                    let local = self.local_values(*field, cell, direction)?;
                    values.insert(
                        pair.derivative,
                        self.evaluate_trace(spec, *evaluation, &basis, &local),
                    );
                }
                let buffers = execute(executable, &values)?;
                let derivative_kernel = executable.kernel().as_kernel();
                for pair in &product.product.dependent_operands {
                    let InterfaceOperand::Residual {
                        field,
                        side,
                        evaluation,
                    } = &bound.kernel.operands[pair.primal.index()]
                    else {
                        unreachable!("dependent operands are residuals");
                    };
                    let count = component_count(
                        &derivative_kernel.operands[pair.derivative.index()].shape,
                    )?;
                    self.scatter_output(
                        facet,
                        point,
                        *field,
                        *side,
                        *evaluation,
                        &buffers[pair.derivative.index()][..count],
                        output,
                    )?;
                }
            }
            Action::Vjp { adjoint } => {
                let Some((product, executable)) = &bound.vjp else {
                    return Ok(());
                };
                let mut values = BTreeMap::new();
                for pair in &product.product.primal_operands {
                    if readable(pair.primal.index()) {
                        values.insert(pair.derivative, primal[pair.primal.index()].clone());
                    }
                }
                for pair in &product.product.dependent_operands {
                    let InterfaceOperand::Residual {
                        field,
                        side,
                        evaluation,
                    } = &bound.kernel.operands[pair.primal.index()]
                    else {
                        unreachable!("dependent operands are residuals");
                    };
                    let spec = self.space.field(*field)?;
                    let cell = Self::side_cell(facet, *side);
                    let basis = self.side_basis(spec, facet, point, *side)?;
                    let local = self.local_values(*field, cell, adjoint)?;
                    values.insert(
                        pair.derivative,
                        self.evaluate_trace(spec, *evaluation, &basis, &local),
                    );
                }
                let buffers = execute(executable, &values)?;
                let derivative_kernel = executable.kernel().as_kernel();
                for pair in &product.product.independent_operands {
                    let InterfaceOperand::Trace {
                        field,
                        side,
                        evaluation,
                    } = &bound.kernel.operands[pair.primal.index()]
                    else {
                        unreachable!("independent operands are traces");
                    };
                    let count = component_count(
                        &derivative_kernel.operands[pair.derivative.index()].shape,
                    )?;
                    self.scatter_output(
                        facet,
                        point,
                        *field,
                        *side,
                        *evaluation,
                        &buffers[pair.derivative.index()][..count],
                        output,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Scatters a point output of a side-owned field through that side's trace basis,
    /// scaled by the quadrature weight.
    #[allow(clippy::too_many_arguments)]
    fn scatter_output(
        &self,
        facet: &FacetQuadrature,
        point: &FacetPoint,
        field: SymbolId,
        side: FacetSide,
        evaluation: TraceEvaluation,
        values: &[f64],
        output: &mut [f64],
    ) -> Result<(), FinitumError> {
        let spec = self.space.field(field)?;
        let cell = Self::side_cell(facet, side);
        let basis = self.side_basis(spec, facet, point, side)?;
        let mut local = vec![0.0; basis.0.len() * spec.components];
        self.scatter_trace(spec, evaluation, &basis, values, point.scale, &mut local);
        self.scatter_local(field, cell, &local, output);
        Ok(())
    }
}

impl LinearOperator for InterfaceOperator {
    fn rows(&self) -> usize {
        self.dimension()
    }

    fn columns(&self) -> usize {
        self.dimension()
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
        self.apply_action(input, output)
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

impl TransposableOperator for InterfaceOperator {
    fn apply_transpose(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; self.dimension()];
        self.vector_jacobian_product(&zero, input, output)
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

fn validate_operand_shapes(
    space: &InterfaceSpace,
    kernel: &InterfaceKernel,
) -> Result<(), FinitumError> {
    let dimension = space.mesh.dimension();
    let definition = kernel.validated.kernel().as_kernel();
    for (index, operand) in kernel.operands.iter().enumerate() {
        let actual = component_count(&definition.operands[index].shape)?;
        let expected = match operand {
            InterfaceOperand::Trace {
                field, evaluation, ..
            }
            | InterfaceOperand::Residual {
                field, evaluation, ..
            } => {
                let spec = space.field(*field)?;
                match evaluation {
                    TraceEvaluation::Value => spec.components,
                    TraceEvaluation::Gradient => spec.components * dimension,
                }
            }
            InterfaceOperand::Normal | InterfaceOperand::Coordinates => dimension,
            InterfaceOperand::FacetMeasure => 1,
            InterfaceOperand::Constant { values } => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(FinitumError::InvalidRealization(format!(
                        "interface kernel `{}` operand {index} constant is not finite",
                        definition.name
                    )));
                }
                values.len()
            }
            InterfaceOperand::Ignored => actual,
        };
        if actual != expected {
            return Err(FinitumError::InvalidRealization(format!(
                "interface kernel `{}` operand {index} has {actual} components, its binding \
                 {operand:?} needs {expected}",
                definition.name
            )));
        }
    }
    Ok(())
}

fn facet_quadrature(
    mesh: &Mesh,
    topology: &FacetTopology,
    pair: OrientedFacetPair,
) -> Result<FacetQuadrature, FinitumError> {
    let dimension = mesh.dimension();
    let facet = topology.facets().get(pair.facet.0).ok_or_else(|| {
        FinitumError::InvalidRealization(format!("facet {} does not exist", pair.facet.0))
    })?;
    if !facet.is_interior() {
        return Err(FinitumError::InvalidRealization(format!(
            "facet {} is not two-sided; interface measures admit interior facets only",
            pair.facet.0
        )));
    }
    for incidence in [pair.minus, pair.plus] {
        if !facet
            .incidences
            .iter()
            .any(|own| own.cell == incidence.cell && own.local_facet == incidence.local_facet)
        {
            return Err(FinitumError::InvalidRealization(format!(
                "cell {} (local facet {}) is not incident to facet {}",
                incidence.cell.0, incidence.local_facet, pair.facet.0
            )));
        }
    }
    let vertices = facet
        .vertices
        .iter()
        .map(|vertex| mesh.vertices()[vertex.0].clone())
        .collect::<Vec<_>>();
    let origin = &vertices[0];
    let tangents = vertices[1..]
        .iter()
        .map(|vertex| {
            (0..dimension)
                .map(|axis| vertex[axis] - origin[axis])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (jacobian, mut normal, rule) = match dimension {
        2 => {
            let t = &tangents[0];
            (
                (t[0] * t[0] + t[1] * t[1]).sqrt(),
                vec![t[1], -t[0]],
                gauss_legendre_unit_interval(3),
            )
        }
        3 => {
            let (t1, t2) = (&tangents[0], &tangents[1]);
            let cross = vec![
                t1[1] * t2[2] - t1[2] * t2[1],
                t1[2] * t2[0] - t1[0] * t2[2],
                t1[0] * t2[1] - t1[1] * t2[0],
            ];
            let norm = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
            (norm, cross, triangle_degree4_quadrature())
        }
        other => {
            return Err(FinitumError::UnsupportedRealization(format!(
                "interface realization is supported for mesh dimension 2 or 3, got {other}"
            )));
        }
    };
    if !(jacobian.is_finite() && jacobian > 0.0) {
        return Err(FinitumError::InvalidRealization(format!(
            "facet {} has a degenerate geometry",
            pair.facet.0
        )));
    }
    let norm = normal.iter().map(|v| v * v).sum::<f64>().sqrt();
    for value in &mut normal {
        *value /= norm;
    }
    // Orient from the minus cell to the plus cell: away from the minus cell's opposite vertex.
    let minus_cell = mesh.cell(pair.minus.cell).expect("validated incidence");
    let opposite = &mesh.vertices()[minus_cell.vertices[pair.minus.local_facet].0];
    let towards_opposite = (0..dimension)
        .map(|axis| normal[axis] * (opposite[axis] - origin[axis]))
        .sum::<f64>();
    if towards_opposite > 0.0 {
        for value in &mut normal {
            *value = -*value;
        }
    }
    let plus_cell = mesh.cell(pair.plus.cell).expect("validated incidence");
    let plus_opposite = &mesh.vertices()[plus_cell.vertices[pair.plus.local_facet].0];
    let towards_plus = (0..dimension)
        .map(|axis| normal[axis] * (plus_opposite[axis] - origin[axis]))
        .sum::<f64>();
    if towards_plus <= 0.0 {
        return Err(FinitumError::InvalidRealization(format!(
            "facet {} normal does not point into the plus cell {}",
            pair.facet.0, pair.plus.cell.0
        )));
    }
    let minus_map = AffineMap::from_cell(mesh, pair.minus.cell)?;
    let plus_map = AffineMap::from_cell(mesh, pair.plus.cell)?;
    let mut points = Vec::with_capacity(rule.len());
    let mut measure = 0.0;
    for quadrature_point in rule {
        let physical = (0..dimension)
            .map(|axis| {
                origin[axis]
                    + quadrature_point
                        .coordinates
                        .iter()
                        .zip(&tangents)
                        .map(|(s, t)| s * t[axis])
                        .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let scale = quadrature_point.weight * jacobian;
        measure += scale;
        points.push(FacetPoint {
            minus_reference: minus_map.reference_point(&physical)?,
            plus_reference: plus_map.reference_point(&physical)?,
            physical,
            scale,
        });
    }
    Ok(FacetQuadrature {
        pair,
        points,
        normal,
        measure,
        minus_map,
        plus_map,
    })
}

fn interface_digest(
    space: &InterfaceSpace,
    measure: &InterfaceMeasure,
    kernels: &[BoundKernel],
) -> Digest {
    #[derive(Serialize)]
    struct PairIdentity {
        facet: usize,
        minus: usize,
        plus: usize,
    }
    #[derive(Serialize)]
    struct KernelIdentity<'a> {
        kernel: &'a Digest,
        operands: &'a [InterfaceOperand],
    }
    #[derive(Serialize)]
    struct BlockIdentity {
        symbol: u32,
        offset: usize,
        extent: usize,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        dimension: usize,
        vertices: &'a [Vec<f64>],
        cells: Vec<Vec<usize>>,
        fields: &'a [TraceFieldSpec],
        blocks: Vec<BlockIdentity>,
        facets: Vec<PairIdentity>,
        kernels: Vec<KernelIdentity<'a>>,
    }
    let bytes = serde_json::to_vec(&Payload {
        schema: INTERFACE_REALIZATION_SCHEMA,
        dimension: space.mesh.dimension(),
        vertices: space.mesh.vertices(),
        cells: space
            .mesh
            .cells()
            .iter()
            .map(|cell| cell.vertices.iter().map(|vertex| vertex.0).collect())
            .collect(),
        fields: &space.fields,
        blocks: space
            .layout
            .blocks()
            .iter()
            .map(|block| BlockIdentity {
                symbol: block.symbol.0,
                offset: block.offset,
                extent: block.extent,
            })
            .collect(),
        facets: measure
            .pairs
            .iter()
            .map(|pair| PairIdentity {
                facet: pair.facet.0,
                minus: pair.minus.cell.0,
                plus: pair.plus.cell.0,
            })
            .collect(),
        kernels: kernels
            .iter()
            .map(|bound| KernelIdentity {
                kernel: &bound.kernel.digest,
                operands: &bound.kernel.operands,
            })
            .collect(),
    })
    .expect("interface realization identity is serializable");
    Digest::blake3(&bytes)
}
