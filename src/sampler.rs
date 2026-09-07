//! Public field sampling (W8 lane F1, `finitum-field-sampler/1`): value, physical gradient,
//! divergence and exterior-facet traces of one realized field at physical points, with the
//! element family's own reference basis and Piola maps, plus the cell/facet geometry and the
//! quadrature rules a consumer integrates functionals with.
//!
//! This is the public evaluation authority for realized fields (`sinbad/ARCHITECTURE.md` §8,
//! "Field sampling authority is Finitum"): a consumer such as Sinbad's observable evaluator
//! samples through [`FieldSampler`] instead of restating Finitum's DOF ordering, reference bases
//! and pullbacks. Nothing here is a second copy of a convention: Lagrange fields evaluate
//! [`simplex_basis`] (the crate's single P1/P2 source of truth), RT0 fields evaluate
//! [`rt0_reference_basis`] under [`AffineMap::contravariant_piola`] with
//! [`CompatibleDofMaps::hdiv`]'s own orientation table, physical gradients go through
//! [`AffineMap::covariant_piola`], DOF maps are the canonical `crate::space` constructors (or the
//! plan's own map), and exterior-facet geometry is `crate::realization`'s `FacetGeometry`.
//!
//! [`QuadratureView`] exposes the rule a plan integrates with (reference and physical points,
//! physical weights, verified exactness degree) and [`QuadratureRule::for_degree`] selects the
//! smallest rule of this crate exact to a requested polynomial degree, so a functional such as
//! `dot(u, u)` of a P1 field (degree 2) is integrated exactly instead of with the barycenter
//! rule (W8 decision 1: quadrature is a recorded numerical contract).
//!
//! Honest limits: families other than P1/P2 Lagrange (scalar or dimension-vector), P0 and RT0
//! are refused typed ([`FinitumError::SamplingUnsupported`]); geometry is affine (straight-sided
//! simplices) only; traces are exterior facets only (an interior facet has no outward normal;
//! two-sided traces belong with the SC-W2 interface realization); a point is evaluated in the
//! cell the caller names (no point location -- the polynomial is extrapolated outside the cell,
//! see [`FieldSampler::cell_contains`]).

use std::borrow::Cow;

use scientia::scientific::ValueShape;
use scientia::{Digest, ElementFamilyRequirement, SymbolId};
use serde::Serialize;

use crate::element::{
    barycenter_quadrature, gauss_legendre_unit_interval, rt0_reference_basis, simplex_basis,
    tetrahedron_degree2_quadrature, tetrahedron_degree5_quadrature, triangle_degree4_quadrature,
    triangle_degree5_quadrature,
};
use crate::realization::FacetGeometry;
use crate::space::cell_constant_dof_map;
use crate::{
    AffineMap, CellId, CompatibleDofMaps, DofMap, ElementRestriction, FacetId, FacetTopology,
    FinitumError, Mesh, MixedSpace, PreparedElement, QuadraturePoint, RealizationPlan,
    SystemRealizationPlan, quadratic_simplex_dof_map, vector_nodal_dof_map,
};

/// Schema of [`FieldSamplerConventions`] and its digest.
pub const FIELD_SAMPLER_SCHEMA: &str = "finitum-field-sampler/1";

/// Schema of [`QuadratureRule::identity`].
pub const QUADRATURE_RULE_SCHEMA: &str = "finitum-quadrature-rule/1";

/// Absolute tolerance of the monomial exactness probe ([`QuadratureRule::verified_degree`]).
const MOMENT_TOLERANCE: f64 = 1.0e-13;

/// Highest degree the monomial probe checks for a caller-supplied table.
const PROBE_DEGREE_CAP: u16 = 10;

/// The element family of one sampled field -- exactly the families the plans realize.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum SampledFamily {
    /// Continuous nodal Lagrange simplex element of `order` 1 (vertex nodes) or 2 (vertex then
    /// edge-midpoint nodes) with `components` DOFs per node (`1` scalar, or the mesh
    /// dimension): local DOF `node * components + component`, the order
    /// [`vector_nodal_dof_map`] / [`quadratic_simplex_dof_map`] and every generated-kernel
    /// execution path use.
    Lagrange { order: u8, components: usize },
    /// Piecewise-constant P0 (`L2(order=0)`): one DOF per cell, `DofId(cell)`.
    CellConstant,
    /// Lowest-order Raviart-Thomas RT0 (`Hdiv(order=0)`): one oriented flux DOF per mesh facet
    /// (`DofId(facet)`), reference basis `phi_i(x) = x - p_i` on the facet opposite local
    /// vertex `i`, pushed forward by the contravariant Piola map; the physical coefficient is
    /// `orientation * dof` with [`CompatibleDofMaps::hdiv`]'s per-cell sign table.
    RaviartThomas0,
}

impl SampledFamily {
    /// Components of a sampled value: `components` for a Lagrange field, `1` for P0, the mesh
    /// dimension for RT0 (a vector field with one scalar DOF per facet).
    pub fn component_count(self, dimension: usize) -> usize {
        match self {
            Self::Lagrange { components, .. } => components,
            Self::CellConstant => 1,
            Self::RaviartThomas0 => dimension,
        }
    }

    fn validate(self, dimension: usize) -> Result<(), FinitumError> {
        match self {
            Self::Lagrange { order, components } => {
                if !matches!(order, 1 | 2) {
                    return Err(FinitumError::SamplingUnsupported {
                        family: format!("Lagrange(order={order})"),
                        reason: "the sampler reconstructs P1 and P2 simplex Lagrange fields only"
                            .into(),
                    });
                }
                if components != 1 && components != dimension {
                    return Err(FinitumError::SamplingUnsupported {
                        family: format!("Lagrange(order={order}, components={components})"),
                        reason: format!(
                            "a Lagrange field is scalar or dimension-{dimension}-vector valued"
                        ),
                    });
                }
                Ok(())
            }
            Self::CellConstant => Ok(()),
            Self::RaviartThomas0 => {
                if (2..=3).contains(&dimension) {
                    Ok(())
                } else {
                    Err(FinitumError::SamplingUnsupported {
                        family: "Hdiv(order=0)".into(),
                        reason: format!(
                            "RT0 is realized on triangles and tetrahedra, not in dimension \
                             {dimension}"
                        ),
                    })
                }
            }
        }
    }

    fn describe(self) -> String {
        match self {
            Self::Lagrange { order, components } => {
                format!("Lagrange(order={order}, components={components})")
            }
            Self::CellConstant => "L2(order=0)".into(),
            Self::RaviartThomas0 => "Hdiv(order=0)".into(),
        }
    }

    fn dof_ordering(self) -> &'static str {
        match self {
            Self::Lagrange { order: 1, .. } => "vertex-major nodes; node * components + component",
            Self::Lagrange { .. } => {
                "vertex-then-canonical-edge-major nodes; node * components + component"
            }
            Self::CellConstant => "one DOF per cell, DofId(cell)",
            Self::RaviartThomas0 => {
                "one DOF per facet, DofId(facet); cell coefficient = incidence orientation * dof"
            }
        }
    }

    fn reference_basis(self) -> &'static str {
        match self {
            Self::Lagrange { order: 1, .. } => {
                "barycentric P1: lambda_0 = 1 - sum x, lambda_k = x_k"
            }
            Self::Lagrange { .. } => "P2: lambda_i (2 lambda_i - 1), 4 lambda_i lambda_j",
            Self::CellConstant => "constant 1",
            Self::RaviartThomas0 => "RT0: phi_i(x) = x - p_i, facet opposite local vertex i",
        }
    }

    fn pullback(self) -> &'static str {
        match self {
            Self::Lagrange { .. } | Self::CellConstant => {
                "composition; gradient by the covariant Piola map J^{-T}"
            }
            Self::RaviartThomas0 => {
                "contravariant Piola map J / det J; divergence scaled by 1 / det J"
            }
        }
    }
}

/// The conventions a [`FieldSampler`] evaluates with, serialized under
/// [`FIELD_SAMPLER_SCHEMA`] and digested by [`Self::digest`]. This covers the conventions only
/// (family, DOF ordering, reference basis, pullback, facet convention); the concrete mesh, DOF
/// map and values are identified by the realization's own digest, which a receipt records
/// alongside this one.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FieldSamplerConventions {
    pub schema: &'static str,
    pub dimension: usize,
    pub family: SampledFamily,
    pub dof_ordering: &'static str,
    pub reference_basis: &'static str,
    pub pullback: &'static str,
    pub facet_convention: &'static str,
}

impl FieldSamplerConventions {
    pub fn new(dimension: usize, family: SampledFamily) -> Self {
        Self {
            schema: FIELD_SAMPLER_SCHEMA,
            dimension,
            family,
            dof_ordering: family.dof_ordering(),
            reference_basis: family.reference_basis(),
            pullback: family.pullback(),
            facet_convention: "local facet i omits local vertex i; exterior trace from the \
                               single incident cell; outward unit normal away from the omitted \
                               vertex; measure = physical facet length or area",
        }
    }

    /// Content-addressed identity of these conventions (`finitum-field-sampler/1`).
    pub fn digest(&self) -> Digest {
        Digest::blake3(
            &serde_json::to_vec(self).expect("sampler conventions are plain serializable data"),
        )
    }
}

/// One field's reconstruction at one point.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSample {
    /// Physical coordinates of the sampled point.
    pub coordinates: Vec<f64>,
    /// One entry per component (see [`SampledFamily::component_count`]).
    pub value: Vec<f64>,
    /// Physical gradient, one row per component, `dimension` entries per row.
    pub gradient: Vec<Vec<f64>>,
    /// Pointwise divergence when the field is dimension-vector valued (vector Lagrange, RT0);
    /// `None` for a scalar field.
    pub divergence: Option<f64>,
}

/// The trace of a field on an exterior facet at one point.
#[derive(Clone, Debug, PartialEq)]
pub struct FacetTrace {
    pub facet: FacetId,
    /// The single incident cell the trace is taken from.
    pub cell: CellId,
    /// The facet's local index in that cell (the omitted local vertex).
    pub local_facet: usize,
    /// Physical coordinates of the sampled point.
    pub coordinates: Vec<f64>,
    /// One entry per component.
    pub value: Vec<f64>,
    /// Outward unit normal of the domain on this facet.
    pub normal: Vec<f64>,
    /// Physical facet measure (length in 2-D, area in 3-D, `1` in 1-D).
    pub measure: f64,
}

impl FacetTrace {
    /// `value . normal` -- defined when the value has one component per axis (a vector
    /// Lagrange or RT0 field).
    pub fn normal_component(&self) -> Result<f64, FinitumError> {
        if self.value.len() != self.normal.len() {
            return Err(FinitumError::UnsupportedRealization(format!(
                "normal component of a {}-component trace in dimension {} is undefined",
                self.value.len(),
                self.normal.len()
            )));
        }
        Ok(self
            .value
            .iter()
            .zip(&self.normal)
            .map(|(value, normal)| value * normal)
            .sum())
    }
}

/// Geometry of one exterior facet.
#[derive(Clone, Debug, PartialEq)]
pub struct ExteriorFacet {
    pub facet: FacetId,
    pub cell: CellId,
    pub local_facet: usize,
    pub centroid: Vec<f64>,
    /// Physical facet measure (length in 2-D, area in 3-D, `1` in 1-D).
    pub measure: f64,
    /// Outward unit normal of the domain.
    pub normal: Vec<f64>,
}

/// Point reconstruction of one realized field over a simplex mesh.
///
/// `values` is the field's own DOF vector (block-local for a system field) in the DOF map's
/// numbering -- the *physical* vector: a caller holding a constrained state with affine
/// dependencies expands it first (`ConstraintSet::expand`).
#[derive(Clone, Debug)]
pub struct FieldSampler<'a> {
    mesh: &'a Mesh,
    family: SampledFamily,
    dofs: Cow<'a, DofMap>,
    /// RT0 only: `orientations[cell][local_facet]`, [`CompatibleDofMaps::hdiv`]'s table.
    orientations: Vec<Vec<i8>>,
    facets: Cow<'a, FacetTopology>,
    values: &'a [f64],
    conventions: FieldSamplerConventions,
    digest: Digest,
}

struct Local {
    value: Vec<f64>,
    gradient: Vec<Vec<f64>>,
}

impl<'a> FieldSampler<'a> {
    /// A sampler over the canonical DOF map of `family` on `mesh` ([`vector_nodal_dof_map`],
    /// [`quadratic_simplex_dof_map`], the cell-constant map, or [`CompatibleDofMaps::hdiv`]).
    pub fn new(
        mesh: &'a Mesh,
        family: SampledFamily,
        values: &'a [f64],
    ) -> Result<Self, FinitumError> {
        family.validate(mesh.dimension())?;
        let facets = FacetTopology::from_mesh(mesh)?;
        let (dofs, orientations) = match family {
            SampledFamily::Lagrange {
                order: 1,
                components,
            } => (vector_nodal_dof_map(mesh, components)?, Vec::new()),
            SampledFamily::Lagrange { components, .. } => {
                (quadratic_simplex_dof_map(mesh, components)?, Vec::new())
            }
            SampledFamily::CellConstant => (cell_constant_dof_map(mesh)?, Vec::new()),
            SampledFamily::RaviartThomas0 => {
                rt0_maps(mesh, &CompatibleDofMaps::simplex(mesh, &facets)?)?
            }
        };
        Self::assemble(
            mesh,
            family,
            Cow::Owned(dofs),
            orientations,
            Cow::Owned(facets),
            values,
        )
    }

    /// The sampler of a single-field [`RealizationPlan`]'s own field: the plan's element table
    /// fixes the order (basis count), the plan's DOF map fixes the numbering and component
    /// stride, and `values` is the plan's physical DOF vector (length
    /// [`RealizationPlan::dimension`]).
    pub fn from_realization_plan(
        plan: &'a RealizationPlan,
        values: &'a [f64],
    ) -> Result<Self, FinitumError> {
        let mesh = plan.mesh();
        let element = plan.element();
        let dofs = plan.dofs();
        let dimension = mesh.dimension();
        let order = if element.basis_count() == dimension + 1 {
            1
        } else if element.basis_count() == (dimension + 1) * (dimension + 2) / 2 {
            2
        } else {
            return Err(FinitumError::SamplingUnsupported {
                family: format!("{}-function element table", element.basis_count()),
                reason: format!(
                    "a realization plan in dimension {dimension} samples as P1 ({}) or P2 ({}) \
                     Lagrange only",
                    dimension + 1,
                    (dimension + 1) * (dimension + 2) / 2
                ),
            });
        };
        let first = dofs.restrictions().first().ok_or_else(|| {
            FinitumError::InvalidRealization("realization plan has no cells".into())
        })?;
        let components = first.dofs.len() / element.basis_count();
        if components * element.basis_count() != first.dofs.len() {
            return Err(FinitumError::InvalidRealization(format!(
                "cell 0 restriction has {} DOFs, not a multiple of the {}-function basis",
                first.dofs.len(),
                element.basis_count()
            )));
        }
        let family = SampledFamily::Lagrange { order, components };
        family.validate(dimension)?;
        Self::assemble(
            mesh,
            family,
            Cow::Borrowed(dofs),
            Vec::new(),
            Cow::Owned(FacetTopology::from_mesh(mesh)?),
            values,
        )
    }

    /// The sampler of one field of a [`SystemRealizationPlan`]: the family comes from the
    /// system's typed element requirement for `field` (mirroring the admission rule the bound
    /// `SystemOperator` realizes with: H1/L2 order 1 or 2 scalar or dimension-vector Lagrange,
    /// L2 order 0, Hdiv order 0), the DOF map is the canonical one that operator builds, and
    /// `solution` is the layout-wide vector (`plan.layout().extent()` entries), sliced to the
    /// field's block.
    pub fn from_system_plan(
        plan: &'a SystemRealizationPlan,
        field: SymbolId,
        solution: &'a [f64],
    ) -> Result<Self, FinitumError> {
        let layout = plan.layout();
        if solution.len() != layout.extent() {
            return Err(FinitumError::InvalidRealization(format!(
                "solution vector has {} entries, the block layout {}",
                solution.len(),
                layout.extent()
            )));
        }
        let block = layout.block(field).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("layout has no block for field {field}"))
        })?;
        let mesh = plan.mesh();
        let dimension = mesh.dimension();
        let requirement = plan
            .system()
            .blocks
            .iter()
            .flat_map(|block| block.requirements.elements.iter())
            .find(|requirement| requirement.symbol == field)
            .ok_or_else(|| {
                FinitumError::ArtifactMismatch(format!(
                    "system field {field} has no element requirement in any block"
                ))
            })?;
        let family = match (requirement.family, requirement.polynomial_order) {
            (ElementFamilyRequirement::H1 | ElementFamilyRequirement::L2, order @ (1 | 2)) => {
                let components = match &requirement.value_shape {
                    ValueShape::Scalar => 1,
                    ValueShape::Vector(extent) if *extent as usize == dimension => dimension,
                    other => {
                        return Err(FinitumError::SamplingUnsupported {
                            family: format!(
                                "{:?}(order={order}) with value shape {other:?}",
                                requirement.family
                            ),
                            reason: format!(
                                "a Lagrange system field is scalar or dimension-{dimension}-\
                                 vector valued"
                            ),
                        });
                    }
                };
                SampledFamily::Lagrange { order, components }
            }
            (ElementFamilyRequirement::L2, 0) => SampledFamily::CellConstant,
            (ElementFamilyRequirement::Hdiv, 0) => SampledFamily::RaviartThomas0,
            (family, order) => {
                return Err(FinitumError::SamplingUnsupported {
                    family: format!("{family:?}(order={order})"),
                    reason: "the sampler reconstructs H1/L2 order 1 or 2 Lagrange, L2 order 0 \
                             and Hdiv order 0 system fields"
                        .into(),
                });
            }
        };
        family.validate(dimension)?;
        let facets = Cow::Borrowed(plan.facets());
        let (dofs, orientations) = match family {
            SampledFamily::Lagrange {
                order: 1,
                components,
            } => (vector_nodal_dof_map(mesh, components)?, Vec::new()),
            SampledFamily::Lagrange { components, .. } => {
                (quadratic_simplex_dof_map(mesh, components)?, Vec::new())
            }
            SampledFamily::CellConstant => (cell_constant_dof_map(mesh)?, Vec::new()),
            SampledFamily::RaviartThomas0 => match plan.compatible_dofs() {
                Some(compatible) => rt0_maps(mesh, compatible)?,
                None => rt0_maps(mesh, &CompatibleDofMaps::simplex(mesh, &facets)?)?,
            },
        };
        let values = &solution[block.offset..block.offset + block.extent];
        Self::assemble(mesh, family, Cow::Owned(dofs), orientations, facets, values)
    }

    /// The sampler of one field of a [`MixedSpace`] (its own `FieldSpec` order/components and
    /// DOF map), `solution` being the layout-wide vector.
    pub fn from_mixed_space(
        space: &'a MixedSpace,
        field: SymbolId,
        solution: &'a [f64],
    ) -> Result<Self, FinitumError> {
        let spec = space.field(field).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("mixed space has no field {field}"))
        })?;
        let dofs = space.dof_map(field).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("mixed space has no DOF map for {field}"))
        })?;
        let layout = space.layout();
        if solution.len() != layout.extent() {
            return Err(FinitumError::InvalidRealization(format!(
                "solution vector has {} entries, the block layout {}",
                solution.len(),
                layout.extent()
            )));
        }
        let block = layout.block(field).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("layout has no block for field {field}"))
        })?;
        let family = SampledFamily::Lagrange {
            order: spec.order,
            components: spec.components,
        };
        let mesh = space.mesh();
        family.validate(mesh.dimension())?;
        Self::assemble(
            mesh,
            family,
            Cow::Borrowed(dofs),
            Vec::new(),
            Cow::Owned(FacetTopology::from_mesh(mesh)?),
            &solution[block.offset..block.offset + block.extent],
        )
    }

    fn assemble(
        mesh: &'a Mesh,
        family: SampledFamily,
        dofs: Cow<'a, DofMap>,
        orientations: Vec<Vec<i8>>,
        facets: Cow<'a, FacetTopology>,
        values: &'a [f64],
    ) -> Result<Self, FinitumError> {
        let dimension = mesh.dimension();
        if dofs.restrictions().len() != mesh.cells().len() {
            return Err(FinitumError::InvalidRealization(format!(
                "DOF map has {} restrictions, mesh has {} cells",
                dofs.restrictions().len(),
                mesh.cells().len()
            )));
        }
        let local_extent = match family {
            SampledFamily::Lagrange { order, components } => {
                crate::element::simplex_basis_count(dimension, order) * components
            }
            SampledFamily::CellConstant => 1,
            SampledFamily::RaviartThomas0 => dimension + 1,
        };
        if let Some((cell, restriction)) = dofs
            .restrictions()
            .iter()
            .enumerate()
            .find(|(_, restriction)| restriction.dofs.len() != local_extent)
        {
            return Err(FinitumError::InvalidRealization(format!(
                "cell {cell} restriction has {} DOFs, {} needs {local_extent}",
                restriction.dofs.len(),
                family.describe()
            )));
        }
        if family == SampledFamily::RaviartThomas0
            && (orientations.len() != mesh.cells().len()
                || orientations
                    .iter()
                    .any(|orientations| orientations.len() != local_extent))
        {
            return Err(FinitumError::InvalidRealization(
                "RT0 orientation table does not cover every cell facet".into(),
            ));
        }
        if values.len() != dofs.dof_count() {
            return Err(FinitumError::InvalidRealization(format!(
                "field values have {} entries, the {} DOF map has {}",
                values.len(),
                family.describe(),
                dofs.dof_count()
            )));
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::InvalidRealization(format!(
                "field values contain a non-finite entry at index {index}"
            )));
        }
        let conventions = FieldSamplerConventions::new(dimension, family);
        let digest = conventions.digest();
        Ok(Self {
            mesh,
            family,
            dofs,
            orientations,
            facets,
            values,
            conventions,
            digest,
        })
    }

    pub fn mesh(&self) -> &'a Mesh {
        self.mesh
    }

    pub fn family(&self) -> SampledFamily {
        self.family
    }

    /// The DOF map the values are numbered by.
    pub fn dofs(&self) -> &DofMap {
        &self.dofs
    }

    /// RT0 only: the per-cell orientation table (`[cell][local_facet]`); empty otherwise.
    pub fn orientations(&self) -> &[Vec<i8>] {
        &self.orientations
    }

    pub fn facets(&self) -> &FacetTopology {
        &self.facets
    }

    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    pub fn component_count(&self) -> usize {
        self.family.component_count(self.mesh.dimension())
    }

    pub fn conventions(&self) -> &FieldSamplerConventions {
        &self.conventions
    }

    /// The `finitum-field-sampler/1` digest of this sampler's conventions.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    fn affine(&self, cell: CellId) -> Result<AffineMap, FinitumError> {
        AffineMap::from_cell(self.mesh, cell)
    }

    fn restriction(&self, cell: CellId) -> &ElementRestriction {
        &self.dofs.restrictions()[cell.0]
    }

    /// Value and physical gradient at a reference point of `cell` (the affine map is passed in
    /// so physical-point callers compute it once).
    fn local(
        &self,
        cell: CellId,
        affine: &AffineMap,
        reference: &[f64],
    ) -> Result<Local, FinitumError> {
        let dimension = self.mesh.dimension();
        let restriction = self.restriction(cell);
        match self.family {
            SampledFamily::Lagrange { order, components } => {
                let (basis, reference_gradients) = simplex_basis(dimension, order, reference)?;
                let mut value = vec![0.0; components];
                let mut reference_gradient = vec![vec![0.0; dimension]; components];
                for (node, (basis_value, basis_gradient)) in
                    basis.iter().zip(&reference_gradients).enumerate()
                {
                    for component in 0..components {
                        let coefficient =
                            self.values[restriction.dofs[node * components + component].0];
                        value[component] += basis_value * coefficient;
                        for (axis, entry) in reference_gradient[component].iter_mut().enumerate() {
                            *entry += basis_gradient[axis] * coefficient;
                        }
                    }
                }
                let gradient = reference_gradient
                    .iter()
                    .map(|row| affine.covariant_piola(row))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Local { value, gradient })
            }
            SampledFamily::CellConstant => Ok(Local {
                value: vec![self.values[restriction.dofs[0].0]],
                gradient: vec![vec![0.0; dimension]],
            }),
            SampledFamily::RaviartThomas0 => {
                let (basis, _divergence) = rt0_reference_basis(dimension, reference)?;
                let orientations = &self.orientations[cell.0];
                let mut reference_value = vec![0.0; dimension];
                let mut coefficient_sum = 0.0;
                for ((dof, orientation), basis) in
                    restriction.dofs.iter().zip(orientations).zip(&basis)
                {
                    let signed = f64::from(*orientation) * self.values[dof.0];
                    coefficient_sum += signed;
                    for (entry, component) in reference_value.iter_mut().zip(basis) {
                        *entry += signed * component;
                    }
                }
                let value = affine.contravariant_piola(&reference_value)?;
                // phi_i(x) = (x - x_0 - J p_i) / det J in physical coordinates, so the
                // gradient of the field is (sum_i orientation_i dof_i / det J) times the
                // identity -- constant per cell, with trace `map_hdiv_divergence(d * sum)`.
                let scale = coefficient_sum / affine.determinant();
                let gradient = (0..dimension)
                    .map(|row| {
                        let mut entries = vec![0.0; dimension];
                        entries[row] = scale;
                        entries
                    })
                    .collect();
                Ok(Local { value, gradient })
            }
        }
    }

    /// The divergence a field with one component per axis defines (a vector Lagrange field, or
    /// RT0; a scalar Lagrange field in one dimension has one component per axis too); `None`
    /// for P0 and every other scalar field.
    fn divergence_of(&self, gradient: &[Vec<f64>]) -> Option<f64> {
        (self.family != SampledFamily::CellConstant && gradient.len() == self.mesh.dimension())
            .then(|| {
                gradient
                    .iter()
                    .enumerate()
                    .map(|(axis, row)| row[axis])
                    .sum()
            })
    }

    /// Everything at a reference point of `cell`.
    pub fn sample_at_reference(
        &self,
        cell: CellId,
        reference: &[f64],
    ) -> Result<FieldSample, FinitumError> {
        let affine = self.affine(cell)?;
        let coordinates = affine.physical_point(reference)?;
        let local = self.local(cell, &affine, reference)?;
        let divergence = self.divergence_of(&local.gradient);
        Ok(FieldSample {
            coordinates,
            value: local.value,
            gradient: local.gradient,
            divergence,
        })
    }

    /// Everything at a physical point, evaluated in `cell` (the point is mapped to the cell's
    /// reference coordinates through the inverse affine map; a point outside the cell is
    /// extrapolated, see [`Self::cell_contains`]).
    pub fn sample_at(&self, cell: CellId, physical: &[f64]) -> Result<FieldSample, FinitumError> {
        let affine = self.affine(cell)?;
        let reference = affine.reference_point(physical)?;
        let local = self.local(cell, &affine, &reference)?;
        let divergence = self.divergence_of(&local.gradient);
        Ok(FieldSample {
            coordinates: physical.to_vec(),
            value: local.value,
            gradient: local.gradient,
            divergence,
        })
    }

    /// Components of the field at a physical point of `cell`.
    pub fn value_at(&self, cell: CellId, physical: &[f64]) -> Result<Vec<f64>, FinitumError> {
        let affine = self.affine(cell)?;
        let reference = affine.reference_point(physical)?;
        Ok(self.local(cell, &affine, &reference)?.value)
    }

    /// Components of the field at a reference point of `cell`.
    pub fn value_at_reference(
        &self,
        cell: CellId,
        reference: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let affine = self.affine(cell)?;
        Ok(self.local(cell, &affine, reference)?.value)
    }

    /// Physical gradient at a physical point of `cell`: one row per component (the covariant
    /// Piola map of the reference gradient for Lagrange fields; the per-cell constant
    /// `sum / det J` times the identity for RT0; zero for P0).
    pub fn gradient_at(
        &self,
        cell: CellId,
        physical: &[f64],
    ) -> Result<Vec<Vec<f64>>, FinitumError> {
        let affine = self.affine(cell)?;
        let reference = affine.reference_point(physical)?;
        Ok(self.local(cell, &affine, &reference)?.gradient)
    }

    /// Physical gradient at a reference point of `cell`.
    pub fn gradient_at_reference(
        &self,
        cell: CellId,
        reference: &[f64],
    ) -> Result<Vec<Vec<f64>>, FinitumError> {
        let affine = self.affine(cell)?;
        Ok(self.local(cell, &affine, reference)?.gradient)
    }

    /// Divergence at a physical point of `cell`; refused for a field without one component per
    /// axis.
    pub fn divergence_at(&self, cell: CellId, physical: &[f64]) -> Result<f64, FinitumError> {
        let gradient = self.gradient_at(cell, physical)?;
        self.divergence_of(&gradient).ok_or_else(|| {
            FinitumError::UnsupportedRealization(format!(
                "divergence of a {}-component {} field in dimension {} is undefined",
                self.component_count(),
                self.family.describe(),
                self.mesh.dimension()
            ))
        })
    }

    /// Whether `physical` lies in `cell` up to `tolerance` on every barycentric coordinate.
    pub fn cell_contains(
        &self,
        cell: CellId,
        physical: &[f64],
        tolerance: f64,
    ) -> Result<bool, FinitumError> {
        let reference = self.affine(cell)?.reference_point(physical)?;
        let last = 1.0 - reference.iter().sum::<f64>();
        Ok(reference.iter().all(|value| *value >= -tolerance) && last >= -tolerance)
    }

    /// The trace of the field on exterior facet `facet` at a physical point of that facet,
    /// with the domain's outward unit normal and the facet measure. Refused typed for an
    /// interior facet.
    pub fn trace_at(&self, facet: FacetId, physical: &[f64]) -> Result<FacetTrace, FinitumError> {
        let geometry = exterior_facet(self.mesh, &self.facets, facet)?;
        let value = self.value_at(geometry.cell, physical)?;
        Ok(FacetTrace {
            facet,
            cell: geometry.cell,
            local_facet: geometry.local_facet,
            coordinates: physical.to_vec(),
            value,
            normal: geometry.normal,
            measure: geometry.measure,
        })
    }

    /// [`Self::trace_at`] at the facet centroid.
    pub fn trace_at_centroid(&self, facet: FacetId) -> Result<FacetTrace, FinitumError> {
        let geometry = exterior_facet(self.mesh, &self.facets, facet)?;
        let value = self.value_at(geometry.cell, &geometry.centroid)?;
        Ok(FacetTrace {
            facet,
            cell: geometry.cell,
            local_facet: geometry.local_facet,
            coordinates: geometry.centroid,
            value,
            normal: geometry.normal,
            measure: geometry.measure,
        })
    }

    pub fn cell_measure(&self, cell: CellId) -> Result<f64, FinitumError> {
        cell_measure(self.mesh, cell)
    }

    pub fn cell_centroid(&self, cell: CellId) -> Result<Vec<f64>, FinitumError> {
        cell_centroid(self.mesh, cell)
    }

    /// Geometry of an exterior facet (refused typed for an interior one).
    pub fn exterior_facet(&self, facet: FacetId) -> Result<ExteriorFacet, FinitumError> {
        exterior_facet(self.mesh, &self.facets, facet)
    }

    /// Centroid of any facet (interior or exterior).
    pub fn facet_centroid(&self, facet: FacetId) -> Result<Vec<f64>, FinitumError> {
        let facet = self.facets.facets().get(facet.0).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("facet {} does not exist", facet.0))
        })?;
        Ok(mean_of(
            self.mesh,
            facet.vertices.iter().map(|vertex| vertex.0),
        ))
    }
}

/// RT0 DOF map and orientation table from [`CompatibleDofMaps::hdiv`].
fn rt0_maps(
    mesh: &Mesh,
    compatible: &CompatibleDofMaps,
) -> Result<(DofMap, Vec<Vec<i8>>), FinitumError> {
    if compatible.hdiv.len() != mesh.cells().len() {
        return Err(FinitumError::ArtifactMismatch(
            "RT0 compatible DOF map has a different cell count than the mesh".into(),
        ));
    }
    let restrictions = compatible
        .hdiv
        .iter()
        .map(|restriction| ElementRestriction {
            dofs: restriction.dofs.clone(),
        })
        .collect();
    let orientations = compatible
        .hdiv
        .iter()
        .map(|restriction| restriction.orientations.clone())
        .collect();
    Ok((
        DofMap::new(compatible.hdiv_dof_count, restrictions)?,
        orientations,
    ))
}

fn mean_of(mesh: &Mesh, vertices: impl Iterator<Item = usize>) -> Vec<f64> {
    let mut mean = vec![0.0; mesh.dimension()];
    let mut count = 0.0;
    for vertex in vertices {
        count += 1.0;
        for (entry, coordinate) in mean.iter_mut().zip(&mesh.vertices()[vertex]) {
            *entry += coordinate;
        }
    }
    for entry in &mut mean {
        *entry /= count;
    }
    mean
}

/// Measure (length, area, volume) of a simplex cell: `|det J| / dimension!`.
pub fn cell_measure(mesh: &Mesh, cell: CellId) -> Result<f64, FinitumError> {
    let affine = AffineMap::from_cell(mesh, cell)?;
    let factorial: f64 = (1..=mesh.dimension()).map(|k| k as f64).product();
    Ok(affine.volume_scale() / factorial)
}

/// Centroid (vertex mean) of a simplex cell.
pub fn cell_centroid(mesh: &Mesh, cell: CellId) -> Result<Vec<f64>, FinitumError> {
    let cell = mesh
        .cell(cell)
        .ok_or_else(|| FinitumError::InvalidRealization(format!("mesh has no cell {}", cell.0)))?;
    Ok(mean_of(mesh, cell.vertices.iter().map(|vertex| vertex.0)))
}

/// Geometry of exterior facet `facet` of `facets` (built from `mesh`): its single incident
/// cell, centroid, physical measure and the domain's outward unit normal. Dimensions 2 and 3
/// reuse `crate::realization`'s facet geometry (GX-C4); dimension 1 is the endpoint with normal
/// `+1`/`-1` away from the cell's other vertex and measure `1`. An interior facet is refused
/// typed (it has no outward normal).
pub fn exterior_facet(
    mesh: &Mesh,
    facets: &FacetTopology,
    facet: FacetId,
) -> Result<ExteriorFacet, FinitumError> {
    let data = facets.facets().get(facet.0).ok_or_else(|| {
        FinitumError::InvalidRealization(format!("facet {} does not exist", facet.0))
    })?;
    if !data.is_exterior() {
        return Err(FinitumError::UnsupportedRealization(format!(
            "facet {} is interior; only an exterior facet carries an outward normal and a \
             one-sided trace",
            facet.0
        )));
    }
    let incidence = data.minus();
    if mesh.dimension() == 1 {
        let cell = mesh.cell(incidence.cell).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("mesh has no cell {}", incidence.cell.0))
        })?;
        let point = mesh.vertices()[data.vertices[0].0][0];
        let opposite = mesh.vertices()[cell.vertices[incidence.local_facet].0][0];
        return Ok(ExteriorFacet {
            facet,
            cell: incidence.cell,
            local_facet: incidence.local_facet,
            centroid: vec![point],
            measure: 1.0,
            normal: vec![if point >= opposite { 1.0 } else { -1.0 }],
        });
    }
    let geometry = FacetGeometry::compute(mesh, incidence)?;
    Ok(ExteriorFacet {
        facet,
        cell: geometry.cell,
        local_facet: geometry.local_facet,
        measure: geometry.scale(mesh.dimension()),
        centroid: geometry.physical_centroid,
        normal: geometry.normal,
    })
}

/// Exact moment of a monomial over the unit reference simplex of dimension `exponents.len()`:
/// `integral prod x_i^{a_i} dx = prod a_i! / (sum a_i + d)!`.
pub fn simplex_monomial_moment(exponents: &[usize]) -> f64 {
    fn factorial(n: usize) -> f64 {
        (1..=n).map(|k| k as f64).product()
    }
    let total = exponents.iter().sum::<usize>();
    exponents.iter().map(|&a| factorial(a)).product::<f64>() / factorial(total + exponents.len())
}

/// Every exponent tuple of total degree `degree` in `dimension` variables.
fn exponent_tuples(dimension: usize, degree: usize) -> Vec<Vec<usize>> {
    if dimension == 1 {
        return vec![vec![degree]];
    }
    let mut tuples = Vec::new();
    for first in 0..=degree {
        for mut rest in exponent_tuples(dimension - 1, degree - first) {
            rest.insert(0, first);
            tuples.push(rest);
        }
    }
    tuples
}

/// One reference quadrature rule with a stable identity and a verified exactness degree.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuadratureRule {
    /// Stable identifier (`simplex-barycenter`, `gauss-legendre-3`, `triangle-edge-midpoints`,
    /// `triangle-dunavant-6`, `triangle-radon-7`, `tetrahedron-symmetric-4`,
    /// `tetrahedron-symmetric-14`, or `caller-table` for a table this crate does not name).
    pub id: String,
    pub dimension: usize,
    /// Highest polynomial degree the rule integrates exactly on the reference simplex --
    /// declared for a named rule and verified by [`Self::verified_degree`] in this module's
    /// tests; probed (up to degree 10) for a caller table.
    pub degree: u16,
    pub points: Vec<QuadraturePoint>,
}

impl QuadratureRule {
    fn named(
        id: impl Into<String>,
        dimension: usize,
        degree: u16,
        points: Vec<QuadraturePoint>,
    ) -> Self {
        Self {
            id: id.into(),
            dimension,
            degree,
            points,
        }
    }

    /// Every rule this crate names in `dimension`, by increasing degree.
    pub fn known(dimension: usize) -> Result<Vec<Self>, FinitumError> {
        let barycenter = Self::named(
            "simplex-barycenter",
            dimension,
            1,
            barycenter_quadrature(dimension)?,
        );
        Ok(match dimension {
            1 => std::iter::once(barycenter)
                .chain((2..=8).map(|count| {
                    Self::named(
                        format!("gauss-legendre-{count}"),
                        1,
                        2 * count as u16 - 1,
                        gauss_legendre_unit_interval(count),
                    )
                }))
                .collect(),
            2 => vec![
                barycenter,
                Self::named(
                    "triangle-edge-midpoints",
                    2,
                    2,
                    PreparedElement::linear_simplex_with_degree(2, 2)?
                        .quadrature()
                        .to_vec(),
                ),
                Self::named("triangle-dunavant-6", 2, 4, triangle_degree4_quadrature()),
                Self::named("triangle-radon-7", 2, 5, triangle_degree5_quadrature()),
            ],
            3 => vec![
                barycenter,
                Self::named(
                    "tetrahedron-symmetric-4",
                    3,
                    2,
                    tetrahedron_degree2_quadrature(),
                ),
                Self::named(
                    "tetrahedron-symmetric-14",
                    3,
                    5,
                    tetrahedron_degree5_quadrature(),
                ),
            ],
            _ => return Err(FinitumError::InvalidDimension(dimension)),
        })
    }

    /// The one-point barycenter rule (degree 1).
    pub fn barycenter(dimension: usize) -> Result<Self, FinitumError> {
        Ok(Self::named(
            "simplex-barycenter",
            dimension,
            1,
            barycenter_quadrature(dimension)?,
        ))
    }

    /// The smallest rule this crate names in `dimension` that is exact to polynomial `degree`
    /// (segments: Gauss-Legendre up to degree 15; triangles and tetrahedra: up to degree 5).
    /// Higher degrees are refused typed rather than silently under-integrated.
    pub fn for_degree(dimension: usize, degree: u16) -> Result<Self, FinitumError> {
        let known = Self::known(dimension)?;
        let richest = known.iter().map(|rule| rule.degree).max().unwrap_or(0);
        known
            .into_iter()
            .find(|rule| rule.degree >= degree)
            .ok_or_else(|| {
                FinitumError::UnsupportedRealization(format!(
                    "no quadrature rule exact to polynomial degree {degree} is realized in \
                     dimension {dimension}; the richest is degree {richest}"
                ))
            })
    }

    /// Identify a reference table: a table equal to one this crate names takes that name and
    /// declared degree; any other finite table is `caller-table` with its probed degree.
    pub fn from_table(
        dimension: usize,
        points: Vec<QuadraturePoint>,
    ) -> Result<Self, FinitumError> {
        if points.is_empty() {
            return Err(FinitumError::InvalidElementShape(
                "a quadrature rule needs at least one point".into(),
            ));
        }
        for (index, point) in points.iter().enumerate() {
            if point.coordinates.len() != dimension {
                return Err(FinitumError::InvalidElementShape(format!(
                    "quadrature point {index} has dimension {}, expected {dimension}",
                    point.coordinates.len()
                )));
            }
            if !point.weight.is_finite() || point.coordinates.iter().any(|value| !value.is_finite())
            {
                return Err(FinitumError::NonFiniteElementData {
                    location: format!("quadrature point {index}"),
                });
            }
        }
        if let Some(rule) = Self::known(dimension)?
            .into_iter()
            .find(|rule| rule.points == points)
        {
            return Ok(rule);
        }
        let candidate = Self::named("caller-table", dimension, 0, points);
        let degree = candidate.verified_degree(PROBE_DEGREE_CAP).ok_or_else(|| {
            FinitumError::InvalidElementShape(
                "quadrature table does not integrate constants exactly".into(),
            )
        })?;
        Ok(Self {
            degree,
            ..candidate
        })
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// The largest degree `<= cap` such that every monomial of every degree up to it integrates
    /// to its closed-form simplex moment within `1e-13`; `None` when even constants fail.
    pub fn verified_degree(&self, cap: u16) -> Option<u16> {
        let mut verified = None;
        for degree in 0..=cap {
            let exact_for_degree = exponent_tuples(self.dimension, degree as usize)
                .into_iter()
                .all(|exponents| {
                    let approximate = self
                        .points
                        .iter()
                        .map(|point| {
                            point.weight
                                * point
                                    .coordinates
                                    .iter()
                                    .zip(&exponents)
                                    .map(|(x, &a)| x.powi(a as i32))
                                    .product::<f64>()
                        })
                        .sum::<f64>();
                    (approximate - simplex_monomial_moment(&exponents)).abs() <= MOMENT_TOLERANCE
                });
            if !exact_for_degree {
                break;
            }
            verified = Some(degree);
        }
        verified
    }

    /// Content-addressed identity (`finitum-quadrature-rule/1`) over id, dimension, degree and
    /// the table itself.
    pub fn identity(&self) -> Digest {
        #[derive(Serialize)]
        struct Payload<'a> {
            schema: &'static str,
            rule: &'a QuadratureRule,
        }
        Digest::blake3(
            &serde_json::to_vec(&Payload {
                schema: QUADRATURE_RULE_SCHEMA,
                rule: self,
            })
            .expect("quadrature rules are plain serializable data"),
        )
    }
}

/// One quadrature point of one cell in physical space.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalQuadraturePoint {
    pub reference: Vec<f64>,
    pub physical: Vec<f64>,
    /// Physical weight: the reference weight times `|det J|`, so the weights of a cell sum to
    /// its measure.
    pub weight: f64,
}

/// A reference rule bound to a mesh: per-cell physical points and weights for integrating
/// functionals with a plan's own rule ([`Self::of_realization_plan`],
/// [`Self::of_system_plan`]) or with a rule selected from the integrand's degree
/// ([`Self::rule_for_degree`]).
#[derive(Clone, Debug)]
pub struct QuadratureView<'a> {
    mesh: &'a Mesh,
    rule: QuadratureRule,
}

impl<'a> QuadratureView<'a> {
    pub fn new(mesh: &'a Mesh, rule: QuadratureRule) -> Result<Self, FinitumError> {
        if rule.dimension != mesh.dimension() {
            return Err(FinitumError::InvalidRealization(format!(
                "quadrature rule {} has dimension {}, mesh has dimension {}",
                rule.id,
                rule.dimension,
                mesh.dimension()
            )));
        }
        Ok(Self { mesh, rule })
    }

    /// The rule a [`RealizationPlan`]'s prepared element integrates with.
    pub fn of_realization_plan(plan: &'a RealizationPlan) -> Result<Self, FinitumError> {
        let rule = QuadratureRule::from_table(
            plan.mesh().dimension(),
            plan.element().quadrature().to_vec(),
        )?;
        Self::new(plan.mesh(), rule)
    }

    /// The shared cell rule a [`SystemRealizationPlan`] integrates every field with
    /// (`SystemQuadrature::Barycenter` or `Richest`, named).
    pub fn of_system_plan(plan: &'a SystemRealizationPlan) -> Result<Self, FinitumError> {
        let rule = QuadratureRule::from_table(plan.mesh().dimension(), plan.quadrature()?)?;
        Self::new(plan.mesh(), rule)
    }

    pub fn mesh(&self) -> &'a Mesh {
        self.mesh
    }

    pub fn rule(&self) -> &QuadratureRule {
        &self.rule
    }

    /// The same mesh under [`QuadratureRule::for_degree`]`(degree)`.
    pub fn rule_for_degree(&self, degree: u16) -> Result<Self, FinitumError> {
        Self::new(
            self.mesh,
            QuadratureRule::for_degree(self.mesh.dimension(), degree)?,
        )
    }

    /// The rule's points on `cell` in physical coordinates with physical weights.
    pub fn cell_points(&self, cell: CellId) -> Result<Vec<PhysicalQuadraturePoint>, FinitumError> {
        let affine = AffineMap::from_cell(self.mesh, cell)?;
        let scale = affine.volume_scale();
        self.rule
            .points
            .iter()
            .map(|point| {
                Ok(PhysicalQuadraturePoint {
                    reference: point.coordinates.clone(),
                    physical: affine.physical_point(&point.coordinates)?,
                    weight: point.weight * scale,
                })
            })
            .collect()
    }

    /// `sum_cells sum_points weight * integrand(cell, point)` over the named cells.
    pub fn integrate_over(
        &self,
        cells: impl IntoIterator<Item = CellId>,
        mut integrand: impl FnMut(CellId, &PhysicalQuadraturePoint) -> Result<f64, FinitumError>,
    ) -> Result<f64, FinitumError> {
        let mut total = 0.0;
        for cell in cells {
            for point in self.cell_points(cell)? {
                let value = integrand(cell, &point)?;
                if !value.is_finite() {
                    return Err(FinitumError::InvalidRealization(format!(
                        "integrand is not finite at cell {} point {:?}",
                        cell.0, point.physical
                    )));
                }
                total += point.weight * value;
            }
        }
        Ok(total)
    }

    /// [`Self::integrate_over`] every cell of the mesh.
    pub fn integrate(
        &self,
        integrand: impl FnMut(CellId, &PhysicalQuadraturePoint) -> Result<f64, FinitumError>,
    ) -> Result<f64, FinitumError> {
        self.integrate_over((0..self.mesh.cells().len()).map(CellId), integrand)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell, VertexId, quadratic_simplex_node_points};

    /// Two sheared triangles (no edge axis-aligned), one positively and one negatively
    /// oriented, sharing the diagonal edge.
    fn sheared_triangles() -> Mesh {
        Mesh::new(
            2,
            vec![
                vec![0.1, 0.2],
                vec![1.3, 0.5],
                vec![1.1, 1.7],
                vec![-0.2, 1.1],
            ],
            vec![
                Cell {
                    vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
                },
                Cell {
                    vertices: vec![VertexId(2), VertexId(0), VertexId(3)],
                },
            ],
        )
        .unwrap()
    }

    /// Two skewed tetrahedra sharing a face, with opposite orientations.
    fn skewed_tetrahedra() -> Mesh {
        Mesh::new(
            3,
            vec![
                vec![0.1, 0.0, 0.2],
                vec![1.2, 0.3, 0.1],
                vec![0.4, 1.1, 0.3],
                vec![0.3, 0.2, 1.3],
                vec![1.1, 1.2, 1.4],
            ],
            vec![
                Cell {
                    vertices: vec![VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
                },
                Cell {
                    vertices: vec![VertexId(1), VertexId(2), VertexId(3), VertexId(4)],
                },
            ],
        )
        .unwrap()
    }

    fn segments() -> Mesh {
        Mesh::new(
            1,
            vec![vec![0.3], vec![1.1], vec![2.6]],
            vec![
                Cell {
                    vertices: vec![VertexId(0), VertexId(1)],
                },
                Cell {
                    vertices: vec![VertexId(2), VertexId(1)],
                },
            ],
        )
        .unwrap()
    }

    fn meshes() -> Vec<Mesh> {
        vec![segments(), sheared_triangles(), skewed_tetrahedra()]
    }

    /// A few reference points strictly inside the simplex plus one outside (extrapolation).
    fn probes(dimension: usize) -> Vec<Vec<f64>> {
        let mut probes = vec![
            vec![0.21; dimension],
            vec![1.0 / (dimension as f64 + 1.0); dimension],
            (0..dimension).map(|k| 0.05 + 0.15 * k as f64).collect(),
            vec![1.2 / dimension as f64; dimension],
        ];
        if dimension > 1 {
            probes.push(
                (0..dimension)
                    .map(|k| if k == 0 { 0.7 } else { 0.05 })
                    .collect(),
            );
        }
        probes
    }

    fn assert_near(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}: {actual} != {expected} (difference {})",
            (actual - expected).abs()
        );
    }

    /// Affine field for component `c`: `a_c + b_c . x`.
    fn affine_field(dimension: usize, component: usize, x: &[f64]) -> (f64, Vec<f64>) {
        let a = 0.3 + 0.7 * component as f64;
        let b = (0..dimension)
            .map(|axis| 1.5 - 0.4 * axis as f64 + 0.25 * component as f64)
            .collect::<Vec<_>>();
        let value = a + b.iter().zip(x).map(|(b, x)| b * x).sum::<f64>();
        (value, b)
    }

    /// Quadratic field for component `c`: `a + b . x + x^T C x`.
    fn quadratic_field(dimension: usize, component: usize, x: &[f64]) -> (f64, Vec<f64>) {
        let (affine, mut gradient) = affine_field(dimension, component, x);
        let mut value = affine;
        for i in 0..dimension {
            for j in 0..dimension {
                let c = 0.6 - 0.2 * (i + j) as f64 + 0.1 * component as f64 * (i as f64 + 1.0);
                value += c * x[i] * x[j];
                gradient[i] += c * x[j];
                gradient[j] += c * x[i];
            }
        }
        (value, gradient)
    }

    #[test]
    fn p1_scalar_and_vector_reproduce_affine_fields_to_roundoff() {
        for mesh in meshes() {
            let dimension = mesh.dimension();
            for components in [1, dimension] {
                let family = SampledFamily::Lagrange {
                    order: 1,
                    components,
                };
                let mut values = vec![0.0; mesh.vertices().len() * components];
                for (vertex, x) in mesh.vertices().iter().enumerate() {
                    for component in 0..components {
                        values[vertex * components + component] =
                            affine_field(dimension, component, x).0;
                    }
                }
                let sampler = FieldSampler::new(&mesh, family, &values).unwrap();
                assert_eq!(sampler.component_count(), components);
                for cell in 0..mesh.cells().len() {
                    for reference in probes(dimension) {
                        let sample = sampler
                            .sample_at_reference(CellId(cell), &reference)
                            .unwrap();
                        let physical = sampler
                            .sample_at(CellId(cell), &sample.coordinates)
                            .unwrap();
                        for component in 0..components {
                            let (value, gradient) =
                                affine_field(dimension, component, &sample.coordinates);
                            assert_near(sample.value[component], value, 1.0e-14, "P1 value");
                            assert_near(
                                physical.value[component],
                                value,
                                1.0e-14,
                                "P1 value (physical)",
                            );
                            for (axis, expected) in gradient.iter().enumerate() {
                                assert_near(
                                    sample.gradient[component][axis],
                                    *expected,
                                    1.0e-14,
                                    "P1 gradient",
                                );
                                assert_near(
                                    physical.gradient[component][axis],
                                    *expected,
                                    1.0e-14,
                                    "P1 gradient (physical)",
                                );
                            }
                        }
                        if components == dimension {
                            let expected = (0..dimension)
                                .map(|axis| {
                                    affine_field(dimension, axis, &sample.coordinates).1[axis]
                                })
                                .sum::<f64>();
                            assert_near(
                                sample.divergence.unwrap(),
                                expected,
                                1.0e-14,
                                "P1 divergence",
                            );
                            assert_near(
                                sampler
                                    .divergence_at(CellId(cell), &sample.coordinates)
                                    .unwrap(),
                                expected,
                                1.0e-14,
                                "P1 divergence_at",
                            );
                        } else {
                            assert_eq!(sample.divergence, None);
                            assert!(matches!(
                                sampler.divergence_at(CellId(cell), &sample.coordinates),
                                Err(FinitumError::UnsupportedRealization(_))
                            ));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn p2_scalar_and_vector_reproduce_quadratic_fields_to_roundoff() {
        for mesh in meshes() {
            let dimension = mesh.dimension();
            let nodes = quadratic_simplex_node_points(&mesh);
            for components in [1, dimension] {
                let family = SampledFamily::Lagrange {
                    order: 2,
                    components,
                };
                let mut values = vec![0.0; nodes.len() * components];
                for (node, x) in nodes.iter().enumerate() {
                    for component in 0..components {
                        values[node * components + component] =
                            quadratic_field(dimension, component, x).0;
                    }
                }
                let sampler = FieldSampler::new(&mesh, family, &values).unwrap();
                assert_eq!(sampler.dofs().dof_count(), values.len());
                for cell in 0..mesh.cells().len() {
                    for reference in probes(dimension) {
                        let sample = sampler
                            .sample_at_reference(CellId(cell), &reference)
                            .unwrap();
                        for component in 0..components {
                            let (value, gradient) =
                                quadratic_field(dimension, component, &sample.coordinates);
                            assert_near(sample.value[component], value, 1.0e-13, "P2 value");
                            for (axis, expected) in gradient.iter().enumerate() {
                                assert_near(
                                    sample.gradient[component][axis],
                                    *expected,
                                    1.0e-13,
                                    "P2 gradient",
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn p0_samples_the_cell_constant_with_zero_gradient() {
        for mesh in meshes() {
            let values = (0..mesh.cells().len())
                .map(|cell| 2.0 + cell as f64)
                .collect::<Vec<_>>();
            let sampler = FieldSampler::new(&mesh, SampledFamily::CellConstant, &values).unwrap();
            for (cell, expected) in values.iter().enumerate() {
                let centroid = sampler.cell_centroid(CellId(cell)).unwrap();
                let sample = sampler.sample_at(CellId(cell), &centroid).unwrap();
                assert_eq!(sample.value, vec![*expected]);
                assert_eq!(sample.gradient, vec![vec![0.0; mesh.dimension()]]);
                assert_eq!(sample.divergence, None);
            }
        }
    }

    /// DOF values of a constant vector field `q` in RT0: `c_F = (d-1)! * orientation *
    /// sign(det J) * |F| * (q . n_out)` from any incident cell (consistency across the two
    /// incident cells of an interior facet is what the orientation table guarantees).
    fn rt0_constant_field_dofs(mesh: &Mesh, facets: &FacetTopology, q: &[f64]) -> Vec<f64> {
        let dimension = mesh.dimension();
        let compatible = CompatibleDofMaps::simplex(mesh, facets).unwrap();
        let basis_flux = 1.0 / (1..dimension).product::<usize>() as f64;
        let mut dofs = vec![0.0; compatible.hdiv_dof_count];
        for facet in facets.facets() {
            let incidence = facet.minus();
            let cell = incidence.cell;
            let geometry = FacetGeometry::compute(mesh, incidence).unwrap();
            let measure = geometry.scale(dimension);
            let outward_flux = q
                .iter()
                .zip(&geometry.normal)
                .map(|(q, n)| q * n)
                .sum::<f64>();
            let sign = AffineMap::from_cell(mesh, cell)
                .unwrap()
                .determinant()
                .signum();
            let orientation =
                f64::from(compatible.hdiv[cell.0].orientations[incidence.local_facet]);
            dofs[facet.id.0] = orientation * sign * measure * outward_flux / basis_flux;
        }
        dofs
    }

    #[test]
    fn rt0_reproduces_a_constant_flux_through_the_piola_map_on_skewed_cells() {
        for mesh in [sheared_triangles(), skewed_tetrahedra()] {
            let dimension = mesh.dimension();
            let facets = FacetTopology::from_mesh(&mesh).unwrap();
            let q = (0..dimension)
                .map(|axis| 0.8 - 0.5 * axis as f64)
                .collect::<Vec<_>>();
            let dofs = rt0_constant_field_dofs(&mesh, &facets, &q);
            let sampler = FieldSampler::new(&mesh, SampledFamily::RaviartThomas0, &dofs).unwrap();
            assert_eq!(sampler.component_count(), dimension);
            assert_eq!(sampler.orientations().len(), mesh.cells().len());
            for cell in 0..mesh.cells().len() {
                for reference in probes(dimension) {
                    let sample = sampler
                        .sample_at_reference(CellId(cell), &reference)
                        .unwrap();
                    for (axis, expected) in q.iter().enumerate() {
                        assert_near(sample.value[axis], *expected, 1.0e-13, "RT0 constant value");
                        for entry in &sample.gradient[axis] {
                            assert_near(*entry, 0.0, 1.0e-13, "RT0 gradient");
                        }
                    }
                    assert_near(sample.divergence.unwrap(), 0.0, 1.0e-13, "RT0 divergence");
                }
            }
            // Exterior traces: outward, unit, and the normal component is the constant flux;
            // an interior facet is refused.
            for facet in facets.facets() {
                if facet.is_interior() {
                    assert!(matches!(
                        sampler.trace_at_centroid(facet.id),
                        Err(FinitumError::UnsupportedRealization(_))
                    ));
                    continue;
                }
                let trace = sampler.trace_at_centroid(facet.id).unwrap();
                let norm = trace.normal.iter().map(|n| n * n).sum::<f64>().sqrt();
                assert_near(norm, 1.0, 1.0e-14, "unit normal");
                let cell_centroid = sampler.cell_centroid(trace.cell).unwrap();
                let outward = trace
                    .normal
                    .iter()
                    .zip(&trace.coordinates)
                    .zip(&cell_centroid)
                    .map(|((n, f), c)| n * (f - c))
                    .sum::<f64>();
                assert!(outward > 0.0, "normal points away from the cell centroid");
                let expected = q.iter().zip(&trace.normal).map(|(q, n)| q * n).sum::<f64>();
                assert_near(
                    trace.normal_component().unwrap(),
                    expected,
                    1.0e-13,
                    "RT0 normal trace",
                );
                assert!(trace.measure > 0.0);
            }
        }
    }

    #[test]
    fn rt0_normal_trace_of_one_facet_dof_is_its_flux_density() {
        for mesh in [sheared_triangles(), skewed_tetrahedra()] {
            let dimension = mesh.dimension();
            let facets = FacetTopology::from_mesh(&mesh).unwrap();
            let basis_flux = 1.0 / (1..dimension).product::<usize>() as f64;
            for facet in facets.exterior() {
                let mut dofs = vec![0.0; facets.facets().len()];
                dofs[facet.id.0] = 1.7;
                let sampler =
                    FieldSampler::new(&mesh, SampledFamily::RaviartThomas0, &dofs).unwrap();
                let trace = sampler.trace_at_centroid(facet.id).unwrap();
                let orientation =
                    f64::from(sampler.orientations()[trace.cell.0][trace.local_facet]);
                let sign = AffineMap::from_cell(&mesh, trace.cell)
                    .unwrap()
                    .determinant()
                    .signum();
                let expected = sign * orientation * 1.7 * basis_flux / trace.measure;
                assert_near(
                    trace.normal_component().unwrap(),
                    expected,
                    1.0e-13,
                    "flux density of one RT0 DOF",
                );
                // Every other exterior facet of the same cell sees zero normal trace from it.
                for other in facets.exterior() {
                    if other.id == facet.id || other.minus().cell != trace.cell {
                        continue;
                    }
                    let other_trace = sampler.trace_at_centroid(other.id).unwrap();
                    assert_near(
                        other_trace.normal_component().unwrap(),
                        0.0,
                        1.0e-13,
                        "zero flux",
                    );
                }
            }
        }
    }

    #[test]
    fn one_dimensional_exterior_facets_carry_signed_unit_normals() {
        let mesh = segments();
        let facets = FacetTopology::from_mesh(&mesh).unwrap();
        let values = vec![1.0, 2.0, 3.0];
        let sampler = FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 1,
                components: 1,
            },
            &values,
        )
        .unwrap();
        let mut seen = 0;
        for facet in facets.exterior() {
            let trace = sampler.trace_at_centroid(facet.id).unwrap();
            let x = trace.coordinates[0];
            let expected_normal = if x < 1.0 { -1.0 } else { 1.0 };
            assert_eq!(trace.normal, vec![expected_normal]);
            assert_eq!(trace.measure, 1.0);
            let expected_value = if x < 1.0 { 1.0 } else { 3.0 };
            assert_near(trace.value[0], expected_value, 1.0e-15, "1-D trace value");
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[test]
    fn geometry_measures_centroids_and_containment() {
        let mesh = sheared_triangles();
        let facets = FacetTopology::from_mesh(&mesh).unwrap();
        // Shoelace areas.
        let area = |a: &[f64], b: &[f64], c: &[f64]| {
            0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs()
        };
        let v = mesh.vertices();
        assert_near(
            cell_measure(&mesh, CellId(0)).unwrap(),
            area(&v[0], &v[1], &v[2]),
            1.0e-15,
            "area 0",
        );
        assert_near(
            cell_measure(&mesh, CellId(1)).unwrap(),
            area(&v[2], &v[0], &v[3]),
            1.0e-15,
            "area 1",
        );
        let centroid = cell_centroid(&mesh, CellId(0)).unwrap();
        assert_near(centroid[0], (0.1 + 1.3 + 1.1) / 3.0, 1.0e-15, "centroid x");
        assert_near(centroid[1], (0.2 + 0.5 + 1.7) / 3.0, 1.0e-15, "centroid y");
        let exterior_measure = facets
            .exterior()
            .map(|facet| exterior_facet(&mesh, &facets, facet.id).unwrap().measure)
            .sum::<f64>();
        let perimeter = [(0, 1), (1, 2), (2, 3), (3, 0)]
            .iter()
            .map(|&(a, b)| ((v[a][0] - v[b][0]).powi(2) + (v[a][1] - v[b][1]).powi(2)).sqrt())
            .sum::<f64>();
        assert_near(exterior_measure, perimeter, 1.0e-14, "perimeter");
        let values = vec![0.0; 4];
        let sampler = FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 1,
                components: 1,
            },
            &values,
        )
        .unwrap();
        assert!(sampler.cell_contains(CellId(0), &centroid, 0.0).unwrap());
        assert!(!sampler.cell_contains(CellId(1), &centroid, 0.0).unwrap());
        assert!(sampler.cell_contains(CellId(0), &v[0], 1.0e-12).unwrap());
        assert!(matches!(
            cell_measure(&mesh, CellId(7)),
            Err(FinitumError::InvalidRealization(_))
        ));
        let interior = facets.interior().next().unwrap();
        let facet_centroid = sampler.facet_centroid(interior.id).unwrap();
        assert_near(
            facet_centroid[0],
            0.5 * (v[0][0] + v[2][0]),
            1.0e-15,
            "facet centroid",
        );
    }

    #[test]
    fn unsupported_families_and_malformed_values_are_refused_typed() {
        let mesh = sheared_triangles();
        let values = vec![0.0; 4];
        match FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 3,
                components: 1,
            },
            &values,
        ) {
            Err(FinitumError::SamplingUnsupported { family, .. }) => {
                assert_eq!(family, "Lagrange(order=3)")
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }
        assert!(matches!(
            FieldSampler::new(
                &mesh,
                SampledFamily::Lagrange {
                    order: 1,
                    components: 3,
                },
                &values,
            ),
            Err(FinitumError::SamplingUnsupported { .. })
        ));
        assert!(matches!(
            FieldSampler::new(&segments(), SampledFamily::RaviartThomas0, &values),
            Err(FinitumError::SamplingUnsupported { .. })
        ));
        assert!(matches!(
            FieldSampler::new(
                &mesh,
                SampledFamily::Lagrange {
                    order: 1,
                    components: 1,
                },
                &values[..3],
            ),
            Err(FinitumError::InvalidRealization(_))
        ));
        let non_finite = vec![0.0, f64::NAN, 0.0, 0.0];
        assert!(matches!(
            FieldSampler::new(
                &mesh,
                SampledFamily::Lagrange {
                    order: 1,
                    components: 1,
                },
                &non_finite,
            ),
            Err(FinitumError::InvalidRealization(_))
        ));
        let sampler = FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 1,
                components: 1,
            },
            &values,
        )
        .unwrap();
        assert!(sampler.value_at(CellId(9), &[0.5, 0.5]).is_err());
        assert!(sampler.value_at(CellId(0), &[0.5]).is_err());
        assert!(sampler.trace_at_centroid(FacetId(99)).is_err());
    }

    #[test]
    fn conventions_digest_is_stable_per_family_and_dimension() {
        let mesh = sheared_triangles();
        let scalar = FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 1,
                components: 1,
            },
            &[1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let other_values = FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 1,
                components: 1,
            },
            &[4.0, 3.0, 2.0, 1.0],
        )
        .unwrap();
        assert_eq!(scalar.digest(), other_values.digest());
        assert_eq!(scalar.conventions().schema, FIELD_SAMPLER_SCHEMA);
        let vector = FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 1,
                components: 2,
            },
            &[0.0; 8],
        )
        .unwrap();
        assert_ne!(scalar.digest(), vector.digest());
        let p0 = FieldSampler::new(&mesh, SampledFamily::CellConstant, &[1.0, 2.0]).unwrap();
        assert_ne!(scalar.digest(), p0.digest());
        assert_eq!(
            FieldSamplerConventions::new(2, SampledFamily::CellConstant).digest(),
            *p0.digest()
        );
    }

    #[test]
    fn every_named_rule_verifies_its_declared_degree_against_closed_form_moments() {
        for dimension in 1..=3 {
            let known = QuadratureRule::known(dimension).unwrap();
            assert!(known.len() >= 3);
            for rule in &known {
                let verified = rule.verified_degree(rule.degree + 2).unwrap();
                assert_eq!(
                    verified, rule.degree,
                    "{} declares degree {} but verifies {verified}",
                    rule.id, rule.degree
                );
                let volume = rule.points.iter().map(|point| point.weight).sum::<f64>();
                assert_near(
                    volume,
                    simplex_monomial_moment(&vec![0; dimension]),
                    1.0e-14,
                    "volume",
                );
                assert!(
                    rule.points.iter().all(|point| point.weight > 0.0),
                    "{} positive",
                    rule.id
                );
                assert_eq!(
                    QuadratureRule::from_table(dimension, rule.points.clone()).unwrap(),
                    *rule
                );
            }
        }
        let tetrahedron = QuadratureRule::for_degree(3, 4).unwrap();
        assert_eq!(tetrahedron.id, "tetrahedron-symmetric-14");
        assert_eq!(tetrahedron.point_count(), 14);
        let triangle = QuadratureRule::for_degree(2, 5).unwrap();
        assert_eq!(triangle.id, "triangle-radon-7");
        assert_eq!(
            QuadratureRule::for_degree(2, 2).unwrap().id,
            "triangle-edge-midpoints"
        );
        assert_eq!(
            QuadratureRule::for_degree(2, 0).unwrap().id,
            "simplex-barycenter"
        );
        assert_eq!(
            QuadratureRule::for_degree(1, 6).unwrap().id,
            "gauss-legendre-4"
        );
        assert!(matches!(
            QuadratureRule::for_degree(3, 6),
            Err(FinitumError::UnsupportedRealization(_))
        ));
        assert!(matches!(
            QuadratureRule::for_degree(4, 1),
            Err(FinitumError::InvalidDimension(4))
        ));
        assert_ne!(tetrahedron.identity(), triangle.identity());
    }

    #[test]
    fn caller_tables_are_probed_and_the_richest_system_rules_are_named() {
        // A midpoint rule on the segment authored by hand: identified as the barycenter rule.
        let barycenter = QuadratureRule::from_table(
            1,
            vec![QuadraturePoint {
                coordinates: vec![0.5],
                weight: 1.0,
            }],
        )
        .unwrap();
        assert_eq!(barycenter.id, "simplex-barycenter");
        // A two-point trapezoid rule on the segment: unnamed, degree 1.
        let trapezoid = QuadratureRule::from_table(
            1,
            vec![
                QuadraturePoint {
                    coordinates: vec![0.0],
                    weight: 0.5,
                },
                QuadraturePoint {
                    coordinates: vec![1.0],
                    weight: 0.5,
                },
            ],
        )
        .unwrap();
        assert_eq!(trapezoid.id, "caller-table");
        assert_eq!(trapezoid.degree, 1);
        assert!(matches!(
            QuadratureRule::from_table(
                1,
                vec![QuadraturePoint {
                    coordinates: vec![0.5],
                    weight: 0.7,
                }],
            ),
            Err(FinitumError::InvalidElementShape(_))
        ));
        assert!(QuadratureRule::from_table(2, Vec::new()).is_err());
        for (dimension, id, degree) in [
            (1, "gauss-legendre-3", 5),
            (2, "triangle-dunavant-6", 4),
            (3, "tetrahedron-symmetric-4", 2),
        ] {
            let richest = QuadratureRule::from_table(
                dimension,
                crate::SystemQuadrature::Richest.table(dimension).unwrap(),
            )
            .unwrap();
            assert_eq!(richest.id, id);
            assert_eq!(richest.degree, degree);
            let barycenter = QuadratureRule::from_table(
                dimension,
                crate::SystemQuadrature::Barycenter
                    .table(dimension)
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(barycenter.id, "simplex-barycenter");
            assert_eq!(barycenter.degree, 1);
        }
        for dimension in 1..=3 {
            let p1 = QuadratureRule::from_table(
                dimension,
                PreparedElement::linear_simplex(dimension)
                    .unwrap()
                    .quadrature()
                    .to_vec(),
            )
            .unwrap();
            assert_eq!(p1.id, "simplex-barycenter");
        }
    }

    #[test]
    fn a_degree_two_rule_integrates_dot_u_u_of_a_p1_interpolant_exactly_and_barycenter_does_not() {
        // Hand case: u = (x, y) on the reference triangle; integral of x^2 + y^2 is 1/6, the
        // barycenter rule gives 1/9 (a 33 % relative error).
        let reference = Mesh::new(
            2,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![Cell {
                vertices: vec![VertexId(0), VertexId(1), VertexId(2)],
            }],
        )
        .unwrap();
        let values = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let sampler = FieldSampler::new(
            &reference,
            SampledFamily::Lagrange {
                order: 1,
                components: 2,
            },
            &values,
        )
        .unwrap();
        let dot_u_u =
            |sampler: &FieldSampler<'_>, cell: CellId, point: &PhysicalQuadraturePoint| {
                let u = sampler.value_at(cell, &point.physical)?;
                Ok(u.iter().map(|u| u * u).sum::<f64>())
            };
        let barycenter =
            QuadratureView::new(&reference, QuadratureRule::barycenter(2).unwrap()).unwrap();
        let exact = barycenter.rule_for_degree(2).unwrap();
        assert_eq!(exact.rule().id, "triangle-edge-midpoints");
        assert_near(
            exact
                .integrate(|cell, point| dot_u_u(&sampler, cell, point))
                .unwrap(),
            1.0 / 6.0,
            1.0e-15,
            "exact dot(u, u)",
        );
        assert_near(
            barycenter
                .integrate(|cell, point| dot_u_u(&sampler, cell, point))
                .unwrap(),
            1.0 / 9.0,
            1.0e-15,
            "barycenter dot(u, u)",
        );

        // Sheared mesh, affine vector field: degree-2 and degree-5 rules agree to roundoff;
        // the barycenter rule is off by a recorded relative error.
        let mesh = sheared_triangles();
        let mut values = vec![0.0; mesh.vertices().len() * 2];
        for (vertex, x) in mesh.vertices().iter().enumerate() {
            for component in 0..2 {
                values[vertex * 2 + component] = affine_field(2, component, x).0;
            }
        }
        let sampler = FieldSampler::new(
            &mesh,
            SampledFamily::Lagrange {
                order: 1,
                components: 2,
            },
            &values,
        )
        .unwrap();
        let view = QuadratureView::new(&mesh, QuadratureRule::barycenter(2).unwrap()).unwrap();
        let degree_two = view
            .rule_for_degree(2)
            .unwrap()
            .integrate(|cell, point| dot_u_u(&sampler, cell, point))
            .unwrap();
        let degree_five = view
            .rule_for_degree(5)
            .unwrap()
            .integrate(|cell, point| dot_u_u(&sampler, cell, point))
            .unwrap();
        let barycenter = view
            .integrate(|cell, point| dot_u_u(&sampler, cell, point))
            .unwrap();
        assert_near(
            degree_two,
            degree_five,
            1.0e-14 * degree_five.abs(),
            "degree 2 vs 5",
        );
        let relative_error = (barycenter - degree_five).abs() / degree_five.abs();
        assert!(
            relative_error > 1.0e-3,
            "the barycenter rule should mis-integrate a degree-2 integrand, error {relative_error}"
        );
        // The physical weights of every cell sum to its measure.
        for cell in 0..mesh.cells().len() {
            let weights = view
                .rule_for_degree(4)
                .unwrap()
                .cell_points(CellId(cell))
                .unwrap()
                .iter()
                .map(|point| point.weight)
                .sum::<f64>();
            assert_near(
                weights,
                cell_measure(&mesh, CellId(cell)).unwrap(),
                1.0e-15,
                "weights",
            );
        }
    }

    #[test]
    fn quadrature_views_refuse_dimension_mismatch_and_non_finite_integrands() {
        let mesh = sheared_triangles();
        assert!(matches!(
            QuadratureView::new(&mesh, QuadratureRule::barycenter(3).unwrap()),
            Err(FinitumError::InvalidRealization(_))
        ));
        let view = QuadratureView::new(&mesh, QuadratureRule::barycenter(2).unwrap()).unwrap();
        assert!(matches!(
            view.integrate(|_, _| Ok(f64::NAN)),
            Err(FinitumError::InvalidRealization(_))
        ));
        assert!(view.cell_points(CellId(5)).is_err());
    }
}
