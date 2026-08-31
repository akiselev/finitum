//! GX-C1/GX-C2: structured `MeshProfile` realization, region tags derived from
//! [`crate::FacetTopology`], and tag-driven constraints/partitions.
//!
//! `MeshProfile::SimplexBox` realizes deterministic structured simplex meshes
//! on an axis-aligned box (1-D segments, 2-D two-triangles-per-quad, 3-D Kuhn
//! six-tet bricks) with positive orientation and exterior facets tagged by
//! box face. `MeshProfile::CadRectangle`/`CadFamily` wrap the existing
//! [`CadGeometryRealization`] providers and derive tags from their boundary
//! and region associations instead of duplicating their geometry.
//!
//! `RegionTags` sits on top of [`FacetTopology`] rather than replacing it:
//! every tag is either a per-cell region label or a set of exterior facet
//! identities, so `essential_constraints_from` and `check_boundary_partition`
//! discharge Scientia's region-keyed requirements against real topology
//! instead of caller-supplied index arithmetic.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cadabra_provider::{AnalyticProvider, RectangleProvider};
use malleus::{BufferBinding, ExecutableModule, Interpreter, OperandId, validate_module};
use scientia::Digest;
use serde::Serialize;

use crate::{
    AffineConstraint, CadGeometryRealization, Cell, ConstraintSet, DofId, DofMap, FacetId,
    FacetTopology, FinitumError, Mesh, VertexId,
};

const MESH_PROFILE_ITEM_CAP: usize = 1_000_000;

/// Finitum-owned stable region/boundary label.
///
/// `RegionMap` maps Scientia [`scientia::RegionId`]s onto sets of these; the
/// realized [`RegionTags`] never stores a Scientia region identity directly.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RegionTagId(pub String);

impl RegionTagId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RegionTagId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Per-cell region labels and per-exterior-facet boundary labels for one mesh.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RegionTags {
    /// One region tag per cell, in mesh cell order.
    pub cell_regions: Vec<RegionTagId>,
    /// Exterior facets grouped by boundary tag.
    pub facet_regions: BTreeMap<RegionTagId, Vec<FacetId>>,
}

/// How a [`TaggedMesh`] was produced.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum MeshProvenance {
    /// Realized directly from a [`MeshProfile`]; refinement level zero.
    Profile,
    /// Produced by [`refine_uniform`] from a parent whose digest and level are retained.
    Refined { parent: Digest, level: u32 },
}

impl MeshProvenance {
    pub fn level(&self) -> u32 {
        match self {
            MeshProvenance::Profile => 0,
            MeshProvenance::Refined { level, .. } => *level,
        }
    }
}

/// A concrete mesh with region tags, a canonical identity, and its construction history.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TaggedMesh {
    pub mesh: Mesh,
    pub tags: RegionTags,
    pub digest: Digest,
    pub provenance: MeshProvenance,
}

/// Deterministic structured mesh recipes and CAD-provider realizations under one contract.
#[derive(Clone, Debug, PartialEq)]
pub enum MeshProfile {
    /// Structured simplex box: 1-D segments, 2-D two-triangles-per-quad, 3-D Kuhn six-tet bricks.
    SimplexBox {
        dimension: u8,
        /// `[min, max]` per axis, length `dimension`.
        extent: Vec<[f64; 2]>,
        /// Cell subdivisions per axis, length `dimension`.
        subdivisions: Vec<usize>,
    },
    /// Delegates to [`CadGeometryRealization::from_rectangle`].
    CadRectangle {
        provider: RectangleProvider,
        expected_revision: u64,
        subdivisions: [usize; 2],
    },
    /// Delegates to [`CadGeometryRealization::from_family`].
    CadFamily {
        provider: AnalyticProvider,
        expected_revision: u64,
        subdivisions: [usize; 2],
    },
}

/// Realizes one [`MeshProfile`] into a [`TaggedMesh`].
pub fn realize(profile: &MeshProfile) -> Result<TaggedMesh, FinitumError> {
    match profile {
        MeshProfile::SimplexBox {
            dimension,
            extent,
            subdivisions,
        } => realize_simplex_box(*dimension, extent, subdivisions),
        MeshProfile::CadRectangle {
            provider,
            expected_revision,
            subdivisions,
        } => {
            let geometry = CadGeometryRealization::from_rectangle(
                provider,
                *expected_revision,
                *subdivisions,
            )?;
            cad_tagged_mesh(&geometry)
        }
        MeshProfile::CadFamily {
            provider,
            expected_revision,
            subdivisions,
        } => {
            let geometry =
                CadGeometryRealization::from_family(provider, *expected_revision, *subdivisions)?;
            cad_tagged_mesh(&geometry)
        }
    }
}

/// Red refinement of a [`TaggedMesh`]: every cell splits into `2^dimension` children, tags are
/// inherited (a child facet on a tagged parent facet keeps the parent tag), and the provenance
/// chain records the parent digest and an incremented level.
pub fn refine_uniform(mesh: &TaggedMesh) -> Result<TaggedMesh, FinitumError> {
    let dimension = mesh.mesh.dimension();
    if !(1..=3).contains(&dimension) {
        return Err(FinitumError::MeshProfileUnsupported(format!(
            "uniform refinement is defined for dimensions 1..=3, got {dimension}"
        )));
    }
    let expected_arity = dimension + 1;
    if mesh
        .mesh
        .cells()
        .iter()
        .any(|cell| cell.vertices.len() != expected_arity)
    {
        return Err(FinitumError::MeshProfileUnsupported(
            "uniform refinement requires a simplex mesh".into(),
        ));
    }
    if mesh.tags.cell_regions.len() != mesh.mesh.cells().len() {
        return Err(FinitumError::MeshProfileUnsupported(
            "region tags do not cover every mesh cell".into(),
        ));
    }

    let mut edges = BTreeSet::<[usize; 2]>::new();
    for cell in mesh.mesh.cells() {
        for left in 0..cell.vertices.len() {
            for right in left + 1..cell.vertices.len() {
                edges.insert(sorted_edge(cell.vertices[left].0, cell.vertices[right].0));
            }
        }
    }
    let ordered_edges = edges.into_iter().collect::<Vec<_>>();
    let original_vertex_count = mesh.mesh.vertices().len();
    let children_per_cell = match dimension {
        1 => 2,
        2 => 4,
        3 => 8,
        _ => unreachable!("dimension was checked"),
    };
    let new_vertex_count = original_vertex_count
        .checked_add(ordered_edges.len())
        .ok_or_else(|| {
            FinitumError::MeshProfileUnsupported("refined vertex count overflows usize".into())
        })?;
    let new_cell_count = mesh
        .mesh
        .cells()
        .len()
        .checked_mul(children_per_cell)
        .ok_or_else(|| {
            FinitumError::MeshProfileUnsupported("refined cell count overflows usize".into())
        })?;
    if new_vertex_count > MESH_PROFILE_ITEM_CAP || new_cell_count > MESH_PROFILE_ITEM_CAP {
        return Err(FinitumError::MeshProfileUnsupported(
            "uniform refinement exceeds the one-million-item work cap".into(),
        ));
    }

    let mut edge_midpoint = BTreeMap::<[usize; 2], usize>::new();
    let mut new_vertices = mesh.mesh.vertices().to_vec();
    for (offset, edge) in ordered_edges.iter().enumerate() {
        edge_midpoint.insert(*edge, original_vertex_count + offset);
        let a = &mesh.mesh.vertices()[edge[0]];
        let b = &mesh.mesh.vertices()[edge[1]];
        new_vertices.push(a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).collect());
    }

    let mut new_cells = Vec::with_capacity(new_cell_count);
    let mut new_cell_regions = Vec::with_capacity(new_cell_count);
    for (cell_index, cell) in mesh.mesh.cells().iter().enumerate() {
        let region = mesh.tags.cell_regions[cell_index].clone();
        let children = match dimension {
            1 => refine_segment(cell, &edge_midpoint),
            2 => refine_triangle(cell, &edge_midpoint),
            3 => refine_tet(cell, &edge_midpoint),
            _ => unreachable!("dimension was checked"),
        };
        for child in children {
            new_cells.push(child);
            new_cell_regions.push(region.clone());
        }
    }
    let new_mesh = Mesh::new(dimension, new_vertices, new_cells)?;

    let old_facets = FacetTopology::from_mesh(&mesh.mesh)?;
    let mut old_facet_tag = BTreeMap::<FacetId, RegionTagId>::new();
    for (tag, facet_ids) in &mesh.tags.facet_regions {
        for facet_id in facet_ids {
            old_facet_tag.insert(*facet_id, tag.clone());
        }
    }
    let old_exterior = old_facets
        .exterior()
        .map(|facet| (facet.id, facet))
        .collect::<Vec<_>>();

    let new_facets = FacetTopology::from_mesh(&new_mesh)?;
    let mut new_facet_regions = BTreeMap::<RegionTagId, Vec<FacetId>>::new();
    for facet in new_facets.exterior() {
        let mut unfolded = BTreeSet::<usize>::new();
        for vertex in &facet.vertices {
            if vertex.0 < original_vertex_count {
                unfolded.insert(vertex.0);
            } else if let Some(edge) = ordered_edges.get(vertex.0 - original_vertex_count) {
                unfolded.insert(edge[0]);
                unfolded.insert(edge[1]);
            }
        }
        for (old_id, old_facet) in &old_exterior {
            let old_vertices = old_facet
                .vertices
                .iter()
                .map(|vertex| vertex.0)
                .collect::<BTreeSet<_>>();
            if unfolded.is_subset(&old_vertices) {
                if let Some(tag) = old_facet_tag.get(old_id) {
                    new_facet_regions
                        .entry(tag.clone())
                        .or_default()
                        .push(facet.id);
                }
                break;
            }
        }
    }
    for facet_ids in new_facet_regions.values_mut() {
        facet_ids.sort();
        facet_ids.dedup();
    }

    let new_tags = RegionTags {
        cell_regions: new_cell_regions,
        facet_regions: new_facet_regions,
    };
    let provenance = MeshProvenance::Refined {
        parent: mesh.digest.clone(),
        level: mesh.provenance.level() + 1,
    };
    let digest = tagged_mesh_digest(&new_mesh, &new_tags, &provenance);
    Ok(TaggedMesh {
        mesh: new_mesh,
        tags: new_tags,
        digest,
        provenance,
    })
}

/// Caller-supplied mapping from a Scientia [`scientia::RegionId`] to the set of
/// [`RegionTagId`]s that realize it on one [`TaggedMesh`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegionMap(BTreeMap<scientia::RegionId, BTreeSet<RegionTagId>>);

impl RegionMap {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Adds the given tags to the set mapped for `region`.
    pub fn insert(
        &mut self,
        region: scientia::RegionId,
        tags: impl IntoIterator<Item = RegionTagId>,
    ) -> &mut Self {
        self.0.entry(region).or_default().extend(tags);
        self
    }

    pub fn tags(&self, region: scientia::RegionId) -> Option<&BTreeSet<RegionTagId>> {
        self.0.get(&region)
    }
}

/// One non-basis essential value source: a constant, a caller-projected nodal field, a
/// coordinate sampler, a Scientia property table, or a validated Scientia property kernel
/// (GX-C3, contract C7). `identity()` returns a canonical digest of the source, replacing the
/// caller-invented strings dynamic bindings used before this landed.
type SampledFieldFn = dyn Fn(&[f64]) -> Vec<f64> + Send + Sync;

#[derive(Clone)]
pub enum FieldSource {
    /// One value per component, applied uniformly to every tagged vertex.
    Constant(Vec<f64>),
    /// A caller-projected vertex-major field, `components` entries per vertex.
    Nodal(Vec<f64>),
    /// A coordinate-driven evaluator returning `components` values per call.
    Sampled(Arc<SampledFieldFn>),
    /// A Scientia interpolation table, validated for shape/finiteness at construction.
    Table(scientia::PropertyTable),
    /// A Scientia property kernel, revalidated through Malleus at construction and bound to its
    /// executable module so repeated evaluation does not re-validate.
    Kernel {
        kernel: scientia::PropertyKernel,
        executable: ExecutableModule,
    },
}

impl std::fmt::Debug for FieldSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldSource::Constant(values) => {
                formatter.debug_tuple("Constant").field(values).finish()
            }
            FieldSource::Nodal(values) => formatter.debug_tuple("Nodal").field(values).finish(),
            FieldSource::Sampled(_) => formatter.debug_tuple("Sampled").field(&"<fn>").finish(),
            FieldSource::Table(table) => formatter.debug_tuple("Table").field(table).finish(),
            FieldSource::Kernel { kernel, .. } => formatter
                .debug_struct("Kernel")
                .field("kernel", &kernel.identity)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Serialize)]
struct FieldSourceDigestPayload<'a, T: Serialize> {
    schema: &'static str,
    kind: &'static str,
    data: &'a T,
}

fn field_source_digest<T: Serialize>(kind: &'static str, data: &T) -> Digest {
    let payload = FieldSourceDigestPayload {
        schema: "finitum.field-source/1",
        kind,
        data,
    };
    Digest::blake3(&serde_json::to_vec(&payload).expect("field source payload is serializable"))
}

impl FieldSource {
    pub fn constant(components: impl Into<Vec<f64>>) -> Self {
        Self::Constant(components.into())
    }

    pub fn nodal(values: impl Into<Vec<f64>>) -> Self {
        Self::Nodal(values.into())
    }

    pub fn sampled(sampler: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static) -> Self {
        Self::Sampled(Arc::new(sampler))
    }

    /// Validates axis/value shape, monotone finite axis points, and finite table values.
    pub fn table(table: scientia::PropertyTable) -> Result<Self, FinitumError> {
        validate_property_table(&table)?;
        Ok(Self::Table(table))
    }

    /// Revalidates every kernel in `kernel.module` through Malleus before wrapping it, per
    /// contract C7 ("kernel module revalidated via malleus::validate before wrapping").
    pub fn kernel(kernel: scientia::PropertyKernel) -> Result<Self, FinitumError> {
        let module = validate_module(kernel.module.clone())
            .map_err(|error| FinitumError::KernelValidation(error.to_string()))?;
        let executable = ExecutableModule::reference(module);
        if kernel.value_kernel >= executable.kernels().len() {
            return Err(FinitumError::InvalidRealization(
                "property kernel value_kernel index is out of range".into(),
            ));
        }
        for tangent in &kernel.tangents {
            if tangent.kernel >= executable.kernels().len() {
                return Err(FinitumError::InvalidRealization(format!(
                    "property kernel tangent for input {:?} has an out-of-range kernel index",
                    tangent.input
                )));
            }
        }
        Ok(Self::Kernel { kernel, executable })
    }

    /// Canonical digest of this source: for `Kernel`, the `PropertyKernel` identity Scientia
    /// already computed; for `Table`, a digest of its axes/values/policies; for `Constant`/
    /// `Nodal`, a digest of the owned data. `Sampled` wraps an opaque closure with no structural
    /// identity, so its digest distinguishes closures only within one process run (the `Arc`'s
    /// address), not across runs or processes.
    pub fn identity(&self) -> Digest {
        match self {
            FieldSource::Constant(values) => field_source_digest("constant", values),
            FieldSource::Nodal(values) => field_source_digest("nodal", values),
            FieldSource::Sampled(sampler) => {
                let marker = format!("sampled:{:p}", Arc::as_ptr(sampler));
                Digest::blake3(marker.as_bytes())
            }
            FieldSource::Table(table) => field_source_digest("table", table),
            FieldSource::Kernel { kernel, .. } => kernel.identity.clone(),
        }
    }
}

/// Runs one scalar-in/scalar-out Malleus kernel: `inputs.len()` leading read operands followed
/// by one write operand, matching the layout `lower_property_kernel` emits (contract C7).
fn run_scalar_kernel(
    executable: &malleus::Executable,
    inputs: &[f64],
) -> Result<f64, FinitumError> {
    let kernel = executable.kernel().as_kernel();
    if kernel.operands.len() != inputs.len() + 1 {
        return Err(FinitumError::InvalidRealization(format!(
            "property kernel expects {} inputs, got {}",
            kernel.operands.len().saturating_sub(1),
            inputs.len()
        )));
    }
    let mut buffers = kernel
        .operands
        .iter()
        .map(|operand| vec![0.0; operand.region.offset + operand.region.length])
        .collect::<Vec<_>>();
    for (index, value) in inputs.iter().enumerate() {
        let definition = &kernel.operands[index];
        buffers[index][definition.region.offset] = *value;
    }
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(executable, &mut bindings)
        .map_err(|error| FinitumError::KernelExecution(error.to_string()))?;
    drop(bindings);
    let output_index = kernel.operands.len() - 1;
    let output_definition = &kernel.operands[output_index];
    Ok(buffers[output_index][output_definition.region.offset])
}

/// Evaluates a [`FieldSource::Kernel`]'s value kernel. `inputs_by_name` must supply exactly one
/// value per `kernel.inputs` entry, matched by name; the caller resolves each declared input's
/// meaning (coordinate, time, or active field trace) before calling, since Scientia's
/// `SlotInput` carries no role marker distinguishing them (contract C7).
pub(crate) fn evaluate_kernel_value(
    kernel: &scientia::PropertyKernel,
    executable: &ExecutableModule,
    inputs_by_name: &BTreeMap<String, f64>,
) -> Result<f64, FinitumError> {
    let ordered = ordered_kernel_inputs(kernel, inputs_by_name)?;
    let value_kernel = executable
        .kernels()
        .get(kernel.value_kernel)
        .ok_or_else(|| {
            FinitumError::InvalidRealization(
                "property kernel value_kernel index is out of range".into(),
            )
        })?;
    run_scalar_kernel(value_kernel, &ordered)
}

/// Evaluates the partial derivative of a [`FieldSource::Kernel`]'s value with respect to one
/// declared input, as a function of the same primal inputs (contract C7 tangent kernels are
/// closed-form partials, not JVP-style directional propagators). Returns `Ok(None)` when the
/// input's `DerivativeContract` supplied no tangent kernel (`Piecewise`/`AnalyticProvided`/
/// `NumericalAllowed`/`None`); callers building a state-dependent chain rule must refuse rather
/// than trust an absent tangent (`REALIZATION_TANGENT_UNAVAILABLE`).
pub(crate) fn evaluate_kernel_partial(
    kernel: &scientia::PropertyKernel,
    executable: &ExecutableModule,
    inputs_by_name: &BTreeMap<String, f64>,
    with_respect_to: &str,
) -> Result<Option<f64>, FinitumError> {
    let Some(tangent) = kernel
        .tangents
        .iter()
        .find(|tangent| tangent.input == with_respect_to)
    else {
        return Ok(None);
    };
    let ordered = ordered_kernel_inputs(kernel, inputs_by_name)?;
    let tangent_kernel = executable.kernels().get(tangent.kernel).ok_or_else(|| {
        FinitumError::InvalidRealization(format!(
            "property kernel tangent for input {with_respect_to:?} has an out-of-range kernel index"
        ))
    })?;
    Ok(Some(run_scalar_kernel(tangent_kernel, &ordered)?))
}

fn ordered_kernel_inputs(
    kernel: &scientia::PropertyKernel,
    inputs_by_name: &BTreeMap<String, f64>,
) -> Result<Vec<f64>, FinitumError> {
    kernel
        .inputs
        .iter()
        .map(|input| {
            inputs_by_name.get(&input.name).copied().ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "property kernel input {:?} was not supplied",
                    input.name
                ))
            })
        })
        .collect()
}

fn validate_property_table(table: &scientia::PropertyTable) -> Result<(), FinitumError> {
    if table.axes.is_empty() {
        return Err(FinitumError::InvalidRealization(
            "property table must declare at least one axis".into(),
        ));
    }
    let mut expected = 1usize;
    for axis in &table.axes {
        if axis.points.len() < 2 {
            return Err(FinitumError::InvalidRealization(format!(
                "property table axis {:?} needs at least two points",
                axis.name
            )));
        }
        if axis.points.iter().any(|point| !point.is_finite())
            || !axis.points.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(FinitumError::InvalidRealization(format!(
                "property table axis {:?} points must be finite and strictly increasing",
                axis.name
            )));
        }
        expected = expected.checked_mul(axis.points.len()).ok_or_else(|| {
            FinitumError::InvalidRealization("property table extent overflows usize".into())
        })?;
    }
    if table.values.len() != expected {
        return Err(FinitumError::InvalidRealization(format!(
            "property table has {} values, expected {expected} for its axis grid",
            table.values.len()
        )));
    }
    if table.values.iter().any(|value| !value.is_finite()) {
        return Err(FinitumError::InvalidRealization(
            "property table contains a non-finite value".into(),
        ));
    }
    Ok(())
}

/// Multilinear interpolation (reducing to linear for one axis) of `table` at `point`, one
/// coordinate per declared axis in `table.axes` order. Out-of-range coordinates are handled per
/// `table.out_of_range`: `Error` refuses, `Warn` clamps into range (Finitum has no separate
/// warning channel, so this is a silent clamp), and `ExplicitExtrapolation` extends the boundary
/// segment's slope. `Interpolation::MonotoneCubic` is not yet implemented and is refused.
pub(crate) fn evaluate_table_value(
    table: &scientia::PropertyTable,
    point: &[f64],
) -> Result<f64, FinitumError> {
    if matches!(
        table.interpolation,
        scientia::scientific::Interpolation::MonotoneCubic
    ) {
        return Err(FinitumError::UnsupportedRealization(
            "monotone cubic table interpolation is not yet implemented".into(),
        ));
    }
    if point.len() != table.axes.len() {
        return Err(FinitumError::InvalidRealization(format!(
            "property table point has {} coordinates, expected {}",
            point.len(),
            table.axes.len()
        )));
    }
    let mut segments = Vec::with_capacity(table.axes.len());
    for (axis, coordinate) in table.axes.iter().zip(point) {
        segments.push(table_axis_segment(axis, *coordinate, &table.out_of_range)?);
    }
    Ok(multilinear_evaluate(table, &segments, 0))
}

/// Exact partial derivative of [`evaluate_table_value`] with respect to axis `axis_index`
/// (piecewise-constant between grid points, matching `TableDerivativePolicy::PiecewiseConstantSlope`).
pub(crate) fn evaluate_table_slope(
    table: &scientia::PropertyTable,
    point: &[f64],
    axis_index: usize,
) -> Result<f64, FinitumError> {
    if point.len() != table.axes.len() || axis_index >= table.axes.len() {
        return Err(FinitumError::InvalidRealization(
            "property table slope request has an invalid axis index or point extent".into(),
        ));
    }
    let mut segments = Vec::with_capacity(table.axes.len());
    for (axis, coordinate) in table.axes.iter().zip(point) {
        segments.push(table_axis_segment(axis, *coordinate, &table.out_of_range)?);
    }
    Ok(multilinear_slope(table, &segments, axis_index, 0))
}

/// One axis's bracketing lower grid index and interpolation fraction `t` in `[0, 1]`
/// (extrapolation may push `t` outside that range under `ExplicitExtrapolation`).
struct AxisSegment {
    lower: usize,
    fraction: f64,
}

fn table_axis_segment(
    axis: &scientia::scientific::TableAxis,
    coordinate: f64,
    policy: &scientia::scientific::OutOfValidityPolicy,
) -> Result<AxisSegment, FinitumError> {
    if !coordinate.is_finite() {
        return Err(FinitumError::InvalidRealization(format!(
            "property table axis {:?} coordinate is not finite",
            axis.name
        )));
    }
    let points = &axis.points;
    let min = points[0];
    let max = points[points.len() - 1];
    let clamped = if coordinate < min || coordinate > max {
        match policy {
            scientia::scientific::OutOfValidityPolicy::Error => {
                return Err(FinitumError::InvalidRealization(format!(
                    "property table axis {:?} coordinate {coordinate} is outside [{min}, {max}]",
                    axis.name
                )));
            }
            scientia::scientific::OutOfValidityPolicy::Warn => coordinate.clamp(min, max),
            scientia::scientific::OutOfValidityPolicy::ExplicitExtrapolation(_) => coordinate,
        }
    } else {
        coordinate
    };
    let lower = match points.binary_search_by(|probe| probe.total_cmp(&clamped)) {
        Ok(index) => index.min(points.len() - 2),
        Err(index) => index.saturating_sub(1).min(points.len() - 2),
    };
    let span = points[lower + 1] - points[lower];
    let fraction = if span.abs() <= f64::EPSILON {
        0.0
    } else {
        (clamped - points[lower]) / span
    };
    Ok(AxisSegment { lower, fraction })
}

fn multilinear_evaluate(
    table: &scientia::PropertyTable,
    segments: &[AxisSegment],
    axis: usize,
) -> f64 {
    if axis == segments.len() {
        return table_value_at(table, segments);
    }
    let low = corner_value(table, segments, axis, false);
    let high = corner_value(table, segments, axis, true);
    low + segments[axis].fraction * (high - low)
}

fn multilinear_slope(
    table: &scientia::PropertyTable,
    segments: &[AxisSegment],
    target_axis: usize,
    axis: usize,
) -> f64 {
    if axis == segments.len() {
        return table_value_at(table, segments);
    }
    if axis == target_axis {
        let low = corner_value(table, segments, axis, false);
        let high = corner_value(table, segments, axis, true);
        let span_index = segments[axis].lower;
        let span = table.axes[axis].points[span_index + 1] - table.axes[axis].points[span_index];
        return if span.abs() <= f64::EPSILON {
            0.0
        } else {
            (high - low) / span
        };
    }
    let low = corner_slope(table, segments, axis, false, target_axis);
    let high = corner_slope(table, segments, axis, true, target_axis);
    low + segments[axis].fraction * (high - low)
}

/// Value at one grid corner along `axis` (`high` selects `lower+1` instead of `lower`), resolved
/// through the remaining axes by full multilinear interpolation.
fn corner_value(
    table: &scientia::PropertyTable,
    segments: &[AxisSegment],
    axis: usize,
    high: bool,
) -> f64 {
    let pinned = pin_axis(segments, axis, high);
    multilinear_evaluate(table, &pinned, axis + 1)
}

fn corner_slope(
    table: &scientia::PropertyTable,
    segments: &[AxisSegment],
    axis: usize,
    high: bool,
    target_axis: usize,
) -> f64 {
    let pinned = pin_axis(segments, axis, high);
    multilinear_slope(table, &pinned, target_axis, axis + 1)
}

fn pin_axis(segments: &[AxisSegment], axis: usize, high: bool) -> Vec<AxisSegment> {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            if index == axis {
                AxisSegment {
                    lower: segment.lower,
                    fraction: if high { 1.0 } else { 0.0 },
                }
            } else {
                AxisSegment {
                    lower: segment.lower,
                    fraction: segment.fraction,
                }
            }
        })
        .collect()
}

/// Row-major flat index into `table.values`: axis 0 varies slowest, the last axis fastest. Every
/// segment reaching this point has been pinned by an enclosing `pin_axis` call, so its fraction
/// is exactly `0.0` (lower corner) or `1.0` (upper corner).
fn table_value_at(table: &scientia::PropertyTable, segments: &[AxisSegment]) -> f64 {
    let mut flat = 0usize;
    let mut multiplier = 1usize;
    for axis_index in (0..table.axes.len()).rev() {
        let segment = &segments[axis_index];
        let corner = segment.lower + if segment.fraction >= 1.0 { 1 } else { 0 };
        flat += corner * multiplier;
        multiplier *= table.axes[axis_index].points.len();
    }
    table.values[flat]
}

/// Which components of a [`FieldSource`]'s evaluated value one
/// `scientia::EssentialConstraintRequirement` actually constrains (GX-C6). The
/// source still evaluates every component (a vector `FieldSource` is not reshaped), but only the
/// selected component indices become DOF rows; the rest are left free.
#[derive(Clone, Debug, PartialEq)]
pub enum ComponentSelection {
    /// Every component of the evaluated value is constrained.
    All,
    /// Only these zero-based component indices are constrained; each must be `< components`.
    Only(Vec<usize>),
}

/// Derives essential (Dirichlet) constraints from region tags rather than positional CAD ids or
/// index arithmetic. `dof_map` must be a vertex-major scalar or vector map (see
/// [`crate::vector_nodal_dof_map`]); `values` supplies one [`FieldSource`] per requirement.
/// Every component of every tagged vertex is constrained; see
/// [`essential_constraints_from_selected`] to constrain a subset of a vector field's components.
pub fn essential_constraints_from(
    mesh: &TaggedMesh,
    dof_map: &DofMap,
    requirements: &[scientia::EssentialConstraintRequirement],
    region_map: &RegionMap,
    values: &[FieldSource],
) -> Result<ConstraintSet, FinitumError> {
    essential_constraints_from_selected(
        mesh,
        dof_map,
        requirements,
        region_map,
        values,
        &vec![ComponentSelection::All; requirements.len()],
    )
}

/// As [`essential_constraints_from`], with an explicit per-requirement [`ComponentSelection`]
/// (GX-C6). `values` accepts [`FieldSource::Table`]/[`FieldSource::Kernel`] alongside the landed
/// variants: both evaluate as scalars (`components` must be `1`) at the tagged vertex's physical
/// coordinates and time `0.0` (`Kernel` inputs are resolved by name from `{"x", "y", "z", "t",
/// "time"}` covering the mesh's coordinate axes; any other declared input name is refused, since
/// a boundary value has no active-field state to supply). A time-dependent boundary kernel's
/// identity ([`FieldSource::identity`]) still changes with the kernel, so callers that need a
/// different time rebuild the constraint set with a fresh evaluation rather than mutating this
/// one; the `t = 0.0` evaluation is a fixed convention, not a runtime parameter of this function.
pub fn essential_constraints_from_selected(
    mesh: &TaggedMesh,
    dof_map: &DofMap,
    requirements: &[scientia::EssentialConstraintRequirement],
    region_map: &RegionMap,
    values: &[FieldSource],
    selection: &[ComponentSelection],
) -> Result<ConstraintSet, FinitumError> {
    if requirements.len() != values.len() || requirements.len() != selection.len() {
        return Err(FinitumError::InvalidRealization(format!(
            "essential constraint requirements has {} entries but {} field sources and {} \
             component selections were supplied",
            requirements.len(),
            values.len(),
            selection.len()
        )));
    }
    let vertex_count = mesh.mesh.vertices().len();
    if vertex_count == 0 || dof_map.dof_count() % vertex_count != 0 {
        return Err(FinitumError::InvalidRealization(
            "degree-of-freedom map is not a vertex-major layout over the tagged mesh".into(),
        ));
    }
    let components = dof_map.dof_count() / vertex_count;
    let facets = FacetTopology::from_mesh(&mesh.mesh)?;
    let facet_vertices = facets
        .facets()
        .iter()
        .map(|facet| (facet.id, facet.vertices.as_slice()))
        .collect::<BTreeMap<_, _>>();

    let mut recorded = BTreeMap::<DofId, (String, f64)>::new();
    for ((requirement, source), selection) in requirements.iter().zip(values).zip(selection) {
        let selected_components = match selection {
            ComponentSelection::All => (0..components).collect::<BTreeSet<_>>(),
            ComponentSelection::Only(indices) => {
                let mut selected = BTreeSet::new();
                for &index in indices {
                    if index >= components {
                        return Err(FinitumError::InvalidRealization(format!(
                            "component selection index {index} is outside the {components}-component field"
                        )));
                    }
                    selected.insert(index);
                }
                selected
            }
        };
        let tags = region_map
            .tags(requirement.region)
            .filter(|tags| !tags.is_empty())
            .ok_or_else(|| {
                FinitumError::RealizationRegionUnmapped(format!("{:?}", requirement.region))
            })?;
        let mut vertices = BTreeSet::<VertexId>::new();
        for tag in tags {
            if let Some(facet_ids) = mesh.tags.facet_regions.get(tag) {
                for facet_id in facet_ids {
                    if let Some(facet_vertex_ids) = facet_vertices.get(facet_id) {
                        vertices.extend(facet_vertex_ids.iter().copied());
                    }
                }
            }
        }
        let tag_label = tags
            .iter()
            .next()
            .map(RegionTagId::to_string)
            .unwrap_or_default();
        for vertex in vertices {
            let coordinates = &mesh.mesh.vertices()[vertex.0];
            let evaluated = match source {
                FieldSource::Constant(constant) => constant.clone(),
                FieldSource::Nodal(nodal) => {
                    let start = vertex.0 * components;
                    if nodal.len() < start + components {
                        return Err(FinitumError::InvalidRealization(
                            "nodal field source does not cover every tagged vertex".into(),
                        ));
                    }
                    nodal[start..start + components].to_vec()
                }
                FieldSource::Sampled(sampler) => sampler(coordinates),
                FieldSource::Table(table) => {
                    let point = named_axis_point(&table.axes, coordinates, 0.0)?;
                    vec![evaluate_table_value(table, &point)?]
                }
                FieldSource::Kernel { kernel, executable } => {
                    let inputs = named_coordinate_inputs(coordinates, 0.0);
                    vec![evaluate_kernel_value(kernel, executable, &inputs)?]
                }
            };
            if evaluated.len() != components {
                return Err(FinitumError::InvalidRealization(format!(
                    "essential value has {} components, expected {components}",
                    evaluated.len()
                )));
            }
            for (component, value) in evaluated.iter().copied().enumerate() {
                if !selected_components.contains(&component) {
                    continue;
                }
                if !value.is_finite() {
                    return Err(FinitumError::InvalidRealization(
                        "essential value is not finite".into(),
                    ));
                }
                let dof = DofId(vertex.0 * components + component);
                if let Some((other_tag, other_value)) = recorded.get(&dof) {
                    if other_value.to_bits() != value.to_bits() {
                        return Err(FinitumError::ConflictingRegionValue {
                            left: other_tag.clone(),
                            right: tag_label.clone(),
                            vertex: vertex.0,
                        });
                    }
                } else {
                    recorded.insert(dof, (tag_label.clone(), value));
                }
            }
        }
    }
    ConstraintSet::new(
        dof_map.dof_count(),
        recorded
            .into_iter()
            .map(|(target, (_, offset))| AffineConstraint {
                target,
                dependencies: Vec::new(),
                offset,
            }),
    )
}

/// Builds the `{"x", "y", "z", "t", "time"}` named-input convention `evaluate_kernel_value` uses
/// for coordinate/time-only kernels (no active field trace is available for boundary data).
pub(crate) fn named_coordinate_inputs(coordinates: &[f64], time: f64) -> BTreeMap<String, f64> {
    const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];
    let mut inputs = BTreeMap::new();
    for (axis, value) in AXIS_NAMES.iter().zip(coordinates) {
        inputs.insert((*axis).to_string(), *value);
    }
    inputs.insert("t".to_string(), time);
    inputs.insert("time".to_string(), time);
    inputs
}

/// Resolves a [`scientia::PropertyTable`]'s axis point from the same named-coordinate convention
/// as [`named_coordinate_inputs`], in `table.axes` order.
fn named_axis_point(
    axes: &[scientia::scientific::TableAxis],
    coordinates: &[f64],
    time: f64,
) -> Result<Vec<f64>, FinitumError> {
    let named = named_coordinate_inputs(coordinates, time);
    axes.iter()
        .map(|axis| {
            named.get(&axis.name).copied().ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "property table axis {:?} is not one of the coordinate/time names \
                     {{x, y, z, t, time}}",
                    axis.name
                ))
            })
        })
        .collect()
}

/// GX-C4: resolves a set of Scientia `RegionId`s into concrete exterior `FacetId` lists through
/// `mesh`'s `RegionTags` and the caller-supplied `RegionMap`, for use with
/// [`crate::RealizationPlan::new_with_facets`]. Mirrors [`essential_constraints_from`]'s own
/// region-tag resolution exactly, so the same `RegionMap`/`TaggedMesh` drives both essential
/// constraints and exterior facet integrals consistently. Refuses an unmapped or empty-mapped
/// region (`FinitumError::RealizationRegionUnmapped`), matching `essential_constraints_from`.
pub fn facet_membership_from(
    mesh: &TaggedMesh,
    region_map: &RegionMap,
    regions: impl IntoIterator<Item = scientia::RegionId>,
) -> Result<BTreeMap<scientia::RegionId, Vec<FacetId>>, FinitumError> {
    let mut membership = BTreeMap::new();
    for region in regions {
        let tags = region_map
            .tags(region)
            .filter(|tags| !tags.is_empty())
            .ok_or_else(|| FinitumError::RealizationRegionUnmapped(format!("{region:?}")))?;
        let mut facet_ids = BTreeSet::new();
        for tag in tags {
            if let Some(ids) = mesh.tags.facet_regions.get(tag) {
                facet_ids.extend(ids.iter().copied());
            }
        }
        membership.insert(region, facet_ids.into_iter().collect::<Vec<_>>());
    }
    Ok(membership)
}

/// Result of a successful [`check_boundary_partition`] discharge.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartitionReport {
    pub exterior_facet_count: usize,
    pub region_count: usize,
}

/// Checks that every exterior facet of `mesh` is covered by exactly one region mapped from
/// `requirement.exterior_regions`. Refuses with [`FinitumError::RealizationPartitionFailed`]
/// carrying the uncovered/overlapping facet counts otherwise.
pub fn check_boundary_partition(
    mesh: &TaggedMesh,
    requirement: &scientia::BoundaryPartitionRequirement,
    region_map: &RegionMap,
) -> Result<PartitionReport, FinitumError> {
    partition_report_for(&mesh.mesh, &mesh.tags, requirement, region_map)
}

pub(crate) fn partition_report_for(
    mesh: &Mesh,
    tags: &RegionTags,
    requirement: &scientia::BoundaryPartitionRequirement,
    region_map: &RegionMap,
) -> Result<PartitionReport, FinitumError> {
    let facets = FacetTopology::from_mesh(mesh)?;
    let exterior_ids = facets
        .exterior()
        .map(|facet| facet.id)
        .collect::<BTreeSet<_>>();
    let mut coverage = BTreeMap::<FacetId, usize>::new();
    for region in &requirement.exterior_regions {
        let mapped = region_map
            .tags(*region)
            .filter(|tags| !tags.is_empty())
            .ok_or_else(|| FinitumError::RealizationRegionUnmapped(format!("{region:?}")))?;
        for tag in mapped {
            if let Some(facet_ids) = tags.facet_regions.get(tag) {
                for facet_id in facet_ids {
                    if exterior_ids.contains(facet_id) {
                        *coverage.entry(*facet_id).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    let mut uncovered = 0usize;
    let mut overlapping = 0usize;
    for facet_id in &exterior_ids {
        match coverage.get(facet_id).copied().unwrap_or(0) {
            0 => uncovered += 1,
            1 => {}
            _ => overlapping += 1,
        }
    }
    if uncovered > 0 || overlapping > 0 {
        return Err(FinitumError::RealizationPartitionFailed {
            uncovered,
            overlapping,
        });
    }
    Ok(PartitionReport {
        exterior_facet_count: exterior_ids.len(),
        region_count: requirement.exterior_regions.len(),
    })
}

fn realize_simplex_box(
    dimension: u8,
    extent: &[[f64; 2]],
    subdivisions: &[usize],
) -> Result<TaggedMesh, FinitumError> {
    let dim = dimension as usize;
    if !(1..=3).contains(&dim) {
        return Err(FinitumError::MeshProfileUnsupported(format!(
            "SimplexBox dimension must be in 1..=3, got {dimension}"
        )));
    }
    if extent.len() != dim {
        return Err(FinitumError::MeshProfileUnsupported(format!(
            "SimplexBox extent has {} axes, expected {dim}",
            extent.len()
        )));
    }
    if subdivisions.len() != dim {
        return Err(FinitumError::MeshProfileUnsupported(format!(
            "SimplexBox subdivisions has {} axes, expected {dim}",
            subdivisions.len()
        )));
    }
    for (axis, [min, max]) in extent.iter().enumerate() {
        if !min.is_finite() || !max.is_finite() || *max <= *min {
            return Err(FinitumError::MeshProfileUnsupported(format!(
                "SimplexBox extent on axis {axis} must be finite with max > min, got [{min}, {max}]"
            )));
        }
    }
    if subdivisions.contains(&0) {
        return Err(FinitumError::MeshProfileUnsupported(
            "SimplexBox subdivisions must be positive on every axis".into(),
        ));
    }

    let node_shape = subdivisions
        .iter()
        .map(|count| count + 1)
        .collect::<Vec<_>>();
    let node_count = node_shape
        .iter()
        .try_fold(1usize, |total, count| total.checked_mul(*count))
        .ok_or_else(|| {
            FinitumError::MeshProfileUnsupported("SimplexBox node count overflows usize".into())
        })?;
    let simplices_per_brick: usize = match dim {
        1 => 1,
        2 => 2,
        3 => 6,
        _ => unreachable!("dimension was checked"),
    };
    let brick_count = subdivisions
        .iter()
        .try_fold(1usize, |total, count| total.checked_mul(*count))
        .ok_or_else(|| {
            FinitumError::MeshProfileUnsupported("SimplexBox brick count overflows usize".into())
        })?;
    let cell_count = brick_count
        .checked_mul(simplices_per_brick)
        .ok_or_else(|| {
            FinitumError::MeshProfileUnsupported("SimplexBox cell count overflows usize".into())
        })?;
    if node_count > MESH_PROFILE_ITEM_CAP || cell_count > MESH_PROFILE_ITEM_CAP {
        return Err(FinitumError::MeshProfileUnsupported(
            "SimplexBox realization exceeds the one-million-item work cap".into(),
        ));
    }

    let (vertices, cells, grid) = match dim {
        1 => build_segment_grid(extent, subdivisions[0]),
        2 => build_quad_triangle_grid(extent, subdivisions[0], subdivisions[1]),
        3 => build_brick_tet_grid(extent, subdivisions[0], subdivisions[1], subdivisions[2]),
        _ => unreachable!("dimension was checked"),
    };
    let mesh = Mesh::new(dim, vertices, cells)?;
    let facets = FacetTopology::from_mesh(&mesh)?;
    let facet_regions = box_face_tags(dim, subdivisions, &grid, &facets);
    let cell_regions = vec![RegionTagId::new("interior"); mesh.cells().len()];
    let tags = RegionTags {
        cell_regions,
        facet_regions,
    };
    let provenance = MeshProvenance::Profile;
    let digest = tagged_mesh_digest(&mesh, &tags, &provenance);
    Ok(TaggedMesh {
        mesh,
        tags,
        digest,
        provenance,
    })
}

fn coordinate(extent: &[[f64; 2]], axis: usize, index: usize, subdivisions_axis: usize) -> f64 {
    let [min, max] = extent[axis];
    min + (max - min) * (index as f64 / subdivisions_axis as f64)
}

type GridBuild = (Vec<Vec<f64>>, Vec<Cell>, Vec<Vec<usize>>);

fn build_segment_grid(extent: &[[f64; 2]], n: usize) -> GridBuild {
    let mut vertices = Vec::with_capacity(n + 1);
    let mut grid = Vec::with_capacity(n + 1);
    for index in 0..=n {
        vertices.push(vec![coordinate(extent, 0, index, n)]);
        grid.push(vec![index]);
    }
    let mut cells = Vec::with_capacity(n);
    for index in 0..n {
        cells.push(Cell {
            vertices: vec![VertexId(index), VertexId(index + 1)],
        });
    }
    (vertices, cells, grid)
}

fn build_quad_triangle_grid(extent: &[[f64; 2]], nx: usize, ny: usize) -> GridBuild {
    let width = nx + 1;
    let mut vertices = Vec::with_capacity(width * (ny + 1));
    let mut grid = Vec::with_capacity(width * (ny + 1));
    for y in 0..=ny {
        for x in 0..=nx {
            vertices.push(vec![
                coordinate(extent, 0, x, nx),
                coordinate(extent, 1, y, ny),
            ]);
            grid.push(vec![x, y]);
        }
    }
    let mut cells = Vec::with_capacity(nx * ny * 2);
    for y in 0..ny {
        for x in 0..nx {
            let lower_left = y * width + x;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(Cell {
                vertices: vec![
                    VertexId(lower_left),
                    VertexId(lower_right),
                    VertexId(upper_right),
                ],
            });
            cells.push(Cell {
                vertices: vec![
                    VertexId(lower_left),
                    VertexId(upper_right),
                    VertexId(upper_left),
                ],
            });
        }
    }
    (vertices, cells, grid)
}

/// Positively oriented Kuhn six-tet decomposition of one axis-aligned brick. The tets share the
/// main diagonal from the brick's minimum to its maximum corner, matching the decomposition in
/// `tests/sv2_elasticity.rs::brick_tets`; two of its six tets are reordered here (their last two
/// vertices swapped) so every child carries a positive determinant.
fn build_brick_tet_grid(extent: &[[f64; 2]], nx: usize, ny: usize, nz: usize) -> GridBuild {
    let width = nx + 1;
    let depth = ny + 1;
    let mut vertices = Vec::with_capacity(width * depth * (nz + 1));
    let mut grid = Vec::with_capacity(width * depth * (nz + 1));
    for z in 0..=nz {
        for y in 0..=ny {
            for x in 0..=nx {
                vertices.push(vec![
                    coordinate(extent, 0, x, nx),
                    coordinate(extent, 1, y, ny),
                    coordinate(extent, 2, z, nz),
                ]);
                grid.push(vec![x, y, z]);
            }
        }
    }
    let index = |x: usize, y: usize, z: usize| z * depth * width + y * width + x;
    let mut cells = Vec::with_capacity(nx * ny * nz * 6);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let corner =
                    |dx: usize, dy: usize, dz: usize| VertexId(index(x + dx, y + dy, z + dz));
                let a = corner(0, 0, 0);
                let b = corner(1, 0, 0);
                let c = corner(1, 1, 0);
                let d = corner(0, 1, 0);
                let e = corner(0, 0, 1);
                let f = corner(1, 0, 1);
                let g = corner(1, 1, 1);
                let h = corner(0, 1, 1);
                for tet in [
                    [a, b, c, g],
                    [a, b, g, f],
                    [a, e, f, g],
                    [a, e, g, h],
                    [a, d, h, g],
                    [a, d, g, c],
                ] {
                    cells.push(Cell {
                        vertices: tet.to_vec(),
                    });
                }
            }
        }
    }
    (vertices, cells, grid)
}

fn box_face_tags(
    dim: usize,
    subdivisions: &[usize],
    grid: &[Vec<usize>],
    facets: &FacetTopology,
) -> BTreeMap<RegionTagId, Vec<FacetId>> {
    const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];
    let mut result = BTreeMap::<RegionTagId, Vec<FacetId>>::new();
    for facet in facets.exterior() {
        let mut candidates: Option<BTreeSet<RegionTagId>> = None;
        for vertex in &facet.vertices {
            let indices = &grid[vertex.0];
            let mut vertex_candidates = BTreeSet::new();
            for axis in 0..dim {
                if indices[axis] == 0 {
                    vertex_candidates.insert(RegionTagId::new(format!("{}_min", AXIS_NAMES[axis])));
                }
                if indices[axis] == subdivisions[axis] {
                    vertex_candidates.insert(RegionTagId::new(format!("{}_max", AXIS_NAMES[axis])));
                }
            }
            candidates = Some(match candidates {
                None => vertex_candidates,
                Some(existing) => existing.intersection(&vertex_candidates).cloned().collect(),
            });
        }
        if let Some(tag) = candidates.into_iter().flatten().next() {
            result.entry(tag).or_default().push(facet.id);
        }
    }
    for facet_ids in result.values_mut() {
        facet_ids.sort();
        facet_ids.dedup();
    }
    result
}

fn cad_tagged_mesh(geometry: &CadGeometryRealization) -> Result<TaggedMesh, FinitumError> {
    let mesh = geometry.mesh().clone();
    let facets = FacetTopology::from_mesh(&mesh)?;
    let mut facet_lookup = BTreeMap::<Vec<usize>, FacetId>::new();
    for facet in facets.facets() {
        let mut key = facet
            .vertices
            .iter()
            .map(|vertex| vertex.0)
            .collect::<Vec<_>>();
        key.sort_unstable();
        facet_lookup.insert(key, facet.id);
    }
    let mut facet_regions = BTreeMap::<RegionTagId, Vec<FacetId>>::new();
    for boundary in geometry.boundaries() {
        let tag = RegionTagId::new(boundary.entity_id.clone());
        let mut facet_ids = Vec::new();
        for window in boundary.vertices.windows(2) {
            if let Some(id) = facet_lookup.get(&sorted_pair(window[0].0, window[1].0)) {
                facet_ids.push(*id);
            }
        }
        if boundary.vertices.len() > 2 {
            let first = boundary.vertices[0].0;
            let last = boundary.vertices[boundary.vertices.len() - 1].0;
            if let Some(id) = facet_lookup.get(&sorted_pair(last, first)) {
                facet_ids.push(*id);
            }
        }
        facet_ids.sort();
        facet_ids.dedup();
        facet_regions.insert(tag, facet_ids);
    }
    let cell_regions = geometry
        .cells()
        .iter()
        .map(|association| RegionTagId::new(association.region_id.clone()))
        .collect();
    let tags = RegionTags {
        cell_regions,
        facet_regions,
    };
    let provenance = MeshProvenance::Profile;
    let digest = tagged_mesh_digest(&mesh, &tags, &provenance);
    Ok(TaggedMesh {
        mesh,
        tags,
        digest,
        provenance,
    })
}

fn refine_segment(cell: &Cell, edge_midpoint: &BTreeMap<[usize; 2], usize>) -> Vec<Cell> {
    let v0 = cell.vertices[0].0;
    let v1 = cell.vertices[1].0;
    let m = midpoint(edge_midpoint, v0, v1);
    vec![
        Cell {
            vertices: vec![VertexId(v0), m],
        },
        Cell {
            vertices: vec![m, VertexId(v1)],
        },
    ]
}

fn refine_triangle(cell: &Cell, edge_midpoint: &BTreeMap<[usize; 2], usize>) -> Vec<Cell> {
    let v0 = cell.vertices[0].0;
    let v1 = cell.vertices[1].0;
    let v2 = cell.vertices[2].0;
    let m01 = midpoint(edge_midpoint, v0, v1);
    let m12 = midpoint(edge_midpoint, v1, v2);
    let m02 = midpoint(edge_midpoint, v0, v2);
    vec![
        Cell {
            vertices: vec![VertexId(v0), m01, m02],
        },
        Cell {
            vertices: vec![m01, VertexId(v1), m12],
        },
        Cell {
            vertices: vec![m02, m12, VertexId(v2)],
        },
        Cell {
            vertices: vec![m01, m12, m02],
        },
    ]
}

/// Standard red refinement of a tetrahedron into eight children: four corner tets peeling each
/// original vertex, plus the medial octahedron split into four tets along one fixed diagonal
/// (`m02`-`m13`). Vertex order was chosen so every child keeps a positive determinant given a
/// positively oriented parent (verified against a reference unit tet).
fn refine_tet(cell: &Cell, edge_midpoint: &BTreeMap<[usize; 2], usize>) -> Vec<Cell> {
    let p0 = cell.vertices[0].0;
    let p1 = cell.vertices[1].0;
    let p2 = cell.vertices[2].0;
    let p3 = cell.vertices[3].0;
    let m01 = midpoint(edge_midpoint, p0, p1);
    let m02 = midpoint(edge_midpoint, p0, p2);
    let m03 = midpoint(edge_midpoint, p0, p3);
    let m12 = midpoint(edge_midpoint, p1, p2);
    let m13 = midpoint(edge_midpoint, p1, p3);
    let m23 = midpoint(edge_midpoint, p2, p3);
    vec![
        Cell {
            vertices: vec![VertexId(p0), m01, m02, m03],
        },
        Cell {
            vertices: vec![m01, VertexId(p1), m12, m13],
        },
        Cell {
            vertices: vec![m02, m12, VertexId(p2), m23],
        },
        Cell {
            vertices: vec![m03, m13, m23, VertexId(p3)],
        },
        Cell {
            vertices: vec![m01, m02, m03, m13],
        },
        Cell {
            vertices: vec![m01, m02, m13, m12],
        },
        Cell {
            vertices: vec![m02, m03, m13, m23],
        },
        Cell {
            vertices: vec![m02, m12, m23, m13],
        },
    ]
}

fn midpoint(edge_midpoint: &BTreeMap<[usize; 2], usize>, a: usize, b: usize) -> VertexId {
    VertexId(edge_midpoint[&sorted_edge(a, b)])
}

fn sorted_edge(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

fn sorted_pair(a: usize, b: usize) -> Vec<usize> {
    if a < b { vec![a, b] } else { vec![b, a] }
}

#[derive(Serialize)]
struct TaggedMeshDigestPayload<'a> {
    schema: &'static str,
    mesh: &'a Mesh,
    tags: &'a RegionTags,
    provenance: &'a MeshProvenance,
}

fn tagged_mesh_digest(mesh: &Mesh, tags: &RegionTags, provenance: &MeshProvenance) -> Digest {
    let payload = TaggedMeshDigestPayload {
        schema: "finitum.tagged-mesh/v1",
        mesh,
        tags,
        provenance,
    };
    Digest::blake3(&serde_json::to_vec(&payload).expect("tagged mesh payload is serializable"))
}
