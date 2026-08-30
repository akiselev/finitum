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

/// One non-basis essential value source: a constant, a caller-projected nodal field, or a
/// coordinate sampler. This is a deliberately small slice of the eventual `FieldSource` (C7):
/// `Table`/`Kernel` variants arrive once Scientia's `PropertyKernel` contract lands.
type SampledFieldFn = dyn Fn(&[f64]) -> Vec<f64> + Send + Sync;

#[derive(Clone)]
pub enum FieldSource {
    /// One value per component, applied uniformly to every tagged vertex.
    Constant(Vec<f64>),
    /// A caller-projected vertex-major field, `components` entries per vertex.
    Nodal(Vec<f64>),
    /// A coordinate-driven evaluator returning `components` values per call.
    Sampled(Arc<SampledFieldFn>),
}

impl std::fmt::Debug for FieldSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldSource::Constant(values) => {
                formatter.debug_tuple("Constant").field(values).finish()
            }
            FieldSource::Nodal(values) => formatter.debug_tuple("Nodal").field(values).finish(),
            FieldSource::Sampled(_) => formatter.debug_tuple("Sampled").field(&"<fn>").finish(),
        }
    }
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
}

/// Derives essential (Dirichlet) constraints from region tags rather than positional CAD ids or
/// index arithmetic. `dof_map` must be a vertex-major scalar or vector map (see
/// [`crate::vector_nodal_dof_map`]); `values` supplies one [`FieldSource`] per requirement.
pub fn essential_constraints_from(
    mesh: &TaggedMesh,
    dof_map: &DofMap,
    requirements: &[scientia::EssentialConstraintRequirement],
    region_map: &RegionMap,
    values: &[FieldSource],
) -> Result<ConstraintSet, FinitumError> {
    if requirements.len() != values.len() {
        return Err(FinitumError::InvalidRealization(format!(
            "essential constraint requirements has {} entries but {} field sources were supplied",
            requirements.len(),
            values.len()
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
    for (requirement, source) in requirements.iter().zip(values) {
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
                FieldSource::Sampled(sampler) => sampler(&mesh.mesh.vertices()[vertex.0]),
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
