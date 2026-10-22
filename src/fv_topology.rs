//! SV2-C2a finite-volume topology (`finitum-fv-topology/1`, W12 lane L4).
//!
//! [`FvTopology`] is the cell-centred view of a mesh: cell volumes and centroids, and face
//! blocks in structure-of-arrays form (owner/neighbour, area, unit normal per axis, centroid,
//! centroid-to-centroid vector) so a face kernel binds every column at a trivial stride. It is
//! built from a simplex [`TaggedMesh`] (through [`FacetTopology`] and its exterior region tags)
//! or from a [`TensorBox`] profile (decision D2: 1-D/2-D axis-aligned cells, uniform or graded
//! spacing, optional periodic axes).
//!
//! Orientation: an interior face's normal points from `minus` to `plus`; a boundary face's
//! normal is outward from its only cell (`minus`). A periodic face joins the last cell of a
//! periodic axis (`minus`) to the first (`plus`) across the wrap and records the translation
//! that carries the `plus` cell next to the face.
//!
//! SV2-C2c (W13 lane `finitum-data`): a [`TensorBox`] may be partitioned by a
//! [`TensorRegionPartition`] into axis-aligned sub-boxes of cells, each with a region tag, and
//! every face between two regions carries an interface tag. The topology then records one
//! region per cell ([`FvTopology::cell_regions`]) and the interface faces as indices into the
//! interior and periodic blocks ([`FvTopology::interfaces`]); the block layout, the scatter
//! order and the digest of an unpartitioned box are unchanged.

use std::collections::{BTreeMap, BTreeSet};

use scientia::Digest;
use serde::{Deserialize, Serialize};

use crate::{FacetId, FacetTopology, FinitumError, MeshProfile, RegionTagId, TaggedMesh};

pub const FV_TOPOLOGY_SCHEMA: &str = "finitum-fv-topology/1";
/// Schema of a [`TensorRegionPartition`] and of the source digest of a partitioned box.
pub const FV_REGION_PARTITION_SCHEMA: &str = "finitum-fv-region-partition/1";

const FV_TOPOLOGY_CELL_CAP: usize = 1_000_000;
const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];

fn topology_error(code: &str, message: impl std::fmt::Display) -> FinitumError {
    FinitumError::InvalidRealization(format!("{code}: {message}"))
}

/// Node placement along one tensor axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AxisSpacing {
    /// `cells` equal cells on `[min, max]`; the last node is `max` exactly.
    Uniform { min: f64, max: f64, cells: usize },
    /// Explicit strictly increasing node coordinates (graded spacing); at least two.
    Nodes { nodes: Vec<f64> },
}

/// One axis of a [`TensorBox`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorAxis {
    pub spacing: AxisSpacing,
    /// A periodic axis has no boundary faces; its end faces are paired across the wrap.
    pub periodic: bool,
}

impl TensorAxis {
    pub fn uniform(min: f64, max: f64, cells: usize) -> Self {
        Self {
            spacing: AxisSpacing::Uniform { min, max, cells },
            periodic: false,
        }
    }

    pub fn graded(nodes: Vec<f64>) -> Self {
        Self {
            spacing: AxisSpacing::Nodes { nodes },
            periodic: false,
        }
    }

    pub fn periodic(mut self) -> Self {
        self.periodic = true;
        self
    }

    fn nodes(&self, axis: usize) -> Result<Vec<f64>, FinitumError> {
        let nodes = match &self.spacing {
            AxisSpacing::Uniform { min, max, cells } => {
                if *cells == 0 || !min.is_finite() || !max.is_finite() || max <= min {
                    return Err(FinitumError::MeshProfileUnsupported(format!(
                        "TensorBox axis {axis} needs finite min < max and a positive cell count"
                    )));
                }
                if *cells > FV_TOPOLOGY_CELL_CAP {
                    return Err(FinitumError::MeshProfileUnsupported(
                        "TensorBox realization exceeds the one-million-item work cap".into(),
                    ));
                }
                let mut nodes = (0..=*cells)
                    .map(|index| min + (max - min) * (index as f64 / *cells as f64))
                    .collect::<Vec<_>>();
                nodes[*cells] = *max;
                nodes
            }
            AxisSpacing::Nodes { nodes } => nodes.clone(),
        };
        if nodes.len() < 2
            || nodes.iter().any(|node| !node.is_finite())
            || nodes.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(FinitumError::MeshProfileUnsupported(format!(
                "TensorBox axis {axis} nodes must be finite, strictly increasing and at least two"
            )));
        }
        Ok(nodes)
    }
}

/// Axis-aligned tensor-product cells in 1-D (segments) or 2-D (rectangles), decision D2.
/// Cells are numbered lexicographically with axis 0 fastest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorBox {
    pub axes: Vec<TensorAxis>,
}

impl TensorBox {
    /// Cell counts per axis (after the node checks of [`FvTopology::from_tensor_box`]).
    pub fn cell_counts(&self) -> Result<Vec<usize>, FinitumError> {
        self.axes
            .iter()
            .enumerate()
            .map(|(axis, item)| item.nodes(axis).map(|nodes| nodes.len() - 1))
            .collect()
    }

    /// The index of the node of `axis` at `coordinate`, so a region boundary given as a
    /// coordinate can be turned into an exact cell-index range. The coordinate must coincide
    /// with a node to `4ε` of the axis extent; anything else is refused
    /// (`FV_REGION_BOUNDARY_OFF_NODE`) rather than snapped, because a region boundary inside a
    /// cell is not representable.
    pub fn node_index_at(&self, axis: usize, coordinate: f64) -> Result<usize, FinitumError> {
        let item = self.axes.get(axis).ok_or_else(|| {
            topology_error(
                "FV_REGION_RANGE_INVALID",
                format!("axis {axis} does not exist in a {}-D box", self.axes.len()),
            )
        })?;
        let nodes = item.nodes(axis)?;
        let extent = nodes[nodes.len() - 1] - nodes[0];
        let tolerance = 4.0 * f64::EPSILON * extent;
        nodes
            .iter()
            .position(|node| (node - coordinate).abs() <= tolerance)
            .ok_or_else(|| {
                topology_error(
                    "FV_REGION_BOUNDARY_OFF_NODE",
                    format!(
                        "coordinate {coordinate} on axis {axis} is not a node (to {tolerance:e})"
                    ),
                )
            })
    }
}

/// One region of a [`TensorRegionPartition`]: the cells whose per-axis index lies in the
/// half-open range `cells[axis] = [start, end)` (one range per axis of the box).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRegion {
    pub tag: RegionTagId,
    pub cells: Vec<[usize; 2]>,
}

/// The faces between two regions of a [`TensorRegionPartition`], under one tag.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInterface {
    pub tag: RegionTagId,
    /// The two (distinct) region tags the interface separates, in either order.
    pub between: [RegionTagId; 2],
}

/// A partition of a [`TensorBox`] into tagged axis-aligned sub-boxes of cells
/// (`finitum-fv-region-partition/1`). Regions are given by cell-index ranges, never by
/// coordinates, so membership is exact and every region boundary lies on a face;
/// [`TensorBox::node_index_at`] converts a coordinate that is a node. Invariants checked by
/// [`FvTopology::from_partitioned_tensor_box`]: every cell lies in exactly one region
/// (`FV_REGION_OVERLAP`, `FV_REGION_UNCOVERED`), ranges fit the box
/// (`FV_REGION_RANGE_INVALID`), tags are unique and disjoint from the box's boundary tags
/// (`FV_REGION_TAG_DUPLICATE`, `FV_REGION_TAG_OVERLAP`), every pair of regions that shares a
/// face is named by exactly one interface and every interface has faces
/// (`FV_REGION_INTERFACE_UNTAGGED`, `_DUPLICATE`, `_EMPTY`, `_UNKNOWN`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRegionPartition {
    pub regions: Vec<TensorRegion>,
    #[serde(default)]
    pub interfaces: Vec<TensorInterface>,
}

/// The faces of one interface tag, as indices into the interior and periodic blocks of the
/// topology. A face's two cells lie in the two regions of `between` (either side).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FvInterface {
    pub between: [RegionTagId; 2],
    pub interior: Vec<usize>,
    #[serde(default)]
    pub periodic: Vec<usize>,
}

impl FvInterface {
    pub fn len(&self) -> usize {
        self.interior.len() + self.periodic.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The cell family a topology was built from; it fixes the face count of every cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FvCellShape {
    Simplex,
    Tensor,
}

/// One block of faces in structure-of-arrays form. `plus` is empty on boundary blocks;
/// `translation` is set only on the periodic block.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvFaceBlock {
    pub minus: Vec<usize>,
    pub plus: Vec<usize>,
    pub area: Vec<f64>,
    /// Unit normal components, one column per axis (`normal[axis][face]`).
    pub normal: Vec<Vec<f64>>,
    /// Face centroids, `dimension` values per face.
    pub centroid: Vec<f64>,
    /// `plus` centroid (translated across a periodic wrap) minus the `minus` centroid, or the
    /// face centroid minus the `minus` centroid at a boundary; `dimension` values per face.
    pub delta: Vec<f64>,
    /// Translation carrying the `plus` cell next to a periodic face; `dimension` per face.
    pub translation: Vec<f64>,
}

impl FvFaceBlock {
    fn new(dimension: usize) -> Self {
        Self {
            minus: Vec::new(),
            plus: Vec::new(),
            area: Vec::new(),
            normal: vec![Vec::new(); dimension],
            centroid: Vec::new(),
            delta: Vec::new(),
            translation: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.minus.len()
    }

    pub fn is_empty(&self) -> bool {
        self.minus.is_empty()
    }

    fn push(&mut self, face: Face<'_>) {
        self.minus.push(face.minus);
        if let Some(plus) = face.plus {
            self.plus.push(plus);
        }
        self.area.push(face.area);
        for (column, value) in self.normal.iter_mut().zip(face.normal) {
            column.push(*value);
        }
        self.centroid.extend_from_slice(face.centroid);
        self.delta.extend_from_slice(face.delta);
        if let Some(translation) = face.translation {
            self.translation.extend_from_slice(translation);
        }
    }
}

struct Face<'a> {
    minus: usize,
    plus: Option<usize>,
    area: f64,
    normal: &'a [f64],
    centroid: &'a [f64],
    delta: &'a [f64],
    translation: Option<&'a [f64]>,
}

/// Cells and faces of a finite-volume mesh with a content identity.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FvTopology {
    schema: String,
    dimension: usize,
    shape: FvCellShape,
    volume: Vec<f64>,
    centroid: Vec<f64>,
    interior: FvFaceBlock,
    periodic: FvFaceBlock,
    boundary: BTreeMap<RegionTagId, FvFaceBlock>,
    /// One region tag per cell when the topology was built from a partitioned box; empty
    /// (and absent from the digest) otherwise.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cell_regions: Vec<RegionTagId>,
    /// Interface faces by tag (empty and absent from the digest without a partition).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    interfaces: BTreeMap<RegionTagId, FvInterface>,
    /// The tagged-mesh digest or the digest of the tensor profile it was built from.
    source: Digest,
    #[serde(skip)]
    digest: Digest,
}

/// The checked invariants of an [`FvTopology`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FvTopologyInvariants {
    pub cells: usize,
    pub interior_faces: usize,
    pub periodic_faces: usize,
    pub boundary_faces: usize,
    /// Faces of every cell (`2·dimension` tensor, `dimension + 1` simplex); each cell was
    /// found on exactly this many face sides, so every face appears once per incident side.
    pub faces_per_cell: usize,
    /// `max_i ‖Σ_f s_if |f| n_f‖_∞ / Σ_f |f|` over cells, `s = +1` on the minus side.
    pub max_closure_defect: f64,
    pub closure_bound: f64,
    /// `max_f |‖n_f‖ − 1|`.
    pub max_normal_defect: f64,
    /// Distinct cell regions (`0` without a partition).
    #[serde(default)]
    pub regions: usize,
    /// Faces tagged as interfaces; equals the number of faces joining two different regions.
    #[serde(default)]
    pub interface_faces: usize,
}

impl FvTopology {
    /// The topology of any [`MeshProfile`]: a [`TensorBox`] directly, every other profile
    /// through [`crate::realize`] and [`Self::from_tagged_mesh`].
    pub fn from_profile(profile: &MeshProfile) -> Result<Self, FinitumError> {
        match profile {
            MeshProfile::TensorBox(spec) => Self::from_tensor_box(spec),
            other => Self::from_tagged_mesh(&crate::realize(other)?),
        }
    }

    /// The cell-centred topology of a simplex mesh. Every exterior facet must carry exactly one
    /// boundary tag; interior blocks follow [`FacetTopology`] order and boundary blocks keep
    /// facet order within each tag.
    pub fn from_tagged_mesh(mesh: &TaggedMesh) -> Result<Self, FinitumError> {
        let geometry = &mesh.mesh;
        let dimension = geometry.dimension();
        if geometry
            .cells()
            .iter()
            .any(|cell| cell.vertices.len() != dimension + 1)
        {
            return Err(topology_error(
                "FV_MESH_UNSUPPORTED",
                "simplex FV topology needs simplex cells",
            ));
        }
        let point = |vertex: crate::VertexId| geometry.vertices()[vertex.0].as_slice();
        let mut volume = Vec::with_capacity(geometry.cells().len());
        let mut centroid = Vec::with_capacity(geometry.cells().len() * dimension);
        for (index, cell) in geometry.cells().iter().enumerate() {
            let points = cell.vertices.iter().map(|v| point(*v)).collect::<Vec<_>>();
            let measure = simplex_volume(&points);
            if !measure.is_finite() || measure <= 0.0 {
                return Err(topology_error(
                    "FV_MESH_UNSUPPORTED",
                    format!("cell {index} is degenerate"),
                ));
            }
            volume.push(measure);
            centroid.extend(mean(&points));
        }

        let facets = FacetTopology::from_mesh(geometry)?;
        let mut tag_of = BTreeMap::<FacetId, &RegionTagId>::new();
        for (tag, ids) in &mesh.tags.facet_regions {
            for id in ids {
                let Some(facet) = facets.facets().get(id.0) else {
                    return Err(topology_error(
                        "FV_BOUNDARY_TAG_INVALID",
                        format!("tag `{tag}` names missing facet {}", id.0),
                    ));
                };
                if !facet.is_exterior() {
                    return Err(topology_error(
                        "FV_BOUNDARY_TAG_INVALID",
                        format!("tag `{tag}` names interior facet {}", id.0),
                    ));
                }
                if tag_of.insert(*id, tag).is_some() {
                    return Err(topology_error(
                        "FV_BOUNDARY_TAG_OVERLAP",
                        format!("facet {} carries more than one boundary tag", id.0),
                    ));
                }
            }
        }

        let mut interior = FvFaceBlock::new(dimension);
        let mut boundary = BTreeMap::<RegionTagId, FvFaceBlock>::new();
        for facet in facets.facets() {
            let points = facet.vertices.iter().map(|v| point(*v)).collect::<Vec<_>>();
            let face_centroid = mean(&points);
            let minus = facet.minus().cell.0;
            let minus_centroid = &centroid[minus * dimension..(minus + 1) * dimension];
            let (area, mut normal) = simplex_facet_normal(&points);
            let outward = difference(&face_centroid, minus_centroid);
            if dot(&normal, &outward) < 0.0 {
                normal.iter_mut().for_each(|value| *value = -*value);
            }
            if let Some(plus) = facet.plus() {
                let plus = plus.cell.0;
                let delta = difference(
                    &centroid[plus * dimension..(plus + 1) * dimension],
                    minus_centroid,
                );
                interior.push(Face {
                    minus,
                    plus: Some(plus),
                    area,
                    normal: &normal,
                    centroid: &face_centroid,
                    delta: &delta,
                    translation: None,
                });
            } else {
                let tag = tag_of.get(&facet.id).ok_or_else(|| {
                    topology_error(
                        "FV_BOUNDARY_UNTAGGED",
                        format!("exterior facet {} carries no boundary tag", facet.id.0),
                    )
                })?;
                boundary
                    .entry((*tag).clone())
                    .or_insert_with(|| FvFaceBlock::new(dimension))
                    .push(Face {
                        minus,
                        plus: None,
                        area,
                        normal: &normal,
                        centroid: &face_centroid,
                        delta: &outward,
                        translation: None,
                    });
            }
        }
        Ok(Self::finish(
            dimension,
            FvCellShape::Simplex,
            volume,
            centroid,
            interior,
            FvFaceBlock::new(dimension),
            boundary,
            Regions::default(),
            mesh.digest.clone(),
        ))
    }

    /// The topology of a [`TensorBox`]. Faces are ordered by axis, then by the cell row across
    /// the axis, then along it; boundary tags are `x_min`, `x_max`, `y_min`, `y_max` on the
    /// non-periodic axes (the [`crate::MeshProfile::SimplexBox`] names).
    pub fn from_tensor_box(spec: &TensorBox) -> Result<Self, FinitumError> {
        Self::build_tensor_box(spec, None)
    }

    /// The topology of a [`TensorBox`] partitioned into regions: as [`Self::from_tensor_box`]
    /// (same cells, faces, blocks and order), plus one region tag per cell and the faces
    /// between two regions under their interface tag. See [`TensorRegionPartition`] for the
    /// checked invariants and refusal codes.
    pub fn from_partitioned_tensor_box(
        spec: &TensorBox,
        partition: &TensorRegionPartition,
    ) -> Result<Self, FinitumError> {
        Self::build_tensor_box(spec, Some(partition))
    }

    fn build_tensor_box(
        spec: &TensorBox,
        partition: Option<&TensorRegionPartition>,
    ) -> Result<Self, FinitumError> {
        let dimension = spec.axes.len();
        if !(1..=2).contains(&dimension) {
            return Err(FinitumError::MeshProfileUnsupported(format!(
                "TensorBox dimension must be 1 or 2, got {dimension}"
            )));
        }
        let nodes = spec
            .axes
            .iter()
            .enumerate()
            .map(|(axis, item)| item.nodes(axis))
            .collect::<Result<Vec<_>, _>>()?;
        let counts = nodes.iter().map(|axis| axis.len() - 1).collect::<Vec<_>>();
        let cells = counts
            .iter()
            .try_fold(1usize, |total, count| total.checked_mul(*count))
            .filter(|cells| *cells <= FV_TOPOLOGY_CELL_CAP)
            .ok_or_else(|| {
                FinitumError::MeshProfileUnsupported(
                    "TensorBox realization exceeds the one-million-item work cap".into(),
                )
            })?;
        let width = |axis: usize, index: usize| nodes[axis][index + 1] - nodes[axis][index];
        let middle =
            |axis: usize, index: usize| (nodes[axis][index] + nodes[axis][index + 1]) * 0.5;
        // Cell index of per-axis indices (axis 0 fastest).
        let cell_of = |index: &[usize]| index[0] + index.get(1).map_or(0, |j| j * counts[0]);

        let mut volume = Vec::with_capacity(cells);
        let mut centroid = Vec::with_capacity(cells * dimension);
        for cell in 0..cells {
            let index = [cell % counts[0], cell / counts[0]];
            let mut measure = 1.0;
            for (axis, at) in index.iter().copied().enumerate().take(dimension) {
                measure *= width(axis, at);
                centroid.push(middle(axis, at));
            }
            volume.push(measure);
        }

        let mut interior = FvFaceBlock::new(dimension);
        let mut periodic = FvFaceBlock::new(dimension);
        let mut boundary = BTreeMap::<RegionTagId, FvFaceBlock>::new();
        for axis in 0..dimension {
            let other = 1 - axis;
            let rows = if dimension == 2 { counts[other] } else { 1 };
            let count = counts[axis];
            for row in 0..rows {
                let cell_at = |k: usize| {
                    let mut index = [0usize; 2];
                    index[axis] = k;
                    if dimension == 2 {
                        index[other] = row;
                    }
                    cell_of(&index[..dimension])
                };
                let area = if dimension == 2 {
                    width(other, row)
                } else {
                    1.0
                };
                for k in 0..=count {
                    let mut face_centroid = vec![0.0; dimension];
                    face_centroid[axis] = nodes[axis][k];
                    if dimension == 2 {
                        face_centroid[other] = middle(other, row);
                    }
                    let mut normal = vec![0.0; dimension];
                    normal[axis] = 1.0;
                    let at = |cell: usize| &centroid[cell * dimension..(cell + 1) * dimension];
                    if 0 < k && k < count {
                        let (minus, plus) = (cell_at(k - 1), cell_at(k));
                        interior.push(Face {
                            minus,
                            plus: Some(plus),
                            area,
                            normal: &normal,
                            centroid: &face_centroid,
                            delta: &difference(at(plus), at(minus)),
                            translation: None,
                        });
                    } else if spec.axes[axis].periodic {
                        if k == 0 {
                            continue;
                        }
                        let (minus, plus) = (cell_at(count - 1), cell_at(0));
                        let mut translation = vec![0.0; dimension];
                        translation[axis] = nodes[axis][count] - nodes[axis][0];
                        let shifted = at(plus)
                            .iter()
                            .zip(&translation)
                            .map(|(c, t)| c + t)
                            .collect::<Vec<_>>();
                        periodic.push(Face {
                            minus,
                            plus: Some(plus),
                            area,
                            normal: &normal,
                            centroid: &face_centroid,
                            delta: &difference(&shifted, at(minus)),
                            translation: Some(&translation),
                        });
                    } else {
                        let (minus, side) = if k == 0 {
                            normal[axis] = -1.0;
                            (cell_at(0), "min")
                        } else {
                            (cell_at(count - 1), "max")
                        };
                        let tag = RegionTagId::new(format!("{}_{side}", AXIS_NAMES[axis]));
                        boundary
                            .entry(tag)
                            .or_insert_with(|| FvFaceBlock::new(dimension))
                            .push(Face {
                                minus,
                                plus: None,
                                area,
                                normal: &normal,
                                centroid: &face_centroid,
                                delta: &difference(&face_centroid, at(minus)),
                                translation: None,
                            });
                    }
                }
            }
        }
        let (regions, source) = match partition {
            None => (
                Regions::default(),
                Digest::blake3(
                    &serde_json::to_vec(&("finitum-tensor-box/1", spec))
                        .expect("tensor box is serializable"),
                ),
            ),
            Some(partition) => (
                Regions::of_tensor_box(
                    partition,
                    &counts,
                    &interior,
                    &periodic,
                    boundary.keys(),
                    cells,
                )?,
                Digest::blake3(
                    &serde_json::to_vec(&(FV_REGION_PARTITION_SCHEMA, spec, partition))
                        .expect("tensor box partition is serializable"),
                ),
            ),
        };
        Ok(Self::finish(
            dimension,
            FvCellShape::Tensor,
            volume,
            centroid,
            interior,
            periodic,
            boundary,
            regions,
            source,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        dimension: usize,
        shape: FvCellShape,
        volume: Vec<f64>,
        centroid: Vec<f64>,
        interior: FvFaceBlock,
        periodic: FvFaceBlock,
        boundary: BTreeMap<RegionTagId, FvFaceBlock>,
        regions: Regions,
        source: Digest,
    ) -> Self {
        let mut topology = Self {
            schema: FV_TOPOLOGY_SCHEMA.into(),
            dimension,
            shape,
            volume,
            centroid,
            interior,
            periodic,
            boundary,
            cell_regions: regions.cell_regions,
            interfaces: regions.interfaces,
            source,
            digest: Digest {
                algorithm: "blake3".into(),
                hex: String::new(),
            },
        };
        topology.digest =
            Digest::blake3(&serde_json::to_vec(&topology).expect("FV topology is serializable"));
        topology
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn shape(&self) -> FvCellShape {
        self.shape
    }

    pub fn cell_count(&self) -> usize {
        self.volume.len()
    }

    pub fn cell_volumes(&self) -> &[f64] {
        &self.volume
    }

    /// Cell centroids, `dimension` values per cell.
    pub fn cell_centroids(&self) -> &[f64] {
        &self.centroid
    }

    pub fn interior(&self) -> &FvFaceBlock {
        &self.interior
    }

    pub fn periodic(&self) -> &FvFaceBlock {
        &self.periodic
    }

    pub fn boundary(&self) -> &BTreeMap<RegionTagId, FvFaceBlock> {
        &self.boundary
    }

    /// One region tag per cell, or empty when the topology carries no region partition.
    pub fn cell_regions(&self) -> &[RegionTagId] {
        &self.cell_regions
    }

    /// Whether every cell carries a region tag (a partitioned tensor box).
    pub fn has_regions(&self) -> bool {
        !self.cell_regions.is_empty()
    }

    /// The distinct region tags, in tag order (empty without a partition).
    pub fn region_tags(&self) -> BTreeSet<&RegionTagId> {
        self.cell_regions.iter().collect()
    }

    /// Interface faces by tag (empty without a partition).
    pub fn interfaces(&self) -> &BTreeMap<RegionTagId, FvInterface> {
        &self.interfaces
    }

    /// The digest of the mesh or tensor profile this topology was built from.
    pub fn source_digest(&self) -> &Digest {
        &self.source
    }

    /// Content identity over every cell and face column (`finitum-fv-topology/1`).
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// Faces of every cell for this cell family.
    pub fn faces_per_cell(&self) -> usize {
        match self.shape {
            FvCellShape::Simplex => self.dimension + 1,
            FvCellShape::Tensor => 2 * self.dimension,
        }
    }

    /// Interior, periodic, then boundary blocks in tag order: the fixed scatter order.
    pub(crate) fn blocks(&self) -> impl Iterator<Item = &FvFaceBlock> {
        [&self.interior, &self.periodic]
            .into_iter()
            .chain(self.boundary.values())
    }

    /// The largest angle (radians) between a face's centroid offset `delta` and its unit
    /// normal over every face; `0` on tensor meshes. Two-point diffusion needs it near zero.
    pub fn max_nonorthogonality(&self) -> f64 {
        let dimension = self.dimension;
        let mut worst = 0.0_f64;
        for block in self.blocks() {
            for face in 0..block.len() {
                let delta = &block.delta[face * dimension..(face + 1) * dimension];
                let length = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
                let projected = (0..dimension)
                    .map(|axis| delta[axis] * block.normal[axis][face])
                    .sum::<f64>();
                worst = worst.max((projected / length).clamp(-1.0, 1.0).acos());
            }
        }
        worst
    }

    /// Check positivity, unit normals, face-count and closure (`Σ_f s_if |f| n_f = 0` for every
    /// closed cell, to `8·faces·ε` relative to `Σ_f |f|`) and interior pairing.
    pub fn check_invariants(&self) -> Result<FvTopologyInvariants, FinitumError> {
        let dimension = self.dimension;
        let cells = self.cell_count();
        if let Some(cell) = self
            .volume
            .iter()
            .position(|volume| !(volume.is_finite() && *volume > 0.0))
        {
            return Err(topology_error(
                "FV_TOPOLOGY_INVARIANT",
                format!("cell {cell} has a non-positive volume"),
            ));
        }
        let mut closure = vec![0.0; cells * dimension];
        let mut area_sum = vec![0.0; cells];
        let mut sides = vec![0usize; cells];
        let mut max_normal_defect = 0.0_f64;
        for block in self.blocks() {
            for face in 0..block.len() {
                let area = block.area[face];
                if !(area.is_finite() && area > 0.0) {
                    return Err(topology_error(
                        "FV_TOPOLOGY_INVARIANT",
                        format!("face {face} has a non-positive area"),
                    ));
                }
                let norm = block
                    .normal
                    .iter()
                    .map(|column| column[face] * column[face])
                    .sum::<f64>()
                    .sqrt();
                max_normal_defect = max_normal_defect.max((norm - 1.0).abs());
                let minus = block.minus[face];
                let mut incident = vec![(minus, 1.0)];
                if let Some(plus) = block.plus.get(face).copied() {
                    if plus == minus && block.translation.is_empty() {
                        return Err(topology_error(
                            "FV_TOPOLOGY_INVARIANT",
                            format!("interior face {face} joins cell {minus} to itself"),
                        ));
                    }
                    incident.push((plus, -1.0));
                }
                for (cell, sign) in incident {
                    sides[cell] += 1;
                    area_sum[cell] += area;
                    for axis in 0..dimension {
                        closure[cell * dimension + axis] += sign * area * block.normal[axis][face];
                    }
                }
            }
        }
        let faces_per_cell = self.faces_per_cell();
        if let Some(cell) = sides.iter().position(|count| *count != faces_per_cell) {
            return Err(topology_error(
                "FV_TOPOLOGY_INVARIANT",
                format!(
                    "cell {cell} lies on {} face sides, expected {faces_per_cell}",
                    sides[cell]
                ),
            ));
        }
        let (regions, interface_faces) = self.check_region_invariants()?;
        let closure_bound = 8.0 * faces_per_cell as f64 * f64::EPSILON;
        let mut max_closure_defect = 0.0_f64;
        for cell in 0..cells {
            let defect = closure[cell * dimension..(cell + 1) * dimension]
                .iter()
                .fold(0.0_f64, |worst, value| worst.max(value.abs()))
                / area_sum[cell];
            max_closure_defect = max_closure_defect.max(defect);
        }
        if max_closure_defect > closure_bound || max_normal_defect > 4.0 * f64::EPSILON {
            return Err(topology_error(
                "FV_TOPOLOGY_INVARIANT",
                format!(
                    "closure defect {max_closure_defect:e} (bound {closure_bound:e}) or normal \
                     defect {max_normal_defect:e}"
                ),
            ));
        }
        Ok(FvTopologyInvariants {
            cells,
            interior_faces: self.interior.len(),
            periodic_faces: self.periodic.len(),
            boundary_faces: self.boundary.values().map(FvFaceBlock::len).sum(),
            faces_per_cell,
            max_closure_defect,
            closure_bound,
            max_normal_defect,
            regions,
            interface_faces,
        })
    }

    /// Every cell in exactly one region, every interface face joining two different regions
    /// that match its `between`, every region-crossing face in exactly one interface, and no
    /// interface face inside one region.
    fn check_region_invariants(&self) -> Result<(usize, usize), FinitumError> {
        if self.cell_regions.is_empty() {
            if !self.interfaces.is_empty() {
                return Err(topology_error(
                    "FV_TOPOLOGY_INVARIANT",
                    "interfaces without cell regions",
                ));
            }
            return Ok((0, 0));
        }
        if self.cell_regions.len() != self.cell_count() {
            return Err(topology_error(
                "FV_TOPOLOGY_INVARIANT",
                format!(
                    "{} cell regions for {} cells",
                    self.cell_regions.len(),
                    self.cell_count()
                ),
            ));
        }
        let mut tagged = [
            vec![false; self.interior.len()],
            vec![false; self.periodic.len()],
        ];
        let mut interface_faces = 0;
        for (tag, interface) in &self.interfaces {
            for (which, (block, faces)) in [
                (&self.interior, &interface.interior),
                (&self.periodic, &interface.periodic),
            ]
            .into_iter()
            .enumerate()
            {
                for face in faces {
                    let (Some(minus), Some(plus)) =
                        (block.minus.get(*face), block.plus.get(*face))
                    else {
                        return Err(topology_error(
                            "FV_TOPOLOGY_INVARIANT",
                            format!("interface `{tag}` names missing face {face}"),
                        ));
                    };
                    let pair = [&self.cell_regions[*minus], &self.cell_regions[*plus]];
                    let declared = [&interface.between[0], &interface.between[1]];
                    if pair[0] == pair[1]
                        || (pair != declared && pair != [declared[1], declared[0]])
                    {
                        return Err(topology_error(
                            "FV_TOPOLOGY_INVARIANT",
                            format!(
                                "interface `{tag}` face {face} joins regions `{}` and `{}`",
                                pair[0], pair[1]
                            ),
                        ));
                    }
                    if std::mem::replace(&mut tagged[which][*face], true) {
                        return Err(topology_error(
                            "FV_TOPOLOGY_INVARIANT",
                            format!("face {face} lies on two interfaces"),
                        ));
                    }
                    interface_faces += 1;
                }
            }
        }
        for (block, tagged) in [(&self.interior, &tagged[0]), (&self.periodic, &tagged[1])] {
            for face in 0..block.len() {
                let crossing =
                    self.cell_regions[block.minus[face]] != self.cell_regions[block.plus[face]];
                if crossing && !tagged[face] {
                    return Err(topology_error(
                        "FV_TOPOLOGY_INVARIANT",
                        format!("face {face} joins two regions but no interface tags it"),
                    ));
                }
            }
        }
        Ok((self.region_tags().len(), interface_faces))
    }
}

/// The region columns of a topology under construction.
#[derive(Default)]
struct Regions {
    cell_regions: Vec<RegionTagId>,
    interfaces: BTreeMap<RegionTagId, FvInterface>,
}

impl Regions {
    /// Validate `partition` against a tensor box with `counts` cells per axis and tag its
    /// region-crossing faces; see [`TensorRegionPartition`] for the refusal codes.
    fn of_tensor_box<'a>(
        partition: &TensorRegionPartition,
        counts: &[usize],
        interior: &FvFaceBlock,
        periodic: &FvFaceBlock,
        boundary_tags: impl Iterator<Item = &'a RegionTagId>,
        cells: usize,
    ) -> Result<Self, FinitumError> {
        let dimension = counts.len();
        let mut tags = boundary_tags.cloned().collect::<BTreeSet<_>>();
        let reserve = |tags: &mut BTreeSet<RegionTagId>, tag: &RegionTagId, what: &str| {
            if tags.insert(tag.clone()) {
                Ok(())
            } else {
                Err(topology_error(
                    "FV_REGION_TAG_OVERLAP",
                    format!("{what} tag `{tag}` is already a boundary, region or interface tag"),
                ))
            }
        };
        let mut region_index = BTreeMap::<&RegionTagId, usize>::new();
        for (index, region) in partition.regions.iter().enumerate() {
            if region_index.insert(&region.tag, index).is_some() {
                return Err(topology_error(
                    "FV_REGION_TAG_DUPLICATE",
                    format!("region `{}` is declared twice", region.tag),
                ));
            }
            reserve(&mut tags, &region.tag, "region")?;
            if region.cells.len() != dimension {
                return Err(topology_error(
                    "FV_REGION_RANGE_INVALID",
                    format!(
                        "region `{}` gives {} ranges for a {dimension}-D box",
                        region.tag,
                        region.cells.len()
                    ),
                ));
            }
            for (axis, ([start, end], count)) in region.cells.iter().zip(counts).enumerate() {
                if start >= end || end > count {
                    return Err(topology_error(
                        "FV_REGION_RANGE_INVALID",
                        format!(
                            "region `{}` range [{start}, {end}) on axis {axis} is empty or \
                             exceeds {count} cells",
                            region.tag
                        ),
                    ));
                }
            }
        }
        let mut of_cell = vec![usize::MAX; cells];
        for (index, region) in partition.regions.iter().enumerate() {
            for cell in 0..cells {
                let at = [cell % counts[0], cell / counts[0]];
                if region
                    .cells
                    .iter()
                    .zip(at)
                    .all(|([start, end], k)| (*start..*end).contains(&k))
                {
                    if of_cell[cell] != usize::MAX {
                        return Err(topology_error(
                            "FV_REGION_OVERLAP",
                            format!(
                                "cell {cell} lies in regions `{}` and `{}`",
                                partition.regions[of_cell[cell]].tag, region.tag
                            ),
                        ));
                    }
                    of_cell[cell] = index;
                }
            }
        }
        if let Some(cell) = of_cell.iter().position(|index| *index == usize::MAX) {
            return Err(topology_error(
                "FV_REGION_UNCOVERED",
                format!("cell {cell} lies in no region"),
            ));
        }
        let mut pair_interface = BTreeMap::<[usize; 2], &RegionTagId>::new();
        let mut interfaces = BTreeMap::new();
        for interface in &partition.interfaces {
            let pair = interface
                .between
                .iter()
                .map(|tag| {
                    region_index.get(tag).copied().ok_or_else(|| {
                        topology_error(
                            "FV_REGION_INTERFACE_UNKNOWN",
                            format!(
                                "interface `{}` names undeclared region `{tag}`",
                                interface.tag
                            ),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut key = [pair[0], pair[1]];
            key.sort_unstable();
            if key[0] == key[1] {
                return Err(topology_error(
                    "FV_REGION_INTERFACE_UNKNOWN",
                    format!("interface `{}` names one region twice", interface.tag),
                ));
            }
            reserve(&mut tags, &interface.tag, "interface")?;
            if pair_interface.insert(key, &interface.tag).is_some() {
                return Err(topology_error(
                    "FV_REGION_INTERFACE_DUPLICATE",
                    format!(
                        "regions `{}` and `{}` have two interfaces",
                        interface.between[0], interface.between[1]
                    ),
                ));
            }
            interfaces.insert(
                interface.tag.clone(),
                FvInterface {
                    between: interface.between.clone(),
                    interior: Vec::new(),
                    periodic: Vec::new(),
                },
            );
        }
        for (block, is_periodic) in [(interior, false), (periodic, true)] {
            for face in 0..block.len() {
                let (minus, plus) = (of_cell[block.minus[face]], of_cell[block.plus[face]]);
                if minus == plus {
                    continue;
                }
                let mut key = [minus, plus];
                key.sort_unstable();
                let tag = pair_interface.get(&key).ok_or_else(|| {
                    topology_error(
                        "FV_REGION_INTERFACE_UNTAGGED",
                        format!(
                            "face between regions `{}` and `{}` has no interface tag",
                            partition.regions[minus].tag, partition.regions[plus].tag
                        ),
                    )
                })?;
                let interface = interfaces.get_mut(*tag).expect("declared interface");
                if is_periodic {
                    interface.periodic.push(face);
                } else {
                    interface.interior.push(face);
                }
            }
        }
        if let Some((tag, _)) = interfaces.iter().find(|(_, faces)| faces.is_empty()) {
            return Err(topology_error(
                "FV_REGION_INTERFACE_EMPTY",
                format!("interface `{tag}` separates no faces"),
            ));
        }
        Ok(Self {
            cell_regions: of_cell
                .iter()
                .map(|index| partition.regions[*index].tag.clone())
                .collect(),
            interfaces,
        })
    }
}

fn mean(points: &[&[f64]]) -> Vec<f64> {
    let dimension = points[0].len();
    let scale = 1.0 / points.len() as f64;
    (0..dimension)
        .map(|axis| points.iter().map(|point| point[axis]).sum::<f64>() * scale)
        .collect()
}

fn difference(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| a - b).collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn simplex_volume(points: &[&[f64]]) -> f64 {
    let edge = |index: usize| difference(points[index], points[0]);
    match points.len() {
        2 => edge(1)[0].abs(),
        3 => {
            let (a, b) = (edge(1), edge(2));
            (a[0] * b[1] - a[1] * b[0]).abs() * 0.5
        }
        4 => {
            let (a, b, c) = (edge(1), edge(2), edge(3));
            dot(&a, &cross(&b, &c)).abs() / 6.0
        }
        _ => f64::NAN,
    }
}

fn cross(a: &[f64], b: &[f64]) -> Vec<f64> {
    vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Area and an (unoriented) unit normal of a simplex facet with `dimension` vertices.
fn simplex_facet_normal(points: &[&[f64]]) -> (f64, Vec<f64>) {
    match points.len() {
        1 => (1.0, vec![1.0]),
        2 => {
            let edge = difference(points[1], points[0]);
            let length = edge[0].hypot(edge[1]);
            (length, vec![edge[1] / length, -edge[0] / length])
        }
        _ => {
            let normal = cross(
                &difference(points[1], points[0]),
                &difference(points[2], points[0]),
            );
            let norm = dot(&normal, &normal).sqrt();
            (
                norm * 0.5,
                normal.iter().map(|value| value / norm).collect(),
            )
        }
    }
}
