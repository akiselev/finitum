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

use std::collections::BTreeMap;

use scientia::Digest;
use serde::Serialize;

use crate::{FacetId, FacetTopology, FinitumError, MeshProfile, RegionTagId, TaggedMesh};

pub const FV_TOPOLOGY_SCHEMA: &str = "finitum-fv-topology/1";

const FV_TOPOLOGY_CELL_CAP: usize = 1_000_000;
const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];

fn topology_error(code: &str, message: impl std::fmt::Display) -> FinitumError {
    FinitumError::InvalidRealization(format!("{code}: {message}"))
}

/// Node placement along one tensor axis.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AxisSpacing {
    /// `cells` equal cells on `[min, max]`; the last node is `max` exactly.
    Uniform { min: f64, max: f64, cells: usize },
    /// Explicit strictly increasing node coordinates (graded spacing); at least two.
    Nodes { nodes: Vec<f64> },
}

/// One axis of a [`TensorBox`].
#[derive(Clone, Debug, PartialEq, Serialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TensorBox {
    pub axes: Vec<TensorAxis>,
}

/// The cell family a topology was built from; it fixes the face count of every cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
    /// The tagged-mesh digest or the digest of the tensor profile it was built from.
    source: Digest,
    #[serde(skip)]
    digest: Digest,
}

/// The checked invariants of an [`FvTopology`].
#[derive(Clone, Debug, PartialEq, Serialize)]
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
            mesh.digest.clone(),
        ))
    }

    /// The topology of a [`TensorBox`]. Faces are ordered by axis, then by the cell row across
    /// the axis, then along it; boundary tags are `x_min`, `x_max`, `y_min`, `y_max` on the
    /// non-periodic axes (the [`crate::MeshProfile::SimplexBox`] names).
    pub fn from_tensor_box(spec: &TensorBox) -> Result<Self, FinitumError> {
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
        let source = Digest::blake3(
            &serde_json::to_vec(&("finitum-tensor-box/1", spec))
                .expect("tensor box is serializable"),
        );
        Ok(Self::finish(
            dimension,
            FvCellShape::Tensor,
            volume,
            centroid,
            interior,
            periodic,
            boundary,
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
