//! Tetrahedral realization of a committed CADabra B-rep body (Provider V1).
//!
//! The first slice realizes bodies that are extrusions of a planar profile: two planar caps
//! with opposite normals, every other face perpendicular to them, and a published display
//! tessellation whose side-face samples all lie on the caps' rims. The boundary is the published
//! tessellation: the bottom cap's rim samples are the profile's boundary and the side faces'
//! samples. Finitum generates the interior deterministically: rim edges longer than the target
//! spacing are split, a fixed grid of interior points is inserted and the profile is flipped to
//! its Delaunay triangulation (`cad_profile`), then `layers` uniform layers between the caps,
//! three tetrahedra per triangular prism split by the global vertex order (so neighbouring
//! prisms share their quadrilateral diagonals).
//!
//! Every boundary node knows the faces it lies on, and the edge or vertex where it lies on two
//! or three. A boundary design velocity `V_n` from
//! [`BrepBodyProvider::boundary_velocity`] (one value per node-face incidence) drives the mesh
//! motion, which is the derivative of the generator itself: a published bottom-cap sample
//! moves by the minimum-norm displacement whose normal component on each of its faces is that
//! face's `V_n` (no tangential gauge), a rim split by the interpolation of its end points, a
//! grid point (fixed in the profile plane) with its cap only, and
//! layer `k` adds `k / layers` of the extent's rate (the caps' normal speeds) along the
//! extrusion direction. So the motion is what regenerating the mesh from the moved published
//! samples does, while the body stays an extrusion with the same tessellation topology. It is
//! linear in `V_n`; its transpose maps a node gradient back to incidence weights, which
//! [`BrepBodyProvider::boundary_velocity_transpose`] turns into a design gradient. Incidences
//! the generator does not read (side and top samples, rim splits) carry zero weight.

use std::collections::BTreeMap;

use cadabra_provider::{
    BodyClosure, BrepBodyProvider, DesignName, DesignSeed, DesignVelocity, DisplayArtifact,
    DisplaySource, ProviderError,
};
use serde::Serialize;

use crate::cad_profile::{ProfileOrigin, PublishedProfile, mesh_profile};
use crate::{Cell, DofId, DofMap, ElementRestriction, FinitumError, Mesh, VertexId};

/// Hard cap for realized vertices or tetrahedra.
const WORK_CAP: usize = 2_000_000;

/// Immutable identity of the committed body a realization was built from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CadBodySource {
    /// Durable identity of the body (hex).
    pub body: String,
    /// The committed B-rep revision.
    pub revision: [u8; 32],
    /// Digest of the provider's whole body binding.
    pub binding_digest: [u8; 32],
}

/// How the interior of an extruded profile is generated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ExtrusionRefinement {
    /// Target edge length in the profile: longer rim edges are split, and interior grid points
    /// are this far apart (and at least half of it from the rim).
    pub spacing: f64,
    /// Layers of prisms between the caps (at least one).
    pub layers: usize,
}

/// The B-rep entity a realized vertex lies on. Indices are into the provider binding's
/// `faces`, `edges` and `vertices`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum CadBodySupport {
    /// Strictly inside the body.
    Interior,
    /// On one face.
    Face {
        /// The face.
        face: usize,
    },
    /// On the edge where two faces meet.
    Edge {
        /// The edge.
        edge: usize,
        /// Its two faces, ascending.
        faces: [usize; 2],
    },
    /// At a vertex where three or more faces meet.
    Vertex {
        /// The vertex.
        vertex: usize,
        /// Its faces, ascending.
        faces: Vec<usize>,
    },
}

/// One realized vertex and the B-rep entity it lies on.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadBodyNode {
    /// The mesh vertex.
    pub vertex: VertexId,
    /// Its support.
    pub support: CadBodySupport,
}

/// One node-face incidence: a boundary vertex on one face, the point a boundary velocity is
/// read at, and the face's outward unit normal there.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadBoundaryIncidence {
    /// The mesh vertex.
    pub vertex: VertexId,
    /// The face (index into the binding's faces).
    pub face: usize,
    /// The vertex position.
    pub point: [f64; 3],
    /// Outward unit normal of the face at the point.
    pub normal: [f64; 3],
}

/// One row of the mesh motion: a node's displacement as a combination of incidence speeds.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct MotionRow {
    terms: Vec<(usize, [f64; 3])>,
}

/// Finitum-owned P1 tetrahedral realization of one committed extruded body.
#[derive(Clone, Debug, PartialEq)]
pub struct CadBodyRealization {
    source: CadBodySource,
    refinement: ExtrusionRefinement,
    mesh: Mesh,
    nodes: Vec<CadBodyNode>,
    incidences: Vec<CadBoundaryIncidence>,
    motion: Vec<MotionRow>,
    digest: [u8; 32],
}

fn invalid(message: impl Into<String>) -> FinitumError {
    FinitumError::InvalidCadGeometry(message.into())
}

fn provider_error(error: ProviderError) -> FinitumError {
    FinitumError::InvalidCadGeometry(error.to_string())
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(a: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(a, a).sqrt();
    (length > 0.0 && length.is_finite()).then(|| a.map(|c| c / length))
}

impl CadBodyRealization {
    /// Realizes the bound body of `provider` from its published display tessellation
    /// `display`, as an extrusion (see the module documentation).
    ///
    /// # Errors
    /// [`FinitumError::StaleBrepRevision`] when `display` is not a tessellation of the
    /// provider's committed revision, and [`FinitumError::InvalidCadGeometry`] when the body is
    /// not an admitted extrusion or the tessellation does not realize it as one.
    pub fn extruded(
        provider: &BrepBodyProvider,
        display: &DisplayArtifact,
        refinement: ExtrusionRefinement,
    ) -> Result<Self, FinitumError> {
        let binding = provider.binding();
        match &display.source {
            DisplaySource::Revision { revision, .. } if *revision == binding.brep_revision => {}
            _ => return Err(FinitumError::StaleBrepRevision),
        }
        if refinement.layers == 0 {
            return Err(invalid("an extrusion needs at least one layer"));
        }
        let body = display
            .bodies
            .iter()
            .find(|body| body.body == binding.body)
            .ok_or_else(|| invalid("the display artifact has no mesh of the bound body"))?;
        if body.closure != BodyClosure::ClosedSolid {
            return Err(invalid("the body's display mesh is not a closed solid"));
        }
        let scale = body
            .positions
            .iter()
            .flatten()
            .fold(1.0_f64, |m, c| m.max(c.abs()));
        let tolerance = 64.0 * f64::EPSILON * scale;

        // Faces of the display, by binding index, with their display vertices.
        let mut faces = Vec::with_capacity(body.faces.len());
        for display_face in &body.faces {
            let index = binding
                .faces
                .iter()
                .position(|f| f.uid == display_face.face)
                .ok_or_else(|| invalid("a display face is not a face of the binding"))?;
            let range = display_face.triangles.start as usize..display_face.triangles.end as usize;
            let triangles = body
                .triangles
                .get(range.clone())
                .ok_or_else(|| invalid("a display face's triangle range is out of bounds"))?
                .iter()
                .map(|t| t.map(|i| i as usize))
                .collect::<Vec<_>>();
            if triangles.is_empty() {
                return Err(invalid("a display face has no triangles"));
            }
            let normals = body.triangle_normals[range].to_vec();
            faces.push((index, triangles, normals));
        }
        if faces.len() != binding.faces.len() {
            return Err(invalid(
                "the display does not mesh every face of the binding",
            ));
        }

        // The caps: two planar faces with opposite constant normals, every other face's
        // normals perpendicular to them. The first such pair in face order is used.
        let constant_normal = |(index, _, normals): &(usize, Vec<[usize; 3]>, Vec<[f64; 3]>)| {
            (binding.faces[*index].support == "plane").then_some(())?;
            let n = normals[0];
            normals
                .iter()
                .all(|m| dot(*m, n) >= 1.0 - 1e-12)
                .then_some(n)
        };
        let perpendicular = |face: &(usize, Vec<[usize; 3]>, Vec<[f64; 3]>), d: [f64; 3]| {
            face.1
                .iter()
                .flatten()
                .all(|&i| dot(body.normals[i], d).abs() <= 1e-9)
        };
        let mut caps = None;
        'search: for (i, first) in faces.iter().enumerate() {
            let Some(n) = constant_normal(first) else {
                continue;
            };
            for (j, second) in faces.iter().enumerate().skip(i + 1) {
                let Some(m) = constant_normal(second) else {
                    continue;
                };
                if dot(n, m) > -1.0 + 1e-12 {
                    continue;
                }
                let sides_perpendicular = faces
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .all(|(_, face)| perpendicular(face, n));
                if sides_perpendicular {
                    caps = Some((i, j, n.map(|c| -c)));
                    break 'search;
                }
            }
        }
        let (bottom, top, direction) =
            caps.ok_or_else(|| invalid("the body is not an extrusion of a planar profile"))?;

        // Profile frame: `direction` from the bottom cap up, `e1`, `e2` spanning the profile.
        let least = (0..3)
            .min_by(|&a, &b| direction[a].abs().total_cmp(&direction[b].abs()))
            .unwrap_or(0);
        let mut axis = [0.0; 3];
        axis[least] = 1.0;
        let e1 = normalized(sub(axis, direction.map(|c| c * direction[least])))
            .ok_or_else(|| invalid("degenerate extrusion direction"))?;
        let e2 = cross(direction, e1);
        let planar = |x: [f64; 3]| [dot(x, e1), dot(x, e2)];
        let height = |x: [f64; 3]| dot(x, direction);

        // The bottom cap's samples, numbered in the mesher's emission order (stable under small
        // design steps, unlike the order of its triangles), seed the profile.
        let mut local = BTreeMap::<usize, usize>::new();
        for &display_index in faces[bottom].1.iter().flatten() {
            local.insert(display_index, 0);
        }
        let mut points = Vec::with_capacity(local.len());
        for (index, (&display_index, slot)) in local.iter_mut().enumerate() {
            *slot = index;
            points.push(body.positions[display_index]);
        }
        let triangles = faces[bottom]
            .1
            .iter()
            .map(|t| t.map(|display_index| local[&display_index]))
            .collect::<Vec<_>>();
        let bottom_height = height(points[0]);
        if points
            .iter()
            .any(|p| (height(*p) - bottom_height).abs() > tolerance)
        {
            return Err(invalid("the bottom cap is not planar"));
        }
        let top_height = height(body.positions[faces[top].1[0][0]]);
        let extent = top_height - bottom_height;
        if extent <= tolerance {
            return Err(invalid("the caps do not bound a positive extrusion height"));
        }
        let planar_points = points.iter().map(|p| planar(*p)).collect::<Vec<_>>();
        let find_profile_point = |x: [f64; 3]| {
            let q = planar(x);
            planar_points
                .iter()
                .position(|p| (p[0] - q[0]).abs() <= tolerance && (p[1] - q[1]).abs() <= tolerance)
        };
        let mut edge_uses = BTreeMap::<(usize, usize), usize>::new();
        for [a, b, c] in &triangles {
            for (p, q) in [(*a, *b), (*b, *c), (*c, *a)] {
                *edge_uses.entry((p.min(q), p.max(q))).or_default() += 1;
            }
        }
        if edge_uses.values().any(|&uses| uses > 2) {
            return Err(invalid("the profile triangulation is not manifold"));
        }

        // Side faces through the rim points, from the side faces' own samples.
        let mut sides = vec![Vec::<(usize, [f64; 3])>::new(); points.len()];
        for (k, (index, face_triangles, _)) in faces.iter().enumerate() {
            if k == bottom || k == top {
                continue;
            }
            for &display_index in face_triangles.iter().flatten() {
                let x = body.positions[display_index];
                let h = height(x);
                let at_bottom = (h - bottom_height).abs() <= tolerance;
                if !at_bottom && (h - top_height).abs() > tolerance {
                    return Err(invalid("a side face is sampled between the caps"));
                }
                let point = find_profile_point(x)
                    .ok_or_else(|| invalid("a side face sample is not on the profile rim"))?;
                let normal = body.normals[display_index];
                match sides[point].iter_mut().find(|(f, _)| f == index) {
                    // Prefer the bottom sample's normal; a prism's side normal is constant
                    // along the extrusion.
                    Some(entry) if at_bottom => entry.1 = normal,
                    Some(_) => {}
                    None => sides[point].push((*index, normal)),
                }
            }
        }
        for &display_index in faces[top].1.iter().flatten() {
            if find_profile_point(body.positions[display_index]).is_none() {
                return Err(invalid("a top cap sample is not above a profile point"));
            }
        }
        let mut rim = BTreeMap::new();
        for (&(p, q), &uses) in &edge_uses {
            if uses != 1 {
                continue;
            }
            let mut common = sides[p]
                .iter()
                .filter(|(f, _)| sides[q].iter().any(|(g, _)| g == f))
                .map(|(f, _)| *f);
            let (Some(face), None) = (common.next(), common.next()) else {
                return Err(invalid(
                    "a profile rim edge does not lie on exactly one side face",
                ));
            };
            rim.insert((p, q), face);
        }
        let on_rim = |point: usize| rim.keys().any(|&(p, q)| p == point || q == point);
        for (point, point_sides) in sides.iter().enumerate() {
            if point_sides.is_empty() == on_rim(point) {
                return Err(invalid("the profile rim and the side faces disagree"));
            }
        }

        let lift = |q: [f64; 2]| {
            [0, 1, 2].map(|k| q[0] * e1[k] + q[1] * e2[k] + bottom_height * direction[k])
        };
        let profile = mesh_profile(
            PublishedProfile {
                points,
                planar: planar_points,
                triangles,
                sides,
                rim,
            },
            refinement.spacing,
            lift,
            WORK_CAP,
        )?;

        // Layers and nodes.
        let layers = refinement.layers;
        let column = profile.points.len();
        let node_count = column
            .checked_mul(layers + 1)
            .filter(|&n| n <= WORK_CAP)
            .ok_or_else(|| invalid("the realization exceeds the work cap"))?;
        let cell_count = profile
            .triangles
            .len()
            .checked_mul(3 * layers)
            .filter(|&n| n <= WORK_CAP)
            .ok_or_else(|| invalid("the realization exceeds the work cap"))?;
        let bottom_face = faces[bottom].0;
        let top_face = faces[top].0;
        let mut vertices = Vec::with_capacity(node_count);
        let mut nodes = Vec::with_capacity(node_count);
        let mut incidences = Vec::new();
        let mut incidence_ranges = Vec::with_capacity(node_count);
        for layer in 0..=layers {
            let lift = extent * layer as f64 / layers as f64;
            for p in 0..column {
                let x = profile.points[p];
                let position = [0, 1, 2].map(|k| x[k] + lift * direction[k]);
                let vertex = VertexId(vertices.len());
                vertices.push(position.to_vec());
                let mut on = Vec::new();
                if layer == 0 {
                    on.push((bottom_face, direction.map(|c| -c)));
                }
                on.extend(profile.sides[p].iter().copied());
                if layer == layers {
                    on.push((top_face, direction));
                }
                on.sort_by_key(|(face, _)| *face);
                let start = incidences.len();
                for &(face, normal) in &on {
                    incidences.push(CadBoundaryIncidence {
                        vertex,
                        face,
                        point: position,
                        normal,
                    });
                }
                incidence_ranges.push(start..incidences.len());
                let face_set = on.iter().map(|(f, _)| *f).collect::<Vec<_>>();
                let support = support_of(provider, body, &face_set, position, tolerance)?;
                nodes.push(CadBodyNode { vertex, support });
            }
        }

        // Mesh motion: the derivative of this generator. A published bottom sample moves by
        // the minimum-norm displacement whose normal component on each of its faces is that
        // face's speed; a rim split by the interpolation of its end points; layer `k` adds
        // `k / layers` of the extent's rate along the extrusion direction (the speeds of the
        // two caps along their outward normals).
        let mut profile_rows: Vec<BTreeMap<usize, [f64; 3]>> = Vec::with_capacity(column);
        for (origin, range) in profile.origins.iter().zip(&incidence_ranges) {
            let row = match *origin {
                ProfileOrigin::Sample | ProfileOrigin::Grid => {
                    let range = range.clone();
                    let normals = incidences[range.clone()]
                        .iter()
                        .map(|i| i.normal)
                        .collect::<Vec<_>>();
                    range.zip(minimum_norm_columns(&normals)?).collect()
                }
                ProfileOrigin::RimSplit {
                    from: (a, b),
                    weight,
                } => {
                    let mut row = BTreeMap::<usize, [f64; 3]>::new();
                    for (parent, share) in [(a, 1.0 - weight), (b, weight)] {
                        for (&incidence, c) in &profile_rows[parent] {
                            let entry = row.entry(incidence).or_insert([0.0; 3]);
                            for axis in 0..3 {
                                entry[axis] += share * c[axis];
                            }
                        }
                    }
                    row
                }
            };
            profile_rows.push(row);
        }
        let cap_incidence = |node: usize, face: usize| {
            incidence_ranges[node]
                .clone()
                .find(|&i| incidences[i].face == face)
                .ok_or_else(|| invalid("a cap node has no cap incidence"))
        };
        let extent_rate = [
            cap_incidence(0, bottom_face)?,
            cap_incidence(layers * column, top_face)?,
        ];
        let mut motion = Vec::with_capacity(node_count);
        for layer in 0..=layers {
            let t = layer as f64 / layers as f64;
            for row in &profile_rows {
                let mut terms = row.iter().map(|(&i, c)| (i, *c)).collect::<Vec<_>>();
                if layer > 0 {
                    terms.extend(extent_rate.map(|i| (i, direction.map(|c| c * t))));
                }
                motion.push(MotionRow { terms });
            }
        }

        // Three tetrahedra per prism, split by the global vertex order.
        let mut cells = Vec::with_capacity(cell_count);
        for layer in 0..layers {
            for triangle in &profile.triangles {
                let mut t = *triangle;
                t.sort_unstable();
                let low = t.map(|p| layer * column + p);
                let high = t.map(|p| (layer + 1) * column + p);
                for mut tet in [
                    [low[0], low[1], low[2], high[0]],
                    [low[1], low[2], high[0], high[1]],
                    [low[2], high[0], high[1], high[2]],
                ] {
                    let corner = |k: usize| -> [f64; 3] {
                        let v = &vertices[tet[k]];
                        [v[0], v[1], v[2]]
                    };
                    let volume = dot(
                        sub(corner(1), corner(0)),
                        cross(sub(corner(2), corner(0)), sub(corner(3), corner(0))),
                    );
                    if volume == 0.0 || !volume.is_finite() {
                        return Err(invalid("the extrusion produced a degenerate tetrahedron"));
                    }
                    if volume < 0.0 {
                        tet.swap(2, 3);
                    }
                    cells.push(Cell {
                        vertices: tet.into_iter().map(VertexId).collect(),
                    });
                }
            }
        }
        let mesh = Mesh::new(3, vertices, cells)?;
        let source = CadBodySource {
            body: binding.body.to_hex(),
            revision: binding.brep_revision.bytes(),
            binding_digest: binding.semantic_digest.bytes(),
        };
        let digest = realization_digest(&source, refinement, &mesh, &nodes, &incidences, &motion)?;
        Ok(Self {
            source,
            refinement,
            mesh,
            nodes,
            incidences,
            motion,
            digest,
        })
    }

    /// The committed body this realization was built from.
    pub fn source(&self) -> &CadBodySource {
        &self.source
    }

    /// The interior generation it used.
    pub fn refinement(&self) -> ExtrusionRefinement {
        self.refinement
    }

    /// The P1 tetrahedral mesh.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// Every vertex with its B-rep support, in vertex order.
    pub fn nodes(&self) -> &[CadBodyNode] {
        &self.nodes
    }

    /// The node-face incidences, by vertex then face.
    pub fn incidences(&self) -> &[CadBoundaryIncidence] {
        &self.incidences
    }

    /// The vertices on face `face` (index into the binding's faces), ascending.
    pub fn face_vertices(&self, face: usize) -> Vec<VertexId> {
        let mut vertices = self
            .incidences
            .iter()
            .filter(|i| i.face == face)
            .map(|i| i.vertex)
            .collect::<Vec<_>>();
        vertices.dedup();
        vertices
    }

    /// Digest over the source, refinement, mesh, supports, incidences and mesh motion.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// The nodal P1 degree-of-freedom map of the mesh.
    pub fn nodal_dof_map(&self) -> Result<DofMap, FinitumError> {
        DofMap::new(
            self.mesh.vertices().len(),
            self.mesh
                .cells()
                .iter()
                .map(|cell| ElementRestriction {
                    dofs: cell.vertices.iter().map(|vertex| DofId(vertex.0)).collect(),
                })
                .collect(),
        )
    }

    /// Requires `provider` to bind the same committed revision and body as this realization.
    ///
    /// # Errors
    /// [`FinitumError::StaleBrepRevision`] for another revision and
    /// [`FinitumError::CadGeometrySourceMismatch`] for another body or binding.
    pub fn require_source(&self, provider: &BrepBodyProvider) -> Result<(), FinitumError> {
        let binding = provider.binding();
        if binding.brep_revision.bytes() != self.source.revision {
            return Err(FinitumError::StaleBrepRevision);
        }
        if binding.body.to_hex() != self.source.body
            || binding.semantic_digest.bytes() != self.source.binding_digest
        {
            return Err(FinitumError::CadGeometrySourceMismatch);
        }
        Ok(())
    }

    /// The boundary design velocity `V_n` of every incidence along `seeds`, from the provider.
    ///
    /// # Errors
    /// As [`Self::require_source`]; [`FinitumError::SnappedCadDesign`] when a snapped decision
    /// of the body moves along the seeds (no smooth derivative exists there); and
    /// [`FinitumError::InvalidCadGeometry`] for a provider refusal.
    pub fn boundary_velocity(
        &self,
        provider: &BrepBodyProvider,
        seeds: &[DesignSeed],
    ) -> Result<Vec<f64>, FinitumError> {
        self.require_source(provider)?;
        let faces = &provider.binding().faces;
        let points = self
            .incidences
            .iter()
            .map(|i| (faces[i.face].uid, i.point))
            .collect::<Vec<_>>();
        smooth(
            provider
                .boundary_velocity(&points, seeds)
                .map_err(provider_error)?,
        )
    }

    /// The mesh-motion JVP: vertex-major node velocities (three per vertex) from the incidence
    /// speeds `boundary_velocity`.
    ///
    /// # Errors
    /// [`FinitumError::InvalidCadGeometry`] for a length that is not the incidence count or a
    /// non-finite speed.
    pub fn mesh_velocity(&self, boundary_velocity: &[f64]) -> Result<Vec<f64>, FinitumError> {
        check_finite(
            boundary_velocity,
            self.incidences.len(),
            "boundary velocity",
        )?;
        let mut out = vec![0.0; 3 * self.nodes.len()];
        for (node, row) in self.motion.iter().enumerate() {
            for &(incidence, c) in &row.terms {
                for axis in 0..3 {
                    out[3 * node + axis] += c[axis] * boundary_velocity[incidence];
                }
            }
        }
        Ok(out)
    }

    /// The transpose of [`Self::mesh_velocity`]: incidence weights from a vertex-major node
    /// gradient.
    ///
    /// # Errors
    /// [`FinitumError::InvalidCadGeometry`] for a length that is not three per vertex or a
    /// non-finite component.
    pub fn mesh_velocity_transpose(&self, node_gradient: &[f64]) -> Result<Vec<f64>, FinitumError> {
        check_finite(node_gradient, 3 * self.nodes.len(), "node gradient")?;
        let mut out = vec![0.0; self.incidences.len()];
        for (node, row) in self.motion.iter().enumerate() {
            for &(incidence, c) in &row.terms {
                for axis in 0..3 {
                    out[incidence] += c[axis] * node_gradient[3 * node + axis];
                }
            }
        }
        Ok(out)
    }

    /// The parameter JVP: vertex-major node velocities along `seeds`
    /// ([`Self::boundary_velocity`] then [`Self::mesh_velocity`]).
    ///
    /// # Errors
    /// As [`Self::boundary_velocity`].
    pub fn design_velocity(
        &self,
        provider: &BrepBodyProvider,
        seeds: &[DesignSeed],
    ) -> Result<Vec<f64>, FinitumError> {
        self.mesh_velocity(&self.boundary_velocity(provider, seeds)?)
    }

    /// The parameter VJP: `dJ/dp` for each of `parameters` from the vertex-major node gradient
    /// `dJ/dX` ([`Self::mesh_velocity_transpose`] then
    /// [`BrepBodyProvider::boundary_velocity_transpose`]).
    ///
    /// # Errors
    /// As [`Self::boundary_velocity`] and [`Self::mesh_velocity_transpose`].
    pub fn design_gradient(
        &self,
        provider: &BrepBodyProvider,
        node_gradient: &[f64],
        parameters: &[(String, DesignName)],
    ) -> Result<Vec<f64>, FinitumError> {
        self.require_source(provider)?;
        let weights = self.mesh_velocity_transpose(node_gradient)?;
        let faces = &provider.binding().faces;
        let points = self
            .incidences
            .iter()
            .map(|i| (faces[i.face].uid, i.point))
            .collect::<Vec<_>>();
        smooth(
            provider
                .boundary_velocity_transpose(&points, &weights, parameters)
                .map_err(provider_error)?,
        )
    }
}

fn smooth(velocity: DesignVelocity<Vec<f64>>) -> Result<Vec<f64>, FinitumError> {
    match velocity {
        DesignVelocity::Defined(values) => Ok(values),
        DesignVelocity::Snapped { exit, .. } => Err(FinitumError::SnappedCadDesign { exit }),
        other => Err(invalid(format!(
            "the provider gives no smooth boundary velocity: {other:?}"
        ))),
    }
}

fn check_finite(values: &[f64], expected: usize, what: &str) -> Result<(), FinitumError> {
    if values.len() != expected {
        return Err(invalid(format!(
            "{what} has length {}, expected {expected}",
            values.len()
        )));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("{what} has a non-finite component")));
    }
    Ok(())
}

/// Columns `a_j` of the minimum-norm solution `x = sum_j a_j v_j` of `n_j . x = v_j`:
/// `A = N^T (N N^T)^{-1}`.
fn minimum_norm_columns(normals: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, FinitumError> {
    let k = normals.len();
    if k > 3 {
        return Err(invalid("a node lies on more than three faces"));
    }
    let mut gram = [[0.0; 3]; 3];
    for i in 0..k {
        for j in 0..k {
            gram[i][j] = dot(normals[i], normals[j]);
        }
    }
    let inverse = invert_small(&gram, k)
        .ok_or_else(|| invalid("a node's face normals are linearly dependent"))?;
    Ok((0..k)
        .map(|j| {
            let mut column = [0.0; 3];
            for (i, normal) in normals.iter().enumerate() {
                for axis in 0..3 {
                    column[axis] += normal[axis] * inverse[i][j];
                }
            }
            column
        })
        .collect())
}

fn invert_small(m: &[[f64; 3]; 3], k: usize) -> Option<[[f64; 3]; 3]> {
    let mut out = [[0.0; 3]; 3];
    match k {
        1 => {
            (m[0][0] > 1e-12).then_some(())?;
            out[0][0] = 1.0 / m[0][0];
        }
        2 => {
            let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            (det > 1e-12).then_some(())?;
            out[0][0] = m[1][1] / det;
            out[1][1] = m[0][0] / det;
            out[0][1] = -m[0][1] / det;
            out[1][0] = -m[1][0] / det;
        }
        3 => {
            let c = |i: usize, j: usize| {
                let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
                let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
                m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
            };
            let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
            (det > 1e-12).then_some(())?;
            for (i, row) in out.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value = c(j, i) / det;
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

/// The support of a node on `faces` (ascending binding indices): the edge both faces use, or
/// the vertex at `position` where three meet.
fn support_of(
    provider: &BrepBodyProvider,
    body: &cadabra_provider::BodyDisplay,
    faces: &[usize],
    position: [f64; 3],
    tolerance: f64,
) -> Result<CadBodySupport, FinitumError> {
    let binding = provider.binding();
    let uid = |face: usize| binding.faces[face].uid;
    Ok(match faces {
        [] => CadBodySupport::Interior,
        [face] => CadBodySupport::Face { face: *face },
        [a, b] => {
            let candidates = binding
                .edges
                .iter()
                .enumerate()
                .filter(|(_, e)| e.faces.contains(&uid(*a)) && e.faces.contains(&uid(*b)))
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let edge = match candidates.as_slice() {
                [only] => *only,
                [] => return Err(invalid("two faces of a node share no edge")),
                several => nearest_edge(body, binding, several, position)?,
            };
            CadBodySupport::Edge {
                edge,
                faces: [*a, *b],
            }
        }
        several => {
            let vertex = binding
                .vertices
                .iter()
                .position(|v| (0..3).all(|k| (v.position[k] - position[k]).abs() <= tolerance))
                .ok_or_else(|| invalid("a node on three faces is not at a B-rep vertex"))?;
            CadBodySupport::Vertex {
                vertex,
                faces: several.to_vec(),
            }
        }
    })
}

/// Of several edges between the same two faces, the one whose display polyline passes
/// nearest `position`.
fn nearest_edge(
    body: &cadabra_provider::BodyDisplay,
    binding: &cadabra_provider::BrepBodyBinding,
    candidates: &[usize],
    position: [f64; 3],
) -> Result<usize, FinitumError> {
    let mut best: Option<(f64, usize)> = None;
    for &candidate in candidates {
        let uid = binding.edges[candidate].uid;
        let Some(display) = body.edges.iter().find(|e| e.edge == uid) else {
            continue;
        };
        for segment in &body.lines[display.lines.start as usize..display.lines.end as usize] {
            for &i in segment {
                let d = sub(body.line_positions[i as usize], position);
                let distance = dot(d, d);
                if best.is_none_or(|(b, _)| distance < b) {
                    best = Some((distance, candidate));
                }
            }
        }
    }
    best.map(|(_, edge)| edge)
        .ok_or_else(|| invalid("no display polyline for the edges between two faces"))
}

#[derive(Serialize)]
struct DigestPayload<'a> {
    schema: &'static str,
    source: &'a CadBodySource,
    refinement: ExtrusionRefinement,
    mesh: &'a Mesh,
    nodes: &'a [CadBodyNode],
    incidences: &'a [CadBoundaryIncidence],
    motion: &'a [MotionRow],
}

fn realization_digest(
    source: &CadBodySource,
    refinement: ExtrusionRefinement,
    mesh: &Mesh,
    nodes: &[CadBodyNode],
    incidences: &[CadBoundaryIncidence],
    motion: &[MotionRow],
) -> Result<[u8; 32], FinitumError> {
    let payload = DigestPayload {
        schema: "finitum.cad-body-realization/v0",
        source,
        refinement,
        mesh,
        nodes,
        incidences,
        motion,
    };
    let bytes = serde_json::to_vec(&payload).map_err(|error| invalid(error.to_string()))?;
    Ok(*blake3::hash(&bytes).as_bytes())
}
