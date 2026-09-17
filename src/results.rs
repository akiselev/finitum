//! SHOW-1: portable affine-simplex geometry and exact P1 field samples for inspection.
//! Rendering consumes these owner-produced samples; it does not reconstruct DOF conventions.
use crate::realization::CellGeometry;
use crate::{AffineMap, Cell, CellId, FieldSampler, FinitumError, Mesh, SampledFamily, VertexId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultMesh {
    pub dimension: usize,
    pub vertices: Vec<Vec<f64>>,
    pub cells: Vec<Vec<usize>>,
}
impl ResultMesh {
    pub fn capture(mesh: &Mesh) -> Self {
        Self {
            dimension: mesh.dimension(),
            vertices: mesh.vertices().to_vec(),
            cells: mesh
                .cells()
                .iter()
                .map(|c| c.vertices.iter().map(|v| v.0).collect())
                .collect(),
        }
    }
    pub fn realize(&self) -> Result<Mesh, FinitumError> {
        let mesh = Mesh::new(
            self.dimension,
            self.vertices.clone(),
            self.cells
                .iter()
                .map(|c| Cell {
                    vertices: c.iter().copied().map(VertexId).collect(),
                })
                .collect(),
        )?;
        if mesh.cells().is_empty() {
            return Err(invalid("result mesh is empty"));
        }
        for cell in 0..mesh.cells().len() {
            CellGeometry::new(&mesh, CellId(cell))?;
        }
        Ok(mesh)
    }
    pub fn identity(&self) -> Result<String, FinitumError> {
        let bytes = serde_json::to_vec(self).map_err(|e| invalid(&e.to_string()))?;
        Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
    }
}
fn invalid(message: &str) -> FinitumError {
    FinitumError::InvalidRealization(message.into())
}

/// Exact values at each cell's ordered simplex vertices. The location is cell-vertex,
/// not a claim that discontinuous fields share one global value at a vertex.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearFieldSamples {
    pub schema: String,
    pub mesh_identity: String,
    pub components: usize,
    pub cell_values: Vec<Vec<Vec<f64>>>,
}
impl LinearFieldSamples {
    pub fn capture(sampler: &FieldSampler<'_>) -> Result<Self, FinitumError> {
        if !matches!(sampler.family(), SampledFamily::Lagrange { order: 1, .. }) {
            return Err(FinitumError::SamplingUnsupported { family: format!("{:?}", sampler.family()), reason: "saved linear inspection requires a P1 field; higher-order and Piola fields must not be relabeled P1".into() });
        }
        let mesh = sampler.mesh();
        let cell_values = mesh
            .cells()
            .iter()
            .enumerate()
            .map(|(cell, c)| {
                c.vertices
                    .iter()
                    .map(|v| sampler.value_at(CellId(cell), &mesh.vertices()[v.0]))
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            schema: "finitum-linear-field-samples/1".into(),
            mesh_identity: ResultMesh::capture(mesh).identity()?,
            components: sampler.component_count(),
            cell_values,
        })
    }
    pub fn validate(&self, mesh: &ResultMesh) -> Result<(), FinitumError> {
        mesh.realize()?;
        if self.schema != "finitum-linear-field-samples/1"
            || self.mesh_identity != mesh.identity()?
            || ![1, mesh.dimension].contains(&self.components)
            || self.cell_values.len() != mesh.cells.len()
        {
            return Err(invalid(
                "result field schema, mesh identity or extent mismatch",
            ));
        }
        for cell in &self.cell_values {
            if cell.len() != mesh.dimension + 1
                || cell
                    .iter()
                    .any(|v| v.len() != self.components || v.iter().any(|x| !x.is_finite()))
            {
                return Err(invalid("invalid cell-vertex field samples"));
            }
        }
        Ok(())
    }
    pub fn value_at_reference(
        &self,
        cell: usize,
        reference: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        let values = self
            .cell_values
            .get(cell)
            .ok_or_else(|| invalid("missing result cell"))?;
        if reference.iter().any(|v| !v.is_finite())
            || values.len() != reference.len() + 1
            || values.iter().any(|v| v.len() != self.components)
        {
            return Err(invalid("invalid result sample extent"));
        }
        let (basis, _) = crate::element::simplex_basis(reference.len(), 1, reference)?;
        let mut out = vec![0.0; self.components];
        for (weight, value) in basis.iter().zip(values) {
            for (o, v) in out.iter_mut().zip(value) {
                *o += weight * v;
            }
        }
        if out.iter().any(|value| !value.is_finite()) {
            return Err(invalid("non-finite saved field sample"));
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DisplayPoint {
    pub cell: usize,
    pub coordinates: Vec<f64>,
    pub reference: Vec<f64>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DisplaySurface {
    pub points: Vec<DisplayPoint>,
    pub triangles: Vec<[usize; 3]>,
}
impl DisplaySurface {
    fn polygon(&mut self, points: &[DisplayPoint]) {
        if points.len() < 3 {
            return;
        }
        let start = self.points.len();
        self.points.extend_from_slice(points);
        for i in 1..points.len() - 1 {
            self.triangles.push([start, start + i, start + i + 1]);
        }
    }
}
fn interpolate(a: &DisplayPoint, b: &DisplayPoint, t: f64) -> DisplayPoint {
    DisplayPoint {
        cell: a.cell,
        coordinates: a
            .coordinates
            .iter()
            .zip(&b.coordinates)
            .map(|(a, b)| a + t * (b - a))
            .collect(),
        reference: a
            .reference
            .iter()
            .zip(&b.reference)
            .map(|(a, b)| a + t * (b - a))
            .collect(),
    }
}
/// Exterior surface, optionally clipped to x <= cut, including an owner-sampled cut cap.
/// Clipping is display geometry only; caps do not become physical boundary conditions.
pub fn inspection_surface(
    mesh: &ResultMesh,
    cut: Option<f64>,
) -> Result<DisplaySurface, FinitumError> {
    let realized = mesh.realize()?;
    if ![2, 3].contains(&mesh.dimension) || cut.is_some_and(|x| !x.is_finite()) {
        return Err(invalid("inspection needs finite cuts and 2D/3D simplices"));
    }
    let point = |cell: usize, vertex: usize| -> Result<DisplayPoint, FinitumError> {
        let coordinates = mesh.vertices[vertex].clone();
        let reference =
            AffineMap::from_cell(&realized, CellId(cell))?.reference_point(&coordinates)?;
        Ok(DisplayPoint {
            cell,
            coordinates,
            reference,
        })
    };
    let mut faces: BTreeMap<Vec<usize>, Vec<(usize, Vec<usize>)>> = BTreeMap::new();
    for (cell, vertices) in mesh.cells.iter().enumerate() {
        if mesh.dimension == 2 {
            faces.insert(vec![cell], vec![(cell, vertices.clone())]);
        } else {
            for omitted in 0..4 {
                let face: Vec<_> = vertices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != omitted)
                    .map(|(_, v)| *v)
                    .collect();
                let mut key = face.clone();
                key.sort();
                faces.entry(key).or_default().push((cell, face));
            }
        }
    }
    let mut surface = DisplaySurface::default();
    for entries in faces.values().filter(|e| e.len() == 1) {
        let (cell, face) = &entries[0];
        let polygon = face
            .iter()
            .map(|v| point(*cell, *v))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(cut) = cut {
            let mut clipped = Vec::new();
            for i in 0..polygon.len() {
                let a = &polygon[i];
                let b = &polygon[(i + 1) % polygon.len()];
                let ai = a.coordinates[0] <= cut;
                let bi = b.coordinates[0] <= cut;
                if ai {
                    clipped.push(a.clone());
                }
                if ai != bi {
                    clipped.push(interpolate(
                        a,
                        b,
                        (cut - a.coordinates[0]) / (b.coordinates[0] - a.coordinates[0]),
                    ));
                }
            }
            surface.polygon(&clipped);
        } else {
            surface.polygon(&polygon);
        }
    }
    if let Some(cut) = cut.filter(|_| mesh.dimension == 3) {
        for (cell, vertices) in mesh.cells.iter().enumerate() {
            let mut cap: Vec<DisplayPoint> = Vec::new();
            for i in 0..4 {
                for j in i + 1..4 {
                    let a = point(cell, vertices[i])?;
                    let b = point(cell, vertices[j])?;
                    if (a.coordinates[0] < cut && b.coordinates[0] >= cut)
                        || (b.coordinates[0] < cut && a.coordinates[0] >= cut)
                    {
                        let p = interpolate(
                            &a,
                            &b,
                            (cut - a.coordinates[0]) / (b.coordinates[0] - a.coordinates[0]),
                        );
                        if !cap.iter().any(|q| {
                            q.reference
                                .iter()
                                .zip(&p.reference)
                                .all(|(a, b)| (a - b).abs() < 1e-12)
                        }) {
                            cap.push(p);
                        }
                    }
                }
            }
            if cap.len() >= 3 {
                let y = cap.iter().map(|p| p.coordinates[1]).sum::<f64>() / cap.len() as f64;
                let z = cap.iter().map(|p| p.coordinates[2]).sum::<f64>() / cap.len() as f64;
                cap.sort_by(|a, b| {
                    (a.coordinates[2] - z)
                        .atan2(a.coordinates[1] - y)
                        .total_cmp(&(b.coordinates[2] - z).atan2(b.coordinates[1] - y))
                });
                surface.polygon(&cap);
            }
        }
    }
    Ok(surface)
}
