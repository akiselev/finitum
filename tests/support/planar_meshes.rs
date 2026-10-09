//! Mesh transforms that build genuinely NON-nested planar interface pairs from the structured
//! boxes, and the exterior facets on a coordinate plane. Each transform keeps the cells and
//! vertex numbering, so facet ids are the structured box's own.
use finitum::{Cell, FacetId, FacetTopology, Mesh, VertexId};
use std::f64::consts::PI;

fn with_vertices(mesh: &Mesh, vertices: Vec<Vec<f64>>) -> Mesh {
    Mesh::new(mesh.dimension(), vertices, mesh.cells().to_vec()).unwrap()
}

/// Every exterior facet whose vertices all lie on `coordinate[axis] == value` (within 1e-12).
pub fn plane_facets(mesh: &Mesh, axis: usize, value: f64) -> Vec<FacetId> {
    FacetTopology::from_mesh(mesh)
        .unwrap()
        .exterior()
        .filter(|facet| {
            facet
                .vertices
                .iter()
                .all(|v| (mesh.vertices()[v.0][axis] - value).abs() <= 1e-12)
        })
        .map(|facet| facet.id)
        .collect()
}

/// A quarter turn about the `x` axis through `(y, z) = center`: `(y, z) -> (cy - (z - cz),
/// cz + (y - cy))`. A rotation keeps every cell's orientation, and it turns a square face's
/// diagonal into the other diagonal.
pub fn quarter_turn_about_x(mesh: &Mesh, center: [f64; 2]) -> Mesh {
    with_vertices(
        mesh,
        mesh.vertices()
            .iter()
            .map(|p| {
                vec![
                    p[0],
                    center[0] - (p[2] - center[1]),
                    center[1] + (p[1] - center[0]),
                ]
            })
            .collect(),
    )
}

/// In-plane deformation `(y, z) += amplitude * sin(pi y) sin(pi z)` of a unit-section box: `x`
/// is untouched (planar faces `x == const` stay planar), the lateral faces `y, z in {0, 1}`
/// are fixed (so the interface polygon stays the unit square and its boundary vertices stay
/// put), and every interior face vertex moves off the structured grid lines and diagonals.
pub fn deformed_in_plane(mesh: &Mesh, amplitude: [f64; 2]) -> Mesh {
    with_vertices(
        mesh,
        mesh.vertices()
            .iter()
            .map(|p| {
                let bump = (PI * p[1]).sin() * (PI * p[2]).sin();
                vec![p[0], p[1] + amplitude[0] * bump, p[2] + amplitude[1] * bump]
            })
            .collect(),
    )
}

pub fn translated(mesh: &Mesh, shift: [f64; 3]) -> Mesh {
    with_vertices(
        mesh,
        mesh.vertices()
            .iter()
            .map(|p| (0..3).map(|i| p[i] + shift[i]).collect())
            .collect(),
    )
}

fn signed_volume(mesh: &Mesh, cell: &Cell) -> f64 {
    let p: Vec<&Vec<f64>> = cell
        .vertices
        .iter()
        .map(|v| &mesh.vertices()[v.0])
        .collect();
    let e: Vec<[f64; 3]> = (1..4)
        .map(|k| [p[k][0] - p[0][0], p[k][1] - p[0][1], p[k][2] - p[0][2]])
        .collect();
    (e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
        - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
        + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]))
        / 6.0
}

/// A transform is admissible only if no cell degenerated or inverted: every signed volume
/// keeps the sign the structured cell had and stays above a fraction of it.
pub fn assert_same_orientation(original: &Mesh, transformed: &Mesh) {
    assert_eq!(original.cells().len(), transformed.cells().len());
    for (index, (before, after)) in original.cells().iter().zip(transformed.cells()).enumerate() {
        let (before, after) = (
            signed_volume(original, before),
            signed_volume(transformed, after),
        );
        assert!(
            before != 0.0 && after.signum() == before.signum() && after.abs() > 0.1 * before.abs(),
            "cell {index}: volume {before} -> {after}"
        );
    }
}

/// A one-sided interface mesh of tetrahedra sharing the apex `(1, 0.5, 0.5)`, one per face
/// triangle on the plane `x == 0.5` given in `(y, z)`; disjoint triangles are not required, so
/// it can carry an overlapping or a degenerate interface triangulation on purpose.
pub fn fan_mesh_on_plane(triangles: &[[[f64; 2]; 3]], apex: [f64; 3]) -> Mesh {
    let mut vertices: Vec<Vec<f64>> = Vec::new();
    let mut cells = Vec::new();
    let index = |vertices: &mut Vec<Vec<f64>>, p: Vec<f64>| {
        if let Some(i) = vertices.iter().position(|q| *q == p) {
            i
        } else {
            vertices.push(p);
            vertices.len() - 1
        }
    };
    for triangle in triangles {
        let ids: Vec<usize> = triangle
            .iter()
            .map(|p| index(&mut vertices, vec![0.5, p[0], p[1]]))
            .collect();
        let apex_id = index(&mut vertices, apex.to_vec());
        cells.push(Cell {
            vertices: vec![
                VertexId(ids[0]),
                VertexId(ids[1]),
                VertexId(ids[2]),
                VertexId(apex_id),
            ],
        });
    }
    Mesh::new(3, vertices, cells).unwrap()
}
