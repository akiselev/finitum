//! The planar profile mesh of an extruded body: the published rim samples, long rim edges split
//! to the target spacing, a fixed hexagonal lattice of interior points, and Lawson flips to the Delaunay
//! triangulation of those points (the rim edges are the only constraints, and as boundary
//! edges they are never flipped).
//!
//! The published display triangulation of a cap fans its few interior-free triangles between
//! distant rim samples, which leaves angles near a half turn; P1 elements do not converge on
//! such triangles, so the profile is re-triangulated here and the published triangulation only
//! seeds the topology.

use std::collections::BTreeMap;

use crate::FinitumError;

fn invalid(message: impl Into<String>) -> FinitumError {
    FinitumError::InvalidCadGeometry(message.into())
}

/// Where a profile point comes from; the mesh motion differentiates this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ProfileOrigin {
    /// A published sample of the bottom cap.
    Sample,
    /// A grid point fixed in the profile plane.
    Grid,
    /// A split of the rim edge from `from.0` to `from.1`, at `(1 - weight) a + weight b`.
    RimSplit { from: (usize, usize), weight: f64 },
}

/// The profile triangulation, in the bottom cap's plane.
pub(crate) struct ProfileMesh {
    /// Bottom-cap positions.
    pub(crate) points: Vec<[f64; 3]>,
    /// Profile-plane coordinates.
    pub(crate) planar: Vec<[f64; 2]>,
    /// Counter-clockwise in the profile plane.
    pub(crate) triangles: Vec<[usize; 3]>,
    /// Side faces through each point, with their outward normals.
    pub(crate) sides: Vec<Vec<(usize, [f64; 3])>>,
    pub(crate) origins: Vec<ProfileOrigin>,
}

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Positive when `d` is strictly inside the circumcircle of the counter-clockwise `a, b, c`,
/// beyond a relative rounding bound.
fn inside_circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let rows = [a, b, c].map(|p| {
        let (x, y) = (p[0] - d[0], p[1] - d[1]);
        [x, y, x * x + y * y]
    });
    let det = rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
    let bound = rows[0][0].abs()
        * (rows[1][1] * rows[2][2])
            .abs()
            .max((rows[1][2] * rows[2][1]).abs())
        + rows[0][1].abs()
            * (rows[1][0] * rows[2][2])
                .abs()
                .max((rows[1][2] * rows[2][0]).abs())
        + rows[0][2].abs()
            * (rows[1][0] * rows[2][1])
                .abs()
                .max((rows[1][1] * rows[2][0]).abs());
    det > 1e-10 * bound
}

fn distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (ab, ap) = ([b[0] - a[0], b[1] - a[1]], [p[0] - a[0], p[1] - a[1]]);
    let length = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if length > 0.0 {
        ((ap[0] * ab[0] + ap[1] * ab[1]) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = [a[0] + t * ab[0] - p[0], a[1] + t * ab[1] - p[1]];
    (q[0] * q[0] + q[1] * q[1]).sqrt()
}

/// The published bottom cap: its samples (positions and plane coordinates), triangles, the side
/// faces through each sample, and the side face of each rim edge (keyed by its ascending end
/// points).
pub(crate) struct PublishedProfile {
    pub(crate) points: Vec<[f64; 3]>,
    pub(crate) planar: Vec<[f64; 2]>,
    pub(crate) triangles: Vec<[usize; 3]>,
    pub(crate) sides: Vec<Vec<(usize, [f64; 3])>>,
    pub(crate) rim: BTreeMap<(usize, usize), usize>,
}

/// Meshes the `published` profile at the target `spacing`; `lift` maps plane coordinates back
/// to the bottom cap.
pub(crate) fn mesh_profile(
    published: PublishedProfile,
    spacing: f64,
    lift: impl Fn([f64; 2]) -> [f64; 3],
    work_cap: usize,
) -> Result<ProfileMesh, FinitumError> {
    if !(spacing.is_finite() && spacing > 0.0) {
        return Err(invalid("the interior spacing must be positive and finite"));
    }
    let PublishedProfile {
        points,
        planar,
        triangles,
        sides,
        rim,
    } = published;
    let origins = vec![ProfileOrigin::Sample; points.len()];
    let mut mesh = ProfileMesh {
        points,
        planar,
        triangles: Vec::with_capacity(triangles.len()),
        sides,
        origins,
    };
    for &[a, b, c] in &triangles {
        let area = orient(mesh.planar[a], mesh.planar[b], mesh.planar[c]);
        if area == 0.0 {
            return Err(invalid("the published profile has a zero-area triangle"));
        }
        mesh.triangles
            .push(if area > 0.0 { [a, b, c] } else { [a, c, b] });
    }

    // Split rim edges longer than the spacing into equal parts.
    let mut rim_segments = Vec::new();
    for (&(p, q), &face) in &rim {
        let (u, v) = (mesh.planar[p], mesh.planar[q]);
        let length = ((v[0] - u[0]).powi(2) + (v[1] - u[1]).powi(2)).sqrt();
        // The part count changes a quarter spacing past each whole multiple, so neither an edge
        // of a whole number of spacings (a box side) nor a chord near half a spacing sits on a
        // threshold where a small design step would change the topology.
        let parts = (length / spacing - 0.25).ceil().max(1.0) as usize;
        if mesh.points.len() + parts > work_cap {
            return Err(invalid("the profile exceeds the work cap"));
        }
        let normal = |point: usize| {
            mesh.sides[point]
                .iter()
                .find(|(f, _)| *f == face)
                .map(|(_, n)| *n)
                .ok_or_else(|| invalid("a rim edge's end points are not on its side face"))
        };
        let (np, nq) = (normal(p)?, normal(q)?);
        let mut chain = vec![p];
        for part in 1..parts {
            let weight = part as f64 / parts as f64;
            let m = mesh.points.len();
            let (x, y) = (mesh.points[p], mesh.points[q]);
            mesh.points
                .push([0, 1, 2].map(|k| (1.0 - weight) * x[k] + weight * y[k]));
            mesh.planar.push([
                (1.0 - weight) * u[0] + weight * v[0],
                (1.0 - weight) * u[1] + weight * v[1],
            ]);
            let n = [0, 1, 2].map(|k| (1.0 - weight) * np[k] + weight * nq[k]);
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if length <= 0.0 {
                return Err(invalid("a rim edge turns through a half turn"));
            }
            mesh.sides.push(vec![(face, n.map(|c| c / length))]);
            mesh.origins.push(ProfileOrigin::RimSplit {
                from: (p, q),
                weight,
            });
            chain.push(m);
        }
        chain.push(q);
        if parts > 1 {
            // The one triangle on this boundary edge becomes a fan over the chain.
            let (index, triangle) = mesh
                .triangles
                .iter()
                .enumerate()
                .find(|(_, t)| t.contains(&p) && t.contains(&q))
                .map(|(i, t)| (i, *t))
                .ok_or_else(|| invalid("a rim edge has no profile triangle"))?;
            let at = (0..3)
                .find(|&k| {
                    let (x, y) = (triangle[k], triangle[(k + 1) % 3]);
                    (x, y) == (p, q) || (x, y) == (q, p)
                })
                .ok_or_else(|| invalid("a rim edge is not an edge of its triangle"))?;
            let (x, apex) = (triangle[at], triangle[(at + 2) % 3]);
            let ordered = if x == p {
                chain.clone()
            } else {
                chain.iter().rev().copied().collect()
            };
            mesh.triangles[index] = [ordered[0], ordered[1], apex];
            for pair in ordered[1..].windows(2) {
                mesh.triangles.push([pair[0], pair[1], apex]);
            }
        }
        for pair in chain.windows(2) {
            rim_segments.push((pair[0], pair[1]));
        }
    }

    // Interior grid points, away from the rim.
    let (mut low, mut high) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &mesh.planar {
        for k in 0..2 {
            low[k] = low[k].min(p[k]);
            high[k] = high[k].max(p[k]);
        }
    }
    // A hexagonal lattice: no four lattice points are cocircular, so the Delaunay triangulation
    // of the interior is unique and does not depend on the insertion history (a square grid's
    // cells are cocircular and would keep whichever diagonal the published triangulation led
    // to). The offset keeps lattice rows off axis-aligned rim edges.
    let row_height = 0.5 * 3.0_f64.sqrt() * spacing;
    let offset = 0.5371 * spacing;
    let counts = [
        ((high[0] - low[0] - offset) / spacing).floor().max(-1.0) + 1.0,
        ((high[1] - low[1] - offset) / row_height).floor().max(-1.0) + 1.0,
    ];
    if counts[0] * counts[1] + mesh.points.len() as f64 > work_cap as f64 {
        return Err(invalid("the profile exceeds the work cap"));
    }
    for j in 0..counts[1] as usize {
        for i in 0..counts[0] as usize {
            let shift = if j % 2 == 1 { 0.5 } else { 0.0 };
            let g = [
                low[0] + offset + (i as f64 + shift) * spacing,
                low[1] + offset + j as f64 * row_height,
            ];
            let near_rim = rim_segments.iter().any(|&(a, b)| {
                distance_to_segment(g, mesh.planar[a], mesh.planar[b]) < 0.5 * spacing
            });
            if near_rim {
                continue;
            }
            let containing = mesh.triangles.iter().position(|&[a, b, c]| {
                let (pa, pb, pc) = (mesh.planar[a], mesh.planar[b], mesh.planar[c]);
                let area = orient(pa, pb, pc);
                let tolerance = 1e-9 * area;
                orient(pa, pb, g) > tolerance
                    && orient(pb, pc, g) > tolerance
                    && orient(pc, pa, g) > tolerance
            });
            let Some(index) = containing else {
                continue;
            };
            let m = mesh.points.len();
            mesh.points.push(lift(g));
            mesh.planar.push(g);
            mesh.sides.push(Vec::new());
            mesh.origins.push(ProfileOrigin::Grid);
            let [a, b, c] = mesh.triangles[index];
            mesh.triangles[index] = [a, b, m];
            mesh.triangles.push([b, c, m]);
            mesh.triangles.push([c, a, m]);
        }
    }

    // Lawson flips to the Delaunay triangulation.
    for _pass in 0..10_000 {
        let mut edges = BTreeMap::<(usize, usize), Vec<usize>>::new();
        for (index, t) in mesh.triangles.iter().enumerate() {
            for k in 0..3 {
                let (x, y) = (t[k], t[(k + 1) % 3]);
                edges.entry((x.min(y), x.max(y))).or_default().push(index);
            }
        }
        let mut touched = vec![false; mesh.triangles.len()];
        let mut flipped = false;
        for (&(x, y), incident) in &edges {
            let [t1, t2] = incident[..] else {
                continue;
            };
            if touched[t1] || touched[t2] {
                continue;
            }
            // Orient the edge as (a, b) in t1, so it is (b, a) in t2.
            let first = mesh.triangles[t1];
            let k = (0..3)
                .find(|&k| {
                    let (u, v) = (first[k], first[(k + 1) % 3]);
                    (u, v) == (x, y) || (u, v) == (y, x)
                })
                .ok_or_else(|| invalid("inconsistent profile edge map"))?;
            let (a, b, c) = (first[k], first[(k + 1) % 3], first[(k + 2) % 3]);
            let second = mesh.triangles[t2];
            let d = second
                .into_iter()
                .find(|&v| v != a && v != b)
                .ok_or_else(|| invalid("inconsistent profile edge map"))?;
            let (pa, pb, pc, pd) = (
                mesh.planar[a],
                mesh.planar[b],
                mesh.planar[c],
                mesh.planar[d],
            );
            if inside_circumcircle(pa, pb, pc, pd)
                && orient(pc, pa, pd) > 0.0
                && orient(pd, pb, pc) > 0.0
            {
                mesh.triangles[t1] = [c, a, d];
                mesh.triangles[t2] = [d, b, c];
                touched[t1] = true;
                touched[t2] = true;
                flipped = true;
            }
        }
        if !flipped {
            // Canonical order: each triangle from its smallest index, then sorted, so the
            // mesh does not depend on the flip history.
            for t in &mut mesh.triangles {
                let first = (0..3).min_by_key(|&k| t[k]).unwrap_or(0);
                t.rotate_left(first);
            }
            mesh.triangles.sort_unstable();
            return Ok(mesh);
        }
    }
    Err(invalid("profile edge flips did not terminate"))
}
