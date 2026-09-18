//! Bounded SC-W3: planar nested triangular P1 traces with complete area coverage.
use super::*;
impl ConnectionRealizationPlan {
    /// Endpoint zero is the coarse/master trace. Endpoint one must be a true
    /// triangulated refinement with opposite outward normal and complete coverage.
    /// No extrapolation, curved surface, crossing partition or nearest-neighbor fallback.
    pub fn nested_p1(
        system: &ScientificSystem,
        relation: &ConnectionSet,
        meshes: [&Mesh; 2],
        facets: [&[FacetId]; 2],
        tolerance: f64,
    ) -> Result<Self, FinitumError> {
        system.validate().map_err(|e| failure(e.to_string()))?;
        relation.validate().map_err(|e| failure(e.to_string()))?;
        if !system.connections.contains(relation)
            || meshes.iter().any(|m| m.dimension() != 3)
            || facets.iter().any(|f| f.is_empty())
            || !tolerance.is_finite()
            || tolerance <= 0.0
        {
            return Err(failure(
                "nested P1 needs an admitted relation and nonempty 3-D traces",
            ));
        }
        let topology = [
            FacetTopology::from_mesh(meshes[0])?,
            FacetTopology::from_mesh(meshes[1])?,
        ];
        let mut triangles: [Vec<[usize; 3]>; 2] = [vec![], vec![]];
        let mut geometry = [vec![], vec![]];
        for side in 0..2 {
            let mut seen = BTreeSet::new();
            for id in facets[side] {
                if !seen.insert(id.0) {
                    return Err(failure("duplicate interface facet"));
                }
                let vertices = &topology[side]
                    .facets()
                    .get(id.0)
                    .ok_or_else(|| failure("facet out of range"))?
                    .vertices;
                triangles[side].push(
                    vertices
                        .iter()
                        .map(|v| v.0)
                        .collect::<Vec<_>>()
                        .try_into()
                        .map_err(|_| failure("triangular traces required"))?,
                );
                geometry[side].push(exterior_facet(meshes[side], &topology[side], *id)?);
            }
        }
        let normal = &geometry[0][0].normal;
        let origin = &meshes[0].vertices()[triangles[0][0][0]];
        for side in 0..2 {
            for (triangle, geo) in triangles[side].iter().zip(&geometry[side]) {
                let dot: f64 = normal.iter().zip(&geo.normal).map(|(a, b)| a * b).sum();
                if (dot - if side == 0 { 1.0 } else { -1.0 }).abs() > 1e-10 {
                    return Err(failure("planar opposite normals required"));
                }
                for vertex in triangle {
                    let distance: f64 = meshes[side].vertices()[*vertex]
                        .iter()
                        .zip(origin)
                        .zip(normal)
                        .map(|((x, o), n)| (x - o) * n)
                        .sum();
                    if distance.abs() > tolerance {
                        return Err(failure("interface planes do not coincide"));
                    }
                }
            }
        }
        // Project along the dominant normal coordinate; check both triangulations
        // for positive-area overlaps so area equality cannot hide a hole/overlap pair.
        let drop_axis = (0..3)
            .max_by(|a, b| normal[*a].abs().total_cmp(&normal[*b].abs()))
            .unwrap();
        let axes: Vec<_> = (0..3).filter(|i| *i != drop_axis).collect();
        for side in 0..2 {
            let projected: Vec<[[f64; 2]; 3]> = triangles[side]
                .iter()
                .map(|t| {
                    t.map(|v| {
                        [
                            meshes[side].vertices()[v][axes[0]],
                            meshes[side].vertices()[v][axes[1]],
                        ]
                    })
                })
                .collect();
            for a in 0..projected.len() {
                for b in 0..a {
                    if intersection_area(projected[a], projected[b])
                        > 1e-10
                            * geometry[side][a].measure.min(geometry[side][b].measure)
                            * normal[drop_axis].abs()
                    {
                        return Err(failure("overlapping interface triangles"));
                    }
                }
            }
        }
        let source: Vec<[f64; 3]> = meshes[0]
            .vertices()
            .iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let mut covered = vec![0.0; triangles[0].len()];
        for (fine, geo) in triangles[1].iter().zip(&geometry[1]) {
            let points = fine
                .iter()
                .map(|v| {
                    let p = &meshes[1].vertices()[*v];
                    [p[0], p[1], p[2]]
                })
                .collect::<Vec<_>>();
            let parents: Vec<_> = triangles[0]
                .iter()
                .enumerate()
                .filter_map(|(i, t)| {
                    crate::SurfaceTransfer::p1(source.clone(), vec![*t], points.clone(), tolerance)
                        .ok()
                        .map(|_| i)
                })
                .collect();
            if parents.len() != 1 {
                return Err(failure("fine facet must lie wholly in one coarse facet"));
            }
            covered[parents[0]] += geo.measure;
        }
        if covered
            .iter()
            .zip(&geometry[0])
            .any(|(a, b)| (a - b.measure).abs() > 1e-10 * b.measure)
        {
            return Err(failure("nonmatching interface coverage is incomplete"));
        }
        let targets: Vec<_> = triangles[1]
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let points = targets
            .iter()
            .map(|v| {
                let p = &meshes[1].vertices()[*v];
                [p[0], p[1], p[2]]
            })
            .collect();
        let transfer = crate::SurfaceTransfer::p1(source, triangles[0].clone(), points, tolerance)?;
        Self::finish(
            system,
            relation,
            meshes,
            vec![],
            vec![],
            Some((transfer, targets)),
            tolerance,
        )
    }
    pub fn is_nonmatching(&self) -> bool {
        self.nonmatching.is_some()
    }
    /// Each target vertex and its weighted source vertices; scalar P1 primal map.
    pub fn trace_rows(&self) -> Vec<TraceInterpolationRow> {
        if let Some((transfer, targets)) = &self.nonmatching {
            targets
                .iter()
                .copied()
                .zip(transfer.rows().iter().cloned())
                .collect()
        } else {
            self.vertex_pairs
                .iter()
                .map(|p| (p[1], vec![(p[0], 1.0)]))
                .collect()
        }
    }
    pub fn trace_vertices(&self, side: usize) -> Vec<usize> {
        self.trace_rows()
            .into_iter()
            .flat_map(|(target, row)| {
                if side == 0 {
                    row.into_iter().map(|(i, _)| i).collect()
                } else {
                    vec![target]
                }
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    /// Continuity and dual residual balance in the actual trace spaces.
    pub fn trace_defects(
        &self,
        operators: [&crate::ReducedSystemOperator; 2],
        values: [&[f64]; 2],
        residuals: [&[f64]; 2],
    ) -> Result<(f64, f64), FinitumError> {
        self.constraints(operators)?;
        for side in 0..2 {
            let n = operators[side].operator().dimension();
            if values[side].len() != n
                || residuals[side].len() != n
                || values[side]
                    .iter()
                    .chain(residuals[side])
                    .any(|x| !x.is_finite())
            {
                return Err(failure("invalid trace sample extents or values"));
            }
        }
        let offsets = [0, 1].map(|side| {
            operators[side]
                .operator()
                .layout()
                .block(self.fields[side])
                .unwrap()
                .offset
        });
        let mut balance: BTreeMap<usize, f64> = self
            .trace_vertices(0)
            .into_iter()
            .map(|v| (v, residuals[0][offsets[0] + v]))
            .collect();
        let mut continuity = 0.0_f64;
        for (target, row) in self.trace_rows() {
            let expected: f64 = row
                .iter()
                .map(|(source, weight)| weight * values[0][offsets[0] + source])
                .sum();
            continuity = continuity.max((values[1][offsets[1] + target] - expected).abs());
            for (source, weight) in row {
                *balance.get_mut(&source).unwrap() += weight * residuals[1][offsets[1] + target];
            }
        }
        if !continuity.is_finite() || balance.values().any(|v| !v.is_finite()) {
            return Err(failure("nonfinite trace defects"));
        }
        Ok((
            continuity,
            balance.values().map(|x| x.abs()).fold(0.0, f64::max),
        ))
    }
}
fn intersection_area(a: [[f64; 2]; 3], mut b: [[f64; 2]; 3]) -> f64 {
    let cross = |a: [f64; 2], b: [f64; 2], p: [f64; 2]| {
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    };
    if cross(b[0], b[1], b[2]) < 0.0 {
        b.swap(1, 2);
    }
    let mut polygon = a.to_vec();
    for i in 0..3 {
        let mut clipped = vec![];
        for j in 0..polygon.len() {
            let p = polygon[j];
            let q = polygon[(j + 1) % polygon.len()];
            let dp = cross(b[i], b[(i + 1) % 3], p);
            let dq = cross(b[i], b[(i + 1) % 3], q);
            if dp >= 0.0 {
                clipped.push(p);
            }
            if (dp >= 0.0) != (dq >= 0.0) {
                let t = dp / (dp - dq);
                clipped.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        polygon = clipped;
    }
    if polygon.len() < 3 {
        return 0.0;
    }
    (1..polygon.len() - 1)
        .map(|i| cross(polygon[0], polygon[i], polygon[i + 1]).abs() * 0.5)
        .sum()
}
