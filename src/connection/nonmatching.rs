//! Bounded SC-W3: planar triangular P1 traces. `nested_p1` admits a coarse trace whose every
//! fine facet nests in one coarse facet; `planar_p1` admits two arbitrary triangulations of the
//! SAME planar polygon, proves their coverage through exact pairwise triangle intersections and
//! builds mortar-type (dual-basis) weighted constraints from the intersection integrals.
use super::*;
use crate::ExteriorFacet;

/// Relative tolerance of every planar coverage comparison: `nested_p1`'s covered area per
/// coarse facet, `planar_p1`'s covered area per triangle on both sides and its aggregate
/// `Σ|T_a ∩ T_b|` against each side's area, and the relative part of the same-side overlap floor.
const COVERAGE_RELATIVE_TOLERANCE: f64 = 1e-10;

/// Tolerance of the mortar rows' constant reproduction (each row sum against one). Complete
/// coverage to [`COVERAGE_RELATIVE_TOLERANCE`] with a dual basis bounded by 3 in magnitude and
/// exact degree-2 quadrature bounds the defect below this; a larger one is an integration defect.
const CONSTANT_REPRODUCTION_TOLERANCE: f64 = 1e-9;

/// The coverage proof a [`ConnectionRealizationPlan::planar_p1`] plan carries: both trace
/// polygons' areas, the total of the exact pairwise triangle-intersection areas that must equal
/// both, the worst relative defect of every comparison that held, the floor below which an
/// intersection counts as empty (and a triangle as degenerate), and how many pairs were kept or
/// classified as sub-floor slivers. `nested_p1` and `matching` plans carry none.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct PlanarCoverageReport {
    /// Physical area of each side's trace polygon (sum of its admitted triangles).
    pub area: [f64; 2],
    /// `Σ_{a,b} |T_a ∩ T_b|` over the opposite-side pairs above the area floor.
    pub intersection_total: f64,
    /// Worst relative defect over every per-triangle (both sides) and aggregate comparison.
    pub worst_relative_defect: f64,
    /// Worst `|Σ_k w_ik − 1|` over the mortar rows (constant reproduction).
    pub worst_constant_defect: f64,
    /// The relative tolerance every coverage comparison held to.
    pub relative_tolerance: f64,
    /// `tolerance × diameter`: an intersection at or below it is empty, a triangle degenerate.
    pub area_floor: f64,
    /// Opposite-side pairs whose intersection exceeds the floor (each carries integrals).
    pub pairs: usize,
    /// Opposite-side pairs with a positive but sub-floor intersection, classified as empty.
    pub slivers_classified: usize,
}

/// Both sides' admitted interface triangles, their exterior geometry, and their in-plane
/// coordinates in one orthonormal frame of the shared plane (so in-plane areas are physical).
struct PlanarTraces {
    triangles: [Vec<[usize; 3]>; 2],
    geometry: [Vec<ExteriorFacet>; 2],
    projected: [Vec<[[f64; 2]; 3]>; 2],
}

impl PlanarTraces {
    /// Shared admission of `nested_p1` and `planar_p1`: an admitted relation, nonempty 3-D
    /// traces of distinct exterior triangular facets, every vertex on the plane of the first
    /// master facet within `tolerance`, master normals equal and target normals opposite to it.
    fn admit(
        label: &str,
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
            return Err(failure(format!(
                "{label} needs an admitted relation and nonempty 3-D traces"
            )));
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
        // Orthonormal in-plane frame: the unit axis least aligned with the normal, made
        // orthogonal to it, and the normal's cross product with that. In-plane areas and the
        // integrals below are then physical, not scaled by a dropped coordinate.
        let n = [normal[0], normal[1], normal[2]];
        let weakest = (0..3)
            .min_by(|a, b| n[*a].abs().total_cmp(&n[*b].abs()))
            .unwrap();
        let mut e1 = [0.0; 3];
        e1[weakest] = 1.0;
        let along = e1[weakest] * n[weakest];
        for i in 0..3 {
            e1[i] -= along * n[i];
        }
        let length = e1.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut e1 {
            *x /= length;
        }
        let e2 = [
            n[1] * e1[2] - n[2] * e1[1],
            n[2] * e1[0] - n[0] * e1[2],
            n[0] * e1[1] - n[1] * e1[0],
        ];
        let project = |side: usize, vertex: usize| -> [f64; 2] {
            let p = &meshes[side].vertices()[vertex];
            let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
            [
                d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2],
                d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2],
            ]
        };
        let projected = [0, 1].map(|side| {
            triangles[side]
                .iter()
                .map(|t| t.map(|v| project(side, v)))
                .collect::<Vec<_>>()
        });
        Ok(Self {
            triangles,
            geometry,
            projected,
        })
    }

    /// No two triangles of one side overlap: a same-side pair whose in-plane intersection area
    /// exceeds `floor(measure_a, measure_b)` refuses, so area equality cannot hide a hole/overlap
    /// pair.
    fn refuse_same_side_overlap(
        &self,
        floor: impl Fn(f64, f64) -> f64,
    ) -> Result<(), FinitumError> {
        for side in 0..2 {
            let projected = &self.projected[side];
            for a in 0..projected.len() {
                for b in 0..a {
                    let measures = (
                        self.geometry[side][a].measure,
                        self.geometry[side][b].measure,
                    );
                    if intersection_area(projected[a], projected[b]) > floor(measures.0, measures.1)
                    {
                        return Err(failure("overlapping interface triangles"));
                    }
                }
            }
        }
        Ok(())
    }

    fn points(&self, meshes: [&Mesh; 2], side: usize, vertices: &[usize]) -> Vec<[f64; 3]> {
        vertices
            .iter()
            .map(|v| {
                let p = &meshes[side].vertices()[*v];
                [p[0], p[1], p[2]]
            })
            .collect()
    }

    /// Every distinct target-side trace vertex, ascending.
    fn targets(&self) -> Vec<usize> {
        self.triangles[1]
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// The exact pairwise-intersection coverage proof and the dual-basis mortar rows it yields.
struct PlanarMortar {
    report: PlanarCoverageReport,
    targets: Vec<usize>,
    rows: Vec<Vec<(usize, f64)>>,
}

impl PlanarMortar {
    fn prove(traces: &PlanarTraces, tolerance: f64) -> Result<Self, FinitumError> {
        let diameter = traces
            .projected
            .iter()
            .flatten()
            .flat_map(|t| (0..3).map(move |i| distance(t[i], t[(i + 1) % 3])))
            .fold(0.0, f64::max);
        let area_floor = tolerance * diameter;
        let areas = [0, 1].map(|side| {
            traces.projected[side]
                .iter()
                .map(|t| polygon_area(t))
                .collect::<Vec<_>>()
        });
        if areas
            .iter()
            .flatten()
            .any(|area| !area.is_finite() || *area <= area_floor)
        {
            return Err(failure(
                "degenerate interface triangle below the area floor",
            ));
        }
        traces.refuse_same_side_overlap(|a, b| {
            (COVERAGE_RELATIVE_TOLERANCE * a.min(b)).max(area_floor)
        })?;
        let targets = traces.targets();
        let position: BTreeMap<usize, usize> = targets
            .iter()
            .enumerate()
            .map(|(index, vertex)| (*vertex, index))
            .collect();
        let mut patch = vec![0.0; targets.len()];
        for (triangle, area) in traces.triangles[1].iter().zip(&areas[1]) {
            for vertex in triangle {
                patch[position[vertex]] += area;
            }
        }
        let mut covered = [vec![0.0; areas[0].len()], vec![0.0; areas[1].len()]];
        let mut moments: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); targets.len()];
        let mut pairs = 0;
        let mut slivers_classified = 0;
        let mut intersection_total = 0.0;
        for (a, coarse) in traces.projected[0].iter().enumerate() {
            for (b, fine) in traces.projected[1].iter().enumerate() {
                let polygon = clip(*fine, *coarse);
                let area = polygon_area(&polygon);
                if area <= area_floor {
                    slivers_classified += usize::from(area > 0.0);
                    continue;
                }
                pairs += 1;
                intersection_total += area;
                covered[0][a] += area;
                covered[1][b] += area;
                // Exact integrals of (dual target basis) x (master P1 basis) over the convex
                // intersection: fan sub-triangles, three edge midpoints each (exact to degree 2).
                for window in 1..polygon.len() - 1 {
                    let corners = [polygon[0], polygon[window], polygon[window + 1]];
                    let weight = polygon_area(&corners) / 3.0;
                    for edge in 0..3 {
                        let point = midpoint(corners[edge], corners[(edge + 1) % 3]);
                        let target_bary = barycentric(*fine, point);
                        let master_bary = barycentric(*coarse, point);
                        for (local, vertex) in traces.triangles[1][b].iter().enumerate() {
                            let dual = 4.0 * target_bary[local] - 1.0;
                            let row = &mut moments[position[vertex]];
                            for (master_local, master) in traces.triangles[0][a].iter().enumerate()
                            {
                                *row.entry(*master).or_insert(0.0) +=
                                    weight * dual * master_bary[master_local];
                            }
                        }
                    }
                }
            }
        }
        let mut worst_relative_defect = 0.0_f64;
        for side in 0..2 {
            for (index, (covered, area)) in covered[side].iter().zip(&areas[side]).enumerate() {
                let defect = (covered - area).abs() / area;
                worst_relative_defect = worst_relative_defect.max(defect);
                if !defect.is_finite() || defect > COVERAGE_RELATIVE_TOLERANCE {
                    return Err(failure(format!(
                        "interface triangulations do not cover the same polygon: side {side} \
                         triangle {index} is covered on {covered:e} of its {area:e}"
                    )));
                }
            }
        }
        let area = [0, 1].map(|side| areas[side].iter().sum::<f64>());
        for total in area {
            let defect = (intersection_total - total).abs() / total;
            worst_relative_defect = worst_relative_defect.max(defect);
            if !defect.is_finite() || defect > COVERAGE_RELATIVE_TOLERANCE {
                return Err(failure(format!(
                    "interface triangulations do not cover the same polygon: intersections \
                     total {intersection_total:e} against a side of {total:e}"
                )));
            }
        }
        // Mortar row of target vertex i: w_ik = ∫ ψ_i φ_k / ∫ ψ_i φ_i with ψ_i = 4λ_i − 1 the
        // dual basis (∫_T ψ_i φ_j = δ_ij |T|/3), so ∫ ψ_i φ_i = |patch_i| / 3. Exactly zero
        // moments (a master whose basis vanishes on the whole patch) carry no dependency.
        let mut worst_constant_defect = 0.0_f64;
        let mut rows = Vec::with_capacity(targets.len());
        for (index, moments) in moments.into_iter().enumerate() {
            let scale = 3.0 / patch[index];
            let row: Vec<(usize, f64)> = moments
                .into_iter()
                .filter(|(_, moment)| *moment != 0.0)
                .map(|(master, moment)| (master, moment * scale))
                .collect();
            let sum: f64 = row.iter().map(|(_, w)| w).sum();
            worst_constant_defect = worst_constant_defect.max((sum - 1.0).abs());
            rows.push(row);
        }
        if !worst_constant_defect.is_finite()
            || worst_constant_defect > CONSTANT_REPRODUCTION_TOLERANCE
        {
            return Err(failure(format!(
                "mortar rows do not reproduce constants: worst row-sum defect \
                 {worst_constant_defect:e}"
            )));
        }
        Ok(Self {
            report: PlanarCoverageReport {
                area,
                intersection_total,
                worst_relative_defect,
                worst_constant_defect,
                relative_tolerance: COVERAGE_RELATIVE_TOLERANCE,
                area_floor,
                pairs,
                slivers_classified,
            },
            targets,
            rows,
        })
    }
}

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
        let traces = PlanarTraces::admit("nested P1", system, relation, meshes, facets, tolerance)?;
        traces.refuse_same_side_overlap(|a, b| COVERAGE_RELATIVE_TOLERANCE * a.min(b))?;
        let source: Vec<[f64; 3]> = meshes[0]
            .vertices()
            .iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let mut covered = vec![0.0; traces.triangles[0].len()];
        for (fine, geo) in traces.triangles[1].iter().zip(&traces.geometry[1]) {
            let points = traces.points(meshes, 1, fine);
            let parents: Vec<_> = traces.triangles[0]
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
            .zip(&traces.geometry[0])
            .any(|(a, b)| (a - b.measure).abs() > COVERAGE_RELATIVE_TOLERANCE * b.measure)
        {
            return Err(failure("nonmatching interface coverage is incomplete"));
        }
        let targets = traces.targets();
        let points = traces.points(meshes, 1, &targets);
        let transfer =
            crate::SurfaceTransfer::p1(source, traces.triangles[0].clone(), points, tolerance)?;
        Self::finish(
            system,
            relation,
            meshes,
            vec![],
            vec![],
            Some((transfer, targets)),
            None,
            tolerance,
        )
    }

    /// Planar NON-nested scalar P1 traces: endpoint zero is the master trace whose vertex values
    /// the constraints depend on, endpoint one the target trace whose every vertex value is
    /// eliminated. Unlike [`Self::nested_p1`], neither triangulation needs to refine the other;
    /// the two sides must be two planar triangulations of the SAME polygon, which is proved, not
    /// assumed:
    ///
    /// - the shared admission of `nested_p1` (admitted relation, exterior triangular facets,
    ///   coplanar within `tolerance`, opposite outward normals);
    /// - no triangle is degenerate: its area exceeds the area floor `tolerance × diameter`
    ///   (`diameter` the longest admitted edge); an opposite-side intersection at or below that
    ///   floor is classified as empty (roundoff slivers of shared edges and vertices);
    /// - no two same-side triangles overlap above `max(1e-10 · min area, floor)`;
    /// - exact pairwise intersections (Sutherland–Hodgman clip of each target triangle against
    ///   each master triangle, orientation-independent) cover every triangle of BOTH sides to
    ///   relative `1e-10`, hence `Σ_{a,b} |T_a ∩ T_b| = |Γ_master| = |Γ_target|` to the same
    ///   tolerance; the proof is carried as [`Self::planar_coverage`].
    ///
    /// The constraints are the mortar projection of the master P1 trace onto the target P1 trace
    /// tested against the **biorthogonal dual basis** `ψ_i = 4λ_i − 1` per target triangle
    /// (`∫_T ψ_i φ_j = δ_ij |T|/3`): target value `u_i = Σ_k w_ik u_k` with `w_ik = ∫ ψ_i φ_k /
    /// ∫ ψ_i φ_i`, every integral exact over the intersection polygons (degree-2 edge-midpoint
    /// rule on their fan triangles). Chosen over the standard-basis `M_ff⁻¹ M_fc` because the
    /// rows stay local (a target vertex depends only on the master vertices whose triangles meet
    /// its patch), need no dense inverse, and are still a projection: a master trace already in
    /// the target space -- every affine trace, and every trace of a nested pair -- is reproduced
    /// exactly, so on a pair `nested_p1` admits the weights coincide with its interpolation
    /// weights. Every row sums to one (constant reproduction, hence the transposed dual load is
    /// conserved); weights may be negative (no maximum principle, as for any L2-type
    /// projection). The result is the same `ConstraintSet` shape `nested_p1` emits (target,
    /// weighted dependencies, zero offset), so [`Self::constraints`],
    /// [`Self::system_linearized_diagonal`], [`Self::trace_defects`] and Krasis's generic
    /// composition apply unchanged.
    pub fn planar_p1(
        system: &ScientificSystem,
        relation: &ConnectionSet,
        meshes: [&Mesh; 2],
        facets: [&[FacetId]; 2],
        tolerance: f64,
    ) -> Result<Self, FinitumError> {
        let traces = PlanarTraces::admit("planar P1", system, relation, meshes, facets, tolerance)?;
        let mortar = PlanarMortar::prove(&traces, tolerance)?;
        let source: Vec<[f64; 3]> = meshes[0]
            .vertices()
            .iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let points = traces.points(meshes, 1, &mortar.targets);
        let transfer = crate::SurfaceTransfer::from_rows(
            source,
            traces.triangles[0].clone(),
            points,
            mortar.rows,
            tolerance,
        )?;
        Self::finish(
            system,
            relation,
            meshes,
            vec![],
            vec![],
            Some((transfer, mortar.targets)),
            Some(mortar.report),
            tolerance,
        )
    }
    pub fn is_nonmatching(&self) -> bool {
        self.nonmatching.is_some()
    }
    /// The coverage proof of a [`Self::planar_p1`] plan; `None` for `matching` and `nested_p1`.
    pub fn planar_coverage(&self) -> Option<&PlanarCoverageReport> {
        self.planar_coverage.as_ref()
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

/// Twice the signed area of `(a, b, c)`; positive for a counter-clockwise turn.
fn signed_double_area(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
}

/// The triangle turned counter-clockwise, or `None` when it has no area.
fn counter_clockwise(mut triangle: [[f64; 2]; 3]) -> Option<[[f64; 2]; 3]> {
    let orientation = signed_double_area(triangle[0], triangle[1], triangle[2]);
    if !orientation.is_finite() || orientation == 0.0 {
        return None;
    }
    if orientation < 0.0 {
        triangle.swap(1, 2);
    }
    Some(triangle)
}

/// Sutherland–Hodgman clip of `subject` against the three half-planes of `window`, independent
/// of either triangle's orientation: the convex intersection polygon, counter-clockwise, or
/// empty when fewer than three points survive (or either triangle has no area).
fn clip(subject: [[f64; 2]; 3], window: [[f64; 2]; 3]) -> Vec<[f64; 2]> {
    let (Some(subject), Some(window)) = (counter_clockwise(subject), counter_clockwise(window))
    else {
        return vec![];
    };
    let mut polygon = subject.to_vec();
    for i in 0..3 {
        let (from, to) = (window[i], window[(i + 1) % 3]);
        let mut clipped = Vec::with_capacity(polygon.len() + 1);
        for j in 0..polygon.len() {
            let p = polygon[j];
            let q = polygon[(j + 1) % polygon.len()];
            let dp = signed_double_area(from, to, p);
            let dq = signed_double_area(from, to, q);
            if dp >= 0.0 {
                clipped.push(p);
            }
            if (dp >= 0.0) != (dq >= 0.0) {
                let t = dp / (dp - dq);
                clipped.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        polygon = clipped;
        if polygon.len() < 3 {
            return vec![];
        }
    }
    polygon
}

/// Shoelace area of a simple polygon, orientation-independent.
fn polygon_area(polygon: &[[f64; 2]]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    (1..polygon.len() - 1)
        .map(|i| signed_double_area(polygon[0], polygon[i], polygon[i + 1]))
        .sum::<f64>()
        .abs()
        * 0.5
}

fn intersection_area(a: [[f64; 2]; 3], b: [[f64; 2]; 3]) -> f64 {
    polygon_area(&clip(a, b))
}

/// Barycentric coordinates of `point` in `triangle` (any orientation).
fn barycentric(triangle: [[f64; 2]; 3], point: [f64; 2]) -> [f64; 3] {
    let total = signed_double_area(triangle[0], triangle[1], triangle[2]);
    [
        signed_double_area(point, triangle[1], triangle[2]) / total,
        signed_double_area(triangle[0], point, triangle[2]) / total,
        signed_double_area(triangle[0], triangle[1], point) / total,
    ]
}
