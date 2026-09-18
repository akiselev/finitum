//! SC-W2 matching scalar P1 trace elimination, geometry and coverage owned by Finitum.
use crate::{
    AffineConstraint, ConstraintSet, DofId, FacetId, FacetTopology, FinitumError, Mesh, ResultMesh,
    WeightedDof, exterior_facet,
};
use scientia::composition::ScientificSystem;
use scientia::ports::ConnectionSet;
use scientia::{RegionId, SymbolId};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
mod nonmatching;

/// Target trace vertex followed by weighted source trace vertices.
pub type TraceInterpolationRow = (usize, Vec<(usize, f64)>);

#[derive(Clone, Debug, Serialize)]
pub struct ConnectionRealizationPlan {
    relation: ConnectionSet,
    mesh_identities: [String; 2],
    source_identities: [scientia::Digest; 2],
    model_names: [String; 2],
    regions: [RegionId; 2],
    fields: [SymbolId; 2],
    facet_pairs: Vec<[usize; 2]>,
    vertex_pairs: Vec<[usize; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nonmatching: Option<(crate::SurfaceTransfer, Vec<usize>)>,
    tolerance: f64,
    identity: String,
}
fn failure(message: impl Into<String>) -> FinitumError {
    FinitumError::InvalidRealization(format!("CONNECTION_TRACE: {}", message.into()))
}
impl ConnectionRealizationPlan {
    /// Every selected exterior facet must have exactly one geometrically coincident partner
    /// with opposite outward normal. No nearest-neighbour transfer or partial coverage.
    pub fn matching(
        system: &ScientificSystem,
        relation: &ConnectionSet,
        meshes: [&Mesh; 2],
        facets: [&[FacetId]; 2],
        tolerance: f64,
    ) -> Result<Self, FinitumError> {
        relation.validate().map_err(|e| failure(e.to_string()))?;
        system.validate().map_err(|e| failure(e.to_string()))?;
        if !system.connections.contains(relation) {
            return Err(failure("relation is not part of the system"));
        }
        if !tolerance.is_finite()
            || tolerance <= 0.0
            || meshes[0].dimension() != meshes[1].dimension()
            || !(2..=3).contains(&meshes[0].dimension())
        {
            return Err(failure("invalid tolerance or dimension"));
        }
        if facets.iter().any(|s| s.is_empty()) {
            return Err(failure("empty interface coverage"));
        }
        let topology = [
            FacetTopology::from_mesh(meshes[0])?,
            FacetTopology::from_mesh(meshes[1])?,
        ];
        let mut selected = [BTreeSet::new(), BTreeSet::new()];
        for side in 0..2 {
            for facet in facets[side] {
                if !selected[side].insert(facet.0) {
                    return Err(failure("duplicate interface facet"));
                }
                exterior_facet(meshes[side], &topology[side], *facet)?;
            }
        }
        let mut pairs = Vec::new();
        let mut vertex_map = BTreeMap::new();
        let mut reverse = BTreeMap::new();
        let mut used = BTreeSet::new();
        for first in facets[0] {
            let a = &topology[0].facets()[first.0];
            let mut matches = vec![];
            for second in facets[1] {
                let b = &topology[1].facets()[second.0];
                let mut mapping = vec![];
                for vertex in &a.vertices {
                    let candidates = b
                        .vertices
                        .iter()
                        .filter(|other| {
                            meshes[0].vertices()[vertex.0]
                                .iter()
                                .zip(&meshes[1].vertices()[other.0])
                                .all(|(x, y)| (x - y).abs() <= tolerance)
                        })
                        .collect::<Vec<_>>();
                    if candidates.len() != 1 {
                        mapping.clear();
                        break;
                    }
                    mapping.push([vertex.0, candidates[0].0]);
                }
                if mapping.len() == a.vertices.len()
                    && mapping.iter().map(|p| p[1]).collect::<BTreeSet<_>>().len()
                        == b.vertices.len()
                {
                    matches.push((*second, mapping));
                }
            }
            if matches.len() != 1 {
                return Err(failure("facet coverage is incomplete or ambiguous"));
            }
            let (second, mapping) = matches.pop().unwrap();
            if !used.insert(second.0) {
                return Err(failure("facet used by two partners"));
            }
            let ga = exterior_facet(meshes[0], &topology[0], *first)?;
            let gb = exterior_facet(meshes[1], &topology[1], second)?;
            if ga
                .normal
                .iter()
                .zip(&gb.normal)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                > -1.0 + 1e-10
            {
                return Err(failure("interface normals are not opposite"));
            }
            if (ga.measure - gb.measure).abs() > 1e-10 * ga.measure.max(gb.measure) {
                return Err(failure("interface facet measures differ"));
            }
            for [a, b] in mapping {
                if vertex_map
                    .insert(a, b)
                    .is_some_and(|previous| previous != b)
                    || reverse.insert(b, a).is_some_and(|previous| previous != a)
                {
                    return Err(failure("vertex correspondence is not bijective"));
                }
            }
            pairs.push([first.0, second.0]);
        }
        if used != selected[1] {
            return Err(failure("uncovered partner facets"));
        }
        Self::finish(
            system,
            relation,
            meshes,
            pairs,
            vertex_map.into_iter().map(|(a, b)| [a, b]).collect(),
            None,
            tolerance,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn finish(
        system: &ScientificSystem,
        relation: &ConnectionSet,
        meshes: [&Mesh; 2],
        facet_pairs: Vec<[usize; 2]>,
        vertex_pairs: Vec<[usize; 2]>,
        nonmatching: Option<(crate::SurfaceTransfer, Vec<usize>)>,
        tolerance: f64,
    ) -> Result<Self, FinitumError> {
        let mut fields = [SymbolId(0); 2];
        let mut regions = [RegionId(0); 2];
        for side in 0..2 {
            let port = &relation.ports[side];
            let instance = &system.instances[port.instance.index()];
            fields[side] = system
                .variables
                .iter()
                .find(|v| v.id == port.variable && v.owner == port.instance)
                .ok_or_else(|| failure("unknown port field"))?
                .local;
            regions[side] = instance
                .region_map
                .iter()
                .find(|(_, r)| *r == port.region)
                .ok_or_else(|| failure("unknown port region"))?
                .0;
        }
        let mut plan = Self {
            relation: relation.clone(),
            model_names: [
                system.instances[relation.ports[0].instance.index()]
                    .model_name
                    .clone(),
                system.instances[relation.ports[1].instance.index()]
                    .model_name
                    .clone(),
            ],
            source_identities: [
                system.instances[relation.ports[0].instance.index()]
                    .semantic_digest
                    .clone(),
                system.instances[relation.ports[1].instance.index()]
                    .semantic_digest
                    .clone(),
            ],
            mesh_identities: [
                ResultMesh::capture(meshes[0]).identity()?,
                ResultMesh::capture(meshes[1]).identity()?,
            ],
            regions,
            fields,
            facet_pairs,
            vertex_pairs,
            nonmatching,
            tolerance,
            identity: String::new(),
        };
        plan.identity = format!(
            "blake3:{}",
            blake3::hash(&serde_json::to_vec(&plan).expect("finite matching plan")).to_hex()
        );
        Ok(plan)
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn relation(&self) -> &ConnectionSet {
        &self.relation
    }
    pub fn vertex_pairs(&self) -> &[[usize; 2]] {
        &self.vertex_pairs
    }
    pub fn facet_pairs(&self) -> &[[usize; 2]] {
        &self.facet_pairs
    }
    pub(crate) fn admits(
        &self,
        mesh: &Mesh,
        source: &scientia::Digest,
        model: &str,
        region: RegionId,
        field: SymbolId,
        port: &str,
    ) -> bool {
        let Ok(identity) = ResultMesh::capture(mesh).identity() else {
            return false;
        };
        (0..2).any(|side| {
            self.source_identities[side] == *source
                && self.model_names[side] == model
                && self.mesh_identities[side] == identity
                && self.regions[side] == region
                && self.fields[side] == field
                && self.relation.ports[side].name == port
        })
    }
    /// Matching traces become affine DOF equalities in the concatenated group layout.
    /// The transpose restriction sums the outward equations at the retained trace DOF.
    pub fn constraints(
        &self,
        operators: [&crate::ReducedSystemOperator; 2],
    ) -> Result<ConstraintSet, FinitumError> {
        let mut offsets = [0; 2];
        for side in 0..2 {
            let operator = operators[side].operator();
            if operator.plan().system().source_semantic_digest != self.source_identities[side]
                || operator.plan().system().model != self.model_names[side]
            {
                return Err(failure(
                    "operator scientific source differs from matching plan",
                ));
            }
            if ResultMesh::capture(operator.plan().mesh()).identity()? != self.mesh_identities[side]
            {
                return Err(failure("operator mesh differs from matching plan"));
            }
            if operator.connection_row_scale(self.fields[side])?
                != f64::from(self.relation.ports[side].orientation)
            {
                return Err(failure(
                    "operator residual orientation does not match outward balance convention",
                ));
            }
            let block = operator
                .layout()
                .block(self.fields[side])
                .ok_or_else(|| failure("port field missing from layout"))?;
            if block.component_count != 1 || block.extent != operator.plan().mesh().vertices().len()
            {
                return Err(failure("matching elimination requires scalar nodal P1"));
            }
            offsets[side] = block.offset;
            if side == 1 {
                offsets[side] += operators[0].operator().dimension();
            }
            for vertex in self.trace_vertices(side) {
                if operators[side]
                    .constraints()
                    .is_constrained(DofId(block.offset + vertex))
                {
                    return Err(failure(
                        "interface trace overlaps an essential or affine constraint",
                    ));
                }
            }
        }
        ConstraintSet::new(
            operators.iter().map(|o| o.operator().dimension()).sum(),
            self.trace_rows()
                .into_iter()
                .map(|(target, row)| AffineConstraint {
                    target: DofId(offsets[1] + target),
                    dependencies: row
                        .into_iter()
                        .map(|(source, weight)| WeightedDof {
                            dof: DofId(offsets[0] + source),
                            weight,
                        })
                        .collect(),
                    offset: 0.0,
                }),
        )
    }

    /// Compose pairwise matching interfaces in scientific instance order. Shared
    /// edge/corner DOFs form one equivalence class, independent of connection order;
    /// transpose restriction accumulates every participating component's residual.
    pub fn system_constraints(
        connections: &[Self],
        operators: &[&crate::ReducedSystemOperator],
    ) -> Result<ConstraintSet, FinitumError> {
        if connections.is_empty() || operators.is_empty() {
            return Err(failure("empty connected system"));
        }
        let mut offsets = vec![0usize];
        for operator in operators {
            offsets.push(offsets.last().unwrap() + operator.operator().dimension());
        }
        let dimension = *offsets.last().unwrap();
        if connections.iter().any(|c| c.nonmatching.is_some()) {
            if connections.len() != 1 || operators.len() != 2 {
                return Err(failure(
                    "nested nonmatching currently requires one two-component relation",
                ));
            }
            let c = &connections[0];
            let ids = c.relation.ports.each_ref().map(|p| p.instance.index());
            let [Some(a), Some(b)] = ids.map(|i| operators.get(i).copied()) else {
                return Err(failure("connection instance absent from operator list"));
            };
            let pair = c.constraints([a, b])?;
            let split = a.operator().dimension();
            let global = |i: usize| {
                if i < split {
                    offsets[ids[0]] + i
                } else {
                    offsets[ids[1]] + i - split
                }
            };
            return ConstraintSet::new(
                dimension,
                pair.constraints().map(|c| AffineConstraint {
                    target: DofId(global(c.target.0)),
                    dependencies: c
                        .dependencies
                        .iter()
                        .map(|d| WeightedDof {
                            dof: DofId(global(d.dof.0)),
                            weight: d.weight,
                        })
                        .collect(),
                    offset: c.offset,
                }),
            );
        }

        let mut parents: Vec<usize> = (0..dimension).collect();
        fn root(parents: &[usize], mut index: usize) -> usize {
            while parents[index] != index {
                index = parents[index];
            }
            index
        }
        let mut seen = BTreeSet::new();
        for connection in connections {
            if !seen.insert(connection.identity()) {
                return Err(failure("duplicate matching relation"));
            }
            let indices = connection
                .relation
                .ports
                .each_ref()
                .map(|p| p.instance.index());
            let [Some(a), Some(b)] = indices.map(|i| operators.get(i).copied()) else {
                return Err(failure("connection instance absent from operator list"));
            };
            let local = connection.constraints([a, b])?;
            let split = a.operator().dimension();
            let global = |local: usize| {
                if local < split {
                    offsets[indices[0]] + local
                } else {
                    offsets[indices[1]] + local - split
                }
            };
            for constraint in local.constraints() {
                let a = root(&parents, global(constraint.target.0));
                let b = root(&parents, global(constraint.dependencies[0].dof.0));
                parents[a.max(b)] = a.min(b);
            }
        }
        ConstraintSet::new(
            dimension,
            (0..dimension).filter_map(|index| {
                let representative = root(&parents, index);
                (representative != index).then(|| AffineConstraint {
                    target: DofId(index),
                    dependencies: vec![WeightedDof {
                        dof: DofId(representative),
                        weight: 1.0,
                    }],
                    offset: 0.0,
                })
            }),
        )
    }
}
