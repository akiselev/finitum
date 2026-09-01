use crate::FinitumError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DofId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementRestriction {
    pub dofs: Vec<DofId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DofMap {
    dof_count: usize,
    restrictions: Vec<ElementRestriction>,
}

impl DofMap {
    pub fn new(
        dof_count: usize,
        restrictions: Vec<ElementRestriction>,
    ) -> Result<Self, FinitumError> {
        for (restriction_index, restriction) in restrictions.iter().enumerate() {
            if restriction.dofs.is_empty() {
                return Err(FinitumError::EmptyRestriction {
                    restriction: restriction_index,
                });
            }
            let mut distinct = BTreeSet::new();
            for dof in &restriction.dofs {
                if dof.0 >= dof_count {
                    return Err(FinitumError::MissingDof(dof.0));
                }
                if !distinct.insert(*dof) {
                    return Err(FinitumError::DuplicateRestrictionDof {
                        restriction: restriction_index,
                        dof: dof.0,
                    });
                }
            }
        }
        Ok(Self {
            dof_count,
            restrictions,
        })
    }

    pub fn dof_count(&self) -> usize {
        self.dof_count
    }

    pub fn restrictions(&self) -> &[ElementRestriction] {
        &self.restrictions
    }
}

/// Canonical vertex-major vector nodal DOF map for P1 simplices.
///
/// Node `i` of cell `c` owns `components` consecutive DOFs starting at
/// `i * components`, so local restriction order matches the vertex-major
/// layout the generated kernels and basis evaluation read.
pub fn vector_nodal_dof_map(
    mesh: &crate::Mesh,
    components: usize,
) -> Result<DofMap, crate::FinitumError> {
    if components == 0 {
        return Err(crate::FinitumError::InvalidRealization(
            "vector DOF maps require at least one component".into(),
        ));
    }
    let vertices = mesh.vertices().len();
    let dof_count = vertices
        .checked_mul(components)
        .ok_or_else(|| crate::FinitumError::InvalidRealization("dof overflow".into()))?;
    let restrictions = mesh
        .cells()
        .iter()
        .map(|cell| {
            let mut dofs = Vec::with_capacity(cell.vertices.len() * components);
            for vertex in &cell.vertices {
                for component in 0..components {
                    dofs.push(DofId(vertex.0 * components + component));
                }
            }
            ElementRestriction { dofs }
        })
        .collect();
    DofMap::new(dof_count, restrictions)
}

/// Canonical vertex-then-edge-major vector nodal DOF map for P2 simplices.
///
/// Node ordering is the mesh's vertices `0..vertex_count`, followed by its unique edges (from
/// `crate::topology::mesh_edges`'s canonical, deterministic order) at
/// `vertex_count..vertex_count + edge_count`. Node `i` owns `components` consecutive DOFs
/// starting at `i * components`, matching [`vector_nodal_dof_map`]'s convention. Each cell's
/// local restriction lists its `dimension + 1` vertex nodes in cell-local vertex order, then its
/// `(dimension + 1) * dimension / 2` edge nodes in the nested `(left, right)` pair order with
/// `left < right`, matching [`crate::element::PreparedElement::quadratic_simplex`]'s basis
/// ordering exactly, so a restriction's local DOF `k` always corresponds to basis function `k`.
pub fn quadratic_simplex_dof_map(
    mesh: &crate::Mesh,
    components: usize,
) -> Result<DofMap, crate::FinitumError> {
    if components == 0 {
        return Err(crate::FinitumError::InvalidRealization(
            "quadratic simplex DOF maps require at least one component".into(),
        ));
    }
    let edges = crate::topology::mesh_edges(mesh);
    let edge_index: BTreeMap<Vec<usize>, usize> = edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge.clone(), index))
        .collect();
    let vertex_count = mesh.vertices().len();
    let node_count = vertex_count
        .checked_add(edges.len())
        .ok_or_else(|| crate::FinitumError::InvalidRealization("node count overflow".into()))?;
    let dof_count = node_count
        .checked_mul(components)
        .ok_or_else(|| crate::FinitumError::InvalidRealization("dof overflow".into()))?;
    let restrictions = mesh
        .cells()
        .iter()
        .map(|cell| {
            let mut dofs = Vec::with_capacity(
                (cell.vertices.len() + cell.vertices.len() * (cell.vertices.len() - 1) / 2)
                    * components,
            );
            for vertex in &cell.vertices {
                for component in 0..components {
                    dofs.push(DofId(vertex.0 * components + component));
                }
            }
            for left in 0..cell.vertices.len() {
                for right in left + 1..cell.vertices.len() {
                    let key = sorted_pair(cell.vertices[left].0, cell.vertices[right].0);
                    let node = vertex_count + edge_index[&key];
                    for component in 0..components {
                        dofs.push(DofId(node * components + component));
                    }
                }
            }
            ElementRestriction { dofs }
        })
        .collect();
    DofMap::new(dof_count, restrictions)
}

/// Physical coordinates of every P2 node in [`quadratic_simplex_dof_map`]'s node order: mesh
/// vertices, then edge midpoints in `crate::topology::mesh_edges`'s canonical order.
pub fn quadratic_simplex_node_points(mesh: &crate::Mesh) -> Vec<Vec<f64>> {
    let mut points = mesh.vertices().to_vec();
    for edge in crate::topology::mesh_edges(mesh) {
        let a = &mesh.vertices()[edge[0]];
        let b = &mesh.vertices()[edge[1]];
        points.push(
            a.iter()
                .zip(b)
                .map(|(left, right)| 0.5 * (left + right))
                .collect(),
        );
    }
    points
}

/// One scalar DOF per mesh cell (P0/L2(order=0) piecewise-constant field).
///
/// Cell `c` owns exactly `DofId(c)`, matching this crate's convention (used throughout
/// `crate::system`) that a P0 field's global vector is indexed identically to
/// `crate::mesh::CellId`.
pub(crate) fn cell_constant_dof_map(mesh: &crate::Mesh) -> Result<DofMap, crate::FinitumError> {
    let dof_count = mesh.cells().len();
    let restrictions = (0..dof_count)
        .map(|cell| ElementRestriction {
            dofs: vec![DofId(cell)],
        })
        .collect();
    DofMap::new(dof_count, restrictions)
}

fn sorted_pair(left: usize, right: usize) -> Vec<usize> {
    if left < right {
        vec![left, right]
    } else {
        vec![right, left]
    }
}
