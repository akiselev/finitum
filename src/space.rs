use crate::FinitumError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
