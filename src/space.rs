use crate::FinitumError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DofId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementRestriction {
    pub dofs: Vec<DofId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DofMap {
    dof_count: usize,
    restrictions: Vec<ElementRestriction>,
}

impl DofMap {
    pub fn new(
        dof_count: usize,
        restrictions: Vec<ElementRestriction>,
    ) -> Result<Self, FinitumError> {
        for restriction in &restrictions {
            for dof in &restriction.dofs {
                if dof.0 >= dof_count {
                    return Err(FinitumError::MissingDof(dof.0));
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
