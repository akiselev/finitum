use crate::{
    AffineConstraint, CellId, DofId, DofMap, FinitumError, Mesh, PreparedElement, WeightedDof,
};

/// Per-cell variable-order basis tables for one-dimensional segments.
///
/// This validates local tables and restrictions only. It does not provide mesh refinement,
/// AMR topology, or integration with [`crate::RealizationPlan`].
#[derive(Clone, Debug, PartialEq)]
pub struct VariableOrderSegmentElements {
    orders: Vec<usize>,
    elements: Vec<PreparedElement>,
}

impl VariableOrderSegmentElements {
    /// Prepare nodal segment elements for independently selected cell orders.
    pub fn lagrange_segments(
        mesh: &Mesh,
        dofs: &DofMap,
        orders: impl Into<Vec<usize>>,
    ) -> Result<Self, FinitumError> {
        if mesh.dimension() != 1 {
            return Err(FinitumError::UnsupportedRealization(
                "variable-order segment tables require a one-dimensional mesh".into(),
            ));
        }
        let orders = orders.into();
        if orders.len() != mesh.cells().len() || orders.len() != dofs.restrictions().len() {
            return Err(FinitumError::InvalidRealization(format!(
                "segment order count {}, mesh cell count {}, and restriction count {} must agree",
                orders.len(),
                mesh.cells().len(),
                dofs.restrictions().len()
            )));
        }
        let mut elements = Vec::with_capacity(orders.len());
        for (cell, order) in orders.iter().copied().enumerate() {
            let element = PreparedElement::lagrange_segment(order)?;
            let actual = dofs.restrictions()[cell].dofs.len();
            if actual != element.basis_count() {
                return Err(FinitumError::InvalidRealization(format!(
                    "segment cell {cell} at order {order} needs {} DOFs, got {actual}",
                    element.basis_count()
                )));
            }
            elements.push(element);
        }
        Ok(Self { orders, elements })
    }

    pub fn order(&self, cell: CellId) -> Option<usize> {
        self.orders.get(cell.0).copied()
    }

    pub fn element(&self, cell: CellId) -> Option<&PreparedElement> {
        self.elements.get(cell.0)
    }

    pub fn cell_count(&self) -> usize {
        self.elements.len()
    }
}

/// A hanging DOF expressed as an affine interpolation of its coarse masters.
#[derive(Clone, Debug, PartialEq)]
pub struct HangingNodeConstraint {
    constraint: AffineConstraint,
}

impl HangingNodeConstraint {
    /// Linear interpolation at `fraction` along a coarse edge.
    pub fn linear(
        target: DofId,
        left: DofId,
        right: DofId,
        fraction: f64,
    ) -> Result<Self, FinitumError> {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(FinitumError::InvalidRealization(
                "hanging-node edge fraction must be finite and in [0, 1]".into(),
            ));
        }
        if target == left || target == right || left == right {
            return Err(FinitumError::InvalidRealization(
                "hanging-node target and coarse masters must be distinct".into(),
            ));
        }
        Ok(Self {
            constraint: AffineConstraint {
                target,
                dependencies: vec![
                    WeightedDof {
                        dof: left,
                        weight: 1.0 - fraction,
                    },
                    WeightedDof {
                        dof: right,
                        weight: fraction,
                    },
                ],
                offset: 0.0,
            },
        })
    }

    pub fn as_affine(&self) -> &AffineConstraint {
        &self.constraint
    }

    pub fn into_affine(self) -> AffineConstraint {
        self.constraint
    }
}
