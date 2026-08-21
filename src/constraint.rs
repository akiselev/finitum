use crate::{DofId, FinitumError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeightedDof {
    pub dof: DofId,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AffineConstraint {
    pub target: DofId,
    pub dependencies: Vec<WeightedDof>,
    pub offset: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConstraintSet {
    dof_count: usize,
    constraints: BTreeMap<DofId, AffineConstraint>,
}

impl ConstraintSet {
    pub fn new(
        dof_count: usize,
        constraints: impl IntoIterator<Item = AffineConstraint>,
    ) -> Result<Self, FinitumError> {
        let mut map = BTreeMap::new();
        for constraint in constraints {
            if constraint.target.0 >= dof_count {
                return Err(FinitumError::InvalidConstraintTarget(constraint.target.0));
            }
            let mut dependencies = BTreeSet::new();
            for dependency in &constraint.dependencies {
                if dependency.dof.0 >= dof_count {
                    return Err(FinitumError::InvalidConstraintDependency(dependency.dof.0));
                }
                if !dependencies.insert(dependency.dof) {
                    return Err(FinitumError::DuplicateConstraintDependency {
                        target: constraint.target.0,
                        dependency: dependency.dof.0,
                    });
                }
                if !dependency.weight.is_finite() {
                    return Err(FinitumError::InvalidConstraintCoefficient {
                        target: constraint.target.0,
                    });
                }
            }
            if !constraint.offset.is_finite() {
                return Err(FinitumError::InvalidConstraintCoefficient {
                    target: constraint.target.0,
                });
            }
            let target = constraint.target;
            if map.insert(target, constraint).is_some() {
                return Err(FinitumError::DuplicateConstraintTarget(target.0));
            }
        }
        let set = Self {
            dof_count,
            constraints: map,
        };
        set.validate_acyclic()?;
        Ok(set)
    }

    pub fn constraints(&self) -> impl Iterator<Item = &AffineConstraint> {
        self.constraints.values()
    }

    pub fn dof_count(&self) -> usize {
        self.dof_count
    }

    pub fn is_constrained(&self, dof: DofId) -> bool {
        self.constraints.contains_key(&dof)
    }

    pub(crate) fn has_affine_dependencies(&self) -> bool {
        self.constraints
            .values()
            .any(|constraint| !constraint.dependencies.is_empty())
    }

    pub fn expand(&self, unconstrained: &[f64]) -> Result<Vec<f64>, FinitumError> {
        if unconstrained.len() != self.dof_count {
            return Err(FinitumError::ConstraintInputLength {
                actual: unconstrained.len(),
                expected: self.dof_count,
            });
        }
        if let Some(dof) = unconstrained.iter().position(|value| !value.is_finite()) {
            return Err(FinitumError::NonFiniteConstraintInput(dof));
        }
        let mut values = unconstrained.to_vec();
        let mut resolved = BTreeSet::new();
        for target in self.constraints.keys().copied() {
            self.resolve(target, &mut values, &mut resolved)?;
        }
        Ok(values)
    }

    fn resolve(
        &self,
        target: DofId,
        values: &mut [f64],
        resolved: &mut BTreeSet<DofId>,
    ) -> Result<(), FinitumError> {
        if resolved.contains(&target) {
            return Ok(());
        }
        let Some(constraint) = self.constraints.get(&target) else {
            return Ok(());
        };
        let mut value = constraint.offset;
        for dependency in &constraint.dependencies {
            self.resolve(dependency.dof, values, resolved)?;
            value += dependency.weight * values[dependency.dof.0];
        }
        if !value.is_finite() {
            return Err(FinitumError::NonFiniteConstraintResult(target.0));
        }
        values[target.0] = value;
        resolved.insert(target);
        Ok(())
    }

    fn validate_acyclic(&self) -> Result<(), FinitumError> {
        fn visit(
            dof: DofId,
            constraints: &BTreeMap<DofId, AffineConstraint>,
            active: &mut BTreeSet<DofId>,
            done: &mut BTreeSet<DofId>,
        ) -> Result<(), FinitumError> {
            if done.contains(&dof) {
                return Ok(());
            }
            if !active.insert(dof) {
                return Err(FinitumError::ConstraintCycle(dof.0));
            }
            if let Some(constraint) = constraints.get(&dof) {
                for dependency in &constraint.dependencies {
                    if constraints.contains_key(&dependency.dof) {
                        visit(dependency.dof, constraints, active, done)?;
                    }
                }
            }
            active.remove(&dof);
            done.insert(dof);
            Ok(())
        }

        let mut active = BTreeSet::new();
        let mut done = BTreeSet::new();
        for dof in self.constraints.keys().copied() {
            visit(dof, &self.constraints, &mut active, &mut done)?;
        }
        Ok(())
    }
}
