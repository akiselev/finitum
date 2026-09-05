//! System-level identities for realization (SC-W1, `sinbad/ARCHITECTURE.md` §2.3/§2.4/§2.6).
//!
//! Finitum keys its realization by its **own** newtypes of the same `u32` wire width as
//! Scientia's `scientia-system/1` ids and carries an explicit origin table ([`SystemIdMap`])
//! that says, for every system-level id, which instance and which per-model coordinate it came
//! from. [`SystemIdMap::from_scientia`] builds the map from Scientia's
//! `scientia-operator-system/2` `SystemOperator` (its `variables`, `residuals`, `instances`,
//! and `instance_artifacts`; the ids are copied by value, never re-allocated);
//! [`SystemIdMap::compose`] is Finitum's own allocation of the same canonical order, proven
//! equal to Scientia's on a declared two-instance system (`tests/w7_sc_w1_scientia_ids.rs`),
//! and kept for callers composing per-model `/1` artifacts by hand.
//!
//! Allocation is canonical (§2.3): instance declaration order, then local id order (variables
//! by `SymbolId`, residuals by equation order within the instance's `OperatorSystem`), dense
//! from zero. A single-model system is the degenerate one-instance composition with the
//! **identity** maps (`SysVarId(symbol.0)`, `SysResId(block index)`), which is what keeps every
//! existing single-model realization -- and Krasis's `SemanticId = SysVarId` convention, read
//! today from `FieldBlock::symbol` -- numerically unchanged.

use crate::FinitumError;
use scientia::{Digest, OperatorSystem, ResidualOrigin, SymbolId};
use serde::Serialize;
use std::collections::BTreeMap;

/// Schema of [`SystemIdMap::identity`].
pub const SYSTEM_ID_MAP_SCHEMA: &str = "finitum-system-ids/1";

/// One instance of a model inside a realization group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct InstanceId(pub u32);

/// A system-level variable (field) id: dense across the whole system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SysVarId(pub u32);

/// A system-level residual (equation row) id: dense across the whole system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SysResId(pub u32);

impl std::fmt::Display for InstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "instance#{}", self.0)
    }
}

impl std::fmt::Display for SysVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sysvar#{}", self.0)
    }
}

impl std::fmt::Display for SysResId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sysres#{}", self.0)
    }
}

/// Origin of one system variable: the instance and its per-model symbol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SysVar {
    pub id: SysVarId,
    pub instance: InstanceId,
    pub local: SymbolId,
}

/// Origin of one system residual: the instance, its equation name, and the per-model row
/// symbol (the field the equation's test function belongs to).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SysRes {
    pub id: SysResId,
    pub instance: InstanceId,
    pub equation: String,
    pub row: SymbolId,
}

/// The per-instance receipt chain entry (§2.4): which model the instance is, and the semantic
/// and artifact digests of the per-model `OperatorSystem` it reuses verbatim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InstanceRecord {
    pub instance: InstanceId,
    pub name: String,
    pub model: String,
    pub semantic_digest: Digest,
    pub artifact_digest: Digest,
}

/// Finitum's origin table over one realization group: every [`SysVarId`]/[`SysResId`] with
/// its instance and per-model coordinate, plus the per-instance receipt list, under one
/// content-addressed identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemIdMap {
    instances: Vec<InstanceRecord>,
    variables: Vec<SysVar>,
    residuals: Vec<SysRes>,
    variable_index: BTreeMap<(InstanceId, SymbolId), usize>,
    residual_index: BTreeMap<(InstanceId, String), usize>,
    identity: Digest,
}

#[derive(Serialize)]
struct MapPayload<'a> {
    schema: &'static str,
    instances: &'a [InstanceRecord],
    variables: &'a [SysVar],
    residuals: &'a [SysRes],
}

impl SystemIdMap {
    /// The degenerate one-instance system of a model (§2.6): `InstanceId(0)`, and the identity
    /// maps `SysVarId(symbol.0)` for every field in `field_order` and `SysResId(k)` for the
    /// `k`-th equation block.
    pub fn one_instance(system: &OperatorSystem) -> Result<Self, FinitumError> {
        let instance = InstanceId(0);
        let mut variables = Vec::with_capacity(system.field_order.len());
        for symbol in &system.field_order {
            variables.push(SysVar {
                id: SysVarId(symbol.0),
                instance,
                local: *symbol,
            });
        }
        variables.sort_by_key(|variable| variable.id);
        let residuals = system
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| SysRes {
                id: SysResId(u32::try_from(index).expect("block count fits u32")),
                instance,
                equation: block.equation.clone(),
                row: block.row,
            })
            .collect();
        Self::build(
            vec![InstanceRecord {
                instance,
                name: system.model.clone(),
                model: system.model.clone(),
                semantic_digest: system.source_semantic_digest.clone(),
                artifact_digest: system.artifact_digest.clone(),
            }],
            variables,
            residuals,
        )
    }

    /// Dense canonical allocation over several instances (§2.3): instance order, then local
    /// `SymbolId` order for variables and equation order for residuals. Each entry is
    /// `(instance name, per-model OperatorSystem)`; the per-model artifacts are referenced, not
    /// re-numbered (§2.2).
    ///
    /// Recorded Scientia surface (the exact shape this constructor replaces once
    /// `scientia-system/1` lands): `OriginMap.variables: Vec<(SysVarId, InstanceId,
    /// GlobalDeclId, SymbolId, SourceLocator)>` and `OriginMap.residuals: Vec<(SysResId,
    /// ResidualOrigin::Equation { instance, decl })>` plus `OperatorSystem/2.instances[i].{model,
    /// semantic_digest}`; Finitum then keeps `SysVar.local`/`SysRes.equation` as the
    /// per-model coordinates and drops its own allocation rule.
    pub fn compose(instances: &[(&str, &OperatorSystem)]) -> Result<Self, FinitumError> {
        if instances.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "a system id map needs at least one instance".into(),
            ));
        }
        let mut records = Vec::with_capacity(instances.len());
        let mut variables = Vec::new();
        let mut residuals = Vec::new();
        for (index, (name, system)) in instances.iter().enumerate() {
            let instance = InstanceId(u32::try_from(index).expect("instance count fits u32"));
            if name.is_empty() {
                return Err(FinitumError::InvalidRealization(format!(
                    "{instance} has an empty name"
                )));
            }
            records.push(InstanceRecord {
                instance,
                name: (*name).to_string(),
                model: system.model.clone(),
                semantic_digest: system.source_semantic_digest.clone(),
                artifact_digest: system.artifact_digest.clone(),
            });
            let mut locals = system.field_order.clone();
            locals.sort();
            for local in locals {
                variables.push(SysVar {
                    id: SysVarId(u32::try_from(variables.len()).expect("fits u32")),
                    instance,
                    local,
                });
            }
            for block in &system.blocks {
                residuals.push(SysRes {
                    id: SysResId(u32::try_from(residuals.len()).expect("fits u32")),
                    instance,
                    equation: block.equation.clone(),
                    row: block.row,
                });
            }
        }
        Self::build(records, variables, residuals)
    }

    /// The map Scientia's `scientia-operator-system/2` artifact fixes (HANDOFF §6, W7): every
    /// `SysVar { id, owner, local }` and `SysResBlock { id, origin: Equation { instance, name },
    /// row }` copied by value into Finitum's newtypes (same `u32` wire width, no re-allocation),
    /// with one [`InstanceRecord`] per `instances` entry whose `artifact_digest` is the
    /// instance's entry in `instance_artifacts` (the per-model `/1` artifact digest). The
    /// implicit root instance (empty `name`) is recorded under its model name, which is exactly
    /// what [`Self::one_instance`] records, so the two maps -- and their `finitum-system-ids/1`
    /// identities -- coincide for an implicit one-instance system. An instance without an
    /// artifact digest, or an id/origin the map's own consistency rules reject (duplicate ids,
    /// duplicate instance names, a row that is not a system variable), is refused typed.
    pub fn from_scientia(operator: &scientia::SystemOperator) -> Result<Self, FinitumError> {
        let mut records = Vec::with_capacity(operator.instances.len());
        for record in &operator.instances {
            let artifact_digest = operator
                .instance_artifacts
                .iter()
                .find(|(instance, _)| *instance == record.instance)
                .map(|(_, digest)| digest.clone())
                .ok_or_else(|| {
                    FinitumError::ArtifactMismatch(format!(
                        "scientia system operator carries no `/1` artifact digest for instance \
                         {} (`{}`)",
                        record.instance.0, record.model_name
                    ))
                })?;
            records.push(InstanceRecord {
                instance: InstanceId(record.instance.0),
                name: if record.name.is_empty() {
                    record.model_name.clone()
                } else {
                    record.name.clone()
                },
                model: record.model_name.clone(),
                semantic_digest: record.semantic_digest.clone(),
                artifact_digest,
            });
        }
        let variables = operator
            .variables
            .iter()
            .map(|variable| SysVar {
                id: SysVarId(variable.id.0),
                instance: InstanceId(variable.owner.0),
                local: variable.local,
            })
            .collect();
        let residuals = operator
            .residuals
            .iter()
            .map(|residual| {
                let ResidualOrigin::Equation { instance, name, .. } = &residual.origin;
                SysRes {
                    id: SysResId(residual.id.0),
                    instance: InstanceId(instance.0),
                    equation: name.clone(),
                    row: residual.row,
                }
            })
            .collect();
        Self::build(records, variables, residuals)
    }

    fn build(
        instances: Vec<InstanceRecord>,
        variables: Vec<SysVar>,
        residuals: Vec<SysRes>,
    ) -> Result<Self, FinitumError> {
        let mut names = BTreeMap::new();
        for record in &instances {
            if names.insert(record.name.clone(), record.instance).is_some() {
                return Err(FinitumError::InvalidRealization(format!(
                    "instance name `{}` is used more than once",
                    record.name
                )));
            }
        }
        let mut variable_index = BTreeMap::new();
        let mut seen_ids = BTreeMap::new();
        for (position, variable) in variables.iter().enumerate() {
            if seen_ids.insert(variable.id, position).is_some() {
                return Err(FinitumError::InvalidRealization(format!(
                    "{} is allocated more than once",
                    variable.id
                )));
            }
            if variable_index
                .insert((variable.instance, variable.local), position)
                .is_some()
            {
                return Err(FinitumError::InvalidRealization(format!(
                    "field {} of {} is allocated more than once",
                    variable.local, variable.instance
                )));
            }
        }
        let mut residual_index = BTreeMap::new();
        let mut seen_res = BTreeMap::new();
        for (position, residual) in residuals.iter().enumerate() {
            if seen_res.insert(residual.id, position).is_some() {
                return Err(FinitumError::InvalidRealization(format!(
                    "{} is allocated more than once",
                    residual.id
                )));
            }
            if residual_index
                .insert((residual.instance, residual.equation.clone()), position)
                .is_some()
            {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{}` of {} is allocated more than once",
                    residual.equation, residual.instance
                )));
            }
            if !variable_index.contains_key(&(residual.instance, residual.row)) {
                return Err(FinitumError::InvalidRealization(format!(
                    "equation `{}` of {} has row field {} which is not a system variable",
                    residual.equation, residual.instance, residual.row
                )));
            }
        }
        let bytes = serde_json::to_vec(&MapPayload {
            schema: SYSTEM_ID_MAP_SCHEMA,
            instances: &instances,
            variables: &variables,
            residuals: &residuals,
        })
        .map_err(|error| FinitumError::InvalidRealization(error.to_string()))?;
        Ok(Self {
            instances,
            variables,
            residuals,
            variable_index,
            residual_index,
            identity: Digest::blake3(&bytes),
        })
    }

    pub fn instances(&self) -> &[InstanceRecord] {
        &self.instances
    }

    /// Every system variable in `SysVarId` order.
    pub fn variables(&self) -> &[SysVar] {
        &self.variables
    }

    /// Every system residual in `SysResId` order.
    pub fn residuals(&self) -> &[SysRes] {
        &self.residuals
    }

    pub fn identity(&self) -> &Digest {
        &self.identity
    }

    /// The system variable of a per-model field of one instance.
    pub fn variable(&self, instance: InstanceId, local: SymbolId) -> Option<SysVarId> {
        self.variable_index
            .get(&(instance, local))
            .map(|&position| self.variables[position].id)
    }

    /// The origin of a system variable.
    pub fn variable_origin(&self, id: SysVarId) -> Option<&SysVar> {
        self.variables.iter().find(|variable| variable.id == id)
    }

    /// The system residual of one instance's equation.
    pub fn residual(&self, instance: InstanceId, equation: &str) -> Option<SysResId> {
        self.residual_index
            .get(&(instance, equation.to_string()))
            .map(|&position| self.residuals[position].id)
    }

    /// The origin of a system residual.
    pub fn residual_origin(&self, id: SysResId) -> Option<&SysRes> {
        self.residuals.iter().find(|residual| residual.id == id)
    }

    /// The display path of a residual (§2.4): `<instance name>.<equation>`; the root of a
    /// one-instance system is unprefixed.
    pub fn residual_path(&self, id: SysResId) -> Option<String> {
        let residual = self.residual_origin(id)?;
        let record = self
            .instances
            .iter()
            .find(|record| record.instance == residual.instance)?;
        Some(if self.instances.len() == 1 {
            residual.equation.clone()
        } else {
            format!("{}.{}", record.name, residual.equation)
        })
    }

    /// The display path of a variable: `<instance name>/<local symbol>`, unprefixed for the
    /// root of a one-instance system.
    pub fn variable_path(&self, id: SysVarId) -> Option<String> {
        let variable = self.variable_origin(id)?;
        let record = self
            .instances
            .iter()
            .find(|record| record.instance == variable.instance)?;
        Some(if self.instances.len() == 1 {
            variable.local.to_string()
        } else {
            format!("{}/{}", record.name, variable.local)
        })
    }
}
