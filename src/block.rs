use crate::FinitumError;
use crate::system_ids::SysVarId;
use scientia::SymbolId;
use std::collections::{BTreeMap, BTreeSet};

/// One field's block of a product-space layout. `symbol` is the per-model coordinate (kept
/// for every single-model consumer, including Krasis's `SemanticId::new(symbol.0)` read);
/// `variable` is the system-level key (SC-W1, `sinbad/ARCHITECTURE.md` §2.4) -- equal to
/// `SysVarId(symbol.0)` for a layout built by [`BlockLayout::new`], and the composed dense id
/// for one built by [`BlockLayout::new_keyed`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldBlock {
    pub symbol: SymbolId,
    pub variable: SysVarId,
    pub entity_count: usize,
    pub component_count: usize,
    pub offset: usize,
    pub extent: usize,
}

/// Product-space global layout with explicit field and component extents.
///
/// Blocks are keyed by [`SysVarId`] (always unique) and, for the single-model case, by the
/// per-model [`SymbolId`]; in a composed layout ([`Self::new_keyed`]) two instances of one model
/// share symbols, so [`Self::block`] by symbol answers only for symbols that occur exactly once
/// and every symbol-keyed accessor documents that rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    blocks: Vec<FieldBlock>,
    by_symbol: BTreeMap<SymbolId, usize>,
    by_variable: BTreeMap<SysVarId, usize>,
    extent: usize,
}

impl BlockLayout {
    /// Single-model layout: every block keyed by its per-model symbol, with the identity
    /// system id `SysVarId(symbol.0)`.
    pub fn new(
        specifications: impl IntoIterator<Item = (SymbolId, usize, usize)>,
    ) -> Result<Self, FinitumError> {
        Self::new_keyed(
            specifications
                .into_iter()
                .map(|(symbol, entities, components)| {
                    (SysVarId(symbol.0), symbol, entities, components)
                }),
        )
    }

    /// System-keyed layout (SC-W1): `(system variable, per-model symbol, entity count,
    /// component count)` per block. System variables must be distinct; symbols may repeat
    /// across instances (and then are not addressable by symbol).
    pub fn new_keyed(
        specifications: impl IntoIterator<Item = (SysVarId, SymbolId, usize, usize)>,
    ) -> Result<Self, FinitumError> {
        let mut blocks = Vec::new();
        let mut by_variable = BTreeMap::new();
        let mut symbol_counts: BTreeMap<SymbolId, usize> = BTreeMap::new();
        let mut offset = 0usize;
        for (variable, symbol, entity_count, component_count) in specifications {
            if entity_count == 0 || component_count == 0 {
                return Err(FinitumError::InvalidRealization(format!(
                    "block {variable} ({symbol}) must have non-zero entity and component counts"
                )));
            }
            if by_variable.contains_key(&variable) {
                return Err(FinitumError::InvalidRealization(format!(
                    "block {variable} is declared more than once"
                )));
            }
            let extent = entity_count.checked_mul(component_count).ok_or_else(|| {
                FinitumError::InvalidRealization(format!("block {variable} extent overflows usize"))
            })?;
            let index = blocks.len();
            by_variable.insert(variable, index);
            *symbol_counts.entry(symbol).or_default() += 1;
            blocks.push(FieldBlock {
                symbol,
                variable,
                entity_count,
                component_count,
                offset,
                extent,
            });
            offset = offset.checked_add(extent).ok_or_else(|| {
                FinitumError::InvalidRealization("product-space extent overflows usize".into())
            })?;
        }
        if blocks.is_empty() {
            return Err(FinitumError::InvalidRealization(
                "product-space layout must contain at least one field block".into(),
            ));
        }
        let by_symbol = blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| symbol_counts[&block.symbol] == 1)
            .map(|(index, block)| (block.symbol, index))
            .collect();
        Ok(Self {
            blocks,
            by_symbol,
            by_variable,
            extent: offset,
        })
    }

    pub fn blocks(&self) -> &[FieldBlock] {
        &self.blocks
    }

    pub fn extent(&self) -> usize {
        self.extent
    }

    /// The block of a per-model symbol; `None` when no block carries it or when several do
    /// (a composed layout with repeated instances -- use [`Self::block_by_variable`]).
    pub fn block(&self, symbol: SymbolId) -> Option<&FieldBlock> {
        self.by_symbol
            .get(&symbol)
            .map(|index| &self.blocks[*index])
    }

    /// The block of a system variable.
    pub fn block_by_variable(&self, variable: SysVarId) -> Option<&FieldBlock> {
        self.by_variable
            .get(&variable)
            .map(|index| &self.blocks[*index])
    }

    /// Every system variable in block order.
    pub fn variables(&self) -> impl Iterator<Item = SysVarId> + '_ {
        self.blocks.iter().map(|block| block.variable)
    }

    /// The slice of `vector` owned by a system variable's block.
    pub fn values_by_variable<'a>(
        &self,
        vector: &'a [f64],
        variable: SysVarId,
    ) -> Result<&'a [f64], FinitumError> {
        self.validate_vector(vector)?;
        let block = self.block_by_variable(variable).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("layout has no block for {variable}"))
        })?;
        Ok(&vector[block.offset..block.offset + block.extent])
    }

    pub fn values<'a>(
        &self,
        vector: &'a [f64],
        symbol: SymbolId,
    ) -> Result<&'a [f64], FinitumError> {
        self.validate_vector(vector)?;
        let block = self.block(symbol).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("layout has no block for {symbol}"))
        })?;
        Ok(&vector[block.offset..block.offset + block.extent])
    }

    pub fn gather(
        &self,
        vector: &[f64],
        symbol: SymbolId,
        entities: &[usize],
    ) -> Result<Vec<f64>, FinitumError> {
        let values = self.values(vector, symbol)?;
        let block = self.block(symbol).expect("looked up by values");
        let mut distinct = BTreeSet::new();
        let mut gathered = Vec::with_capacity(entities.len() * block.component_count);
        for entity in entities {
            if *entity >= block.entity_count || !distinct.insert(*entity) {
                return Err(FinitumError::InvalidRealization(format!(
                    "block {symbol} has an invalid or repeated entity {entity}"
                )));
            }
            let start = entity * block.component_count;
            gathered.extend_from_slice(&values[start..start + block.component_count]);
        }
        Ok(gathered)
    }

    pub fn scatter_add(
        &self,
        vector: &mut [f64],
        symbol: SymbolId,
        entities: &[usize],
        local: &[f64],
    ) -> Result<(), FinitumError> {
        if vector.len() != self.extent {
            return Err(FinitumError::InvalidRealization(format!(
                "block vector has length {}, expected {}",
                vector.len(),
                self.extent
            )));
        }
        let block = self.block(symbol).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("layout has no block for {symbol}"))
        })?;
        if local.len() != entities.len() * block.component_count {
            return Err(FinitumError::InvalidRealization(format!(
                "local block has length {}, expected {}",
                local.len(),
                entities.len() * block.component_count
            )));
        }
        let mut distinct = BTreeSet::new();
        for (local_entity, entity) in entities.iter().enumerate() {
            if *entity >= block.entity_count || !distinct.insert(*entity) {
                return Err(FinitumError::InvalidRealization(format!(
                    "block {symbol} has an invalid or repeated entity {entity}"
                )));
            }
            for component in 0..block.component_count {
                vector[block.offset + entity * block.component_count + component] +=
                    local[local_entity * block.component_count + component];
            }
        }
        Ok(())
    }

    fn validate_vector(&self, vector: &[f64]) -> Result<(), FinitumError> {
        if vector.len() != self.extent || vector.iter().any(|value| !value.is_finite()) {
            Err(FinitumError::InvalidRealization(format!(
                "block vector must contain {} finite values",
                self.extent
            )))
        } else {
            Ok(())
        }
    }
}
