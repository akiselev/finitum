use crate::FinitumError;
use scientia::SymbolId;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldBlock {
    pub symbol: SymbolId,
    pub entity_count: usize,
    pub component_count: usize,
    pub offset: usize,
    pub extent: usize,
}

/// Product-space global layout with explicit field and component extents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    blocks: Vec<FieldBlock>,
    by_symbol: BTreeMap<SymbolId, usize>,
    extent: usize,
}

impl BlockLayout {
    pub fn new(
        specifications: impl IntoIterator<Item = (SymbolId, usize, usize)>,
    ) -> Result<Self, FinitumError> {
        let mut blocks = Vec::new();
        let mut by_symbol = BTreeMap::new();
        let mut offset = 0usize;
        for (symbol, entity_count, component_count) in specifications {
            if entity_count == 0 || component_count == 0 {
                return Err(FinitumError::InvalidRealization(format!(
                    "block {symbol} must have non-zero entity and component counts"
                )));
            }
            if by_symbol.contains_key(&symbol) {
                return Err(FinitumError::InvalidRealization(format!(
                    "block {symbol} is declared more than once"
                )));
            }
            let extent = entity_count.checked_mul(component_count).ok_or_else(|| {
                FinitumError::InvalidRealization(format!("block {symbol} extent overflows usize"))
            })?;
            let index = blocks.len();
            by_symbol.insert(symbol, index);
            blocks.push(FieldBlock {
                symbol,
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
        Ok(Self {
            blocks,
            by_symbol,
            extent: offset,
        })
    }

    pub fn blocks(&self) -> &[FieldBlock] {
        &self.blocks
    }

    pub fn extent(&self) -> usize {
        self.extent
    }

    pub fn block(&self, symbol: SymbolId) -> Option<&FieldBlock> {
        self.by_symbol
            .get(&symbol)
            .map(|index| &self.blocks[*index])
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
