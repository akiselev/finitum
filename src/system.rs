use crate::{BlockLayout, CompatibleDofMaps, ExactSequence, FacetTopology, FinitumError, Mesh};
use scientia::scientific::ValueShape;
use scientia::{Digest, ElementFamilyRequirement, OperatorSystem, SemanticMeasure};
use serde::Serialize;
use std::sync::Arc;

/// Digest-bound concrete ownership plan for an FC8 mixed operator system.
#[derive(Clone, Debug)]
pub struct SystemRealizationPlan {
    system: Arc<OperatorSystem>,
    mesh: Mesh,
    layout: BlockLayout,
    facets: FacetTopology,
    compatible_dofs: Option<CompatibleDofMaps>,
    exact_sequence: Option<ExactSequence>,
    artifact_digest: Digest,
}

impl SystemRealizationPlan {
    pub fn new(
        system: OperatorSystem,
        mesh: Mesh,
        layout: BlockLayout,
    ) -> Result<Self, FinitumError> {
        for symbol in &system.field_order {
            if layout.block(*symbol).is_none() {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "mixed layout has no block for system field {symbol}"
                )));
            }
        }
        for block in &system.blocks {
            if block.form.source_semantic_digest != system.source_semantic_digest
                || block.factorization.receipt.source_form_digest != block.form.artifact_digest
                || block.kernels.source_factorization_digest != block.factorization.artifact_digest
            {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "system equation `{}` has a broken form/factorization/kernel receipt chain",
                    block.equation
                )));
            }
            for coordinate in &block.coordinates {
                if layout.block(coordinate.row).is_none()
                    || layout.block(coordinate.column).is_none()
                {
                    return Err(FinitumError::ArtifactMismatch(format!(
                        "system coordinate ({}, {}) is absent from the concrete layout",
                        coordinate.row, coordinate.column
                    )));
                }
            }
        }
        validate_components(&system, &layout)?;
        let facets = FacetTopology::from_mesh(&mesh)?;
        let uses_facets = system.blocks.iter().any(|block| {
            block
                .factorization
                .integrals
                .iter()
                .any(|integral| !matches!(integral.measure, SemanticMeasure::Cell { .. }))
        });
        if uses_facets && facets.facets().is_empty() {
            return Err(FinitumError::InvalidRealization(
                "facet operator system requires a nonempty facet topology".into(),
            ));
        }
        let uses_compatible = system.blocks.iter().any(|block| {
            block.requirements.elements.iter().any(|element| {
                matches!(
                    element.family,
                    ElementFamilyRequirement::Hcurl | ElementFamilyRequirement::Hdiv
                )
            })
        });
        let (compatible_dofs, exact_sequence) = if uses_compatible {
            (
                Some(CompatibleDofMaps::simplex(&mesh, &facets)?),
                Some(ExactSequence::simplex(&mesh, &facets)?),
            )
        } else {
            (None, None)
        };
        let artifact_digest = digest_plan(&system, &mesh, &layout, &facets);
        Ok(Self {
            system: Arc::new(system),
            mesh,
            layout,
            facets,
            compatible_dofs,
            exact_sequence,
            artifact_digest,
        })
    }

    pub fn system(&self) -> &OperatorSystem {
        &self.system
    }

    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    pub fn facets(&self) -> &FacetTopology {
        &self.facets
    }

    pub fn compatible_dofs(&self) -> Option<&CompatibleDofMaps> {
        self.compatible_dofs.as_ref()
    }

    pub fn exact_sequence(&self) -> Option<&ExactSequence> {
        self.exact_sequence.as_ref()
    }

    pub fn artifact_digest(&self) -> &Digest {
        &self.artifact_digest
    }
}

fn validate_components(system: &OperatorSystem, layout: &BlockLayout) -> Result<(), FinitumError> {
    for block in &system.blocks {
        for space in &block.requirements.spaces {
            let Some(concrete) = layout.block(space.symbol) else {
                continue;
            };
            let expected = match space.value_shape {
                ValueShape::Scalar => 1,
                ValueShape::Vector(extent) => usize::from(extent),
                ValueShape::Tensor { rows, cols } => usize::from(rows) * usize::from(cols),
                ValueShape::SymmetricTensor(extent) => usize::from(extent) * usize::from(extent),
            };
            if concrete.component_count != expected {
                return Err(FinitumError::ArtifactMismatch(format!(
                    "field {} has {} concrete components, typed space requires {expected}",
                    space.symbol, concrete.component_count
                )));
            }
        }
    }
    Ok(())
}

fn digest_plan(
    system: &OperatorSystem,
    mesh: &Mesh,
    layout: &BlockLayout,
    facets: &FacetTopology,
) -> Digest {
    #[derive(Serialize)]
    struct BlockIdentity {
        symbol: u32,
        entity_count: usize,
        component_count: usize,
        offset: usize,
    }
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        system: &'a Digest,
        dimension: usize,
        vertices: &'a [Vec<f64>],
        cells: Vec<Vec<usize>>,
        blocks: Vec<BlockIdentity>,
        facet_count: usize,
    }
    let bytes = serde_json::to_vec(&Payload {
        schema: "finitum-system-realization/1",
        system: &system.artifact_digest,
        dimension: mesh.dimension(),
        vertices: mesh.vertices(),
        cells: mesh
            .cells()
            .iter()
            .map(|cell| cell.vertices.iter().map(|vertex| vertex.0).collect())
            .collect(),
        blocks: layout
            .blocks()
            .iter()
            .map(|block| BlockIdentity {
                symbol: block.symbol.0,
                entity_count: block.entity_count,
                component_count: block.component_count,
                offset: block.offset,
            })
            .collect(),
        facet_count: facets.facets().len(),
    })
    .expect("system realization identity is serializable");
    Digest {
        algorithm: "blake3".into(),
        hex: blake3::hash(&bytes).to_hex().to_string(),
    }
}
