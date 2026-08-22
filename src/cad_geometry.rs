use std::collections::BTreeMap;

use cadabra_provider::{RectangleProvider, StableId};
use serde::Serialize;

use crate::{
    AffineConstraint, AssembledOperator, Cell, CellId, ConstraintSet, DofId, DofMap,
    ElementRestriction, FinitumError, MatrixFreeOperator, Mesh, RealizationPlan, VertexId,
};

/// Immutable CAD provider identity captured by a concrete realization.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadGeometrySource {
    /// Provider-owned geometry identity.
    pub geometry_id: String,
    /// Exact immutable provider revision used to create the realization.
    pub revision: u64,
    /// Provider semantic digest, independent of the mesh resolution.
    pub semantic_digest: [u8; 32],
}

/// Stable CAD design-parameter identity and admitted coordinate value.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadParameterCoordinate {
    /// Provider-owned parameter identity.
    pub parameter_id: String,
    /// Coordinate value frozen by the source snapshot.
    pub value: f64,
}

/// Deterministic association between one realized vertex and its CAD chart coordinate.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadNodeAssociation {
    /// Realization-owned stable node identity.
    pub node_id: String,
    /// Concrete mesh vertex.
    pub vertex: VertexId,
    /// Provider chart coordinate used to evaluate the vertex.
    pub reference_coordinate: [f64; 2],
}

/// Deterministic association between one simplex and its CAD region.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadCellAssociation {
    /// Realization-owned stable cell identity.
    pub cell_id: String,
    /// Concrete mesh cell.
    pub cell: CellId,
    /// Provider-owned material-region identity.
    pub region_id: String,
}

/// Stable CAD boundary identity and the concrete vertices associated with it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadBoundaryAssociation {
    /// Provider-owned boundary identity.
    pub entity_id: String,
    /// Deterministically ordered boundary vertices.
    pub vertices: Vec<VertexId>,
}

/// Constant essential value selected by stable CAD boundary identity.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CadBoundaryCondition {
    /// Provider-owned boundary identity.
    pub entity_id: String,
    /// Essential value applied to every associated vertex.
    pub value: f64,
}

/// Finitum-owned deterministic primal realization of one admitted CAD rectangle.
///
/// R3P deliberately realizes only rectangles whose world-space carrier is the
/// XY plane. Supporting embedded surface finite elements would require a
/// distinct concrete mapping capability; silently dropping the third
/// coordinate is refused.
#[derive(Clone, Debug, PartialEq)]
pub struct CadGeometryRealization {
    source: CadGeometrySource,
    parameters: Vec<CadParameterCoordinate>,
    mesh: Mesh,
    nodes: Vec<CadNodeAssociation>,
    cells: Vec<CadCellAssociation>,
    boundaries: Vec<CadBoundaryAssociation>,
    digest: [u8; 32],
}

/// Primal operator plan that keeps its CAD association identity attached.
#[derive(Clone, Debug)]
pub struct CadPrimalPlan {
    geometry: CadGeometryRealization,
    boundary_conditions: Vec<CadBoundaryCondition>,
    realization: RealizationPlan,
    digest: scientia::Digest,
}

impl CadGeometryRealization {
    /// Samples an admitted affine rectangle into a deterministic triangular
    /// P1 mesh, after checking the caller's immutable provider revision.
    pub fn from_rectangle(
        provider: &RectangleProvider,
        expected_revision: u64,
        subdivisions: [usize; 2],
    ) -> Result<Self, FinitumError> {
        let snapshot = provider.snapshot();
        if expected_revision != snapshot.revision {
            return Err(FinitumError::StaleGeometryRevision {
                expected: snapshot.revision,
                actual: expected_revision,
            });
        }
        let [u_cells, v_cells] = subdivisions;
        if u_cells == 0 || v_cells == 0 {
            return Err(FinitumError::InvalidCadGeometry(
                "rectangle subdivisions must be positive".into(),
            ));
        }
        let node_count = u_cells
            .checked_add(1)
            .and_then(|u| v_cells.checked_add(1).and_then(|v| u.checked_mul(v)))
            .ok_or_else(|| FinitumError::InvalidCadGeometry("node count overflow".into()))?;
        let cell_count = u_cells
            .checked_mul(v_cells)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| FinitumError::InvalidCadGeometry("cell count overflow".into()))?;
        if node_count > 1_000_000 || cell_count > 1_000_000 {
            return Err(FinitumError::InvalidCadGeometry(
                "rectangle realization exceeds the one-million-item work cap".into(),
            ));
        }
        validate_xy_carrier(snapshot.frame.axes)?;

        let width = u_cells + 1;
        let mut vertices = Vec::with_capacity(node_count);
        let mut nodes = Vec::with_capacity(node_count);
        for row in 0..=v_cells {
            for column in 0..=u_cells {
                let reference_coordinate =
                    [column as f64 / u_cells as f64, row as f64 / v_cells as f64];
                let evaluation = provider
                    .evaluate(reference_coordinate)
                    .map_err(|error| FinitumError::InvalidCadGeometry(error.to_string()))?;
                if evaluation.position[2] != snapshot.frame.origin[2] {
                    return Err(FinitumError::InvalidCadGeometry(
                        "rectangle is not contained in a constant-Z XY carrier".into(),
                    ));
                }
                let vertex = VertexId(vertices.len());
                vertices.push(vec![evaluation.position[0], evaluation.position[1]]);
                nodes.push(CadNodeAssociation {
                    node_id: format!(
                        "{}/realization/node/{column}/{row}",
                        snapshot.geometry_id.as_str()
                    ),
                    vertex,
                    reference_coordinate,
                });
            }
        }

        let positive_xy_orientation = snapshot.frame.axes[2][2] > 0.0;
        let mut mesh_cells = Vec::with_capacity(cell_count);
        for row in 0..v_cells {
            for column in 0..u_cells {
                let lower_left = row * width + column;
                let lower_right = lower_left + 1;
                let upper_left = lower_left + width;
                let upper_right = upper_left + 1;
                let triangles = if positive_xy_orientation {
                    [
                        [lower_left, lower_right, upper_right],
                        [lower_left, upper_right, upper_left],
                    ]
                } else {
                    [
                        [lower_left, upper_right, lower_right],
                        [lower_left, upper_left, upper_right],
                    ]
                };
                for triangle in triangles {
                    mesh_cells.push(Cell {
                        vertices: triangle.into_iter().map(VertexId).collect(),
                    });
                }
            }
        }
        let mesh = Mesh::new(2, vertices, mesh_cells)?;
        let cells = (0..mesh.cells().len())
            .map(|cell| CadCellAssociation {
                cell_id: format!("{}/realization/cell/{cell}", snapshot.geometry_id.as_str()),
                cell: CellId(cell),
                region_id: snapshot.region.id.as_str().to_owned(),
            })
            .collect::<Vec<_>>();
        let boundaries = rectangle_boundaries(
            snapshot.boundaries.clone().map(|entity| entity.id),
            width,
            u_cells,
            v_cells,
        );
        let source = CadGeometrySource {
            geometry_id: snapshot.geometry_id.as_str().to_owned(),
            revision: snapshot.revision,
            semantic_digest: snapshot.semantic_digest.bytes(),
        };
        let parameters = snapshot
            .design_parameters
            .iter()
            .map(|parameter| CadParameterCoordinate {
                parameter_id: parameter.id.as_str().to_owned(),
                value: parameter.value,
            })
            .collect::<Vec<_>>();
        let digest = association_digest(&source, &parameters, &mesh, &nodes, &cells, &boundaries)?;
        Ok(Self {
            source,
            parameters,
            mesh,
            nodes,
            cells,
            boundaries,
            digest,
        })
    }

    /// Provider identity frozen by this realization.
    pub fn source(&self) -> &CadGeometrySource {
        &self.source
    }

    /// Stable design coordinates frozen by the provider snapshot.
    pub fn parameters(&self) -> &[CadParameterCoordinate] {
        &self.parameters
    }

    /// Concrete P1 simplex mesh.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// Stable node-to-chart associations.
    pub fn nodes(&self) -> &[CadNodeAssociation] {
        &self.nodes
    }

    /// Stable cell-to-region associations.
    pub fn cells(&self) -> &[CadCellAssociation] {
        &self.cells
    }

    /// Stable boundary associations.
    pub fn boundaries(&self) -> &[CadBoundaryAssociation] {
        &self.boundaries
    }

    /// Digest covering source revision, parameter coordinates, mesh, and all associations.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Requires the complete provider source identity, not merely a numeric revision.
    pub fn require_rectangle_source(
        &self,
        provider: &RectangleProvider,
    ) -> Result<(), FinitumError> {
        let snapshot = provider.snapshot();
        if snapshot.revision != self.source.revision {
            return Err(FinitumError::StaleGeometryRevision {
                expected: self.source.revision,
                actual: snapshot.revision,
            });
        }
        if snapshot.geometry_id.as_str() != self.source.geometry_id
            || snapshot.semantic_digest.bytes() != self.source.semantic_digest
        {
            return Err(FinitumError::CadGeometrySourceMismatch);
        }
        Ok(())
    }

    /// Returns one unambiguous concrete boundary association.
    pub fn boundary(&self, entity_id: &str) -> Result<&CadBoundaryAssociation, FinitumError> {
        let mut matches = self
            .boundaries
            .iter()
            .filter(|association| association.entity_id == entity_id);
        let Some(association) = matches.next() else {
            return Err(FinitumError::MissingCadBoundary(entity_id.to_owned()));
        };
        if matches.next().is_some() {
            return Err(FinitumError::AmbiguousCadBoundary(entity_id.to_owned()));
        }
        Ok(association)
    }

    /// Creates the nodal P1 degree-of-freedom map for the associated mesh.
    pub fn nodal_dof_map(&self) -> Result<DofMap, FinitumError> {
        DofMap::new(
            self.mesh.vertices().len(),
            self.mesh
                .cells()
                .iter()
                .map(|cell| ElementRestriction {
                    dofs: cell.vertices.iter().map(|vertex| DofId(vertex.0)).collect(),
                })
                .collect(),
        )
    }

    /// Resolves constant essential conditions by stable CAD boundary identity.
    /// Duplicate identities and conflicting values at shared CAD corners are
    /// refused rather than resolved by ordering.
    pub fn essential_constraints(
        &self,
        conditions: &[CadBoundaryCondition],
    ) -> Result<ConstraintSet, FinitumError> {
        let mut seen = BTreeMap::<&str, f64>::new();
        let mut vertex_values = BTreeMap::<VertexId, (&str, f64)>::new();
        for condition in conditions {
            if !condition.value.is_finite() {
                return Err(FinitumError::InvalidCadGeometry(format!(
                    "boundary value for {} is not finite",
                    condition.entity_id
                )));
            }
            if seen.insert(&condition.entity_id, condition.value).is_some() {
                return Err(FinitumError::AmbiguousCadBoundary(
                    condition.entity_id.clone(),
                ));
            }
            let association = self.boundary(&condition.entity_id)?;
            for vertex in &association.vertices {
                if let Some((other, value)) = vertex_values.get(vertex) {
                    if value.to_bits() != condition.value.to_bits() {
                        return Err(FinitumError::AmbiguousCadBoundary(format!(
                            "{} and {} overlap at vertex {} with different values",
                            other, condition.entity_id, vertex.0
                        )));
                    }
                } else {
                    vertex_values.insert(*vertex, (&condition.entity_id, condition.value));
                }
            }
        }
        ConstraintSet::new(
            self.mesh.vertices().len(),
            vertex_values
                .into_iter()
                .map(|(target, (_, offset))| AffineConstraint {
                    target: DofId(target.0),
                    dependencies: Vec::new(),
                    offset,
                }),
        )
    }
}

impl CadPrimalPlan {
    /// Binds a concrete operator plan to exactly the CAD-associated mesh from
    /// which it was realized.
    pub fn new(
        geometry: CadGeometryRealization,
        mut boundary_conditions: Vec<CadBoundaryCondition>,
        realization: RealizationPlan,
    ) -> Result<Self, FinitumError> {
        let artifact = realization.artifact();
        if &artifact.mesh != geometry.mesh() {
            return Err(FinitumError::InvalidCadGeometry(
                "operator mesh differs from the CAD-associated mesh".into(),
            ));
        }
        boundary_conditions.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
        let expected_dofs = geometry.nodal_dof_map()?;
        if artifact.dofs != expected_dofs {
            return Err(FinitumError::InvalidCadGeometry(
                "operator DOF map is not the canonical CAD nodal map".into(),
            ));
        }
        let expected_constraints = geometry.essential_constraints(&boundary_conditions)?;
        if artifact.constraints != expected_constraints {
            return Err(FinitumError::InvalidCadGeometry(
                "operator constraints do not match the retained CAD boundary conditions".into(),
            ));
        }
        #[derive(Serialize)]
        struct DigestPayload<'a> {
            schema: &'static str,
            geometry_digest: [u8; 32],
            boundary_conditions: &'a [CadBoundaryCondition],
            realization_digest: &'a scientia::Digest,
        }
        let payload = DigestPayload {
            schema: "finitum.cad-primal-plan/v0",
            geometry_digest: geometry.digest(),
            boundary_conditions: &boundary_conditions,
            realization_digest: realization.digest(),
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| FinitumError::InvalidCadGeometry(error.to_string()))?;
        Ok(Self {
            geometry,
            boundary_conditions,
            realization,
            digest: scientia::Digest::blake3(&bytes),
        })
    }

    /// CAD geometry and association record retained by the operator plan.
    pub fn geometry(&self) -> &CadGeometryRealization {
        &self.geometry
    }

    /// Underlying Finitum operator realization.
    pub fn realization(&self) -> &RealizationPlan {
        &self.realization
    }

    /// Canonically ordered CAD boundary conditions whose exact constraint
    /// projection was validated at construction.
    pub fn boundary_conditions(&self) -> &[CadBoundaryCondition] {
        &self.boundary_conditions
    }

    /// Composite identity covering both operator and CAD association digests.
    pub fn digest(&self) -> &scientia::Digest {
        &self.digest
    }

    /// Rechecks the provider revision before reuse.
    pub fn require_rectangle_source(
        &self,
        provider: &RectangleProvider,
    ) -> Result<(), FinitumError> {
        self.geometry.require_rectangle_source(provider)
    }

    /// Matrix-free primal operator.
    pub fn matrix_free(&self) -> MatrixFreeOperator {
        self.realization.matrix_free()
    }

    /// Canonical assembled primal operator.
    pub fn assemble(&self) -> Result<AssembledOperator, FinitumError> {
        self.realization.assemble()
    }

    /// Generated primal load vector with CAD-selected essential lifting.
    pub fn load_vector(&self) -> Result<Vec<f64>, FinitumError> {
        self.realization.load_vector()
    }
}

fn validate_xy_carrier(axes: [[f64; 3]; 3]) -> Result<(), FinitumError> {
    let tolerance = 64.0 * f64::EPSILON;
    if axes[0][2].abs() > tolerance
        || axes[1][2].abs() > tolerance
        || axes[2][0].abs() > tolerance
        || axes[2][1].abs() > tolerance
        || (axes[2][2].abs() - 1.0).abs() > tolerance
    {
        return Err(FinitumError::InvalidCadGeometry(
            "R3P supports affine rectangles in the XY carrier only".into(),
        ));
    }
    Ok(())
}

fn rectangle_boundaries(
    ids: [StableId; 4],
    width: usize,
    u_cells: usize,
    v_cells: usize,
) -> Vec<CadBoundaryAssociation> {
    let bottom = (0..=u_cells).map(VertexId).collect();
    let right = (0..=v_cells)
        .map(|row| VertexId(row * width + u_cells))
        .collect();
    let top = (0..=u_cells)
        .map(|column| VertexId(v_cells * width + column))
        .collect();
    let left = (0..=v_cells).map(|row| VertexId(row * width)).collect();
    ids.into_iter()
        .zip([bottom, right, top, left])
        .map(|(id, vertices)| CadBoundaryAssociation {
            entity_id: id.as_str().to_owned(),
            vertices,
        })
        .collect()
}

#[derive(Serialize)]
struct AssociationDigestPayload<'a> {
    schema: &'static str,
    source: &'a CadGeometrySource,
    parameters: &'a [CadParameterCoordinate],
    mesh: &'a Mesh,
    nodes: &'a [CadNodeAssociation],
    cells: &'a [CadCellAssociation],
    boundaries: &'a [CadBoundaryAssociation],
}

fn association_digest(
    source: &CadGeometrySource,
    parameters: &[CadParameterCoordinate],
    mesh: &Mesh,
    nodes: &[CadNodeAssociation],
    cells: &[CadCellAssociation],
    boundaries: &[CadBoundaryAssociation],
) -> Result<[u8; 32], FinitumError> {
    let payload = AssociationDigestPayload {
        schema: "finitum.cad-primal-realization/v0",
        source,
        parameters,
        mesh,
        nodes,
        cells,
        boundaries,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| FinitumError::InvalidCadGeometry(error.to_string()))?;
    Ok(*blake3::hash(&bytes).as_bytes())
}
