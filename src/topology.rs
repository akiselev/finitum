use crate::{CellId, DofId, FinitumError, Mesh, VertexId};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FacetId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FacetIncidence {
    pub cell: CellId,
    pub local_facet: usize,
    /// Sign relating the induced cell-boundary orientation to the canonical sorted facet.
    pub orientation: i8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshFacet {
    pub id: FacetId,
    pub vertices: Vec<VertexId>,
    pub incidences: Vec<FacetIncidence>,
}

impl MeshFacet {
    pub fn is_exterior(&self) -> bool {
        self.incidences.len() == 1
    }

    pub fn is_interior(&self) -> bool {
        self.incidences.len() == 2
    }

    pub fn minus(&self) -> FacetIncidence {
        self.incidences[0]
    }

    pub fn plus(&self) -> Option<FacetIncidence> {
        self.incidences.get(1).copied()
    }
}

/// Deterministic simplex facet traversal with explicit two-sided orientation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FacetTopology {
    facets: Vec<MeshFacet>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrientedFacetPair {
    pub facet: FacetId,
    pub minus: FacetIncidence,
    pub plus: FacetIncidence,
}

impl OrientedFacetPair {
    /// Product of the two cell-to-facet orientations. It is `-1` for consistently oriented
    /// neighboring cells, making the opposite-normal relation explicit to interface kernels.
    pub fn relative_orientation(self) -> i8 {
        self.minus.orientation * self.plus.orientation
    }
}

impl FacetTopology {
    pub fn from_mesh(mesh: &Mesh) -> Result<Self, FinitumError> {
        let mut by_vertices = BTreeMap::<Vec<usize>, usize>::new();
        let mut facets = Vec::<MeshFacet>::new();
        for (cell_index, cell) in mesh.cells().iter().enumerate() {
            for omitted in 0..cell.vertices.len() {
                let local = cell
                    .vertices
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != omitted)
                    .map(|(_, vertex)| vertex.0)
                    .collect::<Vec<_>>();
                let mut canonical = local.clone();
                canonical.sort_unstable();
                let orientation = alternating_sign(omitted) * permutation_sign(&local, &canonical);
                let index = if let Some(index) = by_vertices.get(&canonical).copied() {
                    index
                } else {
                    let index = facets.len();
                    by_vertices.insert(canonical.clone(), index);
                    facets.push(MeshFacet {
                        id: FacetId(index),
                        vertices: canonical.iter().copied().map(VertexId).collect(),
                        incidences: Vec::new(),
                    });
                    index
                };
                facets[index].incidences.push(FacetIncidence {
                    cell: CellId(cell_index),
                    local_facet: omitted,
                    orientation,
                });
                if facets[index].incidences.len() > 2 {
                    return Err(FinitumError::InvalidRealization(format!(
                        "facet {index} has more than two incident cells"
                    )));
                }
            }
        }
        for facet in &mut facets {
            facet.incidences.sort_by_key(|incidence| incidence.cell);
        }
        Ok(Self { facets })
    }

    pub fn facets(&self) -> &[MeshFacet] {
        &self.facets
    }

    pub fn interior(&self) -> impl Iterator<Item = &MeshFacet> {
        self.facets.iter().filter(|facet| facet.is_interior())
    }

    pub fn exterior(&self) -> impl Iterator<Item = &MeshFacet> {
        self.facets.iter().filter(|facet| facet.is_exterior())
    }

    /// Select an explicit minus cell for an interior/interface facet. Reversing `minus_cell`
    /// reverses the returned trace ordering without changing the canonical facet identity.
    pub fn oriented_pair(
        &self,
        facet: FacetId,
        minus_cell: CellId,
    ) -> Result<OrientedFacetPair, FinitumError> {
        let facet_data = self.facets.get(facet.0).ok_or_else(|| {
            FinitumError::InvalidRealization(format!("facet {} does not exist", facet.0))
        })?;
        if !facet_data.is_interior() {
            return Err(FinitumError::InvalidRealization(format!(
                "facet {} is not two-sided",
                facet.0
            )));
        }
        let minus = facet_data
            .incidences
            .iter()
            .find(|incidence| incidence.cell == minus_cell)
            .copied()
            .ok_or_else(|| {
                FinitumError::InvalidRealization(format!(
                    "cell {} is not incident to facet {}",
                    minus_cell.0, facet.0
                ))
            })?;
        let plus = facet_data
            .incidences
            .iter()
            .find(|incidence| incidence.cell != minus_cell)
            .copied()
            .expect("validated two-sided facet");
        Ok(OrientedFacetPair { facet, minus, plus })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrientedRestriction {
    pub dofs: Vec<DofId>,
    pub orientations: Vec<i8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompatibleDofMaps {
    pub hcurl_dof_count: usize,
    pub hcurl: Vec<OrientedRestriction>,
    pub hdiv_dof_count: usize,
    pub hdiv: Vec<OrientedRestriction>,
}

impl CompatibleDofMaps {
    pub fn simplex(mesh: &Mesh, facets: &FacetTopology) -> Result<Self, FinitumError> {
        if mesh.dimension() < 2 {
            return Err(FinitumError::UnsupportedRealization(
                "compatible edge/facet spaces require dimension two or three".into(),
            ));
        }
        let edges = mesh_edges(mesh);
        let edge_ids = edges
            .iter()
            .enumerate()
            .map(|(index, edge)| (edge.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let mut hcurl = Vec::with_capacity(mesh.cells().len());
        for cell in mesh.cells() {
            let mut dofs = Vec::new();
            let mut orientations = Vec::new();
            for left in 0..cell.vertices.len() {
                for right in left + 1..cell.vertices.len() {
                    let from = cell.vertices[left].0;
                    let to = cell.vertices[right].0;
                    let key = sorted_pair(from, to);
                    dofs.push(DofId(edge_ids[&key]));
                    orientations.push(if [from, to] == key.as_slice() { 1 } else { -1 });
                }
            }
            hcurl.push(OrientedRestriction { dofs, orientations });
        }
        let mut facet_ids = BTreeMap::new();
        for facet in facets.facets() {
            facet_ids.insert(
                facet
                    .vertices
                    .iter()
                    .map(|vertex| vertex.0)
                    .collect::<Vec<_>>(),
                facet.id.0,
            );
        }
        let mut hdiv = Vec::with_capacity(mesh.cells().len());
        for (cell_index, cell) in mesh.cells().iter().enumerate() {
            let mut dofs = Vec::new();
            let mut orientations = Vec::new();
            for omitted in 0..cell.vertices.len() {
                let mut key = cell
                    .vertices
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != omitted)
                    .map(|(_, vertex)| vertex.0)
                    .collect::<Vec<_>>();
                key.sort_unstable();
                let facet = &facets.facets()[facet_ids[&key]];
                let incidence = facet
                    .incidences
                    .iter()
                    .find(|incidence| incidence.cell == CellId(cell_index))
                    .expect("facet topology was built from the same mesh");
                dofs.push(DofId(facet.id.0));
                orientations.push(incidence.orientation);
            }
            hdiv.push(OrientedRestriction { dofs, orientations });
        }
        Ok(Self {
            hcurl_dof_count: edges.len(),
            hcurl,
            hdiv_dof_count: facets.facets().len(),
            hdiv,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SignedIncidence {
    rows: usize,
    columns: usize,
    values: Vec<i8>,
}

impl SignedIncidence {
    /// Construct a checked dense signed-incidence action.
    ///
    /// This validates the representation only. Topological exactness is a
    /// separate property checked by [`crate::check_exact_sequence`].
    pub fn new(rows: usize, columns: usize, values: Vec<i8>) -> Result<Self, FinitumError> {
        let expected = rows.checked_mul(columns).ok_or_else(|| {
            FinitumError::InvalidRealization("signed-incidence extent overflows usize".into())
        })?;
        if rows == 0 || columns == 0 || values.len() != expected {
            return Err(FinitumError::InvalidRealization(format!(
                "signed incidence needs {expected} entries for a nonempty {rows} by {columns} action, got {}",
                values.len()
            )));
        }
        if values.iter().any(|value| !matches!(value, -1..=1)) {
            return Err(FinitumError::InvalidRealization(
                "signed-incidence entries must be -1, 0, or 1".into(),
            ));
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn value(&self, row: usize, column: usize) -> Option<i8> {
        (row < self.rows && column < self.columns).then(|| self.values[row * self.columns + column])
    }

    pub fn product_is_zero(&self, right: &Self) -> bool {
        self.columns == right.rows
            && (0..self.rows).all(|row| {
                (0..right.columns).all(|column| {
                    (0..self.columns)
                        .map(|inner| {
                            i32::from(self.values[row * self.columns + inner])
                                * i32::from(right.values[inner * right.columns + column])
                        })
                        .sum::<i32>()
                        == 0
                })
            })
    }

    pub fn rank(&self) -> usize {
        let mut values = self
            .values
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let mut rank = 0;
        for column in 0..self.columns {
            let Some(pivot) = (rank..self.rows)
                .max_by(|left, right| {
                    values[*left * self.columns + column]
                        .abs()
                        .total_cmp(&values[*right * self.columns + column].abs())
                })
                .filter(|row| values[*row * self.columns + column].abs() > f64::EPSILON)
            else {
                continue;
            };
            for entry in 0..self.columns {
                values.swap(rank * self.columns + entry, pivot * self.columns + entry);
            }
            let diagonal = values[rank * self.columns + column];
            for entry in column..self.columns {
                values[rank * self.columns + entry] /= diagonal;
            }
            for row in 0..self.rows {
                if row == rank {
                    continue;
                }
                let factor = values[row * self.columns + column];
                for entry in column..self.columns {
                    values[row * self.columns + entry] -=
                        factor * values[rank * self.columns + entry];
                }
            }
            rank += 1;
            if rank == self.rows {
                break;
            }
        }
        rank
    }
}

/// Topological de Rham incidence sequence. Construction verifies boundary-of-boundary is zero.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExactSequence {
    pub gradient: SignedIncidence,
    pub curl: SignedIncidence,
    pub divergence: Option<SignedIncidence>,
}

impl ExactSequence {
    pub fn simplex(mesh: &Mesh, facets: &FacetTopology) -> Result<Self, FinitumError> {
        if !(2..=3).contains(&mesh.dimension()) {
            return Err(FinitumError::UnsupportedRealization(
                "exact-sequence incidence is implemented for triangles and tetrahedra".into(),
            ));
        }
        let edges = mesh_edges(mesh);
        let edge_ids = edges
            .iter()
            .enumerate()
            .map(|(index, edge)| (edge.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let mut gradient = SignedIncidence {
            rows: edges.len(),
            columns: mesh.vertices().len(),
            values: vec![0; edges.len() * mesh.vertices().len()],
        };
        for (row, edge) in edges.iter().enumerate() {
            gradient.values[row * gradient.columns + edge[0]] = -1;
            gradient.values[row * gradient.columns + edge[1]] = 1;
        }
        let (curl, divergence) = if mesh.dimension() == 2 {
            let mut curl = SignedIncidence {
                rows: mesh.cells().len(),
                columns: edges.len(),
                values: vec![0; mesh.cells().len() * edges.len()],
            };
            for facet in facets.facets() {
                for incidence in &facet.incidences {
                    curl.values[incidence.cell.0 * curl.columns
                        + edge_ids[&facet
                            .vertices
                            .iter()
                            .map(|vertex| vertex.0)
                            .collect::<Vec<_>>()]] = incidence.orientation;
                }
            }
            (curl, None)
        } else {
            let mut curl = SignedIncidence {
                rows: facets.facets().len(),
                columns: edges.len(),
                values: vec![0; facets.facets().len() * edges.len()],
            };
            for facet in facets.facets() {
                let vertices = facet
                    .vertices
                    .iter()
                    .map(|vertex| vertex.0)
                    .collect::<Vec<_>>();
                for omitted in 0..vertices.len() {
                    let local = vertices
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != omitted)
                        .map(|(_, vertex)| *vertex)
                        .collect::<Vec<_>>();
                    let key = sorted_pair(local[0], local[1]);
                    let direction = if local == key { 1 } else { -1 };
                    curl.values[facet.id.0 * curl.columns + edge_ids[&key]] =
                        alternating_sign(omitted) * direction;
                }
            }
            let mut divergence = SignedIncidence {
                rows: mesh.cells().len(),
                columns: facets.facets().len(),
                values: vec![0; mesh.cells().len() * facets.facets().len()],
            };
            for facet in facets.facets() {
                for incidence in &facet.incidences {
                    divergence.values[incidence.cell.0 * divergence.columns + facet.id.0] =
                        incidence.orientation;
                }
            }
            (curl, Some(divergence))
        };
        if !curl.product_is_zero(&gradient)
            || divergence
                .as_ref()
                .is_some_and(|divergence| !divergence.product_is_zero(&curl))
        {
            return Err(FinitumError::InvalidRealization(
                "simplex incidence violates the exact-sequence boundary identity".into(),
            ));
        }
        if gradient.rank() + curl.rank() != gradient.rows()
            || divergence
                .as_ref()
                .is_some_and(|divergence| curl.rank() + divergence.rank() != curl.rows())
        {
            return Err(FinitumError::InvalidRealization(
                "simplex incidence complex is not exact at an edge or facet space".into(),
            ));
        }
        Ok(Self {
            gradient,
            curl,
            divergence,
        })
    }
}

fn mesh_edges(mesh: &Mesh) -> Vec<Vec<usize>> {
    let mut edges = BTreeMap::new();
    for cell in mesh.cells() {
        for left in 0..cell.vertices.len() {
            for right in left + 1..cell.vertices.len() {
                edges.insert(
                    sorted_pair(cell.vertices[left].0, cell.vertices[right].0),
                    (),
                );
            }
        }
    }
    edges.into_keys().collect()
}

fn sorted_pair(left: usize, right: usize) -> Vec<usize> {
    if left < right {
        vec![left, right]
    } else {
        vec![right, left]
    }
}

fn alternating_sign(index: usize) -> i8 {
    if index % 2 == 0 { 1 } else { -1 }
}

fn permutation_sign(values: &[usize], sorted: &[usize]) -> i8 {
    let positions = sorted
        .iter()
        .enumerate()
        .map(|(index, value)| (*value, index))
        .collect::<BTreeMap<_, _>>();
    let permutation = values
        .iter()
        .map(|value| positions[value])
        .collect::<Vec<_>>();
    let inversions = (0..permutation.len())
        .flat_map(|left| (left + 1..permutation.len()).map(move |right| (left, right)))
        .filter(|(left, right)| permutation[*left] > permutation[*right])
        .count();
    alternating_sign(inversions)
}
