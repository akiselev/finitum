use crate::FinitumError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VertexId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CellId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub vertices: Vec<VertexId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    dimension: usize,
    vertices: Vec<Vec<f64>>,
    cells: Vec<Cell>,
}

impl Mesh {
    pub fn new(
        dimension: usize,
        vertices: Vec<Vec<f64>>,
        cells: Vec<Cell>,
    ) -> Result<Self, FinitumError> {
        if !(1..=3).contains(&dimension) {
            return Err(FinitumError::InvalidDimension(dimension));
        }
        for (vertex, coordinates) in vertices.iter().enumerate() {
            if coordinates.len() != dimension {
                return Err(FinitumError::CoordinateDimension {
                    vertex,
                    actual: coordinates.len(),
                    expected: dimension,
                });
            }
        }
        let expected = dimension + 1;
        for (cell_id, cell) in cells.iter().enumerate() {
            if cell.vertices.len() != expected {
                return Err(FinitumError::CellArity {
                    cell: cell_id,
                    actual: cell.vertices.len(),
                    expected,
                });
            }
            for vertex in &cell.vertices {
                if vertex.0 >= vertices.len() {
                    return Err(FinitumError::MissingVertex {
                        cell: cell_id,
                        vertex: vertex.0,
                    });
                }
            }
        }
        Ok(Self {
            dimension,
            vertices,
            cells,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn vertices(&self) -> &[Vec<f64>] {
        &self.vertices
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn cell(&self, id: CellId) -> Option<&Cell> {
        self.cells.get(id.0)
    }
}
