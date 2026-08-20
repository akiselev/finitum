use crate::FinitumError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealizationKind {
    Assembled,
    Element,
    Partial,
    MatrixFree,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CsrMatrix {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    pub fn new(
        rows: usize,
        columns: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, FinitumError> {
        if row_offsets.len() != rows + 1
            || row_offsets.first() != Some(&0)
            || row_offsets.last() != Some(&values.len())
            || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(FinitumError::InvalidRowOffsets);
        }
        if column_indices.len() != values.len() {
            return Err(FinitumError::InvalidCsrStorage);
        }
        if let Some(column) = column_indices
            .iter()
            .copied()
            .find(|column| *column >= columns)
        {
            return Err(FinitumError::InvalidColumn { column, columns });
        }
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices,
            values,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        if input.len() != self.columns {
            return Err(FinitumError::InputLength {
                actual: input.len(),
                expected: self.columns,
            });
        }
        if output.len() != self.rows {
            return Err(FinitumError::InputLength {
                actual: output.len(),
                expected: self.rows,
            });
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = (self.row_offsets[row]..self.row_offsets[row + 1])
                .map(|entry| self.values[entry] * input[self.column_indices[entry]])
                .sum();
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssembledOperator {
    matrix: CsrMatrix,
}

impl AssembledOperator {
    pub fn new(matrix: CsrMatrix) -> Self {
        Self { matrix }
    }

    pub fn matrix(&self) -> &CsrMatrix {
        &self.matrix
    }

    pub fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), FinitumError> {
        self.matrix.apply(input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembled_action_is_deterministic() {
        let matrix = CsrMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, -1.0, -1.0, 2.0],
        )
        .unwrap();
        let mut output = [0.0; 2];
        matrix.apply(&[1.0, 3.0], &mut output).unwrap();
        assert_eq!(output, [-1.0, 5.0]);
    }
}
