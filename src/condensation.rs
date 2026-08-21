use crate::FinitumError;
use std::collections::BTreeSet;

/// Schur-complement result for one element-local system.
#[derive(Clone, Debug, PartialEq)]
pub struct CondensedLocalSystem {
    size: usize,
    pub interior_dofs: Vec<usize>,
    pub trace_dofs: Vec<usize>,
    /// Row-major trace-by-trace Schur complement.
    pub schur: Vec<f64>,
    pub rhs: Vec<f64>,
    interior_response: Vec<f64>,
    interior_offset: Vec<f64>,
}

impl CondensedLocalSystem {
    /// Recover the complete element vector after solving the condensed trace system.
    pub fn recover(&self, trace_values: &[f64]) -> Result<Vec<f64>, FinitumError> {
        if trace_values.len() != self.trace_dofs.len()
            || trace_values.iter().any(|value| !value.is_finite())
        {
            return Err(FinitumError::InvalidRealization(format!(
                "trace solution must contain {} finite values",
                self.trace_dofs.len()
            )));
        }
        let mut complete = vec![0.0; self.size];
        for (local, dof) in self.trace_dofs.iter().enumerate() {
            complete[*dof] = trace_values[local];
        }
        for (interior, dof) in self.interior_dofs.iter().enumerate() {
            complete[*dof] = self.interior_offset[interior]
                - (0..self.trace_dofs.len())
                    .map(|trace| {
                        self.interior_response[interior * self.trace_dofs.len() + trace]
                            * trace_values[trace]
                    })
                    .sum::<f64>();
        }
        Ok(complete)
    }
}

/// Eliminate element-interior unknowns while retaining a trace Schur system and recovery map.
pub fn static_condense(
    size: usize,
    matrix: &[f64],
    rhs: &[f64],
    interior_dofs: &[usize],
) -> Result<CondensedLocalSystem, FinitumError> {
    if size == 0
        || matrix.len() != size * size
        || rhs.len() != size
        || matrix.iter().chain(rhs).any(|value| !value.is_finite())
    {
        return Err(FinitumError::InvalidRealization(
            "local condensation requires a finite square matrix and matching right-hand side"
                .into(),
        ));
    }
    let mut distinct = BTreeSet::new();
    for dof in interior_dofs {
        if *dof >= size || !distinct.insert(*dof) {
            return Err(FinitumError::InvalidRealization(format!(
                "interior condensation index {dof} is invalid or repeated"
            )));
        }
    }
    if interior_dofs.is_empty() || interior_dofs.len() == size {
        return Err(FinitumError::InvalidRealization(
            "static condensation requires both interior and trace unknowns".into(),
        ));
    }
    let interior_dofs = interior_dofs.to_vec();
    let trace_dofs = (0..size)
        .filter(|dof| !distinct.contains(dof))
        .collect::<Vec<_>>();
    let ni = interior_dofs.len();
    let nt = trace_dofs.len();
    let aii = submatrix(matrix, size, &interior_dofs, &interior_dofs);
    let ait = submatrix(matrix, size, &interior_dofs, &trace_dofs);
    let ati = submatrix(matrix, size, &trace_dofs, &interior_dofs);
    let att = submatrix(matrix, size, &trace_dofs, &trace_dofs);
    let mut right = vec![0.0; ni * (nt + 1)];
    for row in 0..ni {
        right[row * (nt + 1)..row * (nt + 1) + nt].copy_from_slice(&ait[row * nt..row * nt + nt]);
        right[row * (nt + 1) + nt] = rhs[interior_dofs[row]];
    }
    solve_dense(aii, ni, &mut right, nt + 1)?;
    let mut interior_response = vec![0.0; ni * nt];
    let mut interior_offset = vec![0.0; ni];
    for row in 0..ni {
        interior_response[row * nt..row * nt + nt]
            .copy_from_slice(&right[row * (nt + 1)..row * (nt + 1) + nt]);
        interior_offset[row] = right[row * (nt + 1) + nt];
    }
    let mut schur = att;
    for row in 0..nt {
        for column in 0..nt {
            schur[row * nt + column] -= (0..ni)
                .map(|inner| ati[row * ni + inner] * interior_response[inner * nt + column])
                .sum::<f64>();
        }
    }
    let mut condensed_rhs = trace_dofs.iter().map(|dof| rhs[*dof]).collect::<Vec<_>>();
    for row in 0..nt {
        condensed_rhs[row] -= (0..ni)
            .map(|inner| ati[row * ni + inner] * interior_offset[inner])
            .sum::<f64>();
    }
    Ok(CondensedLocalSystem {
        size,
        interior_dofs,
        trace_dofs,
        schur,
        rhs: condensed_rhs,
        interior_response,
        interior_offset,
    })
}

fn submatrix(matrix: &[f64], size: usize, rows: &[usize], columns: &[usize]) -> Vec<f64> {
    rows.iter()
        .flat_map(|row| {
            columns
                .iter()
                .map(move |column| matrix[row * size + column])
        })
        .collect()
}

fn solve_dense(
    mut matrix: Vec<f64>,
    size: usize,
    rhs: &mut [f64],
    rhs_columns: usize,
) -> Result<(), FinitumError> {
    for pivot in 0..size {
        let selected = (pivot..size)
            .max_by(|left, right| {
                matrix[*left * size + pivot]
                    .abs()
                    .total_cmp(&matrix[*right * size + pivot].abs())
            })
            .expect("nonempty pivot range");
        if matrix[selected * size + pivot].abs() <= f64::EPSILON {
            return Err(FinitumError::InvalidRealization(
                "interior block is singular during static condensation".into(),
            ));
        }
        if selected != pivot {
            for column in 0..size {
                matrix.swap(pivot * size + column, selected * size + column);
            }
            for column in 0..rhs_columns {
                rhs.swap(
                    pivot * rhs_columns + column,
                    selected * rhs_columns + column,
                );
            }
        }
        let diagonal = matrix[pivot * size + pivot];
        for column in pivot..size {
            matrix[pivot * size + column] /= diagonal;
        }
        for column in 0..rhs_columns {
            rhs[pivot * rhs_columns + column] /= diagonal;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = matrix[row * size + pivot];
            for column in pivot..size {
                matrix[row * size + column] -= factor * matrix[pivot * size + column];
            }
            for column in 0..rhs_columns {
                rhs[row * rhs_columns + column] -= factor * rhs[pivot * rhs_columns + column];
            }
        }
    }
    Ok(())
}
