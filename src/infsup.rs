//! Runtime inf-sup (Ladyzhenskaya-Babuska-Brezzi) checker for realized mixed pairs (W7 package
//! 3; SV2-B4 evidence for `@inf_sup` obligations).
//!
//! For a realized saddle-point operator
//!
//! ```text
//! [ A   B^T ] [ v ]     A: constrained-field block (SPD on the free DOFs),
//! [ B   0   ] [ q ]     B: multiplier-row, constrained-column coupling block,
//! ```
//!
//! the discrete inf-sup constant in the `A`-energy norm on the constrained space and a
//! caller-chosen norm `M_Q` on the multiplier space is
//!
//! ```text
//! beta_h = inf_q sup_v (q^T B v) / (|v|_A |q|_{M_Q}) = sqrt(lambda_min(M_Q^{-1} B A^{-1} B^T))
//! ```
//!
//! taken over the multiplier modes outside the *declared* kernel (a pure-Dirichlet Stokes
//! pairing legitimately carries one constant-pressure mode; the caller declares it). Every
//! further zero eigenvalue is a spurious multiplier mode -- the discrete signature of an
//! unstable pairing (P1-P1 checkerboards, rank-deficient counts), reported deterministically as
//! [`InfSupInstability::SpuriousModes`] and refused by [`require_inf_sup_stable`].
//!
//! The estimate is computed densely (unit-column probing of the operator, dense Cholesky and a
//! cyclic Jacobi eigenvalue sweep): it is a fixture-grade *checker*, bounded by
//! [`INF_SUP_DIMENSION_CAP`], not a production eigensolver. It names no physics: the pairing is
//! either supplied explicitly or derived structurally from Scientia's `OperatorStructure`
//! ([`InfSupPairing::from_structure`]).

use crate::{BlockLayout, ConstraintSet, DofId, FinitumError};
use methodus::{EvaluationContext, LinearOperator};
use scientia::{Digest, OperatorStructure, SymbolId};
use serde::Serialize;

/// Schema of [`InfSupEstimate`].
pub const INF_SUP_REPORT_SCHEMA: &str = "finitum-inf-sup-report/1";

/// Largest operator dimension the dense estimate is attempted for.
pub const INF_SUP_DIMENSION_CAP: usize = 2048;

/// The two fields of a mixed pair: the field the multiplier constrains (velocity, flux) and the
/// multiplier field itself (pressure).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct InfSupPairing {
    pub constrained: SymbolId,
    pub multiplier: SymbolId,
}

impl InfSupPairing {
    pub fn new(constrained: SymbolId, multiplier: SymbolId) -> Result<Self, FinitumError> {
        if constrained == multiplier {
            return Err(FinitumError::InvalidRealization(format!(
                "inf-sup pairing needs two distinct fields, got {constrained} twice"
            )));
        }
        Ok(Self {
            constrained,
            multiplier,
        })
    }

    /// Derives the pairing from Scientia's structural `OperatorStructure` (C5.4) without any
    /// physics-name dispatch: the multiplier is the one field whose diagonal block is absent
    /// (the structure's own `saddle_point` rule), and the constrained field is the one field
    /// with a present diagonal block that the multiplier couples to through a present
    /// off-diagonal block in either direction. Anything else (no saddle point, several
    /// multipliers, a multiplier coupled to several fields, a multiplier coupled to nothing) is
    /// refused typed rather than guessed.
    ///
    /// Recorded need: Scientia's `VerificationObligationKind::InfSup { pair }` carries only a
    /// display string (`"Taylor-Hood"`, `"RT0-P0"`), not the two `SymbolId`s; a typed
    /// `InfSup { pair, constrained, multiplier }` would let a case bind this pairing directly
    /// instead of re-deriving it here.
    pub fn from_structure(structure: &OperatorStructure) -> Result<Self, FinitumError> {
        let present_diagonal = |field: SymbolId| {
            structure
                .blocks
                .iter()
                .any(|block| block.row == field && block.column == field && block.present)
        };
        let mut fields = Vec::new();
        for block in &structure.blocks {
            for symbol in [block.row, block.column] {
                if !fields.contains(&symbol) {
                    fields.push(symbol);
                }
            }
        }
        let multipliers = fields
            .iter()
            .copied()
            .filter(|&field| !present_diagonal(field))
            .collect::<Vec<_>>();
        let [multiplier] = multipliers[..] else {
            return Err(FinitumError::UnsupportedRealization(format!(
                "inf-sup pairing derivation needs exactly one field without a diagonal block \
                 (a saddle-point multiplier); structure `{}` has {} such field(s)",
                structure.model,
                multipliers.len()
            )));
        };
        let mut constrained = Vec::new();
        for block in &structure.blocks {
            if !block.present || block.row == block.column {
                continue;
            }
            let other = if block.row == multiplier {
                block.column
            } else if block.column == multiplier {
                block.row
            } else {
                continue;
            };
            if present_diagonal(other) && !constrained.contains(&other) {
                constrained.push(other);
            }
        }
        let [constrained] = constrained[..] else {
            return Err(FinitumError::UnsupportedRealization(format!(
                "inf-sup pairing derivation needs the multiplier {multiplier} coupled to exactly \
                 one field with a diagonal block; structure `{}` couples it to {}",
                structure.model,
                constrained.len()
            )));
        };
        Self::new(constrained, multiplier)
    }
}

/// The norm on the multiplier space the estimate is taken in.
#[derive(Clone, Debug, PartialEq)]
pub enum InfSupNorm {
    /// The Euclidean norm of the multiplier coefficient vector.
    Euclidean,
    /// A dense, row-major, symmetric positive definite Gram matrix over the multiplier
    /// block's *full* extent (constrained multiplier DOFs are removed by the estimate).
    /// [`crate::SystemOperator::mass_matrix`] produces the L2 mass matrix.
    Gram(Vec<f64>),
}

/// Tolerances and declarations for one estimate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct InfSupConfig {
    /// A generalized eigenvalue at or below `kernel_tolerance * lambda_max` counts as a
    /// kernel mode.
    pub kernel_tolerance: f64,
    /// Kernel dimension the caller declares legitimate (one constant multiplier mode under
    /// pure essential constraints on the constrained field; zero for a full-rank pairing).
    pub declared_kernel_dimension: usize,
    /// The estimate is unstable when the constant falls at or below this value.
    pub minimum_constant: f64,
}

impl Default for InfSupConfig {
    fn default() -> Self {
        Self {
            kernel_tolerance: 1.0e-10,
            declared_kernel_dimension: 0,
            minimum_constant: 0.0,
        }
    }
}

/// Why a pairing was judged unstable.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InfSupInstability {
    /// More kernel modes than declared: every extra one is a spurious multiplier mode.
    SpuriousModes { observed: usize, declared: usize },
    /// Every multiplier mode lies in the kernel (or the constrained space is empty).
    DegenerateMultiplierSpace,
    /// The constant is positive but at or below the configured minimum.
    BelowMinimum { constant: f64, minimum: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InfSupVerdict {
    Stable,
    Unstable(InfSupInstability),
}

#[derive(Serialize)]
struct EstimateBody<'a> {
    schema: &'static str,
    pairing: InfSupPairing,
    constrained_dimension: usize,
    multiplier_dimension: usize,
    count_deficit: usize,
    eigenvalues: &'a [f64],
    kernel_dimension: usize,
    declared_kernel_dimension: usize,
    spurious_mode_count: usize,
    inf_sup_constant: Option<f64>,
    verdict: &'a InfSupVerdict,
    config: InfSupConfig,
}

/// One inf-sup estimate; a serialized, digest-identified record.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InfSupEstimate {
    pub schema: String,
    pub pairing: InfSupPairing,
    /// Free (unconstrained) constrained-field DOFs.
    pub constrained_dimension: usize,
    /// Free multiplier DOFs.
    pub multiplier_dimension: usize,
    /// `max(multiplier_dimension - constrained_dimension, 0)`: a lower bound on the kernel
    /// dimension by counting alone.
    pub count_deficit: usize,
    /// Generalized eigenvalues of `M_Q^{-1} B A^{-1} B^T`, ascending.
    pub eigenvalues: Vec<f64>,
    pub kernel_dimension: usize,
    pub declared_kernel_dimension: usize,
    pub spurious_mode_count: usize,
    /// `sqrt` of the smallest eigenvalue beyond the declared kernel, when one exists.
    pub inf_sup_constant: Option<f64>,
    pub verdict: InfSupVerdict,
    pub config: InfSupConfig,
    pub identity: Digest,
}

impl InfSupEstimate {
    pub fn is_stable(&self) -> bool {
        matches!(self.verdict, InfSupVerdict::Stable)
    }
}

/// Deterministic refusal of an unstable pairing: `Ok(())` exactly when the estimate is
/// [`InfSupVerdict::Stable`], otherwise [`FinitumError::InfSupUnstable`] naming the cause.
pub fn require_inf_sup_stable(estimate: &InfSupEstimate) -> Result<(), FinitumError> {
    match &estimate.verdict {
        InfSupVerdict::Stable => Ok(()),
        InfSupVerdict::Unstable(instability) => {
            let cause = match instability {
                InfSupInstability::SpuriousModes { observed, declared } => format!(
                    "{} spurious multiplier mode(s): kernel dimension {observed} exceeds the \
                     declared {declared}",
                    observed - declared
                ),
                InfSupInstability::DegenerateMultiplierSpace => {
                    "every multiplier mode is in the kernel".to_string()
                }
                InfSupInstability::BelowMinimum { constant, minimum } => {
                    format!("inf-sup constant {constant} is at or below the minimum {minimum}")
                }
            };
            Err(FinitumError::InfSupUnstable(format!(
                "pairing (constrained {}, multiplier {}): {cause} (constrained free DOFs {}, \
                 multiplier free DOFs {})",
                estimate.pairing.constrained,
                estimate.pairing.multiplier,
                estimate.constrained_dimension,
                estimate.multiplier_dimension
            )))
        }
    }
}

/// Estimates the inf-sup constant of `pairing` in `operator` (the *linear* action over
/// `layout`; a `ReducedSystemOperator`/`ReducedMixedOperator` is admitted as-is, or pass the
/// physical operator with `constraints` and the constrained DOFs are removed here -- both give
/// the same free-DOF blocks). Refuses typed: affine (dependency-carrying) constraints, an
/// operator above [`INF_SUP_DIMENSION_CAP`], a non-symmetric or non-positive-definite
/// constrained block (the energy norm needs SPD), a nonzero multiplier diagonal block (a
/// stabilized pairing is not judged by this constraint-only estimate), or a Gram norm of the
/// wrong extent.
pub fn estimate_inf_sup(
    operator: &dyn LinearOperator,
    layout: &BlockLayout,
    constraints: Option<&ConstraintSet>,
    pairing: InfSupPairing,
    multiplier_norm: &InfSupNorm,
    config: &InfSupConfig,
) -> Result<InfSupEstimate, FinitumError> {
    validate_config(config)?;
    let dimension = layout.extent();
    if operator.rows() != dimension || operator.columns() != dimension {
        return Err(FinitumError::InvalidRealization(format!(
            "inf-sup operator is {}x{}, layout extent is {dimension}",
            operator.rows(),
            operator.columns()
        )));
    }
    if dimension > INF_SUP_DIMENSION_CAP {
        return Err(FinitumError::UnsupportedRealization(format!(
            "inf-sup estimate is refused above {INF_SUP_DIMENSION_CAP} degrees of freedom \
             (operator has {dimension})"
        )));
    }
    let constrained_block = layout.block(pairing.constrained).ok_or_else(|| {
        FinitumError::InvalidRealization(format!(
            "layout has no block for the constrained field {}",
            pairing.constrained
        ))
    })?;
    let multiplier_block = layout.block(pairing.multiplier).ok_or_else(|| {
        FinitumError::InvalidRealization(format!(
            "layout has no block for the multiplier field {}",
            pairing.multiplier
        ))
    })?;
    if let Some(constraints) = constraints {
        if constraints.dof_count() != dimension {
            return Err(FinitumError::InvalidRealization(format!(
                "constraint set has {} degrees of freedom, layout has {dimension}",
                constraints.dof_count()
            )));
        }
        if let Some(constraint) = constraints
            .constraints()
            .find(|constraint| !constraint.dependencies.is_empty())
        {
            return Err(FinitumError::UnsupportedRealization(format!(
                "inf-sup estimate admits essential (dependency-free) constraints only; \
                 degree of freedom {} carries an affine dependency",
                constraint.target.0
            )));
        }
    }
    let free = |offset: usize, extent: usize| -> Vec<usize> {
        (offset..offset + extent)
            .filter(|&dof| constraints.is_none_or(|set| !set.is_constrained(DofId(dof))))
            .collect()
    };
    let v_index = free(constrained_block.offset, constrained_block.extent);
    let q_index = free(multiplier_block.offset, multiplier_block.extent);
    let n_v = v_index.len();
    let n_q = q_index.len();
    let count_deficit = n_q.saturating_sub(n_v);

    let matrix = probe_dense(operator, dimension)?;
    let a = submatrix(&matrix, dimension, &v_index, &v_index);
    let b = submatrix(&matrix, dimension, &q_index, &v_index);
    let c = submatrix(&matrix, dimension, &q_index, &q_index);
    let scale = max_abs(&a).max(max_abs(&b)).max(1.0e-300);
    if max_abs(&c) > 1.0e-12 * scale {
        return Err(FinitumError::UnsupportedRealization(format!(
            "multiplier field {} has a nonzero diagonal block (max |entry| {:e}); the \
             constraint-only inf-sup estimate does not judge stabilized pairings",
            pairing.multiplier,
            max_abs(&c)
        )));
    }
    if !is_symmetric(&a, n_v, 1.0e-10 * scale) {
        return Err(FinitumError::UnsupportedRealization(format!(
            "constrained field {} block is not symmetric on its free DOFs; the energy-norm \
             inf-sup estimate needs a symmetric positive definite block",
            pairing.constrained
        )));
    }
    let a_factor = cholesky(&a, n_v).ok_or_else(|| {
        FinitumError::UnsupportedRealization(format!(
            "constrained field {} block is not positive definite on its free DOFs; the \
             energy-norm inf-sup estimate needs a symmetric positive definite block",
            pairing.constrained
        ))
    })?;

    // S = B A^{-1} B^T, symmetrized.
    let mut schur = vec![0.0; n_q * n_q];
    let mut column = vec![0.0; n_v];
    for j in 0..n_q {
        for (i, value) in column.iter_mut().enumerate() {
            *value = b[j * n_v + i];
        }
        let solved = cholesky_solve(&a_factor, n_v, &column);
        for i in 0..n_q {
            let mut sum = 0.0;
            for k in 0..n_v {
                sum += b[i * n_v + k] * solved[k];
            }
            schur[i * n_q + j] = sum;
        }
    }
    for i in 0..n_q {
        for j in 0..i {
            let mean = 0.5 * (schur[i * n_q + j] + schur[j * n_q + i]);
            schur[i * n_q + j] = mean;
            schur[j * n_q + i] = mean;
        }
    }

    let reduced = match multiplier_norm {
        InfSupNorm::Euclidean => schur,
        InfSupNorm::Gram(gram) => {
            let extent = multiplier_block.extent;
            if gram.len() != extent * extent {
                return Err(FinitumError::InvalidRealization(format!(
                    "multiplier Gram norm has {} entries, expected {extent}x{extent}",
                    gram.len()
                )));
            }
            if gram.iter().any(|value| !value.is_finite()) {
                return Err(FinitumError::InvalidRealization(
                    "multiplier Gram norm contains non-finite entries".into(),
                ));
            }
            let local = q_index
                .iter()
                .map(|dof| dof - multiplier_block.offset)
                .collect::<Vec<_>>();
            let gram_free = submatrix(gram, extent, &local, &local);
            if !is_symmetric(&gram_free, n_q, 1.0e-10 * max_abs(&gram_free).max(1.0e-300)) {
                return Err(FinitumError::InvalidRealization(
                    "multiplier Gram norm is not symmetric".into(),
                ));
            }
            let factor = cholesky(&gram_free, n_q).ok_or_else(|| {
                FinitumError::InvalidRealization(
                    "multiplier Gram norm is not positive definite on the free DOFs".into(),
                )
            })?;
            congruence(&schur, &factor, n_q)
        }
    };
    let mut eigenvalues = symmetric_eigenvalues(&reduced, n_q);
    eigenvalues.sort_by(|left, right| left.total_cmp(right));
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(FinitumError::InvalidRealization(
            "inf-sup eigenvalue sweep produced non-finite values".into(),
        ));
    }
    let largest = eigenvalues.last().copied().unwrap_or(0.0).max(0.0);
    let kernel_dimension = if largest <= 0.0 {
        eigenvalues.len()
    } else {
        eigenvalues
            .iter()
            .filter(|&&value| value <= config.kernel_tolerance * largest)
            .count()
    };
    let declared = config.declared_kernel_dimension;
    let spurious_mode_count = kernel_dimension.saturating_sub(declared);
    let inf_sup_constant = if kernel_dimension > declared {
        None
    } else {
        eigenvalues.get(declared).map(|value| value.max(0.0).sqrt())
    };
    let verdict = if spurious_mode_count > 0 {
        InfSupVerdict::Unstable(InfSupInstability::SpuriousModes {
            observed: kernel_dimension,
            declared,
        })
    } else {
        match inf_sup_constant {
            None => InfSupVerdict::Unstable(InfSupInstability::DegenerateMultiplierSpace),
            Some(constant) if constant <= config.minimum_constant => {
                InfSupVerdict::Unstable(InfSupInstability::BelowMinimum {
                    constant,
                    minimum: config.minimum_constant,
                })
            }
            Some(_) => InfSupVerdict::Stable,
        }
    };
    let body = EstimateBody {
        schema: INF_SUP_REPORT_SCHEMA,
        pairing,
        constrained_dimension: n_v,
        multiplier_dimension: n_q,
        count_deficit,
        eigenvalues: &eigenvalues,
        kernel_dimension,
        declared_kernel_dimension: declared,
        spurious_mode_count,
        inf_sup_constant,
        verdict: &verdict,
        config: *config,
    };
    let bytes = serde_json::to_vec(&body).map_err(|error| {
        FinitumError::InvalidRealization(format!("inf-sup estimate serialization failed: {error}"))
    })?;
    let identity = Digest::blake3(&bytes);
    Ok(InfSupEstimate {
        schema: INF_SUP_REPORT_SCHEMA.into(),
        pairing,
        constrained_dimension: n_v,
        multiplier_dimension: n_q,
        count_deficit,
        eigenvalues,
        kernel_dimension,
        declared_kernel_dimension: declared,
        spurious_mode_count,
        inf_sup_constant,
        verdict,
        config: *config,
        identity,
    })
}

fn validate_config(config: &InfSupConfig) -> Result<(), FinitumError> {
    if !(config.kernel_tolerance.is_finite() && config.kernel_tolerance >= 0.0) {
        return Err(FinitumError::InvalidRealization(
            "inf-sup kernel tolerance must be finite and nonnegative".into(),
        ));
    }
    if !(config.minimum_constant.is_finite() && config.minimum_constant >= 0.0) {
        return Err(FinitumError::InvalidRealization(
            "inf-sup minimum constant must be finite and nonnegative".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Dense helpers (row-major, fixture-grade).
// ---------------------------------------------------------------------------------------------

fn probe_dense(operator: &dyn LinearOperator, dimension: usize) -> Result<Vec<f64>, FinitumError> {
    let context = EvaluationContext::default();
    let mut matrix = vec![0.0; dimension * dimension];
    let mut direction = vec![0.0; dimension];
    let mut output = vec![0.0; dimension];
    for column in 0..dimension {
        direction[column] = 1.0;
        operator
            .apply(&context, &direction, &mut output)
            .map_err(|error| FinitumError::Assembly(error.to_string()))?;
        for (row, value) in output.iter().enumerate() {
            if !value.is_finite() {
                return Err(FinitumError::Assembly(format!(
                    "operator action produced a non-finite entry at ({row}, {column})"
                )));
            }
            matrix[row * dimension + column] = *value;
        }
        direction[column] = 0.0;
    }
    Ok(matrix)
}

fn submatrix(matrix: &[f64], stride: usize, rows: &[usize], columns: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * columns.len());
    for &row in rows {
        for &column in columns {
            out.push(matrix[row * stride + column]);
        }
    }
    out
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn is_symmetric(matrix: &[f64], n: usize, tolerance: f64) -> bool {
    for i in 0..n {
        for j in 0..i {
            if (matrix[i * n + j] - matrix[j * n + i]).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Lower Cholesky factor `L` (row-major, `A = L L^T`), `None` when `A` is not positive
/// definite to working precision.
fn cholesky(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; n * n];
    let scale = max_abs(matrix).max(1.0e-300);
    for j in 0..n {
        let mut diagonal = matrix[j * n + j];
        for k in 0..j {
            diagonal -= lower[j * n + k] * lower[j * n + k];
        }
        if !(diagonal.is_finite() && diagonal > 1.0e-14 * scale) {
            return None;
        }
        let root = diagonal.sqrt();
        lower[j * n + j] = root;
        for i in j + 1..n {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            lower[i * n + j] = sum / root;
        }
    }
    Some(lower)
}

fn cholesky_solve(lower: &[f64], n: usize, rhs: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= lower[i * n + k] * y[k];
        }
        y[i] = sum / lower[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= lower[k * n + i] * x[k];
        }
        x[i] = sum / lower[i * n + i];
    }
    x
}

/// `L^{-1} S L^{-T}` for a lower factor `L`.
fn congruence(schur: &[f64], lower: &[f64], n: usize) -> Vec<f64> {
    // Y = L^{-1} S (solve L Y = S column by column).
    let mut y = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = schur[i * n + j];
            for k in 0..i {
                sum -= lower[i * n + k] * y[k * n + j];
            }
            y[i * n + j] = sum / lower[i * n + i];
        }
    }
    // C = Y L^{-T}: solve L C^T = Y^T, i.e. per row of Y solve L c = y_row.
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        let mut row = vec![0.0; n];
        for j in 0..n {
            let mut sum = y[i * n + j];
            for k in 0..j {
                sum -= lower[j * n + k] * row[k];
            }
            row[j] = sum / lower[j * n + j];
        }
        c[i * n..(i + 1) * n].copy_from_slice(&row);
    }
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (c[i * n + j] + c[j * n + i]);
            c[i * n + j] = mean;
            c[j * n + i] = mean;
        }
    }
    c
}

/// Eigenvalues of a dense symmetric matrix by cyclic Jacobi rotations.
fn symmetric_eigenvalues(matrix: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let mut a = matrix.to_vec();
    let frobenius = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    if frobenius == 0.0 {
        return vec![0.0; n];
    }
    for _sweep in 0..200 {
        let off = (0..n)
            .flat_map(|i| (0..i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum::<f64>()
            .sqrt();
        if off <= 1.0e-15 * frobenius {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1.0e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_recovers_known_spectrum() {
        // Symmetric 3x3 with eigenvalues 1, 2, 3 (diagonal after an orthogonal similarity).
        let matrix = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let mut values = symmetric_eigenvalues(&matrix, 3);
        values.sort_by(f64::total_cmp);
        let expected = [2.0 - 2.0_f64.sqrt(), 2.0, 2.0 + 2.0_f64.sqrt()];
        for (value, expected) in values.iter().zip(expected) {
            assert!((value - expected).abs() < 1.0e-12, "{value} != {expected}");
        }
    }

    #[test]
    fn cholesky_solve_inverts_spd_matrix() {
        let matrix = [4.0, 2.0, 2.0, 3.0];
        let factor = cholesky(&matrix, 2).unwrap();
        let solution = cholesky_solve(&factor, 2, &[1.0, 2.0]);
        assert!((4.0 * solution[0] + 2.0 * solution[1] - 1.0).abs() < 1.0e-14);
        assert!((2.0 * solution[0] + 3.0 * solution[1] - 2.0).abs() < 1.0e-14);
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    }

    #[test]
    fn congruence_matches_direct_product() {
        let schur = [2.0, 1.0, 1.0, 2.0];
        let gram = [4.0, 1.0, 1.0, 2.0];
        let factor = cholesky(&gram, 2).unwrap();
        let reduced = congruence(&schur, &factor, 2);
        // Eigenvalues of L^{-1} S L^{-T} equal those of the pencil (S, M): check via det.
        let mut values = symmetric_eigenvalues(&reduced, 2);
        values.sort_by(f64::total_cmp);
        for lambda in values {
            let m = [
                schur[0] - lambda * gram[0],
                schur[1] - lambda * gram[1],
                schur[2] - lambda * gram[2],
                schur[3] - lambda * gram[3],
            ];
            let determinant = m[0] * m[3] - m[1] * m[2];
            assert!(determinant.abs() < 1.0e-12, "det {determinant}");
        }
    }
}
