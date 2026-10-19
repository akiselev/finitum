//! SV2-C2b linear reconstruction for the cell-centred finite-volume realization (W12 lane L6).
//!
//! [`FvReconstruction`] stores, per cell, the unweighted linear least-squares gradient
//! `g_c = Σ_k w_k (u_k − u_c)`, `w_k = M_c⁻¹ d_k`, `M_c = Σ_k d_k d_kᵀ`, over the face
//! neighbours of the cell (periodic neighbours translated across the wrap), gathered in the
//! fixed face order of [`FvTopology`] (interior, then periodic). On a tensor mesh the
//! neighbours are axis aligned, `M_c` is diagonal and the gradient is the per-axis 1-D
//! least-squares slope (central difference on a uniform axis); at a non-periodic boundary it is
//! one-sided. A cell whose face neighbours do not span the space (a simplex corner with a
//! single neighbour) extends its stencil to its neighbours' face neighbours; a cell still
//! rank deficient gets a zero gradient and is counted by [`FvReconstruction::degenerate_cells`].
//! Face states are `u_f = u_c + σ_c·(x_f − x_c)`.
//!
//! Limiting:
//! - **Sweby** `φ(r)` (the Scientia `Limiter` kernel over its `SlopeRatio` operand, one batched
//!   `run_batch` over every cell-axis pair), tensor meshes only: `σ_a = φ(r)·s⁻` with
//!   `s⁻ = (u_c − u_back)/h⁻`, `s⁺ = (u_fwd − u_c)/h⁺` and `r = s⁺/s⁻`, `r = 0` when `s⁻ = 0`
//!   (every Sweby limiter then gives zero slope). A cell lacking either axis neighbour (a
//!   non-periodic boundary) has zero slope along that axis, i.e. first order there.
//! - **Barth–Jespersen** and **Venkatakrishnan** (any mesh) scale the least-squares gradient by
//!   the smallest factor over the cell's faces keeping each face state inside the bounds of the
//!   cell and its face neighbours; Venkatakrishnan uses `ε² = (k·|Ω_c|^{1/d})³` and is clamped
//!   to 1. These are algorithmic multidimensional limiters, not named physics.
//!
//! Only the unlimited reconstruction is differentiated (it is linear: exact JVP and transpose).
//! Every limited configuration is a typed derivative refusal (decision D6); Sweby kernels are
//! checked through `malleus::ad` with [`NonsmoothPolicy::Refuse`].

use std::sync::Arc;

use malleus::ad::{DerivativeOptions, NonsmoothPolicy, differentiate_with};
use malleus::{DerivativeMode, DerivativeRequest, DifferentiationError};
use scientia::{FvKernel, FvKernelRole, FvOperandMeaning};

use crate::fv::{BoundKernel, Column, Io, Side, fv_error};
use crate::{FinitumError, FvCellShape, FvFaceBlock, FvTopology};

/// The Scientia/Finitum refusal code for a derivative through a limiter or flux kink (D6).
pub const FV_NONSMOOTH_DERIVATIVE_REFUSED: &str = "FV_NONSMOOTH_DERIVATIVE_REFUSED";

/// Relative determinant floor below which a least-squares normal matrix is degenerate.
const DEGENERATE_DETERMINANT: f64 = 1e-10;

/// The slope limiter of an [`FvReconstruction`].
#[derive(Clone, Debug)]
pub enum FvLimiter {
    /// Unlimited linear reconstruction from least-squares gradients.
    Unlimited,
    /// A Sweby `φ(r)` kernel (`FvKernelRole::Limiter`, operands `SlopeRatio` and `Output`).
    Sweby(Box<FvKernel>),
    BarthJespersen,
    Venkatakrishnan {
        k: f64,
    },
}

#[derive(Clone, Debug)]
enum Limiting {
    Unlimited,
    Sweby(Box<BoundKernel>),
    BarthJespersen,
    /// `ε²` per cell.
    Venkatakrishnan(Vec<f64>),
}

/// Per-block offsets `x_f − x_c`, `dimension` values per face; `plus` empty on boundaries.
#[derive(Clone, Debug)]
struct SideOffsets {
    minus: Vec<f64>,
    plus: Vec<f64>,
}

/// Face states of one face block, in [`FvTopology`] block order.
#[derive(Clone, Debug, PartialEq)]
pub struct FvBlockStates {
    pub minus: Vec<f64>,
    /// Empty on boundary blocks (the ghost is the realization's concern).
    pub plus: Vec<f64>,
}

/// Least-squares gradients, limiting and face-state reconstruction over one [`FvTopology`].
#[derive(Clone, Debug)]
pub struct FvReconstruction {
    topology: Arc<FvTopology>,
    limiting: Limiting,
    /// CSR gradient stencil per cell: `neighbours[offsets[c]..offsets[c + 1]]`, with
    /// `dimension` weights per entry.
    offsets: Vec<usize>,
    neighbours: Vec<usize>,
    weights: Vec<f64>,
    degenerate: usize,
    /// Per cell and axis (tensor meshes): backward and forward axis neighbour and distance.
    axes: Vec<[Option<(usize, f64)>; 2]>,
    sides: Vec<SideOffsets>,
    /// Per cell, `x_f − x_c` of every face (CSR by `face_offsets`), for multidimensional limiters.
    face_offsets: Vec<usize>,
    cell_faces: Vec<f64>,
    refusal: Option<(String, String)>,
}

impl FvReconstruction {
    pub fn new(topology: Arc<FvTopology>, limiter: FvLimiter) -> Result<Self, FinitumError> {
        let dimension = topology.dimension();
        let cells = topology.cell_count();
        let centroids = topology.cell_centroids();
        let interior_blocks = [topology.interior(), topology.periodic()];

        let mut entries = vec![Vec::<(usize, Vec<f64>)>::new(); cells];
        for block in interior_blocks {
            for face in 0..block.len() {
                let (minus, plus) = (block.minus[face], block.plus[face]);
                let delta = &block.delta[face * dimension..(face + 1) * dimension];
                entries[minus].push((plus, delta.to_vec()));
                entries[plus].push((minus, delta.iter().map(|d| -d).collect()));
            }
        }
        // A cell whose face neighbours do not span the space (a simplex corner) extends its
        // stencil to the face neighbours of its neighbours, in the same gather order.
        let spans = |stencil: &[(usize, Vec<f64>)]| {
            invert(&normal_matrix(stencil, dimension), dimension).is_some()
        };
        let extended = (0..cells)
            .map(|cell| {
                if spans(&entries[cell]) {
                    return None;
                }
                let mut stencil = entries[cell].clone();
                for (neighbour, d) in &entries[cell] {
                    for (second, e) in &entries[*neighbour] {
                        let displacement = d.iter().zip(e).map(|(a, b)| a + b).collect::<Vec<_>>();
                        let known = *second == cell
                            || stencil
                                .iter()
                                .any(|(other, f)| other == second && f == &displacement);
                        if !known {
                            stencil.push((*second, displacement));
                        }
                    }
                }
                Some(stencil)
            })
            .collect::<Vec<_>>();
        for (cell, stencil) in extended.into_iter().enumerate() {
            if let Some(stencil) = stencil {
                entries[cell] = stencil;
            }
        }
        let mut offsets = Vec::with_capacity(cells + 1);
        let mut neighbours = Vec::new();
        let mut weights = Vec::new();
        let mut degenerate = 0;
        offsets.push(0);
        for stencil in &entries {
            let inverse = invert(&normal_matrix(stencil, dimension), dimension);
            if inverse.is_none() {
                degenerate += 1;
            }
            for (neighbour, d) in stencil {
                neighbours.push(*neighbour);
                for row in 0..dimension {
                    weights.push(match &inverse {
                        Some(inverse) => (0..dimension).map(|k| inverse[row][k] * d[k]).sum(),
                        None => 0.0,
                    });
                }
            }
            offsets.push(neighbours.len());
        }

        let mut axes = Vec::new();
        if topology.shape() == FvCellShape::Tensor {
            axes = vec![[None, None]; cells * dimension];
            for block in interior_blocks {
                for face in 0..block.len() {
                    let axis = (0..dimension)
                        .max_by(|a, b| {
                            block.normal[*a][face]
                                .abs()
                                .total_cmp(&block.normal[*b][face].abs())
                        })
                        .unwrap_or(0);
                    let distance = block.delta[face * dimension + axis].abs();
                    let (minus, plus) = (block.minus[face], block.plus[face]);
                    let (back, forward) = if block.normal[axis][face] > 0.0 {
                        (minus, plus)
                    } else {
                        (plus, minus)
                    };
                    axes[back * dimension + axis][1] = Some((forward, distance));
                    axes[forward * dimension + axis][0] = Some((back, distance));
                }
            }
        }

        let mut sides = Vec::new();
        let mut per_cell = vec![Vec::<f64>::new(); cells];
        for block in topology.blocks() {
            let mut minus = Vec::with_capacity(block.len() * dimension);
            let mut plus = Vec::with_capacity(block.plus.len() * dimension);
            for face in 0..block.len() {
                let centroid = &block.centroid[face * dimension..(face + 1) * dimension];
                let owner = block.minus[face];
                let start = minus.len();
                for axis in 0..dimension {
                    minus.push(centroid[axis] - centroids[owner * dimension + axis]);
                }
                per_cell[owner].extend_from_slice(&minus[start..]);
                if let Some(neighbour) = block.plus.get(face) {
                    let start = plus.len();
                    for axis in 0..dimension {
                        plus.push(
                            minus[face * dimension + axis] - block.delta[face * dimension + axis],
                        );
                    }
                    per_cell[*neighbour].extend_from_slice(&plus[start..]);
                }
            }
            sides.push(SideOffsets { minus, plus });
        }
        let mut face_offsets = Vec::with_capacity(cells + 1);
        let mut cell_faces = Vec::new();
        face_offsets.push(0);
        for faces in per_cell {
            cell_faces.extend(faces);
            face_offsets.push(cell_faces.len() / dimension.max(1));
        }

        let (limiting, refusal) = match limiter {
            FvLimiter::Unlimited => (Limiting::Unlimited, None),
            FvLimiter::Sweby(kernel) => {
                if topology.shape() != FvCellShape::Tensor {
                    return Err(fv_error(
                        "FV_LIMITER_UNSUPPORTED",
                        "Sweby slope-ratio limiters need a tensor mesh; use Barth-Jespersen or Venkatakrishnan",
                    ));
                }
                let (bound, refusal) = sweby(&kernel)?;
                (Limiting::Sweby(Box::new(bound)), Some(refusal))
            }
            FvLimiter::BarthJespersen => (
                Limiting::BarthJespersen,
                Some(multidimensional_refusal("Barth-Jespersen")),
            ),
            FvLimiter::Venkatakrishnan { k } => {
                if !(k.is_finite() && k > 0.0) {
                    return Err(fv_error(
                        "FV_LIMITER_UNSUPPORTED",
                        format!("Venkatakrishnan k = {k} must be finite and positive"),
                    ));
                }
                let exponent = 1.0 / dimension as f64;
                let epsilon = topology
                    .cell_volumes()
                    .iter()
                    .map(|volume| (k * volume.powf(exponent)).powi(3))
                    .collect();
                (
                    Limiting::Venkatakrishnan(epsilon),
                    Some(multidimensional_refusal("Venkatakrishnan")),
                )
            }
        };
        Ok(Self {
            topology,
            limiting,
            offsets,
            neighbours,
            weights,
            degenerate,
            axes,
            sides,
            face_offsets,
            cell_faces,
            refusal,
        })
    }

    /// Cells whose least-squares stencil is rank deficient (zero gradient, first order there).
    pub fn degenerate_cells(&self) -> usize {
        self.degenerate
    }

    pub fn is_limited(&self) -> bool {
        !matches!(self.limiting, Limiting::Unlimited)
    }

    /// The typed D6 refusal `(code, message)` of a derivative through this reconstruction;
    /// `None` for the unlimited (linear) reconstruction.
    pub fn derivative_refusal(&self) -> Option<(&str, &str)> {
        self.refusal
            .as_ref()
            .map(|(code, message)| (code.as_str(), message.as_str()))
    }

    /// Limited slopes `σ_c`, `dimension` values per cell (diagnostic; allocates).
    pub fn slopes(&self, state: &[f64]) -> Result<Vec<f64>, FinitumError> {
        let length = self.topology.cell_count() * self.topology.dimension();
        let (mut slopes, mut ratio, mut phi) =
            (vec![0.0; length], vec![0.0; length], vec![0.0; length]);
        self.check_state(state)?;
        self.slopes_into(state, &mut slopes, &mut ratio, &mut phi)?;
        Ok(slopes)
    }

    /// Reconstructed face states for every face block (diagnostic; allocates).
    pub fn face_states(&self, state: &[f64]) -> Result<Vec<FvBlockStates>, FinitumError> {
        let slopes = self.slopes(state)?;
        Ok(self
            .topology
            .blocks()
            .enumerate()
            .map(|(index, block)| {
                let mut minus = vec![0.0; block.len()];
                let mut plus = vec![0.0; block.plus.len()];
                self.face_values(index, block, state, &slopes, &mut minus, &mut plus);
                FvBlockStates { minus, plus }
            })
            .collect())
    }

    fn check_state(&self, state: &[f64]) -> Result<(), FinitumError> {
        if state.len() == self.topology.cell_count() {
            Ok(())
        } else {
            Err(fv_error(
                "FV_STATE_LENGTH",
                format!(
                    "reconstruction state has length {}, expected {}",
                    state.len(),
                    self.topology.cell_count()
                ),
            ))
        }
    }

    /// The (linear) least-squares gradient of `values`.
    pub(crate) fn gradient_into(&self, values: &[f64], gradient: &mut [f64]) {
        let dimension = self.topology.dimension();
        for cell in 0..values.len() {
            let out = &mut gradient[cell * dimension..(cell + 1) * dimension];
            out.fill(0.0);
            for entry in self.offsets[cell]..self.offsets[cell + 1] {
                let difference = values[self.neighbours[entry]] - values[cell];
                for (axis, slot) in out.iter_mut().enumerate() {
                    *slot += self.weights[entry * dimension + axis] * difference;
                }
            }
        }
    }

    /// `values += (∂g/∂u)ᵀ gradient_cotangent`: the transpose of [`Self::gradient_into`].
    pub(crate) fn gradient_transpose(&self, gradient_cotangent: &[f64], values: &mut [f64]) {
        let dimension = self.topology.dimension();
        for cell in 0..values.len() {
            let bar = &gradient_cotangent[cell * dimension..(cell + 1) * dimension];
            for entry in self.offsets[cell]..self.offsets[cell + 1] {
                let weight = &self.weights[entry * dimension..(entry + 1) * dimension];
                let contribution = weight.iter().zip(bar).map(|(w, b)| w * b).sum::<f64>();
                values[self.neighbours[entry]] += contribution;
                values[cell] -= contribution;
            }
        }
    }

    /// Limited slopes into `slopes`; `ratio`/`phi` are Sweby batch buffers (cells × dimension).
    pub(crate) fn slopes_into(
        &self,
        state: &[f64],
        slopes: &mut [f64],
        ratio: &mut [f64],
        phi: &mut [f64],
    ) -> Result<(), FinitumError> {
        let dimension = self.topology.dimension();
        match &self.limiting {
            Limiting::Unlimited => self.gradient_into(state, slopes),
            Limiting::Sweby(kernel) => {
                let mut count = 0;
                for (index, neighbours) in self.axes.iter().enumerate() {
                    slopes[index] = 0.0;
                    if let [Some((back, h_back)), Some((forward, h_forward))] = *neighbours {
                        let cell = index / dimension;
                        let backward = (state[cell] - state[back]) / h_back;
                        let forward = (state[forward] - state[cell]) / h_forward;
                        ratio[count] = if backward == 0.0 {
                            0.0
                        } else {
                            forward / backward
                        };
                        slopes[index] = backward;
                        count += 1;
                    }
                }
                kernel.run(
                    count,
                    &Io {
                        minus: ratio,
                        plus: &[],
                        minus_tangent: &[],
                        plus_tangent: &[],
                        normal: &[],
                        data: &[],
                    },
                    phi,
                )?;
                let mut point = 0;
                for (index, neighbours) in self.axes.iter().enumerate() {
                    if let [Some(_), Some(_)] = neighbours {
                        let factor = phi[point];
                        if !factor.is_finite() {
                            return Err(fv_error(
                                "FV_LIMITER_INVALID",
                                format!(
                                    "limiter value {factor} at slope ratio {} (cell {})",
                                    ratio[point],
                                    index / dimension
                                ),
                            ));
                        }
                        slopes[index] *= factor;
                        point += 1;
                    }
                }
            }
            Limiting::BarthJespersen | Limiting::Venkatakrishnan(_) => {
                self.gradient_into(state, slopes);
                for cell in 0..state.len() {
                    let value = state[cell];
                    let (mut high, mut low) = (value, value);
                    for entry in self.offsets[cell]..self.offsets[cell + 1] {
                        let neighbour = state[self.neighbours[entry]];
                        high = high.max(neighbour);
                        low = low.min(neighbour);
                    }
                    let gradient = &mut slopes[cell * dimension..(cell + 1) * dimension];
                    let mut factor = 1.0_f64;
                    for face in self.face_offsets[cell]..self.face_offsets[cell + 1] {
                        let offset = &self.cell_faces[face * dimension..(face + 1) * dimension];
                        let change = gradient.iter().zip(offset).map(|(g, o)| g * o).sum::<f64>();
                        let bound = if change > 0.0 {
                            high - value
                        } else if change < 0.0 {
                            low - value
                        } else {
                            continue;
                        };
                        let face_factor = match &self.limiting {
                            Limiting::Venkatakrishnan(epsilon) => {
                                let e = epsilon[cell];
                                (bound * bound + e + 2.0 * bound * change)
                                    / (bound * bound + 2.0 * change * change + bound * change + e)
                            }
                            _ => bound / change,
                        };
                        factor = factor.min(face_factor.min(1.0));
                    }
                    for slot in gradient.iter_mut() {
                        *slot *= factor;
                    }
                }
            }
        }
        Ok(())
    }

    /// Face states of block `index`: `u_c + σ_c·(x_f − x_c)` per side.
    pub(crate) fn face_values(
        &self,
        index: usize,
        block: &FvFaceBlock,
        values: &[f64],
        slopes: &[f64],
        minus: &mut [f64],
        plus: &mut [f64],
    ) {
        let dimension = self.topology.dimension();
        let sides = &self.sides[index];
        let side = |cell: usize, offsets: &[f64], face: usize| {
            let slope = &slopes[cell * dimension..(cell + 1) * dimension];
            let offset = &offsets[face * dimension..(face + 1) * dimension];
            values[cell] + slope.iter().zip(offset).map(|(s, o)| s * o).sum::<f64>()
        };
        for (face, slot) in minus[..block.len()].iter_mut().enumerate() {
            *slot = side(block.minus[face], &sides.minus, face);
        }
        for (face, slot) in plus[..block.plus.len()].iter_mut().enumerate() {
            *slot = side(block.plus[face], &sides.plus, face);
        }
    }

    /// Transpose of [`Self::face_values`]: `values[c] += ū_f`, `slopes[c] += ū_f (x_f − x_c)`.
    pub(crate) fn face_transpose(
        &self,
        index: usize,
        block: &FvFaceBlock,
        minus: &[f64],
        plus: &[f64],
        values: &mut [f64],
        slopes: &mut [f64],
    ) {
        let dimension = self.topology.dimension();
        let sides = &self.sides[index];
        let mut add = |cell: usize, offsets: &[f64], face: usize, bar: f64| {
            values[cell] += bar;
            let offset = &offsets[face * dimension..(face + 1) * dimension];
            for (slot, o) in slopes[cell * dimension..(cell + 1) * dimension]
                .iter_mut()
                .zip(offset)
            {
                *slot += bar * o;
            }
        };
        for (face, (cell, bar)) in block.minus.iter().zip(minus).enumerate() {
            add(*cell, &sides.minus, face, *bar);
        }
        for (face, (cell, bar)) in block.plus.iter().zip(plus).enumerate() {
            add(*cell, &sides.plus, face, *bar);
        }
    }
}

/// Bind a Sweby kernel (its slope ratios ride in the `minus` column) and check its derivative
/// through `malleus::ad` with [`NonsmoothPolicy::Refuse`].
fn sweby(kernel: &FvKernel) -> Result<(BoundKernel, (String, String)), FinitumError> {
    if kernel.role != FvKernelRole::Limiter {
        return Err(fv_error(
            "FV_KERNEL_SHAPE",
            format!("expected a Limiter kernel, got {:?}", kernel.role),
        ));
    }
    let columns = kernel
        .operands
        .iter()
        .map(|meaning| match meaning {
            FvOperandMeaning::SlopeRatio => Ok(Column::State(Side::Minus)),
            FvOperandMeaning::Output => Ok(Column::Write),
            other => Err(fv_error(
                "FV_KERNEL_SHAPE",
                format!("limiter operand {other:?} is not a slope ratio"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ratio = kernel
        .operand(&FvOperandMeaning::SlopeRatio)
        .ok_or_else(|| fv_error("FV_KERNEL_SHAPE", "limiter has no slope ratio"))?;
    let output = kernel
        .operand(&FvOperandMeaning::Output)
        .ok_or_else(|| fv_error("FV_KERNEL_SHAPE", "limiter has no output"))?;
    let request = DerivativeRequest {
        mode: DerivativeMode::Jvp,
        independent_operands: vec![ratio],
        dependent_operands: vec![output],
    };
    let options = DerivativeOptions {
        nonsmooth: NonsmoothPolicy::Refuse,
        ..DerivativeOptions::default()
    };
    let refusal = match differentiate_with(&kernel.kernel, &request, &options) {
        Err(DifferentiationError::NonsmoothRefused(sites)) => (
            FV_NONSMOOTH_DERIVATIVE_REFUSED.to_string(),
            format!(
                "limiter `{}` differentiates {} kink site(s) ({}) under NonsmoothPolicy::Refuse",
                kernel.kernel.name,
                sites.len(),
                sites
                    .iter()
                    .map(|site| format!("{:?}@{}", site.kind, site.statement))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ),
        Err(error) => ("FV_DERIVATIVE_FAILED".to_string(), format!("{error:?}")),
        Ok(_) => (
            "FV_LIMITED_DERIVATIVE_UNSUPPORTED".to_string(),
            format!(
                "limiter `{}` is smooth but the limited-slope JVP chain is not realized",
                kernel.kernel.name
            ),
        ),
    };
    Ok((BoundKernel::new(kernel.kernel.clone(), columns)?, refusal))
}

fn multidimensional_refusal(name: &str) -> (String, String) {
    (
        FV_NONSMOOTH_DERIVATIVE_REFUSED.to_string(),
        format!("{name} limiting takes min/max over neighbour bounds and faces (decision D6)"),
    )
}

/// `Σ_k d_k d_kᵀ` over a stencil (leading `dimension × dimension` block).
fn normal_matrix(stencil: &[(usize, Vec<f64>)], dimension: usize) -> [[f64; 3]; 3] {
    let mut normal = [[0.0; 3]; 3];
    for (_, d) in stencil {
        for row in 0..dimension {
            for column in 0..dimension {
                normal[row][column] += d[row] * d[column];
            }
        }
    }
    normal
}

/// Inverse of the leading `dimension × dimension` block, `None` when it is degenerate.
fn invert(matrix: &[[f64; 3]; 3], dimension: usize) -> Option<[[f64; 3]; 3]> {
    let m = matrix;
    let trace = (0..dimension).map(|i| m[i][i]).sum::<f64>();
    let (determinant, adjugate) = match dimension {
        1 => (m[0][0], [[1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]]),
        2 => (
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
            [[m[1][1], -m[0][1], 0.0], [-m[1][0], m[0][0], 0.0], [0.0; 3]],
        ),
        3 => {
            let cofactor = |r0: usize, r1: usize, c0: usize, c1: usize| {
                m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
            };
            let adjugate = [
                [
                    cofactor(1, 2, 1, 2),
                    -cofactor(0, 2, 1, 2),
                    cofactor(0, 1, 1, 2),
                ],
                [
                    -cofactor(1, 2, 0, 2),
                    cofactor(0, 2, 0, 2),
                    -cofactor(0, 1, 0, 2),
                ],
                [
                    cofactor(1, 2, 0, 1),
                    -cofactor(0, 2, 0, 1),
                    cofactor(0, 1, 0, 1),
                ],
            ];
            let determinant =
                m[0][0] * adjugate[0][0] + m[0][1] * adjugate[1][0] + m[0][2] * adjugate[2][0];
            (determinant, adjugate)
        }
        _ => return None,
    };
    let scale = (trace / dimension as f64).powi(dimension as i32);
    if !(trace > 0.0 && determinant > DEGENERATE_DETERMINANT * scale) {
        return None;
    }
    let mut inverse = [[0.0; 3]; 3];
    for row in 0..dimension {
        for column in 0..dimension {
            inverse[row][column] = adjugate[row][column] / determinant;
        }
    }
    Some(inverse)
}
