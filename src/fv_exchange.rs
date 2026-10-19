//! FV ↔ P1 FE face-flux exchange over a matching interface (W13 A4, SV2C L9, gate 1c).
//!
//! A [`FaceFluxExchange`] pairs every face of one FV boundary tag with exactly one exterior
//! facet of a P1 finite-element mesh (coincident centroid and measure, opposite normals) and
//! realizes the weakly enforced conjugate condition: one face flux
//! `q_f = g_f (T_P − T_s(x_f))` per face, outward from the FV cell `P` into the FE side, where
//! `T_s(x_f)` is the P1 trace at the face centroid (the facet-vertex mean of a matching facet)
//! and `g_f` is
//!
//! * [`FaceFluxExchangeLaw::DirichletNeumann`]: the half-cell two-point conductance
//!   `k_P/(δ·n)` (the FE trace is the FV face's Dirichlet state; the FE side receives the flux
//!   as a Neumann load), with `k_P` the FV realization's own cell conductivity;
//! * [`FaceFluxExchangeLaw::Robin`]: a given transfer coefficient `h` (contact conductance).
//!
//! The FV residual gains `+|f|q_f` on the owner cell (the FV outward-flux convention) and the
//! FE residual gains `−|f|q_f/n_v` on each of the facet's `n_v` vertices: the exact integral of
//! the constant `q_f` against each P1 test trace (`∫_f φ_i = |f|/n_v`). P1 traces are a
//! partition of unity, so the two sides' contributions cancel to round-off; that is the
//! [`InterfaceBalance`] receipt. The FE convention is the one Finitum realizes for
//! `div(q) = 0`, `q = −k grad T`: `R_i = ∫ k∇T·∇φ_i − ∫_Γ g φ_i` with `g` the inward boundary
//! flux. The equality of temperatures is enforced weakly through the flux, not by elimination.
//!
//! The exchange is linear and state independent in its derivative: the JVP is the residual
//! map applied to the direction and the VJP is its exact transpose.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::{
    FacetId, FacetTopology, FinitumError, FvRealization, Mesh, RegionTagId, exterior_facet,
};
use scientia::Digest;

pub const FACE_FLUX_EXCHANGE_SCHEMA: &str = "finitum-face-flux-exchange/1";
pub const INTERFACE_BALANCE_SCHEMA: &str = "finitum-interface-balance/1";

fn exchange_error(code: &str, message: impl std::fmt::Display) -> FinitumError {
    FinitumError::InvalidRealization(format!("{code}: {message}"))
}

/// How the face flux is formed from the FV cell value and the FE trace.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FaceFluxExchangeLaw {
    /// `q = k_P (T_P − T_s)/(δ·n)`: half-cell TPFA against the FE trace.
    DirichletNeumann,
    /// `q = h (T_P − T_s)` with a finite, positive `h`.
    Robin { coefficient: f64 },
}

/// What an exchange bound, for run evidence and Krasis's operator identity.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FaceFluxExchangeReceipt {
    pub schema: String,
    pub digest: Digest,
    pub fv_digest: Digest,
    pub fv_tag: RegionTagId,
    pub fe_vertices: usize,
    pub faces: usize,
    pub law: FaceFluxExchangeLaw,
    /// Always `weak_flux`: equal temperature is enforced through the flux, not elimination.
    pub enforcement: String,
    pub matching_tolerance: f64,
    pub max_centroid_mismatch: f64,
}

/// `InterfaceBalance`: Σ FV face flux + Σ FE facet load = 0 to round-off.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InterfaceBalance {
    pub schema: String,
    pub faces: usize,
    /// `Σ_f |f| q_f` (what the FV residual gains), Neumaier.
    pub fv_flux_sum: f64,
    /// `Σ_f Σ_v (−|f| q_f / n_v)` (what the FE residual gains), Neumaier.
    pub fe_flux_sum: f64,
    pub imbalance: f64,
    /// `Σ_f |f||q_f|`.
    pub scale: f64,
    /// `4·n_v·ε`.
    pub relative_bound: f64,
    pub balanced: bool,
}

/// A matching FV boundary-face ↔ P1 exterior-facet flux exchange.
#[derive(Clone, Debug)]
pub struct FaceFluxExchange {
    law: FaceFluxExchangeLaw,
    tag: RegionTagId,
    fv_cells: usize,
    fe_vertices: usize,
    /// FV owner cell per face (FV block order).
    owner: Vec<usize>,
    /// FE facet vertices per face (matched facet).
    vertices: Vec<Vec<usize>>,
    area: Vec<f64>,
    conductance: Vec<f64>,
    receipt: FaceFluxExchangeReceipt,
}

impl FaceFluxExchange {
    /// Pair the faces of FV boundary `tag` with `fe_facets` (exterior facets of `fe_mesh`).
    /// The FV tag must be a prescribed-flux boundary whose flux vanishes at the states 0 and 1
    /// (an adiabatic placeholder the exchange supplies; `FV_EXCHANGE_BOUNDARY`). Every FV face
    /// needs exactly one FE facet with centroid within `tolerance` (max norm), measure within
    /// `tolerance·max(1, |f|)` and normal sum within `tolerance`, and every FE facet must be
    /// used once (`FV_EXCHANGE_NONMATCHING`). Dimensions must agree (`FV_EXCHANGE_SHAPE`).
    pub fn new(
        fv: &FvRealization,
        tag: &RegionTagId,
        fe_mesh: &Mesh,
        fe_facets: &[FacetId],
        law: FaceFluxExchangeLaw,
        tolerance: f64,
    ) -> Result<Self, FinitumError> {
        let topology = fv.topology();
        let dimension = topology.dimension();
        if fe_mesh.dimension() != dimension || !(tolerance.is_finite() && tolerance > 0.0) {
            return Err(exchange_error(
                "FV_EXCHANGE_SHAPE",
                format!(
                    "FV is {dimension}-D, FE mesh {}-D, tolerance {tolerance}",
                    fe_mesh.dimension()
                ),
            ));
        }
        let block = topology.boundary().get(tag).ok_or_else(|| {
            exchange_error(
                "FV_EXCHANGE_BOUNDARY",
                format!("FV has no boundary tag `{tag}`"),
            )
        })?;
        if !fv.is_prescribed_boundary(tag) {
            return Err(exchange_error(
                "FV_EXCHANGE_BOUNDARY",
                format!("FV boundary `{tag}` must be a prescribed-flux boundary"),
            ));
        }
        for value in [0.0, 1.0] {
            let fluxes = fv.face_fluxes(0.0, &vec![value; fv.dimension()])?;
            if fluxes.boundary[tag].iter().any(|flux| *flux != 0.0) {
                return Err(exchange_error(
                    "FV_EXCHANGE_BOUNDARY",
                    format!(
                        "FV boundary `{tag}` prescribes a nonzero flux; the exchange supplies it"
                    ),
                ));
            }
        }
        if let FaceFluxExchangeLaw::Robin { coefficient } = law
            && !(coefficient.is_finite() && coefficient > 0.0)
        {
            return Err(exchange_error(
                "FV_EXCHANGE_LAW",
                format!("Robin coefficient {coefficient} must be finite and positive"),
            ));
        }
        let facets = FacetTopology::from_mesh(fe_mesh)?;
        let geometry = fe_facets
            .iter()
            .map(|facet| exterior_facet(fe_mesh, &facets, *facet))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| exchange_error("FV_EXCHANGE_NONMATCHING", error))?;
        if fe_facets
            .iter()
            .map(|facet| facet.0)
            .collect::<BTreeSet<_>>()
            .len()
            != fe_facets.len()
            || fe_facets.len() != block.len()
        {
            return Err(exchange_error(
                "FV_EXCHANGE_NONMATCHING",
                format!(
                    "{} FV faces against {} distinct FE facets",
                    block.len(),
                    fe_facets.len()
                ),
            ));
        }
        let centroids = topology.cell_centroids();
        let mut used = vec![false; fe_facets.len()];
        let (mut owner, mut vertices, mut area, mut conductance) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut max_mismatch = 0.0_f64;
        for face in 0..block.len() {
            let centroid = &block.centroid[face * dimension..(face + 1) * dimension];
            let mismatch = |candidate: &crate::ExteriorFacet| {
                candidate
                    .centroid
                    .iter()
                    .zip(centroid)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
            };
            let matches = geometry
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    mismatch(candidate) <= tolerance
                        && (candidate.measure - block.area[face]).abs()
                            <= tolerance * block.area[face].max(1.0)
                        && (0..dimension).all(|axis| {
                            (candidate.normal[axis] + block.normal[axis][face]).abs() <= tolerance
                        })
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let [index] = matches[..] else {
                return Err(exchange_error(
                    "FV_EXCHANGE_NONMATCHING",
                    format!(
                        "FV face {face} of `{tag}` has {} coincident FE facets, expected 1",
                        matches.len()
                    ),
                ));
            };
            if std::mem::replace(&mut used[index], true) {
                return Err(exchange_error(
                    "FV_EXCHANGE_NONMATCHING",
                    format!("FE facet {} is matched twice", fe_facets[index].0),
                ));
            }
            max_mismatch = max_mismatch.max(mismatch(&geometry[index]));
            let cell = block.minus[face];
            let g = match law {
                FaceFluxExchangeLaw::Robin { coefficient } => coefficient,
                FaceFluxExchangeLaw::DirichletNeumann => {
                    let k = fv.cell_conductivity(cell).ok_or_else(|| {
                        exchange_error(
                            "FV_EXCHANGE_LAW",
                            "Dirichlet-Neumann needs FV diffusion (uniform or regions)",
                        )
                    })?;
                    let projected = (0..dimension)
                        .map(|axis| {
                            (centroid[axis] - centroids[cell * dimension + axis])
                                * block.normal[axis][face]
                        })
                        .sum::<f64>();
                    k / projected
                }
            };
            owner.push(cell);
            vertices.push(
                facets.facets()[fe_facets[index].0]
                    .vertices
                    .iter()
                    .map(|vertex| vertex.0)
                    .collect::<Vec<_>>(),
            );
            area.push(block.area[face]);
            conductance.push(g);
        }
        let digest = Digest::blake3(
            &serde_json::to_vec(&(
                FACE_FLUX_EXCHANGE_SCHEMA,
                fv.digest(),
                tag,
                law,
                &vertices,
                fe_mesh.vertices().len(),
                tolerance,
            ))
            .expect("exchange identity is serializable"),
        );
        Ok(Self {
            law,
            tag: tag.clone(),
            fv_cells: fv.dimension(),
            fe_vertices: fe_mesh.vertices().len(),
            receipt: FaceFluxExchangeReceipt {
                schema: FACE_FLUX_EXCHANGE_SCHEMA.into(),
                digest,
                fv_digest: fv.digest().clone(),
                fv_tag: tag.clone(),
                fe_vertices: fe_mesh.vertices().len(),
                faces: owner.len(),
                law,
                enforcement: "weak_flux".into(),
                matching_tolerance: tolerance,
                max_centroid_mismatch: max_mismatch,
            },
            owner,
            vertices,
            area,
            conductance,
        })
    }

    pub fn law(&self) -> FaceFluxExchangeLaw {
        self.law
    }

    pub fn fv_tag(&self) -> &RegionTagId {
        &self.tag
    }

    /// FV state length (cells).
    pub fn fv_dimension(&self) -> usize {
        self.fv_cells
    }

    /// FE state length (P1 vertices of the FE mesh, one scalar per vertex).
    pub fn fe_dimension(&self) -> usize {
        self.fe_vertices
    }

    pub fn faces(&self) -> usize {
        self.owner.len()
    }

    pub fn digest(&self) -> &Digest {
        &self.receipt.digest
    }

    pub fn receipt(&self) -> &FaceFluxExchangeReceipt {
        &self.receipt
    }

    fn check(&self, label: &str, fv: &[f64], fe: &[f64]) -> Result<(), FinitumError> {
        if fv.len() != self.fv_cells || fe.len() != self.fe_vertices {
            return Err(exchange_error(
                "FV_EXCHANGE_SHAPE",
                format!(
                    "{label}: FV length {} (expected {}), FE length {} (expected {})",
                    fv.len(),
                    self.fv_cells,
                    fe.len(),
                    self.fe_vertices
                ),
            ));
        }
        Ok(())
    }

    fn flux(&self, face: usize, fv: &[f64], fe: &[f64]) -> f64 {
        let vertices = &self.vertices[face];
        let trace = vertices.iter().map(|vertex| fe[*vertex]).sum::<f64>() / vertices.len() as f64;
        self.conductance[face] * (fv[self.owner[face]] - trace)
    }

    /// `q_f` per face (per unit area, outward from the FV cell), FV block order.
    pub fn face_fluxes(
        &self,
        fv_state: &[f64],
        fe_state: &[f64],
    ) -> Result<Vec<f64>, FinitumError> {
        self.check("exchange state", fv_state, fe_state)?;
        Ok((0..self.faces())
            .map(|face| self.flux(face, fv_state, fe_state))
            .collect())
    }

    /// The FE trace `T_s(x_f)` per face, FV block order.
    pub fn fe_trace(&self, fe_state: &[f64]) -> Result<Vec<f64>, FinitumError> {
        self.check("exchange trace", &vec![0.0; self.fv_cells], fe_state)?;
        Ok(self
            .vertices
            .iter()
            .map(|vertices| {
                vertices.iter().map(|vertex| fe_state[*vertex]).sum::<f64>() / vertices.len() as f64
            })
            .collect())
    }

    fn scatter(&self, fv: &[f64], fe: &[f64], fv_out: &mut [f64], fe_out: &mut [f64]) {
        for face in 0..self.faces() {
            let flow = self.area[face] * self.flux(face, fv, fe);
            fv_out[self.owner[face]] += flow;
            let share = flow / self.vertices[face].len() as f64;
            for vertex in &self.vertices[face] {
                fe_out[*vertex] -= share;
            }
        }
    }

    /// ADD the exchange to both residuals: `fv_out[P] += |f|q_f`, `fe_out[v] −= |f|q_f/n_v`.
    pub fn residual(
        &self,
        fv_state: &[f64],
        fe_state: &[f64],
        fv_out: &mut [f64],
        fe_out: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.check("exchange state", fv_state, fe_state)?;
        self.check("exchange residual", fv_out, fe_out)?;
        self.scatter(fv_state, fe_state, fv_out, fe_out);
        Ok(())
    }

    /// ADD the JVP in the direction `(fv_direction, fe_direction)`; the exchange is linear, so
    /// this is the residual map of the direction at any state.
    pub fn jacobian_vector_product(
        &self,
        fv_direction: &[f64],
        fe_direction: &[f64],
        fv_out: &mut [f64],
        fe_out: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.residual(fv_direction, fe_direction, fv_out, fe_out)
    }

    /// ADD the exact transpose: for cotangents `(w_fv, w_fe)`, with
    /// `c_f = |f| g_f (w_fv[P] − Σ_v w_fe[v]/n_v)`, `fv_out[P] += c_f` and
    /// `fe_out[v] −= c_f/n_v`.
    pub fn vector_jacobian_product(
        &self,
        fv_cotangent: &[f64],
        fe_cotangent: &[f64],
        fv_out: &mut [f64],
        fe_out: &mut [f64],
    ) -> Result<(), FinitumError> {
        self.check("exchange cotangent", fv_cotangent, fe_cotangent)?;
        self.check("exchange cotangent output", fv_out, fe_out)?;
        for face in 0..self.faces() {
            let vertices = &self.vertices[face];
            let count = vertices.len() as f64;
            let weight = fv_cotangent[self.owner[face]]
                - vertices
                    .iter()
                    .map(|vertex| fe_cotangent[*vertex])
                    .sum::<f64>()
                    / count;
            let c = self.area[face] * self.conductance[face] * weight;
            fv_out[self.owner[face]] += c;
            for vertex in vertices {
                fe_out[*vertex] -= c / count;
            }
        }
        Ok(())
    }

    /// The `InterfaceBalance` receipt at `(fv_state, fe_state)`: both sides' contributions are
    /// summed exactly as [`FaceFluxExchange::residual`] adds them.
    pub fn balance(
        &self,
        fv_state: &[f64],
        fe_state: &[f64],
    ) -> Result<InterfaceBalance, FinitumError> {
        self.check("exchange state", fv_state, fe_state)?;
        let (mut fv_sum, mut fe_sum) = (
            crate::fv::Neumaier::default(),
            crate::fv::Neumaier::default(),
        );
        let mut scale = 0.0;
        let mut widest = 1usize;
        for face in 0..self.faces() {
            let flow = self.area[face] * self.flux(face, fv_state, fe_state);
            fv_sum.add(flow);
            scale += flow.abs();
            let share = flow / self.vertices[face].len() as f64;
            widest = widest.max(self.vertices[face].len());
            for _ in &self.vertices[face] {
                fe_sum.add(-share);
            }
        }
        let (fv_flux_sum, fe_flux_sum) = (fv_sum.total(), fe_sum.total());
        let imbalance = fv_flux_sum + fe_flux_sum;
        let relative_bound = 4.0 * widest as f64 * f64::EPSILON;
        Ok(InterfaceBalance {
            schema: INTERFACE_BALANCE_SCHEMA.into(),
            faces: self.faces(),
            fv_flux_sum,
            fe_flux_sum,
            imbalance,
            scale,
            relative_bound,
            balanced: imbalance.abs() <= relative_bound * scale,
        })
    }
}
