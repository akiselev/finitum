//! Reusable checks for concrete discretizations and global realizations.
//!
//! Reports are implementation evidence, not scientific acceptance or support claims.

use methodus::{
    ComparisonReport, ComparisonTolerance, ConvergenceOrderReport, ConvergenceSample,
    EvaluationContext, LinearOperator, check_solve_strategy_agreement, estimate_convergence_order,
};
use scientia::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    ConstraintSet, ExactSequence, FinitumError, Mesh, NonmatchingTransfer, RealizationPlan,
};

pub const VERIFICATION_REPORT_SCHEMA: &str = "finitum.verification-report/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationCheckKind {
    NodalPatch,
    RealizationAgreement,
    GlobalTransposeWork,
    ConstraintWork,
    TransferConservation,
    ExactSequence,
    MeshRefinement,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationSubject {
    pub identity: String,
    pub digest: Digest,
}

impl VerificationSubject {
    pub fn from_serializable(
        identity: impl Into<String>,
        subject: &impl Serialize,
    ) -> Result<Self, FinitumError> {
        let identity = identity.into();
        if identity.trim().is_empty() {
            return Err(invalid("verification subject identity must not be empty"));
        }
        Ok(Self {
            identity,
            digest: digest(subject)?,
        })
    }

    fn validate(&self) -> Result<(), FinitumError> {
        if self.identity.trim().is_empty()
            || self.digest.algorithm != "blake3"
            || self.digest.hex.len() != 64
            || !self
                .digest
                .hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid(
                "verification subject needs a nonempty identity and canonical blake3 digest",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReportHeader {
    pub schema: String,
    pub check_kind: VerificationCheckKind,
    pub subject: VerificationSubject,
    pub report_digest: Digest,
}

/// Acceptance available only after source-aware recomputation succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedVerification {
    pub accepted: bool,
}

macro_rules! report {
    ($body:ident, $name:ident, { $($field:tt)* }) => {
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct $body { $($field)* }
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct $name {
            pub header: VerificationReportHeader,
            pub body: $body,
        }
    };
}

report!(PatchCheckBody, PatchCheckReport, {
    pub tolerance: ComparisonTolerance,
    pub coordinates: Vec<Vec<f64>>,
    pub component_count: usize,
    pub candidate_values: Vec<f64>,
    pub exact_values: Vec<f64>,
    pub comparison: ComparisonReport,
});

report!(RealizationAgreementBody, RealizationAgreementReport, {
    pub tolerance: ComparisonTolerance,
    pub lane_width: usize,
    pub probe: Vec<f64>,
    pub matrix_free_output: Vec<f64>,
    pub assembled_output: Vec<f64>,
    pub element_assembled_output: Vec<f64>,
    pub partial_assembled_output: Vec<f64>,
    pub assembled: ComparisonReport,
    pub element_assembled: ComparisonReport,
    pub partial_assembled: ComparisonReport,
});

report!(GlobalTransposeWorkBody, GlobalTransposeWorkReport, {
    pub tolerance: ComparisonTolerance,
    pub left_probe: Vec<f64>,
    pub right_probe: Vec<f64>,
    pub forward_action: Vec<f64>,
    pub transpose_action: Vec<f64>,
    pub forward_work: f64,
    pub transpose_work: f64,
    pub comparison: ComparisonReport,
});

report!(ConstraintWorkBody, ConstraintWorkReport, {
    pub tolerance: ComparisonTolerance,
    pub unconstrained: Vec<f64>,
    pub physical_residual: Vec<f64>,
    pub prolonged: Vec<f64>,
    pub restricted: Vec<f64>,
    pub forward_work: f64,
    pub transpose_work: f64,
    pub comparison: ComparisonReport,
});

report!(TransferConservationBody, TransferConservationReport, {
    pub tolerance: ComparisonTolerance,
    pub source_values: Vec<f64>,
    pub target_residual: Vec<f64>,
    pub target_weights: Vec<f64>,
    pub target_values: Vec<f64>,
    pub source_residual: Vec<f64>,
    pub forward_work: f64,
    pub transpose_work: f64,
    pub comparison: ComparisonReport,
});

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactSequenceCheckBody {
    pub dimension: usize,
    pub expected_stage_count: usize,
    pub observed_stage_count: usize,
    pub stage_complete: bool,
    pub gradient_shape: [usize; 2],
    pub curl_shape: [usize; 2],
    pub divergence_shape: Option<[usize; 2]>,
    pub gradient_rank: usize,
    pub curl_rank: usize,
    pub divergence_rank: Option<usize>,
    pub curl_gradient_zero: bool,
    pub divergence_curl_zero: Option<bool>,
    pub exact_at_edges: bool,
    pub exact_at_facets: Option<bool>,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactSequenceCheckReport {
    pub header: VerificationReportHeader,
    pub body: ExactSequenceCheckBody,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshRefinementSample<'a> {
    pub mesh: &'a Mesh,
    pub error: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshRefinementLevel {
    pub mesh_digest: Digest,
    pub maximum_cell_diameter: f64,
    pub error: f64,
}

report!(MeshRefinementCheckBody, MeshRefinementCheckReport, {
    pub minimum_required_order: f64,
    pub levels: Vec<MeshRefinementLevel>,
    pub convergence: ConvergenceOrderReport,
    pub accepted: bool,
});

impl RealizationAgreementReport {
    /// Re-execute every realization action and validate the complete report.
    pub fn validate(&self, plan: &RealizationPlan) -> Result<ValidatedVerification, FinitumError> {
        let expected = check_realization_agreement(
            plan,
            &self.body.probe,
            self.body.lane_width,
            self.body.tolerance,
        )?;
        require_report(self, &expected)?;
        Ok(ValidatedVerification {
            accepted: expected.body.assembled.accepted
                && expected.body.element_assembled.accepted
                && expected.body.partial_assembled.accepted,
        })
    }
}

impl PatchCheckReport {
    /// Recompute the patch comparison against the bound concrete mesh and stored exact values.
    pub fn validate(&self, mesh: &Mesh) -> Result<ValidatedVerification, FinitumError> {
        if self.body.component_count == 0 {
            return Err(invalid("patch component count must be nonzero"));
        }
        let expected_len = mesh
            .vertices()
            .len()
            .checked_mul(self.body.component_count)
            .ok_or_else(|| invalid("patch value extent overflows usize"))?;
        if self.body.coordinates.len() != mesh.vertices().len()
            || self.body.candidate_values.len() != expected_len
            || self.body.exact_values.len() != expected_len
        {
            return Err(invalid(
                "patch report coordinates and value extents do not match the concrete mesh",
            ));
        }
        let mut exact = self
            .body
            .exact_values
            .chunks_exact(self.body.component_count);
        let expected = check_nodal_patch(
            mesh,
            self.body.component_count,
            &self.body.candidate_values,
            self.body.tolerance,
            |_| exact.next().unwrap_or(&[]).to_vec(),
        )?;
        if !exact.remainder().is_empty() || exact.next().is_some() {
            return Err(invalid("patch report contains excess exact values"));
        }
        require_report(self, &expected)?;
        Ok(ValidatedVerification {
            accepted: expected.body.comparison.accepted,
        })
    }
}

impl GlobalTransposeWorkReport {
    /// Re-execute both actions and recompute the complete work report.
    pub fn validate(
        &self,
        subject: VerificationSubject,
        forward: &dyn LinearOperator,
        transpose: &dyn LinearOperator,
    ) -> Result<ValidatedVerification, FinitumError> {
        let expected = check_global_transpose(
            subject,
            forward,
            transpose,
            &self.body.left_probe,
            &self.body.right_probe,
            self.body.tolerance,
        )?;
        require_report(self, &expected)?;
        Ok(ValidatedVerification {
            accepted: expected.body.comparison.accepted,
        })
    }
}

impl ConstraintWorkReport {
    /// Recompute prolongation, restriction, work, comparison, and identity.
    pub fn validate(
        &self,
        constraints: &ConstraintSet,
    ) -> Result<ValidatedVerification, FinitumError> {
        let expected = check_constraint_work(
            constraints,
            &self.body.unconstrained,
            &self.body.physical_residual,
            self.body.tolerance,
        )?;
        require_report(self, &expected)?;
        Ok(ValidatedVerification {
            accepted: expected.body.comparison.accepted,
        })
    }
}

impl TransferConservationReport {
    /// Recompute interpolation, transpose scatter, work, comparison, and identity.
    pub fn validate(
        &self,
        transfer: &NonmatchingTransfer,
    ) -> Result<ValidatedVerification, FinitumError> {
        let expected = check_transfer_conservation(
            transfer,
            &self.body.source_values,
            &self.body.target_residual,
            &self.body.target_weights,
            self.body.tolerance,
        )?;
        require_report(self, &expected)?;
        Ok(ValidatedVerification {
            accepted: expected.body.comparison.accepted,
        })
    }
}

impl ExactSequenceCheckReport {
    /// Recompute dimensional completeness, products, ranks, acceptance, and identity.
    pub fn validate(
        &self,
        sequence: &ExactSequence,
    ) -> Result<ValidatedVerification, FinitumError> {
        let expected = check_exact_sequence(self.body.dimension, sequence)?;
        require_report(self, &expected)?;
        Ok(ValidatedVerification {
            accepted: expected.body.accepted,
        })
    }
}

impl MeshRefinementCheckReport {
    /// Recompute mesh identities, diameters, convergence fit, acceptance, and report identity.
    pub fn validate(
        &self,
        samples: &[MeshRefinementSample<'_>],
    ) -> Result<ValidatedVerification, FinitumError> {
        let expected = check_mesh_refinement(samples, self.body.minimum_required_order)?;
        require_report(self, &expected)?;
        Ok(ValidatedVerification {
            accepted: expected.body.accepted,
        })
    }
}

pub fn check_nodal_patch(
    mesh: &Mesh,
    component_count: usize,
    nodal_values: &[f64],
    tolerance: ComparisonTolerance,
    mut exact: impl FnMut(&[f64]) -> Vec<f64>,
) -> Result<PatchCheckReport, FinitumError> {
    if component_count == 0 {
        return Err(invalid("patch component count must be nonzero"));
    }
    let expected_len = mesh
        .vertices()
        .len()
        .checked_mul(component_count)
        .ok_or_else(|| invalid("patch value extent overflows usize"))?;
    if nodal_values.len() != expected_len {
        return Err(invalid(format!(
            "patch values contain {} entries, expected {expected_len}",
            nodal_values.len()
        )));
    }
    let mut exact_values = Vec::with_capacity(expected_len);
    for (vertex, coordinates) in mesh.vertices().iter().enumerate() {
        let values = exact(coordinates);
        if values.len() != component_count {
            return Err(invalid(format!(
                "exact patch field returned {} components at vertex {vertex}, expected {component_count}",
                values.len()
            )));
        }
        exact_values.extend(values);
    }
    let body = PatchCheckBody {
        tolerance,
        coordinates: mesh.vertices().to_vec(),
        component_count,
        candidate_values: nodal_values.to_vec(),
        comparison: compare(nodal_values, &exact_values, tolerance)?,
        exact_values,
    };
    make_report(
        VerificationCheckKind::NodalPatch,
        VerificationSubject::from_serializable("concrete-mesh", mesh)?,
        body,
    )
    .map(|(header, body)| PatchCheckReport { header, body })
}

pub fn check_realization_agreement(
    plan: &RealizationPlan,
    probe: &[f64],
    lane_width: usize,
    tolerance: ComparisonTolerance,
) -> Result<RealizationAgreementReport, FinitumError> {
    if probe.len() != plan.dimension() || probe.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "realization probe must contain {} finite values",
            plan.dimension()
        )));
    }
    let context = EvaluationContext::reproducible();
    let matrix_free_output = apply(&plan.matrix_free(), &context, probe)?;
    let assembled_output = apply(&plan.assemble()?, &context, probe)?;
    let element_assembled_output = apply(&plan.element_assembly(lane_width)?, &context, probe)?;
    let partial_assembled_output = apply(&plan.partial_assembly(lane_width)?, &context, probe)?;
    let body = RealizationAgreementBody {
        tolerance,
        lane_width,
        probe: probe.to_vec(),
        assembled: compare(&matrix_free_output, &assembled_output, tolerance)?,
        element_assembled: compare(&matrix_free_output, &element_assembled_output, tolerance)?,
        partial_assembled: compare(&matrix_free_output, &partial_assembled_output, tolerance)?,
        matrix_free_output,
        assembled_output,
        element_assembled_output,
        partial_assembled_output,
    };
    let subject = VerificationSubject {
        identity: "realization-plan".into(),
        digest: plan.digest().clone(),
    };
    make_report(VerificationCheckKind::RealizationAgreement, subject, body)
        .map(|(header, body)| RealizationAgreementReport { header, body })
}

pub fn check_global_transpose(
    subject: VerificationSubject,
    forward: &dyn LinearOperator,
    transpose: &dyn LinearOperator,
    left: &[f64],
    right: &[f64],
    tolerance: ComparisonTolerance,
) -> Result<GlobalTransposeWorkReport, FinitumError> {
    if forward.rows() != transpose.columns() || forward.columns() != transpose.rows() {
        return Err(invalid(
            "global transpose operator dimensions do not reverse",
        ));
    }
    if left.len() != forward.rows() || right.len() != forward.columns() {
        return Err(invalid("global transpose probes have incorrect dimensions"));
    }
    let context = EvaluationContext::reproducible();
    let forward_action = apply(forward, &context, right)?;
    let transpose_action = apply(transpose, &context, left)?;
    let forward_work = dot(left, &forward_action)?;
    let transpose_work = dot(right, &transpose_action)?;
    let body = GlobalTransposeWorkBody {
        tolerance,
        left_probe: left.to_vec(),
        right_probe: right.to_vec(),
        forward_action,
        transpose_action,
        forward_work,
        transpose_work,
        comparison: compare(&[forward_work], &[transpose_work], tolerance)?,
    };
    make_report(VerificationCheckKind::GlobalTransposeWork, subject, body)
        .map(|(header, body)| GlobalTransposeWorkReport { header, body })
}

pub fn check_constraint_work(
    constraints: &ConstraintSet,
    unconstrained: &[f64],
    physical_residual: &[f64],
    tolerance: ComparisonTolerance,
) -> Result<ConstraintWorkReport, FinitumError> {
    let prolonged = constraints.expand_homogeneous(unconstrained)?;
    let restricted = constraints.restrict_transpose(physical_residual)?;
    let forward_work = dot(&prolonged, physical_residual)?;
    let transpose_work = dot(unconstrained, &restricted)?;
    let body = ConstraintWorkBody {
        tolerance,
        unconstrained: unconstrained.to_vec(),
        physical_residual: physical_residual.to_vec(),
        prolonged,
        restricted,
        forward_work,
        transpose_work,
        comparison: compare(&[forward_work], &[transpose_work], tolerance)?,
    };
    let subject = VerificationSubject::from_serializable("affine-constraint-set", constraints)?;
    make_report(VerificationCheckKind::ConstraintWork, subject, body)
        .map(|(header, body)| ConstraintWorkReport { header, body })
}

pub fn check_transfer_conservation(
    transfer: &NonmatchingTransfer,
    source_values: &[f64],
    target_residual: &[f64],
    target_weights: &[f64],
    tolerance: ComparisonTolerance,
) -> Result<TransferConservationReport, FinitumError> {
    let target_values = transfer.apply(source_values)?;
    let source_residual = transfer.apply_weighted_transpose(target_residual, target_weights)?;
    let weighted_target = target_residual
        .iter()
        .zip(target_weights)
        .map(|(v, w)| v * w)
        .collect::<Vec<_>>();
    let forward_work = dot(&target_values, &weighted_target)?;
    let transpose_work = dot(source_values, &source_residual)?;
    let body = TransferConservationBody {
        tolerance,
        source_values: source_values.to_vec(),
        target_residual: target_residual.to_vec(),
        target_weights: target_weights.to_vec(),
        target_values,
        source_residual,
        forward_work,
        transpose_work,
        comparison: compare(&[forward_work], &[transpose_work], tolerance)?,
    };
    let subject = VerificationSubject::from_serializable("nonmatching-transfer", transfer)?;
    make_report(VerificationCheckKind::TransferConservation, subject, body)
        .map(|(header, body)| TransferConservationReport { header, body })
}

pub fn check_exact_sequence(
    dimension: usize,
    sequence: &ExactSequence,
) -> Result<ExactSequenceCheckReport, FinitumError> {
    if !(2..=3).contains(&dimension) {
        return Err(invalid("exact-sequence checks require dimension 2 or 3"));
    }
    let expected_stage_count = if dimension == 2 { 2 } else { 3 };
    let observed_stage_count = 2 + usize::from(sequence.divergence.is_some());
    let stage_complete = (dimension == 2) == sequence.divergence.is_none();
    let gradient_rank = sequence.gradient.rank();
    let curl_rank = sequence.curl.rank();
    let divergence_rank = sequence.divergence.as_ref().map(|value| value.rank());
    let curl_gradient_zero = sequence.curl.product_is_zero(&sequence.gradient);
    let exact_at_edges = gradient_rank + curl_rank == sequence.gradient.rows();
    let divergence_curl_zero = sequence
        .divergence
        .as_ref()
        .map(|d| d.product_is_zero(&sequence.curl));
    let exact_at_facets = sequence
        .divergence
        .as_ref()
        .map(|d| curl_rank + d.rank() == sequence.curl.rows());
    let accepted = stage_complete
        && curl_gradient_zero
        && exact_at_edges
        && divergence_curl_zero.unwrap_or(dimension == 2)
        && exact_at_facets.unwrap_or(dimension == 2);
    let body = ExactSequenceCheckBody {
        dimension,
        expected_stage_count,
        observed_stage_count,
        stage_complete,
        gradient_shape: [sequence.gradient.rows(), sequence.gradient.columns()],
        curl_shape: [sequence.curl.rows(), sequence.curl.columns()],
        divergence_shape: sequence
            .divergence
            .as_ref()
            .map(|v| [v.rows(), v.columns()]),
        gradient_rank,
        curl_rank,
        divergence_rank,
        curl_gradient_zero,
        divergence_curl_zero,
        exact_at_edges,
        exact_at_facets,
        accepted,
    };
    let subject =
        VerificationSubject::from_serializable("exact-sequence-incidence", &(dimension, sequence))?;
    make_report(VerificationCheckKind::ExactSequence, subject, body)
        .map(|(header, body)| ExactSequenceCheckReport { header, body })
}

pub fn check_mesh_refinement(
    samples: &[MeshRefinementSample<'_>],
    minimum_required_order: f64,
) -> Result<MeshRefinementCheckReport, FinitumError> {
    if !minimum_required_order.is_finite() {
        return Err(invalid("minimum required convergence order must be finite"));
    }
    let levels = samples
        .iter()
        .map(|sample| {
            Ok(MeshRefinementLevel {
                mesh_digest: digest(sample.mesh)?,
                maximum_cell_diameter: maximum_cell_diameter(sample.mesh)?,
                error: sample.error,
            })
        })
        .collect::<Result<Vec<_>, FinitumError>>()?;
    let convergence = estimate_convergence_order(
        &levels
            .iter()
            .map(|level| ConvergenceSample {
                resolution: level.maximum_cell_diameter,
                error: level.error,
            })
            .collect::<Vec<_>>(),
    )
    .map_err(numeric)?;
    let accepted = convergence.minimum_pair_order >= minimum_required_order;
    let body = MeshRefinementCheckBody {
        minimum_required_order,
        levels,
        convergence,
        accepted,
    };
    let subject = VerificationSubject::from_serializable(
        "mesh-refinement-sequence",
        &body
            .levels
            .iter()
            .map(|l| &l.mesh_digest)
            .collect::<Vec<_>>(),
    )?;
    make_report(VerificationCheckKind::MeshRefinement, subject, body)
        .map(|(header, body)| MeshRefinementCheckReport { header, body })
}

fn make_report<T: Serialize>(
    kind: VerificationCheckKind,
    subject: VerificationSubject,
    body: T,
) -> Result<(VerificationReportHeader, T), FinitumError> {
    let header = header(kind, subject, &body)?;
    Ok((header, body))
}

fn header(
    kind: VerificationCheckKind,
    subject: VerificationSubject,
    body: &impl Serialize,
) -> Result<VerificationReportHeader, FinitumError> {
    subject.validate()?;
    #[derive(Serialize)]
    struct Identity<'a, T> {
        schema: &'static str,
        check_kind: VerificationCheckKind,
        subject: &'a VerificationSubject,
        body: &'a T,
    }
    let report_digest = digest(&Identity {
        schema: VERIFICATION_REPORT_SCHEMA,
        check_kind: kind,
        subject: &subject,
        body,
    })?;
    Ok(VerificationReportHeader {
        schema: VERIFICATION_REPORT_SCHEMA.into(),
        check_kind: kind,
        subject,
        report_digest,
    })
}

fn require_report<T: PartialEq>(actual: &T, expected: &T) -> Result<(), FinitumError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(
            "verification report does not match recomputed source, inputs, outputs, or identity",
        ))
    }
}

fn digest(value: &impl Serialize) -> Result<Digest, FinitumError> {
    let value = serde_json::to_value(value).map_err(|error| invalid(error.to_string()))?;
    let mut bytes = Vec::new();
    write_canonical_json(&value, &mut bytes)?;
    Ok(Digest::blake3(&bytes))
}

fn write_canonical_json(
    value: &serde_json::Value,
    output: &mut Vec<u8>,
) -> Result<(), FinitumError> {
    match value {
        serde_json::Value::Null => output.extend_from_slice(b"null"),
        serde_json::Value::Bool(value) => {
            output.extend_from_slice(if *value { b"true" } else { b"false" })
        }
        serde_json::Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
        serde_json::Value::String(value) => {
            output.extend(serde_json::to_vec(value).map_err(|error| invalid(error.to_string()))?)
        }
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            output.push(b'{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend(serde_json::to_vec(key).map_err(|error| invalid(error.to_string()))?);
                output.push(b':');
                write_canonical_json(&values[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn maximum_cell_diameter(mesh: &Mesh) -> Result<f64, FinitumError> {
    if mesh.cells().is_empty() {
        return Err(invalid("mesh-refinement checks require nonempty meshes"));
    }
    let mut maximum = 0.0_f64;
    for cell in mesh.cells() {
        for a in 0..cell.vertices.len() {
            for b in a + 1..cell.vertices.len() {
                let left = &mesh.vertices()[cell.vertices[a].0];
                let right = &mesh.vertices()[cell.vertices[b].0];
                maximum = maximum.max(
                    left.iter()
                        .zip(right)
                        .map(|(l, r)| (l - r) * (l - r))
                        .sum::<f64>()
                        .sqrt(),
                );
            }
        }
    }
    if maximum.is_finite() && maximum > 0.0 {
        Ok(maximum)
    } else {
        Err(invalid(
            "mesh maximum cell diameter must be finite and positive",
        ))
    }
}

fn apply(
    operator: &dyn LinearOperator,
    context: &EvaluationContext,
    input: &[f64],
) -> Result<Vec<f64>, FinitumError> {
    let mut output = vec![0.0; operator.rows()];
    operator
        .apply(context, input, &mut output)
        .map_err(numeric)?;
    Ok(output)
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, FinitumError> {
    if left.len() != right.len() || left.iter().chain(right).any(|v| !v.is_finite()) {
        return Err(invalid(
            "work vectors must have equal length and finite values",
        ));
    }
    let value = left.iter().zip(right).map(|(l, r)| l * r).sum();
    if f64::is_finite(value) {
        Ok(value)
    } else {
        Err(invalid("work inner product is not finite"))
    }
}

fn compare(
    left: &[f64],
    right: &[f64],
    tolerance: ComparisonTolerance,
) -> Result<ComparisonReport, FinitumError> {
    check_solve_strategy_agreement(left, right, tolerance).map_err(numeric)
}

fn numeric(error: methodus::NumericError) -> FinitumError {
    invalid(error.to_string())
}
fn invalid(message: impl Into<String>) -> FinitumError {
    FinitumError::InvalidRealization(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell, VertexId};

    #[test]
    fn rehashed_inconsistent_acceptance_is_refused() {
        let mesh = Mesh::new(
            1,
            vec![vec![0.0], vec![1.0]],
            vec![Cell {
                vertices: vec![VertexId(0), VertexId(1)],
            }],
        )
        .unwrap();
        let mut report = check_nodal_patch(
            &mesh,
            1,
            &[2.0, 4.0],
            ComparisonTolerance {
                absolute: 1.0e-14,
                relative: 1.0e-14,
            },
            |point| vec![2.0 + 2.0 * point[0]],
        )
        .unwrap();
        report.body.comparison.accepted = false;
        report.header = header(
            VerificationCheckKind::NodalPatch,
            report.header.subject.clone(),
            &report.body,
        )
        .unwrap();
        assert!(report.validate(&mesh).is_err());

        let mut zero_components: PatchCheckReport =
            serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        zero_components.body.component_count = 0;
        zero_components.header = header(
            VerificationCheckKind::NodalPatch,
            zero_components.header.subject.clone(),
            &zero_components.body,
        )
        .unwrap();
        assert!(matches!(
            zero_components.validate(&mesh),
            Err(FinitumError::InvalidRealization(message))
                if message.contains("component count must be nonzero")
        ));
    }
}
