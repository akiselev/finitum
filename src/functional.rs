//! Compiler/kernel-owned point evaluation and Finitum-owned cell functional accumulation.
//! Provider callbacks implement only external primitives; scientific expressions and their
//! argument graphs execute the compiler's kernels. Physical weights are applied exactly once.
use crate::realization::{
    BoundBundle, bind_kernels, component_count, execute, execute_jvp_values, execute_primal_values,
    execute_vjp_values, operand_values,
};
use crate::{FinitumError, InputEvaluationError, InputLocation, InputOrigin, PointEvaluation};
use scientia::{
    DerivativeEvaluation, ExprId, InputSourceRequirement, PointExpressionKernels,
    PointExpressionNode, ProviderId, QFunctionInput, SymbolId, TensorInputId,
};
use std::{collections::BTreeMap, sync::Arc};

/// Identity of an external primitive or independently varied point input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptureKey {
    Provider(ProviderId),
    Input(SymbolId),
}
type ValueCallback =
    dyn Fn(&InputLocation, &[Vec<f64>]) -> Result<Vec<f64>, InputEvaluationError> + Send + Sync;
type DirectionCallback = dyn Fn(&InputLocation, &[Vec<f64>], &[Vec<f64>]) -> Result<Vec<f64>, InputEvaluationError>
    + Send
    + Sync;
/// Explicit derivative contract of a provider primitive.
#[derive(Clone)]
pub enum PointDerivative {
    Available(Arc<DirectionCallback>),
    ProvablyZero { reason: String },
    Frozen { reason: String },
    Unavailable { reason: String },
}
impl std::fmt::Debug for PointDerivative {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Available(_) => f.write_str("Available"),
            Self::ProvablyZero { reason } => f.debug_tuple("ProvablyZero").field(reason).finish(),
            Self::Frozen { reason } => f.debug_tuple("Frozen").field(reason).finish(),
            Self::Unavailable { reason } => f.debug_tuple("Unavailable").field(reason).finish(),
        }
    }
}
impl PointDerivative {
    pub fn available(
        f: impl Fn(&InputLocation, &[Vec<f64>], &[Vec<f64>]) -> Result<Vec<f64>, InputEvaluationError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::Available(Arc::new(f))
    }
}
/// A primitive's value and argument derivative, with stable attribution.
#[derive(Clone)]
pub struct PointProvider {
    pub identity: String,
    pub origin: InputOrigin,
    value: Arc<ValueCallback>,
    pub derivative: PointDerivative,
}
impl std::fmt::Debug for PointProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointProvider")
            .field("identity", &self.identity)
            .field("derivative", &self.derivative)
            .finish_non_exhaustive()
    }
}
impl PointProvider {
    pub fn new(
        identity: impl Into<String>,
        origin: InputOrigin,
        value: impl Fn(&InputLocation, &[Vec<f64>]) -> Result<Vec<f64>, InputEvaluationError>
        + Send
        + Sync
        + 'static,
        derivative: PointDerivative,
    ) -> Self {
        Self {
            identity: identity.into(),
            origin,
            value: Arc::new(value),
            derivative,
        }
    }
    fn direction(
        &self,
        location: &InputLocation,
        args: &[Vec<f64>],
        directions: &[Vec<f64>],
        extent: usize,
    ) -> Result<Vec<f64>, FinitumError> {
        let value = match &self.derivative {
            PointDerivative::Available(callback) => {
                callback(location, args, directions).map_err(|e| located(e, location))?
            }
            PointDerivative::ProvablyZero { .. } | PointDerivative::Frozen { .. } => {
                vec![0.0; extent]
            }
            PointDerivative::Unavailable { reason } => {
                return Err(located(
                    InputEvaluationError::new(
                        "POINT_TANGENT_UNAVAILABLE",
                        self.origin.clone(),
                        reason,
                    ),
                    location,
                ));
            }
        };
        check_values(&value, extent)?;
        Ok(value)
    }
}
/// Independent captures are supplied explicitly at evaluation, including their directions.
#[derive(Clone, Debug)]
pub enum CaptureBinding {
    Provider(PointProvider),
    Independent,
}
/// A semantic field leaf; local kernel input identifiers are never shared across graph nodes.
#[derive(Clone, Debug)]
pub struct PointFieldValue {
    pub symbol: SymbolId,
    pub derivative: DerivativeEvaluation,
    pub values: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct PointExpressionPoint {
    pub location: InputLocation,
    pub fields: Vec<PointFieldValue>,
    pub captures: BTreeMap<CaptureKey, Vec<f64>>,
}
impl PointExpressionPoint {
    /// Adapt an operator callback point using that operator integral's own input bindings.
    pub fn from_evaluation(
        point: &PointEvaluation,
        inputs: &[QFunctionInput],
    ) -> Result<Self, FinitumError> {
        let mut fields = Vec::new();
        for input in inputs
            .iter()
            .filter(|input| input.source == InputSourceRequirement::Basis)
        {
            let values = point.input_values(input.id).ok_or_else(|| {
                invalid(format!("callback point lacks active input {:?}", input.id))
            })?;
            fields.push(PointFieldValue {
                symbol: input.binding.symbol,
                derivative: input.binding.evaluation.derivative,
                values: values.to_vec(),
            });
        }
        Ok(Self {
            location: InputLocation {
                cell: Some(point.cell),
                point: point.coordinates.clone(),
                time: Some(point.time),
            },
            fields,
            captures: point
                .bound
                .iter()
                .map(|input| (CaptureKey::Input(input.symbol), input.values.clone()))
                .collect(),
        })
    }
    fn field(&self, input: &QFunctionInput) -> Result<&[f64], FinitumError> {
        self.fields
            .iter()
            .find(|value| {
                value.symbol == input.binding.symbol
                    && value.derivative == input.binding.evaluation.derivative
            })
            .map(|value| value.values.as_slice())
            .ok_or_else(|| {
                invalid(format!(
                    "point lacks {:?} of field {}",
                    input.binding.evaluation.derivative, input.binding.symbol
                ))
            })
    }
}
#[derive(Clone, Debug, Default)]
pub struct PointPullback {
    pub fields: Vec<PointFieldValue>,
    pub captures: BTreeMap<CaptureKey, Vec<f64>>,
}
#[derive(Clone, Debug)]
struct Node {
    source: PointExpressionNode,
    bundle: BoundBundle,
    parameter_reverse: malleus::Executable,
}
impl Node {
    fn parameter_cotangents(
        &self,
        inputs: &BTreeMap<TensorInputId, Vec<f64>>,
        seed: &[f64],
    ) -> Result<BTreeMap<TensorInputId, Vec<f64>>, FinitumError> {
        let product = &self.source.parameter_vjp[0];
        let by_operand = self
            .bundle
            .bundle
            .primal_inputs
            .iter()
            .map(|binding| (binding.operand, binding.input))
            .collect::<BTreeMap<_, _>>();
        let mut values = BTreeMap::new();
        for remap in &product.primal_operands {
            if let Some(input) = by_operand.get(&remap.primal) {
                values.insert(remap.derivative, inputs[input].clone());
            }
        }
        let dependent = product
            .dependent_operands
            .first()
            .ok_or_else(|| invalid("capture VJP has no dependent operand"))?;
        values.insert(dependent.derivative, seed.to_vec());
        let buffers = execute(&self.parameter_reverse, &values)?;
        let mut result = BTreeMap::new();
        for pair in &product.independent_operands {
            let input = by_operand
                .get(&pair.primal)
                .ok_or_else(|| invalid("capture VJP operand lacks primal input binding"))?;
            let values = operand_values(&self.parameter_reverse, &buffers, pair.derivative)?;
            add(result.entry(*input).or_default(), &values)?;
        }
        Ok(result)
    }
}
/// Reusable executable graph over the compiler's existing point kernels.
#[derive(Clone, Debug)]
pub struct BoundPointExpression {
    kernels: PointExpressionKernels,
    nodes: BTreeMap<ExprId, Node>,
    bindings: BTreeMap<CaptureKey, CaptureBinding>,
}
fn invalid(message: impl Into<String>) -> FinitumError {
    FinitumError::InvalidRealization(message.into())
}
fn located(mut error: InputEvaluationError, location: &InputLocation) -> FinitumError {
    if error.location.is_none() {
        error.location = Some(Box::new(location.clone()));
    }
    error.into()
}
fn check_result(
    values: &[f64],
    extent: usize,
    point: &PointExpressionPoint,
    expression: ExprId,
) -> Result<(), FinitumError> {
    check_values(values, extent).map_err(|error| {
        located(
            InputEvaluationError::new(
                "POINT_RESULT_INVALID",
                InputOrigin::ExpressionPath(format!("expression/{expression}")),
                error.to_string(),
            ),
            &point.location,
        )
    })
}
fn check_values(values: &[f64], extent: usize) -> Result<(), FinitumError> {
    if values.len() != extent || values.iter().any(|v| !v.is_finite()) {
        Err(invalid(format!(
            "point result must contain {extent} finite components"
        )))
    } else {
        Ok(())
    }
}
fn key(capture: &scientia::PointCapture) -> CaptureKey {
    capture
        .provider
        .map(CaptureKey::Provider)
        .unwrap_or(CaptureKey::Input(capture.symbol))
}
fn add(to: &mut Vec<f64>, value: &[f64]) -> Result<(), FinitumError> {
    if to.is_empty() {
        to.resize(value.len(), 0.0)
    }
    if to.len() != value.len() {
        return Err(invalid("point cotangent extent mismatch"));
    }
    for (a, b) in to.iter_mut().zip(value) {
        *a += b;
    }
    Ok(())
}
#[derive(Clone)]
struct Tape {
    inputs: BTreeMap<TensorInputId, Vec<f64>>,
    value: Vec<f64>,
    direction: Vec<f64>,
}
impl BoundPointExpression {
    pub fn new(
        kernels: PointExpressionKernels,
        bindings: BTreeMap<CaptureKey, CaptureBinding>,
    ) -> Result<Self, FinitumError> {
        kernels
            .validate_identity()
            .map_err(|e| FinitumError::ArtifactMismatch(e.to_string()))?;
        for binding in bindings.values() {
            if let CaptureBinding::Provider(provider) = binding {
                if provider.identity.trim().is_empty() {
                    return Err(invalid("point provider requires identity"));
                }
                let reason = match &provider.derivative {
                    PointDerivative::Available(_) => None,
                    PointDerivative::ProvablyZero { reason }
                    | PointDerivative::Frozen { reason }
                    | PointDerivative::Unavailable { reason } => Some(reason),
                };
                if reason.is_some_and(|reason| reason.trim().is_empty()) {
                    return Err(invalid("provider derivative disposition requires reason"));
                }
            }
        }
        if kernels.root.expression != kernels.expression
            || kernels
                .arguments
                .iter()
                .any(|(id, node)| *id != node.expression || *id == kernels.expression)
        {
            return Err(invalid("point graph node identity/key mismatch"));
        }
        let mut nodes = BTreeMap::new();
        for source in std::iter::once(&kernels.root).chain(kernels.arguments.values()) {
            if source.factorization.integrals.len() != 1
                || source.kernels.bundles.len() != 1
                || source.parameter_vjp.len() != 1
            {
                return Err(invalid(
                    "point graph node needs one integral and one bundle",
                ));
            }
            for capture in &source.captures {
                if capture.provider.is_some()
                    && matches!(
                        bindings.get(&key(capture)),
                        Some(CaptureBinding::Independent)
                    )
                {
                    let constant_arguments =
                        capture.arguments.iter().all(|argument| {
                            kernels.arguments.get(argument).is_some_and(|node| {
                                node.captures.is_empty()
                                    && node.factorization.integrals.iter().all(|integral| {
                                        integral.primal.inputs.iter().all(|input| {
                                            input.source != InputSourceRequirement::Basis
                                        })
                                    })
                            })
                        });
                    if !constant_arguments {
                        return Err(invalid(
                            "POINT_PROVIDER_DEPENDENCY: an independent provider binding cannot drop state or capture dependencies of its arguments; bind an explicit provider derivative disposition",
                        ));
                    }
                }
                if !bindings.contains_key(&key(capture)) {
                    return Err(invalid(format!(
                        "point capture {:?} is not bound",
                        key(capture)
                    )));
                }
            }
            let mut bundles = bind_kernels(&source.factorization, source.kernels.clone())?;
            let bundle = bundles
                .pop_first()
                .ok_or_else(|| invalid("point node has no executable"))?
                .1;
            nodes.insert(
                source.expression,
                Node {
                    source: source.clone(),
                    bundle,
                    parameter_reverse: malleus::Executable::reference(
                        malleus::validate(source.parameter_vjp[0].kernel.clone())
                            .map_err(|error| FinitumError::KernelValidation(error.to_string()))?,
                    ),
                },
            );
        }
        fn visit(
            id: ExprId,
            nodes: &BTreeMap<ExprId, Node>,
            active: &mut std::collections::BTreeSet<ExprId>,
            done: &mut std::collections::BTreeSet<ExprId>,
        ) -> Result<(), FinitumError> {
            if done.contains(&id) {
                return Ok(());
            }
            if !active.insert(id) {
                return Err(invalid("point argument graph contains a cycle"));
            }
            let node = nodes
                .get(&id)
                .ok_or_else(|| invalid("point argument graph names missing node"))?;
            for argument in node
                .source
                .captures
                .iter()
                .flat_map(|capture| &capture.arguments)
            {
                visit(*argument, nodes, active, done)?;
            }
            active.remove(&id);
            done.insert(id);
            Ok(())
        }
        let mut done = std::collections::BTreeSet::new();
        for id in nodes.keys() {
            visit(
                *id,
                &nodes,
                &mut std::collections::BTreeSet::new(),
                &mut done,
            )?;
        }
        Ok(Self {
            kernels,
            nodes,
            bindings,
        })
    }
    pub fn kernels(&self) -> &PointExpressionKernels {
        &self.kernels
    }
    /// Identity covers compiler payload, primitive identities, and every derivative disposition.
    pub fn identity(&self) -> scientia::Digest {
        let bindings = self
            .bindings
            .iter()
            .map(|(key, binding)| {
                let (identity, disposition, origin) = match binding {
                    CaptureBinding::Independent => (
                        "independent".to_owned(),
                        "available".to_owned(),
                        String::new(),
                    ),
                    CaptureBinding::Provider(provider) => (
                        provider.identity.clone(),
                        format!("{:?}", provider.derivative),
                        provider.origin.to_string(),
                    ),
                };
                (format!("{key:?}"), identity, disposition, origin)
            })
            .collect::<Vec<_>>();
        scientia::Digest::blake3(
            &serde_json::to_vec(&(
                "finitum-point-expression/1",
                &self.kernels.artifact_digest,
                bindings,
            ))
            .expect("serializable identity"),
        )
    }
    pub fn bindings(&self) -> &BTreeMap<CaptureKey, CaptureBinding> {
        &self.bindings
    }
    /// Every semantic basis requirement, including leaves of nested provider arguments.
    pub fn field_inputs(&self) -> Vec<QFunctionInput> {
        let mut inputs = Vec::<QFunctionInput>::new();
        for node in self.nodes.values() {
            for input in &node.source.factorization.integrals[0].primal.inputs {
                if input.source == InputSourceRequirement::Basis
                    && !inputs.iter().any(|other| other.binding == input.binding)
                {
                    inputs.push(input.clone());
                }
            }
        }
        inputs
    }
    pub fn independent_captures(&self) -> impl Iterator<Item = CaptureKey> + '_ {
        self.bindings.iter().filter_map(|(key, binding)| {
            matches!(binding, CaptureBinding::Independent).then_some(*key)
        })
    }
    fn forward(
        &self,
        id: ExprId,
        point: &PointExpressionPoint,
        direction: Option<&PointExpressionPoint>,
        tape: &mut BTreeMap<ExprId, Tape>,
        order: &mut Vec<ExprId>,
    ) -> Result<(), FinitumError> {
        if tape.contains_key(&id) {
            return Ok(());
        }
        let node = self
            .nodes
            .get(&id)
            .ok_or_else(|| invalid("point graph argument is absent"))?;
        let program = &node.source.factorization.integrals[0].primal;
        let mut values = BTreeMap::new();
        let mut directions = BTreeMap::new();
        for input in &program.inputs {
            let (value, tangent) = if input.source == InputSourceRequirement::Basis {
                (
                    point.field(input)?.to_vec(),
                    match direction {
                        Some(d) => d.field(input)?.to_vec(),
                        None => vec![0.0; component_count(&input.shape)?],
                    },
                )
            } else {
                let capture = node
                    .source
                    .captures
                    .iter()
                    .find(|capture| capture.input == input.id)
                    .ok_or_else(|| invalid("nonbasis point input has no capture contract"))?;
                let capture_key = key(capture);
                match &self.bindings[&capture_key] {
                    CaptureBinding::Independent => {
                        let value = point
                            .captures
                            .get(&capture_key)
                            .ok_or_else(|| {
                                invalid(format!("independent capture {capture_key:?} missing"))
                            })?
                            .clone();
                        let tangent = match direction {
                            Some(d) => d
                                .captures
                                .get(&capture_key)
                                .ok_or_else(|| {
                                    invalid(format!(
                                        "independent capture {capture_key:?} direction missing"
                                    ))
                                })?
                                .clone(),
                            None => vec![0.0; value.len()],
                        };
                        (value, tangent)
                    }
                    CaptureBinding::Provider(provider) => {
                        for argument in &capture.arguments {
                            self.forward(*argument, point, direction, tape, order)?;
                        }
                        let args = capture
                            .arguments
                            .iter()
                            .map(|id| tape[id].value.clone())
                            .collect::<Vec<_>>();
                        let value = (provider.value)(&point.location, &args)
                            .map_err(|e| located(e, &point.location))?;
                        let tangent = if direction.is_some() {
                            let ds = capture
                                .arguments
                                .iter()
                                .map(|id| tape[id].direction.clone())
                                .collect::<Vec<_>>();
                            provider.direction(&point.location, &args, &ds, value.len())?
                        } else {
                            vec![0.0; value.len()]
                        };
                        (value, tangent)
                    }
                }
            };
            check_values(&value, component_count(&input.shape)?)?;
            check_values(&tangent, value.len())?;
            values.insert(input.id, value);
            directions.insert(input.id, tangent);
        }
        let value = execute_primal_values(&node.bundle, &values)?;
        let tangent = if direction.is_some() {
            execute_jvp_values(&node.bundle, &values, &directions)?
        } else {
            vec![0.0; value.len()]
        };
        check_result(&value, value.len(), point, id)?;
        check_result(&tangent, value.len(), point, id)?;
        tape.insert(
            id,
            Tape {
                inputs: values,
                value,
                direction: tangent,
            },
        );
        order.push(id);
        Ok(())
    }
    pub fn value(&self, point: &PointExpressionPoint) -> Result<Vec<f64>, FinitumError> {
        let mut tape = BTreeMap::new();
        self.forward(
            self.kernels.expression,
            point,
            None,
            &mut tape,
            &mut Vec::new(),
        )?;
        Ok(tape.remove(&self.kernels.expression).unwrap().value)
    }
    pub fn jvp(
        &self,
        point: &PointExpressionPoint,
        direction: &PointExpressionPoint,
    ) -> Result<Vec<f64>, FinitumError> {
        let mut tape = BTreeMap::new();
        self.forward(
            self.kernels.expression,
            point,
            Some(direction),
            &mut tape,
            &mut Vec::new(),
        )?;
        Ok(tape.remove(&self.kernels.expression).unwrap().direction)
    }
    pub fn vjp(
        &self,
        point: &PointExpressionPoint,
        seed: &[f64],
    ) -> Result<PointPullback, FinitumError> {
        let mut tape = BTreeMap::new();
        let mut order = Vec::new();
        self.forward(self.kernels.expression, point, None, &mut tape, &mut order)?;
        check_values(seed, tape[&self.kernels.expression].value.len())?;
        let mut seeds = BTreeMap::from([(self.kernels.expression, seed.to_vec())]);
        let mut result = PointPullback::default();
        for id in order.into_iter().rev() {
            let Some(seed) = seeds.remove(&id) else {
                continue;
            };
            let node = &self.nodes[&id];
            let point_tape = &tape[&id];
            let mut cotangents =
                execute_vjp_values(&node.bundle, &point_tape.inputs, seed.clone())?;
            for (input, contribution) in node.parameter_cotangents(&point_tape.inputs, &seed)? {
                add(cotangents.entry(input).or_default(), &contribution)?;
            }
            for input in &node.source.factorization.integrals[0].primal.inputs {
                let Some(cotangent) = cotangents.get(&input.id) else {
                    continue;
                };
                check_result(cotangent, component_count(&input.shape)?, point, id)?;
                if input.source == InputSourceRequirement::Basis {
                    if let Some(existing) = result.fields.iter_mut().find(|field| {
                        field.symbol == input.binding.symbol
                            && field.derivative == input.binding.evaluation.derivative
                    }) {
                        add(&mut existing.values, cotangent)?;
                    } else {
                        result.fields.push(PointFieldValue {
                            symbol: input.binding.symbol,
                            derivative: input.binding.evaluation.derivative,
                            values: cotangent.clone(),
                        });
                    }
                } else {
                    let capture = node
                        .source
                        .captures
                        .iter()
                        .find(|capture| capture.input == input.id)
                        .ok_or_else(|| invalid("point capture absent"))?;
                    let capture_key = key(capture);
                    match &self.bindings[&capture_key] {
                        CaptureBinding::Independent => {
                            add(result.captures.entry(capture_key).or_default(), cotangent)?
                        }
                        CaptureBinding::Provider(provider) => {
                            let args = capture
                                .arguments
                                .iter()
                                .map(|id| tape[id].value.clone())
                                .collect::<Vec<_>>();
                            // Exact transpose of the primitive's small argument JVP. This is
                            // local argument probing, never a global finite difference.
                            let mut ds =
                                args.iter().map(|a| vec![0.0; a.len()]).collect::<Vec<_>>();
                            // Missing derivatives refuse even when this particular seed is zero.
                            if matches!(provider.derivative, PointDerivative::Unavailable { .. }) {
                                provider.direction(&point.location, &args, &ds, cotangent.len())?;
                            }
                            for (arg_index, arg_id) in capture.arguments.iter().enumerate() {
                                let mut contribution = vec![0.0; args[arg_index].len()];
                                for (component, value) in contribution.iter_mut().enumerate() {
                                    ds[arg_index][component] = 1.0;
                                    let column = provider.direction(
                                        &point.location,
                                        &args,
                                        &ds,
                                        cotangent.len(),
                                    )?;
                                    *value = column.iter().zip(cotangent).map(|(a, b)| a * b).sum();
                                    ds[arg_index][component] = 0.0;
                                }
                                add(seeds.entry(*arg_id).or_default(), &contribution)?;
                            }
                        }
                    }
                }
            }
        }
        for field in &result.fields {
            check_result(
                &field.values,
                field.values.len(),
                point,
                self.kernels.expression,
            )?;
        }
        for values in result.captures.values() {
            check_result(values, values.len(), point, self.kernels.expression)?;
        }
        Ok(result)
    }
}

use crate::realization::CellGeometry;
use crate::sampler::{QuadratureRule, QuadratureView};
use crate::system::{
    FieldElement, apply_field_basis_adjoint, build_field_elements, evaluate_field_basis_input,
};
use crate::{CellId, CoefficientLayout, InstanceId, SysVarId, SystemRealizationPlan};
/// Concrete independent data/design layout. All interpolation and transpose weights are
/// owned by Finitum, independently of the state field's polynomial order.
#[derive(Clone, Debug)]
pub struct FunctionalDesign {
    pub key: CaptureKey,
    pub layout: CoefficientLayout,
    pub components: usize,
}
#[derive(Clone, Debug)]
pub struct FunctionalPullback {
    pub state: Vec<f64>,
    pub rate: Vec<f64>,
    pub design: BTreeMap<CaptureKey, Vec<f64>>,
}
/// Frozen integration rule with an explicit reason; its identity is available in the receipt.
#[derive(Clone, Debug)]
pub struct FunctionalQuadrature {
    pub rule: QuadratureRule,
    pub reason: String,
}
#[derive(Clone, Debug)]
pub struct CellFunctionalPlan {
    realization: SystemRealizationPlan,
    instance: InstanceId,
    expression: BoundPointExpression,
    fields: BTreeMap<SysVarId, FieldElement>,
    inputs: Vec<QFunctionInput>,
    designs: Vec<FunctionalDesign>,
    quadrature: FunctionalQuadrature,
}
impl CellFunctionalPlan {
    pub fn new(
        realization: &SystemRealizationPlan,
        instance: InstanceId,
        expression: BoundPointExpression,
        quadrature: FunctionalQuadrature,
        designs: Vec<FunctionalDesign>,
    ) -> Result<Self, FinitumError> {
        let model = realization
            .instance_system(instance)
            .ok_or_else(|| invalid("functional instance is absent"))?;
        if model.model != expression.kernels.model
            || model.source_semantic_digest != expression.kernels.root.form.source_semantic_digest
        {
            return Err(FinitumError::ArtifactMismatch(
                "functional and realization originate in different models".into(),
            ));
        }
        if expression
            .nodes
            .values()
            .flat_map(|node| &node.source.captures)
            .any(|capture| {
                capture.provider.is_none() && realization.has_bound_input(instance, capture.symbol)
            })
        {
            return Err(FinitumError::UnsupportedRealization("POINT_BOUND_FUNCTIONAL_UNSUPPORTED: functional derivatives through composed input binds require the system bind pullback; they cannot be independent design captures".into()));
        }
        if quadrature.reason.trim().is_empty() {
            return Err(invalid("functional quadrature needs a recorded reason"));
        }
        QuadratureView::new(realization.mesh(), quadrature.rule.clone())?;
        let checked = QuadratureRule::from_table(
            realization.mesh().dimension(),
            quadrature.rule.points.clone(),
        )?;
        if checked.degree != quadrature.rule.degree {
            return Err(invalid(
                "functional quadrature degree does not match its numerical table",
            ));
        }
        let fields = build_field_elements(realization, &quadrature.rule.points)?;
        let inputs = expression.field_inputs();
        for input in &inputs {
            let variable = realization
                .system_ids()
                .variable(instance, input.binding.symbol)
                .ok_or_else(|| {
                    invalid(format!(
                        "functional field {} is not realized in {instance}",
                        input.binding.symbol
                    ))
                })?;
            if !fields.contains_key(&variable) {
                return Err(invalid("functional field has no numerical basis"));
            }
        }
        let expected = expression
            .independent_captures()
            .collect::<std::collections::BTreeSet<_>>();
        let actual = designs
            .iter()
            .map(|design| design.key)
            .collect::<std::collections::BTreeSet<_>>();
        if expected != actual
            || actual.len() != designs.len()
            || designs.iter().any(|design| design.components == 0)
        {
            return Err(invalid(
                "functional design layouts must cover independent captures exactly once",
            ));
        }
        Ok(Self {
            realization: realization.clone(),
            instance,
            expression,
            fields,
            inputs,
            designs,
            quadrature,
        })
    }
    pub fn quadrature(&self) -> &FunctionalQuadrature {
        &self.quadrature
    }
    pub fn identity(&self) -> scientia::Digest {
        let designs = self
            .designs
            .iter()
            .map(|design| {
                (
                    format!("{:?}", design.key),
                    design.layout,
                    design.components,
                )
            })
            .collect::<Vec<_>>();
        scientia::Digest::blake3(
            &serde_json::to_vec(&(
                "finitum-cell-functional/1",
                self.realization.artifact_digest(),
                self.instance.0,
                self.expression.identity(),
                self.quadrature.rule.identity(),
                &self.quadrature.reason,
                designs,
            ))
            .expect("serializable identity"),
        )
    }
    pub fn realization(&self) -> &SystemRealizationPlan {
        &self.realization
    }
    pub fn expression(&self) -> &BoundPointExpression {
        &self.expression
    }
    fn check(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        design: &BTreeMap<CaptureKey, Vec<f64>>,
    ) -> Result<(), FinitumError> {
        if !time.is_finite() {
            return Err(invalid("functional time must be finite"));
        }
        check_values(state, self.realization.layout().extent())?;
        check_values(rate, state.len())?;
        if design.len() != self.designs.len() {
            return Err(invalid("functional design keys differ from plan"));
        }
        for binding in &self.designs {
            let n = binding.layout.dimension_at(
                self.realization.mesh(),
                self.quadrature.rule.points.len(),
                binding.components,
            )?;
            check_values(
                design
                    .get(&binding.key)
                    .ok_or_else(|| invalid("functional design absent"))?,
                n,
            )?;
        }
        Ok(())
    }
    fn point(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        design: &BTreeMap<CaptureKey, Vec<f64>>,
        cell: usize,
        point: usize,
    ) -> Result<PointExpressionPoint, FinitumError> {
        let mesh = self.realization.mesh();
        let geometry = CellGeometry::new(mesh, CellId(cell))?;
        let affine = crate::AffineMap::from_cell(mesh, CellId(cell))?;
        let reference = &self.quadrature.rule.points[point].coordinates;
        let mut fields = Vec::new();
        for input in &self.inputs {
            let variable = self
                .realization
                .system_ids()
                .variable(self.instance, input.binding.symbol)
                .expect("validated functional field");
            let element = &self.fields[&variable];
            let block = self
                .realization
                .layout()
                .block_by_variable(variable)
                .unwrap();
            let vector =
                if input.binding.evaluation.derivative == DerivativeEvaluation::TimeDerivative {
                    rate
                } else {
                    state
                };
            let local = element.dofs.restrictions()[cell]
                .dofs
                .iter()
                .map(|dof| vector[block.offset + dof.0])
                .collect::<Vec<_>>();
            let values = evaluate_field_basis_input(
                element, &geometry, &affine, cell, point, reference, input, &local,
            )?;
            fields.push(PointFieldValue {
                symbol: input.binding.symbol,
                derivative: input.binding.evaluation.derivative,
                values,
            });
        }
        let mut captures = BTreeMap::new();
        for binding in &self.designs {
            let mut values = vec![0.0; binding.components];
            for (entity, weight) in
                binding
                    .layout
                    .weights_at(mesh, &self.quadrature.rule.points, cell, point)?
            {
                for (component, value) in values.iter_mut().enumerate() {
                    *value +=
                        weight * design[&binding.key][entity * binding.components + component];
                }
            }
            captures.insert(binding.key, values);
        }
        Ok(PointExpressionPoint {
            location: InputLocation {
                cell: Some(CellId(cell)),
                point: affine.physical_point(reference)?,
                time: Some(time),
            },
            fields,
            captures,
        })
    }
    pub fn value(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        design: &BTreeMap<CaptureKey, Vec<f64>>,
    ) -> Result<f64, FinitumError> {
        self.check(time, state, rate, design)?;
        let mut result = 0.0;
        let view = QuadratureView::new(self.realization.mesh(), self.quadrature.rule.clone())?;
        for cell in 0..self.realization.mesh().cells().len() {
            for (point, q) in view.cell_points(CellId(cell))?.iter().enumerate() {
                let value = self
                    .expression
                    .value(&self.point(time, state, rate, design, cell, point)?)?;
                check_values(&value, 1)?;
                result += q.weight * value[0];
            }
        }
        check_values(&[result], 1)?;
        Ok(result)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn jvp(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        design: &BTreeMap<CaptureKey, Vec<f64>>,
        state_direction: &[f64],
        rate_direction: &[f64],
        design_direction: &BTreeMap<CaptureKey, Vec<f64>>,
    ) -> Result<f64, FinitumError> {
        self.check(time, state, rate, design)?;
        self.check(time, state_direction, rate_direction, design_direction)?;
        let mut result = 0.0;
        let view = QuadratureView::new(self.realization.mesh(), self.quadrature.rule.clone())?;
        for cell in 0..self.realization.mesh().cells().len() {
            for (point, q) in view.cell_points(CellId(cell))?.iter().enumerate() {
                let base = self.point(time, state, rate, design, cell, point)?;
                let direction = self.point(
                    time,
                    state_direction,
                    rate_direction,
                    design_direction,
                    cell,
                    point,
                )?;
                let value = self.expression.jvp(&base, &direction)?;
                check_values(&value, 1)?;
                result += q.weight * value[0];
            }
        }
        check_values(&[result], 1)?;
        Ok(result)
    }
    pub fn vjp(
        &self,
        time: f64,
        state: &[f64],
        rate: &[f64],
        design: &BTreeMap<CaptureKey, Vec<f64>>,
        seed: f64,
    ) -> Result<FunctionalPullback, FinitumError> {
        self.check(time, state, rate, design)?;
        check_values(&[seed], 1)?;
        let n = self.realization.layout().extent();
        let mut result = FunctionalPullback {
            state: vec![0.0; n],
            rate: vec![0.0; n],
            design: design
                .iter()
                .map(|(key, values)| (*key, vec![0.0; values.len()]))
                .collect(),
        };
        let mesh = self.realization.mesh();
        let view = QuadratureView::new(mesh, self.quadrature.rule.clone())?;
        for cell in 0..mesh.cells().len() {
            let geometry = CellGeometry::new(mesh, CellId(cell))?;
            let affine = crate::AffineMap::from_cell(mesh, CellId(cell))?;
            for (point, q) in view.cell_points(CellId(cell))?.iter().enumerate() {
                let pullback = self.expression.vjp(
                    &self.point(time, state, rate, design, cell, point)?,
                    &[seed * q.weight],
                )?;
                for field in &pullback.fields {
                    let variable = self
                        .realization
                        .system_ids()
                        .variable(self.instance, field.symbol)
                        .ok_or_else(|| invalid("functional pullback has unbound field"))?;
                    let element = &self.fields[&variable];
                    let block = self
                        .realization
                        .layout()
                        .block_by_variable(variable)
                        .unwrap();
                    let restriction = &element.dofs.restrictions()[cell];
                    let mut local = vec![0.0; restriction.dofs.len()];
                    let (target, derivative) =
                        if field.derivative == DerivativeEvaluation::TimeDerivative {
                            (&mut result.rate, DerivativeEvaluation::Value)
                        } else {
                            (&mut result.state, field.derivative)
                        };
                    apply_field_basis_adjoint(
                        element,
                        &geometry,
                        &affine,
                        cell,
                        point,
                        &q.reference,
                        &derivative,
                        &field.values,
                        1.0,
                        &mut local,
                    )?;
                    for (dof, value) in restriction.dofs.iter().zip(local) {
                        target[block.offset + dof.0] += value;
                        if !target[block.offset + dof.0].is_finite() {
                            return Err(located(
                                InputEvaluationError::new(
                                    "FUNCTIONAL_NONFINITE",
                                    InputOrigin::ExpressionPath(format!(
                                        "expression/{}",
                                        self.expression.kernels.expression
                                    )),
                                    "functional state/rate scatter overflow",
                                ),
                                &InputLocation {
                                    cell: Some(CellId(cell)),
                                    point: q.physical.clone(),
                                    time: Some(time),
                                },
                            ));
                        }
                    }
                }
                for binding in &self.designs {
                    if let Some(values) = pullback.captures.get(&binding.key) {
                        check_values(values, binding.components)?;
                        for (entity, weight) in binding.layout.weights_at(
                            mesh,
                            &self.quadrature.rule.points,
                            cell,
                            point,
                        )? {
                            for (component, value) in values.iter().enumerate() {
                                result.design.get_mut(&binding.key).unwrap()
                                    [entity * binding.components + component] += weight * value;
                                if !result.design[&binding.key]
                                    [entity * binding.components + component]
                                    .is_finite()
                                {
                                    return Err(located(
                                        InputEvaluationError::new(
                                            "FUNCTIONAL_NONFINITE",
                                            InputOrigin::ExpressionPath(format!(
                                                "expression/{}",
                                                self.expression.kernels.expression
                                            )),
                                            "functional design scatter overflow",
                                        ),
                                        &InputLocation {
                                            cell: Some(CellId(cell)),
                                            point: q.physical.clone(),
                                            time: Some(time),
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        check_values(&result.state, n)?;
        check_values(&result.rate, n)?;
        for values in result.design.values() {
            check_values(values, values.len())?;
        }
        Ok(result)
    }
}
