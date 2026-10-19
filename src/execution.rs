//! Prepared Malleus execution (W11 F1) and the execution backend choice (W13 A1).
//!
//! Every executable Finitum runs is held next to its `malleus::prepared` form, built once by
//! the struct that owns the executable (never a global cache keyed by address: addresses are
//! reused after drop). Calls bind operands positionally over inline binding tables and a
//! per-thread reusable register file, so a steady-state call allocates nothing inside
//! Malleus. The prepared path is bitwise identical to `Interpreter::run`; a refused call is
//! re-issued through the checked one-shot `PreparedKernel::run`, which reports exactly the
//! interpreter's `ExecutionError`, so `FinitumError::KernelExecution` messages are unchanged.
//!
//! [`ExecutionBackend`] chooses who executes those calls. `Interpreter` is the prepared
//! bytecode above. `Cranelift` (cargo feature `cranelift`) compiles every kernel once per
//! process through a shared digest-keyed `malleus_cranelift::JitCache` and runs the same
//! positional batch calls natively; results are bitwise those of the interpreter, so the
//! backend is in no realization or kernel identity. A kernel the JIT refuses runs on the
//! interpreter and is recorded as a fallback with its reason in [`ExecutionReceipt`]; binding
//! refusals still come from the interpreter's checked path, so error values do not depend on
//! the backend. The backend is captured when a kernel is prepared, from
//! [`with_execution_backend`]'s scope or else the process default ([`EXECUTION_BACKEND_ENV`],
//! `interpreter` when unset).

use crate::FinitumError;
use malleus::prepared::{
    CompositionState, PreparedComposition, PreparedKernel, PreparedState, ReadBinding, WriteBinding,
};
use malleus::{
    BufferBinding, CompositionBinding, Executable, ExecutableComposition, ExecutableModule,
    OperandId,
};
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

/// Operand count up to which binding tables live on the stack.
const INLINE_OPERANDS: usize = 16;

thread_local! {
    static KERNEL_STATE: RefCell<PreparedState> = RefCell::new(PreparedState::new());
    static COMPOSITION_STATE: RefCell<CompositionState> =
        RefCell::new(CompositionState::default());
    static SELECTED_BACKEND: Cell<Option<ExecutionBackend>> = const { Cell::new(None) };
}

/// Who executes Finitum's Malleus kernels. Not part of any identity: every backend computes
/// the interpreter's bits.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBackend {
    /// Malleus prepared bytecode (`PreparedKernel::run_batch`), the reference semantics.
    #[default]
    Interpreter,
    /// Native code from `malleus-cranelift` (cargo feature `cranelift`), compiled once per
    /// process per kernel digest; refused kernels fall back to the interpreter, recorded.
    Cranelift,
}

/// Environment variable naming the process-default backend (`interpreter` or `cranelift`),
/// read once. Unset means `interpreter`; any other value refuses kernel preparation.
pub const EXECUTION_BACKEND_ENV: &str = "FINITUM_EXECUTION_BACKEND";

/// Schema of [`ExecutionReceipt`].
pub const EXECUTION_RECEIPT_SCHEMA: &str = "finitum-execution-receipt/1";

/// Identity of the interpreter backend in receipts.
const INTERPRETER_IDENTITY: &str = "malleus-prepared run_batch";

impl ExecutionBackend {
    /// The name used by [`EXECUTION_BACKEND_ENV`] and serde.
    pub const fn name(self) -> &'static str {
        match self {
            ExecutionBackend::Interpreter => "interpreter",
            ExecutionBackend::Cranelift => "cranelift",
        }
    }

    /// Parses [`Self::name`].
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "interpreter" => Some(ExecutionBackend::Interpreter),
            "cranelift" => Some(ExecutionBackend::Cranelift),
            _ => None,
        }
    }

    /// Whether this build of Finitum can execute the backend (`Cranelift` needs the
    /// `cranelift` feature).
    pub const fn available(self) -> bool {
        match self {
            ExecutionBackend::Interpreter => true,
            ExecutionBackend::Cranelift => cfg!(feature = "cranelift"),
        }
    }

    /// The code generator identity a receipt records: for `Cranelift`, the
    /// `malleus-cranelift` backend version, batch ABI and host CPU features.
    pub fn identity(self) -> String {
        match self {
            ExecutionBackend::Interpreter => INTERPRETER_IDENTITY.to_owned(),
            #[cfg(feature = "cranelift")]
            ExecutionBackend::Cranelift => format!(
                "{} {} {}",
                malleus_cranelift::BACKEND_VERSION,
                malleus_cranelift::BATCH_ABI,
                malleus_cranelift::cpu_features()
            ),
            #[cfg(not(feature = "cranelift"))]
            ExecutionBackend::Cranelift => "unavailable (feature `cranelift` not built)".to_owned(),
        }
    }
}

fn process_default() -> &'static Result<ExecutionBackend, String> {
    static DEFAULT: OnceLock<Result<ExecutionBackend, String>> = OnceLock::new();
    DEFAULT.get_or_init(|| match std::env::var(EXECUTION_BACKEND_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(ExecutionBackend::Interpreter),
        Ok(name) => ExecutionBackend::from_name(&name).ok_or_else(|| {
            format!("{EXECUTION_BACKEND_ENV}=`{name}` names no backend (interpreter, cranelift)")
        }),
        Err(error) => Err(format!("{EXECUTION_BACKEND_ENV} is unreadable: {error}")),
    })
}

/// The backend kernels prepared on this thread now use: the innermost
/// [`with_execution_backend`] scope, else the process default. Refuses
/// (`EXECUTION_BACKEND_UNAVAILABLE`) a backend this build cannot run or a malformed
/// [`EXECUTION_BACKEND_ENV`].
pub fn selected_execution_backend() -> Result<ExecutionBackend, FinitumError> {
    let backend = match SELECTED_BACKEND.with(Cell::get) {
        Some(backend) => backend,
        None => process_default()
            .clone()
            .map_err(FinitumError::ExecutionBackendUnavailable)?,
    };
    if backend.available() {
        Ok(backend)
    } else {
        Err(FinitumError::ExecutionBackendUnavailable(format!(
            "backend `{}` needs Finitum's `cranelift` feature",
            backend.name()
        )))
    }
}

/// Runs `body` with `backend` selected for every kernel prepared on this thread inside it
/// (realization plans, system operators, FV realizations, methods, samplers). Objects keep
/// the backend they were built with; the previous selection is restored on exit or unwind.
pub fn with_execution_backend<R>(backend: ExecutionBackend, body: impl FnOnce() -> R) -> R {
    struct Restore(Option<ExecutionBackend>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SELECTED_BACKEND.with(|selected| selected.set(self.0));
        }
    }
    let _restore = Restore(SELECTED_BACKEND.with(|selected| selected.replace(Some(backend))));
    body()
}

/// Kernels compiled natively in this process (the shared JIT cache size); 0 without the
/// `cranelift` feature.
pub fn native_kernel_cache_len() -> usize {
    #[cfg(feature = "cranelift")]
    {
        jit::cache().len()
    }
    #[cfg(not(feature = "cranelift"))]
    {
        0
    }
}

/// How one kernel executes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KernelEngine {
    /// Selected backend `Interpreter`.
    Interpreter,
    /// Native code from the shared JIT cache.
    Native,
    /// Selected backend `Cranelift`, refused by the JIT: runs on the interpreter.
    Fallback { reason: String },
}

/// One distinct kernel (by `kernel_digest`) of an [`ExecutionReceipt`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExecutionKernelReceipt {
    pub name: String,
    pub kernel_digest: String,
    pub engine: KernelEngine,
}

/// What executed an object's kernels: the backend, the code generator identity, and every
/// distinct kernel with its engine. Descriptive only; it is in no realization digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExecutionReceipt {
    pub schema: String,
    pub backend: ExecutionBackend,
    pub backend_identity: String,
    /// Distinct kernels executed natively.
    pub native: usize,
    /// Distinct kernels executed by the interpreter (by selection or fallback).
    pub interpreted: usize,
    /// Distinct kernels the JIT refused (each also counted in `interpreted`).
    pub fallbacks: usize,
    pub kernels: Vec<ExecutionKernelReceipt>,
}

impl ExecutionReceipt {
    /// The receipt of `kernels`; the backend is the one they were prepared under
    /// (`Interpreter` for an object with no kernels).
    pub(crate) fn collect<'a>(kernels: impl IntoIterator<Item = &'a PreparedExecutable>) -> Self {
        let mut distinct = BTreeMap::new();
        let mut backend = ExecutionBackend::Interpreter;
        for kernel in kernels {
            backend = backend.max(kernel.backend());
            let digest = malleus::kernel_digest(kernel.kernel().as_kernel()).hex;
            distinct
                .entry(digest)
                .or_insert_with(|| (kernel.kernel().as_kernel().name.clone(), kernel.engine()));
        }
        let kernels = distinct
            .into_iter()
            .map(|(kernel_digest, (name, engine))| ExecutionKernelReceipt {
                name,
                kernel_digest,
                engine,
            })
            .collect::<Vec<_>>();
        let count = |pick: fn(&KernelEngine) -> bool| {
            kernels.iter().filter(|kernel| pick(&kernel.engine)).count()
        };
        Self {
            schema: EXECUTION_RECEIPT_SCHEMA.to_owned(),
            backend,
            backend_identity: backend.identity(),
            native: count(|engine| matches!(engine, KernelEngine::Native)),
            interpreted: count(|engine| !matches!(engine, KernelEngine::Native)),
            fallbacks: count(|engine| matches!(engine, KernelEngine::Fallback { .. })),
            kernels,
        }
    }
}

fn prepare_error(error: malleus::prepared::PrepareError) -> FinitumError {
    FinitumError::KernelValidation(error.to_string())
}

/// The engine chosen for one prepared kernel when it was built.
#[derive(Clone)]
enum Engine {
    Interpreter,
    #[cfg(feature = "cranelift")]
    Native(Arc<malleus_cranelift::JitKernel>),
    Fallback(Arc<str>),
}

impl fmt::Debug for Engine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Engine::Interpreter => formatter.write_str("Interpreter"),
            #[cfg(feature = "cranelift")]
            Engine::Native(_) => formatter.write_str("Native"),
            Engine::Fallback(reason) => write!(formatter, "Fallback({reason:?})"),
        }
    }
}

impl Engine {
    fn select(executable: &Executable) -> Result<Self, FinitumError> {
        match selected_execution_backend()? {
            ExecutionBackend::Interpreter => Ok(Engine::Interpreter),
            ExecutionBackend::Cranelift => Ok(jit::engine(executable)),
        }
    }

    fn receipt(&self) -> KernelEngine {
        match self {
            Engine::Interpreter => KernelEngine::Interpreter,
            #[cfg(feature = "cranelift")]
            Engine::Native(_) => KernelEngine::Native,
            Engine::Fallback(reason) => KernelEngine::Fallback {
                reason: reason.to_string(),
            },
        }
    }
}

#[cfg(feature = "cranelift")]
mod jit {
    use super::Engine;
    use malleus::Executable;
    use malleus_cranelift::{JitCache, JitState};
    use std::cell::RefCell;
    use std::sync::LazyLock;

    thread_local! {
        pub(super) static JIT_STATE: RefCell<JitState> = RefCell::new(JitState::default());
    }

    pub(super) fn cache() -> &'static JitCache {
        static CACHE: LazyLock<JitCache> = LazyLock::new(JitCache::default);
        &CACHE
    }

    pub(super) fn engine(executable: &Executable) -> Engine {
        match cache().get_or_compile(executable) {
            Ok(kernel) => Engine::Native(kernel),
            Err(error) => Engine::Fallback(error.to_string().into()),
        }
    }
}

#[cfg(not(feature = "cranelift"))]
mod jit {
    use super::Engine;
    use malleus::Executable;

    /// Unreachable: `selected_execution_backend` refuses `Cranelift` in this build.
    pub(super) fn engine(_: &Executable) -> Engine {
        Engine::Fallback("feature `cranelift` not built".into())
    }
}

/// An executable, its prepared form and its engine. Dereferences to the executable so IR
/// inspection (`kernel()`, operands, regions) reads exactly what it read before.
#[derive(Clone, Debug)]
pub(crate) struct PreparedExecutable {
    executable: Executable,
    prepared: Arc<PreparedKernel>,
    engine: Engine,
}

impl PreparedExecutable {
    pub(crate) fn new(executable: Executable) -> Result<Self, FinitumError> {
        let prepared = PreparedKernel::new(&executable).map_err(prepare_error)?;
        let engine = Engine::select(&executable)?;
        Ok(Self {
            executable,
            prepared: Arc::new(prepared),
            engine,
        })
    }

    pub(crate) fn prepared(&self) -> &PreparedKernel {
        &self.prepared
    }

    pub(crate) fn engine(&self) -> KernelEngine {
        self.engine.receipt()
    }

    /// The backend selected when this kernel was prepared.
    pub(crate) fn backend(&self) -> ExecutionBackend {
        match self.engine {
            Engine::Interpreter => ExecutionBackend::Interpreter,
            _ => ExecutionBackend::Cranelift,
        }
    }
}

impl Deref for PreparedExecutable {
    type Target = Executable;

    fn deref(&self) -> &Executable {
        &self.executable
    }
}

/// An executable module with every kernel prepared, in module order.
#[derive(Clone, Debug)]
pub(crate) struct PreparedExecutableModule {
    kernels: Vec<PreparedExecutable>,
}

impl PreparedExecutableModule {
    pub(crate) fn new(module: &ExecutableModule) -> Result<Self, FinitumError> {
        let kernels = module
            .kernels()
            .iter()
            .cloned()
            .map(PreparedExecutable::new)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { kernels })
    }

    pub(crate) fn kernels(&self) -> &[PreparedExecutable] {
        &self.kernels
    }
}

/// A composition, its prepared form, and (under `Cranelift`) its stages compiled natively.
#[derive(Clone, Debug)]
pub(crate) struct PreparedExecutableComposition {
    executable: ExecutableComposition,
    prepared: Arc<PreparedComposition>,
    /// Every stage with its own engine; the composition runs natively only when all are.
    stages: Vec<PreparedExecutable>,
    #[cfg(feature = "cranelift")]
    native: Option<Arc<native::NativeComposition>>,
}

impl PreparedExecutableComposition {
    pub(crate) fn new(executable: ExecutableComposition) -> Result<Self, FinitumError> {
        let prepared = PreparedComposition::new(&executable).map_err(prepare_error)?;
        let stages = executable
            .stages()
            .iter()
            .cloned()
            .map(PreparedExecutable::new)
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "cranelift")]
        let native = native::NativeComposition::new(&executable, &stages).map(Arc::new);
        Ok(Self {
            executable,
            prepared: Arc::new(prepared),
            stages,
            #[cfg(feature = "cranelift")]
            native,
        })
    }

    /// The stage kernels with their engines (for receipts).
    pub(crate) fn stages(&self) -> &[PreparedExecutable] {
        &self.stages
    }

    /// Runs every stage with `Interpreter::run_composition`'s contract and errors, natively
    /// when every stage compiled, else over the per-thread reusable composition state. A
    /// binding the native path does not accept is re-run by the interpreter for its error.
    pub(crate) fn run(&self, bindings: &mut [CompositionBinding<'_>]) -> Result<(), FinitumError> {
        #[cfg(feature = "cranelift")]
        if let Some(native) = &self.native
            && native.run(bindings)?
        {
            return Ok(());
        }
        COMPOSITION_STATE
            .with(|state| match state.try_borrow_mut() {
                Ok(mut state) => self.prepared.run(&mut state, bindings),
                Err(_) => self.prepared.run(&mut self.prepared.state(), bindings),
            })
            .map_err(|error| FinitumError::KernelExecution(error.to_string()))
    }
}

impl Deref for PreparedExecutableComposition {
    type Target = ExecutableComposition;

    fn deref(&self) -> &ExecutableComposition {
        &self.executable
    }
}

/// `PreparedComposition::run`'s copy-in/copy-out stage semantics over natively compiled
/// stages: each stage runs on private copies of its operand windows, writable copies are
/// written back in operand order, unbound shared buffers start at zero.
#[cfg(feature = "cranelift")]
mod native {
    use super::{Engine, INLINE_OPERANDS, PreparedExecutable, bind};
    use crate::FinitumError;
    use malleus::prepared::{ReadBinding, WriteBinding};
    use malleus::{
        CompositionBinding, CompositionTarget, ExecutableComposition, OperandId, StageOperand,
    };
    use malleus_cranelift::{JitKernel, JitState};
    use std::cell::RefCell;
    use std::sync::Arc;

    const UNBOUND: usize = usize::MAX;

    #[derive(Default)]
    struct Scratch {
        buffers: Vec<Vec<f64>>,
        shared: Vec<Vec<f64>>,
        operand_binding: Vec<usize>,
        shared_binding: Vec<usize>,
        jit: JitState,
    }

    thread_local! {
        static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
    }

    pub(super) struct NativeComposition {
        stages: Vec<Arc<JitKernel>>,
        /// First flat operand index of every stage, plus the total.
        offsets: Vec<usize>,
        shared_of: Vec<Option<usize>>,
        lengths: Vec<usize>,
        writable: Vec<bool>,
        shared_lengths: Vec<usize>,
    }

    impl std::fmt::Debug for NativeComposition {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("NativeComposition")
                .field("stages", &self.stages.len())
                .field("operands", &self.lengths.len())
                .finish()
        }
    }

    impl NativeComposition {
        pub(super) fn new(
            composition: &ExecutableComposition,
            stages: &[PreparedExecutable],
        ) -> Option<Self> {
            let compiled = stages
                .iter()
                .map(|stage| match &stage.engine {
                    Engine::Native(kernel) => Some(Arc::clone(kernel)),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let mut offsets = vec![0];
            let (mut shared_of, mut lengths, mut writable) = (vec![], vec![], vec![]);
            for (stage_index, stage) in composition.stages().iter().enumerate() {
                for (operand_index, definition) in
                    stage.kernel().as_kernel().operands.iter().enumerate()
                {
                    let operand = StageOperand::new(stage_index, OperandId::new(operand_index));
                    shared_of.push(composition.shared_buffer_of(operand));
                    lengths.push(definition.region.offset + definition.region.length);
                    writable.push(definition.access.can_write());
                }
                offsets.push(lengths.len());
            }
            let shared_lengths = (0..)
                .map_while(|buffer| composition.shared_length(buffer))
                .collect();
            Some(Self {
                stages: compiled,
                offsets,
                shared_of,
                lengths,
                writable,
                shared_lengths,
            })
        }

        fn flat(&self, operand: StageOperand) -> Option<usize> {
            let start = *self.offsets.get(operand.stage)?;
            let end = *self.offsets.get(operand.stage + 1)?;
            let flat = start.checked_add(operand.operand.index())?;
            (flat < end).then_some(flat)
        }

        /// `Ok(false)` when the bindings are not exactly the accepted shape (the caller then
        /// runs the interpreter, which reports the error); nothing is written in that case.
        pub(super) fn run(
            &self,
            bindings: &mut [CompositionBinding<'_>],
        ) -> Result<bool, FinitumError> {
            SCRATCH.with(|scratch| match scratch.try_borrow_mut() {
                Ok(mut scratch) => self.run_in(&mut scratch, bindings),
                Err(_) => self.run_in(&mut Scratch::default(), bindings),
            })
        }

        fn run_in(
            &self,
            scratch: &mut Scratch,
            bindings: &mut [CompositionBinding<'_>],
        ) -> Result<bool, FinitumError> {
            let Scratch {
                buffers,
                shared,
                operand_binding,
                shared_binding,
                jit,
            } = scratch;
            operand_binding.clear();
            operand_binding.resize(self.lengths.len(), UNBOUND);
            shared_binding.clear();
            shared_binding.resize(self.shared_lengths.len(), UNBOUND);
            for (index, binding) in bindings.iter().enumerate() {
                match binding.target {
                    CompositionTarget::Operand(operand) => {
                        let Some(flat) = self.flat(operand) else {
                            return Ok(false);
                        };
                        if self.shared_of[flat].is_some()
                            || binding.values.len() < self.lengths[flat]
                            || operand_binding[flat] != UNBOUND
                        {
                            return Ok(false);
                        }
                        operand_binding[flat] = index;
                    }
                    CompositionTarget::Shared(buffer) => {
                        let Some(&length) = self.shared_lengths.get(buffer) else {
                            return Ok(false);
                        };
                        if binding.values.len() < length || shared_binding[buffer] != UNBOUND {
                            return Ok(false);
                        }
                        shared_binding[buffer] = index;
                    }
                }
            }
            if self
                .shared_of
                .iter()
                .zip(operand_binding.iter())
                .any(|(shared, binding)| shared.is_none() && *binding == UNBOUND)
            {
                return Ok(false);
            }
            buffers.resize_with(self.lengths.len(), Vec::new);
            shared.resize_with(self.shared_lengths.len(), Vec::new);
            for (buffer, values) in shared.iter_mut().enumerate() {
                values.clear();
                if shared_binding[buffer] == UNBOUND {
                    values.resize(self.shared_lengths[buffer], 0.0);
                }
            }
            for (stage_index, stage) in self.stages.iter().enumerate() {
                let range = self.offsets[stage_index]..self.offsets[stage_index + 1];
                for flat in range.clone() {
                    let length = self.lengths[flat];
                    let source = match self.shared_of[flat] {
                        Some(buffer) if shared_binding[buffer] == UNBOUND => {
                            &shared[buffer][..length]
                        }
                        Some(buffer) => &bindings[shared_binding[buffer]].values[..length],
                        None => &bindings[operand_binding[flat]].values[..length],
                    };
                    buffers[flat].clear();
                    buffers[flat].extend_from_slice(source);
                }
                run_stage(stage, jit, &mut buffers[range.clone()])?;
                for flat in range.filter(|flat| self.writable[*flat]) {
                    let length = self.lengths[flat];
                    let target = match self.shared_of[flat] {
                        Some(buffer) if shared_binding[buffer] == UNBOUND => {
                            &mut shared[buffer][..length]
                        }
                        Some(buffer) => &mut bindings[shared_binding[buffer]].values[..length],
                        None => &mut bindings[operand_binding[flat]].values[..length],
                    };
                    target.copy_from_slice(&buffers[flat]);
                }
            }
            Ok(true)
        }
    }

    fn run_stage(
        stage: &JitKernel,
        state: &mut JitState,
        buffers: &mut [Vec<f64>],
    ) -> Result<(), FinitumError> {
        let (read_count, write_count) = (stage.read_operands().len(), stage.write_operands().len());
        let called = if buffers.len() <= INLINE_OPERANDS {
            let mut reads = [ReadBinding::new(OperandId::new(0), &[]); INLINE_OPERANDS];
            let mut writes: [WriteBinding<'_>; INLINE_OPERANDS] =
                std::array::from_fn(|_| WriteBinding::new(OperandId::new(0), &mut []));
            bind(
                stage.read_operands(),
                buffers,
                &mut reads[..read_count],
                &mut writes[..write_count],
            ) && stage
                .run_batch(state, 1, &reads[..read_count], &mut writes[..write_count])
                .is_ok()
        } else {
            let mut reads = vec![ReadBinding::new(OperandId::new(0), &[]); read_count];
            let mut writes = (0..write_count)
                .map(|_| WriteBinding::new(OperandId::new(0), &mut []))
                .collect::<Vec<_>>();
            bind(stage.read_operands(), buffers, &mut reads, &mut writes)
                && stage.run_batch(state, 1, &reads, &mut writes).is_ok()
        };
        if called {
            Ok(())
        } else {
            Err(FinitumError::KernelExecution(format!(
                "native composition stage `{}` refused its own copy-in windows",
                stage.executable().kernel().as_kernel().name
            )))
        }
    }
}

/// Runs `points` positional calls (`run_batch`) of `kernel` on its engine over the per-thread
/// state. The native and prepared paths refuse exactly the same bindings (the core
/// `PreparedKernel::check_bindings` decides for both) with the same `CallError`.
pub(crate) fn run_batch(
    kernel: &PreparedExecutable,
    points: usize,
    reads: &[ReadBinding<'_>],
    writes: &mut [WriteBinding<'_>],
) -> Result<(), malleus::prepared::CallError> {
    #[cfg(feature = "cranelift")]
    if let Engine::Native(native) = &kernel.engine {
        return jit::JIT_STATE.with(|state| match state.try_borrow_mut() {
            Ok(mut state) => native.run_batch(&mut state, points, reads, writes),
            Err(_) => native.run_batch(&mut native.state(), points, reads, writes),
        });
    }
    let kernel = kernel.prepared();
    KERNEL_STATE.with(|state| match state.try_borrow_mut() {
        Ok(mut state) => kernel.run_batch(&mut state, points, reads, writes),
        Err(_) => kernel.run_batch(&mut kernel.state(), points, reads, writes),
    })
}

/// Binds `buffers[i]` to operand `i` positionally; false when the buffers do not partition
/// into the kernel's read and writable slots.
fn bind<'b, B: AsMut<[f64]>>(
    read_operands: &[OperandId],
    buffers: &'b mut [B],
    reads: &mut [ReadBinding<'b>],
    writes: &mut [WriteBinding<'b>],
) -> bool {
    let (mut read, mut write) = (0, 0);
    for (index, buffer) in buffers.iter_mut().enumerate() {
        let operand = OperandId::new(index);
        let values = B::as_mut(buffer);
        if read_operands.get(read) == Some(&operand) {
            reads[read] = ReadBinding::new(operand, values);
            read += 1;
        } else if write < writes.len() {
            writes[write] = WriteBinding::new(operand, values);
            write += 1;
        } else {
            return false;
        }
    }
    read == reads.len() && write == writes.len()
}

/// Runs `kernel` once with operand `i` bound to `buffers[i]`: `Interpreter::run`'s semantics
/// and errors (mapped to `FinitumError::KernelExecution`), without its per-call allocations.
pub(crate) fn run_buffers<B: AsMut<[f64]>>(
    kernel: &PreparedExecutable,
    buffers: &mut [B],
) -> Result<(), FinitumError> {
    let prepared = kernel.prepared();
    let (read_count, write_count) = (
        prepared.read_operands().len(),
        prepared.write_operands().len(),
    );
    if buffers.len() == read_count + write_count {
        let called = if buffers.len() <= INLINE_OPERANDS {
            let mut reads = [ReadBinding::new(OperandId::new(0), &[]); INLINE_OPERANDS];
            let mut writes: [WriteBinding<'_>; INLINE_OPERANDS] =
                std::array::from_fn(|_| WriteBinding::new(OperandId::new(0), &mut []));
            bind(
                prepared.read_operands(),
                buffers,
                &mut reads[..read_count],
                &mut writes[..write_count],
            ) && run_batch(kernel, 1, &reads[..read_count], &mut writes[..write_count]).is_ok()
        } else {
            let mut reads = vec![ReadBinding::new(OperandId::new(0), &[]); read_count];
            let mut writes = (0..write_count)
                .map(|_| WriteBinding::new(OperandId::new(0), &mut []))
                .collect::<Vec<_>>();
            bind(prepared.read_operands(), buffers, &mut reads, &mut writes)
                && run_batch(kernel, 1, &reads, &mut writes).is_ok()
        };
        if called {
            return Ok(());
        }
    }
    // Refused positionally (checked before any effect): the one-shot checked path accepts
    // and refuses exactly what the interpreter does, with its error values.
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), B::as_mut(values)))
        .collect::<Vec<_>>();
    prepared
        .run(&mut bindings)
        .map_err(|error| FinitumError::KernelExecution(error.to_string()))
}

/// Runs `points` calls of `kernel` in one `run_batch`: operand `i` is `buffers[i]`, point
/// `p`'s window starting at `p * strides[i]`. Callers lay buffers out so every binding is
/// sized and every writable stride covers its region, so the columnar path applies.
pub(crate) fn run_strided(
    kernel: &PreparedExecutable,
    points: usize,
    buffers: &mut [Vec<f64>],
    strides: &[usize],
) -> Result<(), FinitumError> {
    let prepared = kernel.prepared();
    let mut reads = Vec::with_capacity(prepared.read_operands().len());
    let mut writes = Vec::with_capacity(prepared.write_operands().len());
    for (index, (buffer, stride)) in buffers.iter_mut().zip(strides).enumerate() {
        let operand = OperandId::new(index);
        if prepared.read_operands().get(reads.len()) == Some(&operand) {
            reads.push(ReadBinding::strided(operand, buffer, *stride));
        } else {
            writes.push(WriteBinding::strided(operand, buffer, *stride));
        }
    }
    run_batch(kernel, points, &reads, &mut writes)
        .map_err(|error| FinitumError::KernelExecution(error.to_string()))
}
