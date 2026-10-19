//! Prepared Malleus execution (W11 F1).
//!
//! Every executable Finitum runs is held next to its `malleus::prepared` form, built once by
//! the struct that owns the executable (never a global cache keyed by address: addresses are
//! reused after drop). Calls bind operands positionally over inline binding tables and a
//! per-thread reusable register file, so a steady-state call allocates nothing inside
//! Malleus. The prepared path is bitwise identical to `Interpreter::run`; a refused call is
//! re-issued through the checked one-shot `PreparedKernel::run`, which reports exactly the
//! interpreter's `ExecutionError`, so `FinitumError::KernelExecution` messages are unchanged.

use crate::FinitumError;
use malleus::prepared::{
    CompositionState, PreparedComposition, PreparedKernel, PreparedState, ReadBinding, WriteBinding,
};
use malleus::{
    BufferBinding, CompositionBinding, Executable, ExecutableComposition, ExecutableModule,
    OperandId,
};
use std::cell::RefCell;
use std::ops::Deref;
use std::sync::Arc;

/// Operand count up to which binding tables live on the stack.
const INLINE_OPERANDS: usize = 16;

thread_local! {
    static KERNEL_STATE: RefCell<PreparedState> = RefCell::new(PreparedState::new());
    static COMPOSITION_STATE: RefCell<CompositionState> =
        RefCell::new(CompositionState::default());
}

fn prepare_error(error: malleus::prepared::PrepareError) -> FinitumError {
    FinitumError::KernelValidation(error.to_string())
}

/// An executable and its prepared form. Dereferences to the executable so IR inspection
/// (`kernel()`, operands, regions) reads exactly what it read before.
#[derive(Clone, Debug)]
pub(crate) struct PreparedExecutable {
    executable: Executable,
    prepared: Arc<PreparedKernel>,
}

impl PreparedExecutable {
    pub(crate) fn new(executable: Executable) -> Result<Self, FinitumError> {
        let prepared = PreparedKernel::new(&executable).map_err(prepare_error)?;
        Ok(Self {
            executable,
            prepared: Arc::new(prepared),
        })
    }

    pub(crate) fn prepared(&self) -> &PreparedKernel {
        &self.prepared
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

/// A composition and its prepared form.
#[derive(Clone, Debug)]
pub(crate) struct PreparedExecutableComposition {
    executable: ExecutableComposition,
    prepared: Arc<PreparedComposition>,
}

impl PreparedExecutableComposition {
    pub(crate) fn new(executable: ExecutableComposition) -> Result<Self, FinitumError> {
        let prepared = PreparedComposition::new(&executable).map_err(prepare_error)?;
        Ok(Self {
            executable,
            prepared: Arc::new(prepared),
        })
    }

    /// Runs every stage with `Interpreter::run_composition`'s contract and errors over the
    /// per-thread reusable composition state.
    pub(crate) fn run(&self, bindings: &mut [CompositionBinding<'_>]) -> Result<(), FinitumError> {
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

/// Runs `points` positional calls (`PreparedKernel::run_batch`) over the per-thread state.
pub(crate) fn run_batch(
    kernel: &PreparedKernel,
    points: usize,
    reads: &[ReadBinding<'_>],
    writes: &mut [WriteBinding<'_>],
) -> Result<(), malleus::prepared::CallError> {
    KERNEL_STATE.with(|state| match state.try_borrow_mut() {
        Ok(mut state) => kernel.run_batch(&mut state, points, reads, writes),
        Err(_) => kernel.run_batch(&mut kernel.state(), points, reads, writes),
    })
}

/// Binds `buffers[i]` to operand `i` positionally; false when the buffers do not partition
/// into the kernel's read and writable slots.
fn bind<'b, B: AsMut<[f64]>>(
    kernel: &PreparedKernel,
    buffers: &'b mut [B],
    reads: &mut [ReadBinding<'b>],
    writes: &mut [WriteBinding<'b>],
) -> bool {
    let (mut read, mut write) = (0, 0);
    for (index, buffer) in buffers.iter_mut().enumerate() {
        let operand = OperandId::new(index);
        let values = B::as_mut(buffer);
        if kernel.read_operands().get(read) == Some(&operand) {
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
                prepared,
                buffers,
                &mut reads[..read_count],
                &mut writes[..write_count],
            ) && run_batch(
                prepared,
                1,
                &reads[..read_count],
                &mut writes[..write_count],
            )
            .is_ok()
        } else {
            let mut reads = vec![ReadBinding::new(OperandId::new(0), &[]); read_count];
            let mut writes = (0..write_count)
                .map(|_| WriteBinding::new(OperandId::new(0), &mut []))
                .collect::<Vec<_>>();
            bind(prepared, buffers, &mut reads, &mut writes)
                && run_batch(prepared, 1, &reads, &mut writes).is_ok()
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
    run_batch(prepared, points, &reads, &mut writes)
        .map_err(|error| FinitumError::KernelExecution(error.to_string()))
}
