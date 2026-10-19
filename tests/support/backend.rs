//! W13 A1 differential harness: build and evaluate one fixture under the interpreter and under
//! Cranelift and require bitwise equal outcomes (NaNs equivalent; errors equal as text).

use finitum::{ExecutionBackend, ExecutionReceipt, KernelEngine, with_execution_backend};

/// One evaluated product: its values, or the error it was refused with.
pub type Outcome = Result<Vec<f64>, String>;

pub fn outcome<E: ToString>(result: Result<Vec<f64>, E>) -> Outcome {
    result.map_err(|error| error.to_string())
}

/// `run` (which must build its objects inside, so they capture the backend) on each backend.
pub fn on_both_backends<T>(run: impl Fn() -> T) -> (T, T) {
    let interpreted = with_execution_backend(ExecutionBackend::Interpreter, &run);
    let native = with_execution_backend(ExecutionBackend::Cranelift, &run);
    (interpreted, native)
}

fn same_bits(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan())
}

pub fn assert_bitwise_outcomes(label: &str, interpreted: &[Outcome], native: &[Outcome]) {
    assert_eq!(interpreted.len(), native.len(), "{label}: product count");
    for (product, (left, right)) in interpreted.iter().zip(native).enumerate() {
        match (left, right) {
            (Ok(left), Ok(right)) => {
                assert_eq!(left.len(), right.len(), "{label}[{product}]: length");
                for (index, (a, b)) in left.iter().zip(right).enumerate() {
                    assert!(
                        same_bits(*a, *b),
                        "{label}[{product}][{index}]: interpreter {a:e} ({:#018x}) vs cranelift \
                         {b:e} ({:#018x})",
                        a.to_bits(),
                        b.to_bits()
                    );
                }
            }
            (Err(left), Err(right)) => assert_eq!(left, right, "{label}[{product}]: refusal"),
            _ => panic!("{label}[{product}]: interpreter {left:?} vs cranelift {right:?}"),
        }
    }
}

/// The interpreter run executed nothing natively; the Cranelift run executed every distinct
/// kernel natively (no fallback) and saw the same kernel digests.
pub fn assert_native_receipts(
    label: &str,
    interpreted: &ExecutionReceipt,
    native: &ExecutionReceipt,
) {
    assert_eq!(
        interpreted.backend,
        ExecutionBackend::Interpreter,
        "{label}"
    );
    assert_eq!(interpreted.native, 0, "{label}");
    assert!(
        interpreted
            .kernels
            .iter()
            .all(|kernel| kernel.engine == KernelEngine::Interpreter),
        "{label}"
    );
    assert_eq!(native.backend, ExecutionBackend::Cranelift, "{label}");
    assert!(native.native > 0, "{label}: no native kernel");
    assert_eq!(native.fallbacks, 0, "{label}: {:?}", native.kernels);
    assert_eq!(native.interpreted, 0, "{label}: {:?}", native.kernels);
    let digests = |receipt: &ExecutionReceipt| {
        receipt
            .kernels
            .iter()
            .map(|kernel| kernel.kernel_digest.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        digests(interpreted),
        digests(native),
        "{label}: kernel digests"
    );
}
