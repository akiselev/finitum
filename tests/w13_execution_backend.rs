//! W13 A1: execution backend selection, receipts, identity independence and the
//! compile-once JIT cache. One test per binary on purpose: the JIT cache is process-wide, so
//! its size is only meaningful without concurrently running tests.
use std::sync::Arc;

use finitum::{
    EXECUTION_BACKEND_ENV, ExecutionBackend, FvRealization, FvTopology, RegionMap, TensorAxis,
    TensorBox, selected_execution_backend, with_execution_backend,
};
use quantitas::UnitRegistry;
use scientia::{
    FvMethodIntent, NumericalFluxIntent, PositivityIntent, ReconstructionIntent, SourceTreatment,
    compile_conservation_law_system, compile_semantics, lower_conservation_law,
};

const FIXTURE: &str = include_str!("fixtures/fv-scalar.res");

fn burgers(cells: usize) -> Result<FvRealization, finitum::FinitumError> {
    let module = compile_semantics(FIXTURE, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic;
    let system = compile_conservation_law_system(&module, "Burgers1D", "balance", "u").unwrap();
    let intent = FvMethodIntent {
        numerical_flux: NumericalFluxIntent::LocalLaxFriedrichs,
        reconstruction: ReconstructionIntent::Constant,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    };
    let program = Arc::new(lower_conservation_law(&module, &system, &intent, &[]).unwrap());
    let topology = Arc::new(
        FvTopology::from_tensor_box(&TensorBox {
            axes: vec![TensorAxis::uniform(0.0, 1.0, cells).periodic()],
        })
        .unwrap(),
    );
    FvRealization::new(
        &system,
        program,
        topology,
        &RegionMap::new(),
        &[("k".to_string(), 0.3)].into_iter().collect(),
    )
}

#[test]
fn backend_selection_receipts_identity_and_compile_once_cache() {
    // Names and serde.
    for backend in [ExecutionBackend::Interpreter, ExecutionBackend::Cranelift] {
        assert_eq!(ExecutionBackend::from_name(backend.name()), Some(backend));
        assert_eq!(
            serde_json::to_string(&backend).unwrap(),
            format!("\"{}\"", backend.name())
        );
    }
    assert_eq!(ExecutionBackend::from_name("llvm"), None);
    // The process default follows the environment (unset: interpreter); scopes override it
    // and restore it.
    let default = match std::env::var(EXECUTION_BACKEND_ENV) {
        Ok(name) => ExecutionBackend::from_name(&name).unwrap(),
        Err(_) => ExecutionBackend::Interpreter,
    };
    assert_eq!(selected_execution_backend().unwrap(), default);
    with_execution_backend(ExecutionBackend::Interpreter, || {
        assert_eq!(
            selected_execution_backend().unwrap(),
            ExecutionBackend::Interpreter
        );
    });
    assert_eq!(selected_execution_backend().unwrap(), default);

    let interpreted =
        with_execution_backend(ExecutionBackend::Interpreter, || burgers(12)).unwrap();
    let receipt = interpreted.execution_receipt();
    assert_eq!(receipt.schema, finitum::EXECUTION_RECEIPT_SCHEMA);
    assert_eq!(receipt.backend, ExecutionBackend::Interpreter);
    assert_eq!((receipt.native, receipt.fallbacks), (0, 0));
    assert!(receipt.interpreted >= 2, "{receipt:?}");

    #[cfg(not(feature = "cranelift"))]
    {
        assert!(!ExecutionBackend::Cranelift.available());
        let Err(refused) = with_execution_backend(ExecutionBackend::Cranelift, || burgers(12))
        else {
            panic!("Cranelift needs the `cranelift` feature");
        };
        assert_eq!(refused.code(), Some("EXECUTION_BACKEND_UNAVAILABLE"));
        assert_eq!(finitum::native_kernel_cache_len(), 0);
    }

    #[cfg(feature = "cranelift")]
    {
        assert!(ExecutionBackend::Cranelift.available());
        let before = finitum::native_kernel_cache_len();
        let first = with_execution_backend(ExecutionBackend::Cranelift, || burgers(12)).unwrap();
        let compiled = finitum::native_kernel_cache_len();
        let receipt = first.execution_receipt();
        assert_eq!(receipt.backend, ExecutionBackend::Cranelift);
        assert!(
            receipt
                .backend_identity
                .starts_with("malleus-cranelift/0.1.0"),
            "{}",
            receipt.backend_identity
        );
        assert_eq!((receipt.interpreted, receipt.fallbacks), (0, 0));
        assert_eq!(
            compiled - before,
            receipt.native,
            "one compilation per kernel"
        );
        // A second realization (another mesh size, same kernels) compiles nothing.
        let second = with_execution_backend(ExecutionBackend::Cranelift, || burgers(40)).unwrap();
        assert_eq!(finitum::native_kernel_cache_len(), compiled);
        assert_eq!(second.execution_receipt().kernels, receipt.kernels);
        // The backend is in no identity: digests and descriptive receipts agree.
        assert_eq!(first.digest(), interpreted.digest());
        assert_eq!(first.receipt(), interpreted.receipt());
        assert_eq!(
            serde_json::to_vec(&first.receipt()).unwrap(),
            serde_json::to_vec(&interpreted.receipt()).unwrap()
        );
    }
}
