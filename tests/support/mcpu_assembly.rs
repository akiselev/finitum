use finitum::SystemOperator;
use methodus::{EvaluationContext, LinearOperator, OperatorSymmetry};
use std::time::Instant;

/// The pre-M-CPU-S exhaustive global action is the independent assembly oracle.
/// Compare every column, not just one conveniently chosen solution or checksum.
pub fn compare(operator: &SystemOperator, label: &str) {
    let n = operator.dimension();
    let context = EvaluationContext::reproducible();
    let start = Instant::now();
    let matrix = operator.assemble().unwrap();
    let assembly = start.elapsed();
    let mut direction = vec![0.0; n];
    let mut actual = vec![0.0; n];
    let mut expected = vec![0.0; n];
    let mut dense = vec![0.0; n * n];
    let start = Instant::now();
    for column in 0..n {
        direction[column] = 1.0;
        operator.apply_action(&direction, &mut expected).unwrap();
        matrix.apply(&context, &direction, &mut actual).unwrap();
        for row in 0..n {
            dense[row * n + column] = expected[row];
            let scale = 1.0 + expected[row].abs().max(actual[row].abs());
            assert!(
                (actual[row] - expected[row]).abs() <= 1.0e-12 * scale,
                "{label} ({row}, {column}): {} != {}",
                actual[row],
                expected[row]
            );
        }
        direction[column] = 0.0;
    }
    let probing = start.elapsed();
    let tolerance = 1.0e-10;
    let symmetric = (0..n).all(|i| {
        (0..n).all(|j| {
            let (a, b) = (dense[i * n + j], dense[j * n + i]);
            (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0)
        })
    });
    assert_eq!(
        operator.prove_symmetry(tolerance).unwrap(),
        if symmetric {
            OperatorSymmetry::Symmetric
        } else {
            OperatorSymmetry::Nonsymmetric
        },
        "{label}: numerical symmetry decision"
    );
    eprintln!("M-CPU-S {label}: dimension={n}, assembly={assembly:?}, exhaustive={probing:?}");
}
