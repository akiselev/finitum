# Finitum status

Updated: 2026-08-20
Milestone: clean repository bootstrap

## Implemented

- validated concrete simplex meshes;
- deterministic global degree-of-freedom maps and element restrictions;
- explicit affine constraints with cycle and bounds validation;
- prepared quadrature/basis tables with shape validation;
- assembled sparse operator representation and deterministic action.

## Boundary

Resolvent owns the abstract space and form meaning. Malleus owns executable local kernels.
Finitum owns their concrete mesh/space binding and global realization. Krasis owns coupled
state and Solverang owns numerical algorithms.

## Validation

Passed on 2026-08-20 with Rust 1.97.0:

```text
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test                         # 2 passed
```

## Next

Bind a Resolvent Poisson factorization and Malleus interpreted kernel to the concrete
operator path without adding a compatibility adapter.
