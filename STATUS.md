# Finitum status

Updated: 2026-08-20
Milestone: checked discretization foundation

## Implemented

- validated 1D--3D simplex meshes with finite coordinates, bounded connectivity, and distinct
  vertices per cell;
- deterministic global degree-of-freedom maps with nonempty, bounded, duplicate-free element
  restrictions;
- explicit affine constraints with unique targets/dependencies, finite coefficients, bounds and
  cycle validation, exact input extents, and finite expansion results;
- prepared quadrature/basis tables with checked extents, shape validation, and finite data.

The placeholder CSR/assembled operator has been removed. There is no global operator API until a
real form/kernel/discretization binding exists.

## Boundary

Resolvent owns the abstract space and form meaning. Malleus owns executable local kernels.
Finitum owns their concrete mesh/space binding and global realization. Krasis owns coupled
state and Solverang owns numerical algorithms.

The current manifest has no sibling-repository dependencies. Resolvent, Malleus, and Solverang will
be added only when implemented Finitum code consumes their concrete APIs.

## Validation

Passed on 2026-08-20 with Rust 1.97.0:

```text
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets           # 5 passed, 0 failed
git diff --check
```

## Next

Bind a Resolvent `LocalFormProgram` and Malleus validated executable kernel to the mesh, prepared
element, DOF, and constraint data for Poisson. Introduce a realized operator only when it owns that
complete gather/local-execute/scatter behavior, not as a forwarding wrapper.
