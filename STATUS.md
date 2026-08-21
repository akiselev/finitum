# Finitum status

Updated: 2026-08-21
Milestone: FC6 discrete realization complete

## Implemented

- validated 1D--3D simplex meshes with finite coordinates, bounded connectivity, and distinct
  vertices per cell;
- deterministic global degree-of-freedom maps with nonempty, bounded, duplicate-free element
  restrictions;
- explicit affine constraints with unique targets/dependencies, finite coefficients, bounds and
  cycle validation, exact input extents, and finite expansion results;
- prepared quadrature/basis tables with checked extents, shape validation, and finite data;
- P1 simplex reference basis and barycenter quadrature preparation;
- digest validation across Resolvent `FormRequirements`/`OperatorFactorization` and complete
  Malleus primal/JVP/VJP/parameter bundles;
- concrete affine geometry preprocessing and quadrature-point external-input packing;
- deterministic gather, value/gradient basis forward action, generated primal/JVP execution,
  quadrature weighting, basis transpose, and scatter;
- fixed essential-value lifting and homogeneous identity-row operator action;
- one `RealizationPlan` producing matrix-free and canonical CSR assembled operators through the
  same generated JVP execution; both implement Solverang `LinearOperator` directly.

## Boundary

Resolvent owns the abstract space and form meaning. Malleus owns executable local kernels.
Finitum owns their concrete mesh/space binding and global realization. Krasis owns coupled
state and Solverang owns numerical algorithms.

The manifest now depends directly on Resolvent, Malleus, and Solverang because the realization
plan consumes their concrete artifacts and trait. FC6 supports scalar H1(order=1) cell integrals
and fixed essential constraints. Other spaces, facet traversal, affine dependency elimination,
partial assembly, and optimized backends remain deferred.

The linear operator evaluates generated JVPs at zero active input; nonlinear forms require a
future state-bearing linearization contract. Concrete constrained DOFs and their values are
caller-supplied: FC6 verifies that semantic and concrete constraints are both present and that
their extents are valid, but the mesh has no boundary-region tags with which to prove partition
membership. Point-kernel buffers are allocated per invocation, external sampling prepares its
own cell geometry, and reference CSR assembly performs one operator action per column; these are
deliberate fixture-grade costs, not production performance claims.

## Validation

Passed on 2026-08-21 with Rust 1.97.0:

```text
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets           # 8 passed, 0 failed
git diff --check
```

The realization gate includes an independent affine patch test on a nonuniform sheared mesh:
with `k = 1`, `f = 0`, and nonzero linear Dirichlet data, both realized operators reproduce every
exact P1 nodal value within `1e-12`.

## Next

Extend realization only when FC7 or later acceptance cases require another concrete evaluation,
constraint, traversal, or assembly contract.
