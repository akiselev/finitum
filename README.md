# Finitum

Finitum is Sinbad's concrete discretization and global-realization layer. Its current checked
foundation contains:

- simplex meshes with finite coordinates, valid connectivity, and distinct vertices per cell;
- deterministic element restrictions over a bounded global degree-of-freedom space;
- acyclic affine constraints with finite coefficients and checked expansion; and
- prepared quadrature, basis-value, and basis-gradient tables with checked extents and finite data.

The crate does not yet expose a global operator. A sparse matrix wrapper without the real
form/kernel/discretization binding would only be a forwarding seam, so it was removed.

Resolvent will supply abstract form and local-factorization artifacts. Malleus will supply
validated executable local kernels. Finitum will bind them to its mesh, element, DOF, and
constraint data before adding assembled, partial, element, or matrix-free realizations. Krasis
owns coupled state; Solverang owns numerical algorithms.

There are currently no dependencies on those sibling repositories because no implemented API
consumes them yet. They should be added only with the first end-to-end realized operator.
