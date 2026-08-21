# Finitum

Finitum is Sinbad's concrete discretization and global-realization layer. Its checked FC6/FC7
implementation contains:

- simplex meshes with finite coordinates, valid connectivity, and distinct vertices per cell;
- deterministic element restrictions over a bounded global degree-of-freedom space;
- acyclic affine constraints with finite coefficients and checked expansion;
- prepared quadrature, basis-value, and basis-gradient tables with checked extents and finite data;
- digest-linked `RealizationPlan` bindings from Resolvent FC3/FC4 artifacts and complete Malleus
  FC5 modules to concrete mesh, geometry, DOF, constraint, and coefficient data;
- deterministic P1 simplex gather, value/gradient basis actions, generated primal/JVP execution,
  quadrature weighting, basis transpose, and scatter;
- fixed essential-value lifting and identity rows; and
- matrix-free and canonical CSR operators implementing Solverang's `LinearOperator` directly;
- independent runtime state/rate residual and JVP actions; and
- dynamic point inputs chained through generated parameter-JVP kernels; and
- a concrete realization digest covering artifacts, mesh, element tables, DOF map, constraints,
  stored values, and explicit dynamic-input identities.

The current realization deliberately supports scalar H1(order=1) cell integrals and fixed
essential values. Mixed/compatible spaces, facet traversal, affine dependency elimination,
partial assembly, and optimized backends remain later phases. The FC6 linear view evaluates JVPs
at zero active input; FC7 callers use the runtime state/rate methods. The caller supplies the
mapping from semantic boundary requirements to concrete constrained DOFs; Finitum checks extents
and presence, not boundary-partition membership.

This is a deterministic reference realization, not a production hot path: point execution
allocates interpreter buffers, external sampling prepares geometry independently, and CSR
assembly takes one matrix-free action per column. Krasis owns coupled state; Solverang owns
numerical algorithms.
