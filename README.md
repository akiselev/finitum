# Finitum

Finitum is Sinbad's concrete discretization and global-realization layer. Its checked FC6--FC10
implementation contains:

- simplex meshes with finite coordinates, valid connectivity, and distinct vertices per cell;
- deterministic element restrictions over a bounded global degree-of-freedom space;
- acyclic affine constraints with finite coefficients and checked expansion;
- prepared quadrature, basis-value, and basis-gradient tables with checked extents and finite data;
- digest-linked `RealizationPlan` bindings from Scientia FC3/FC4 artifacts and complete Malleus
  FC5 modules to concrete mesh, geometry, DOF, constraint, and coefficient data;
- deterministic P1 simplex gather, value/gradient basis actions, generated primal/JVP execution,
  quadrature weighting, basis transpose, and scatter;
- affine constraint prolongation/restriction, lifting, and constraint rows, with dependent-row
  actions explicitly classified as nonsymmetric;
- matrix-free and canonical CSR operators implementing Methodus's `LinearOperator` directly;
- independent runtime state/rate residual and JVP actions; and
- dynamic point inputs chained through generated parameter-JVP kernels; and
- a concrete realization digest covering artifacts, mesh, element tables, DOF map, constraints,
  stored values, and explicit dynamic-input identities;
- an FC11 `RealizationArtifact` projection containing that digest and every identity-sensitive
  serializable input for inspection; it is not a reconstruction API, generated executables are
  absent, and dynamic callbacks are represented only by their digest-covered identities;
- component-explicit product layouts, oriented exterior/interior/interface facet traversal,
  covariant and contravariant Piola maps, and oriented edge/facet restrictions;
- exact triangle/tetrahedron incidence sequences and element-local Schur condensation; and
- a digest-linked `SystemRealizationPlan` consuming Scientia block systems and complete Malleus
  bundles;
- nonmatching trace transfer and conservative mortar scatter, standalone per-cell variable-order
  segment tables, and algebraic midpoint interpolation;
- distinct element-assembled and quadrature-partial operators, fixed-width cell batching,
  accelerator-friendly component/lane packing, and tensor-product sum factorization; and
- a concrete level-set policy and exact reference quadrature for clipped segments.
- digest-bound FC10 finite-volume, finite-difference, network DAE, particle-pair, and
  boundary-integral reference realizations, plus one `DiscreteOperator` boundary shared with
  variational FEM for Krasis composition; and
- an R3P affine-rectangle realization that freezes CADabra provider revision, semantic digest,
  stable design-parameter coordinates, node/chart coordinates, cell/region ownership, and
  boundary identities before producing the P1 simplex mesh and CAD-ID-selected constraints;
  `CadPrimalPlan` retains that association digest and canonical boundary-condition projection
  alongside the executable operator identity, rejecting forged DOF or constraint data.

The globally executable operator path deliberately remains scalar H1(order=1) cell integration
with affine essential and algebraic dependency constraints. FC8's mixed/facet/compatible path is a deterministic reference
planning, mapping, topology, and evidence contract; production compatible basis tables and global
mixed solves are not claimed. FC9's advanced paths are likewise bounded reference contracts: the
variable-order segment tables are not integrated into `RealizationPlan`, and the uniform-grid
acceptance constraint is algebraic rather than derived from local refinement. There is no hp/AMR
realization, production multidimensional embedded mesh, or SIMD/GPU backend. The FC6 linear
view evaluates JVPs at zero active input; FC7 callers
use runtime state/rate methods. Generic callers may still supply concrete constrained DOFs, while
the R3P rectangle path derives their boundary membership from stable CADabra identities and
refuses stale or source-mismatched providers, missing identities, duplicate selections, or conflicting values at
shared corners. R3P currently supports affine rectangles contained in an XY carrier; embedded
surface elements and geometry derivatives belong to later capabilities. FC10's method
realizations are bounded deterministic contracts:
FV uses oriented two-cell faces and compiled affine flux kernels, FD uses caller-supplied neighbor
rows and compiled affine stencils, network DAEs use typed-extent dense matrices, particles use
generic radial polynomials and explicit pairs, and boundary integrals use caller-supplied kernel
tables and quadrature weights.

This is a deterministic reference realization, not a production hot path: point execution
allocates interpreter buffers, external sampling prepares geometry independently, and CSR
assembly takes one matrix-free action per column. Krasis owns coupled state; Methodus owns
numerical algorithms.
