# Finitum status

Updated: 2026-09-18

## Role and frontier

Finitum owns concrete meshes, spaces, DOFs, geometry, constraints, sampling,
transfer and global realization. Scientia owns scientific/form meaning, Malleus
owns local executable kernels, Krasis owns state/history, and Methodus owns
numerical algorithms. Finitum implements Methodus operator contracts directly;
it has no dependency on Solverang's constraint engine.

SHOW-3 / SC-W3 additions are implemented and in acceptance. The complete owner
gate and final Sinbad consumer gate remain separate requirements.

## Current additions

- `ConnectionRealizationPlan::system_constraints` checks each pair's scientific
  source, model, mesh, field, orientation and essential-boundary exclusions, then
  forms deterministic flat equivalence classes across multiple matching relations.
  Transpose restriction accumulates all participating residuals.
- `SystemOperator::linearized_diagonal` obtains the dynamic Jacobian diagonal
  through cell-local JVP columns at the actual state/rate and declared rate shift.
  Reduced operators preserve essential rows. Facet kernels and affine dependency
  constraints return no diagonal; there is no whole-mesh probing fallback.
- `SurfaceTransfer::p1` interpolates a triangular surface in physical 3-D to
  nonmatching points and supplies the exact dual transpose. It preserves affine
  traces and total dual loads. Degenerate/ambiguous/outside geometry refuses.
  The point-transfer artifact alone is not a coverage proof.
- `ConnectionRealizationPlan::nested_p1` additionally proves planar, opposite,
  exterior triangular traces, absence of overlap and complete nested fine-facet
  coverage per coarse facet. It constructs weighted constraints and exposes
  continuity and dual-residual balance checks. Steady heat and species consumers
  exercise the product policy; arbitrary nonnested partitions refuse.
- Focused dynamic-diagonal/JVP comparison passes. Complete owner gate pending.

## Accepted recent gates

- SC-W2 first matching-interface implementation: complete bijective exterior
  facet/vertex coverage, opposite normals/measures, source/model/mesh identity,
  scalar P1 traces, row orientation and essential-constraint non-overlap. Only
  a matching proof admits Open boundary terms; gaps, partial coverage, same-facing
  normals and nonmatching meshes refuse. September 17 owner gate: 246 tests across
  35 targets, formatting, strict clippy, rustdoc and doctests passed.
  [Evidence](docs/validation/2026-09-17-sc-w2/README.md).
- SHOW-1: portable simplex meshes, exact P1 samples through FieldSampler, and
  clipped exterior/cap geometry with owner sampling coordinates. Non-P1 exports
  refuse. September 17: 245 tests across 34 targets and the lint/doc gates passed.
  [Evidence](docs/validation/2026-09-17-show1/README.md).
- M-CPU-S: cell-local CSR assembly and scratch reuse, preserving operator/identity/
  symmetry semantics. September 17: 243 owner tests, six focused comparisons and
  195 unchanged Sinbad consumer tests passed. Mixed-case input evaluations fell
  from 2,832 to 720. Exterior-facet/empty-mesh probing, reduced/linearized assembly,
  proof tolerance and dimension cap were unchanged by that bounded fix.
  [Evidence](docs/validation/2026-09-17-mcpu-s/README.md).

## Library surface

- Tagged mesh profiles and refinement ladders; affine simplex maps and concrete
  topology, region/facet membership and essential constraints.
- Source-bound realization plans and state/rate residual/JVP actions over compiled
  kernels; direct Methodus linear/nonlinear/DAE interfaces.
- Multi-instance system realization and generated bind-chain products, with
  system IDs, source provenance, identities and fallible input sampling.
- Owner field sampling, physical/reference mappings and degree-selected quadrature
  for supported H1/mixed/Piola representations. Result export has its narrower P1
  contract; a sampler capability does not automatically imply a product case.
- Assembled, element, matrix-free and partial/batched reference representations,
  with explicit unsupported dispositions and applicable agreement checks.
- Exterior/interface trace and mixed/exact-sequence/condensation primitives;
  affine constraint prolongation, equation rows and transpose work.
- Reference 1-D mortar/nonmatching transfer, variable-order segment, clipped-segment,
  FV/FD/network/particle/boundary method programs and fixtures. These are not a
  claim of complete production execution for every method family.
- Source-aware realization, conservation, transfer and constraint-work verification
  reports. Callers must use the validating APIs, not merely rehash acceptance flags.
- CADabra provider identity/maps are consumed directly for bounded geometry paths.

## Limits and next work

- General hp/AMR topology, geometric hanging-node construction, curved high-order
  realization, moving/remeshed interfaces and production embedded-domain semantics
  remain open. Reference segment tables are not multidimensional hp support.
- Connected product admission includes matching scalar P1 and one steady planar
  nested P1 pair. General nonnested, curved, moving or transient nonmatching
  coupling and multiplier/Nitsche stability remain open.
- Full-coordinate affine constraint rows are generally nonsymmetric even when
  the free-coordinate `P^T A P` block is symmetric; symmetry declarations and solver
  refusals must retain that distinction.
- Runtime point linearization must be distinguished from constructors that freeze
  at zero state/rate. No universal solver/preconditioner or speedup is claimed.
- General SIMD/GPU execution, material/scientific parsing, coupled transactions,
  solver selection and support promotion belong outside this repository.

Detailed FC/GX/W8 history and superseded audit findings remain in Git history.
This compact ledger replaces the accumulated historical work log; recorded gate
artifacts remain in `docs/validation/`.
