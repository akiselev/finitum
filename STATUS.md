# Finitum status

Updated: 2026-09-29

## Role and frontier

Finitum owns concrete meshes, spaces, DOFs, geometry, constraints, sampling,
transfer and global realization. Scientia owns scientific/form meaning, Malleus
owns local executable kernels, Krasis owns state/history, and Methodus owns
numerical algorithms. Finitum implements Methodus operator contracts directly;
it has no dependency on Solverang's constraint engine.

SHOW-3 / bounded SC-W3 additions are accepted. The complete owner gate passes
(250 tests across 37 targets); final Sinbad consumer coverage passes (210 tests
across 35 targets with a documented external-fixture target retry, unaffected
by this package).

## W9 SC-W3 package 1 (2026-09-29): weighted-map Jacobi diagonal

- `ConnectionRealizationPlan::system_linearized_diagonal` lands the reduced
  Jacobian diagonal `diag(Rᵀ J R)` of the two-operator connected system a
  `matching` or `nested_p1` plan couples: essential and connection-target
  (slave) rows are preserved as the reduced system's own identity row (`1.0`,
  exactly as the existing essential-only path already does); a master row
  untouched by the interface keeps its own operator's
  `SystemOperator::linearized_diagonal` value; an interface-adjacent master row
  gains the transpose-scattered cross term `Σ w_β J_B[β,γ] w_γ` over the target
  trace DOFs it prolongates into. Previously "weighted-map Jacobi refuses"
  (any affine-dependency constraint set returned no diagonal at all); this
  closes that gap for the bounded one-relation/two-operator shape
  `system_constraints` already restricts nonmatching connections to.
- New `SystemOperator::linearized_block` (`pub(crate)`) computes the bounded
  dense off-diagonal Jacobian sub-block restricted to a given row/column DOF
  set from the same cell-local JVP machinery `linearized_diagonal` uses: work
  scales with the requested (interface-sized) column count and the cells that
  touch them, never with the whole mesh. It is the only new primitive; no
  change to `linearized_diagonal`, `apply_action`, `apply_reduced_action`, or
  any existing constraint/transfer type.
  Refusal is unchanged and reused, not widened: either operator's own
  facet-kernel operators or its own affine-dependency essential constraints
  (periodic-style, not this connection's) return `Ok(None)`, matching
  `SystemOperator::linearized_diagonal`'s and `ReducedSystemOperator`'s
  existing refusal cases exactly.
- Proof: `tests/nested_p1_diagonal.rs` binds two real `SystemOperator`s (a
  coarse and a fine `Conduction` instance of the `two-material-conduction.res`
  fixture `ConnectionRealizationPlan::nested_p1` already proves coverage over)
  with a constant-conductivity flux closure and no essential constraints, then
  checks `system_linearized_diagonal` against bounded exhaustive probing: every
  unit vector through the same two-operator connected reduced action,
  reconstructed from the public `ConstraintSet::expand_homogeneous` /
  `restrict_transpose` / `direction_residual` primitives exactly as
  `crate::constraint::apply_constrained_action` composes them internally.
  Measured worst relative discrepancy is exactly `0` (the fixture's `k` is
  state-independent, so the Jacobian is exact to roundoff). The test also
  asserts at least one interface-adjacent master row differs from the
  unconnected operator's own diagonal, so the proof cannot pass vacuously
  against a formula that silently ignores the connection.
- Open (not in this package; see "Limits and next work"): the general
  multi-relation `system_constraints` union-find path (matching, N > 1
  relations) has no analogous diagonal yet; only the single-relation,
  two-operator shape (`matching` or `nested_p1`) is proved. Cross-repo need:
  Krasis's `ConnectedSystemOperator::jacobian_diagonal`
  (`krasis/src/connected.rs`) currently returns `Ok(None)` for any connection
  whose constraints carry a dependency weight other than `1.0` (its own
  comment: "Weighted interpolation needs within-leaf cross entries in
  diag(PᵀJP)... A sum of leaf diagonals is not exact for a nonmatching
  trace") -- that is precisely what `system_linearized_diagonal` now supplies;
  the Krasis lane needs to route through it to actually turn weighted-map
  Jacobi on for `nested_p1`. Not fixed here: one writer per repository.

## W9 SC-W3 package 2 investigation (2026-09-29): read-only finding, not landed

Read (not edited, per "one writer per repository") the pinned Krasis clone's
`ConnectedSystemOperator` (`krasis/src/connected.rs`) to find the actual gap
package 2 describes. Finding: `row_kinds()`, `residual()` and
`jacobian_vector_product()` are already fully generic over `ConstraintSet` --
none of the three special-case dependency count or weight, so a `nested_p1`
connection's differential/algebraic row classification, rate expansion
(`expand_homogeneous`) and transpose restriction (`restrict_transpose`)
already work identically to a `matching` connection's today, with **no
Finitum-side realization-plan change needed** for that part of package 2's
intent. The one place that *does* special-case weight is Krasis's own
`jacobian_diagonal` shortcut (see the cross-repo need above), which is a
preconditioner optimization, not a DAE-composition correctness requirement,
and is already addressed on the Finitum side by package 1.
What genuinely remains uncovered on Finitum's side: `nonmatching.rs`'s
`trace_defects` (continuity + dual-residual balance) has **no test at all**
today, for either the steady or the transient/rate case -- this predates this
wave's work. Proving the balance identity correctly needs a residual pair
that actually satisfies interface flux conservation (a real two-sided
evaluation, or a carefully manufactured pair), which is a self-contained task
I did not reach within this lane's budget; landing it rushed and unverified
would be worse than leaving it recorded here. Package 2 is therefore **not
started** as a code change (none was needed for the classification/expansion
half); the `trace_defects` proof-test gap (steady and transient) is the
concrete next step, sized independently of this wave's remaining packages.

Package 3 (planar non-nested triangulated-interface coverage proof) is **not
started**, gated on package 2 per the brief and out of reach within budget.

## Current additions

- `ConnectionRealizationPlan::system_constraints` checks each pair's scientific
  source, model, mesh, field, orientation and essential-boundary exclusions, then
  forms deterministic flat equivalence classes across multiple matching relations.
  Transpose restriction accumulates all participating residuals.
- `SystemOperator::linearized_diagonal` obtains the dynamic Jacobian diagonal
  through cell-local JVP columns at the actual state/rate and declared rate shift.
  Reduced operators preserve essential rows. Facet kernels and (single-operator,
  own) affine dependency constraints return no diagonal; there is no whole-mesh
  probing fallback. `ConnectionRealizationPlan::system_linearized_diagonal` (above)
  now covers the two-operator connected case a `matching`/`nested_p1` plan couples.
- `SurfaceTransfer::p1` interpolates a triangular surface in physical 3-D to
  nonmatching points and supplies the exact dual transpose. It preserves affine
  traces and total dual loads. Degenerate/ambiguous/outside geometry refuses.
  The point-transfer artifact alone is not a coverage proof.
- `ConnectionRealizationPlan::nested_p1` additionally proves planar, opposite,
  exterior triangular traces, absence of overlap and complete nested fine-facet
  coverage per coarse facet. It constructs weighted constraints and exposes
  continuity and dual-residual balance checks. Steady heat and species consumers
  exercise the product policy; arbitrary nonnested partitions refuse.
- Focused dynamic-diagonal/JVP comparison passes. Complete owner gate passed: 250 tests across 37 targets, formatting, check, strict clippy, rustdoc and doctests. Final consumer acceptance passed: 210 tests across 35 targets, with a documented external-fixture target retry and unchanged source.

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
  nested P1 pair; weighted-map Jacobi now has a proved diagonal for that same
  bounded shape (above). General nonnested, curved, moving or transient
  nonmatching coupling and multiplier/Nitsche stability remain open. Planar
  non-nested triangulated-interface coverage (area-weighted mortar constraints
  over arbitrary planar triangulations of the same polygon) is designed in this
  wave's brief but not implemented. Transient (DAE) use of `nested_p1` needs
  its own rate-row differential/algebraic classification and rate expansion,
  mirroring the matching plan's trace-row treatment; not implemented.
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

Final cross-repository evidence: [September 18 acceptance](../sinbad/docs/validation/2026-09-18-assembly/README.md).
