# Finitum status

Updated: 2026-10-19

## Role and frontier

Finitum owns concrete meshes, spaces, DOFs, geometry, constraints, sampling,
transfer and global realization. Scientia owns scientific/form meaning, Malleus
owns local executable kernels, Krasis owns state/history, and Methodus owns
numerical algorithms. Finitum implements Methodus operator contracts directly;
it has no dependency on Solverang's constraint engine.

SHOW-3 / bounded SC-W3 additions are accepted. The complete owner gate passes
(262 tests across 39 targets after the W10 package 3 landing below; 254
across 38 at the W9 head `a75baeb`); final Sinbad consumer coverage passed (210
tests across 35 targets with a documented external-fixture target retry) at the
2026-09-29 head, before the 2026-10-08 fixes and this landing.

## W11 F1 (2026-10-19): prepared and batched Malleus execution

- Every Malleus call site runs prepared (`malleus::prepared`, M1 head `51aa9d3`):
  `src/execution.rs` holds `PreparedExecutable` / `PreparedExecutableModule` /
  `PreparedExecutableComposition` next to the executable in its owning struct
  (`BoundBundle`, interface `BoundKernel`, functional `Node`, method
  `AffineExecutable`, `BoundComposition`; field-source kernels are prepared once
  where their closures are built). No address-keyed cache. Calls bind positionally
  over inline (<=16 operand) binding tables and a per-thread `PreparedState` /
  `CompositionState`; a positional refusal re-runs `PreparedKernel::run`, which
  returns the interpreter's exact `ExecutionError`, so `KernelExecution` text is
  unchanged. No `Interpreter::run`/`run_composition` call remains in `src/`.
- `PointBatch` (realization.rs) packs per-point operand windows strided by
  `region.offset + region.length`; `execute` is its one-point case.
  `execute_{primal,jvp}_values_batch` run one `run_batch` per kernel and return the
  outputs before the first refusal plus that refusal, so callers keep the
  point-by-point error and accumulation order. Batched: the system cell loop
  (`SystemOperator::apply_block_cell`, primal + JVP; bind compositions stay per
  point). Not yet batched (open): single-field realization point loops, facet and
  interface loops, VJP paths, the one-shot scalar property kernels (still allocate
  their operand vectors), and Finitum's per-point `BTreeMap` gathers, which now
  dominate.
- No serialized type, digest or public API changed; Sinbad needs no change.
- Timing (wall s per test binary, opt-level 2, 4 build jobs, shared host;
  before = `e415d8d` load 37-52, P1 = prepared load 33, after = +batching load 31):
  w7_system_path_parity 45.5 / 19.0 / 16.1; gx_c4_facet_integrals 26.7 / 5.7 / 5.2;
  planar_p1 15.9 / 4.6 / 4.6; sv2_elasticity 15.3 / 4.0 / 3.9;
  sv2b4_system_stokes 10.8 / 4.4 / 4.4; w7_rt0_essential 9.5 / 3.0 / 2.3.
  Load was higher for the baseline; the 2-5x ratios exceed what load explains.
- perf (`perf record -F 199`, w7_system_path_parity): before, `malleus::*` 33% and
  malloc/free 18% of samples; after, `malleus::*` 23% (point program 7%, columnar
  blocks 4%) and malloc/free 16%; the rest is Finitum gather (`PointBatch::push`
  5%, `point_inputs_system`, BTreeMap traffic).

## W10 package 3 (2026-10-09): planar non-nested coverage and mortar-type P1 constraints

- `ConnectionRealizationPlan::planar_p1(system, relation, meshes, facets, tolerance)`
  (beside `nested_p1`, same signature; Sinbad product policy value expected:
  `InterfaceTransferPolicy::PlanarP1`, serialized `planar_p1`) admits two
  arbitrary planar triangulations of the SAME polygon: endpoint zero is the
  master trace whose vertex values the constraints depend on, endpoint one the
  target trace whose every vertex is eliminated. Admission is shared with
  `nested_p1` (one `PlanarTraces::admit`, nothing duplicated): admitted
  relation, nonempty 3-D traces, distinct exterior triangular facets, every
  vertex within `tolerance` of the first master facet's plane, master normals
  equal and target normals opposite to it. Both constructors now project into
  one orthonormal in-plane frame (in-plane areas are physical); the former
  dominant-axis projection scaled by `|n_drop|` was the same overlap criterion,
  and `nested_p1`'s weights, refusals and 3 + 2 + 2 owner tests are unchanged.
- Coverage proof (in the constructor, refusing): area floor `tolerance ×
  diameter` (longest admitted edge, `1.4e-12` on the unit square at `1e-12`); a
  triangle at or below it is degenerate; an opposite-side intersection at or
  below it is classified empty; a same-side overlap above `max(1e-10 · min
  area, floor)` refuses; exact orientation-independent Sutherland–Hodgman clips
  of every target triangle against every master triangle must cover every
  triangle of BOTH sides to relative `1e-10`, hence `Σ_{a,b} |T_a ∩ T_b| =
  |Γ_master| = |Γ_target|` to `1e-10`. Carried as `plan.planar_coverage() ->
  Option<&PlanarCoverageReport>` (`area: [f64; 2]`, `intersection_total`,
  `worst_relative_defect`, `worst_constant_defect`, `relative_tolerance`,
  `area_floor`, `pairs`, `slivers_classified`; re-exported from the crate root),
  a new optional `skip_serializing_if` field of the plan: `None` for
  `matching`/`nested_p1`, whose serialized bytes and identities are unchanged.
- Constraints: the mortar projection of the master P1 trace onto the target P1
  trace tested against the **biorthogonal dual basis** `ψ_i = 4λ_i − 1` per
  target triangle (`∫_T ψ_i φ_j = δ_ij |T|/3`): `w_ik = ∫ ψ_i φ_k /
  (|patch_i|/3)`, every integral exact over the intersection polygons (fan
  triangles, degree-2 edge-midpoint rule). Chosen over the standard-basis
  `M_ff⁻¹ M_fc`: rows stay local (a target depends only on the masters whose
  triangles meet its patch), no dense inverse, and still a projection -- every
  trace already in the target space (affine traces; every trace of a nested
  pair) is reproduced, so nested consistency with `nested_p1` holds; rows sum
  to one (constant reproduction, total dual load conserved); weights can be
  negative (no maximum principle, as for any L2-type projection). Emitted
  through the new `pub(crate) SurfaceTransfer::from_rows` into the same
  `nonmatching: (SurfaceTransfer, targets)` slot, so `trace_rows`,
  `trace_vertices`, `constraints` (same `ConstraintSet` shape: target, weighted
  dependencies, zero offset), `trace_defects`, `system_linearized_diagonal`,
  `system_constraints` and Krasis's generic `row_kinds`/`residual`/
  `jacobian_vector_product`/`expand_homogeneous`/`restrict_transpose` path
  apply unchanged.
- Refusals (`FinitumError::InvalidRealization`, `CONNECTION_TRACE:` prefix),
  each asserted by message in `tests/planar_p1.rs`: `planar P1 needs an
  admitted relation and nonempty 3-D traces`; `duplicate interface facet`;
  `facet out of range`; `triangular traces required`; `interface planes do not
  coincide`; `planar opposite normals required`; `degenerate interface triangle
  below the area floor`; `overlapping interface triangles`; `interface
  triangulations do not cover the same polygon: side S triangle T is covered on
  C of its A` (one polygon not the other, either direction, or a dropped
  facet); `mortar rows do not reproduce constants: …` (integration self-check
  at `1e-9`, unreachable while coverage holds); non-exterior traces propagate
  `exterior_facet`'s `UnsupportedRealization` (`facet N is interior; …`).
  `constraints()`'s scalar-P1, orientation and essential-overlap refusals are
  unchanged and apply. A thin-but-admitted fan (vertex `1e-3` off an edge) and
  a grazing fan whose `~2.5e-13` slivers against the master diagonal are
  classified empty (2 classified, worst relative coverage defect `1.0e-12`,
  plan admitted) show the floor classifies rather than forbids.
- Fixtures (`tests/support/planar_meshes.rs`, transforms keep cells and facet
  ids; orientation asserted): the right 1x1x1 box turned a quarter turn about
  `x` against the left 1x1x1 box (the unit square split along opposite
  diagonals, 2 vs 2 triangles, 4 pairs); the right 3x3x3 box deformed in-plane
  by `(0.05, −0.07) sin πy sin πz` against the left 2x2x2 box (8 vs 18
  triangles, 43 pairs; every interior target vertex `> 1e-2` from every master
  vertex and edge, asserted); the nested 1 vs 2 pair; one-apex tetrahedral fans
  for the overlap, degenerate, thin and grazing cases. The operator, closure
  and exhaustive-probe helpers `tests/nested_p1_diagonal.rs` held inline moved,
  unchanged, to `tests/support/connected_conduction.rs`, shared by both proofs.
- Proofs (`tests/planar_p1.rs`, 8 tests; worst discrepancies as printed):
  1. coverage: `Σ|T_a ∩ T_b| = 1 =` both areas on all three pairs; worst
     relative coverage defect `0` / `3.4e-16` / `0` (opposite diagonals /
     irregular / nested), worst row-sum defect `0` / `1.6e-15` / `2.2e-16`;
     `nested_p1` refuses both non-nested pairs and carries no report;
  2. affine reproduction at every target vertex: `0` (opposite diagonals; 2
     rows with several non-trivial weights, 2 negative weights) and `4.5e-13`
     (irregular, values ~300 K; 12 such rows, 34 negative weights), gate `1e-12`;
  3. nested consistency: `1.1e-16` entrywise over 14 entries against
     `nested_p1`, no entry present on one side only;
  4. `<R x, y> = <x, Rᵀ y>`: `1.3e-16` relative; total dual load of target loads
     after `restrict_transpose`: `3.8e-16` relative;
  5. `trace_defects`: continuity `4.5e-13` for constant and linear traces,
     balance `0` relative against `restrict_transpose` for a state residual and
     a mass-matrix (storage) residual;
  6. `system_linearized_diagonal` against the bounded exhaustive reduced-JVP
     probe (`1e-9` gate) on the irregular pair: steady `3.7e-16`; storage
     fixture `two-material-transient-conduction.res` `2.9e-16` at `rate_shift`
     0 and `3.6e-16` at 3, with 5 masters prolongating into several targets
     with non-trivial weights -- the general cross term `Σ w_β J_B[β,γ] w_γ`
     exercised for the first time beyond one dependency -- identity essential
     and target rows, and `shift · rho · cp · M_ii` growth on interior rows;
  7. the refusals above, each by code.
- Open: curved or moving interfaces, multiplier/Nitsche stability, the
  multi-relation (`system_constraints` union-find, N > 1) diagonal, n-way
  junctions, transient nonmatching composition (Krasis's). Recorded
  limitation (pre-publication review, 2026-10-09): the sliver floor is
  absolute (`tolerance × diameter`) while the per-triangle coverage check is
  relative (`1e-10`), so a genuine sub-floor sliver against a triangle whose
  area is below about `floor × 1e10` (`1.4e-2` on the unit square at `1e-12`)
  is dropped from `covered` and the pair refuses "do not cover the same
  polygon" although the geometry is admissible; the direction is
  conservative (a refusal, never a false admission) and no fixture reaches
  it (smallest triangle `1/18`). The
  `system_constraints` refusal wording "nested nonmatching currently requires
  one two-component relation" now also covers `planar_p1` plans (unchanged
  text). Cross-repo need (Sinbad lane, next wave): `InterfaceTransferPolicy::
  PlanarP1` (`planar_p1`) in `sinbad/src/case.rs`, dispatched in
  `connected_run.rs` beside `NestedP1` to `ConnectionRealizationPlan::planar_p1`
  under the same steady-only guard; Sinbad's "arbitrary nonnested interfaces
  refuse" boundary becomes "curved/moving refuse". Krasis needs nothing new.

## W9 SC-W3 package 1 (2026-09-29): weighted-map Jacobi diagonal

- `ConnectionRealizationPlan::system_linearized_diagonal` lands the reduced
  Jacobian diagonal `diag(Rᵀ J R)` of the two-operator connected system a
  `matching` or `nested_p1` plan couples: each operator's own essential rows
  and every connection-target (slave) row are preserved as the reduced
  system's own identity row (`1.0`, as `ReducedSystemOperator`'s own
  `jacobian_diagonal` does) -- true only since the 2026-10-08 review fix
  below: the 2026-09-29 landing identity-rowed the connection targets only
  and left the physical `J_ee` on each operator's own essential rows while
  claiming otherwise. A master row untouched by the interface keeps its own
  operator's `SystemOperator::linearized_diagonal` value; an interface-adjacent
  master row gains the transpose-scattered cross term `Σ w_β J_B[β,γ] w_γ`
  over the target trace DOFs it prolongates into (for `matching`, `J_B[β,β]`,
  the sum-of-leaf-diagonals convention Krasis uses). Contract: the caller
  passes the per-operator slices of the connection-REDUCED state/rate, and
  the point is expanded through the connection first, then through each
  operator's own essential constraints, as every other connected reduced path
  does (also a 2026-10-08 fix: the landing expanded only through each
  operator's own constraints, so operator B was linearized at whatever the
  caller's slice held on the fine-trace rows). Previously "weighted-map
  Jacobi refuses" (any affine-dependency constraint set returned no diagonal
  at all); this closes that gap for the bounded one-relation/two-operator
  shape `system_constraints` already restricts nonmatching connections to.
- New `SystemOperator::linearized_block` (`pub` since 2026-10-08, with its
  contract documented: a bounded dense `rows x columns` sub-block
  `J + rate_shift·M` of the physical operator from the cell-local JVP
  machinery, work scaling with the requested columns and the cells touching
  them, so Krasis can fold N-leaf cross terms itself later) computes the
  bounded dense off-diagonal Jacobian sub-block restricted to a given
  row/column DOF set from the same cell-local JVP machinery
  `linearized_diagonal` uses: work scales with the requested
  (interface-sized) column count and the cells that touch them, never with
  the whole mesh. It is the only new primitive; no change to
  `linearized_diagonal`, `apply_action`, `apply_reduced_action`, or any
  existing constraint/transfer type.
  Refusal is unchanged and reused, not widened: either operator's own
  facet-kernel operators or its own affine-dependency essential constraints
  (periodic-style, not this connection's) return `Ok(None)`, matching
  `SystemOperator::linearized_diagonal`'s and `ReducedSystemOperator`'s
  existing refusal cases exactly.
- Proof (`tests/nested_p1_diagonal.rs`, three tests; oracle revised
  2026-10-08): bounded exhaustive probing -- every unit vector of the
  connection-reduced space, prolongated through the pair and pushed through
  the two REDUCED operators' own `jacobian_vector_product` with the rate
  direction `rate_shift · d` that `linearized_diagonal` probes, restricted by
  `Rᵀ`, target rows from `direction_residual` -- composed exactly as Krasis's
  `ConnectedSystemOperator::jacobian_vector_product` composes its leaves,
  including the pair-expanded linearization point. (The 2026-09-29 oracle
  was the raw `apply_action`: a zero rate direction on the unreduced
  operators, which could see neither essential rows nor `rate_shift`.) Gate
  `1e-9` relative per row; measured worst discrepancies are printed by each
  test (`0` on the two state-independent fixtures; `2.5e-16` and `2.1e-16` on
  the storage fixture at shifts 0 and 3).
  - `nested_p1_diagonal_matches_bounded_exhaustive_probing`: the steady
    `two-material-conduction.res` coarse/fine `Conduction` pair with a
    constant-conductivity flux closure and one essential constraint per side
    on a non-interface vertex. Also asserts an interface-adjacent master row
    differs from the unconnected diagonal, every target row is `1.0`, and
    each essential row is `1.0` where the physical diagonal is not.
  - `nested_p1_diagonal_sees_rate_shift_and_expands_the_point_through_the_connection`:
    the storage-bearing, state-dependent fixture
    `tests/fixtures/two-material-transient-conduction.res`
    (`rho * cp * dt(T) + div(q) = 0`, `q = -k(T) grad(T)`,
    `k(T) = k0 (1 + k1 T)`, all bound as closures). The caller's fine-trace
    entries are deliberately wrong (0 K where the expansion gives ~300 K) and
    operator B's own diagonal is first shown to depend on them; the connected
    diagonal then matches the probe at `rate_shift` 0 and 3, grows with the
    shift at every free/master row, and on interior coarse rows grows by
    exactly `shift · rho · cp · M_ii` against `SystemOperator::mass_matrix`
    (independent of the JVP machinery under test).
  - `matching_diagonal_matches_probing_and_the_sum_of_leaf_diagonals_convention`:
    the `matching` plan on the 2x2x2 pair (9 weight-1 vertex pairs) with one
    essential constraint per side: the same probe, plus entrywise equality
    (`1e-12`) with the sum-of-leaf-diagonals convention Krasis's
    `ConnectedSystemOperator::jacobian_diagonal` uses (`Rᵀ` of the
    concatenated leaf `DaeOperator::jacobian_diagonal`s, identity targets).
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

## W9 SC-W3 package 2 (2026-09-29): transient nonmatching proof checks

Read (not edited, per "one writer per repository") the pinned Krasis clone's
`ConnectedSystemOperator` (`krasis/src/connected.rs`) to find the actual gap
package 2 describes. Finding: `row_kinds()`, `residual()` and
`jacobian_vector_product()` there are already fully generic over
`ConstraintSet` -- none of the three special-case dependency count or weight,
so a `nested_p1` connection's differential/algebraic row classification, rate
expansion (`expand_homogeneous`) and transpose restriction
(`restrict_transpose`) already work identically to a `matching` connection's
today. **No `ConnectionRealizationPlan`/`nested_p1` shape change was needed**
to make that true; the deliverable is the proof that it holds, which is what
this package lands. The one place Krasis *does* special-case weight is its
own `jacobian_diagonal` shortcut (the cross-repo need above), a preconditioner
optimization already addressed on the Finitum side by package 1, not a
DAE-composition correctness requirement.

`tests/nested_p1_transient.rs` gives `nonmatching.rs`'s `trace_defects`
(continuity + dual-residual balance) its first Finitum-owned test, steady or
transient/rate (Sinbad's connected run consumed it before, so "no test
coverage at all" overstated the gap):

- `fine_side_trace_continuity_holds_for_constant_and_linear_in_time_traces`:
  `trace_defects`'s own continuity number is `<= 1e-12` for a constant trace,
  for a linear-in-time trace `value(x, t) = value0(x) + rate(x) * t` sampled
  at two different times (the map carries no time dependence of its own, so
  continuity holding at each independent snapshot is the concrete meaning of
  "at any state"), and for the bare rate field `rate(x)` itself -- rate
  expansion is the same primal interpolation as value expansion, since a
  trace constraint's offset is always zero.
- `dual_residual_balance_identity_matches_restrict_transpose_for_state_and_storage_residuals`:
  `trace_defects`'s returned balance number matches an independently computed
  `ConstraintSet::restrict_transpose` reference to `1e-12` relative, for an
  arbitrary state-like residual pair *and* for a genuinely storage-bearing
  residual pair built from each side's own `SystemOperator::mass_matrix`
  action on a rate field (the discrete term a `rho * cp * dt(T)` storage
  equation contributes) -- "on both sides" literally, one mass matrix per
  mesh. A further direct check confirms the same result is the general
  adjoint identity `<R x, y> = <x, Rᵀ y>` via `expand_homogeneous` /
  `restrict_transpose` alone, not an artifact of `trace_defects`'s own
  bookkeeping. Limitation: this proves the transpose bookkeeping (the adjoint
  identity holds for any vector, storage-shaped or not), not a physical
  storage-equation balance across the interface; no time-stepped two-side
  energy balance is asserted anywhere in Finitum.

Both tests bind two real `SystemOperator`s (coarse/fine `Conduction`) over the
same `two-material-conduction.res` fixture and `nested_p1` plan package 1's
proof uses. Package 2 needed no time stepping or history (Krasis's own); no
production code change, only the proof.

Package 3 (planar non-nested triangulated-interface coverage proof) was not
started in W9; it landed 2026-10-09 as `planar_p1` (W10 section above).

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
  now covers the two-operator connected case a `matching`/`nested_p1` plan couples,
  each operator's essential rows included (since the 2026-10-08 review fix).
- `SurfaceTransfer::p1` interpolates a triangular surface in physical 3-D to
  nonmatching points and supplies the exact dual transpose. It preserves affine
  traces and total dual loads. Degenerate/ambiguous/outside geometry refuses.
  The point-transfer artifact alone is not a coverage proof.
- `ConnectionRealizationPlan::nested_p1` additionally proves planar, opposite,
  exterior triangular traces, absence of overlap and complete nested fine-facet
  coverage per coarse facet. It constructs weighted constraints and exposes
  continuity and dual-residual balance checks. Steady heat and species consumers
  exercise the product policy. `planar_p1` (W10, above) admits arbitrary planar
  triangulations of the same polygon with exact pairwise-intersection coverage
  and dual-basis mortar rows; curved and moving interfaces refuse.
- Focused dynamic-diagonal/JVP comparison passes. Complete owner gate passed: 262 tests across 39 targets, formatting, check, strict clippy, rustdoc and doctests (2026-10-09; 254 across 38 on 2026-10-08). Final consumer acceptance passed at the 2026-09-29 head: 210 tests across 35 targets, with a documented external-fixture target retry and unchanged source.

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
  nested P1 pair; weighted-map Jacobi has a proved diagonal for that same
  bounded shape, essential rows included only since the 2026-10-08 review fix
  (above), and the transient/rate proof checks Krasis needs for
  a `nested_p1` connection (fine-side continuity, dual-residual balance, both
  at any state and rate) are proved in `tests/nested_p1_transient.rs`.
  Planar non-nested triangulations of the same polygon are admitted by
  `planar_p1` with a proved coverage and dual-basis mortar rows (W10, above);
  curved or moving nonmatching coupling and multiplier/Nitsche stability remain
  open; actual time stepping/history for a transient nonmatching connection is
  Krasis's own composition, not built here.
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
