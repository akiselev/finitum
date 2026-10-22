# Batch 13 lane `finitum-data` — handoff report (batch paused by the coordinator)

Status: **paused mid-lane** on the coordinator's instruction (2026-10-22). Packages 1–4 are
landed in the library and compile clean; package 5 (tests) is written and 6 of 7 tests pass;
STATUS.md section not yet written; clippy/fmt not yet run on the final state.

## 1. Branch and final sha

`akiselev/finitum`, branch `claude/b13-finitum-data`, base `90c4789`.
Commits: `c078b06` (library), `d0f7964` (tests + fixture), plus this report commit (see `git log`).

## 2. What landed, package by package

Built against siblings at the pinned shas: scientia `84188ac`, malleus `7a43e7f`, methodus
`8bd8333`, quantitas `0d682ac`, resolvent `b604f93`, cadabra3 `ecdb54b`.

1. **Region partitions of tensor boxes** (`src/fv_topology.rs`): `TensorRegionPartition {
   regions: Vec<TensorRegion { tag, cells: Vec<[usize; 2]> }>, interfaces: Vec<TensorInterface
   { tag, between: [RegionTagId; 2] }> }` — regions by **half-open cell-index ranges per axis**
   (recorded choice: exact membership, boundary always on a face); `TensorBox::node_index_at(axis,
   coordinate)` converts a coordinate that is a node (to 4ε·extent) and refuses otherwise;
   `TensorBox::cell_counts()`. `FvTopology::from_partitioned_tensor_box(&TensorBox,
   &TensorRegionPartition)`; `cell_regions()`, `has_regions()`, `region_tags()`,
   `interfaces() -> &BTreeMap<RegionTagId, FvInterface { between, interior: Vec<usize>,
   periodic: Vec<usize> }>` (face indices into the interior/periodic blocks; block layout and
   scatter order unchanged, so `fv_reconstruction.rs` is untouched in semantics).
   `check_invariants` gains region checks and `FvTopologyInvariants { regions, interface_faces }`.
   Untagged boxes keep their digest (proved: test `w12_identities_of_uniform_data_are_unchanged`).
   Proving tests: `region_partition_tags_cells_and_interfaces_and_keeps_the_block_layout`,
   `region_partition_refusals` (both pass).
2. **Per-region datum values** (`src/fv.rs`): `FvDatumValue::{Uniform(f64),
   PerRegion(BTreeMap<RegionTagId, f64>)}` (`#[serde(untagged)]`: number or tag map on the wire,
   `FV_DATUM_VALUE_SCHEMA = "finitum-fv-datum-value/1"`); `FvRealization::bind(system, program,
   topology, regions, &BTreeMap<String, FvDatumValue>)`; `FvRealization::new` unchanged and
   delegates with `Uniform`. Per-region data bind as per-cell stride-1 columns for the source
   kernel and as the **owner (`minus`) cell's value for every face kernel** (face flux,
   characteristic speed, ghost, prescribed flux) because Scientia's `Datum` operand has no side
   (documented in the module doc and on `FvDatumValue`). Uniform data: same buffers, same
   `digest()`, same receipt bytes (pinned). `FvRealizationReceipt.region_data:
   Vec<FvRegionDatumReceipt { name, values }>`; `FvRealization::data()` returns the bound values.
   Proving test: `per_region_binding_refusals` (passes);
   `per_region_source_and_face_data_match_hand_assembled_residuals_bitwise` (**fails**, see §7).
3. **Regionwise two-point diffusion**: `FvRegionDiffusion { coefficients: BTreeMap<RegionTagId,
   f64>, averaging: FvFaceAveraging::{Harmonic, Arithmetic} }`,
   `FvRealization::with_region_diffusion`; face coefficient `2κ⁻κ⁺/(κ⁻+κ⁺)` (0 when both vanish)
   or `(κ⁻+κ⁺)/2`, owner's `κ` at a ghost boundary; receipt field `region_diffusion:
   Option<FvRegionDiffusionReceipt { scheme: "two_point", averaging, coefficients,
   max_nonorthogonality }>`. `FvDiffusion { coefficient }` / `with_diffusion` unchanged and
   bit-identical (transmissibility path, identity, receipt bytes, residual and JVP bytes pinned).
   Proving test: `two_region_steady_conduction_is_exact_with_harmonic_averaging` (passes: on 2,
   4, 10, 16, 50 cells the harmonic solution matches 400−160x / 320−40(x−0.5) within the
   predeclared bound 6.5·n³·ε·400, interface flux 160 and interface temperature 320 from both
   sides; arithmetic averaging errors recorded: 8.78 K @2, 5.93 @4, 2.69 @10, 1.73 @16, 0.569 @50
   cells, interface flux 195.1…161.2 instead of 160).
4. **Serde requests**: `Deserialize` derived on `FvRealizationReceipt`, `FvKernelReceipt`,
   `FvDerivativeDisposition`, `FvReconstructionReceipt`, `FvDiffusionReceipt`, `FvCfl`,
   `FvConservationReceipt`, `FvFaceFluxes`, `FvTopologyInvariants`, `FvCellShape`,
   `TensorBox`/`TensorAxis`/`AxisSpacing`, `PlanarCoverageReport` (W10), and (cross-lane, additive)
   `RegionTagId`. Optional fields carry `#[serde(default)]`. Proving test:
   `receipts_round_trip_through_serde` (passes; also checks `FvCfl` with `+∞` serializes as
   `null` — see §8).
5. **Tests** `tests/sv2c2c_fv_regions.rs` + fixture `tests/fixtures/fv-regions.res`
   (`Conduction1D` with parameters `a`, `t_hot`, `t_cold` and Dirichlet ghosts; `SourcedAdvection2D`
   with per-cell source parameter `q`).

## 3. Tests run (exact commands and counts)

- `cargo test -p finitum --test sv2c2c_fv_regions`: 7 run, **6 passed, 1 failed**
  (`per_region_source_and_face_data_match_hand_assembled_residuals_bitwise`).
- After commit `c078b06` (library only): `cargo test -p finitum --test sv2c2a_fv --test sv2c2b_fv
  --test fc10_methods --test sv2c2a_fv_measure`: 9 + 9 + 5 + 1 passed, 0 failed.
- `cargo check -p finitum --all-targets`: clean (no warnings) after the library commit.
- **Not run**: `--lib`, clippy `-D warnings`, `cargo fmt -- --check` on the final state
  (coordinator asked to stop without further builds).
- Build note: `--locked` fails at the pinned shas because the committed `Cargo.lock` predates
  cadabra3 `ecdb54b` (`cadabra-arrangements` → `cadabra-predicates`, `slotmap` gone); builds used
  the minimally updated lock locally and `Cargo.lock` is **not committed** (restored).

## 4. Contract shapes landed (for GX-CONTRACTS C12)

- `finitum-fv-region-partition/1`: `TensorRegionPartition`, `TensorRegion`, `TensorInterface`;
  topology source digest of a partitioned box = blake3(json(schema, TensorBox, partition)).
  `FvTopology` (`finitum-fv-topology/1`) gains `cell_regions` / `interfaces` columns, absent
  from the serialization (and digest) when empty. `FvInterface`.
- `finitum-fv-datum-value/1`: `FvDatumValue` (untagged: number | {tag: number});
  `FvRealization::bind`; receipt `region_data: [FvRegionDatumReceipt]`.
- Diffusion: `FvRegionDiffusion`, `FvFaceAveraging` (`"harmonic" | "arithmetic"`),
  receipt `region_diffusion: FvRegionDiffusionReceipt`; identity suffix
  `(FV_REALIZATION_SCHEMA, prior digest, "two_point_regionwise", FvRegionDiffusion)`.
- Refusal codes (all `FinitumError::InvalidRealization("<code>: …")`): topology
  `FV_REGION_RANGE_INVALID`, `FV_REGION_OVERLAP`, `FV_REGION_UNCOVERED`,
  `FV_REGION_TAG_DUPLICATE`, `FV_REGION_TAG_OVERLAP`, `FV_REGION_INTERFACE_UNTAGGED`,
  `FV_REGION_INTERFACE_UNKNOWN`, `FV_REGION_INTERFACE_DUPLICATE`, `FV_REGION_INTERFACE_EMPTY`,
  `FV_REGION_BOUNDARY_OFF_NODE`; binding `FV_DATUM_REGION_UNPARTITIONED`,
  `FV_DATUM_REGION_MISSING`, `FV_DATUM_REGION_UNKNOWN`; diffusion
  `FV_DIFFUSION_REGION_UNPARTITIONED`, `FV_DIFFUSION_REGION_MISSING`,
  `FV_DIFFUSION_REGION_UNKNOWN` (plus existing `FV_DIFFUSION_INVALID`, `FV_TPFA_NONORTHOGONAL`).
- `FvTopologyInvariants` gains `regions`, `interface_faces` (`#[serde(default)]`).

## 5. Expected moves

None. All W12 identities pinned in the new test from `90c4789` before the change are
unchanged (topology digests of uniform/graded/periodic boxes, diffusive realization digest,
receipt, residual and JVP bytes, Burgers and 2-D advection digests/receipts).

## 6. Cross-lane edits

- `src/profile.rs`: `serde::Deserialize` derive on `RegionTagId` (needed by every receipt that
  embeds a tag; additive).
- `src/connection/nonmatching.rs`: `Deserialize` derive on `PlanarCoverageReport` (W10 request,
  named in the lane).
- `src/fv_reconstruction.rs`: one mechanical `fields: &[]` field added to its kernel `Io`
  literal; no semantic change.

## 7. Contradictions and blockers

- `Cargo.lock` at the base does not resolve with cadabra3 `ecdb54b` under `--locked` (see §3).
- Open failure: the 2-D per-region source hand assembly differs by 1 ulp in cells 9 and 11
  (top row) — the 1-D hand LLF reused for 2-D faces, or the realization's periodic face order
  (axis-0 wraps interleaved per row with that row's interior faces, versus the test's "all
  interior then all periodic") differs in operation order. Likely the latter: the realization
  pushes periodic faces into the periodic block, which is scattered *after* all interior
  faces, so the test order should already match; next session should dump both residuals per
  cell and compare the 2-D LLF kernel's `ax·u·nx + ay·u·ny` operation order against
  `hand_llf`. The 1-D owner-side-speed assertions later in the same test have not executed.
- Harmonic averaging per the lane formula is exact only for equal half-cell distances; the
  distance-weighted `(d⁻+d⁺)/(d⁻/κ⁻+d⁺/κ⁺)` would be exact on graded interfaces too. Not done
  (lane specified the formula).

## 8. Carry-overs (what the next session should do first, in order)

1. Fix or re-scope the failing bitwise assertion in `per_region_source_and_face_data_match_
   hand_assembled_residuals_bitwise` (see §7) and run the whole file; then run `cargo test
   --locked -p finitum --test sv2c2a_fv --test sv2c2b_fv --test fc10_methods --lib` (note the
   lock issue), `cargo clippy --locked -p finitum --all-targets -- -D warnings`, `cargo fmt -p
   finitum -- --check` (the new test file was not yet formatted).
2. Add the lane's STATUS.md section under the latest wave heading.
3. `FvCfl::stable_step = +∞` serializes as JSON `null` and does not read back (serde_json has no
   infinity); decide a wire form (e.g. `Option<f64>` or a string) — round trip of an
   unconstrained CFL is not proved.
4. Simplex tagged meshes: `from_tagged_mesh` still carries no `cell_regions`/interfaces (kept
   so simplex topology digests do not move); extend when a simplex regionwise case needs it.
5. `docs/re-requests/batch13-finitum-data.md` is "none" (no sibling change needed).
