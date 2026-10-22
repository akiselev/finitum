# Lane `finitum-data` (repository: finitum, branch `claude/b13-finitum-data`, base `90c4789`)

Work package: **SV2-C2c**: regionwise finite-volume data and the cross-repository requests W10/W12
filed against Finitum. This prepares SV9 E5 stage 1 (conjugate heat transfer on a single FV mesh,
decision D5: one mesh, solid and fluid regions with different coefficients) without needing any
new Scientia API; the systems realization is a later phase of this batch.

## Packages

1. **Region partitions of tensor boxes.** `TensorBox` gains an optional region partition:
   axis-aligned sub-boxes (by cell-index ranges or coordinate intervals, your choice, recorded)
   each with a region tag, plus interface tags for the faces between two regions, so an FV
   topology built from a tensor box carries `cell_regions` and tagged interior face blocks like a
   tagged simplex mesh does. Every cell is in exactly one region (invariant); the untagged box
   keeps its current digest (identity pins in existing tests unchanged).
2. **Per-region datum values.** `FvRealization::new` currently takes `data: &BTreeMap<String, f64>`
   (one uniform value per datum). Add a versioned value type (`FvDatumValue::Uniform(f64) |
   PerRegion(BTreeMap<RegionTagId, f64>)`, or an equivalent) with the uniform case producing the
   same bound buffers and the same `digest()` as today. Per-region values bind as per-cell buffers
   with stride 1 for cell-local kernels (source, admissibility) and, for face kernels, as the
   owner-side value where Scientia's `Datum` operand has no side (document this; a two-sided datum
   operand is the `scientia-diffusion` lane's business and arrives next phase).
3. **Regionwise two-point diffusion.** `FvDiffusion { coefficient: f64 }` gains a per-region form
   with harmonic face averaging `2 k⁻ k⁺ / (k⁻ + k⁺)` (arithmetic as an explicit option), the
   receipt recording the averaging and the per-region coefficients. Uniform coefficient:
   bit-identical transmissibilities and receipt digest as today.
4. **Serde requests.** Derive `Deserialize` on `FvRealizationReceipt`, `FvCfl`,
   `FvConservationReceipt`, `FvDiffusionReceipt`, `FvReconstructionReceipt`, `FvKernelReceipt`,
   `FvTopologyInvariants` and `PlanarCoverageReport` (W10 request) so consumers embed the owner
   types instead of JSON values; round-trip tests.
5. **Tests** (`tests/sv2c2c_fv_regions.rs`): a 1-D two-region steady conduction problem
   (`k = 1` on `[0, 0.5]`, `k = 4` on `[0.5, 1]`, Dirichlet 400 K / 300 K, like Sinbad's
   two-material case) solved through the existing steady path: with harmonic averaging the
   interface temperature and flux equal the analytic values (320 K, 160 W/m²-equivalent) to
   roundoff on every uniform mesh with the interface on a face (TPFA is exact for piecewise-
   constant `k` in 1-D); arithmetic averaging is not exact and the test records its error; a
   2-D two-region box with a per-region source matches a hand-assembled residual bitwise;
   topology invariants with regions; receipts round-trip; digests of uniform data unchanged
   (pin the W12 digests in the test from the current code before you change anything).

## Files you own

`src/fv_topology.rs` (regions), `src/fv.rs` (datum values, diffusion, serde), the serde derive on
`PlanarCoverageReport` wherever it lives (`src/connection.rs` or `src/transfer.rs`), `src/results.rs`
only if `ResultMesh` needs cell-region export for FV, `tests/sv2c2c_fv_regions.rs`, your STATUS
section. Do not change `src/fv_reconstruction.rs` semantics or the scalar face-kernel binding.

## Done means

Packages 1–5 landed, all existing Finitum tests still pass (`cargo test --locked -p finitum
--test sv2c2a_fv --test sv2c2b_fv --test fc10_methods` at least, plus `--lib`), the uniform-data
digests pinned unchanged, and report §4 with the exact types Sinbad binds (`FvDatumValue`,
the region partition spec, the diffusion form).

## Refuse rather than guess

Non-orthogonal meshes (keep `FV_TPFA_NONORTHOGONALITY_TOLERANCE`), moving interfaces, more than
one value per region per datum, and anything needing a new Scientia operand meaning.
