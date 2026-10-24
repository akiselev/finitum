# Finitum status

Updated: 2026-10-24

## Role and frontier

Finitum owns concrete meshes, spaces, DOFs, geometry, constraints, sampling,
transfer and global realization. Scientia owns scientific/form meaning, Malleus
owns local executable kernels, Krasis owns state/history, and Methodus owns
numerical algorithms. Finitum implements Methodus operator contracts directly;
it has no dependency on Solverang's constraint engine.

SHOW-3 / bounded SC-W3 additions are accepted. The complete owner gate passes
(286 tests across 43 targets after the V8 CAD-body landing below; 281 across 42 after the W12 L6 landing; 272 across 41 after L4; 262 across 39 after W10
package 3; 254
across 38 at the W9 head `a75baeb`); final Sinbad consumer coverage passed (210
tests across 35 targets with a documented external-fixture target retry) at the
2026-09-29 head, before the 2026-10-08 fixes and this landing.

## R6D V8 (2026-10-24): tetrahedral realization of a committed CADabra body; the consumer gate

Branch `claude/b7-finitum` from `90c4789` (CADabra batch 7, lane finitum). Siblings: cadabra3
`claude/b7-finitum` (base `3e77160b`, provider unchanged), resolvent `b604f93`, Malleus `7a43e7f`,
Scientia `84188ac`, Methodus `8bd8333`, Quantitas `0d682ac`; Sinbad `fb11d4a`, Krasis `d981980`,
Artifactum `2c016f1` for the consumer run. Not yet integrated.

- `CadBodyRealization::extruded(&BrepBodyProvider, &DisplayArtifact, ExtrusionRefinement{spacing,
  layers})` (`src/cad_body.rs`, `finitum.cad-body-realization/v0`): a committed Provider V1 body
  that is an extrusion (two planar caps with opposite normals, every other face perpendicular,
  side samples only on the cap rims) becomes a P1 tetrahedral `Mesh`. The display artifact must
  be of the provider's revision (`FinitumError::StaleBrepRevision`); a body that is not such an
  extrusion is refused (`InvalidCadGeometry`), never mis-meshed.
- Boundary from the published tessellation, interior from Finitum (`src/cad_profile.rs`): the
  bottom cap's rim samples (numbered in the mesher's emission order); rim edges longer than the
  spacing split evenly; a hexagonal lattice of interior points at least half a spacing from the
  rim; Lawson flips to the Delaunay triangulation; triangles in canonical order; `layers` uniform
  layers; three tetrahedra per prism split by global vertex order. The published cap triangulation
  only seeds the topology (its fans of near-half-turn triangles do not converge for P1: compliance
  on it moved 30-100% between refinements and gave `dJ/dx0 = 3042` where symmetry gives 0). No new
  dependency; `cadabra` (the CADabra facade, path) is a dev-dependency only, to commit Booleans in
  tests (its lock adds `cadabra`, `cadabra-ops`, `cadabra-ssi`, all path crates).
- Association: every vertex has `CadBodySupport::{Interior, Face, Edge{edge, faces}, Vertex{vertex,
  faces}}` (binding indices) and one `CadBoundaryIncidence{vertex, face, point, normal}` per face it
  lies on; `face_vertices`, `nodal_dof_map`, `digest` (source revision, binding digest, mesh,
  supports, incidences, motion), `require_source` (`StaleBrepRevision` /
  `CadGeometrySourceMismatch`).
- Parameter JVP/VJP: `boundary_velocity` (provider `V_n` per incidence; a snapped decision moving
  along the seeds is `FinitumError::SnappedCadDesign{exit}`), `mesh_velocity` (the mesh-motion JVP)
  and `mesh_velocity_transpose`, `design_velocity` (JVP), `design_gradient` (VJP through the
  provider's `boundary_velocity_transpose`). The motion is the derivative of the generator: a
  published bottom sample moves by the minimum-norm displacement meeting each of its faces' `V_n`
  (no tangential gauge), a rim split by interpolation of its end points, a lattice point with its
  cap only, layer `k` by `k/layers` of the extent rate.
- `RealizationPlan::residual_geometry_sensitivity_transpose(time, state, adjoint, node_gradient)`
  (`src/realization.rs`): the exact transpose of `residual_geometry_sensitivity` in the node
  velocities for stored inputs whose design direction is zero (cell-local columns, one vertex
  component at a time; constraint rows ignored as the forward zeroes them). The forward's support
  checks moved to `require_geometry_sensitivity_support` (no behaviour change).
- Evidence `tests/cad_body_campaign.rs` (5; log `docs/validation/2026-10-24-v8-cad-body/`): box
  `[-2,2]^3` minus a through bore, two designs (r 0.6 at x0 0; r 1 at x0 0.3), objectives `J_w =
  sum of the P1 load of -div grad u = 1 + x + 2y` (exact reference from the provider's volume and
  centroid) and the compliance `J_c` of `-div grad u = 1`, `u = 0` on the bore (CG + Jacobi over
  `element_assembly`), parameters bore radius, bore position, box width, refinements (chord 1e-2,
  spacing 0.5, 8 layers: 891/927 vertices) and (1e-3, 0.25, 16: 5610/6086 vertices).
  - Moved mesh (the Finitum chain): centred differences at 1e-2/1e-3/1e-4 second order and Taylor
    remainders falling 100x per decade for every parameter, design, level (`J_c` at the coarse
    level), down to rounding.
  - Through the CAD (publish, tessellate, realize, assemble): `J_w` centred differences equal the
    Finitum rate to 1e-10 for radius and width whenever the topology is unchanged; `J_c` to 8e-7
    (radius, rate ~100-200) and 4e-8 (width) absolute at 1e-4, falling 100x per decade. The bore position differs by the tangential gauge
    (published rim samples ride the tool's chart): `J_w` 0.089/0.153 -> 0.0098/0.016, `J_c`
    0.18/0.63 -> 0.023/0.16 under refinement.
  - Refinement: `J_w` error 0.091/0.202 -> 0.0098/0.021; rate errors radius 0.30/0.40 ->
    0.033/0.043, position 0.18/0.31 -> 0.020/0.033 (first order in the sagitta); width exact.
  - VJP = JVP per parameter to 1e-9; `<dJ/dX, X'(d)> = <dJ/dp, d>` to 1e-10; mesh-motion and
    residual-sensitivity transpose identities to 1e-12/1e-11; assembled columns vs combined
    direction to 1e-13; matrix-free vs element assembly to 1e-12.
  - Refusals: flush cap (snapped, height seed) `SnappedCadDesign{exit: Some}` for JVP and VJP, radius
    defined; stale revision `StaleBrepRevision` (provider of another revision; display of another
    revision); the kernel's `WrongRevision`; blind bore refused as a non-extrusion.
- Open: non-extruded bodies (blind bores, bosses, curved caps) need a general tetrahedral mesher
  from the published surface; the bore position's tangential gauge needs a provider point velocity
  in the face chart's gauge (requested); topology changes of the regenerated mesh at large steps
  (lattice threshold, rim sample counts) are skipped, not differentiated; debug-mode CG on ~6k
  vertices is ~8 s per solve, so `J_c` runs its full campaign at the coarse level only.

## W12 L4 SV2-C2a (2026-10-19): FV topology and first-order realization

Working clone, not yet integrated or consumer-accepted. Siblings: Scientia L1 `7819884`
(`conservation.rs`), Methodus L2 `8bd8333` (`ExplicitOperator`), Malleus `3005489`.

- `MeshProfile::TensorBox(TensorBox{axes: Vec<TensorAxis{spacing: AxisSpacing::{Uniform{min,
  max,cells} | Nodes{nodes}}, periodic}>})`, 1-D/2-D (D2). Existing profiles, their
  `realize` output and identities are unchanged; `realize(TensorBox)` refuses
  (`MeshProfileUnsupported`: a rectangle is not a simplex `Mesh` cell).
- `FvTopology` (`finitum-fv-topology/1`, `src/fv_topology.rs`): `from_tensor_box`,
  `from_tagged_mesh` (simplex 1-3-D via `FacetTopology`; every exterior facet needs exactly
  one tag: `FV_BOUNDARY_UNTAGGED`/`_TAG_OVERLAP`/`_TAG_INVALID`), `from_profile`. Cells:
  volume, centroid. `FvFaceBlock` SoA: `minus`, `plus` (empty on boundary), `area`,
  `normal[axis][face]` (unit, minus->plus / outward), `centroid`, `delta`, `translation`
  (periodic only). Blocks: `interior`, `periodic` (last cell -> first across the wrap),
  `boundary: BTreeMap<RegionTagId, _>` (`x_min/x_max/y_min/y_max`, SimplexBox names). Tensor
  cells are axis-0-fastest; faces by axis, row, position. `check_invariants` ->
  `FvTopologyInvariants` (positive measures, unit normals, every cell on exactly
  `faces_per_cell` face sides, closure `‖Σ s|f|n‖∞/Σ|f| ≤ 8·faces·ε`; tensor closure is 0).
  `digest()` over every column; `source_digest()` = tagged-mesh or tensor-profile digest.
- `FvRealization::new(system, Arc<ConservationLawProgram>, Arc<FvTopology>, &RegionMap,
  &BTreeMap<String,f64>)` (`finitum-fv-realization/1`, `src/fv.rs`). Binds kernels by
  `FvOperandMeaning` (minus = owner, plus = neighbour or the region's `Ghost` output;
  `PrescribedBoundaryFlux` replaces the face flux). Every declared boundary region must map to
  existing tags and every tag be claimed once (`FV_BOUNDARY_UNMAPPED`, `_TAG_UNKNOWN`,
  `_OVERLAP`, `_UNCLOSED`). Data: one finite uniform value per non-time datum
  (`FV_DATUM_MISSING`/`_UNKNOWN`); `Time` data bind `t`. Refuses MUSCL
  (`FV_RECONSTRUCTION_UNSUPPORTED`, L6), positivity, dimension and identity mismatches.
- Evaluation: per face block, gather into reusable `Mutex` scratch, one `run_batch` (state,
  normal, output stride 1; data stride 0), scatter in fixed order interior, periodic,
  boundary tags (owner `+= |f|F*`, neighbour `-=` the same value). Residual
  `R = |Ω|u̇ + Σ s|f|F* − |Ω|S`; `rhs = −(Σ s|f|F* − |Ω|S)/|Ω|`. Diagnostics
  (`face_fluxes`, `conservation_receipt`) allocate; the operator paths reuse scratch.
- `DaeOperator` (residual, JVP) and `methodus::ExplicitOperator` (`rhs`, `stable_step` =
  `min_i |Ω_i|/Σ_f |f|λ_f`, λ_f the larger `characteristic_speed` of the two face states — a
  sufficient bound, half the sharp 1-D upwind step; `linear_invariants` = [cell volumes];
  `invariant_rates` = −Σ_b|f|F_b + Σ|Ω|S, Neumaier). JVPs via `FvKernel::differentiate`:
  a refused kernel (D6) makes the JVP a typed `FinitumError::InputEvaluation` ->
  `NumericError::Evaluation{code: "FV_NONSMOOTH_DERIVATIVE_REFUSED", origin: "fv/face_flux"}`;
  primal unaffected.
- `cfl() -> FvCfl{stable_step, limiting_cell, max_face_speed}`; `receipt() ->
  FvRealizationReceipt{digest, program/intent/topology digests, kernels[{role, tag,
  kernel_digest, derivative: Smooth|Zero|NotDifferentiated|Refused{code}}]}`; `digest()` over program, topology,
  data values and boundary binding (for Krasis `OperatorIdentity`).
- Conservation: `discrete_flux_balance(topology, &FvFaceFluxes, cell_flux_residual) ->
  FvConservationReceipt` (`finitum-fv-conservation/1`): Neumaier Σ cells vs Σ boundary
  `|f|F*`, `balanced` iff relative imbalance `≤ (faces_per_cell+2)·ε` of `Σ|terms|`;
  `scatter_face_fluxes` is the residual's own scatter.
- FC10 `FiniteVolumeRealization` (`finitum-finite-volume/1`) untouched. `Cargo.lock` gains
  `libm 0.2.16` only through the Malleus pin (no new Finitum dependency).
- Evidence: `tests/sv2c2a_fv.rs` (9): tensor numbering/periodic pairing/graded 2-D/invariants
  /identity/`realize` refusal; simplex 1-3-D invariants; residual bitwise equal to a hand
  first-order LLF scatter and within 16ε of pure upwind; ghost/prescribed boundaries; receipt
  balanced and a leaking scatter (1e-3, 1e-9) refused; 2-D free stream to round-off;
  `integrate_explicit` FE and SSP-RK3 on periodic 1-D advection with per-step balance
  residual `≤ (cells+4·stages)ε·scale`, mass drift ≤ 1e-13, TVD bounds; JVP exact for linear
  LLF, O(h²) for central Burgers, refused for LLF Burgers; BDF1 step mass `m/(1+kΔt)`;
  binding refusals. `tests/sv2c2a_fv_measure.rs` (1): repeated evaluation bitwise identical.
- Measurement: residual of 1-D LLF advection, 100 000 cells, 20 repeats, opt-level 2, host
  thinkpadx1 (12 threads, shared with other builds): 106 ns/face. Not optimized; no claim.
- Open: simplex periodic pairing; spatially varying data/provider fields; systems
  (multi-component); admissibility hooks; a zero-allocation proof (the crate forbids
  `unsafe`, so no counting allocator). MUSCL/limiters/LSQ/diffusion/VJP: L6 below.

## W12 L6 SV2-C2b (2026-10-19): MUSCL, limiters, TPFA diffusion, derivative chain

Working clone on L4 `b935dd2`; siblings Malleus `7a43e7f` (`malleus::ad`), Scientia L1
`7819884`. Not yet integrated or consumer-accepted.

- `FvReconstruction` (`src/fv_reconstruction.rs`): unweighted linear LSQ gradients over face
  neighbours in topology face order (periodic translated), CSR weights `M⁻¹d`; rank-deficient
  cells (simplex corners) extend to neighbours' neighbours, else zero slope and counted
  (`degenerate_cells`). `FvLimiter::{Unlimited, Sweby(FvKernel), BarthJespersen,
  Venkatakrishnan{k}}`. Sweby (tensor only, else `FV_LIMITER_UNSUPPORTED`): the Scientia
  `Limiter` kernel over `SlopeRatio`, one `run_batch` over all cell-axis pairs, `σ=φ(r)s⁻`,
  `r=0` when `s⁻=0`, zero slope at non-periodic boundaries; non-finite `φ` ->
  `FV_LIMITER_INVALID`. BJ/Venkatakrishnan (`ε²=(k|Ω|^{1/d})³`, clamped to 1) on any mesh.
  Diagnostics `slopes`, `face_states`; `derivative_refusal()` (Sweby via `malleus::ad`
  `NonsmoothPolicy::Refuse`, `FV_NONSMOOTH_DERIVATIVE_REFUSED` with kink sites).
- `FvRealization` takes `Muscl` intents: conserved variables only (`Primitive` ->
  `FV_PRIMITIVE_RECONSTRUCTION_UNSUPPORTED`), `LeastSquares` anywhere, `Axis1D` tensor only,
  `GreenGauss` -> `FV_GRADIENT_UNSUPPORTED`. Face, ghost (reads the reconstructed interior
  state) and speed kernels read reconstructed face states; CFL uses them too.
- `with_diffusion(FvDiffusion{coefficient})`: TPFA `−κ(u⁺−u⁻)/(δ·n)` with cell values on
  interior/periodic/ghost faces (half-cell `δ` at a ghost; prescribed flux is total).
  Refuses `max_nonorthogonality() > 1e-10` rad (`FV_TPFA_NONORTHOGONAL`, all 2-D/3-D simplex
  boxes) and bad coefficients (`FV_DIFFUSION_INVALID`). CFL sums `|f|(λ_f+κ/(δ·n))`.
  Identity `blake3(schema, base digest, "two_point", κ)`; without diffusion unchanged.
- Derivatives: JVP through unlimited reconstruction (linear); `state_vjp(t,u,w,out)` =
  `(∂R/∂u)ᵀw` from unit-tangent face/ghost/source JVP partials and the exact transpose of
  reconstruction (allocates per call). Any limited reconstruction refuses JVP and VJP
  (`InputEvaluation`, origin `fv/limiter`); primal unaffected.
- Receipt adds skip-if-none `muscl{variables, gradient, limiter, limiter_kernel_digest,
  degenerate_gradient_cells, derivative}`, `diffusion{scheme, coefficient,
  max_nonorthogonality}`, `max_nonorthogonality`; `reconstruction` = `muscl`. L4 first-order
  digests and receipt bytes pinned (`first_order_identities_are_pinned`).
- Evidence `tests/sv2c2b_fv.rs` (9) + fixture `fv-muscl.res`: LSQ exact for linear fields
  (graded tensor, simplex) at cells and faces; BJ strictly bounded, Venkatakrishnan within
  O(ε²) on a simplex step; transpose identity ≤ 64ε·scale (Burgers central periodic, inflow
  LLF + diffusion, 2-D simplex LLF) and Taylor order 1.9–2.1 (Burgers); D6 refusals for the
  four Sweby kernels, Scientia agreeing; TVD + min/max on a 100-cell square wave, 120
  SSP-RK3 steps at the stable step, each Sweby limiter (unlimited control fails); L1 orders
  128->256 on sin(2πx), t=0.5, SSP-RK3: first 0.97, unlimited 2.00, van Leer 2.05, MC 1.99;
  TPFA linear profile exact, Pe=10 boundary layer (central + TPFA) max error 1.75e-3 ->
  4.57e-4 (order 1.94); simplex refusals. L4's MUSCL refusal assertion now asserts success.
- Not measured (no performance claim). Open: Scientia lowering of `div(-k grad u)` and of
  BJ/Venkatakrishnan intents (Finitum realizes both once programs carry them); limited-slope
  frozen-active-set derivatives; non-orthogonal diffusion correction; spatially varying κ.

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
  point). Batching the single-field `RealizationPlan::apply_cell` loop too (outputs
  replayed in point-major order) passed all 262 tests but changed no target time
  measurably (sv2_elasticity 3.9 -> 4.5, planar_p1 4.6 -> 4.3, load 28), so it was
  not landed. Not yet batched (open): single-field realization point loops, facet and
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

Condensed 2026-10-19 (full text in git history at `16dbdf3`).
- `ConnectionRealizationPlan::planar_p1(system, relation, meshes, facets, tolerance)` (beside
  `nested_p1`; Sinbad policy `InterfaceTransferPolicy::PlanarP1`, serialized `planar_p1`)
  admits two arbitrary planar triangulations of the same polygon.
- The constructor proves coverage (area floor `tolerance × diameter`; degenerate triangles,
  empty opposite-side and overlapping same-side intersections refused) and builds the mortar
  projection of the master P1 trace against the biorthogonal dual basis `ψ_i = 4λ_i − 1`,
  every integral exact over the intersection polygons.
- Refusals are `InvalidRealization` with a `CONNECTION_TRACE:` prefix, each asserted by
  message in `tests/planar_p1.rs` (8 tests: coverage defects ≤ 3.4e-16 relative, row sums,
  reproduction and transpose checks over the `tests/support/planar_meshes.rs` fixtures).
- Open: curved/moving interfaces, multiplier/Nitsche stability, the multi-relation diagonal,
  n-way junctions, transient nonmatching composition (Krasis); the recorded sliver-floor
  limitation from the 2026-10-09 review stands.

## W9 SC-W3 packages 1-2 (2026-09-29; details in git history)

- Package 1: `ConnectionRealizationPlan::system_linearized_diagonal` gives the reduced
  Jacobian diagonal `diag(Rᵀ J R)` of a `matching`/`nested_p1` connected pair (essential and
  connection-target rows are identity rows since the 2026-10-08 fix; interface-adjacent
  master rows gain the transpose-scattered cross term). Krasis must route weighted-map Jacobi
  through it for `nested_p1`.
- Package 2: proof-only (no production change) that Krasis `ConnectedSystemOperator` row
  kinds, residual and JVP are generic over `ConstraintSet`, so `nested_p1` transient coupling
  behaves as `matching`; two real coarse/fine `Conduction` operators over
  `two-material-conduction.res`. Package 3 landed in W10 as `planar_p1`.

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
- Cell-centred FV: `TensorBox` profiles, `FvTopology` (tensor/simplex, periodic tensor
  axes) and the first-order `FvRealization` (DaeOperator + ExplicitOperator, CFL,
  conservation receipt) over Scientia conservation-law programs (SV2-C2a).
- Reference 1-D mortar/nonmatching transfer, variable-order segment, clipped-segment,
  FV/FD/network/particle/boundary method programs and fixtures. These are not a
  claim of complete production execution for every method family.
- Source-aware realization, conservation, transfer and constraint-work verification
  reports. Callers must use the validating APIs, not merely rehash acceptance flags.
- CADabra provider identity/maps are consumed directly for bounded geometry paths; a
  committed extruded B-rep body (Provider V1) is realized as a tetrahedral mesh with node
  association and parameter JVP/VJP (V8).

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
