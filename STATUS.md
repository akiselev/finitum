# Finitum status

Updated: 2026-10-24

## Role and frontier

Finitum owns concrete meshes, spaces, DOFs, geometry, constraints, sampling,
transfer and global realization. Scientia owns scientific/form meaning, Malleus
owns local executable kernels, Krasis owns state/history, and Methodus owns
numerical algorithms. Finitum implements Methodus operator contracts directly;
it has no dependency on Solverang's constraint engine.

SHOW-3 / bounded SC-W3 additions are accepted. W13 A4 owner battery (2026-10-19, before the V8
rebase): 293 passed, 1 recorded failing (below) across 46 targets. The complete owner gate passed
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

## W13 A1 (2026-10-19): JIT execution backend (W11 F2)

Working clone, not yet integrated. Siblings: Malleus `7a43e7f` (`crates/malleus-cranelift`).
- `ExecutionBackend::{Interpreter, Cranelift}` (serde `interpreter`/`cranelift`). Cargo
  feature `cranelift` (off by default) adds the optional `malleus-cranelift` path dependency
  (Cranelift `=0.128.3` crates enter `Cargo.lock`); without it `Cranelift` is refused with
  `FinitumError::ExecutionBackendUnavailable` (code `EXECUTION_BACKEND_UNAVAILABLE`).
- Selection is captured when a kernel is prepared: `with_execution_backend(backend, || ..)`
  (thread scope, restored on unwind) else the process default `FINITUM_EXECUTION_BACKEND`
  (`interpreter` when unset; other values refuse). Every kernel-owning object (realization
  plans, system operators and bind compositions, interface operators, FV realizations,
  methods, samplers, functionals) therefore follows the scope it was built in.
- Under `Cranelift` each kernel is compiled once per process via one shared
  `malleus_cranelift::JitCache` (key: kernel digest, schedule, level, backend version, CPU
  features, batch ABI) and every call goes through the same positional `run_batch` (one native
  loop over face/cell/point batches; JVP/VJP kernels included). Bindings are checked by the core
  `check_bindings` on both paths and refusals re-run the interpreter, so errors are identical.
  Compositions run natively stage by stage with `PreparedComposition`'s copy-in/copy-out
  semantics when every stage compiled; a binding shape it does not accept is re-run by the
  interpreter for its error. A kernel the JIT refuses runs on the interpreter: per kernel,
  recorded as `KernelEngine::Fallback{reason}`.
- Receipts (`finitum-execution-receipt/1`): `execution_receipt()` on `RealizationPlan`,
  `SystemOperator`, `InterfaceOperator`, `FvRealization` -> `ExecutionReceipt{backend,
  backend_identity, native, interpreted, fallbacks, kernels[{name, kernel_digest, engine}]}`
  (distinct by digest). Not in any capability, receipt or digest: no serialized type, digest or
  existing public signature changed (pins in `first_order_identities_are_pinned`, capability
  tests, Sinbad identities unchanged; `w13_execution_backend` asserts digest and FV receipt
  bytes equal across backends).
- Differential proof (feature `cranelift`; bitwise, NaN-equivalent; errors equal as text;
  every distinct kernel native, zero fallbacks, same digests): `gx_f7_vjp`
  `cranelift_backend_is_bitwise_equal_on_fe_residual_jvp_vjp_and_assembly` (Poisson, hanging
  constraint, elasticity, nonlinear dynamic input: residual, JVP, VJP, linearized, assembled
  values/action, matrix-free); `w8_multi_instance_system` `..._on_the_system_path_and_bind_
  compositions` (monolithic + composed electrothermal, both quadratures, three states);
  `w7_sc_w1_interface` `..._on_facet_pair_kernels` and
  `cranelift_refusals_fall_back_per_kernel_and_are_recorded` (300-statement kernel over the
  JIT IR budget: interpreter, reason recorded, others native, same bits); `sv2c2a_fv` first
  order LLF/HLL/central 1-D + 2-D HLL with ±0, ±inf, NaN, subnormal, ±huge states;
  `sv2c2b_fv` MUSCL x {none, minmod, van Leer, MC, superbee} x {axis, LSQ, Green-Gauss},
  tensor + simplex, TPFA diffusion (Barth-Jespersen/Venkatakrishnan are refused by Scientia
  lowering, so no FV kernel exists for them). `w13_execution_backend`: selection, scoping,
  compile-once (second realization compiles nothing). Whole battery also green with the
  feature and `FINITUM_EXECUTION_BACKEND=cranelift` (43 targets, 288 tests; default
  build 43 targets, 282 tests).
- Finitum timing (`finished in`, s, feature build, same binaries, env toggled, 3 alternating
  reps, median; opt 2, thinkpadx1 12 threads, load 9-17 from other projects): interpreter /
  cranelift: w7_system_path_parity 7.50/6.02, gx_c4_facet_integrals 1.91/1.75, planar_p1
  1.44/1.06, sv2b4_system_stokes 1.55/1.07, w7_infsup 2.43/1.15, sv2_elasticity 1.64/0.35.
  Includes ~0.2-0.3 ms compile per distinct kernel; remaining time is Finitum gathers.
- Sinbad (read-only clone `0123f8d` built against this Finitum with `--features
  finitum/cranelift`, env toggled, same binaries): sv2c7_scalar_fv 51.6, 54.6 -> 30.2, 35.5 s
  (two reps, alternating order); benchmark_ladder 108.4 -> 95.7 s (one rep each);
  show2_component not measured to completion: > 280 s interpreter and > 585 s with Cranelift
  (two tests still running; beyond the lane's 10-minute command limit).
- Default off: the gain on consumer targets is 1.1-1.7x, while the feature adds 49 lockfile crates (16
  Cranelift) of native code generation and unsafe JIT code to every consumer build and needs Rust 1.93
  (`malleus-cranelift` rust-version; Finitum's own is 1.85). Consumers opt in.
- Open: SIMD across points, batching the remaining per-point loops, a disk cache; Sinbad
  opt-in and recording `ExecutionReceipt` in run evidence (Sinbad's change).

## W13 A4 (2026-10-19): conjugate heat transfer on FV (SV2C L9a, SV9 E5 stage 1)

Working clone on A1 `38556d3`; Scientia A2 `4c34a4c`. Not integrated or consumer-accepted.
- Per-face data: `FvRealization::with_face_fields(system, program, topology, regions, data,
  &BTreeMap<String, FieldSource>)` reads the named datums per face (`Constant`, `Nodal` = one
  value per face in block order, `Fallible` sampled once at face centroids, t = 0); `Table`/
  `Kernel` -> `FV_FACE_FIELD_UNSUPPORTED`; source-kernel datums -> `FV_FACE_FIELD_CELL_KERNEL`;
  `FV_FACE_FIELD_INVALID` (shape, non-finite, given twice). Bound as a stride-1 column in the
  face/ghost/prescribed/speed `run_batch`. Identity adds each field's value-bits digest.
- Regions (D5 gate 1b): `with_regions(FvRegions{cell_regions, properties: {tag ->
  FvRegionProperties{conductivity, capacity, stagnant}}})`: rate `C_i|Ω_i|u̇`, harmonic TPFA
  `T_f = 1/(d⁻/k⁻ + d⁺/k⁺)` (`k⁻/(δ·n)` at ghosts; prescribed fluxes total), invariants and CFL
  on `C|Ω|`; `stagnant` requires every face-flux datum (velocity) to be 0 next to the region
  (`FV_REGION_NOT_STAGNANT`). Refusals `FV_REGION_SHAPE/_UNKNOWN/_PROPERTY_INVALID`,
  `FV_DIFFUSION_INVALID` (second diffusion), `FV_TPFA_NONORTHOGONAL`.
  `region_interface_fluxes(t,u) -> FvRegionInterfaceReceipt` (`finitum-fv-region-interface/1`,
  Neumaier flow per region pair `from < to`). `cell_conductivity`, `is_prescribed_boundary`.
  Receipt adds skip-if-empty `face_fields`, `regions`; `new` digests/receipt bytes unchanged
  (`first_order_identities_are_pinned`, `w13_execution_backend` pass).
- `FaceFluxExchange` (`src/fv_exchange.rs`, `finitum-face-flux-exchange/1`), the Krasis A5 API:
  `new(&FvRealization, &RegionTagId, fe_mesh: &Mesh, fe_facets: &[FacetId],
  FaceFluxExchangeLaw::{DirichletNeumann | Robin{coefficient}}, tolerance)`; the FV tag must
  be a prescribed-flux boundary with zero flux (`FV_EXCHANGE_BOUNDARY`), each FV face matches
  one FE exterior facet (centroid, measure, opposite normal; `FV_EXCHANGE_NONMATCHING`),
  `FV_EXCHANGE_LAW`, `FV_EXCHANGE_SHAPE`. `q_f = g_f(T_P − T_s(x_f))`, `g_f = k_P/(δ·n)` or
  `h`, `T_s` = facet-vertex mean. `residual(fv, fe, fv_out, fe_out)` ADDS `+|f|q_f` to the
  owner cell and `−|f|q_f/n_v` to each facet vertex (FE state = P1 vertex values, the field
  block); `jacobian_vector_product` (= residual of the direction), `vector_jacobian_product`
  (exact transpose, adds), `face_fluxes`, `fe_trace`, `balance -> InterfaceBalance`
  (`finitum-interface-balance/1`, `balanced` iff `|Σfv + Σfe| ≤ 4·n_v·ε·Σ|f||q|`),
  `receipt()` (`enforcement: weak_flux`), `digest`, `fv_dimension`, `fe_dimension`, `faces`.
  FE sign: Finitum's `div(q)=0, q=−k grad T` residual `∫k∇T·∇φ`; verified by the coupled solve.
- Evidence `tests/w13_cht_fv.rs` (8) + `fixtures/fv-cht.res`: Graetz Pe = 100, 120x64, Nu over
  4 ≤ x ≤ 5: isothermal LLF upwind 7.549, MUSCL-LSQ 7.547 (bar 2% of 7.541); uniform flux 8.297
  (bar 2% of 8.235); Newton-GMRES steady, boundary flows cancel ≤ 1e-8, conservation receipts
  balanced; BDF1 (adaptive) to t = 400 within 4.6e-10 of steady; two-slab 1-D uniform and graded
  exact ≤ 1e-9 (320 K, 160 W/m²), capacity BDF storage balance; 2-D zero-velocity limit exact;
  conjugate Graetz with stagnant walls: fluid->wall flow = wall boundary outflow ≤ 1e-8, VJP
  transpose ≤ 64ε; refusals. `tests/w13_face_flux_exchange.rs` (4) + `fixtures/fe-solid.res`:
  FV slab + P1 solid exact (≤ 1e-9; trace 320, flux 160, imbalance exactly 0), Robin series
  resistance exact, JVP/VJP transpose, refusals.
- RECORDED FAILING: `muscl_isothermal_nusselt_error_decreases_under_refinement` (pre-registered
  "|Nu−7.541| decreases ny 16 -> 64"): 7.53882 / 7.54555 / 7.54727 (ny 16/32/64; 240x64
  7.54729): 2nd-order convergence to ≈ 7.548 (+0.09%; window not fully developed, Pe = 100).
- Not measured (no performance claim). Open: time-dependent face fields; Scientia lowering
  of capacity and `div(−k grad u)`; exchange on non-matching facets and P2; Sinbad Graetz
  cases and the FEniCSx `conjugate_graetz` oracle; Krasis composition (A5).

## W11 F1 (2026-10-19): prepared and batched Malleus execution

Condensed (full text at `90c4789`): every Malleus call runs prepared (`src/execution.rs`); the
system cell loop runs one `run_batch` per kernel; single-field, facet/interface and VJP loops
are not batched (open). Timing w7_system_path_parity 45.5 -> 16.1 s (opt 2, loaded host).

## Earlier landings (condensed 2026-10-19; full text at `38556d3`)

W10 package 3 `ConnectionRealizationPlan::planar_p1` (two planar triangulations of one polygon,
proved coverage, biorthogonal dual-basis mortar rows; `tests/planar_p1.rs`); W9 SC-W3
`system_linearized_diagonal` (reduced `diag(RᵀJR)` for `matching`/`nested_p1`) and the
transient `nested_p1` proofs (`tests/nested_p1_transient.rs`); `system_constraints`,
`SystemOperator::linearized_diagonal`, `SurfaceTransfer::p1`. Accepted gates: SC-W2 matching
interface, SHOW-1 portable meshes, M-CPU-S CSR assembly (`docs/validation/2026-09-17-*`).
Open there: curved/moving interfaces, multiplier/Nitsche stability, n-way junctions, the
recorded sliver-floor limitation.

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
  axes) and `FvRealization` (DaeOperator + ExplicitOperator, CFL, conservation receipt,
  MUSCL, TPFA, per-face data, multi-region CHT) over Scientia conservation-law programs, and
  the FV <-> P1 `FaceFluxExchange`.
- Reference 1-D mortar/nonmatching transfer and FV/FD/network/particle/boundary method
  programs (not a claim of production execution for every method family).
- Source-aware realization, conservation, transfer and constraint-work verification
  reports. Callers must use the validating APIs, not merely rehash acceptance flags.
- CADabra provider identity/maps are consumed directly for bounded geometry paths; a
  committed extruded B-rep body (Provider V1) is realized as a tetrahedral mesh with node
  association and parameter JVP/VJP (V8).

## Limits and next work

- General hp/AMR topology, geometric hanging-node construction, curved high-order
  realization, moving/remeshed interfaces and production embedded-domain semantics
  remain open. Reference segment tables are not multidimensional hp support.
- Connected product admission: matching scalar P1, steady/transient `nested_p1` and
  `planar_p1` (proved diagonal and transient checks above); FV <-> P1 face-flux exchange
  (W13 A4). Curved/moving nonmatching coupling and time stepping of transient connections
  (Krasis) are not built here.
- Full-coordinate affine constraint rows are generally nonsymmetric even when
  the free-coordinate `P^T A P` block is symmetric; symmetry declarations and solver
  refusals must retain that distinction.
- Runtime point linearization must be distinguished from constructors that freeze
  at zero state/rate. No universal solver/preconditioner or speedup is claimed.
- General SIMD/GPU execution, material/scientific parsing, coupled transactions,
  solver selection and support promotion belong outside this repository.

Detailed history is in Git; recorded gate artifacts remain in `docs/validation/`.

Final cross-repository evidence: [September 18 acceptance](../sinbad/docs/validation/2026-09-18-assembly/README.md).
