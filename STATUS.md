# Finitum status

Updated: 2026-09-03
Milestone: SV0-B3 checks + R3D/SV1-G0B geometry derivatives + SV2-A vector H1 elasticity +
SV2-B1/B4 P2 elements and mixed product layouts + E6 executable system realization
(Scientia-form-driven SystemOperator with load vector, equation-sign symmetry proof, and
H(div)/RT0 + P0 compatible realization — the real Stokes and mixed-Darcy corpus systems solve)
+ W7/E7 SV1-C1/C3 global transpose operators and distributed-coefficient VJPs
+ W7/P state-dependent `SystemOperator` residual/JVP/VJP with GX-A3 chain-rule tangents
+ W7/package 3 runtime inf-sup checker for realized mixed pairs
+ W7/SC-W1 (Finitum) system-level ids keying `BlockLayout` and public per-block actions/transposes
+ W7/SC-W1 (Finitum) / SV2-B2 interface-measure realization binding Malleus facet-pair kernels
+ W7 follow-ups: C11.8 degree-2 P1 quadrature (opt-in) and C11.22 RT0 essential normal-trace data

## Implemented

- validated 1D--3D simplex meshes with finite coordinates, bounded connectivity, and distinct
  vertices per cell;
- deterministic global degree-of-freedom maps with nonempty, bounded, duplicate-free element
  restrictions;
- explicit affine constraints with unique targets/dependencies, finite coefficients, bounds and
  cycle validation, exact input extents, and finite expansion results;
- prepared quadrature/basis tables with checked extents, shape validation, and finite data;
- P1 simplex reference basis and barycenter quadrature preparation;
- digest validation across Scientia `FormRequirements`/`OperatorFactorization` and complete
  Malleus primal/JVP/VJP/parameter bundles;
- concrete affine geometry preprocessing and quadrature-point external-input packing;
- deterministic gather, value/gradient basis forward action, generated primal/JVP execution,
  quadrature weighting, basis transpose, and scatter;
- affine constraint prolongation, transpose restriction, lifting, and explicit constraint rows for
  both residual and directional actions, including nested hanging-node dependencies;
- one `RealizationPlan` producing matrix-free and canonical CSR assembled operators through the
  same generated JVP execution; both implement Methodus `LinearOperator` directly.
- independent runtime state and state-rate basis bindings for generated primal residuals;
- generated JVP evaluation at the actual linearization point, with independent state/rate
  directions;
- dynamic quadrature-point external inputs and chain-rule composition through generated
  parameter-JVP kernels;
- fixed essential residual rows and their consistent state-direction JVP rows.
- deterministic concrete-plan identity covering the artifact chain, mesh, element, DOF map,
  constraints, stored external values, and caller-declared dynamic-input identities.
- a complete serializable `RealizationArtifact` projection for product inspection and cache
  records, including source artifact digests, mesh, element tables, DOF map, constraints, stored
  values, and dynamic-input identities; it does not deserialize or reconstruct a realization,
  and generated executables remain absent.
- deterministic primal realization of an admitted CADabra affine rectangle, binding provider
  revision and semantic digest, stable design-parameter coordinates, node/chart identities,
  simplex/region identities, and boundary-to-vertex associations into one realization digest,
  retained with the executable operator by `CadPrimalPlan`;
- stable CAD-boundary-selected essential constraints with typed stale-revision, missing-boundary,
  source-mismatch, duplicate-selection, conflicting-corner, and forged-plan refusals;
- product-space block layouts with explicit field/entity/component ownership and deterministic
  gather/scatter;
- deterministic simplex exterior/interior facet topology, explicit reversible minus/plus
  interface ordering, and cell-to-facet orientation signs;
- affine H(curl) covariant and H(div) contravariant Piola maps, oriented edge/facet DOF
  restrictions, and triangle/tetrahedron incidence complexes that verify curl-grad and div-curl
  are exactly zero;
- element-local Schur condensation with a retained trace system and full interior recovery map;
- `SystemRealizationPlan`, which validates Scientia block coordinates and the complete
  form/factorization/Malleus receipt chain before digest-binding it to a mesh, block layout,
  facets, compatible maps, and exact-sequence evidence.
- one-dimensional nonmatching Lagrange transfer and mortar-like common-trace interpolation with
  weighted conservative transpose scatter;
- standalone per-cell variable-order segment basis tables with Gauss-Legendre quadrature, plus an
  explicit algebraic midpoint-constraint constructor;
- quadrature-point partial assembly preserving `E^T B^T D B E`, separate dense element assembly,
  fixed-width cell batches, component/lane accelerator packing, and tensor-product
  sum-factorized value/gradient evaluation; and
- an explicit level-set identity and quadrature policy for linearly clipped segment cells.
- digest-bound finite-volume, finite-difference, network DAE, particle, and boundary-integral
  realizations consuming Scientia `MethodProgram` directly, with typed state-extent checks;
- deterministic DAE residual/JVP actions for every method family, compiled Malleus flux/stencil
  execution where present, and a `DiscreteOperator` enum that preserves variational-versus-sibling
  family identity for Krasis coupling;
- reusable nodal patch and four-way matrix-free/global-assembled/element-assembled/partial
  agreement providers using Methodus componentwise comparisons;
- canonical CSR global transpose realization and a generic forward/transpose work check;
- homogeneous affine-constraint work, weighted nonmatching-transfer conservation,
  dimension-complete exact-sequence boundary/rank, and maximum-cell-diameter mesh-refinement
  order providers;
- versioned, kind-distinct serialized reports whose canonical digest binds subject identity,
  tolerance/policy, probes or refinement samples, measured outputs, and acceptance results;
- SV2-A production slice: vector H1(order=1) blocks execute end-to-end through the same
  generated kernels — vertex-major component DOF maps (`vector_nodal_dof_map`), value,
  gradient, and symmetric-gradient basis evaluations with component-strided state, flux and
  vector-value adjoint scatters, and isotropic constitutive laws wired as dynamic external
  inputs so the tangent flows through the generated parameter kernels (`d sigma = C : d eps`).
- deterministic triangular realization of admitted CADabra planar annuli on the
  wrapped polar chart (`CadGeometryRealization::from_family`), with stable
  inner/outer boundary identity, positively oriented cells, and a family-scoped
  association digest; other Provider V0 families are refused by a typed
  unsupported-family error rather than silently approximated;
- exact per-node design velocities `dx/dp` for rectangle and analytic-family
  realizations, taken from the provider's analytic first design differential at
  each node's frozen chart coordinate and refused on stale revisions or source
  mismatch;
- exact residual geometry sensitivity `dR/dp_k` at a fixed expanded state:
  affine cell-map derivatives (determinant trace identity, inverse-Jacobian
  product rule), basis-gradient direction threading through the generated
  state-JVP kernels, authored stored-external direction tables, and the full
  product rule through measure, test-basis gradients, and kernel outputs.
  Missing, extra, or extent-mismatched external direction tables are refused;
  dynamic callbacks are refused because they cannot declare an exact design
  derivative. Reflected cells are refused.
- GX-C1/C2/C3/C4/C5/C6 and GX-F7 landed (`6e6c4a4`, `89eea19`, `0d378a0`) but were never folded
  into this list by their landing commits: `MeshProfile`/`RegionTags`/`FieldSource`, exterior
  facet integrals, an executed `Symmetric` proof, and executed VJP kernels. This entry only
  records that they exist; their exact behavior is authoritative in
  `sinbad/docs/simulation-vision/GX-CONTRACTS.md` C11.5-C11.7, not re-audited here.
- P2 simplex elements (`PreparedElement::quadratic_simplex`, 1-3-D; vertex-then-edge-major DOF
  maps `quadratic_simplex_dof_map`/`quadratic_simplex_node_points`) alongside P1, sharing every
  existing generated-kernel execution path in `RealizationPlan` unchanged (SV2-B1): basis
  evaluation, gather/scatter, JVP, and assembly are element-basis-count-generic already, so the
  only `RealizationPlan` change was admitting `polynomial_order == 2` (with the matching P2 basis
  count) in `validate_discretization`, and refusing a P2 exterior-facet integral typed (the
  facet trace basis stays hardcoded P1).
- an executable product-space layout (`mixed::MixedSpace`) of per-field blocks with independent
  polynomial order (1 or 2) and component count (scalar or dimension-vector), each with its own
  DOF map, laid out end-to-end by the existing `BlockLayout` (SV2-B1); and a block/coupling
  operator composition (`mixed::MixedOperator`, SV2-B4 start) applying a diagonal
  `GradientGradient` block and an off-diagonal `DivergenceValue` coupling (contributed together
  with its exact transpose, so the composed action is symmetric by construction) into one
  monolithic matrix-free action, implementing Methodus `LinearOperator` and `BlockLinearOperator`
  directly. This is generic structural machinery bypassing Scientia forms and Malleus kernels
  entirely (its own quadrature/basis evaluation, shared across differently-ordered fields); it
  names no physics.
- a typed, representation-only block-nullspace declaration (`mixed::BlockNullspaceCandidate`,
  `NullspaceModeKind::Constant`, mirroring Scientia's structural `NullspaceCandidate::Constant`
  from GX-CONTRACTS C5.4) that resolves against a `BlockLayout` into the exact unit-norm
  constant-mode vector and a `methodus::ConstantModeProjector` a downstream MINRES-family solver
  (SV2-B6) would consume; no solver algorithm is implemented here.
- SV2-B4 continuation: essential-constraint/Dirichlet elimination for `MixedOperator`
  (`BlockEssentialValue`/`essential_constraints_for_blocks`, block-local declarations lifted into
  a global `ConstraintSet` via `BlockLayout` offsets, mirroring `essential_constraints_from`'s
  DOF-indexing convention without the `TaggedMesh`/`RegionMap` wiring it has) and
  `MixedOperator::apply_reduced_action`/`reduced` (`ReducedMixedOperator`), which mirror
  `RealizationPlan::apply_direction`'s identity-row/zero-column treatment exactly. On the
  existing vector-P2/scalar-P1 saddle-point fixture with Dirichlet-constrained boundary `field_a`
  DOFs, `BlockNullspaceCandidate::verify_in_kernel` now passes against the reduced operator (the
  unconstrained-operator demonstration is unchanged and still documents a true property), a
  `methodus::solve_minres` solve with the resolved `ConstantModeProjector` converges and matches
  an independently-derived dense reduced reference, and `ReducedMixedOperator` declares
  `Symmetric` by an analytic argument (proved in its doc comment, not by assembly) whenever its
  constraints carry no affine dependency. `MixedOperator::digest()` is now a content-addressed
  identity over space and couplings (mirroring `RealizationPlan::digest()`), and a P2 vector
  test / P2 *scalar* trial `DivergenceValue` coupling (SV2-B1's deferred order-generic exercise)
  is now covered against an independently-derived reference.

- E6 executable system realization (`739e2aa`): `SystemRealizationPlan::bind_kernels` →
  `SystemOperator`/`ReducedSystemOperator` — real per-`(row, column)` Malleus kernel binding and a
  monolithic multi-block matrix-free action over `BlockLayout` (single-field kernel-execution code
  shared, not duplicated); region-tag multi-field Dirichlet constraints with P2 edge-node
  resolution; Scientia `OperatorStructure` (C5.4) threaded into declared symmetry/properties with
  a realized-coordinate cross-check and auto-derived nullspace candidates; typed `equation_sign`
  (±1, solution-preserving) with assembly-based `SystemOperator::prove_symmetry`; the real
  `25-stokes.res` system agrees entrywise (5e-11) with the independently hand-composed
  `mixed::MixedOperator` and solves under MINRES with the auto-derived pressure projector.
- E6 system load vector (`5da4744`): `SystemOperator::load_vector` (primal kernels at zero active
  state; zero source → exact zero; constant source verified against the Lagrange
  partition-of-unity closed form) and `ReducedSystemOperator::load_vector` (the elimination
  composition) — nontrivial forced solves reachable, cross-checked against an independent dense
  Gaussian-elimination solve.
- E6 H(div)/RT0 + P0 compatible realization (`1af8946`): `rt0_reference_basis` (omitted-vertex
  facet convention shared with `CompatibleDofMaps::hdiv`), `cell_constant_dof_map`,
  `FieldKind::{Lagrange, Hdiv0}` dispatch with per-cell signed contravariant Piola pullback
  (gather/scatter proven exact algebraic transposes by a dot-product identity test), and
  `bind_kernels_with_facets` executing RT0's closed-form exterior-facet normal trace — the real
  `13-mixed-darcy.res` RT0-P0 system solves (MINRES, 79 iterations, 1e-6 against an independent
  dense reference). Load-bearing finding: the corpus's data-independent impermeable boundary term
  makes the discrete system genuinely full rank, so the structural constant-pressure nullspace
  candidate correctly does not verify against the realized operator.

- SV1-C1/C3 (W7/E7): the global transpose as a Methodus operator pair and distributed-
  coefficient derivative products, all executing the bound Malleus VJP/parameter kernels
  point-locally (no assembly):
  - `RealizationPlan::linearize(time, state, rate, rate_shift)` -> `LinearizedOperator`, the
    Jacobian `dR/du + rate_shift * dR/du_t` at a fixed linearization point implementing
    `methodus::LinearOperator` (JVP with rate direction `rate_shift * x`) and
    `methodus::TransposableOperator` (`vector_jacobian_product_shifted`, the new rate-shifted
    generalization of GX-F7's VJP: a `TimeDerivative` active input's cotangent scatters through
    the value basis scaled by the shift, dynamic-input chain rules through `dt(u)` are probed
    the same way). `rate_shift = 0` is byte-identical to `vector_jacobian_product`.
  - `MatrixFreeOperator` and `AssembledOperator` implement `TransposableOperator` (VJP at zero
    state; CSR transposed traversal), so `methodus::TransposeOperator::explicit` and the
    adjoint solve (SV1-D1) consume every realized operator without a symmetry declaration.
  - `DistributedCoefficient { integral_index, input, layout: CoefficientLayout::{Vertex, Cell,
    QuadraturePoint} }` views one stored external input of a cell integral as a caller-owned
    design vector; `ExternalInput::from_coefficient` builds the stored table from that vector,
    `coefficient_jacobian_vector_product` is `dR/dp * d` (parameter kernel with the direction
    routed to that input only) and `coefficient_vector_jacobian_product` its exact transpose
    accumulated through the layout's interpolation transpose. Constraint rows carry zero
    coefficient derivative. Capability reports `DerivativeProduct::{CoefficientJvp,
    CoefficientVjp}` exactly when a stored cell input exists (the VJP under the affine-
    dependency rule GX-F7 already applies).
  - Evidence (`tests/sv1_c1_transpose.rs`, 7 tests): adjoint identities to `1e-12` relative on
    Poisson (nodal `k`) and on a transient nonlinear fixture at a nonzero state with rate
    shifts 0 and 3.7 (the shifted forward action also matches centered differences of the
    residual); matrix-free/assembled/materialized transposes agree to `1e-12`; coefficient
    JVP/VJP transposes for all three layouts to `1e-12`; coefficient JVP against centered
    differences of rebuilt realizations; and the adjoint objective gradient `dJ/dk =
    -lambda^T dR/dk` (adjoint solved by GMRES through the explicit transpose) against centered
    differences of the fully rebuilt solve with tightening error over two step sizes.
  - Not landed: transposes of affine dependency constraints, condensation, and transfer
    (SV1-C2, refused typed as before); coefficient products on facet integrals and for dynamic
    bindings (refused typed); mesh-coordinate JVP/VJP (SV1-C4).

- Batch P (W7; ARCHITECTURE.md §12 P item 2): the system realization is state-dependent.
  `SystemOperator::residual(t, u, u_t)`, `jacobian_vector_product(t, u, u_t, du, du_t)`,
  `vector_jacobian_product[_shifted]`, and `linearize` evaluate every block's bound PRIMAL/JVP/
  VJP kernels at the actual linearization point (basis inputs gathered from the state, or from
  the rate for `TimeDerivative` inputs; constitutive closures see the point's actual active
  values and time), with the chain rule through every closure's exact `direction` composed by
  the generated parameter-JVP kernel (forward) and inverted by unit-perturbation probing
  (transpose), so cross-field property tangents (`ka = ka(b)` inside the `a` equation) land in
  the off-diagonal blocks without any expression rewriting. The `LinearOperator` view
  (`apply_action`) and `load_vector` are now defined as the JVP and `-R` at the zero point and
  are numerically unchanged (E6 Stokes/Darcy tests pass as before). `ReducedSystemOperator`
  gains the essential-constraint-eliminated `residual`/`jacobian_vector_product`/`vector_
  jacobian_product[_shifted]`/`linearize` (constraint rows mirror `RealizationPlan` row for
  row) and implements Methodus `DaeOperator`, `NonlinearOperator` (steady view at `t = 0`,
  `u_t = 0`), and `TransposableOperator`; `SystemOperator` implements `TransposableOperator`;
  `LinearizedSystemOperator` (physical or reduced) implements `LinearOperator +
  TransposableOperator + BlockLinearOperator`. `jacobian_properties` claims the zero-point
  properties only for a structurally linear, non-transient system; otherwise `Unknown` plus
  the block partition. `BlockNonlinearOperator` is deliberately not implemented on
  `ReducedSystemOperator` (its `block_layout` would clash with `BlockLinearOperator`'s at every
  call site); Krasis's `CoupledSystemOperator` owns that view.
  `system_constitutive_from_sources(system, model, sources)` is the GX-A3 resolution for the
  system path, mirroring `external_inputs_from` rule for rule (kernel/table with exactly one
  active-field input -> exact tangent/slope closure, coordinate-only/constant/sampled ->
  zero direction, nodal refused, missing tangent -> `RealizationTangentUnavailable`), looking
  the field's value up by its active input's `TensorInputId` (`PointEvaluation::input_values`)
  so several fields sharing an evaluation kind stay distinct.
  Evidence (`tests/w7_p_system_state_dependent.rs`, 7 tests, on a hermetic two-field
  transient nonlinear system with cross-field property tangents, a state-dependent capacity,
  and a product term): centered residual differences match the JVP; JVP/VJP exact transposes
  to `1e-12` at a nonzero state for rate shifts 0 and 2.5 (physical and reduced); the zero-point
  view and load vector equal the stateful actions at zero bit-for-bit; Methodus `verify_dae_jvp`
  passes on the reduced `DaeOperator`; three Methodus BDF1 steps advance the reduced transient
  system with Dirichlet rows held; kernel-sourced properties reproduce the closure operator's
  residual/JVP/VJP to `1e-12`; a kernel without a tangent is refused typed.
  Not landed: exterior-facet integrals with active inputs in the system path (still refused at
  bind time), interior-facet/interface measures (SC-W1 item below), affine-dependency
  transposes (SV1-C2).

- W7 package 3: runtime inf-sup (LBB) checker for realized mixed pairs (`src/infsup.rs`;
  SV2-B4 evidence for `@inf_sup` obligations). `estimate_inf_sup(operator, layout,
  constraints, pairing, multiplier_norm, config)` probes the linear action densely, removes
  essential-constrained DOFs, and computes `beta_h = sqrt(lambda_min(M_Q^{-1} B A^{-1} B^T))`
  -- the inf-sup constant in the constrained block's energy norm and a caller-chosen multiplier
  norm (`InfSupNorm::{Euclidean, Gram}`; `SystemOperator::mass_matrix(field)` gives the L2
  Gram of a P0/P1/P2 field) -- by dense Cholesky plus a cyclic Jacobi sweep (capped at
  `INF_SUP_DIMENSION_CAP = 2048`). Kernel modes beyond the caller-declared legitimate kernel
  (`declared_kernel_dimension`: one constant pressure under pure Dirichlet velocity, zero for
  a full-rank pairing) are spurious multiplier modes and give a deterministic
  `InfSupVerdict::Unstable(SpuriousModes)`; `require_inf_sup_stable` turns that into the typed
  `FinitumError::InfSupUnstable` (`INF_SUP_UNSTABLE`). `InfSupPairing::from_structure` derives
  (constrained, multiplier) from Scientia's `OperatorStructure` (the one field without a
  diagonal block and the one diagonal-bearing field it couples to), never from a field name.
  Refused typed: affine-dependency constraints, a non-SPD constrained block, a nonzero
  multiplier diagonal (stabilized pairs are not judged by this constraint-only estimate), a
  Gram of the wrong extent, RT0 mass matrices. The estimate uses only the multiplier-row
  coupling block, so it is invariant under the `equation_sign` gauge (verified).
  Evidence (`tests/w7_infsup.rs`, 5 tests, on corpus snapshots under `tests/fixtures/corpus/`):
  Taylor-Hood Stokes (P2-P1) is stable with a mesh-robust constant `0.2171 / 0.2182 / 0.2180`
  at 2x2 / 4x4 / 6x6 (L2 pressure norm, viscosity 1.7, kernel exactly the declared constant
  mode); RT0-P0 Darcy on the 2x2x2 cube is stable with an empty kernel (the E6 full-rank
  finding) in both norms; the same Stokes model with `H1(order=1)` velocity (P1-P1,
  `tests/fixtures/w7_infsup/25-stokes-p1p1.res`) is refused on every mesh -- finding: on the
  structured diagonal triangulation with wall-fixed velocity the P1-P1 pressure kernel has
  dimension 8 (seven spurious modes) at 4x4, 6x6, and 8x8 alike, so the refusal does not rest on
  the 2x2 count deficit alone; unsigned/signed/reduced operators give the identical
  digest-identified record; a non-saddle structure, a same-field pairing, a wrong-extent Gram,
  an affine constraint, and a stabilized P1-P1 `MixedOperator` are refused typed.
  Recorded need (Scientia): `VerificationObligationKind::InfSup { pair }` carries a display
  string only; a typed `{ pair, constrained: SymbolId, multiplier: SymbolId }` would let a case
  bind the pairing without the structural re-derivation. Not landed: a refinement-sequence
  trend verdict (the per-mesh record is the unit; sequences are a Sinbad campaign concern),
  the H(div)-norm variant for RT0 (the energy norm here is the realized mass block).
  Fixture note: `tests/sv2b4_system_stokes.rs` now compiles the corpus *snapshots* in
  `tests/fixtures/corpus/` instead of the live `sinbad/physics/corpus` files, because the live
  `13-mixed-darcy.res` was mid-edit on 2026-09-03 (its `impermeable` block replaced by the
  natural closure, removing the exterior-facet integral the E6 RT0 facet path exercises).

- SC-W1 Finitum side, items 1-2 (`sinbad/ARCHITECTURE.md` §2.3/§2.4/§8; W7 package 4):
  - `src/system_ids.rs`: `InstanceId`/`SysVarId`/`SysResId` (`u32` newtypes, the wire width
    §2.4 fixes) and `SystemIdMap`, Finitum's own origin table (instances with their
    `(model, semantic digest, artifact digest)` receipt, every variable's `(instance, local
    SymbolId)`, every residual's `(instance, equation, row symbol)`, display paths
    `right.eb` / `right/<symbol>`, content-addressed identity `finitum-system-ids/1`).
    `SystemIdMap::one_instance` is the degenerate identity map of §2.6 (`SysVarId(symbol.0)`,
    `SysResId(block index)`, root unprefixed) -- which is exactly why every single-model
    realization and Krasis's `SemanticId::new(block.symbol.0)` read stay numerically
    unchanged; `SystemIdMap::compose(&[(name, &OperatorSystem)])` allocates dense ids in
    instance-then-local order (§2.3) for a multi-instance group. Deviation recorded: Scientia's
    `scientia-system/1` `OriginMap` is not committed yet, so these are Finitum newtypes with an
    explicit mapping; the exact Scientia surface `compose` replaces is recorded in its doc
    comment (`OriginMap.variables` / `OriginMap.residuals` / `OperatorSystem/2.instances`).
  - `BlockLayout` is keyed by `SysVarId`: `FieldBlock` gains `variable: SysVarId` next to the
    per-model `symbol` (kept, so Krasis's consumer compiles unchanged); `BlockLayout::new` is
    the identity keying, `BlockLayout::new_keyed((SysVarId, SymbolId, entities, components))`
    the composed one, where a symbol shared by two instances is deliberately *not* addressable
    by `block(symbol)` (only `block_by_variable`/`values_by_variable`/`variables`).
    `SystemRealizationPlan::new` builds and checks the one-instance map against the layout
    (`plan.system_ids()`, `operator.system_ids()`).
  - Public per-`(row: SysResId, column: SysVarId)` block actions on `SystemOperator`:
    `block_jacobian_vector_product` / `block_vector_jacobian_product` (rate-shifted, at an
    actual `(t, u, u_t)`; the row equation's bound JVP/VJP kernels with the direction/adjoint
    masked to one block, so a cross-field chain-rule tangent is exactly the off-diagonal
    block), the zero-point `block_action`/`block_transpose_action`, and
    `block_operator(...) -> SystemBlockOperator` (rectangular Methodus `LinearOperator +
    TransposableOperator`) for Krasis/Methodus block compositions.
  - Evidence (`tests/w7_sc_w1_block_actions.rs`, 4 tests, on the W7/P two-field nonlinear
    fixture): the one-instance map is the identity and keys the plan's layout; two instances
    compose to dense ids `0..4` with a symbol-ambiguous keyed layout and typed refusals
    (duplicate variable, duplicate instance name, empty composition); every block JVP equals
    the masked-direction slice of the full JVP (1e-13), the blocks of a row sum to the row,
    every block transpose is exact to 1e-12, the off-diagonal blocks are nonzero at a nonzero
    state and vanish at zero; `SystemBlockOperator` satisfies the adjoint identity and refuses
    wrong shapes/ids typed.
  - Not landed: re-keying `SystemOperator`'s *internal* field tables (still per-model
    `SymbolId`, valid for one instance); a multi-instance `SystemRealizationPlan` (two instances
    of one model in one group) waits on Scientia's `OperatorSystem/2`; `RegionMap`/
    `SystemEssentialConstraintRequirement` stay `RegionId`/`SymbolId`-keyed.

- SC-W1 Finitum side, item 3 / SV2-B2 (`src/interface.rs`): interface and interior-facet
  measure realization over shared facets, binding Malleus **facet-pair kernels** through their
  roles (Malleus `9862a08`: `FacetPairKernel`, `FacetOperandRole::{Cell{side, partner},
  Facet{parity}}`, `FACET_NORMAL_CONVENTION = "minus_to_plus"`).
  - `InterfaceSpace` (fields as `TraceFieldSpec { symbol, order 0|1|2, components,
    continuous }` over a `BlockLayout` -- its own via `new`, or an existing `SystemOperator`
    layout via `over_layout` so cell and interface actions add on one vector; cell-owned DG P0/P1
    DOF maps are new, continuous P1/P2 reuse the nodal maps), `InterfaceMeasure` (`interior`:
    every interior facet with the canonical first incidence as minus; `between(minus_cells)`:
    the interface of a cell set, oriented outward from it; `from_pairs`; `flipped`),
    `InterfaceKernel` (a validated Malleus facet-pair kernel plus one `InterfaceOperand` per
    operand: `Trace{field, side, Value|Gradient}` must carry `Cell{side}` and be readable,
    `Normal` must be `Facet{Odd}`, `Coordinates`/`FacetMeasure`/`Constant` must be
    `Facet{Even}`, `Residual{field, side, evaluation}` must be `Cell{side}` and writable, and
    shapes are checked against the field's components and the mesh dimension), and
    `InterfaceOperator` (residual / JVP / VJP at a state, the zero-state `apply_action`,
    CSR `assemble`, Methodus `LinearOperator + TransposableOperator`, content-addressed
    `digest`). Facet quadrature is 3-point Gauss on segments and the degree-4 rule on
    triangles, mapped into both cells' reference coordinates through the exact affine inverse
    (`AffineMap::reference_point`, new); the normal is the minus cell's outward normal, i.e.
    Malleus's minus-to-plus convention, and the plus cell is checked to lie on its far side.
    JVP/VJP execute `differentiate_facet_pair` products over every `Trace` input and every
    `Residual` output, so the pair is an exact transpose by construction.
  - `InterfaceOperator::swap_symmetry_receipt(state, tolerance)` executes Malleus's own
    `check_facet_swap_symmetry` at every facet with the *realized* traces, certifying that the
    global action does not depend on Finitum's minus/plus choice; a kernel with a partnerless
    (one-sided) cell operand is refused typed by the receipt, since Malleus cannot exchange
    its sides. Trace classes recorded here (Malleus deliberately records none): `Value` and
    `Gradient` traces of Lagrange fields on affine simplex facets; `Normal` and `Tangential`
    (Piola) trace mappings and `FacetL2` (a facet-native unknown) are not realized.
  - Evidence (`tests/w7_sc_w1_interface.rs`, 5 tests, hand-written Malleus kernels): a P0
    jump-penalty kernel over every interior facet of a 3x3 square assembles to the hand-built
    facet-length-weighted graph Laplacian (1e-13), annihilates constants, is symmetric, JVP/VJP
    transpose to 1e-12, the flipped measure realizes the bit-identical operator, and the swap
    receipt passes; a `Facet{Odd}`-normal central-flux kernel matches its closed form and is
    flip invariant while a one-sided variant fails the receipt and changes under the flip (the
    receipt detects exactly the orientation-dependent kernels); a two-field P1 coupling across
    the `x = 0.5` interface of a square (`between`) assembles to `[M -M; -M M]` with the segment
    mass matrices on the shared nodes and vanishes for equal constants, and its one-sided
    operands make the receipt refuse typed; a gradient-average consistency kernel on a
    discontinuous P1 field reproduces `-/+ n_x |F| / 2` for `u = x`; every binding refusal
    (side, parity, access, count, no residual, shape, unknown field, exterior facet, repeated
    facet, unsupported orders, `over_layout` extent) is typed.
- W7 follow-ups from the Krasis and Sinbad lanes:
  - C11.8 (P1 mass-matrix rank deficiency): `PreparedElement::linear_simplex_with_degree(d,
    degree)` tabulates P1 on the smallest rule exact for `degree` -- `0 | 1` the barycenter
    rule, `2` two-point Gauss / the three edge midpoints / the symmetric 4-point tetrahedron
    rule (exact for the P1 mass matrix; degrees above 2 refused typed).
    `PreparedElement::linear_simplex` deliberately keeps the barycenter rule: every authored
    per-quadrature-point external table (R3D geometry-sensitivity directions,
    `tests/cad_derivatives.rs`, Sinbad's derivative campaigns) is sized one point per cell,
    and changing the default silently changed that wire size (four `cad_derivatives` tests
    failed on `expected 36 values, got 12`), so the exact rule is opt-in per element. The
    Scientia-system path (`SystemRealizationPlan`, what Krasis's `CoupledLeaf::reduced_system`
    wraps) already integrates with the degree-4 triangle / degree-2 tetrahedron rules and has
    no rank deficiency. Evidence: the `p1_mass_tests` unit test (reference mass
    `|K|(1 + delta_ij)/((d+1)(d+2))` to 1e-15 in 1-3D; the barycenter rule's rank-one
    `|K|/(d+1)^2` recorded; degree 3 refused).
  - C11.22 (Sinbad lane need): `essential_constraints_from_system` admits an RT0
    (`Hdiv(order=0)`) field: a `flux . n = g` datum (`Constant`/`Sampled` scalar at the facet
    centroid) on a tagged exterior facet fixes the facet DOF to `orientation * g * |F| *
    (d - 1)!` -- the RT0 reference basis carries flux `1/(d-1)!` through its own facet
    (`1` on triangles, `1/2` on tetrahedra; `rt0_reference_basis`'s doc corrected, it claimed
    `1` generally). Interior facets, nodal sources, and vector data are refused typed.
    Evidence (`tests/w7_rt0_essential.rs`, 2 tests, on the Darcy snapshot): constraining every
    wall flux to zero puts the constant pressure mode into the reduced operator's kernel
    (the impermeable-box statement; the naturally closed system is full rank) and MINRES
    with the projector still solves; a uniform outward `g` lifts to a flux whose total
    divergence through the `mass_balance` block action is exactly `-g * 6` (divergence
    theorem; this is what caught the `(d-1)!` factor).
  - Not landed: `SystemRealizationPlan::bind_kernels` still refuses `SemanticMeasure::
    {InteriorFacet, Interface}` -- Scientia's factorization does emit `MinusTrace`/`PlusTrace`
    inputs for them (`jump`/`average`/`trace_minus`/`trace_plus`), so the bridge is to wrap
    each such bound kernel as a `FacetPairKernel` with roles derived from the input sites and
    hand it to this module; no corpus model or `.res` equation authors an interior-facet term
    yet (only `form` fixtures do), so the bridge waits for a driving case (SC-W2's CHT
    `ConnectionRealizationPlan`). Time-derivative traces, curved facets, and dimension 1 are
    refused typed.

## Boundary

Scientia owns the abstract space and form meaning. Malleus owns executable local kernels.
Finitum owns their concrete mesh/space binding and global realization. Krasis owns coupled
state and Methodus owns numerical algorithms.

The manifest now depends directly on Scientia, Malleus, Methodus, and CADabra's provider crate
because realization plans consume their concrete artifacts and the R3P path consumes provider
identity/maps directly. The FC6/FC7 executable action remains scalar
H1(order=1) cell integration. FC8 adds deterministic mixed, facet, Piola, exact-sequence, and
condensation reference contracts. FC9 adds reference transfer, affine-constraint, partial,
batched, packed, sum-factorized, variable-order-segment, and clipped-segment paths. The segment
tables are not integrated into `RealizationPlan`; there is no local refinement, AMR topology,
geometrically derived hanging-node map, multidimensional hp realization, production SIMD/GPU
backend, or embedded-domain source semantics.

The numerical dependency moved directly from Solverang to Methodus at Methodus
`d5354abb4dfd197ba5fd66f3742f9820701e4c43`; Finitum has no dependency on the
generalized Solverang constraint engine.

The FC10 `MethodProgram` and FC11 serialized-kernel contracts were validated against Scientia
`215433962c874dfd86b59ffc6d69f017bba2b95a` and Malleus
`09e27a6a23a6a5eab6f881ac0bec9db23046d58e`.

The FC6 linear operator continues to evaluate generated JVPs at zero active input. FC7 callers use
the explicit residual and state/rate JVP actions at a runtime linearization point. Concrete
constrained DOFs and their values are caller-supplied: Finitum verifies that semantic and concrete
constraints are both present and that
their extents are valid. The R3P affine-rectangle path supplies independently digest-bound CAD
region/boundary associations; generic meshes still have no boundary-region tags with which to
prove partition membership. Point-kernel buffers are allocated per invocation, external sampling prepares its
own cell geometry, and reference CSR assembly performs one operator action per column; these are
deliberate fixture-grade costs, not production performance claims.

The matrix-free, element-assembled, and partial constructors all freeze the generated JVP at zero
state/rate. Affine dependency constraints replace target rows with algebraic constraint residuals,
so the full-coordinate action is nonsymmetric even when the reduced `P^T A P` block is symmetric;
the operators declare this to Methodus and conjugate gradient refuses it. Equispaced segment
nodes through order 16 are reference data and are not a well-conditioned high-order basis claim.

FC10 remains a reference contract rather than a production meshless or particle engine. FV uses
explicit oriented faces, FD uses supplied neighbor rows, network realization uses dense matrices,
particle laws are generic radial polynomials over explicit pairs, and boundary kernels are
caller-supplied tables. No named physical law is selected in Finitum.

SV0-B3 consumes Methodus B1 tolerance, comparison, and convergence-order utilities. Malleus B2
owns local primal/JVP/VJP/parameter/backend campaigns; Finitum executes those kernels through its
existing realization plans but does not wrap or duplicate the local campaign API. B3 callers
provide exact fields, probes, tolerances, measured errors, and required orders. The providers do
not derive scientific obligations, select benchmarks, refine meshes, estimate discretization
error, run solvers, or promote support claims. Report consumers recheck the canonical identity;
they must call the fallible source-aware validator, which re-executes the check and refuses even a
rehashed inconsistent acceptance field. Constraint-work and transfer-conservation reports are
distinct, non-interchangeable types. Subject digests use recursively key-sorted canonical JSON or
an explicitly supplied owner digest.

## Validation

The R3D gate rebuilds the complete realization at perturbed design values and
compares `dJ/dp = -lambda^T dR/dp|_u` (adjoint solve through the existing
symmetric conjugate-gradient path) against centered differences of the rebuilt
objective across two step sizes with tightening error, on both the affine
rectangle and its two declared parameters.

```text
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --all-targets           # 156 passed, 0 failed across 25 binaries (W7 follow-ups; 153 at SC-W1 interface, 148 at SC-W1 ids/block actions, 144 at W7 package 3, 136 at W7 SV1-C1/C3 + P, 122 at the E6 close, 103 at SV2-B1 head fae5675, 52 at the R3D-era transcript)
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
git diff --check
python3 ../sinbad/scripts/check-physics-corpus.py        # 50 models
```

The W7 SC-W1 gate (153 tests) ran against Scientia `567251a`, Malleus `9862a08`, Methodus
`bf9082f` through a symlink overlay with a clone of Scientia's committed head, because the
Scientia working tree was mid-edit (not compiling) at the time; `-p finitum` scoping is the
rule for every cargo invocation (a `cargo fmt --all` follows path dependencies into siblings).

The realization gate includes an independent affine patch test on a nonuniform sheared mesh:
with `k = 1`, `f = 0`, and nonzero linear Dirichlet data, both realized operators reproduce every
exact P1 nodal value within `1e-12`.

The FC7 gate compiles a transient nonlinear form and verifies the combined generated state/rate
and dynamic-property JVP against centered differences.

The FC8 gate independently checks vector/tensor block ownership, a Stokes-like Schur complement
against the uncondensed local residual, H(div) flux preservation and shared-facet signs, a
conservative two-sided DG facet scatter, H(curl) circulation preservation, and exact simplex
incidence identities.

The FC9 gate checks nonmatching mortar work conservation, variable-order segment
partition/gradient identities, algebraic midpoint prolongation and constraint rows, CG refusal of
the declared nonsymmetric action, dense-reference sum factorization, an explicit
batch/component/lane packed index plus pack/unpack identity, exact clipped polynomial integration,
and quadrature-partial JVP agreement with generated Malleus interpreter execution and centered
differences.

The FC10 gate independently checks periodic FV conservation, a centered FD stencil, a dense
network DAE, equal/opposite particle forces plus an energy gradient, and weighted boundary-integral
semantics. FV/FD actions execute the digest-linked Malleus kernels emitted by Scientia.

The R3P gate recreates the same mesh and every node/cell/region/boundary association from one
provider revision, rejects stale/source-mismatched or ambiguous associations, and solves a
Scientia-generated zero-source Poisson case whose nonzero constant essential data are selected
only by stable CAD boundary identity. Every nodal value matches the independent manufactured
constant solution; matrix-free and assembled actions and converged primal solutions agree. This first path is
explicitly limited to affine rectangles in an XY carrier.

The R3D gate admits an annulus through the same carrier discipline, refuses
non-annulus families and stale revisions, verifies both design-velocity fields
against centered positions of rebuilt providers, matches the exact residual
sensitivity against centered differences of completely rebuilt realizations at
two step sizes for both rectangle parameters, refuses missing, duplicated,
short, and non-finite sensitivity data, and proves the adjoint identity
`dJ/dp = -lambda^T dR/dp|_u` against centered differences of the fully rebuilt
primal solve and objective with tightening error.

The SV0-B3 gate exercises the generic checker contracts with a synthetic vector-valued nodal
field and a prescribed second-order error sequence across three independently constructed segment
meshes; these are checker tests, not discretization certification. It additionally checks four
concrete global realization strategies on one generated Poisson plan, a nonsymmetric constrained
global transpose work identity, affine prolongation/transpose work, weighted nonmatching
interpolation work, and dimension-complete triangle/tetrahedron exact-sequence identities and
ranks. Hostile patch, transpose, cross-kind report, serialized-report tamper, missing 3-D
divergence, and non-refining mesh fixtures are rejected or produce non-accepted reports. The FC6
nonuniform sheared affine patch above remains the independent realization oracle.

## Known limits recorded by the 2026-08-30 workspace audit (tree `8ec3eac`)

- `Mesh::new(dimension, vertices, cells)` is the only generic mesh
  constructor; there is no structured-grid builder, no refinement facility,
  and no region or boundary tags on generic meshes. Structured meshes are
  hand-rolled in Sinbad four times. The only identity-bearing boundaries are
  CAD `StableId`s wired positionally.
- Scientia `RegionId` on `EssentialConstraintRequirement` and
  `BoundaryPartitionRequirement` is never used to select DOFs; validation
  checks only non-emptiness parity, so the boundary-partition assumption is
  not discharged.
- Superseded by GX-C1-C6/GX-F7 (landed after this audit, `6e6c4a4`/`89eea19`/`0d378a0`; see
  `sinbad/docs/simulation-vision/GX-CONTRACTS.md` C11.5-C11.7 for the authoritative record, not
  re-verified by this entry): non-basis inputs may now be a `FieldSource` (constant/nodal/sampled
  today; table/kernel from GX-C3) rather than only a trusted closure; `FacetTopology` is wired
  into `RealizationPlan` for exterior facet integrals; and `MatrixFreeOperator` can declare
  `Symmetric` from an explicit proof, and Malleus VJP kernels execute through
  `RealizationPlan::vector_jacobian_product`.
- Dirichlet only, as explicit DOF rows; every non-cell measure other than `ExteriorFacet` is
  refused, so there is still no Neumann/Robin/traction path beyond GX-C4's exterior facet data.
- Superseded by SV2-B1 (this milestone): `RealizationPlan` now admits H1 order 1 *or* 2 (one
  order per plan; a P2 exterior-facet integral is refused typed, since GX-C4's trace basis
  stays hardcoded P1). `SystemRealizationPlan` itself remains planning-only (no residual, JVP,
  or operator; no `DofMap`/`ConstraintSet`/bound Malleus executables/external inputs; its own
  `artifact_digest()` hashes only system+mesh+layout+facet *count*, not concrete content the way
  `RealizationPlan::digest()` does). Genuine multi-order product layouts and block operators with
  off-diagonals now exist as a separate, additive structural module (`mixed::MixedSpace`/
  `MixedOperator`, SV2-B1/B4) that bypasses Scientia forms/Malleus kernels entirely -- they are
  not wired into `RealizationPlan` or `SystemRealizationPlan`, and `MixedOperator` represents no
  forcing term (its `residual`/`jacobian_vector_product` are the same linear action).
  `MixedOperator` now supports essential-constraint elimination (`reduced`/
  `ReducedMixedOperator`) and a content-addressed `digest()` (SV2-B4 continuation), but this is
  still entirely structural machinery: no Scientia-form/Malleus-kernel-driven saddle-point case
  can execute through it or through `SystemRealizationPlan` yet (see "Next" below and the SV2-B4
  batch report's realization inventory for the precise gap list).

## Next

The GX-C program listed here previously is complete — `GX-C1/C2/C5`
(`6e6c4a4`), `GX-C3/C4/C6` (`89eea19`), `GX-F7` executed VJPs (`0d378a0`);
`sinbad/docs/simulation-vision/GX-CONTRACTS.md` C11.5–C11.7 is the
authoritative record. The GX exit gate passed on 2026-08-31 (Sinbad
`a1402f2`), and SV2-B1 with the SV2-B4 start landed at `fae5675` (P2
elements, `mixed::MixedSpace`/`MixedOperator`, block nullspace
representation).

Next work, demand-pulled by E6 Stokes (workspace `PLAN.md` §6 batch E6):

1. Done (SV2-B4 continuation): essential-constraint/Dirichlet-elimination handling
   for `MixedOperator` (`reduced`/`ReducedMixedOperator`/`essential_constraints_for_blocks`) --
   the pure-Dirichlet pressure-nullspace candidate now verifies against the reduced operator and
   solves under MINRES with the resolved `ConstantModeProjector`;
2. exterior-facet **element assembly** — `RealizationPlan::assemble` and its
   partial/geometry-sensitivity paths still hard-refuse facet integrals
   (GX-C4 landed the matrix-free path only; registered as a follow-up in
   GX-CONTRACTS C11.15 after it blocked 04-fick-diffusion's Neumann form);
3. P2 facet traces (the trace basis is hardcoded P1) and a
   degree-4-exact tetrahedron quadrature rule for honest 3-D P2 claims;
4. Done (E6: `739e2aa`, `5da4744`, `1af8946`): the executable Scientia-form-driven system
   realization — `SystemOperator` with per-block bound kernels, load vector, equation-sign
   symmetry proof, `OperatorStructure` threading, and H(div)/RT0 + P0 compatible realization;
   the real Stokes and mixed-Darcy corpus systems both solve. Remaining in this area:
   per-block stored/regional external inputs (only the closure-based `SystemConstitutiveInput`
   slice exists), a content-addressed digest over the executable system realization (the
   shape-only `artifact_digest` remains), Hcurl realization, and interior-facet/DG measures;
5. FC3 `minimum_polynomial_degree`-honoring quadrature (the P1 mass-matrix
   under-integration follow-up recorded in GX-CONTRACTS C11.7/C11.8).
6. SC composition (design `sinbad/ARCHITECTURE.md` §8, §12). Landed by W7 (2026-09-03):
   prerequisite batch P (`4a6fe65`, state-dependent `SystemOperator` with GX-A3 tangents);
   SV1-C1/C3 global transposes and coefficient VJPs (`0f570c2`); the runtime inf-sup checker
   (`9c21b67`); SC-W1 system-level ids keying `BlockLayout` plus public per-block
   actions/transposes (`b477525`); and the SV2-B2 interface-measure realization binding Malleus
   facet-pair kernels (`d552da2`). Still open, in order of pull:
   - re-key `SystemOperator`'s internal field tables and admit a multi-instance
     `SystemRealizationPlan` once Scientia's `scientia-system/1` `OriginMap` /
     `OperatorSystem/2` land (replace `SystemIdMap::compose`'s own allocation by Scientia's);
   - bridge Scientia's `InteriorFacet`/`Interface` factorizations (they already carry
     `MinusTrace`/`PlusTrace` inputs) into `InterfaceKernel`s so `bind_kernels` stops refusing
     them -- needs a driving `.res` case (SC-W2 CHT), plus `BoundChain::Composed`
     quadrature-point evaluation of a producer instance's output kernel;
   - SC-W2 `InterfaceRealization`/`ConnectionRealizationPlan` (elimination first) over the
     `InterfaceMeasure::between` machinery, multiplier/Nitsche and transfer beyond 1-D in SC-W3;
   - typed inf-sup pairing from Scientia (`InfSup { pair, constrained, multiplier }`) and an
     H(div)-norm variant of the estimate.

Extend method topology only from concrete acceptance cases, keeping
local-kernel meaning, backend policy, and realization identity explicit.
