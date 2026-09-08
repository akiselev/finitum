# Finitum status

Updated: 2026-09-07
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
+ W7/SC-W1 (Finitum) system-path parity: stored design tables, coefficient JVP/VJP, assembled/
  element/partial representations and agreement/capability/artifact receipts on `SystemOperator`
+ W7/SC-W1 (Finitum) `SystemIdMap::from_scientia` (ids by value from `scientia-operator-system/2`)
  and the typed `@inf_sup` obligation consumed as `InfSupPairing::from_obligation`
+ W7 package 7c (Finitum, single compile path): per-plan `SystemQuadrature` with the P1
  barycenter rule reproducing the single-model default bitwise on the system path,
  proof-aware `symmetry()` (a taken `prove_symmetry` outranks the structural claim), and the
  transient all-table agreement report equal to the single-model one with a typed, named
  `RepresentationUnsupported` refusal otherwise
+ W8 lane F1 (Finitum, strictly additive): the public field sampler `finitum-field-sampler/1`
  (`FieldSampler`, `QuadratureView`/`QuadratureRule`) -- value, physical gradient, divergence
  and exterior-facet traces of P1/P2 Lagrange (scalar and vector), P0 and RT0 fields at
  physical points with the crate's own bases and Piola maps, the plan's named quadrature rule
  and a degree-exact rule selector (new degree-5 triangle and tetrahedron rules)
+ W8 lane F2 (Finitum, a transition inside one wave): fallible external-input and constitutive
  callbacks -- `InputEvaluationError { code, origin: InputOrigin, message, location }`,
  `FinitumError::InputEvaluation` with `code()` returning the producer's own code, `try_new` on
  `DynamicExternalInput` / `SystemConstitutiveInput` (the infallible `new` is a thin wrapper),
  every Methodus boundary mapping it to `NumericError::Evaluation` verbatim; stored-table
  builders `try_sampled[_at]`, `FieldSource::fallible(|x, t| ..)`, the `_at(time)` forms of
  `external_inputs_from` and both essential-constraint samplers; Finitum's own kernel / table
  closures refuse typed (`REALIZATION_PROPERTY_UNAVAILABLE`) instead of NaN or a panic; the
  infallible forms are deleted by slice F3 after Sinbad 7d-2 migrates
+ W8 lane F-MI (Finitum, additive): the multi-instance `SystemRealizationPlan::composed` over
  Scientia's `scientia-operator-system/2` -- rows by `SysResId`, fields by `SysVarId` (two
  instances of one model, or two models with colliding `SymbolId`s, realize as distinct
  blocks), same-mesh `bind` chains realized at bind time (kernel-input path: the producer
  output kernel feeds the consumer operand through the Malleus `BindComposition`, executed
  for the residual; provider-input path: the output's value and tangent reach the consumer
  closures as `PointEvaluation::bound`), cross blocks by the §6 chain rule (JVP and exact
  VJP), keyed essential constraints, `finitum-system-realization/3` identity for composed
  plans; one-instance plans and every `/2` digest bitwise unchanged (`w7_system_path_parity`)

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
    wraps) integrates with the degree-4 triangle / degree-2 tetrahedron rules by default
    (`SystemQuadrature::Richest`) and has no rank deficiency there; on
    `SystemQuadrature::Barycenter` (W7 7c A) it shares the single-model default's rank-one P1
    mass by design. Evidence: the `p1_mass_tests` unit test (reference mass
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

- SC-W1 system-path parity (W7, 2026-09-05; GX-CONTRACTS C12.6 "not landed" items (a) and
  (b)): the one-instance `SystemRealizationPlan` now carries every single-model surface Sinbad's
  E7 path consumes, and reproduces the `RealizationPlan` products to roundoff.
  - (a) Stored tables on the system path: `SystemExternalInput { residual: SysResId, input:
    ExternalInput }` bound through `SystemRealizationPlan::bind_kernels_with_inputs(constitutive,
    stored, equation_sign, facet_regions)` (`bind_kernels[_with_facets]` delegate with no
    tables) to a non-basis input of a cell integral, laid out over the shared quadrature
    (`SystemRealizationPlan::quadrature()` before binding, `SystemOperator::quadrature()`
    after; `ExternalInput::from_coefficient_at(.., quadrature, layout, design)` builds a design
    table over it, `from_coefficient` delegates to it, `CoefficientLayout::dimension_at`
    likewise); a stored input is state-independent (zero direction, no chain rule), so a mixed
    binding (closures for properties, tables for sources/design slots) is admitted. Every
    non-basis cell input must be bound one way; wrong extent, basis inputs, facet integrals,
    double and closure+table bindings refuse typed (`InvalidRealization`/`UnsupportedRealization`).
    `SystemDistributedCoefficient { residual, coefficient: DistributedCoefficient }` keys
    `SystemOperator::{coefficient_dimension, coefficient_jacobian_vector_product,
    coefficient_vector_jacobian_product}` (physical coordinates; the residual's parameter kernels
    with the direction routed to that input, sign-scaled; the VJP through the layout's
    interpolation transpose) and their `ReducedSystemOperator` forms (constraint rows zero /
    masked, affine dependencies refused as SV1-C2). `LinearizedSystemOperator::assemble()` and
    `ReducedSystemOperator::assemble()` give canonical CSR (Methodus `CsrMatrix`, a
    `TransposableOperator`) at the linearization point / zero point. **Digest change:**
    `SystemOperator::digest` payload is `finitum-system-operator/2`
    (`SYSTEM_OPERATOR_DIGEST_SCHEMA`), adding every stored table's values -- two operators
    differing only in a design vector now differ in identity, exactly as two `RealizationPlan`s
    do; every `/1` digest changes.
  - (b) Representations and receipts: `SystemOperator::element_assembly(lane_width)` (per-cell
    local matrices over the concatenated field restrictions, offset into the layout; the existing
    `ElementAssemblyOperator`), `SystemOperator::partial_assembly(lane_width) ->
    SystemPartialAssemblyOperator` (stored per-(block, integral, output, point) Jacobians applied
    through each field's own basis action; refuses a bound constitutive closure exactly as the
    single-model path refuses dynamic inputs, and facet integrals), both on
    `ReducedSystemOperator` with its constraint rows; `check_system_realization_agreement(&reduced,
    probe, lane_width, tolerance) -> RealizationAgreementReport` (subject `system-operator`,
    digest = the operator digest); `ReducedSystemOperator::capability() -> RealizationCapability`
    (`finitum-realization-capability/1`, same type: elements deduplicated by field, all block
    measures, constraint kinds, representation kinds honest about facets/closures, coefficient
    products exactly when a table is bound, `symmetry()` = Scientia's structural claim, receipt
    source digests = the block's for one block / blake3 of the ordered per-block lists otherwise,
    realization digest = operator digest); `ReducedSystemOperator::artifact() ->
    SystemRealizationArtifact` (`finitum-system-realization-artifact/1`: operator/plan/system-id
    digests, per-block `SystemBlockReceipt`, mesh, per-field `SystemFieldArtifact { symbol,
    variable, dofs }`, constraints, bound inputs as `RealizationExternalInput` + residual).
  - Evidence (`tests/w7_system_path_parity.rs`, 5 tests, on the corpus snapshots
    `01-poisson.res` and the new `03-nonlinear-heat.res` snapshot): the one-instance system's
    factorization digest equals the single-model one and `essential_constraints_from_system`
    derives the identical constraint set; with both paths integrating on the shared degree-4
    rule (the test tabulates P1 at the system's table), residual, JVP, rate-shifted VJP,
    coefficient JVP/VJP, linearized action and transpose action, and the assembled CSR action and
    transpose agree to a worst relative difference of `1.9e-16` (Poisson, cell layout),
    `3.5e-16` (Poisson, vertex layout), `4.1e-16` / `2.7e-16` (nonlinear heat at a nonzero
    state and rate, shifts 0 and 2.5, closures `rho = 1 + 0.2T`, `cp = 1 + 0.3T^2`,
    `k = 1 + 0.1T`, `Q` the cell-layout design table); the system coefficient and linearized
    adjoint identities hold to `1e-12`; the zero-point CSR transposes agree; the realization-
    agreement reports carry the same outputs and the same max-abs errors (`8.9e-16` assembled,
    `4.4e-16` element, `0` partial) with all verdicts accepted; the capability's elements,
    measures, constraint kinds, representation kinds, derivative products, source digests, and
    (once the single-model symmetry is proven) symmetry are equal; the artifact's block digests,
    DOF map, constraints, mesh, and inputs equal the single-model artifact's; the design vector
    is part of the `/2` identity; the nonlinear system's capability omits `PartialAssembly` and
    its `partial_assembly` refuses typed; coefficient/binding misuse refuses typed.
  - Deviation from the single-model shape: the coefficient handle and the stored input carry a
    `SysResId` (SC-W1's residual key), not an equation name, so Sinbad's merged runner can key by
    system ids from the start; `SystemConstitutiveInput` keeps its equation-name key unchanged.
    Not landed: stored tables on facet integrals; regional (per-region) tables; the multi-
    instance `SystemRealizationPlan` (unchanged, waits on Scientia `OperatorSystem/2` adoption).

- W7 package 7c, single compile path (Finitum, 2026-09-05; Sinbad's
  `w7-7c-deliverable2-wip.md` "Cross-repo needs (Finitum)"), deliverable A -- per-plan
  quadrature on the system path:
  - `SystemQuadrature { Barycenter, Richest }` is chosen at
    `SystemRealizationPlan::with_quadrature(system, mesh, layout, rule)` (`new` keeps
    `Richest`: the degree-4 triangle / degree-2 tetrahedron / 3-point segment rule every plan
    used before), reported by `quadrature_rule()`, tabulated by `quadrature()` before binding
    and `SystemOperator::quadrature()` after (stored tables are sized per point), and inherited
    by everything derived from the plan. `Barycenter` is the single-model P1 default
    (`PreparedElement::linear_simplex`: one point per cell, C11.8's rank-one P1 mass included)
    and is refused typed (`UnsupportedRealization`) for any field that is not an order-0/1
    Lagrange (H1/L2) field -- one point cannot integrate a P2 stiffness or an RT0 mass (the
    Taylor-Hood refusal is pinned in `tests/sv2b4_system_stokes.rs`).
  - **Digest change (loud):** the rule is part of the plan identity, so the plan digest
    payload is now `finitum-system-realization/2` (adds `quadrature`), and EVERY
    `finitum-system-operator/2` value changes with it (the operator payload embeds the plan
    digest; the operator payload's shape and schema id are unchanged). No sibling pins a
    digest value (Sinbad's `sc_w1_reroute` fixtures pin solutions, observables and verdicts).
  - Evidence (`tests/w7_system_path_parity.rs`, +4 tests): on `Barycenter` the one-instance
    system reproduces the single-model DEFAULT plan bitwise (relative difference exactly 0) --
    residual, JVP, shifted VJP, coefficient JVP/VJP, linearized action and transpose, the
    zero-point CSR action, and the realization-agreement report (outputs, verdicts, max-abs
    errors) -- on `01-poisson` (cell and vertex design layouts) and `03-nonlinear-heat`
    (closures `rho`, `cp`, `k`, stored `Q`, rate shifts 0 and 2.5). The only roundoff left
    (1.2e-16..2.7e-16) is the cross-representation check of the system CSR matvec against the
    single-model matrix-free linearized action, which sums in a different order by
    construction (the single-model linearized operator has no assembly); the `Richest` parity
    stays at 1.3e-16..4.1e-16.
  - Sinbad's `03` final-time nodal ladder reproduced on the system path (BDF2, step 0.05 to
    0.4, `k = 1 + 0.2 (T - 300)` closure with its tangent, stored `rho = cp = 1` and the
    closed-form `Q`, walls at 300, RMS nodal error at `t = 0.4`, dense Newton): `Barycenter`
    2x2/4x4/8x8 errors 1.177e-1 / 3.223e-2 / 8.802e-3, pair orders 1.868 / 1.872 (gate >= 1.8,
    Sinbad's `reference-orders/1` minimum); `Richest` errors 5.185e-2 / 1.692e-2 / 4.786e-3
    -- the very numbers Sinbad's diagnosis recorded under the degree-4 rule -- pair orders
    1.616 / 1.822. The order loss Sinbad saw is the rule's, not the system path's. (The
    `Richest` 8x8 level was measured once for this record; the battery keeps its first pair,
    the six-point 8x8 level under dense Newton costs minutes in a debug build.)
  - Cost (`system_path_residual_and_jvp_cost_is_recorded_against_the_single_model_plan`,
    debug build, residual + JVP wall time, recorded not gated; two runs): `Barycenter` 3x3
    single-model 5.5-6.3 ms vs system 6.8-7.1 ms (1.08-1.28x), 24x24 332-445 ms vs 412-434 ms
    (0.98-1.24x); `Richest` 3x3 30.7-35.6 vs 36.0-39.7 ms (1.01-1.29x), 24x24 1.90-2.03 s vs
    1.95-2.06 s (1.01-1.03x). Within the 1.5x target; the 4-8x Sinbad measured is the
    six-point rule itself (24x24: 0.4 s -> 2.0 s on either path), gone on `Barycenter`.
  - Cross-repo need (Sinbad): build every P1 level with
    `SystemRealizationPlan::with_quadrature(.., SystemQuadrature::Barycenter)` and size tables
    with `plan.quadrature()` as today; keep `new` (`Richest`) for Taylor-Hood and RT0/P0.

- W7 package 7c, deliverable B (Finitum, 2026-09-05) -- proof-aware symmetry on the system
  path:
  - `SystemOperator::symmetry()` -- and so `properties().symmetry()` on `SystemOperator`,
    `ReducedSystemOperator` and `LinearizedSystemOperator`, and the capability's `symmetry` --
    reports a taken `prove_symmetry` first: Scientia's structural claim before a proof (or
    `Unknown` for an `equation_sign`-resigned system, as before), the assembly proof after, in
    both directions: a passed proof upgrades `Unknown`/`Nonsymmetric` to `Symmetric`, a failed
    proof reports `Nonsymmetric` and never upgrades anything, a refused proof (dimension cap,
    bad tolerance) records nothing. Proof, never declaration: a Methodus conjugate-gradient or
    MINRES solve admits the operator with no caller-side `AssumeSymmetric`. A proof is evidence
    about the operator, not identity: the digest is untouched.
  - Evidence (`tests/w7_system_path_symmetry.rs`, 2 tests, on the new corpus snapshot
    `17-linear-elasticity.res` and `01-poisson.res`; `tests/sv2b4_system_stokes.rs` extended):
    17-linear-elasticity as a one-instance 3-D system on a 3x3x3 cube (the opaque `stress`
    constitutive as a closure with its exact Hooke tangent, a stored body-force table, the
    whole boundary clamped, `Barycenter`) carries the structural claim `Nonsymmetric`, is
    refused by Methodus CG under `RequireDeclared` before the proof, is proven `Symmetric` by
    assembly, every symmetry surface then reports `Symmetric` with the digest unchanged, and CG
    under `RequireDeclared` converges in 8 iterations to a relative residual of 2.6e-16.
    01-poisson is structurally `Symmetric` (C5.4/C5.5) already: CG admits it before the proof,
    the proof confirms, the solution is unchanged. The unsigned Stokes system (structural
    `Unknown`) fails its proof: `Nonsymmetric` on the operator, its reduced form and the
    capability, and MINRES refuses on the declared nonsymmetry -- a failed proof never upgrades.
  - Cross-repo need (Sinbad): drop the WIP patch's `ConjugateGradientSymmetryPolicy::
    AssumeSymmetric` fallback under a proof; keep calling `operator.prove_symmetry(tol)` once
    per level when CG/MINRES is requested and hand the reduced operator to Methodus with the
    default `RequireDeclared`.

- W7 package 7c, deliverable C (Finitum, 2026-09-05) -- realization agreement on a transient
  one-block system with all-table inputs (Sinbad's D2):
  - Reproduced on the new corpus snapshot `02-transient-diffusion.res` as a one-instance system
    with every input a stored `SystemExternalInput` table (`capacity`, `k`, `f`; the
    `capacity * dt(u)` mass/rate term present): `check_system_realization_agreement` accepts,
    and its report equals the single-model `check_realization_agreement` report -- all four
    outputs bitwise, the three verdicts accepted, the maximum absolute errors equal -- on
    `Barycenter`, and to 1.9e-16 with equal errors on `Richest`; the state products
    (residual, JVP, shifted VJP, linearized action/transpose, zero-point CSR) are bitwise on
    `Barycenter`. So the D2 refusal is not the mass term: it is a closure. Sinbad's
    `depends_on_fields` rule makes `capacity = storage_capacity(u)` and `k = diffusivity(u)`
    closure pairs because the providers name the state field, although the case binds them to
    constants.
  - Typed refusal: `FinitumError::RepresentationUnsupported { representation:
    RepresentationKind, equation, integral, input: Option<TensorInputId>, reason }` (Display
    `REPRESENTATION_UNSUPPORTED: ...`) from `SystemOperator::partial_assembly` -- and so from
    the agreement check -- names the first closure-bound input (the closure's identity in
    `reason`) or the first non-cell integral (`input: None`). Pinned with `k` a closure on 02:
    the report is refused naming `PartialAssembly`, `evolution`, the integral and `k`'s input
    id; the single-model report refuses too (`UnsupportedRealization`, unchanged); the system
    capability omits `PartialAssembly`; element assembly still agrees bitwise
    (`tests/w7_system_path_parity.rs`, +2 tests; the nonlinear-heat refusal pin updated).
  - Recorded gap (single-model path, not this package): `RealizationPlan::capability` lists
    every representation kind unconditionally, so it claims `PartialAssembly` for a dynamic
    input its own `partial_assembly` refuses; the system capability is honest.
  - Cross-repo need (Sinbad): bind a `ModelDefinedProperty`/`ModelDefinedValue` whose case
    binding is a constant (or a coordinate/time expression) as a stored table even when the
    provider's declared arguments name a field, and reserve the closure pair for definitions
    that actually read the state (an expression over `T`, a `ModelDefinedConstitutive`); then
    D2's agreement report passes on the system path as it did on the single-model path. Carry
    a `RepresentationUnsupported` refusal into the receipt instead of `None`.

- SC-W1 Finitum side, HANDOFF §6 items (W7, 2026-09-05): Scientia's landed system ids and typed
  inf-sup pairing consumed.
  - `SystemIdMap::from_scientia(&scientia::SystemOperator)`: every `SysVar { id, owner, local }`
    and `SysResBlock { id, origin: Equation { instance, name }, row }` copied by value into
    Finitum's `u32` newtypes (no re-allocation), one `InstanceRecord` per instance with the
    `artifact_digest` taken from `instance_artifacts` (an instance without one refuses
    `ArtifactMismatch`); the implicit root (empty name) is recorded under its model name.
    Evidence (`tests/w7_sc_w1_scientia_ids.rs`, 3 tests): the implicit one-instance Poisson map
    from Scientia **equals** `SystemIdMap::one_instance` (same `finitum-system-ids/1` identity,
    `SysVarId(symbol.0)`, `SysResId(0)`, the `/1` artifact digest of the directly compiled
    system); a declared two-instance `TwoHeat { a, b: HeatConduction }` map from Scientia
    **equals** `SystemIdMap::compose(&[("a", ..), ("b", ..)])` (same identity; dense ids
    `0..2`, paths `b.thermal`, rows = Scientia's `SysResBlock.row`), so Finitum's own dense
    allocation is proven to be Scientia's and `compose` is kept only for callers composing `/1`
    artifacts by hand. `InfSupPairing::from_obligation(&VerificationObligationKind)` binds the
    typed `InfSup { constrained: Some, multiplier: Some }` pairing (refuses a non-inf-sup kind
    `InvalidRealization`, an undecided `None` side `UnsupportedRealization`, a same-field pair);
    on the corpus `StokesFlow` and `MixedDarcy` snapshots the typed pairing equals
    `InfSupPairing::from_structure`'s structural derivation. Not landed: re-keying
    `SystemOperator`'s internal field tables by `SysVarId` and the multi-instance
    `SystemRealizationPlan` (two instances of one model in one group) -- the id map is ready
    for it, the per-block field/constraint tables are still per-model `SymbolId`.

- W8 lane F1 (2026-09-07, PLAN §6 W8 decision 4, gate G3): the public field sampler, module
  `src/sampler.rs`, digest schema `finitum-field-sampler/1`. Strictly additive: no existing
  public item, signature, digest value or default changed (Sinbad lane A1 built against this
  tree concurrently).
  - `FieldSampler<'a>`: `new(mesh, SampledFamily, values)` over the canonical DOF maps,
    `from_realization_plan(plan, values)` (order from the plan's element basis count, the
    plan's own `DofMap`), `from_system_plan(plan, field, solution)` (family from the system's
    typed element requirement, mirroring `build_field_elements`' admission rule; the DOF map is
    proven equal to `SystemOperator::dof_map(field)`), `from_mixed_space(space, field,
    solution)`. `SampledFamily::{Lagrange { order: 1 | 2, components: 1 | dimension },
    CellConstant, RaviartThomas0}`; anything else refuses the new typed
    `FinitumError::SamplingUnsupported { family, reason }` (`SAMPLING_UNSUPPORTED:` message).
    Evaluation: `value_at` / `gradient_at` (rows per component, covariant Piola `J^{-T}`) /
    `divergence_at` / `sample_at` at physical points (inverse affine map; `cell_contains`
    tells extrapolation), `*_at_reference` variants, `trace_at` / `trace_at_centroid` on an
    exterior facet (`FacetTrace { value, normal (outward unit), measure, cell, local_facet }`,
    `normal_component()`), `cell_measure`, `cell_centroid`, `facet_centroid`,
    `exterior_facet`. RT0 evaluates `rt0_reference_basis` under `AffineMap::contravariant_piola`
    with `CompatibleDofMaps::hdiv`'s orientation table (the same objects `crate::system`
    executes); its per-cell gradient is `(sum_i orientation_i dof_i / det J) I`, whose trace is
    `map_hdiv_divergence`. `conventions()` / `digest()`: `FieldSamplerConventions` (family, DOF
    ordering, reference basis, pullback, facet convention) hashed under
    `finitum-field-sampler/1` -- conventions only; a receipt pairs it with the realization
    digest for the data. Free functions `cell_measure`, `cell_centroid`, `exterior_facet`
    (dimension 1 handled here; 2/3 reuse GX-C4's `FacetGeometry`), `simplex_monomial_moment`.
  - `QuadratureRule { id, dimension, degree, points }` with `known(dimension)`,
    `for_degree(dimension, degree)` (smallest named rule exact to `degree`: segments
    Gauss-Legendre to degree 15; triangles `simplex-barycenter` 1, `triangle-edge-midpoints` 2,
    `triangle-dunavant-6` 4, `triangle-radon-7` 5; tetrahedra `simplex-barycenter` 1,
    `tetrahedron-symmetric-4` 2, `tetrahedron-symmetric-14` 5 -- the two degree-5 rules are new
    in `element.rs`, closed-form Radon and the positive 14-point Walkington/Yu rule; higher
    degrees refuse typed), `from_table` (a plan's table takes its name; an unnamed table is
    `caller-table` with a probed degree), `verified_degree(cap)` (monomials against the
    closed-form simplex moments, `1e-13`), `identity()` under `finitum-quadrature-rule/1`.
    `QuadratureView<'a>`: `of_realization_plan` / `of_system_plan` / `new(mesh, rule)`,
    `rule_for_degree(d)`, `cell_points(cell)` (`PhysicalQuadraturePoint { reference, physical,
    weight = reference weight * |det J| }`), `integrate` / `integrate_over`.
  - Additive accessors `RealizationPlan::element()` and `RealizationPlan::dofs()`.
  - Evidence: 13 unit tests in `src/sampler.rs` (P1 affine and P2 quadratic reproduction to
    `1e-14` / `1e-13` in value and gradient, scalar and vector, on sheared 1-/2-/3-D meshes with
    mixed cell orientations; RT0 constant flux through the Piola map with outward unit normals
    and `flux . n = sign(det J) * orientation * dof / ((d-1)! |F|)` per facet DOF; every named
    rule verifies its declared degree against the closed-form moments and integrates the
    volume; `rule_for_degree(2)` integrates `dot(u, u)` of a P1 interpolant exactly -- on the
    reference triangle `1/6` against the barycenter rule's `1/9`, a 33 % relative error, and on
    the sheared mesh degree 2 equals degree 5 to `1e-14` while barycenter differs by more than
    `1e-3` relative) and 6 integration tests in `tests/w8_field_sampler.rs` (agreement with
    Finitum's own quadrature-point evaluation, recorded by constitutive closures inside
    `RealizationPlan::residual` and `SystemOperator::residual`: scalar P1, scalar P2, vector P1
    3-D elasticity, RT0 + P0 mixed Darcy and P2-vector/P1 Taylor-Hood Stokes on sheared meshes,
    to `1e-12`..`1e-13`; the Darcy check also closes the divergence theorem between sampled
    normal traces and the sampled divergence; each plan's `QuadratureView` names its rule and
    reproduces the recorded points).
  - Honest limits: affine simplices only; exterior facets only (interior facets have no
    one-sided trace; SC-W2 interface observables need the two-sided `InterfaceMeasure` path);
    no point location (`cell_contains` only); Hcurl, DG, P3+ and any Hdiv order above 0 refuse;
    `from_system_plan` is keyed by per-model `SymbolId` (a composed multi-instance layout would
    need a `SysVarId` constructor); P2 facet traces are exact here but the executable plan's
    facet integrals still refuse P2 (unchanged); the sampler digest covers conventions, not the
    mesh or values.

- W8 lane F2 (2026-09-07, PLAN §6 W8 decision 3, gate G3): fallible external-input and
  constitutive callbacks, a transition inside one wave -- every existing constructor, signature,
  default and digest value keeps working (Sinbad lane A2 builds against this tree through a
  path dependency the whole time); the fallible forms land beside them and the infallible ones
  become thin wrappers; slice F3 (see "Next") deletes the infallible forms after Sinbad 7d-2
  migrates. This is not a compatibility layer: it has a named deletion slice.
  - Types (`src/error.rs`): `InputOrigin { Slot, ExpressionPath, Provider, Table }` as F1
    proposed (`Provider(p)` displays `provider/p`, `Table(t)` displays `t (stored table)`,
    the other two display their string); `InputEvaluationError { code, origin, message,
    location: Option<Box<InputLocation>> }` with `InputLocation { cell: Option<CellId>,
    point: Vec<f64>, time: Option<f64> }` -- F1's proposal plus the cell, with the
    Finitum-filled location boxed into one record so the error stays under clippy's 128-byte
    `Err` threshold in every callback's `Result` (flat: 136 bytes; consumer closures would
    trip `result_large_err`); `new(code, origin, message)`, accessors `cell()` / `point()` /
    `time()`, `location_text()`, Display `<code> at <origin>, point (x, y), t = 0.1, cell 3:
    <message>` with absent parts omitted; `FinitumError::InputEvaluation(Box<InputEvaluationError>)`
    (`#[error(transparent)]`, `From<InputEvaluationError>`; boxed for the same lint on
    `FinitumError` itself); `FinitumError::code() -> Option<&str>` (the original
    producer code for `InputEvaluation`, never a Finitum one; the static code of
    `RepresentationUnsupported` / `RealizationTangentUnavailable` / `SamplingUnsupported` /
    `InfSupUnstable`; `None` for structural errors); `impl From<FinitumError> for
    methodus::NumericError` -- `InputEvaluation` becomes `NumericError::Evaluation { code,
    origin: origin.to_string(), message: "<location>: <message>" }` (Methodus `bec099f`),
    everything else the flat `NumericError::Operator { message }` it always was. Every Methodus
    boundary goes through it (the `numeric_error` helpers in `realization.rs` / `system.rs` /
    `method.rs`, the inline `map_err`s in `optimized.rs`, `mixed.rs`, `interface.rs`,
    `system.rs`).
  - Fallible dynamic callbacks: `DynamicExternalInput::try_new(integral, input, components,
    identity, value: Fn(&PointEvaluation) -> Result<Vec<f64>, InputEvaluationError>,
    direction: Fn(&PointEvaluation, &PointEvaluation) -> Result<..>)` and
    `SystemConstitutiveInput::try_new(equation, ..)` likewise. Naming: F1 proposed reusing the
    plain names with the fallible closure types, which the concurrent-build constraint forbids;
    `try_` is the Rust idiom for the fallible form of an operation and is the name that
    survives F3 (no second rename for consumers). `new` on both is `try_new` with `Ok`, proven
    bitwise (equal digest, equal residual / JVP / VJP / assembled actions on both paths).
    Finitum locates a returned failure at the evaluation's cell, physical point and time
    (overwriting whatever the callback set; Finitum is the authority on where it evaluated)
    and propagates it as `FinitumError::InputEvaluation`. The first failure in cell, then
    quadrature-point, then declared-input order wins, deterministically (same error on every
    repeat).
  - Fallible stored-table builders and time-aware sampling (Sinbad A1's cross-repo need, same
    slice): `ExternalInput::try_sampled(.., FnMut(CellId, &[f64]) -> Result<Vec<f64>,
    InputEvaluationError>)` (steady), `ExternalInput::try_sampled_at(.., time, FnMut(CellId,
    &[f64], f64) -> Result<..>)`, `ExternalInput::try_sampled_on_facets` /
    `try_sampled_on_facets_at`, `ExternalSensitivityInput::try_sampled` (design derivatives are
    steady). A refusal is located at the cell (a facet's owning cell) and physical point, at
    `time` for the `_at` forms and without one for the steady forms, re-labelled
    `InputOrigin::Table(<origin display>)` (a `Table` origin is kept) and returned typed at
    construction -- no table is ever built with a non-finite placeholder, and the old
    `sampler_error`-then-NaN pattern inside `external_inputs_from` (whose captured error was
    shadowed by the non-finite check) is gone: every stored table samples through one
    `sample_cell_table` / `sample_facet_table` core whose first `FinitumError` ends the
    sampling. The infallible `sampled` forms are that core with `Ok`. `FieldSource::Fallible(
    Arc<dyn Fn(&[f64], f64) -> Result<Vec<f64>, InputEvaluationError>>)` /
    `FieldSource::fallible(..)` is a new variant beside `Sampled`: Sinbad `run.rs` and Krasis
    `initial.rs` match `FieldSource::Sampled(sampler)` by payload and call it, so its closure
    type cannot change in this wave, and both consumers' matches carry a wildcard arm, so the
    added variant compiles for them; it is time-aware and its `identity()` hashes the `Arc`
    address like `Sampled`'s. Time-aware forms, each the existing function with the time as a
    parameter (the legacy forms pass `0.0`, their documented convention):
    `external_inputs_from_at(.., time)` (coordinate-only `Kernel` / `Table` sources sampled
    with `t = time`, `Fallible` sources through `try_sampled_at`; the state-dependent dynamic
    bindings are unaffected, they read the runtime `PointEvaluation::time`),
    `essential_constraints_from_at` / `essential_constraints_from_selected_at` (profile path;
    `Table` / `Kernel` / `Fallible` at `time`), `essential_constraints_from_system_at` (system
    path, nodal and RT0 normal-trace data; `Fallible` admitted beside `Constant` / `Nodal` /
    `Sampled`). A `Fallible` Dirichlet refusal is located at the node (no cell) and `time`,
    origin untouched. `system_constitutive_from_sources` binds a `Fallible` source as a
    `try_new` closure at the runtime point's coordinates and time (zero direction).
  - Finitum's own bound property closures are typed too: `system_constitutive_from_sources`
    (`Kernel` / `Table` sources, state-free and state-dependent) and `external_inputs_from[_at]`
    (state-dependent `Kernel` / `Table` bindings) build `try_new` closures. A table axis point
    outside the table's range, a kernel execution failure, a table axis that is neither a
    coordinate nor the bound active input, or an evaluation point without the active input's
    value is `InputEvaluationError { code: REALIZATION_PROPERTY_UNAVAILABLE (new, exported
    constant), origin: ExpressionPath("<model>.<equation>[<integral>].<symbol>") on the system
    path and `"<model>[<integral>].<symbol>"` on the single-model path (its factorization
    carries no equation name), message: the underlying Finitum error }`; a tangent the kernel
    declines at a point is `REALIZATION_TANGENT_UNAVAILABLE` (the build-time refusal's code).
    Before this lane the system path returned NaN placeholders that `validate_finite` reported
    as "constitutive input contains a non-finite value" and the single-model path panicked on
    `.expect("validated at build time")` -- both what decision 3 forbids.
  - Honest limits: `check_global_transpose` (caller-supplied Methodus operators) recovers code
    and origin typed but the location only as message text, and a `Slot` / `ExpressionPath`
    origin round-trips as `Slot`; `essential_constraints_from_system[_at]` still refuses
    `Table` / `Kernel` sources (unchanged); a stored table is sampled at one time -- a
    transient consumer rebuilds it (and the constraint set) per step, there is no per-point
    time; `FieldSource::Fallible`'s identity is its `Arc` address, like `Sampled`'s;
    `InputEvaluationError` is not serialized (`FinitumError` never was); no
    `last_evaluation_failure()` record exists because nothing needed one -- if a future
    Methodus trait method returns no `Result`, that is where it would go.
  - Carrying mechanism, per entry point: **propagated everywhere, nothing recorded**. Every
    Methodus operator trait entry point returns `Result<(), NumericError>` and
    `NumericError::Evaluation` carries the typed payload, so no `last_evaluation_failure()`
    cell exists or is needed. `RealizationPlan` and `SystemOperator` / `ReducedSystemOperator`:
    `residual`, `jacobian_vector_product`, `vector_jacobian_product[_shifted]`, `load_vector`,
    `assemble` (eager, at `t = 0`, zero state), `element_assembly` (eager), the coefficient
    products (through the same point evaluators), `linearize` (lazy: the failure surfaces on
    the linearized operator's first `apply`, as `NumericError::Evaluation`); the Methodus impls
    `LinearOperator::apply`, `TransposableOperator::apply_transpose`, `NonlinearOperator::*`,
    `DaeOperator::{residual, jacobian_vector_product}` -- so a whole `bdf_step` returns
    `SolveError::Numeric(NumericError::Evaluation { code, origin, .. })`; and
    `check_realization_agreement` / `check_system_realization_agreement`, which now drive the
    crate's own representations through a crate-private `TypedAction` so the located error
    reaches the caller as itself instead of flattened through the Methodus trait boundary and
    back. `check_global_transpose` (caller-supplied `&dyn LinearOperator`) recovers a
    `NumericError::Evaluation` as a typed `InputEvaluation` with exact code and origin and the
    location as message text (`InputEvaluationError::from_numeric`; a `Slot` and an
    `ExpressionPath` display identically, so that round trip yields `Slot`). Partial assembly
    refuses dynamic inputs before any callback runs (unchanged). No entry point returns a
    non-finite value in place of a callback failure.
  - Evidence: 4 unit tests in `src/error.rs` (display order, `code()`, the Methodus mapping and
    its round trip, origin displays) and 5 integration tests in `tests/w8_fallible_inputs.rs`:
    a P1 input refusing at the second and third quadrature points of cell 3 and everywhere on
    cell 5 (degree-2 rule, 8 cells) reports cell 3 and the second point's coordinates (equal to
    the plan's own `QuadratureView` point) with the callback's code / origin / message and the
    action's time from `residual`, JVP, VJP, `linearize` + `apply`, `assemble`, `load_vector`,
    `element_assembly`, the matrix-free Methodus action and `check_realization_agreement`
    (`a_p1_input_refusing_at_one_quadrature_point_is_located_and_carried_by_every_action`);
    two inputs refusing at one point yield the first in the factorization's declaration order
    (`among_inputs_refusing_at_the_same_point_the_first_in_declaration_order_wins`); the
    system-path `ka` refusing on cells 4 and 2 reports cell 2's first quadrature point through
    every `SystemOperator` action, the reduced `DaeOperator` / `LinearOperator` impls and a
    Methodus BDF step
    (`a_system_constitutive_refusal_is_located_and_carried_through_the_reduced_dae_operator`);
    the infallible constructors reproduce the fallible-with-`Ok` ones bitwise with equal
    digests on both paths (`the_infallible_*_constructor_is_the_fallible_one_with_ok_bitwise_and_digest_equal`).
    Stored tables and time (3 more integration tests, 1 more unit test): on the transient
    all-table system path (P1 barycenter, `bind_kernels_with_inputs`) an `f` table builder
    refusing on cell 2 refuses at construction with origin
    `Table("TransientNonlinear.evolution[0].f")`, cell 2, the cell centroid and `t = 0.5`, the
    steady form records no time, and without the refusal the `t = 0.5` table holds `0.5`
    everywhere and the all-table operator loads it
    (`a_failing_table_builder_refuses_at_construction_with_a_table_origin_on_the_all_table_transient_path`);
    `g(t) = t` Dirichlet data through `FieldSource::fallible` sampled at `t = 0.5` give `0.5`
    on every constrained DOF on the system path (`essential_constraints_from_system_at`) and
    the profile path (`essential_constraints_from_at`), the legacy forms give `0`, and a
    refusing datum is located at the node without a cell with its `Slot` origin untouched
    (`transient_dirichlet_data_g_of_t_is_sampled_at_the_requested_time_on_both_paths`);
    `external_inputs_from_at` samples a `Fallible` source at the given time (`0.5` everywhere,
    `0` through the legacy form) and a refusing one is `Table`-labelled at the first offending
    cell, while `system_constitutive_from_sources` binds it at the runtime time (the residual
    is affine in `t` through `fa = t`) and a refusal is located through the operator action
    (`fallible_field_sources_feed_time_sampled_tables_and_runtime_time_constitutive_inputs`).
    Finitum's own tables: `ka` tabulated over `b` on `[0, 1]` evaluates at `b = 0.5` and at
    `b = 5` refuses `REALIZATION_PROPERTY_UNAVAILABLE at Coupled.ea[0].ka`, cell 0, `t = 0.1`,
    from residual and JVP, never "non-finite"; the single-model `k` over `u` likewise at
    `TransientNonlinear[<integral>].k` where the dynamic binding used to panic
    (`finitum_s_own_bound_property_table_refuses_typed_at_runtime_instead_of_nan_or_a_panic`).
    The pre-existing 191 tests are unchanged and pass, which is the proof that the wrappers
    change no behaviour and that none of `finitum-system-realization/2`,
    `finitum-system-operator/2`, `finitum-field-sampler/1` moved.

- W8 lane F-MI (2026-09-07, PLAN §6 W8 lane F-MI, gate G2; the exact need of
  GX-CONTRACTS C12.9 "Sinbad `59b00af`/`ec0042f` -- W8 lane A2", items 1-4;
  `sinbad/ARCHITECTURE.md` §2.3/§2.4/§2.6/§6/§8): the multi-instance
  `SystemRealizationPlan` over one realization group (working-tree implementation;
  coordinator landing pending as of 2026-09-08). Additive: every existing signature,
  default and digest value is unchanged (the 205 pre-existing tests pass unchanged; the
  bitwise fixtures of `w7_system_path_parity` are the proof that internal re-keying moved no
  one-instance value).
  - **Item 1, keyed layout and tables.** `SystemRealizationPlan::composed(&scientia::
    SystemOperatorCompilation, mesh, layout, SystemQuadrature)` accepts the `/2`
    `SystemOperator` with its per-instance `/1` `OperatorSystem`s (`model_systems`, checked
    against `instance_artifacts` and each `SysResBlock`'s `model_system`/`block` digests by
    `scientia::block_digest`), its `output_kernels` and `compositions`; ids come from
    `SystemIdMap::from_scientia`; the layout is `BlockLayout::new_keyed` over
    `system_ids().variables()`. Internally the plan is a list of rows (`SysResId` order, each
    `(instance, per-model block)`) and the operator's field tables, local gathers, cotangent
    scatters, partial-assembly actions and receipts are keyed by `SysVarId`; every kernel
    input's `binding.symbol` resolves through its instance's `symbol -> SysVarId` map. On a
    one-instance plan the row index is the block index and `SysVarId(symbol.0)` orders the
    tables exactly as `SymbolId` did, so `(block index, integral, input)` keys, digest payloads
    and floating-point summation order are unchanged. New accessors: `plan.instance_system
    (InstanceId)`, `plan.system_operator()` (the `/2`, `None` on a one-instance plan),
    `SystemOperator::dof_map_by_variable`, `instance_structure(InstanceId)`, `binds()`.
    `system()` / `structure()` answer instance 0's `/1` artifact / structure; the
    symbol-keyed `dof_map(symbol)` / `mass_matrix(symbol)` / `essential_constraints_from_system`
    answer only symbols unique across instances (`BlockLayout::block`'s rule; `V` and `T` of
    the electrothermal pair are both `SymbolId(0)` of their models, so they are ambiguous
    there). Closures are keyed by `SystemConstitutiveInput::try_new_for_residual(SysResId,
    ..)` (an equation-name key is accepted only when unique across instances, else refused
    typed naming the count) and stored tables by `SystemExternalInput.residual` as before;
    `equation_sign` accepts the display path `<instance>.<equation>`. Methodus block names are
    `<instance>/field_<symbol>` on a composed plan (`field_<symbol>` unchanged otherwise);
    `symmetry()` of a plan with binds is `Unknown` until proven, of several instances without
    binds the conjunction of the instances' claims. Nullspace candidates stay symbol-keyed and
    are refused typed when the symbol is ambiguous.
  - **Item 2, kernel-input binds.** At bind time every bind's producer output (one cell
    point function, one bundle) is bound with `bind_kernels`; each `BindComposition` is
    `validate_composition`ed, its `composition_digest` checked against the recorded digest,
    its JVP rebuilt with `differentiate_composition` under Scientia's request (producer active
    operands -> consumer output) and checked against `jvp_digest`. At every consumer quadrature
    point the binds are evaluated in dependency order (an output kernel that reads a bound
    input of its own instance comes after that bind; an algebraic loop among outputs is refused
    typed). Residual: a consumer bundle with a composition runs `Interpreter::run_composition`
    (stage 0 the output primal kernel over the producer's fields gathered at the same point,
    stage 1 the consumer kernel, the shared buffer carrying `Q`). JVP: the §6 chain-rule
    product of local point kernels -- the producer bundle's full JVP (active basis directions
    *and* its frozen-input tangents through `execute_jvp_values`) is the bound operand's
    direction into the consumer's parameter JVP. **Deviation, recorded:** the `jvp_compositions`
    are digest-checked but not executed, because their independent set is the producer's
    active basis operands only (Scientia's request), so executing them alone would drop the
    provider-input tangent `sigma(T)` inside `joule_heat` and `dR_thermal/dT` through `Q`
    would be silently short; the two-kernel chain carries both, and Malleus's STATUS records
    the composition as "the contract, not a mandate to change [Finitum's] loop". VJP: the
    exact transpose -- the consumer's parameter cotangent of the bound operand seeds the
    producer output's VJP (active cotangents scattered through the producer's fields; its
    parameter cotangents chained on into the producer's closures and bound inputs, binds walked
    in reverse dependency order), proven by the adjoint identity to `1e-12` and equal to the
    monolithic transpose.
  - **Item 3, provider-input binds.** `PointEvaluation` gains `bound: Vec<PointBoundInput {
    symbol, slot, values }>` (empty on every one-instance realization; `bound_values(symbol)`,
    `bound_slot_values(slot)`): the value of every bound input of the consumer instance at the
    point, and in the direction callback the output's directional derivative; the transpose
    probes each closure with unit bound perturbations (`probe_bound_direction_evaluation`) and
    holds bound directions at zero on active probes. Output kernels' own non-basis inputs
    (`joule_heat`'s `sigma`) take `SystemConstitutiveInput::try_new_for_output(InstanceId,
    OutputId, integral, input, ..)` closures, which see the producer instance's active inputs
    and its bound inputs that precede the output in dependency order; a bound symbol may not
    also be bound as a closure or table (refused typed), and every other non-basis output input
    must have a closure (refused typed naming the output and the constructor).
  - **Item 4, constraints.** `SystemVariableEssentialConstraint { variable: SysVarId,
    requirement, value }` with `essential_constraints_from_system_by_variable[_at](operator,
    mesh, region_maps: &[(InstanceId, &RegionMap)], requirements[, time])`: the per-model
    requirement's `RegionId` resolves through the variable's instance's own `RegionMap` (an
    instance without one is `RealizationRegionUnmapped` naming region and instance); the
    symbol-keyed forms delegate to it. `mixed::BlockVariableEssentialValue` /
    `essential_constraints_for_variables` are the keyed forms `essential_constraints_for_blocks`
    now delegates to. `reduced` is unchanged (a `ConstraintSet` over the keyed layout).
    **Deviation, recorded:** no `SysRegionId` newtype -- instance regions are keyed by
    `(InstanceId, per-model RegionId)`, which is what Sinbad's per-instance `RegionMap`s carry
    today; a `SysRegionId` table on `SystemIdMap` would change `finitum-system-ids/1` (the
    one-instance identity `from_scientia == one_instance` is a test) for no consumer.
  - **Identity and receipts.** Composed plans digest as `finitum-system-realization/3`
    (`SYSTEM_REALIZATION_COMPOSED_DIGEST_SCHEMA`: the `/2` identity, every instance's name/
    model/`/1` artifact, the layout by variable and symbol, every bind's slot, output, path and
    composition/JVP-composition digests); `with_quadrature` plans keep `/2` bitwise.
    `finitum-system-operator/2` gains an `outputs` list of output-kernel closure identities that
    is omitted from the payload when empty, so one-instance digests are unchanged.
    `SystemRealizationArtifact` gains `instances` and `binds: Vec<SystemBindReceipt {
    consumer_slot, consumer, producer, output, path: BindPath::{KernelInput, ProviderInput},
    rows, columns, compositions, jvp_compositions }>` (both `skip_serializing_if` empty);
    `SystemBlockReceipt.equation` is the display path and `residual` always `Some`.
    `capability()` lists every instance's element requirements and drops `PartialAssembly`
    when binds exist; `partial_assembly` and therefore `check_system_realization_agreement`
    refuse typed (`RepresentationUnsupported { PartialAssembly, equation: <consumer row>,
    reason: "the same-mesh bind on `..`" }`) because a bind chain is a state-dependent point
    chain; matrix-free, `assemble`, `element_assembly`, `linearize`, `block_operator`,
    coefficient JVP/VJP (a stored table on one instance's row) and `prove_symmetry` work.
  - Coordinator review follow-up (2026-09-08, working tree): output closures now remain
    available for every consumer of a shared output, fixing fan-out failure at bind time.
    A three-instance electrothermal fixture checks equal thermal residuals, state/rate finite
    differences, and the shifted transpose identity through both outgoing binds. The bind
    dependency sort now includes self-edges: `Q <- heating`, with
    `heating = Q + rho * cp * dt(T)`, refuses an algebraic output loop at plan admission.
    Pure-basis producer outputs do not depend on provider-input binds; opaque output
    closures conservatively depend on all such binds of their producer instance.
  - Evidence (`tests/w8_multi_instance_system.rs`, 9 tests, snapshots `fixtures/corpus/
    08-electrothermal-joule.res` and `fixtures/corpus/modules/{physics.electrical,
    physics.thermal, systems.electrothermal}.res` copied verbatim from Sinbad): the two-instance
    `Electrothermal` (both binds: `thermal/input/Q <- electrical.joule_heat` on the kernel-input
    path with one composition and one JVP composition, `electrical/input/temperature <-
    thermal.temperature` on the provider-input path) as one plan on a 3x3 unit square versus
    the monolithic 08 one-instance plan, hand-closed the way Sinbad's dual closures close it
    (`current_density = -sigma(T) grad V`, `joule = sigma(T)|grad V|^2`, `k(T)`, constant
    `rho`/`cp`), under both `Barycenter` and `Richest`, at zero, perturbed and nontrivial
    `(t, u, u_t)`: residual, full JVP and all four `(row, column)` blocks agree after the
    `OriginMap` permutation to a measured worst relative discrepancy of `0` to `2.8e-17`
    (gate `1e-10`), both cross blocks `dR_thermal/dV` (max `2.3e-1`) and `dR_electrical/dT`
    (max `4.5e-4`) nonzero on a unit direction at an interior node and equal; the composed
    adjoint identity `<J d, w> = <d, J^T w>` with a rate shift to `1e-12`, the VJP equal to
    the monolithic one, both cross-block transposes adjoint; `linearize().assemble()` equal to
    the JVP and to the monolithic matrix, the Jacobian nonsymmetric at the nontrivial state
    while the zero-point view proves `Symmetric` (both cross blocks vanish with `grad V = 0`);
    receipts (`residual_path` `thermal.thermal`, `<instance>/field_` block names, two
    `SystemBindReceipt`s with their rows/columns/digests, dependency order `temperature` before
    `Q`, four capability elements, the typed partial-assembly refusal, matrix-free = assembled
    = element assembly); two `HeatConduction` instances (`TwoHeat`) with distinct `SysVarId`s
    and the same `SymbolId`, an ambiguous equation-name closure refused typed, each block equal
    to the one-instance realization of the same model with its own state to `1e-12`; keyed
    Dirichlet data on the electrical `V` through the electrical instance's own region map touch
    only `V`'s block, `reduced` keeps the count, the symbol-keyed form refuses the ambiguous
    symbol, a missing instance map is `RealizationRegionUnmapped`, a stored `rho` table on the
    thermal row gives an adjoint-exact coefficient JVP/VJP and the same residual as the
    closure; a `try_new_for_residual` (`k`) and a `try_new_for_output` (`joule_heat`'s
    `sigma`) closure refusing at one cell surface from residual, JVP and VJP as
    `FinitumError::InputEvaluation` with the producer's code, the untouched `Slot` /
    `ExpressionPath` origin, cell, point and time; bind-time refusals (missing output closure,
    a closure on a bound `Q`, an output-keyed closure on a one-instance plan) are typed, and
    a `/2` artifact of an implicit one-instance model is admitted by `composed` with a `/3`
    identity distinct from the `/2` plan's.
  - Honest limits: same-mesh binds only (cross-mesh `Transferred` chains are Krasis's, §8);
    an output is one cell point function with one bundle; a bound output's own closures see
    only the bound inputs that precede the output in dependency order; stored tables bind to
    rows only (an output kernel's non-basis inputs take closures); `system_constitutive_from_
    sources` and `FieldSampler::from_system_plan` stay per-model-symbol keyed; partial assembly
    (and the all-table agreement report) refuse on a plan with binds; interior/interface
    measures and the SC-W2 cases are unchanged (refused as before).

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
cargo test --locked --workspace --all-targets           # 214 passed, 0 failed across 31 binaries (W8 F-MI multi-instance plan, +9 integration, every pre-existing test unchanged; 205 at W8 F2 fallible callbacks, +5 unit +9 integration, every pre-existing test unchanged; 191 at W8 F1 field sampler, +13 unit +6 integration, every pre-existing test unchanged; 172 at W7 7c C typed representation refusal; 170 at W7 7c B proof-aware symmetry; 168 at W7 7c A per-plan quadrature; 164 at SC-W1 Scientia ids + typed inf-sup; 161 at SC-W1 system-path parity; 156 at W7 follow-ups; 153 at SC-W1 interface, 148 at SC-W1 ids/block actions, 144 at W7 package 3, 136 at W7 SV1-C1/C3 + P, 122 at the E6 close, 103 at SV2-B1 head fae5675, 52 at the R3D-era transcript)
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

2026-09-08 review validation: coordinator's full workspace/all-targets run exited 0,
including the final nine-test F-MI binary and all eleven one-instance parity tests.
The run began before the review edits and used the shared build cache; a separate focused
F-MI rerun and clippy/fmt/rustdoc checks validate the final edited tree before handoff.
No commit or push is included in this working-tree evidence.

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
   the real Stokes and mixed-Darcy corpus systems both solve. Per-block stored tables and the
   value-covering `finitum-system-operator/2` digest landed with SC-W1 system-path parity
   (2026-09-05); the plan digest became `finitum-system-realization/2` (quadrature rule) with
   W7 7c A, changing every operator digest value. Remaining in this area: regional (per-region) external tables, Hcurl
   realization, and interior-facet/DG measures;
5. FC3 `minimum_polynomial_degree`-honoring quadrature (the P1 mass-matrix
   under-integration follow-up recorded in GX-CONTRACTS C11.7/C11.8).
6. SC composition (design `sinbad/ARCHITECTURE.md` §8, §12). Landed by W7 (2026-09-03):
   prerequisite batch P (`4a6fe65`, state-dependent `SystemOperator` with GX-A3 tangents);
   SV1-C1/C3 global transposes and coefficient VJPs (`0f570c2`); the runtime inf-sup checker
   (`9c21b67`); SC-W1 system-level ids keying `BlockLayout` plus public per-block
   actions/transposes (`b477525`); and the SV2-B2 interface-measure realization binding Malleus
   facet-pair kernels (`d552da2`). Still open, in order of pull:
   - Done (W8 lane F-MI, 2026-09-07, item 9 below): the internal field tables are keyed by
     `SysVarId` and `SystemRealizationPlan::composed` admits a multi-instance group with its
     same-mesh `Composed` bind chains;
   - bridge Scientia's `InteriorFacet`/`Interface` factorizations (they already carry
     `MinusTrace`/`PlusTrace` inputs) into `InterfaceKernel`s so `bind_kernels` stops refusing
     them -- needs a driving `.res` case (SC-W2 CHT), plus `BoundChain::Composed`
     quadrature-point evaluation of a producer instance's output kernel;
   - SC-W2 `InterfaceRealization`/`ConnectionRealizationPlan` (elimination first) over the
     `InterfaceMeasure::between` machinery, multiplier/Nitsche and transfer beyond 1-D in SC-W3;
   - an H(div)-norm variant of the inf-sup estimate (the typed pairing from Scientia's
     `InfSup { pair, constrained, multiplier }` is consumed by `InfSupPairing::from_obligation`;
     wiring `require_inf_sup_stable` at run time is Sinbad's).
7. W8 lane F1 landed (2026-09-07): the public field sampler `finitum-field-sampler/1`
   (`FieldSampler`, `QuadratureRule`/`QuadratureView`, `SampledFamily`,
   `FinitumError::SamplingUnsupported`; see "Implemented"). Cross-repo needs / follow-ups:
   - Sinbad 7d-1 deletes `observable.rs`'s restated conventions against it: `SystemSampler::new`
     becomes one `FieldSampler::from_system_plan(plan, symbol, solution)` per `layout.blocks()`
     (the plan, not just the `BlockLayout`, carries the family), `sample(cell, reference,
     normal)` becomes `sample_at_reference` plus `exterior_facet` for the normal/measure, the
     private `cell_measure` / `facet_measure_and_outward_normal` / `reference_facet_centroid`
     become `cell_measure`, `exterior_facet(..).measure/normal/centroid`; `integrate_cells`
     picks `QuadratureView::of_system_plan(plan).rule_for_degree(d)` from the integrand degree
     (`dot(u, u)` of P1 is 2, of P2 is 4; the 17-linear-elasticity 22 % under-integration is
     exactly the barycenter-vs-degree-2 gap the unit test records) and the receipt records
     `rule.id`, `rule.degree` and the sampler digest. What Sinbad still needs beyond this lane:
     (a) a `SysVarId`-keyed constructor for composed multi-instance layouts (the F1 sampler is
     keyed by per-model `SymbolId`, as `SystemRealizationPlan` itself still is); (b) interior /
     interface traces for the SC-W2 heat-heat case (two-sided `InterfaceMeasure` sampling, not
     an exterior `FacetTrace`); (c) point location for point observables (`cell_contains` only
     answers membership); (d) facet quadrature rules above the centroid (the facet trace is
     exact at any point, but the exterior-facet functional rule is still the caller's).
   - Slice F2 (after Sinbad A1 lands; deliberately NOT in F1 because it is non-additive):
     fallible external-input and constitutive callbacks end to end (W8 decision 3). Proposed
     signatures: `pub struct InputEvaluationError { pub code: String, pub origin: InputOrigin,
     pub point: Option<Vec<f64>>, pub time: Option<f64>, pub message: String }` with
     `pub enum InputOrigin { Slot(String), ExpressionPath(String), Provider(String), Table(String) }`;
     `FieldSource::sampled(Fn(&[f64]) -> Result<Vec<f64>, InputEvaluationError>)`,
     `DynamicExternalInput::new(.., value: Fn(&PointEvaluation) -> Result<Vec<f64>,
     InputEvaluationError>, direction: Fn(&PointEvaluation, &PointEvaluation) -> Result<..>)`,
     `SystemConstitutiveInput::new(..)` likewise, `ExternalInput::sampled(.., FnMut(CellId,
     &[f64]) -> Result<Vec<f64>, InputEvaluationError>)`, and a carrying variant
     `FinitumError::InputEvaluation(InputEvaluationError)` that `residual` / JVP / VJP /
     `load_vector` / assembly return unchanged (code and origin preserved; never NaN). Changing
     the existing closure types is the non-additive step; a parallel `*_fallible` constructor
     pair would leave two paths and is not proposed. **Landed as item 8** with `try_` names
     and a named deletion slice (F3), because Sinbad A2 builds against the tree concurrently.
8. W8 lane F2 landed (2026-09-07): fallible callbacks (see "Implemented"). Follow-ups:
   - **Slice F3** (after Sinbad 7d-2 has migrated; a deletion, not a compatibility layer),
     the exact list: `DynamicExternalInput::new`, `SystemConstitutiveInput::new`,
     `ExternalInput::sampled`, `ExternalInput::sampled_on_facets`,
     `ExternalSensitivityInput::sampled` (the infallible wrappers; the `try_` forms stay as
     the only forms, no rename), `FieldSource::Sampled` and `FieldSource::sampled` (the
     time-blind infallible variant; `Fallible` stays -- Krasis `initial.rs` and Sinbad
     `run.rs` must match `Fallible` first), and the frozen-`t = 0` conveniences
     `external_inputs_from`, `essential_constraints_from`,
     `essential_constraints_from_selected`, `essential_constraints_from_system` (the `_at`
     forms stay; after 7d-2 no consumer samples at an implicit `t = 0`). Also delete then:
     the `sampler_error`-free `Nodal` refusal stays, the `SampledFieldFn` alias goes.
   - Cross-repo needs: (a) **Krasis K1** -- pass Methodus's typed `NumericError::Evaluation
     { code, origin, message }` through unchanged at its three
     `map_err(.. NumericError::Operator { message })` sites (`coupled.rs`, `coupled_system.rs`)
     so a failure raised inside `attempt_step_with` reaches the transaction outcome as a typed
     refusal rather than a rolled-back "non-finite" step; (b) **Sinbad 7d-2** -- switch
     `system_inputs.rs`'s `SystemConstitutiveInput::new` closures to `try_new` returning
     `InputEvaluationError::new(refusal.code, InputOrigin::Slot(slot) | ExpressionPath(origin),
     refusal.message)` (Finitum fills point / time / cell), then `src/evaluation_failure.rs`
     shrinks to reading `FinitumError::InputEvaluation` / `NumericError::Evaluation` (the
     `EvaluationFailureCell` and the `ClosureSite::fail` NaN placeholder go away);
     `system_inputs.rs`'s `stored_table` switches to `ExternalInput::try_sampled_at(.., time,
     ..)` with the step's time; Dirichlet data become `FieldSource::fallible(|x, t| ..)` and
     the transient path (`coupled_run.rs`) rebuilds `essential_constraints_from_system_at(..,
     time)` / `reduced(..)` per step so `g(t)` stops being frozen at `t = 0`;
     `derivative_campaign.rs` / `advanced.rs` / `artifacts.rs` table builders take the `try_`
     forms; `RUN_*` codes flow through unchanged, and Finitum's own
     `REALIZATION_PROPERTY_UNAVAILABLE` / `REALIZATION_TANGENT_UNAVAILABLE` join the refusal
     vocabulary (C12). (c) **Krasis K1 (addition)** -- `initial.rs` matches
     `FieldSource::Sampled` by payload; add a `FieldSource::Fallible(sampler)` arm evaluating
     `sampler(x, t0)` at the initial time and mapping the `InputEvaluationError` typed (today
     the wildcard arm refuses it as an unsupported source).
   - **P1 mass option on `SystemQuadrature::Barycenter` (recorded, not built; the F2 brief's
     item 6).** Need: a DAE structure with an unconstrained differential field (08 on 3x3)
     cannot take the migration rule under Krasis consistent initialization because the
     barycenter rule integrates the P1 mass `phi_i phi_j` (a degree-2 integrand) to a rank-one
     local block, so the differential rows are singular; Sinbad keeps `Richest` for any
     structure with an algebraic field row. Design: a *per-integral* rule, not a per-plan one.
     The mass term is the factorization integral whose active input carries
     `DerivativeEvaluation::TimeDerivative` against a value-kind test output; that integral
     alone is integrated on a second rule while every other integral (stiffness, sources,
     reactions) keeps the barycenter rule bitwise with the single-model default: **consistent**
     = the degree-2 rule (`PreparedElement::linear_simplex_with_degree(d, 2)`, exact for
     `phi_i phi_j`, the closed-form `|K| (1 + delta_ij) / ((d + 1)(d + 2))`); **lumped** = the
     vertex rule (points at the `d + 1` vertices, weights `|K| / (d + 1)`), exact for degree
     1, whose `phi_i(v_k) = delta_ik` gives the diagonal `|K| / (d + 1)` lumped mass through
     the same point kernels -- both are quadrature rules, so no lumping code path is needed.
     Shape: additive, `SystemQuadrature::BarycenterWithMass { mass: P1Mass::{Consistent,
     Lumped} }` beside `Barycenter` (`Richest` unchanged). Mechanism and cost: the "every
     Lagrange field's basis is tabulated at the same shared quadrature points" invariant
     becomes per-integral (`SystemOperator::quadrature` gains a `quadrature_for(residual,
     integral)`), the stored tables of the mass integral (`SystemExternalInput`,
     `ExternalInput::from_coefficient_at`) sample on that integral's rule, the plan digest
     `finitum-system-realization/2` records the per-integral rule and therefore moves to `/3`
     for the new variant only (`Barycenter` and `Richest` digests unchanged); the parity gate
     (bitwise with the single model) holds for `Barycenter` proper and, for the stiffness
     alone, for the new variant; tests: consistent and lumped P1 mass against the closed
     forms, stiffness bitwise with `Barycenter`, and the 08-style DAE's differential rows
     regular under consistent initialization (Krasis) -- about one lane-day.

9. W8 lane F-MI implemented in the working tree (2026-09-08; coordinator landing pending): the multi-instance `SystemRealizationPlan` (see
   "Implemented"). Cross-repo needs:
   - **Sinbad, G2 completion** (`run_plan` / `coupled_run.rs`): for a declared system whose
     instances share the level mesh, build ONE realization group instead of one per instance:
     keep the `SystemOperatorCompilation` (`compile_system` + `compile_system_operator`) the
     runner already recompiles from the frozen closure and call
     `SystemRealizationPlan::composed(&compilation, mesh, BlockLayout::new_keyed(
     system_ids.variables()...), quadrature)` with `SystemIdMap::from_scientia`; build the
     instance slot closures per residual with `SystemConstitutiveInput::try_new_for_residual
     (system_ids.residual(instance, equation), ..)` (`system_inputs.rs` today keys by
     equation name, which collides for two instances of one model) and stored tables with the
     residual as before; for every `SystemBind` on the provider-input path, `dual_closure`'s
     `DualContext` reads the bound symbol from `PointEvaluation::bound_values(consumer_symbol)`
     (value) / the direction point's `bound_values` (tangent) exactly as it reads an active
     input; for every producer output that a bind reads, build the output kernel's non-basis
     closures (`OutputKernels.factorization.integrals[0].primal.inputs`, the same
     `dual_closure` machinery over the producer model) with `try_new_for_output(InstanceId,
     OutputId, ..)`; on the kernel-input path bind nothing for the bound symbol (the plan
     refuses a closure on it). Dirichlet data: `SystemVariableEssentialConstraint { variable:
     system_ids.variable(instance, symbol), requirement, value }` with
     `essential_constraints_from_system_by_variable_at(operator, mesh, &[(instance,
     &instance_region_map), ..], .., time)` -- the per-instance `RegionMap`s the runner builds
     today, keyed by the instance. One `CoupledLeaf::reduced_system` over the composed
     operator (its `layout()` is already `SysVarId`-keyed, so `SemanticId = SysVarId` needs
     no re-keying); the N-leaf route stays for cross-group systems. Receipts: record the plan
     digest (`finitum-system-realization/3`), `SystemOperator::binds()` (slot, output, path,
     rows/columns, composition and JVP-composition digests) and the input dispositions per
     bound input as `Available` (state tangent through the chain), `realization_agreement` as
     the typed `RepresentationUnsupported { PartialAssembly }` refusal a bind chain earns
     (matrix-free / assembled / element assembly still agree and can be reported directly).
     Then the decision-9 comparison (residuals, state/rate JVPs with both cross blocks,
     consistent initialization, trajectories) against monolithic 08 is measurable; the
     operator-level half is `tests/w8_multi_instance_system.rs`.
   - **Scientia:** nothing. **Malleus:** nothing (the JVP compositions are validated and
     digest-checked; a composition whose independent set also covers the producer's frozen
     inputs would let Finitum execute the JVP composition instead of the two-kernel chain --
     recorded, not needed).

Extend method topology only from concrete acceptance cases, keeping
local-kernel meaning, backend policy, and realization identity explicit.
