# Finitum status

Updated: 2026-09-01
Milestone: SV0-B3 checks + R3D/SV1-G0B geometry derivatives + SV2-A vector H1 elasticity +
SV2-B1/B4 P2 elements and mixed product layouts + E6 executable system realization
(Scientia-form-driven SystemOperator with load vector, equation-sign symmetry proof, and
H(div)/RT0 + P0 compatible realization — the real Stokes and mixed-Darcy corpus systems solve)

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
cargo test --locked --workspace --all-targets           # 103 passed, 0 failed (SV2-B1 head fae5675; 52 at the R3D-era transcript this block was written for)
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
git diff --check
python3 ../sinbad/scripts/check-physics-corpus.py        # 50 models
```

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
6. SC composition (design `sinbad/ARCHITECTURE.md` §8, §12; nothing landed).
   Prerequisite batch P: a state-dependent `SystemOperator` residual/JVP with the
   GX-A3 tangents (today the system path is linear — residual is `A·state`, no VJP)
   so transient nonlinear systems such as 08 can execute. SC-W1: `BlockLayout`,
   `RegionMap`, and `SystemEssentialConstraintRequirement` re-keyed to Scientia's
   system-level ids (same `u32` width), public per-(row, column) block actions and
   transposes, a per-instance receipt chain in `SystemRealizationPlan`, and
   quadrature-point evaluation of a producer instance's output kernel for
   `BoundChain::Composed`. SC-W2: `Interface`/`InteriorFacet` measure realization
   (refused today), `InterfaceRealization` (facet pairing, trace DOF maps,
   orientation), and `ConnectionRealizationPlan` with the elimination path first;
   multiplier/Nitsche and transfer beyond the 1-D `NonmatchingTransfer` in SC-W3.

Extend method topology only from concrete acceptance cases, keeping
local-kernel meaning, backend policy, and realization identity explicit.
