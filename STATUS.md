# Finitum status

Updated: 2026-08-24
Milestone: SV0-B3 checks + R3D/SV1-G0B geometry derivatives + SV2-A vector H1 elasticity

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
cargo test --locked --workspace --all-targets           # 52 passed, 0 failed
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

## Next

Krasis SV0-B4, Sinbad SV0-B5/R4, and Sinbad D4 now consume these landed
providers. R3D/SV1-G0B and SV2-A are complete. SV2-B production Stokes is the
next named realization gap; extend method topology only from its concrete D5
acceptance case, keeping local-kernel meaning, backend policy, and realization
identity explicit.
