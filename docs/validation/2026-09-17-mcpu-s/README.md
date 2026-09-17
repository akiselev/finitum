# M-CPU-S owner validation — September 17, 2026

This isolated candidate reuses the existing cell-local zero-point JVP matrices
for `SystemOperator::assemble` and, consequently, its numerical symmetry proof.
Element assembly allocates its global work vectors once and clears touched DOFs.
Cell contributions accumulate in cell order before CSR construction. No new
backend, scientific model, schema, solver policy or proof cache is introduced.

All six focused cases pass: variable-coefficient scalar Poisson and nonlinear
heat under both quadrature rules, mixed Taylor–Hood Stokes, 3D vector elasticity,
composed electrothermal, exterior-facet Darcy, and an empty mesh. Every matrix
column agrees with exhaustive global-action probing at the existing test
precision, and numerical symmetry classifications agree. The scalar test covers
two mesh sizes. Existing owner tests retain derivative, constraint, prescribed
motion, transpose, representation, identity and failure coverage.

Full owner validation passed 243 tests across 33 targets, zero failures or
ignored tests. Formatting, all-target/all-feature Clippy with warnings denied,
doctests, and rustdoc with warnings denied also passed. The same candidate files
were hashed before and after every check; their bytes were unchanged. Commands,
source hashes, toolchain and sibling heads are in [checks.json](checks.json).
Raw test output is in [focused.log](focused.log) and [full.log](full.log).

The mixed-case input-evaluation count fell from 2,832 to 720. Focused wall times
are diagnostic samples under concurrent host load, not a reproducible full-case
speedup claim. The unchanged Sinbad full suite is running in an isolated consumer
checkout; its complete elasticity, derivative, mutation and oracle results are
still required before integration. W8 remains open on the primary source tuple.

Exterior-facet and empty-mesh operators keep exhaustive probing. Reduced and
linearized operator assembly is unchanged. Symmetry tolerance, the 4096-DOF proof
limit, generated kernels, quadrature, constraints and operator identity are
unchanged. Broad M-CPU work and native scalar execution remain deferred.
