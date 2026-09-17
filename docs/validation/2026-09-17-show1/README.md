# SHOW-1 owner validation

Finitum exports validated simplex geometry and P1 cell-vertex field values through
its existing FieldSampler. Saved values retain component count, location, basis
schema and mesh identity. Other bases refuse explicitly. Surface/cut geometry
carries owner reference coordinates; sampling uses the same simplex basis.
Cut caps are display geometry and never physical boundary conditions.

The full owner gate passes 245 tests across 34 targets. New tests compare scalar
and vector samples on 2D/3D meshes and cut surfaces against the original sampler,
including cuts through mesh vertices; mismatched identities, invalid extents,
nonfinite samples and unsupported bases refuse. Formatting, strict all-target/
all-feature clippy, strict rustdoc and doctests pass. Every gate verifies unchanged
source hashes. Commands and hashes: checks.json; raw full output: full.log.

This validates inspection primitives, not a new discretization, high-order viewer,
point locator, physical extrema search or an independent numerical reference.
