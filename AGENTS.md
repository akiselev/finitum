# Agent instructions

Finitum owns concrete discretization and global operator realization. It may depend on
Scientia scientific compiler artifacts and Malleus executable kernels. Add Methodus only when
a real realized operator can implement its traits directly. Stateful nonlinear, block,
and DAE composition belongs to Krasis.

Do not add scientific parsing, weak-form meaning, kernel scheduling/code generation,
coupled history state, or solver algorithms. Do not create `finitum-core`, contract, bridge,
or ABI crates.

Run formatting, clippy with warnings denied, and all tests before handoff. Keep `STATUS.md`
compact and current.
