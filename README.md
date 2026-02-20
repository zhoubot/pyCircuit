# pyCircuit

pyCircuit is a strict Python frontend for generating hardware from a small MLIR
dialect ("PYC") and emitting:
- Verilog netlists
- a cycle-accurate C++ model

## Model

- `@module`: hierarchy boundary
- `@function`: inline hardware helper
- `@const`: compile-time pure helper (no IR emission, no module mutation)
- `@testbench`: host-side cycle test program carried as a `.pyc` payload

Compile-time + wiring helpers:
- `pycircuit.spec` (compile-time shapes/params)
- `pycircuit.wiring` (binding/unflatten helpers)
- `pycircuit.logic` (combinational helpers)
- `pycircuit.lib` (standard blocks/signatures)

## Quickstart

Build the backend tool (`pycc`):

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/pyc build
```

Run compiler smoke:

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_examples.sh
```

Run simulation smoke (Verilator + `@testbench`):

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_sims.sh
```

## Main docs

- `/Users/zhoubot/pyCircuit/docs/QUICKSTART.md`
- `/Users/zhoubot/pyCircuit/docs/FRONTEND_API.md`
- `/Users/zhoubot/pyCircuit/docs/PIPELINE.md`
- `/Users/zhoubot/pyCircuit/docs/TESTBENCH.md`
- `/Users/zhoubot/pyCircuit/docs/CONST_METAPROGRAMMING.md`
- `/Users/zhoubot/pyCircuit/docs/SPEC_STRUCTURES.md`
- `/Users/zhoubot/pyCircuit/docs/SPEC_COLLECTIONS.md`
- `/Users/zhoubot/pyCircuit/docs/IR_SPEC.md`
- `/Users/zhoubot/pyCircuit/docs/PRIMITIVES.md`
- `/Users/zhoubot/pyCircuit/docs/DIAGNOSTICS.md`

## Regressions

- `bash /Users/zhoubot/pyCircuit/flows/scripts/run_examples.sh`
- `bash /Users/zhoubot/pyCircuit/flows/scripts/run_sims.sh`

