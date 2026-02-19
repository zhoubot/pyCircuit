# pyCircuit v3.1

pyCircuit v3.1 is a strict module/connector-first hardware frontend and MLIR compiler flow.

Design intent:
- Scale to very large designs without single-file C++/Verilog emission bottlenecks.
- Preserve hierarchy by default with explicit module boundaries.
- Support compile-time Python template metaprogramming with zero emitted IR side effects.

## Core authoring model

- `@module`: hierarchy boundary, default non-inline.
- `@function`: inline hardware helper.
- `@template`: compile-time helper, must be pure, emits no IR.
- Inter-module links use connectors (`Connector`, `WireConnector`, `RegConnector`, `ConnectorBundle`).

Public frontend entrypoint:
- `pycircuit.compile_design(...)`

Removed from public API:
- inline alias decorator (removed)
- public compile alias (removed)

## Repository layout

- Frontend: `/Users/zhoubot/pyCircuit/compiler/frontend/pycircuit`
- MLIR compiler: `/Users/zhoubot/pyCircuit/compiler/mlir`
- Runtime: `/Users/zhoubot/pyCircuit/runtime/cpp`, `/Users/zhoubot/pyCircuit/runtime/verilog`
- Flows/tools: `/Users/zhoubot/pyCircuit/flows`
- Examples: `/Users/zhoubot/pyCircuit/designs/examples`

## Quickstart

1. Build compiler tools:

```bash
bash flows/scripts/pyc build
```

2. Emit MLIR from a v3.1 design:

```bash
PYTHONPATH=compiler/frontend python3 -m pycircuit.cli emit \
  designs/examples/template_arith_demo.py \
  -o /tmp/template_arith_demo.pyc
```

3. Compile with split outputs (recommended):

```bash
build/bin/pyc-compile /tmp/template_arith_demo.pyc \
  --emit=cpp --out-dir /tmp/template_arith_demo_cpp --cpp-split=module

build/bin/pyc-compile /tmp/template_arith_demo.pyc \
  --emit=verilog --out-dir /tmp/template_arith_demo_v
```

## Main docs

- `/Users/zhoubot/pyCircuit/docs/USAGE.md`
- `/Users/zhoubot/pyCircuit/docs/TEMPLATE_METAPROGRAMMING.md`
- `/Users/zhoubot/pyCircuit/docs/COMPILER_FLOW.md`
- `/Users/zhoubot/pyCircuit/docs/IR_SPEC.md`
- `/Users/zhoubot/pyCircuit/docs/PRIMITIVES.md`

## Regressions

- `bash /Users/zhoubot/pyCircuit/flows/scripts/run_examples.sh`
- `bash /Users/zhoubot/pyCircuit/flows/tools/run_linx_cpu_pyc_cpp.sh`
- `bash /Users/zhoubot/pyCircuit/flows/tools/run_fastfwd_pyc_cpp.sh`
- `python3 /Users/zhoubot/pyCircuit/flows/tools/perf/run_perf_smoke.py`

## API hygiene check

```bash
python3 /Users/zhoubot/pyCircuit/flows/tools/check_api_hygiene.py
```
