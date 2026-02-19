# pyCircuit v3.2

pyCircuit v3.2 is a strict module/connector-first frontend with explicit compile-time metaprogramming.

Core contracts:
- `@module`: hierarchy boundary (non-inline by default)
- `@function`: inline hardware helper
- `@template`: compile-time pure helper (no IR emission)
- Inter-module connectivity: connectors only (`Connector`, `ConnectorBundle`, etc.)

v3.2 additions:
- Expanded compile-time arithmetic helpers in `pycircuit.ct`
- New `pycircuit.meta` package for immutable template specs and DSE spaces
- New `Circuit` grammar-candy helpers:
  - `io_in(...)`
  - `io_out(...)`
  - `state_regs(...)`
  - `pipe_regs(...)`
  - `instance_bind(...)`

Public frontend entrypoint:
- `pycircuit.compile_design(...)`

## Quickstart

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/pyc build
```

Emit MLIR:

```bash
PYTHONPATH=/Users/zhoubot/pyCircuit/compiler/frontend python3 -m pycircuit.cli emit \
  /Users/zhoubot/pyCircuit/designs/examples/template_pipeline_builder_demo.py \
  -o /tmp/template_pipeline_builder_demo.pyc
```

Compile split C++:

```bash
/Users/zhoubot/pyCircuit/compiler/mlir/build2/bin/pyc-compile /tmp/template_pipeline_builder_demo.pyc \
  --emit=cpp --out-dir /tmp/template_pipeline_builder_demo_cpp --cpp-split=module
```

## Main docs

- `/Users/zhoubot/pyCircuit/docs/USAGE.md`
- `/Users/zhoubot/pyCircuit/docs/TEMPLATE_METAPROGRAMMING.md`
- `/Users/zhoubot/pyCircuit/docs/META_STRUCTURES.md`
- `/Users/zhoubot/pyCircuit/docs/LINXCORE_DSE_GUIDE.md`
- `/Users/zhoubot/pyCircuit/docs/COMPILER_FLOW.md`

## Regressions

- `bash /Users/zhoubot/pyCircuit/flows/scripts/run_examples.sh`
- `bash /Users/zhoubot/pyCircuit/flows/tools/run_linx_cpu_pyc_cpp.sh`
- `bash /Users/zhoubot/pyCircuit/flows/tools/run_fastfwd_pyc_cpp.sh`
- `python3 /Users/zhoubot/pyCircuit/flows/tools/perf/run_perf_smoke.py`

## API hygiene

```bash
python3 /Users/zhoubot/pyCircuit/flows/tools/check_api_hygiene.py
```

Scan LinxCore from pyCircuit checker:

```bash
python3 /Users/zhoubot/pyCircuit/flows/tools/check_api_hygiene.py \
  --scan-root /Users/zhoubot/LinxCore src
```
