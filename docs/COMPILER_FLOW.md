# pyCircuit v3.1 Compiler Flow

This document describes the current strict frontend + MLIR flow.

## 1) End-to-end pipeline

1. Python frontend compiles `@module` top entry via JIT to `.pyc` MLIR.
2. `pyc-compile` runs legality/optimization passes.
3. Emitters write split Verilog/C++ artifacts (per-module by default in out-dir mode).
4. Flow scripts build and run generated simulations/regressions.

## 2) Frontend semantics

Top entry contract:
- `@module def build(m: Circuit, ...)`
- no non-JIT fallback path

Call-kind semantics:
- `@module`: hierarchy boundary, lowered as instances
- `@function`: inline helper
- `@template`: compile-time helper, no IR emission allowed

Inter-module ports:
- Must be connector-based (`Connector` family)
- Raw cross-module `Wire/Reg` connections are rejected

## 3) Template call path

When JIT sees a call to a `@template` function:
- Evaluates the call in Python at compile time.
- Validates purity by checking module state snapshots (`_lines`, `_next_tmp`, `_args`, `_results`).
- Raises hard error if template emitted IR or mutated module interface.
- Enforces template return contract (primitive/allowed container values only).
- Memoizes per-compile call results by function identity + canonicalized arguments.

Guarantee:
- Template logic contributes no MLIR ops and therefore no C++/Verilog emission.

## 4) CLI entrypoints

Frontend emit:

```bash
PYTHONPATH=compiler/frontend python3 -m pycircuit.cli emit <design.py> -o <out.pyc>
```

Backend compile:

```bash
build/bin/pyc-compile <out.pyc> --emit=cpp --out-dir <dir> --cpp-split=module
build/bin/pyc-compile <out.pyc> --emit=verilog --out-dir <dir>
```

## 5) Default compile behavior

Key defaults for large designs:
- Split C++/Verilog out-dir emission.
- Per-module C++ artifacts plus manifest (`cpp_compile_manifest.json`).
- Shard support for oversized generated modules.

Common options:
- `--emit=cpp|verilog`
- `--out-dir=<dir>`
- `--cpp-split=module|none`
- `--cpp-shard-threshold-lines=<N>`
- `--cpp-shard-threshold-bytes=<N>`
- `--logic-depth=<N>`
- `--sim-mode=default|cpp-only`
- `--emit-structural=auto|on|off`

## 6) Hygiene and regression gates

- API hygiene: `python3 flows/tools/check_api_hygiene.py`
- Example sweep: `bash flows/scripts/run_examples.sh`
- Linx CPU C++ flow: `bash flows/tools/run_linx_cpu_pyc_cpp.sh`
- FastFwd C++ flow: `bash flows/tools/run_fastfwd_pyc_cpp.sh`
- Perf smoke: `python3 flows/tools/perf/run_perf_smoke.py`

## 7) Fresh-start scope

v3.1 intentionally removes migration-era surfaces from frontend and flow tooling.
Use only current APIs and connector/template semantics.
