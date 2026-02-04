# `pyc/mlir`: MLIR dialect + tools (prototype)

This folder contains the MLIR-based implementation of the `pyc` dialect, along with:

- `pyc-opt`: `mlir-opt`-style tool with `pyc` dialect + passes
- `pyc-compile`: compile `.pyc` (MLIR) to Verilog or C++ via template libraries

## Build

Recommended: build from the repo root via top-level `CMakeLists.txt` (see `README.md`).

You can also build this subproject standalone if you already have an LLVM+MLIR build/install.

This example assumes an existing `~/llvm-project/build-mlir` containing MLIR.

```bash
cmake -G Ninja -S pyc/mlir -B pyc/mlir/build \
  -DMLIR_DIR=$HOME/llvm-project/build-mlir/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build-mlir/lib/cmake/llvm

ninja -C pyc/mlir/build pyc-opt pyc-compile
```

## Passes (prototype)

### `pyc-fuse-comb`

Fuses consecutive pure combinational ops (`pyc.add/mux/and/or/xor/not/constant`) into
`pyc.comb` regions. This is a codegen-oriented transform intended to enable:

- flattened SystemVerilog emission (SV `assign` instead of many tiny module instantiations)
- inlined C++ combinational evaluation (fewer tiny objects / calls)

`pyc-compile` runs this pass by default before emission.
