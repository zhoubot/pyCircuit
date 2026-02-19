# Compiler Flow (v3.2)

## 1) Frontend authoring model

Frontend decorators define semantic intent:
- `@module`: hierarchy boundary (materializes callable symbol and `pyc.instance` at callsites)
- `@function`: inline helper
- `@template`: compile-time helper (no IR emission)

Inter-module ports are connector-only.

## 2) JIT elaboration

JIT parses and lowers Python AST into pyCircuit IR builders:
- Expression lowering for integer wires and control flow
- Module instancing and specialization
- Template execution for compile-time metaprogramming

Template call behavior:
1. Bind Python args/kwargs
2. Validate first arg is current `Circuit`
3. Snapshot module state
4. Execute template in Python
5. Validate zero-emission purity and return-type contract
6. Canonicalize/memoize result

## 3) Design assembly

`compile_design(...)` builds a multi-function MLIR module from specialized `@module` symbols.

## 4) MLIR pipeline

Typical flow:
- Frontend emits pyc/func/scf-level MLIR
- `pyc-compile` applies legality/normalization/fusion/inlining/statistics passes
- Emit split C++ or split Verilog artifacts by module

## 5) C++ / Verilog emission

- C++: split module artifacts with manifest-driven build support
- Verilog: split module files with deterministic ordering

## 6) v3.2 metaprogramming layer

`pycircuit.meta` lives fully in Python compile-time space:
- Interface/pipeline/parameter specs are immutable template values
- Specs are consumed by frontend helper APIs (`io_in`, `io_out`, `state_regs`, `pipe_regs`, `instance_bind`)
- Spec construction itself emits no IR
