# Examples

This directory contains small frontend demos and a few larger reference designs.

## Main demos

- `counter.py`: minimal register + output
- `counter_tb.py`: `@testbench` example for `counter.py`
- `fifo_loopback.py`: ready/valid FIFO loopback
- `wire_ops.py`: core wire/arithmetic ops
- `jit_control_flow.py`: static `if/for` lowering
- `jit_pipeline_vec.py`: staged pipeline with vectors
- `hier_modules.py`: multi-module hierarchy/instantiation
- `template_struct_transform_demo.py`: immutable struct-transform metaprogramming
- `template_module_collection_demo.py`: module-family vector elaboration
- `template_instance_map_demo.py`: keyed module-map/module-dict elaboration
- `issue_queue_2picker.py`: small issue-queue-like example

## Smoke checks

Compiler smoke (emit + pycc):

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_examples.sh
```

Simulation smoke (Verilator + `@testbench`):

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_sims.sh
```

## Artifact policy

Generated artifacts are local-only and written under:
- `.pycircuit_out/`

They are intentionally not checked into git.

## Linx/board-related designs

Linx CPU / LinxCore / board bring-up examples are kept under `contrib/` and are
not part of the core example smoke gates.

