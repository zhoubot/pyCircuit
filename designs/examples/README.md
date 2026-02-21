# Examples

This directory contains folderized pyCircuit examples.

## Layout contract

Each example case `X` is a folder:
- `X/X.py`: design (`@module build(...)`)
- `X/tb_X.py`: testbench (`@testbench def tb(...)`)
- `X/X_config.py`: default params + TB presets + `SIM_TIER`

## Smoke checks

Compiler smoke (`emit + pycc`):

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_examples.sh
```

Simulation smoke (strict normal-tier examples, C++ + Verilator):

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_sims.sh
```

Nightly simulation smoke (normal + heavy tiers):

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_sims_nightly.sh
```

## Artifact policy

Generated artifacts are local-only and written under:
- `.pycircuit_out/`

They are intentionally not checked into git.

## Linx/board-related designs

Linx CPU / LinxCore / board bring-up examples are kept under `contrib/` and are
not part of the core example smoke gates.
