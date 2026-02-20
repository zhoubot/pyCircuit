# Quickstart

## 1) Build the backend tool (`pycc`)

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/pyc build
```

## 2) Run compiler smoke (emit + pycc)

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_examples.sh
```

## 3) Run simulation smoke (Verilator + `@testbench`)

```bash
bash /Users/zhoubot/pyCircuit/flows/scripts/run_sims.sh
```

## 4) Minimal manual flow

Emit one module:

```bash
PYTHONPATH=/Users/zhoubot/pyCircuit/compiler/frontend \
python3 -m pycircuit.cli emit /Users/zhoubot/pyCircuit/designs/examples/counter.py -o /tmp/counter.pyc
```

Compile to C++:

```bash
/Users/zhoubot/pyCircuit/compiler/mlir/build2/bin/pycc /tmp/counter.pyc --emit=cpp --out-dir /tmp/counter_cpp
```

Build a multi-module project with a testbench:

```bash
PYTHONPATH=/Users/zhoubot/pyCircuit/compiler/frontend \
python3 -m pycircuit.cli build \
  /Users/zhoubot/pyCircuit/designs/examples/counter_tb.py \
  --out-dir /tmp/counter_build \
  --target both \
  --jobs 8
```

