# pyCircuit v3.1 Usage

This is the canonical frontend authoring guide for v3.1.

## 1) Required authoring contracts

- Top entrypoint must be:

```python
@module
def build(m: Circuit, ...):
    ...
```

- Helper calls in design/JIT context must be explicitly marked:
  - `@module` for hierarchy boundaries
  - `@function` for inline hardware helpers
  - `@template` for compile-time pure helpers

- Inter-module boundaries must use connectors. Raw `Wire`/`Reg` across module boundaries are rejected.

## 2) Decorators

## `@module`

- Creates a hierarchy boundary.
- Lowered as callable module symbols + `pyc.instance` at call sites.
- Default non-inline.
- Supports specialization parameters and multiple instances.

## `@function`

- Inline helper for hardware logic.
- Lowered in frontend/MLIR call path and inlined by pipeline rules.
- Use for reusable hardware expressions/mini-stages where hierarchy is not needed.

## `@template`

- Compile-time metaprogramming helper.
- Evaluated during JIT.
- Must be pure: cannot emit IR or mutate module interfaces.
- Allowed returns: `None`, `bool`, `int`, `str`, `LiteralValue`, and container compositions of these.
- Disallowed returns: `Wire`, `Reg`, `Signal`, connectors, modules, arbitrary objects.

See `/Users/zhoubot/pyCircuit/docs/TEMPLATE_METAPROGRAMMING.md` for details.

## 3) Connectors

Main types:
- `Connector`
- `WireConnector`
- `RegConnector`
- `ConnectorBundle`
- `ModuleInstanceHandle`

Common APIs on `Circuit`:
- `m.input_connector(...)`
- `m.output_connector(...)`
- `m.reg_connector(...)`
- `m.bundle_connector(...)`
- `m.as_connector(value, name=...)`
- `m.connect(dst, src, when=...)`

Instantiation:
- `m.instance(fn, name=..., params=..., port_name=connector, ...)`
- Returns a connector for single-output modules or a `ConnectorBundle` for multi-output modules.

## 4) Compile-time arithmetic helpers (`ct`)

Use `/Users/zhoubot/pyCircuit/compiler/frontend/pycircuit/ct.py` via `from pycircuit import ct`.

Available helpers:
- `ct.clog2(n)`
- `ct.flog2(n)`
- `ct.div_ceil(a, b)`
- `ct.align_up(v, a)`
- `ct.pow2_ceil(n)`
- `ct.bitmask(width)`

## 5) Minimal example

```python
from pycircuit import Circuit, compile_design, ct, module, function, template, u

@template
def acc_width(m: Circuit, lanes: int, width: int) -> int:
    _ = m
    return int(width) + ct.clog2(max(1, int(lanes)))

@function
def add_sat(m: Circuit, a, b):
    s = a + b
    return s[0:a.width]

@module
def build(m: Circuit, lanes: int = 4, width: int = 16):
    w = acc_width(m, lanes, width)
    a = m.input("a", width=w)
    b = m.input("b", width=w)
    y = add_sat(m, a, b)
    m.output("y", y)

if __name__ == "__main__":
    print(compile_design(build, name="demo", lanes=4, width=16).emit_mlir())
```

## 6) CLI flow

Emit MLIR:

```bash
PYTHONPATH=compiler/frontend python3 -m pycircuit.cli emit \
  designs/examples/template_arith_demo.py \
  -o /tmp/template_arith_demo.pyc
```

Compile split C++:

```bash
build/bin/pyc-compile /tmp/template_arith_demo.pyc \
  --emit=cpp --out-dir /tmp/template_arith_demo_cpp --cpp-split=module
```

Compile split Verilog:

```bash
build/bin/pyc-compile /tmp/template_arith_demo.pyc \
  --emit=verilog --out-dir /tmp/template_arith_demo_v
```

## 7) High-level blocks

Public blocks in v3+ surface:
- `FIFO`
- `Queue`
- `IssueQueue`
- `Picker`
- `Mem2Port`
- `SRAM`
- `RegFile`
- `Cache`

Use these as composite `@module` building blocks with connectors between instances.

## 8) Fresh-start policy

v3.1 is a hard break. Compatibility/migration shims are intentionally removed.

Not supported:
- inline alias decorator (removed)
- public compile alias (removed)
- non-JIT build fallback paths (removed)

Use `python3 /Users/zhoubot/pyCircuit/flows/tools/check_api_hygiene.py` to enforce repository-wide API hygiene.
