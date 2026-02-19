# Template Metaprogramming in pyCircuit v3.1

`@template` adds explicit compile-time Python metaprogramming to the JIT frontend.

## 1) What `@template` is for

Use templates to compute:
- widths
- loop bounds
- cache/memory parameters
- masks and alignment values
- specialization constants

Template code runs during JIT and must not generate hardware IR.

## 2) Contract

Declare with:

```python
@template
def my_template(m: Circuit, ...):
    ...
```

Rules:
- First argument must be the current `Circuit` builder (`m`).
- Calls must be explicit (`my_template(m, ...)`).
- Template body must be compile-time pure.
- Any IR-emitting/mutation side effect is a hard error.

Purity is conservatively checked using module-state snapshots of:
- `m._lines`
- `m._next_tmp`
- `m._args`
- `m._results`

If changed after template evaluation, compilation fails.

## 3) Allowed return values

Allowed:
- `None`
- `bool`
- `int`
- `str`
- `LiteralValue`
- `tuple`/`list`/`dict` of allowed values

Disallowed (hard error):
- `Wire`
- `Reg`
- `Signal`
- `Connector` / `ConnectorBundle`
- `Module` / `Design`
- arbitrary objects

## 4) Compile-time math helpers (`ct`)

`from pycircuit import ct`

- `ct.clog2(n)`
- `ct.flog2(n)`
- `ct.div_ceil(a, b)`
- `ct.align_up(v, a)`
- `ct.pow2_ceil(n)`
- `ct.bitmask(width)`

## 5) Example

```python
from pycircuit import Circuit, ct, module, template

@template
def cfg(m: Circuit, sets: int, line_bytes: int):
    _ = m
    return {
        "sets": max(1, int(sets)),
        "line_bytes": ct.pow2_ceil(max(1, int(line_bytes))),
        "index_bits": ct.clog2(max(1, int(sets))),
    }

@module
def build(m: Circuit, sets: int = 64, line_bytes: int = 64):
    c = cfg(m, sets, line_bytes)
    addr = m.input("addr", width=40)
    m.output("index", addr[ct.clog2(c["line_bytes"]):ct.clog2(c["line_bytes"]) + c["index_bits"]])
```

## 6) Common failure modes

- Template performs wire arithmetic (`a + b`) and emits IR.
- Template calls mutating builder APIs (`m.output`, `m.instance`, `m.out`, etc.).
- Template returns hardware objects instead of compile-time values.
- Template called without passing current builder (`m`).

All of these fail compilation by design.

## 7) Practical patterns

- Keep templates deterministic and side-effect free.
- Use templates to normalize or validate user parameters.
- Use `ct` helpers instead of ad-hoc duplicated arithmetic.
- Keep hardware construction in `@module` / `@function` only.
