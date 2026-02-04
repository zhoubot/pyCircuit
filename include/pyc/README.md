# `include/pyc/`

Backend “template libraries” used by code generators:

- `include/pyc/cpp/`: cycle-accurate C++ models (header-only, template-heavy)
- `include/pyc/verilog/`: Verilog/SystemVerilog primitives used by emitted RTL

Generated code should only need to include/instantiate these templates.

## Primitive API (prototype)

The intent is for each primitive to exist in both backends with the same name and
port names (e.g. `pyc_add` / `pyc_reg` / `pyc_fifo`), so MLIR lowering and codegen
can stay backend-agnostic.

Examples:

- Verilog: `include/pyc/verilog/pyc_add.sv` defines `module pyc_add #(WIDTH) (a, b, y)`
- C++: `include/pyc/cpp/pyc_primitives.hpp` defines `template<unsigned Width> struct pyc::cpp::pyc_add { a, b, y; eval(); }`

Current checked-in primitives (prototype):

- Combinational: `pyc_add`, `pyc_mux`, `pyc_and`, `pyc_or`, `pyc_xor`, `pyc_not`
- Sequential: `pyc_reg`
- Ready/valid: `pyc_fifo` (strict handshake, single-clock)
- Memory: `pyc_byte_mem` (byte-addressed, async read + sync write, prototype)

Debug/testbench helpers (C++ only):

- `include/pyc/cpp/pyc_print.hpp`: `operator<<` for wires and primitives
- `include/pyc/cpp/pyc_tb.hpp`: small multi-clock-capable testbench harness
- `include/pyc/cpp/pyc_vcd.hpp`: tiny VCD writer (waveform dumping)
- Convenience include: `include/pyc/cpp/pyc_debug.hpp`
