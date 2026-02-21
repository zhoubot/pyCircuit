from __future__ import annotations

import sys
from pathlib import Path

from pycircuit import Tb, compile, testbench

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from interface_wiring import build  # noqa: E402
from interface_wiring_config import DEFAULT_PARAMS, TB_PRESETS  # noqa: E402


@testbench
def tb(t: Tb) -> None:
    p = TB_PRESETS["smoke"]
    t.timeout(int(p["timeout"]))
    t.drive("top_in_left", 1, at=0)
    t.drive("top_in_rhs", 2, at=0)
    t.expect("top_out_left", 1, at=0)
    t.expect("top_out_rhs", 3, at=0)
    t.finish(at=int(p["finish"]))


if __name__ == "__main__":
    print(compile(build, name="tb_interface_wiring_top", **DEFAULT_PARAMS).emit_mlir())
