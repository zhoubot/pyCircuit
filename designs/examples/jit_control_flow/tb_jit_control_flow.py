from __future__ import annotations

import sys
from pathlib import Path

from pycircuit import Tb, compile, testbench

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from jit_control_flow import build  # noqa: E402
from jit_control_flow_config import DEFAULT_PARAMS, TB_PRESETS  # noqa: E402


@testbench
def tb(t: Tb) -> None:
    p = TB_PRESETS["smoke"]
    t.timeout(int(p["timeout"]))
    t.drive("a", 1, at=0)
    t.drive("b", 2, at=0)
    t.drive("op", 0, at=0)
    t.expect("result", 7, at=0)
    t.finish(at=int(p["finish"]))


if __name__ == "__main__":
    print(compile(build, name="tb_jit_control_flow_top", **DEFAULT_PARAMS).emit_mlir())
