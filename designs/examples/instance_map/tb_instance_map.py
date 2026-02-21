from __future__ import annotations

import sys
from pathlib import Path

from pycircuit import Tb, compile, testbench

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from instance_map import build  # noqa: E402
from instance_map_config import DEFAULT_PARAMS, TB_PRESETS  # noqa: E402


@testbench
def tb(t: Tb) -> None:
    p = TB_PRESETS["smoke"]
    t.timeout(int(p["timeout"]))
    t.drive("in_alu", 0, at=0)
    t.drive("in_branch", 0, at=0)
    t.drive("in_lsu", 0, at=0)
    t.expect("alu_y", 1, at=0)
    t.expect("branch_y", 2, at=0)
    t.expect("lsu_y", 3, at=0)
    t.expect("acc", 6, at=0)
    t.finish(at=int(p["finish"]))


if __name__ == "__main__":
    print(compile(build, name="tb_instance_map_top", **DEFAULT_PARAMS).emit_mlir())
