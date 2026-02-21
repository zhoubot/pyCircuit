from __future__ import annotations

import sys
from pathlib import Path

from pycircuit import Tb, compile, testbench

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from struct_transform import build  # noqa: E402
from struct_transform_config import DEFAULT_PARAMS, TB_PRESETS  # noqa: E402


@testbench
def tb(t: Tb) -> None:
    p = TB_PRESETS["smoke"]
    t.clock("clk")
    t.reset("rst", cycles_asserted=2, cycles_deasserted=1)
    t.timeout(int(p["timeout"]))
    t.drive("in_u_hdr_op", 1, at=0)
    t.drive("in_u_hdr_dst", 2, at=0)
    t.drive("in_u_payload_word", 3, at=0)
    t.drive("in_u_ctrl_valid", 1, at=0)
    t.expect("out_u_ctrl_valid", 1, at=0)
    t.expect("out_u_payload_word", 5, at=0)
    t.finish(at=int(p["finish"]))


if __name__ == "__main__":
    print(compile(build, name="tb_struct_transform_top", **DEFAULT_PARAMS).emit_mlir())
