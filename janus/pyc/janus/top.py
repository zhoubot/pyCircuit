from __future__ import annotations

from pycircuit import Circuit

from janus.bcc.ooo.core import build_bcc_ooo


def build(m: Circuit, *, mem_bytes: int = (1 << 20)) -> None:
    """Janus top-level bring-up entry.

    Current bring-up integrates the BCC OOO core as the executable datapath.
    TMU/TMA/CUBE/TAU blocks are factored as pyCircuit module files and can be
    incrementally wired into this top-level as those blocks gain execution
    semantics.
    """

    build_bcc_ooo(m, mem_bytes=mem_bytes)


build.__pycircuit_name__ = "janus_top_pyc"
