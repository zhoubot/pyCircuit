from __future__ import annotations

from pycircuit import CycleAwareCircuit, CycleAwareReg, CycleAwareSignal


def build_if_stage(
    m: CycleAwareCircuit,
    *,
    do_if: CycleAwareSignal,
    ifid_window: CycleAwareReg,
    mem_rdata: CycleAwareSignal,
) -> None:
    # IF stage: latch instruction window.
    ifid_window.set(mem_rdata, when=do_if)
