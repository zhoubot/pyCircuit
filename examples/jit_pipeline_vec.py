# -*- coding: utf-8 -*-
"""JIT pipeline with vector example using cycle-aware API.

Demonstrates multi-stage pipeline with signal packing and unpacking.
"""
from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    CycleAwareSignal,
    ca_cat,
    compile_cycle_aware,
    mux,
)


def _pipe_stage(
    domain: CycleAwareDomain,
    bus: CycleAwareSignal,
    stage_idx: int,
) -> CycleAwareSignal:
    """Create a single pipeline stage (register)."""
    domain.next()
    return domain.cycle(bus, reset_value=0, name="bus_stage")


def build(
    m: CycleAwareCircuit,
    domain: CycleAwareDomain,
    STAGES: int = 3,
) -> None:
    """Build a multi-stage pipeline with packed bus.
    
    Args:
        m: Cycle-aware circuit builder
        domain: Clock domain
        STAGES: Number of pipeline stages (JIT-time parameter)
    """
    # Cycle 0: Inputs
    a = domain.create_signal("a", width=16)
    b = domain.create_signal("b", width=16)
    sel = domain.create_signal("sel", width=1)
    
    # Combinational logic
    sum_ = a + b
    x = a ^ b
    
    # Conditional: data = sum if sel else x
    data = mux(sel, sum_, x)
    
    # Comparison
    tag = a.eq(b)  # 1-bit
    
    # Extract low 8 bits
    lo8 = data[0:8]
    
    # Pack into bus: {tag[0], data[15:0], lo8[7:0]} = 1+16+8 = 25 bits
    # Using ca_cat (MSB-first concatenation)
    bus = ca_cat(tag, data, lo8)
    bus = bus.named("packed_bus")
    
    # Pipeline stages (Python-unrolled)
    for i in range(STAGES):
        bus = _pipe_stage(domain, bus, i)
    
    # Unpack: extract fields from bus
    # Layout: tag (bit 24), data (bits 23:8), lo8 (bits 7:0)
    out_lo8 = bus[0:8]
    out_data = bus[8:24]
    out_tag = bus[24:25]
    
    # Outputs
    m.output("tag", out_tag.sig)
    m.output("data", out_data.sig)
    m.output("lo8", out_lo8.sig)


# Entry point for JIT compilation
if __name__ == "__main__":
    circuit = compile_cycle_aware(build, name="jit_pipeline_vec", STAGES=3)
    print(circuit.emit_mlir())
