# -*- coding: utf-8 -*-
"""Issue queue with 2 pickers using cycle-aware API.

Demonstrates a 4-entry issue queue with 2 output ports.
"""
from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    CycleAwareSignal,
    compile_cycle_aware,
    mux,
)


def build(m: CycleAwareCircuit, domain: CycleAwareDomain) -> None:
    """Build a 4-entry issue queue with 2 output ports.
    
    Args:
        m: Cycle-aware circuit builder
        domain: Clock domain
    """
    # Cycle 0: Inputs
    in_valid = domain.create_signal("in_valid", width=1)
    in_data = domain.create_signal("in_data", width=8)
    out0_ready = domain.create_signal("out0_ready", width=1)
    out1_ready = domain.create_signal("out1_ready", width=1)
    
    # Initial state (constants representing initial values)
    zero1 = domain.create_const(0, width=1)
    zero8 = domain.create_const(0, width=8)
    
    # State registers - use create_const for initial values, will be updated via cycle()
    # These represent "previous cycle" values that we'll use to compute next values
    val0_prev = domain.create_const(0, width=1, name="val0_init")
    val1_prev = domain.create_const(0, width=1, name="val1_init")
    val2_prev = domain.create_const(0, width=1, name="val2_init")
    val3_prev = domain.create_const(0, width=1, name="val3_init")
    data0_prev = domain.create_const(0, width=8, name="data0_init")
    data1_prev = domain.create_const(0, width=8, name="data1_init")
    data2_prev = domain.create_const(0, width=8, name="data2_init")
    data3_prev = domain.create_const(0, width=8, name="data3_init")
    
    # Read current values (same as previous for combinational logic)
    out0_valid = val0_prev
    out0_data = data0_prev
    out1_valid = val1_prev
    out1_data = data1_prev
    
    # Pop control signals
    pop0 = out0_valid & out0_ready
    pop1 = out1_valid & out1_ready & pop0  # out1 can only pop if out0 also pops
    
    # Back-pressure: ready when not full or will pop
    in_ready = (~val3_prev) | pop0
    push = in_valid & in_ready
    
    # Shift stage 1: after pop0 (shift all entries down by 1)
    s1_v0 = val1_prev
    s1_v1 = val2_prev
    s1_v2 = val3_prev
    s1_v3 = zero1
    s1_d0 = data1_prev
    s1_d1 = data2_prev
    s1_d2 = data3_prev
    s1_d3 = data3_prev
    
    # Apply pop0 shift
    a1_v0 = mux(pop0, s1_v0, val0_prev)
    a1_v1 = mux(pop0, s1_v1, val1_prev)
    a1_v2 = mux(pop0, s1_v2, val2_prev)
    a1_v3 = mux(pop0, s1_v3, val3_prev)
    a1_d0 = mux(pop0, s1_d0, data0_prev)
    a1_d1 = mux(pop0, s1_d1, data1_prev)
    a1_d2 = mux(pop0, s1_d2, data2_prev)
    a1_d3 = mux(pop0, s1_d3, data3_prev)
    
    # Shift stage 2: after pop1 (shift entries down by 1 again)
    s2_v0 = a1_v1
    s2_v1 = a1_v2
    s2_v2 = a1_v3
    s2_v3 = zero1
    s2_d0 = a1_d1
    s2_d1 = a1_d2
    s2_d2 = a1_d3
    s2_d3 = a1_d3
    
    # Apply pop1 shift
    a2_v0 = mux(pop1, s2_v0, a1_v0)
    a2_v1 = mux(pop1, s2_v1, a1_v1)
    a2_v2 = mux(pop1, s2_v2, a1_v2)
    a2_v3 = mux(pop1, s2_v3, a1_v3)
    a2_d0 = mux(pop1, s2_d0, a1_d0)
    a2_d1 = mux(pop1, s2_d1, a1_d1)
    a2_d2 = mux(pop1, s2_d2, a1_d2)
    a2_d3 = mux(pop1, s2_d3, a1_d3)
    
    # Push enable signals (find first empty slot)
    en0 = push & (~a2_v0)
    en1 = push & a2_v0 & (~a2_v1)
    en2 = push & a2_v0 & a2_v1 & (~a2_v2)
    en3 = push & a2_v0 & a2_v1 & a2_v2 & (~a2_v3)
    
    # Next state values
    val0_next = a2_v0 | en0
    val1_next = a2_v1 | en1
    val2_next = a2_v2 | en2
    val3_next = a2_v3 | en3
    data0_next = mux(en0, in_data, a2_d0)
    data1_next = mux(en1, in_data, a2_d1)
    data2_next = mux(en2, in_data, a2_d2)
    data3_next = mux(en3, in_data, a2_d3)
    
    # Cycle 1: Register the state
    domain.next()
    val0_reg = domain.cycle(val0_next, reset_value=0, name="val0")
    val1_reg = domain.cycle(val1_next, reset_value=0, name="val1")
    val2_reg = domain.cycle(val2_next, reset_value=0, name="val2")
    val3_reg = domain.cycle(val3_next, reset_value=0, name="val3")
    data0_reg = domain.cycle(data0_next, reset_value=0, name="data0")
    data1_reg = domain.cycle(data1_next, reset_value=0, name="data1")
    data2_reg = domain.cycle(data2_next, reset_value=0, name="data2")
    data3_reg = domain.cycle(data3_next, reset_value=0, name="data3")
    
    # Outputs
    m.output("in_ready", in_ready.sig)
    m.output("out0_valid", out0_valid.sig)
    m.output("out0_data", out0_data.sig)
    m.output("out1_valid", out1_valid.sig)
    m.output("out1_data", out1_data.sig)


# Preserve the historical module name
build.__pycircuit_name__ = "issue_queue_2picker"


# Entry point for JIT compilation
if __name__ == "__main__":
    circuit = compile_cycle_aware(build, name="issue_queue_2picker")
    print(circuit.emit_mlir())
