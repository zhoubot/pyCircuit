# -*- coding: utf-8 -*-
"""Simple counter example using cycle-aware API.

Demonstrates basic counter with enable control and automatic cycle management.
"""
from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    compile_cycle_aware,
    mux,
)


def build(m: CycleAwareCircuit, domain: CycleAwareDomain) -> None:
    """Build a simple 8-bit counter with enable.
    
    Args:
        m: Cycle-aware circuit builder
        domain: Clock domain
    """
    # Cycle 0: Input
    en = domain.create_signal("en", width=1)
    
    # Counter feedback (will be connected to register output)
    count_val = domain.create_const(0, width=8, name="count_init")
    
    # Combinational logic: increment when enabled
    count_next = count_val + 1
    count_with_en = mux(en, count_next, count_val)
    
    # Cycle 1: Register
    domain.next()
    count_reg = domain.cycle(count_with_en, reset_value=0, name="count")
    
    # Output
    m.output("count", count_reg.sig)


# Entry point for JIT compilation
if __name__ == "__main__":
    circuit = compile_cycle_aware(build, name="counter")
    print(circuit.emit_mlir())
