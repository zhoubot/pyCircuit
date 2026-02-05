# -*- coding: utf-8 -*-
"""Wire operations example using cycle-aware API.

Demonstrates combinational logic with conditional mux and register output.
"""
from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    compile_cycle_aware,
    mux,
)


def build(m: CycleAwareCircuit, domain: CycleAwareDomain) -> None:
    """Build a circuit with XOR/AND select and registered output.
    
    Args:
        m: Cycle-aware circuit builder
        domain: Clock domain
    """
    # Cycle 0: Inputs
    a = domain.create_signal("a", width=8)
    b = domain.create_signal("b", width=8)
    sel = domain.create_signal("sel", width=1)
    
    # Combinational logic
    y_xor = a ^ b
    y_and = a & b
    y = mux(sel, y_and, y_xor)  # sel ? (a & b) : (a ^ b)
    
    # Cycle 1: Register the result
    domain.next()
    y_reg = domain.cycle(y, reset_value=0, name="y_reg")
    
    # Output
    m.output("y", y_reg.sig)


# Entry point for JIT compilation
if __name__ == "__main__":
    circuit = compile_cycle_aware(build, name="wire_ops")
    print(circuit.emit_mlir())
