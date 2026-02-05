# -*- coding: utf-8 -*-
"""JIT control flow example using cycle-aware API.

Demonstrates if/else and for loop unrolling in hardware description.
"""
from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    CycleAwareSignal,
    compile_cycle_aware,
    mux,
)


def build(
    m: CycleAwareCircuit,
    domain: CycleAwareDomain,
    N: int = 4,
) -> CycleAwareSignal:
    """Build a circuit with conditional and loop logic.
    
    Computes: x = (a + b) >> 1, then x += 1 if a < b else x += 2,
    then accumulates x += 1 for N iterations.
    
    Args:
        m: Cycle-aware circuit builder
        domain: Clock domain
        N: Number of accumulation iterations (JIT-time parameter)
    
    Returns:
        Final accumulated result
    """
    # Cycle 0: Inputs
    a = domain.create_signal("a", width=8)
    b = domain.create_signal("b", width=8)
    
    # Combinational logic
    x = (a + b) >> 1
    
    # Conditional: if a < b then x + 1 else x + 2
    cond = a.lt(b)
    x_inc1 = x + 1
    x_inc2 = x + 2
    x = mux(cond, x_inc1, x_inc2)
    
    # Loop unrolling: accumulate N times
    acc = x
    for _ in range(N):
        acc = acc + 1
    
    return acc


# Entry point for JIT compilation
if __name__ == "__main__":
    circuit = compile_cycle_aware(build, name="jit_control_flow", N=4)
    print(circuit.emit_mlir())
