# -*- coding: utf-8 -*-
"""FIFO loopback example using Cycle-Aware API.

Demonstrates:
- CycleAwareQueue for FIFO operations
- Push/pop with ready/valid handshaking
"""
from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    compile_cycle_aware,
)


def fifo_loopback(m: CycleAwareCircuit, domain: CycleAwareDomain) -> None:
    """Simple FIFO loopback: data in -> queue -> data out."""
    # Create input signals
    in_valid = domain.create_signal("in_valid", width=1)
    in_data = domain.create_signal("in_data", width=8)
    out_ready = domain.create_signal("out_ready", width=1)
    
    # Create FIFO queue (depth=2, width=8)
    q = m.ca_queue("q", domain=domain, width=8, depth=2)
    
    # Push data when in_valid is high
    q.push(in_data, when=in_valid)
    
    # Pop data when out_ready is high
    p = q.pop(when=out_ready)
    
    # Outputs
    m.output("in_ready", q.in_ready.sig)
    m.output("out_valid", p.valid.sig)
    m.output("out_data", p.data.sig)


if __name__ == "__main__":
    circuit = compile_cycle_aware(fifo_loopback, name="fifo_loopback")
    print(circuit.emit_mlir())
