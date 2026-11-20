"""Shared simulator helpers and low-level reversible building blocks."""
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

_SIM = AerSimulator(method="statevector")


def _idx(qc, qubits):
    return [qc.find_bit(q).index if not isinstance(q, int) else q for q in qubits]


def probabilities(qc, qubits):
    """Exact marginal probabilities over `qubits` (Qubit objects or ints).
    Array index i has bit j == value of qubits[j] (first listed = LSB)."""
    t = qc.copy()
    t.save_probabilities(_idx(qc, qubits))
    return np.asarray(_SIM.run(transpile(t, _SIM)).result().data()["probabilities"])


def run_counts(qc, qubits, shots, seed=None):
    """Sample `qubits`; returns {int_index: count} (same indexing as above)."""
    t = qc.copy()
    cr = ClassicalRegister(len(qubits), "m")
    t.add_register(cr)
    t.measure(_idx(qc, qubits), cr)
    res = _SIM.run(transpile(t, _SIM), shots=shots, seed_simulator=seed).result()
    return {int(k.replace(" ", ""), 2): v for k, v in res.get_counts().items()}


def ge_const(qc, reg, c, target):
    """Toggle `target` iff the unsigned integer in `reg` (LSB first) is >= c.
    Ancilla-free: {D >= c} is a disjoint union of <= m+1 prefix patterns, each
    realised by one multi-controlled X (mutually exclusive => XOR is exact)."""
    reg = list(reg)
    m = len(reg)
    if c <= 0:
        qc.x(target)
        return
    if c >= 2 ** m:
        return
    for i in range(m - 1, -1, -1):          # i = first bit (from MSB) where D>c
        if not (c >> i) & 1:
            pattern = ((c >> (i + 1)) << (i + 1)) | (1 << i)
            qc.mcx(reg[i:], target, ctrl_state=pattern >> i)
    qc.mcx(reg, target, ctrl_state=c)       # equality term
