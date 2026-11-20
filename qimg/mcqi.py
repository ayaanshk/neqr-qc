"""MCQI: multi-channel quantum image (RGB-alpha), after Sun et al.

|I> = 1/2^(n+1) sum_{Y,X} sum_{k=0..3} |C^k_{YX}> |k> |Y>|X>

A 2-qubit channel selector k (0=R,1=G,2=B,3=alpha) joins the position
register as extra controls; each channel value is a q-bit basis state.
Qubits: 2n (position) + 2 (channel) + q (colour).
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from .core import probabilities


def mcqi_circuit(rgba, q=8):
    rgba = np.asarray(rgba, dtype=int)
    N = rgba.shape[0]
    if rgba.ndim == 3 and rgba.shape[2] == 3:       # add opaque alpha
        rgba = np.concatenate([rgba, np.full((N, N, 1), 2 ** q - 1)], axis=2)
    if rgba.shape != (N, N, 4) or N & (N - 1):
        raise ValueError("need 2^n x 2^n x (3|4) array")
    n = N.bit_length() - 1
    x, y = QuantumRegister(n, "x"), QuantumRegister(n, "y")
    ch, c = QuantumRegister(2, "ch"), QuantumRegister(q, "c")
    qc = QuantumCircuit(x, y, ch, c, name="MCQI")
    ctrl_q = list(x) + list(y) + list(ch)
    qc.h(ctrl_q)
    for Y in range(N):
        for X in range(N):
            for k in range(4):
                v = int(rgba[Y, X, k])
                ctrl = X + Y * N + (k << (2 * n))
                for b in range(q):
                    if (v >> b) & 1:
                        qc.mcx(ctrl_q, c[b], ctrl_state=ctrl)
    return qc, n


def mcqi_decode(qc, n, q=8):
    N = 2 ** n
    p = probabilities(qc, list(range(2 * n + 2 + q)))
    out = np.full((N, N, 4), -1, dtype=int)
    for idx in np.flatnonzero(p > 1e-12):
        X, Y = idx & (N - 1), (idx >> n) & (N - 1)
        k, c = (idx >> (2 * n)) & 3, idx >> (2 * n + 2)
        out[Y, X, k] = c
    return out
