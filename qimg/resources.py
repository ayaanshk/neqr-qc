"""Resource accounting: qubits, MCX counts, transpiled CX counts."""
import numpy as np
from qiskit import transpile
from .neqr import neqr_circuit


def neqr_stats(img, q=8, transpile_cx=False):
    qc, n = neqr_circuit(img, q)
    N = img.shape[0]
    s = dict(N=N, qubits=qc.num_qubits, mcx=sum(v for k, v in qc.count_ops().items() if k != "h"),
             classical_bits=N * N * q)
    if transpile_cx:
        t = transpile(qc, basis_gates=["cx", "u"], optimization_level=0)
        s["cx"] = t.count_ops().get("cx", 0)
    return s
