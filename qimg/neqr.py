"""NEQR (Zhang, Lu, Gao, Wang 2013).

|I> = 1/2^n  sum_{Y,X}  |f(Y,X)> |Y> |X>          (q + 2n qubits)

Circuit layout: x[0..n-1], y[0..n-1], c[0..q-1] (LSB first), so the integer
over the position qubits (x then y) is  X + Y*N.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from .core import probabilities, run_counts


def load_image(qc, pos, color, img, dy=0, dx=0):
    """Flip colour bit k iff position==(Y,X) and bit k of f(Y+dy,X+dx) is 1
    (clamp/replicate padding at borders). One MCX per set bit."""
    N = img.shape[0]
    for Y in range(N):
        for X in range(N):
            yy = min(max(Y + dy, 0), N - 1)
            xx = min(max(X + dx, 0), N - 1)
            v = int(img[yy, xx])
            ctrl = X + Y * N
            for k in range(len(color)):
                if (v >> k) & 1:
                    qc.mcx(pos, color[k], ctrl_state=ctrl)


def neqr_circuit(img, q=8):
    img = np.asarray(img, dtype=int)
    N = img.shape[0]
    if img.shape != (N, N) or N < 2 or N & (N - 1):
        raise ValueError("image must be 2^n x 2^n with n >= 1")
    if img.min() < 0 or img.max() >= 2 ** q:
        raise ValueError(f"pixel values must be in [0, {2**q - 1}]")
    n = N.bit_length() - 1
    x, y, c = QuantumRegister(n, "x"), QuantumRegister(n, "y"), QuantumRegister(q, "c")
    qc = QuantumCircuit(x, y, c, name="NEQR")
    pos = list(x) + list(y)
    qc.h(pos)
    load_image(qc, pos, list(c), img)
    return qc, n


def neqr_decode(qc, n, q=8):
    """Exact decode from the full measurement distribution."""
    N = 2 ** n
    p = probabilities(qc, list(range(2 * n + q)))
    img = np.full((N, N), -1, dtype=int)
    for idx in np.flatnonzero(p > 1e-12):
        X, Y, c = idx & (N - 1), (idx >> n) & (N - 1), idx >> (2 * n)
        img[Y, X] = c
    return img


def neqr_decode_shots(qc, n, q=8, shots=4096, seed=1):
    """Finite-shot decode. Returns (image with -1 for unseen pixels, coverage)."""
    N = 2 ** n
    counts = run_counts(qc, list(range(2 * n + q)), shots, seed)
    best = {}
    for idx, k in counts.items():
        pos, c = idx & (N * N - 1), idx >> (2 * n)
        if pos not in best or k > best[pos][1]:
            best[pos] = (c, k)
    img = np.full((N, N), -1, dtype=int)
    for pos, (c, _) in best.items():
        img[pos // N, pos % N] = c
    return img, len(best) / (N * N)
