"""Quantum edge detection on NEQR images.

Circuit for ONE direction (dy,dx):

  pos  : H^(2n)                                 uniform superposition
  A    : NEQR of f(Y,X)                         (q bits, padded to m=q+1)
  B    : NEQR of f(Y+dy, X+dx)                  quantum image translation
  B   <- B - A  (mod 2^m)  [inverse modular adder = reversible subtractor]
         so B = D, the signed difference in two's complement, |D| <= 2^q - 1
  flag : toggled iff T < D < 2^m - T   <=>   |D| > T
         = [D >= T+1] XOR [D >= 2^m - T]       (two ancilla-free comparators)

Measuring (position, flag) gives P(flag=1 | Y,X) in {0,1}: an exact edge map.
Directions: 'h','v' (gradient), 'd','a' (diagonals); maps are OR-ed.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import ModularAdderGate
from .core import probabilities, run_counts, ge_const
from .neqr import load_image

DIRECTIONS = {"h": (0, 1), "v": (1, 0), "d": (1, 1), "a": (1, -1)}


def edge_circuit(img, direction, q=8, T=0):
    img = np.asarray(img, dtype=int)
    N = img.shape[0]
    n = N.bit_length() - 1
    m = q + 1
    dy, dx = DIRECTIONS[direction]
    x, y = QuantumRegister(n, "x"), QuantumRegister(n, "y")
    A, B = QuantumRegister(m, "A"), QuantumRegister(m, "B")
    flag = QuantumRegister(1, "edge")
    qc = QuantumCircuit(x, y, A, B, flag, name=f"edge_{direction}")
    pos = list(x) + list(y)
    qc.h(pos)
    load_image(qc, pos, list(A)[:q], img)                 # A = f(Y,X)
    load_image(qc, pos, list(B)[:q], img, dy=dy, dx=dx)   # B = f(shifted)
    qc.append(ModularAdderGate(m).inverse(), list(A) + list(B))   # B = B - A
    ge_const(qc, list(B), T + 1, flag[0])
    ge_const(qc, list(B), 2 ** m - T, flag[0])
    return qc, n, pos + [flag[0]]


def edge_qubit_count(N, q=8):
    n = int(np.log2(N))
    return 2 * n + 2 * (q + 1) + 1    # position + A + B + flag (adder is ancilla-free)


def _edge_map(qc, n, meas, shots=None, seed=1):
    N = 2 ** n
    if shots is None:
        p = probabilities(qc, meas)
        prob = np.zeros((N, N))
        for idx in np.flatnonzero(p > 1e-12):
            if idx >> (2 * n):
                prob[(idx >> n) & (N - 1), idx & (N - 1)] = p[idx] * N * N
        return prob
    counts = run_counts(qc, meas, shots, seed)
    seen, hit = np.zeros((N, N)), np.zeros((N, N))
    for idx, k in counts.items():
        Y, X = (idx >> n) & (N - 1), idx & (N - 1)
        seen[Y, X] += k
        if idx >> (2 * n):
            hit[Y, X] += k
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, hit / seen, 0.0)


def quantum_edge_detect(img, q=8, T=0, directions="hv", shots=None, seed=1):
    """Returns (edge_bool_map, {direction: P(edge) map}).
    shots=None -> exact statevector probabilities; int -> finite-shot estimate
    (each pixel is seen ~shots/N^2 times, so shots must be >> N^2)."""
    maps = {}
    for d in directions:
        qc, n, meas = edge_circuit(img, d, q, T)
        maps[d] = _edge_map(qc, n, meas, shots, seed)
    out = np.zeros(next(iter(maps.values())).shape, dtype=bool)
    for m in maps.values():
        out |= m > 0.5
    return out, maps


def _shifted(img, dy, dx):
    N = img.shape[0]
    yy = np.clip(np.arange(N)[:, None] + dy, 0, N - 1)
    xx = np.clip(np.arange(N)[None, :] + dx, 0, N - 1)
    return img[yy, xx]


def classical_edge_detect(img, T=0, directions="hv"):
    """Exact classical twin of the circuit: |f(p) - f(p+d)| > T."""
    img = np.asarray(img, dtype=int)
    out = np.zeros(img.shape, dtype=bool)
    for d in directions:
        out |= np.abs(img - _shifted(img, *DIRECTIONS[d])) > T
    return out


def classical_sobel(img, thresh):
    """Standard Sobel magnitude (benchmark reference, not the quantum algorithm)."""
    f = np.pad(np.asarray(img, float), 1, mode="edge")
    gx = (f[:-2, 2:] + 2 * f[1:-1, 2:] + f[2:, 2:]) - (f[:-2, :-2] + 2 * f[1:-1, :-2] + f[2:, :-2])
    gy = (f[2:, :-2] + 2 * f[2:, 1:-1] + f[2:, 2:]) - (f[:-2, :-2] + 2 * f[:-2, 1:-1] + f[:-2, 2:])
    return np.hypot(gx, gy) > thresh
