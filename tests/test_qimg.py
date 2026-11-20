import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qimg import *
from qimg.core import ge_const, probabilities


@pytest.mark.parametrize("m", [2, 3, 4])
def test_ge_const_exhaustive(m):
    for c in range(-1, 2 ** m + 2):
        for D in range(2 ** m):
            qc = QuantumCircuit(m + 1)
            for i in range(m):
                if D >> i & 1: qc.x(i)
            ge_const(qc, list(range(m)), c, m)
            p = probabilities(qc, list(range(m + 1)))
            idx = int(np.argmax(p))
            assert (idx >> m) == int(D >= c), (m, c, D)


@pytest.mark.parametrize("N,q", [(2, 3), (4, 4), (4, 8), (8, 3)])
def test_neqr_roundtrip_random(N, q):
    rng = np.random.default_rng(N * 10 + q)
    img = rng.integers(0, 2 ** q, (N, N))
    qc, n = neqr_circuit(img, q)
    assert qc.num_qubits == 2 * n + q
    assert np.array_equal(neqr_decode(qc, n, q), img)


def test_neqr_state_is_uniform_over_positions():
    img = images.shapes(4, 4)
    qc, n = neqr_circuit(img, 4)
    p = probabilities(qc, list(range(2 * n + 4)))
    nz = p[p > 1e-12]
    assert len(nz) == 16 and np.allclose(nz, 1 / 16)


def test_neqr_input_validation():
    with pytest.raises(ValueError): neqr_circuit(np.zeros((3, 3), int), 4)
    with pytest.raises(ValueError): neqr_circuit(np.full((4, 4), 16), 4)


def test_mcqi_roundtrip():
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 16, (2, 2, 4))
    qc, n = mcqi_circuit(rgba, 4)
    assert qc.num_qubits == 2 * n + 2 + 4
    assert np.array_equal(mcqi_decode(qc, n, 4), rgba)


def test_mcqi_rgb_gets_opaque_alpha():
    rgb = images.color_shapes(4, 4)
    qc, n = mcqi_circuit(rgb, 4)
    out = mcqi_decode(qc, n, 4)
    assert np.array_equal(out[..., :3], rgb) and (out[..., 3] == 15).all()


@pytest.mark.parametrize("T", [0, 1, 3, 7])
@pytest.mark.parametrize("dirs", ["h", "v", "hv", "hvda"])
def test_edge_matches_classical_random(T, dirs):
    rng = np.random.default_rng(T * 7 + len(dirs))
    img = rng.integers(0, 16, (4, 4))
    qe, _ = quantum_edge_detect(img, q=4, T=T, directions=dirs)
    assert np.array_equal(qe, classical_edge_detect(img, T, dirs))


def test_edge_exact_boundary_cases_signed_difference():
    # differences of exactly +-T and +-(T+1) in both signs
    img = np.array([[0, 3], [4, 0]])
    for T in (2, 3, 4):
        qe, _ = quantum_edge_detect(img, q=3, T=T, directions="hvda")
        assert np.array_equal(qe, classical_edge_detect(img, T, "hvda"))


def test_edge_probabilities_are_binary():
    _, maps = quantum_edge_detect(images.shapes(8, 4), q=4, T=1)
    for m in maps.values():
        assert np.allclose(m, np.round(m))


def test_edge_8bit_full_range():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (2, 2))
    qe, _ = quantum_edge_detect(img, q=8, T=40, directions="hvda")
    assert np.array_equal(qe, classical_edge_detect(img, 40, "hvda"))


def test_edge_shots_converge():
    img = images.shapes(4, 4)
    ref = classical_edge_detect(img, 1)
    qe, _ = quantum_edge_detect(img, q=4, T=1, shots=20000, seed=3)
    assert np.array_equal(qe, ref)


def test_flat_image_has_no_edges():
    qe, _ = quantum_edge_detect(np.full((4, 4), 9), q=4)
    assert not qe.any()
