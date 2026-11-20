"""Reproduces every figure and number in the README.   python run_experiments.py"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qimg import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
results = {}


def show(ax, d, title, cmap="gray", **kw):
    ax.imshow(d, cmap=cmap, interpolation="nearest", **kw); ax.set_title(title, fontsize=9); ax.axis("off")


# ---- 1. NEQR round trip + shot-limited decoding ---------------------------
q, N = 4, 8
img = images.requantize(images.gradient_circle(N, 8), 8, q)
qc, n = neqr_circuit(img, q)
dec = neqr_decode(qc, n, q)
results["neqr_roundtrip_exact"] = bool(np.array_equal(dec, img))
results["neqr_qubits_8x8_q4"] = qc.num_qubits

shots_list = [64, 256, 1024, 4096, 16384, 65536]
cov, acc = [], []
for s in shots_list:
    d, c = neqr_decode_shots(qc, n, q, shots=s, seed=7)
    cov.append(c); acc.append(float(np.mean(d == img)))
results["shot_decode"] = dict(shots=shots_list, coverage=cov, pixel_accuracy=acc)

fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
show(ax[0], img, f"Original {N}x{N}, {q}-bit", vmin=0, vmax=2**q-1)
show(ax[1], dec, "NEQR exact decode", vmin=0, vmax=2**q-1)
d, c = neqr_decode_shots(qc, n, q, shots=1024, seed=7)
show(ax[2], np.where(d < 0, 0, d), f"1024 shots (coverage {c:.0%})", vmin=0, vmax=2**q-1)
ax[3].semilogx(shots_list, cov, "o-", label="pixels sampled")
ax[3].semilogx(shots_list, acc, "s-", label="pixels correct")
ax[3].axhline(1, c="gray", ls=":"); ax[3].set_xlabel("shots"); ax[3].legend(); ax[3].set_title("Readout cost (coupon collector)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/1_neqr_roundtrip_and_shots.png", dpi=130); plt.close()

# ---- 2. Edge detection on three images ------------------------------------
tests = {"shapes 8x8": images.requantize(images.shapes(8), 8, 4),
         "gradient+circle 8x8": images.requantize(images.gradient_circle(8), 8, 4),
         "checker 8x8": images.requantize(images.checker(8, block=2), 8, 4)}
fig, axes = plt.subplots(len(tests), 5, figsize=(13, 2.7 * len(tests)))
results["edge"] = {}
T = 2
for r, (name, im) in enumerate(tests.items()):
    t0 = time.time()
    qe, maps = quantum_edge_detect(im, q=4, T=T, directions="hvda")
    dt = time.time() - t0
    ce = classical_edge_detect(im, T, "hvda")
    sb = classical_sobel(im, thresh=T * 4)
    results["edge"][name] = dict(exact_match_classical=bool(np.array_equal(qe, ce)),
                                 vs_sobel=metrics.prf(qe, sb), sim_seconds=round(dt, 1),
                                 edge_pixels=int(qe.sum()))
    show(axes[r, 0], im, name, vmin=0, vmax=15)
    show(axes[r, 1], maps["h"] > .5, "quantum: horizontal")
    show(axes[r, 2], maps["v"] > .5, "quantum: vertical")
    show(axes[r, 3], qe, "quantum: H|V|diag|anti")
    show(axes[r, 4], sb, "classical Sobel (ref.)")
plt.suptitle(f"Quantum edge detection (NEQR + in-circuit subtractor + comparator), threshold |dI|>{T}", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/2_edge_detection.png", dpi=130); plt.close()

# ---- 3. Threshold sweep ----------------------------------------------------
im = tests["gradient+circle 8x8"]
Ts = [0, 1, 2, 4, 8, 12]
fig, ax = plt.subplots(1, len(Ts), figsize=(2.2 * len(Ts), 2.5))
results["threshold_sweep"] = {}
for a, t in zip(ax, Ts):
    qe, _ = quantum_edge_detect(im, q=4, T=t, directions="hv")
    ok = np.array_equal(qe, classical_edge_detect(im, t, "hv"))
    results["threshold_sweep"][t] = dict(edge_pixels=int(qe.sum()), exact_match=bool(ok))
    show(a, qe, f"T={t}  ({int(qe.sum())} px)")
plt.suptitle("Tolerance T: the comparator suppresses small gradients", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/3_threshold_sweep.png", dpi=130); plt.close()

# ---- 4. Shot-based edge detection -----------------------------------------
im = images.requantize(images.shapes(4), 8, 4)
ref = classical_edge_detect(im, 1, "hv")
sh, f1 = [200, 1000, 5000, 25000], []
for s in sh:
    qe, _ = quantum_edge_detect(im, q=4, T=1, directions="hv", shots=s, seed=11)
    f1.append(metrics.prf(qe, ref)["f1"])
results["shot_edge_f1"] = dict(shots=sh, f1=f1)
plt.figure(figsize=(4.5, 3)); plt.semilogx(sh, f1, "o-"); plt.ylim(0, 1.05)
plt.xlabel("shots"); plt.ylabel("F1 vs exact edge map"); plt.title("Finite-shot edge readout (4x4)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/4_shot_edge.png", dpi=130); plt.close()

# ---- 5. MCQI colour image --------------------------------------------------
rgb = images.color_shapes(4, 8)
qc, n = mcqi_circuit(rgb, 8)
out = mcqi_decode(qc, n, 8)
results["mcqi"] = dict(qubits=qc.num_qubits, exact=bool(np.array_equal(out[..., :3], rgb)))
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
show(ax[0], rgb.astype(np.uint8), "Original RGB 4x4", cmap=None)
show(ax[1], out[..., :3].astype(np.uint8), "MCQI decoded", cmap=None)
edge = np.zeros((4, 4), bool)
for ch in range(3):   # per-channel quantum edge detection, union of channels
    e, _ = quantum_edge_detect(rgb[..., ch], q=8, T=30, directions="hv"); edge |= e
results["mcqi"]["colour_edge_pixels"] = int(edge.sum())
show(ax[2], edge, "Colour edges (R|G|B channels)")
plt.tight_layout(); plt.savefig(f"{OUT}/5_mcqi.png", dpi=130); plt.close()

# ---- 6. Resource scaling ---------------------------------------------------
rows = []
for Nn in [2, 4, 8, 16]:
    im8 = images.gradient_circle(Nn, 8)
    rows.append(resources.neqr_stats(im8, 8, transpile_cx=Nn <= 4))
results["resources_neqr_8bit"] = rows
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
Ns = [r["N"] for r in rows]
ax[0].semilogy(Ns, [r["qubits"] for r in rows], "o-", label="NEQR qubits")
ax[0].semilogy(Ns, [r["classical_bits"] for r in rows], "s-", label="classical bits (N²·q)")
ax[0].set_xlabel("image side N"); ax[0].legend(); ax[0].set_title("Storage: log vs quadratic", fontsize=9)
ax[1].semilogy(Ns, [r["mcx"] for r in rows], "o-"); ax[1].set_xlabel("image side N")
ax[1].set_title("Encoding cost: # multi-controlled X (grows ~N²)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/6_resources.png", dpi=130); plt.close()
results["edge_qubits_by_N_q8"] = {N_: edge_qubit_count(N_, 8) for N_ in [2, 4, 8, 16, 32]}

json.dump(results, open(f"{OUT}/results.json", "w"), indent=2, default=float)
print(json.dumps(results, indent=2, default=float))
