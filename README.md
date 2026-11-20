# qimg — Quantum Image Representation (NEQR / MCQI) & Quantum Edge Detection

A complete, tested Qiskit implementation of:

* **NEQR** — Zhang, Lu, Gao & Wang (2013), *Quantum Inf. Process.* 12(8), 2833–2860.
  `|I> = 1/2^n Σ_{Y,X} |f(Y,X)> |Y>|X>` — grayscale in **basis states**, coordinates in a position register.
* **MCQI** — multi-channel (RGBα) extension: a 2-qubit channel selector `|k>` is added so one circuit holds a colour image.
* **Quantum edge detection** done *inside the circuit* (not by classical post-processing).

```
pip install -r requirements.txt
python -m pytest -q            # 32 tests, ~45 s
python run_experiments.py      # regenerates figures/ and figures/results.json
```

## Layout
| file | purpose |
|---|---|
| `qimg/neqr.py` | NEQR encoder, exact decoder, finite-shot decoder |
| `qimg/mcqi.py` | MCQI (RGBα) encoder/decoder |
| `qimg/edge.py` | quantum edge detector + classical twin + Sobel reference |
| `qimg/core.py` | simulator helpers, ancilla-free `ge_const` comparator |
| `qimg/resources.py` | qubit / gate accounting |
| `tests/test_qimg.py` | exhaustive comparator test, random-image round trips, quantum == classical |

## How encoding works
Position qubits `x,y` get Hadamards → uniform superposition over all `N²` coordinates. For each pixel and each
**set bit** of its gray value, one multi-controlled X (controls = the 2n position qubits fixed to that pixel's
coordinate) flips the matching colour qubit. Pixels are therefore stored as `|f(Y,X)>` basis states —
no amplitude precision problem, exact retrieval (unlike FRQI's angle encoding).

## How edge detection works (per direction `(dy,dx)`)
1. `A ← f(Y,X)` (NEQR) and `B ← f(Y+dy, X+dx)` (NEQR of the **translated** image), both loaded under the *same* position superposition, so every pixel sits next to its neighbour.
2. `B ← B − A (mod 2^(q+1))` using the **inverse of Qiskit's `ModularAdderGate`**: a real reversible subtractor. `B` now holds the signed difference `D` (two's complement).
3. Flag qubit toggled iff `|D| > T`. Since `|D| > T ⇔ T < D < 2^m − T`, this is `[D ≥ T+1] ⊕ [D ≥ 2^m − T]`.
   Each `[D ≥ c]` is an **ancilla-free** comparator (`core.ge_const`): ≤ m+1 mutually-exclusive prefix patterns, one MCX each. Verified exhaustively.
4. Measure position + flag → `P(edge | Y,X) ∈ {0,1}`.
5. Run for horizontal, vertical (and optionally the two diagonals) and OR the maps.

`T` is a genuine tolerance, giving a thresholded absolute-gradient detector rather than a mere "pixels differ" test.

## Results (all reproduced by `run_experiments.py`)
* **NEQR round trip**: exact on random images (sizes 2×2…8×8, 3–8 bit).
* **Quantum edge map == classical |ΔI|>T reference: exact on every test** (random images, T∈{0,1,3,7}, all direction sets, signed ±T boundary cases, full 8-bit range).
* **Qubit scaling**: NEQR needs `2·log₂N + q` qubits (10 for 2×2 → 16 for 16×16, 8-bit) versus `N²·q` classical bits.
* **Edge circuit size**: `2·log₂N + 2(q+1) + 1` qubits (21 @2×2, 25 @8×8, 29 @32×32 for 8-bit).
* **Readout cost**: decoding needs ~N²·ln(N²) shots (coupon-collector). 8×8: 98 % of pixels after 256 shots, 100 % at 1024. See `figures/1_*`.
* **vs Sobel** (threshold-matched, F1 on 8×8 synthetic images): 0.89 / 0.77 / 0.95. Not expected to be identical: see limitations.
* **MCQI**: 4×4 RGBα, 8-bit = 14 qubits, exact decode; colour edges via per-channel quantum detection.

Figures: `figures/1_…6_*.png`; raw numbers in `figures/results.json`.

## Honest limitations
* **Simulation only.** Statevector sim limits us to ≈ 25 qubits. Encoding needs ~N² multi-controlled gates, so NEQR trades qubit *count* (log) for gate *count* (quadratic); there's no speed-up claim here — NEQR's real selling points are exact grayscale storage and bit-level operations.
* **Reading the image out costs ≥ N² shots**, which cancels any advantage for a one-off classical output.
* **Not noise-simulated.** Real hardware would need error mitigation; MCX-heavy circuits are deep (4×4, 8-bit NEQR ≈ 1000 CX).
* **Forward-difference detector**: it compares each pixel with its right/down neighbour, so edges appear one-sided (offset ≈ 1 px from centred Sobel). Not Sobel-equivalent; a 3×3 Sobel would need weighted multi-operand adders (extension below).
* Borders use replicate padding (no wrap-around false edges). Images must be 2ⁿ×2ⁿ.
* Colour edge detection is per-channel (3 runs), not a joint colour-space operator.

## Natural extensions
Sobel/Prewitt with multi-operand adders · FRQI comparison · noise models (Aer `NoiseModel`, density-matrix, ≤ 12 qubits) · QFT-based adder for fewer CX · cyclic-shift translation circuits (quantum incrementer on position register) in place of the lookup-style shifted load.
