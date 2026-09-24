"""QuantumBlur port: equivalence with the Python engine reference, then behaviour."""
import sys, json, math, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import qblur

REF = json.loads((pathlib.Path(__file__).parent / "fixtures" / "quantumblur_ref.json").read_text())
L, XI, LOC = 8, 0.31, 0.7                       # the parameters ref.json was generated with (blurbooth/verify.mjs)


def ref_heights():
    h = np.zeros((L, L))
    for k, v in REF["height"].items():
        x, y = map(int, k.split(","))
        h[y, x] = v
    return h


def test_matches_python_engine_reference():
    g = qblur.QBlurGrid(L, L)
    h = ref_heights()
    p = g.probs(h, g.blur_thetas(h, XI, LOC), "rx")
    assert len(REF["probs"]) == 64
    dev = max(abs(p[int(k, 2)] - v) for k, v in REF["probs"].items())
    assert dev < 1e-12, dev
    assert abs(p.sum() - 1) < 1e-12


def test_sampled_readout_converges():
    g = qblur.QBlurGrid(L, L)
    h = ref_heights()
    p = g.probs(h, g.blur_thetas(h, XI, LOC), "rx")
    errs = []
    for shots in (256, 4096, 65536):
        s = np.random.default_rng(1).multinomial(shots, p / p.sum()) / shots
        errs.append(np.abs(s - p).max())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 5e-3


def test_ry_is_micromoth_composite_and_unitary():
    U = qblur._ry(0.7)
    assert np.allclose(U.conj().T @ U, np.eye(2))
    # MicroMoth builds ry from rx/h: on |0> it must put sin^2(t/2) into |1>, like any Y rotation
    assert np.isclose(abs((U @ np.array([1, 0]))[1]) ** 2, math.sin(0.35) ** 2)


def test_zero_strength_is_identity_and_peak_is_kept():
    rng = np.random.default_rng(0)
    img = rng.random((36, 64, 3)).astype(np.float32) * 0.8
    assert qblur.blur_frame(img, 0.0) is img
    out = qblur.blur_frame(img, 0.5, size=32)
    assert out.shape == img.shape and np.all(out >= 0)
    g, h = qblur.QBlurGrid(32, 18), rng.random((18, 32)) * 0.6       # on the grid itself, the peak is exact
    assert abs(g.blur(h, 0.5).max() - h.max()) < 1e-12


def test_blur_spreads_a_dot():
    g = qblur.QBlurGrid(16, 16)
    h = np.zeros((16, 16)); h[8, 8] = 1.0
    out = g.blur(h, 0.5)
    assert (out > 1e-6).sum() > 1 and out[8, 8] == out.max()     # light leaks to neighbours, centre stays brightest


def test_ghost_mode_makes_power_of_two_copies():
    g = qblur.QBlurGrid(16, 16)
    h = np.zeros((16, 16)); h[8, 8] = 1.0
    # rotate by pi/2 (xi 0.5) with a very narrow scale window: only one or two qubits turn -> 2^k copies
    out = g.blur(h, 0.5, mode="ghost", scale=1.0, width=0.3)
    k = int((out > 1e-3).sum())
    assert k in (2, 4) and k & (k - 1) == 0


def test_seeded_shots_repeat():
    img = np.random.default_rng(2).random((20, 30, 3)).astype(np.float32)
    a = qblur.blur_frame(img, 0.4, size=16, shots=512, rng=np.random.default_rng(5))
    b = qblur.blur_frame(img, 0.4, size=16, shots=512, rng=np.random.default_rng(5))
    c = qblur.blur_frame(img, 0.4, size=16, shots=512, rng=np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
