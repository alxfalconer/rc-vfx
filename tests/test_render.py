import sys, pathlib, re
import numpy as np, pytest
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from echo_ir import EchoIR
from render import VideoEcho, RenderParams, _pou_weights, rotate_chroma
import sim

H, W, FPS = 18, 32, 30


def const_ir(val, n=4, T=3):
    F = np.full((n, T), val, complex); F[:, 0] = 1.0     # depth 1 outside cone -> no taps there
    return EchoIR(F=F, shape=(n,), kick_site=1)


def run(ve, frames):
    return list(ve.process(iter(frames)))


# ---- physics seam -----------------------------------------------------------------
def test_sim_matches_published_values():
    F = sim.measure_ir(7, 1, kick_site=3).F[:, 0].real
    assert np.allclose(F, [1, 1, -0.039219, -0.571619, -0.039219, 1, 1], atol=1e-6)

def test_real_F_at_theta_z_zero_and_complex_above():
    assert np.abs(sim.measure_ir(6, 4).F.imag).max() < 1e-12
    assert np.abs(sim.measure_ir(6, 4, theta_z=0.3).F.imag).max() > 0.1

def test_trajectory_round_trip_exact():
    ir = sim.measure_ir(6, 4, theta_z=0.3)
    back = EchoIR.from_trajectory(ir.to_trajectory())
    assert np.array_equal(back.F, ir.F) and back.kick_site == ir.kick_site

def test_nan_collapsed_entries_are_not_taps():
    env = const_ir(-0.5).to_trajectory(); env["data"]["series"]["F_re"][0][1] = float("nan")
    ir = EchoIR.from_trajectory(env)
    assert not ir.light_cone()[0, 1]

def test_render_imports_only_echo_ir():
    src = (pathlib.Path(__file__).parent.parent / "src" / "render.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+(sim|quantum_echo|qiskit)", src, re.M)


# ---- renderer invariants ------------------------------------------------------------
def test_partition_of_unity():
    w = _pou_weights([0.0, 0.3, 0.7, 1.0], 101)
    assert np.allclose(sum(w.values()), 1.0)

def test_positive_map_is_transparent_on_static_frame():
    frame = np.random.default_rng(0).random((H, W, 3)).astype(np.float32)
    ve = VideoEcho(const_ir(0.6), FPS, (H, W), RenderParams(master_s=0.1, mix=0.7))
    out = run(ve, [frame] * 12)
    assert np.allclose(out[10], frame, atol=2e-3)     # all taps positive, full history -> identity

def test_invert_cancels_static_frame():
    frame = np.full((H, W, 3), 0.5, np.float32)
    ve = VideoEcho(const_ir(-0.6), FPS, (H, W), RenderParams(master_s=0.1, mix=0.5))
    assert np.allclose(run(ve, [frame] * 12)[10], 0, atol=2e-3)   # the "hollow" at mix 0.5

def test_invert_vanishes_at_full_wet():
    frame = np.full((H, W, 3), 0.5, np.float32)
    ve = VideoEcho(const_ir(-0.6), FPS, (H, W), RenderParams(master_s=0.1, mix=1.0))
    assert np.clip(run(ve, [frame] * 12)[10], 0, 1).max() == 0   # pure negative light clips: nothing

def test_impulse_lands_on_tap_delays():
    ir = sim.measure_ir(6, 5, kick_site=2)
    ve = VideoEcho(ir, FPS, (H, W), RenderParams(master_s=1.0, mix=1.0, negative_mode="invert"))
    flash = [np.ones((H, W, 3), np.float32)] + [np.zeros((H, W, 3), np.float32)] * 40
    out = run(ve, flash)                                   # unclipped output is signed
    lit = {n for n, f in enumerate(out) if np.abs(f).max() > 1e-4}
    assert lit == {t.delay for t in ve.taps}
    for n in lit:                                          # sign of the flash = sign of the taps there
        signs = {t.sign for t in ve.taps if t.delay == n}
        if len(signs) == 1:
            assert np.sign(out[n].sum()) == signs.pop()

def test_negative_mode_floods_black():
    # documented behaviour: the negative of black is white
    ve = VideoEcho(const_ir(-0.6), FPS, (H, W), RenderParams(master_s=0.1, mix=1.0, negative_mode="negative"))
    assert run(ve, [np.zeros((H, W, 3), np.float32)] * 12)[10].min() > 0.9

def test_bpm_division_is_a_beat_fraction():
    ve = VideoEcho(const_ir(0.5), 30, (H, W), RenderParams(bpm=120, division=0.25))
    assert ve.step == pytest.approx(3.75)             # 1/16 at 120 BPM = 125 ms = 3.75 frames

def test_feedback_converges():
    frame = np.ones((H, W, 3), np.float32)
    ve = VideoEcho(sim.measure_ir(6, 4), FPS, (H, W), RenderParams(master_s=0.2, feedback=0.9))
    out = run(ve, [frame] * 200)
    peaks = [np.abs(o).max() for o in out]
    assert max(peaks) < 20 and abs(peaks[-1] - peaks[-20]) < 1e-2

def test_reverse_is_causal():
    ir = sim.measure_ir(6, 4)
    ve = VideoEcho(ir, FPS, (H, W), RenderParams(master_s=0.1, negative_mode="reverse", grain_frames=5))
    assert all(t.delay >= 5 for t in ve.taps if t.sign < 0)

def test_chroma_rotation_keeps_luminance_and_pi_is_complement():
    c = np.array([[[0.6, 0.2, 0.1]]], np.float32)
    back = rotate_chroma(rotate_chroma(c, np.pi), np.pi)
    assert np.allclose(back, c, atol=1e-4)
    lum = lambda x: np.cbrt(x @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                          [0.2119034982, 0.6806995451, 0.1073969566],
                                          [0.0883024619, 0.2817188376, 0.6299787005]]).T) @ [0.2104542553, 0.7936177850, -0.0040720468]
    assert np.allclose(lum(rotate_chroma(c, 1.1)), lum(c), atol=1e-4)

def test_phase_warns_on_real_F():
    ve = VideoEcho(sim.measure_ir(6, 4), FPS, (H, W), RenderParams(negative_mode="phase"))
    assert any("phase" in w for w in ve.warnings)

def test_square_lattice_windows_cover_frame():
    ir = sim.measure_ir(lattice="square", shape=(3, 3), depth=3)
    ve = VideoEcho(ir, FPS, (H, W), RenderParams(min_level=0))
    assert len({(t.x, t.y) for t in ve.taps}) > 1

def test_phase_pi_tap_keeps_luminance_and_flips_hue():
    frame = np.full((H, W, 3), [0.6, 0.2, 0.1], np.float32)
    ve = VideoEcho(const_ir(-0.6), FPS, (H, W), RenderParams(master_s=0.1, mix=1.0, negative_mode="phase"))
    o = run(ve, [frame] * 12)[10]
    assert np.allclose(o, rotate_chroma(frame, np.pi), atol=2e-3)

def test_grouping_matches_per_tap_sum():
    # the grouped fast path must equal a naive per-tap sum
    from render import to_oklab
    rng = np.random.default_rng(1)
    frames = [rng.random((H, W, 3)).astype(np.float32) for _ in range(30)]
    ir = sim.measure_ir(6, 5, kick_site=2)
    ve = VideoEcho(ir, FPS, (H, W), RenderParams(master_s=0.5, mix=1.0, negative_mode="invert"))
    n = 25
    out = run(ve, frames)[n]
    naive = sum(t.sign * t.level * ve.win[id(t)] * frames[n - t.delay] for t in ve.taps)
    assert np.allclose(out, ve.wet_gain * naive, atol=1e-4)


# ---- cross-echo ---------------------------------------------------------------------
def test_cross_echo_taps_read_the_second_clip():
    ve = VideoEcho(const_ir(0.5), FPS, (H, W), RenderParams(mix=1.0, master_s=0.3, tail_s=0.5))
    n = 20
    dry = [np.full((H, W, 3), 0.25, np.float32)] * n                   # flat grey A
    flash = [np.zeros((H, W, 3), np.float32) for _ in range(n)]; flash[0][:] = 1.0   # B: one flash
    out = run_x(ve, dry, flash)
    delays = sorted({t.delay for t in ve.taps})
    assert out[0].max() < 1e-6                                          # mix 1: no dry, B not yet echoed
    assert all(out[d].max() > 1e-3 for d in delays)                     # B's flash lands on every tap delay
    quiet = [k for k in range(1, n) if k not in delays]
    assert all(out[k].max() < 1e-6 for k in quiet)                      # A's grey never enters the delay line

def test_cross_echo_runs_to_the_longer_clip_then_tail():
    ve = VideoEcho(const_ir(0.5), FPS, (H, W), RenderParams(tail_s=0.2))
    a = [np.zeros((H, W, 3), np.float32)] * 5
    b = [np.zeros((H, W, 3), np.float32)] * 12
    assert len(run_x(ve, a, b)) == 12 + int(0.2 * FPS)
    assert len(run(ve, a)) == 5 + int(0.2 * FPS)                       # unchanged without an echo clip

def run_x(ve, frames, echo):
    return list(ve.process(iter(frames), iter(echo)))


# ---- expressive layer (all default-off; the faithful path is covered above) ------------
def flash_run(p, ir=None, n=40, value=0.5):
    ve = VideoEcho(ir or const_ir(0.5), FPS, (H, W), p)
    flash = [np.full((H, W, 3), value, np.float32)] + [np.zeros((H, W, 3), np.float32)] * n
    return ve, run(ve, flash)

def test_lighten_never_darkens_dry():
    rng = np.random.default_rng(1)
    frames = [rng.random((H, W, 3)).astype(np.float32) for _ in range(20)]
    ve = VideoEcho(const_ir(-0.5), FPS, (H, W), RenderParams(master_s=0.2, mix=1.0, blend="lighten"))
    out = run(ve, frames)
    assert all((o >= f - 1e-6).all() for o, f in zip(out, frames))

def test_difference_of_static_transparent_map_is_black():
    frame = np.random.default_rng(0).random((H, W, 3)).astype(np.float32)
    ve = VideoEcho(const_ir(0.6), FPS, (H, W), RenderParams(master_s=0.1, mix=1.0, blend="difference"))
    assert np.abs(run(ve, [frame] * 12)[10]).max() < 3e-3

def test_punch_brings_echoes_to_full_strength():
    peak = lambda punch: max(f.max() for f in flash_run(RenderParams(master_s=0.3, mix=1.0, punch=punch))[1][1:])
    assert peak(1.0) >= 0.8 * 0.5 and peak(1.0) > 1.6 * peak(0.0)

def test_sparsity_keeps_the_k_strongest_taps():
    ir = sim.measure_ir(8, 6, kick_site=3)
    full = VideoEcho(ir, FPS, (H, W), RenderParams())
    sparse = VideoEcho(ir, FPS, (H, W), RenderParams(sparsity=5))
    assert len(sparse.taps) == 5
    assert min(t.level for t in sparse.taps) >= sorted((t.level for t in full.taps), reverse=True)[4] - 1e-12

def test_zoom_pushes_echoes_away_from_the_kick_site():
    ve = VideoEcho(const_ir(0.5), FPS, (H, W), RenderParams(master_s=0.3, mix=1.0, spatial_width=0, zoom=1.0))
    dot = np.zeros((H, W, 3), np.float32); dot[H // 2 - 1:H // 2 + 1, 14:16] = 1.0     # right of the pivot, stays in frame at 2x
    out = run(ve, [dot] + [np.zeros((H, W, 3), np.float32)] * 20)
    cx = lambda img: (img[..., 1].sum(0) * np.arange(W)).sum() / max(img[..., 1].sum(), 1e-9)
    deepest = max(ve.taps, key=lambda t: t.depth)
    assert cx(out[deepest.delay]) - ve.pivot[0] > cx(dot) - ve.pivot[0] + 1

def test_stutter_is_seeded_and_off_by_default():
    rng = np.random.default_rng(2)
    frames = [rng.random((H, W, 3)).astype(np.float32) for _ in range(30)]
    ir = sim.measure_ir(8, 6, kick_site=3)
    go = lambda **k: run(VideoEcho(ir, FPS, (H, W), RenderParams(master_s=0.5, mix=1.0, **k)), frames)
    a, b, c, base = go(stutter=0.8, stutter_seed=7), go(stutter=0.8, stutter_seed=7), go(stutter=0.8, stutter_seed=8), go()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert any(not np.allclose(x, y) for x, y in zip(a, c)) and any(not np.allclose(x, y) for x, y in zip(a, base))

def test_runaway_feedback_stays_bounded():
    ve = VideoEcho(const_ir(0.5), FPS, (H, W), RenderParams(master_s=0.2, mix=1.0, feedback=1.2, feedback_source="all",
                                                            fb_zoom=0.05, fb_spin=5, fb_hue=0.4, max_tail_s=1))
    out = run(ve, [np.full((H, W, 3), 0.8, np.float32)] * 60)
    assert all(np.isfinite(f).all() and f.max() <= 1.0 + 1e-6 for f in out)
    assert len(out) == 60 + int(1 * FPS)                  # loop gain >= 1 rings out to the tail cap

def test_chroma_split_makes_red_lag_and_blue_lead():
    ve, out = flash_run(RenderParams(master_s=0.6, mix=1.0, chroma_split=1.0, spatial_width=0))
    t = lambda ch: max(range(1, len(out)), key=lambda n: out[n][..., ch].sum())
    first_lit = lambda ch: min(n for n in range(1, len(out)) if out[n][..., ch].max() > 1e-4)
    assert first_lit(0) > first_lit(1) > first_lit(2)

def test_max_accumulation_never_stacks_copies():
    # two equal copies of a flash land on the same pixels: max keeps one copy's brightness, sum would add
    ir = const_ir(0.5)
    rng = np.random.default_rng(3)
    frames = [rng.random((H, W, 3)).astype(np.float32) * 0.5 for _ in range(20)]
    ve = VideoEcho(ir, FPS, (H, W), RenderParams(master_s=0.2, mix=1.0, spatial_width=0, accumulate="max"))
    out = run(ve, frames)
    assert max(float(o.max()) for o in out) <= 0.5 + 1e-5          # never brighter than the brightest source
    assert out[12].mean() > 0.2                                     # but each copy at (near) full strength
