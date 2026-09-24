"""Impulse spacetime: one white flash in, the middle row of each output frame stacked down.
x = site, down = time. Amber = regular taps, blue = inverted. The light cone, made visible."""
import numpy as np, cv2
from dataclasses import replace
from render import VideoEcho, RenderParams


def impulse_spacetime_png(ir, params: RenderParams, fps, width=480, scale=6) -> bytes:
    H = 8
    p = replace(params, negative_mode="invert", mix=1.0, feedback=0.0, drift=0.0, tail_s=None)
    ve = VideoEcho(ir, fps, (H, width), p)
    n = ve.max_delay + 2
    flash = [np.full((H, width, 3), 0.5, np.float32)] + [np.zeros((H, width, 3), np.float32)] * n
    rows = [f[H // 2, :, 0] for _, f in zip(range(n + 1), ve.process(iter(flash)))]
    k = np.array(rows)
    k = k / (np.abs(k).max() or 1)
    img = np.zeros(k.shape + (3,), np.float32)                 # BGR
    img[..., 2] = np.clip(k, 0, 1); img[..., 1] = np.clip(k, 0, 1) * 0.55 + np.clip(-k, 0, 1) * 0.55
    img[..., 0] = np.clip(-k, 0, 1)
    img = cv2.resize(img, (width, k.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", (np.sqrt(np.clip(img, 0, 1)) * 255).astype(np.uint8))
    return buf.tobytes()
