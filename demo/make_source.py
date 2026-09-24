"""Synthetic test clip: textured mid-tone ground, a bright orbiting disc, a sweeping bar."""
import sys, numpy as np
sys.path.insert(0, "src")
from videoio import Writer
from render import srgb_to_lin
W, H, FPS, SECS = 480, 270, 30, 6
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
ground = np.stack([0.30 + 0.10 * xx / W, 0.32 + 0.06 * np.sin(yy / 23), 0.40 - 0.10 * xx / W], -1)
ground += 0.03 * ((xx // 24 + yy // 24) % 2)[..., None]
w = Writer("demo/source.mp4", W, H, FPS)
for n in range(FPS * SECS):
    t = n / FPS
    f = ground.copy()
    cx, cy = W / 2 + 150 * np.sin(1.3 * t), H / 2 + 70 * np.sin(2.1 * t + 0.6)
    d = np.hypot(xx - cx, yy - cy)
    disc = np.clip(22 - d, 0, 1)[..., None]
    f = f * (1 - disc) + disc * np.array([1.0, 0.72, 0.25])
    bx = (40 + 90 * t) % W
    bar = (np.abs(xx - bx) < 6)[..., None] * (yy > H * 0.62)[..., None]
    f = np.where(bar, [0.2, 0.85, 0.9], f)
    w.write(srgb_to_lin(np.clip(f, 0, 1)))
print(w.close(), "frames")
