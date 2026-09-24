import sys, json, time, pathlib, numpy as np, cv2
sys.path.insert(0, "src")
import sim
from echo_ir import EchoIR
from render import VideoEcho, RenderParams, lin_to_srgb
from videoio import read_frames, Writer
out = pathlib.Path("demo/out"); out.mkdir(exist_ok=True)

ir_real = sim.measure_ir(12, 10, kick_site=6)                 # theta_z = 0: F real, two-signed
ir_cplx = sim.measure_ir(12, 10, kick_site=6, theta_z=0.3)    # F complex, for phase mode
for name, ir in [("ir_real", ir_real), ("ir_complex", ir_cplx)]:
    (out / f"{name}.json").write_text(json.dumps(ir.to_trajectory()))

# hardware path is identical: EchoIR.from_trajectory(<otoc-echo-v1 result or job:<id>/ir>)
ir_real = EchoIR.from_trajectory((out / "ir_real.json").read_text())

variants = {
    "invert":   (ir_real, RenderParams(negative_mode="invert", mix=0.5)),
    "negative": (ir_real, RenderParams(negative_mode="negative", mix=0.5)),
    "reverse":  (ir_real, RenderParams(negative_mode="reverse", mix=0.5, grain_frames=8)),
    "phase":    (ir_cplx, RenderParams(negative_mode="phase", mix=0.6, decay=0.9)),
    "tunnel":   (ir_real, RenderParams(negative_mode="invert", mix=0.55, drift=0.35, feedback=0.6)),
}
only = sys.argv[1:]
for name, (ir, p) in variants.items():
    if only and name not in only: continue
    info, frames = read_frames("demo/source.mp4")
    ve = VideoEcho(ir, info["fps"], (info["height"], info["width"]), p)
    w = Writer(str(out / f"{name}.mp4"), info["width"], info["height"], info["fps"])
    t0 = time.time()
    for f in ve.process(frames):
        w.write(f)
    n = w.close()
    (out / f"{name}.taps.json").write_text(json.dumps(ve.tap_map(), indent=1))
    print(f"{name:9s} {len(ve.taps)} taps  {n} frames  {time.time()-t0:.1f}s  warn={ve.warnings}")

if only and 'impulse' not in only: sys.exit()
# visual impulse response: a single white flash, then the middle row of every output frame
# stacked top-to-bottom = a spacetime diagram of the tap map (x = site, down = time)
H, W, FPS = 60, 480, 30
ve = VideoEcho(ir_real, FPS, (H, W), RenderParams(negative_mode="invert", mix=1.0))
flash = [np.full((H, W, 3), 0.5, np.float32)] + [np.zeros((H, W, 3), np.float32)] * 70
rows = [f[H // 2, :, 0] for f in ve.process(iter(flash))]
k = np.array(rows) / 0.5 / ve.wet_gain                     # back to signed tap level
k = k / np.abs(k).max()
img = np.zeros(k.shape + (3,), np.float32)
img[..., 2] = np.clip(k, 0, 1); img[..., 1] = np.clip(k, 0, 1) * 0.55   # regular -> amber
img[..., 0] = np.clip(-k, 0, 1); img[..., 1] += np.clip(-k, 0, 1) * 0.55  # inverted -> blue
img = cv2.resize(img, (W, k.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(out / "impulse_spacetime.png"), (np.sqrt(np.clip(img, 0, 1)) * 255).astype(np.uint8))
print("kymograph", img.shape)
