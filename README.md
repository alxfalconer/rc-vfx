# Retrocausal Echo — Video (prototype)

The visual twin of `retrocausal-echo-v1`: a multi-tap **frame** delay whose tap map is
the measured OTOC `F[i,t] = c_kick / c_ref`. Same `EchoIR`, same measurement, a second
renderer. Measure once on hardware; render audio and video from the same `ir.json`.

| QEcho stack | id | role |
|---|---|---|
| QEcho core (library) | `quantum-echo` | physics, `EchoIR` |
| Quantum Echo (core engine) | `otoc-echo-v1` | publishes `trajectory` |
| Retrocausal Echo (media) | `retrocausal-echo-v1` | renders it as audio |
| **Retrocausal Echo — Video** (proposed) | `retrocausal-echo-video-v1` (working name) | renders it as video |

## Run the dev server

```bash
brew install ffmpeg          # once, if you don't have it
./run.sh                     # first run builds .venv from requirements.txt
open http://127.0.0.1:8000
```

`PORT=8100 ./run.sh` to change port; `RCV_WORKERS=2` for two concurrent renders (CPU-bound —
one is usually right on a laptop). Job files land in `jobs/<id>/`.

| endpoint | does |
|---|---|
| `GET /` | browser UI: upload, tap-map source, circuit + render controls, player, spacetime, job list |
| `GET /engine` | engine record: params schema, file slots, pricing, error codes |
| `POST /validate` | JSON params → ok or `invalid_params` |
| `POST /estimate` | JSON params → credits (2, or 1 with ir), live tap count, output length |
| `POST /process` | multipart `video` + optional `ir` file or `ir_ref=job:<id>/ir` + `params` JSON → `202 {id}` |
| `GET /jobs/{id}` | status, stage, frames done/total, warnings, `media` result |
| `GET /jobs/{id}/files/{video,ir,taps,spacetime}` | outputs (`?download=true` for attachment) |

```bash
curl -F video=@demo/source.mp4 -F 'params={"negative_mode":"reverse"}' localhost:8000/process
curl -F video=@clip.mov -F ir_ref=job:<id>/ir localhost:8000/process      # re-render, no measurement
```

Physics without an `ir` is the local exact sim (≤ 14 qubits). An `ir` upload (an `otoc-echo-v1`
trajectory) or `ir_ref` skips measurement entirely, which is the hardware path.

## Mapping

| measured | audio engine | this renderer |
|---|---|---|
| depth t | tap time | `delay_frames = round(t·step)`, step = `master_s·fps/T` or `60/bpm·division·fps` |
| site i (chain) | pan | horizontal position → cos² spatial window (partition of unity, so a uniform map is transparent) |
| site (x,y) (square) | pan + diffusion | 2D separable window |
| \|F\|·decay^(t−1) | level | blend weight |
| sign / arg F | polarity | `negative_mode` (below) |
| kick site | regeneration source | frame feedback bus, gain-normalised (`feedback` is true loop gain) |
| `stereo_width` | — | `spatial_width` (0 = every tap full-frame) |
| — | — | `drift`: echoes displaced away from the kick site, growing with t (the cone, moving) |

## Negative taps — light can't go below zero

- `invert` — literal: negative taps are subtracted. Visible only against dry (mix < 1); at
  mix 1 negative light clips to nothing. Same caveat as audio. Static regions where the map is
  net-negative cancel toward black: that is the hollow.
- `negative` — photographic negative of the delayed frame. Visible at any mix. **The negative
  of black is white**, so on dark footage negative taps flood with light (tested, by design).
- `reverse` — the delayed grain of `grain_frames` plays backwards; negative-tap delays are
  clamped to ≥ grain so it stays causal.
- `phase` — composited in Oklab: the tap multiplies chroma (a+ib) by F itself; luminance
  weighted by |F|. The only mode that reads Im F. With θ_z = 0 it is a complementary-hue flip
  (warns). Gain is applied inside Oklab (it is cube-root nonlinear — caught by a test).

## Files

```
src/echo_ir.py   EchoIR + trajectory round-trip (stand-in for quantum_echo.ir)
src/sim.py       exact statevector OTOC (stand-in for measure_ir on aer; demo/tests only)
src/render.py    VideoEcho — imports EchoIR and nothing else quantum (grep-tested)
src/videoio.py   ffmpeg decode → linear-RGB float, encode → H.264 MP4
src/params.py    Pydantic wire contract + limits + error codes
src/serve.py     FastAPI dev server, job queue
src/diagnostics.py  impulse-spacetime PNG per job
src/static/      browser UI
tests/           26 tests (19 renderer, 7 server)
demo/            source clip, renders, impulse_spacetime.png, comparison.mp4
```

## Verified

- Sim reproduces published values: n=7, t=1 → −0.039219 / −0.571619 (the kick writeup, 1e-6);
  |Im F| = 0.30 at θ_z = 0.3; Im F = 0 at θ_z = 0; cone grows one site per layer.
- Trajectory round-trips exactly; NaN (collapsed-reference) entries become "no tap".
- Positive map on a static frame is the identity; all-negative at mix 0.5 cancels to black.
- A flash lands on exactly the tap delays, with the tap's sign.
- Grouped fast path equals a naive per-tap sum (1e-4). BPM 1/16 at 120 = 3.75 frames at 30 fps.
- Feedback at 0.9 converges.

Speed at 480×270, 95 taps: 22–30 ms/frame without drift (taps sharing a delay collapse into
one weight map), ~60 ms with drift, ~270 ms with drift + feedback.

## Open before this is an engine

1. **Trajectory field layout** is assumed (`series.F_re`/`F_im`, sites × steps; loader also
   accepts steps × sites). Confirm against a real `otoc-echo-v1` result.
2. **Hardware IR**: all demos use the local exact sim. Use post-layout-fix runs only (the
   2026-09-15 routing bug made pre-fix |c_ref| non-monotonic).
3. **Limits**: frame buffer = (max delay + grain) × frame. v1 caps height at 720 and input at
   30 s; confirm against upload and job-time limits. Audio track is dropped — could pass it
   through, or run it through `retrocausal-echo-v1` with the same `ir`.
4. **Platform**: `engine.yaml` with a `video` file slot + `by_reference` `ir` slot,
   `resultView.kind: video`, result type `media`. Emitting `files` alongside inline output
   hits the known `result: null` issue — same workaround as the audio engine.
5. **Drift + feedback is slow** (per-tap warps); drift makes every tap its own group.
