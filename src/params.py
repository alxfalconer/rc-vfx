"""Wire contract for retrocausal-echo-video-v1 (local dev server).

Two ways in, one renderer — same as retrocausal-echo-v1:
  no `ir` file  -> measure F here (local exact sim; the real engine calls measure_ir)
  an `ir` file  -> EchoIR.from_trajectory, no circuit, physics fields ignored
"""
from __future__ import annotations
import math
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

MAX_SIM_QUBITS = 14          # local statevector ceiling; the platform's aer ceiling is 24
MAX_DEPTH = 16
MAX_UPLOAD_MB = 200
MAX_INPUT_S = 30
MAX_HEIGHT = 720


class Params(BaseModel):
    # --- physics (used only when no ir is supplied) ---
    lattice: Literal["chain", "square"] = "chain"
    n_sites: int = Field(12, ge=2, le=MAX_SIM_QUBITS, description="chain length (ignored for square)")
    shape: Optional[list[int]] = Field(None, description="[nx, ny] for square")
    depth: int = Field(10, ge=1, le=MAX_DEPTH)
    theta_zz: float = Field(0.35 * math.pi, ge=0, le=math.pi / 2)
    theta_x: float = Field(0.3 * math.pi, ge=0, le=math.pi)
    theta_z: float = Field(0.0, ge=0, le=math.pi)
    kick: Literal["X", "Y", "Z"] = "Z"
    kick_site: Optional[int] = Field(None, ge=0)
    disorder: float = Field(0.0, ge=0, le=1)
    seed: int = 0

    # --- render ---
    negative_mode: Literal["invert", "negative", "reverse", "phase"] = "invert"
    master_s: float = Field(2.0, gt=0, le=20)
    bpm: Optional[float] = Field(None, gt=0, le=400)
    division: float = Field(0.25, gt=0, le=4)
    decay: float = Field(0.82, gt=0, le=1)
    mix: float = Field(0.5, ge=0, le=1)
    spatial_width: float = Field(1.0, ge=0, le=1)
    drift: float = Field(0.0, ge=-1, le=1)
    grain_frames: int = Field(6, ge=2, le=60)
    feedback: float = Field(0.0, ge=0, lt=1.25, description=">= 1 runs away; soft-clipped")
    feedback_source: Literal["kick", "all"] = "kick"
    min_level: float = Field(0.02, ge=0, le=1)
    tail_s: Optional[float] = Field(None, ge=0, le=20)
    max_tail_s: float = Field(10.0, ge=0, le=20)
    max_height: int = Field(480, ge=64, le=MAX_HEIGHT)

    # --- expressive layer (defaults = the faithful renderer) ---
    blend: Literal["average", "add", "screen", "lighten", "difference"] = "average"
    punch: float = Field(0.0, ge=0, le=1, description="0 = averaged echoes, 1 = each echo at full strength")
    sparsity: int = Field(0, ge=0, le=400, description="keep only the K strongest taps (0 = all)")
    zoom: float = Field(0.0, ge=-1, le=1, description="deeper taps scale by 1 + zoom * t/T about the kick site")
    spin: float = Field(0.0, ge=-1, le=1, description="deeper taps rotate by spin * 30deg * t/T, direction = sign F")
    fb_zoom: float = Field(0.0, ge=-0.25, le=0.25, description="feedback bus scale per pass")
    fb_spin: float = Field(0.0, ge=-20, le=20, description="feedback bus degrees per pass")
    fb_hue: float = Field(0.0, ge=0, le=1.5, description="feedback bus chroma rotation per pass (radians)")
    chroma_split: float = Field(0.0, ge=0, le=1, description="R lags / B leads by a share of each tap's delay")
    stutter: float = Field(0.0, ge=0, le=1, description="chance a tap group drops out per depth step")
    stutter_seed: int = Field(0, ge=0)
    fb_crossfade: bool = Field(False, description="feedback crossfades input with the bus (tunnels) instead of adding")
    accumulate: Literal["sum", "max"] = Field("sum", description="max = each echo copy at full strength, no stacking")
    # --- QuantumBlur on the echoes (moth-quantum/QuantumBlur, the maths blur-v1 wraps) ---
    qblur: float = Field(0.0, ge=0, le=1, description="strength: rotation per qubit as a fraction of pi")
    qblur_on: Literal["echoes", "depth", "loop"] = "echoes"
    qblur_reach: float = Field(0.0, ge=0, le=1, description="0 local .. 1 non-local (blur-v1 reach)")
    qblur_style: Literal["rx", "ry"] = "rx"
    qblur_mode: Literal["blur", "ghost"] = "blur"
    qblur_scale: float = Field(0.5, ge=0, le=1, description="ghost: displacement scale")
    qblur_width: float = Field(0.3, ge=0.02, le=2, description="ghost: spread of scales")
    qblur_size: int = Field(128, ge=16, le=256, description="grid cells on the long side (blur-v1 size)")
    qblur_shots: int = Field(0, ge=0, le=1_000_000, description="0 = exact; else sampled measurements")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _geometry(self):
        if self.lattice == "square":
            if not self.shape or len(self.shape) != 2 or min(self.shape) < 2:
                raise ValueError("square lattice needs shape [nx, ny] with nx, ny >= 2")
            n = self.shape[0] * self.shape[1]
            if n > MAX_SIM_QUBITS:
                raise ValueError(f"square {self.shape} is {n} sites; local sim ceiling is {MAX_SIM_QUBITS}")
        else:
            n = self.n_sites
        if self.kick_site is not None and self.kick_site >= n:
            raise ValueError(f"kick_site {self.kick_site} out of range for {n} sites")
        return self

    def sim_kwargs(self):
        return dict(n_sites=self.n_sites, depth=self.depth, theta_zz=self.theta_zz, theta_x=self.theta_x,
                    theta_z=self.theta_z, kick=self.kick, kick_site=self.kick_site, lattice=self.lattice,
                    shape=self.shape, disorder=self.disorder, seed=self.seed)

    def render_kwargs(self):
        keys = ("negative_mode master_s bpm division decay mix spatial_width drift grain_frames "
                "feedback feedback_source min_level tail_s max_tail_s blend punch sparsity zoom spin "
                "fb_zoom fb_spin fb_hue chroma_split stutter stutter_seed fb_crossfade accumulate qblur qblur_on qblur_reach qblur_style qblur_mode qblur_scale "
                "qblur_width qblur_size qblur_shots").split()
        return {k: getattr(self, k) for k in keys}


ERROR_CODES = {
    "invalid_params": "Parameters failed validation.",
    "invalid_video": "The video could not be decoded, or exceeds size/duration limits.",
    "invalid_ir": "The ir file is not a valid trajectory envelope, or the job reference is unknown.",
    "no_live_taps": "Every F is ~1 (outside the light cone) or below min_level.",
    "render_failed": "Rendering failed unexpectedly.",
    "moth_unauthorized": "No Moth API key applied, or Moth rejected it.",
    "moth_unavailable": "The Moth API could not be reached, or is rate limiting.",
    "moth_failed": "The Moth measurement job failed or returned no readable trajectory.",
    "timeout": "Serverless only: the render would not finish inside the function time limit."
}


# ---------------------------------------------------------------- presets
# Render-side "characters". Physics fields are left alone, so a preset reshapes how the same
# measured tap map is drawn. `faithful` is the renderer's own defaults.
_OFF = dict(blend="average", punch=0.0, sparsity=0, zoom=0.0, spin=0.0, fb_zoom=0.0, fb_spin=0.0, fb_hue=0.0,
            chroma_split=0.0, stutter=0.0, bpm=None, division=0.25, drift=0.0, feedback=0.0,
            feedback_source="kick", max_tail_s=10.0, fb_crossfade=False, accumulate="sum",
            qblur=0.0, qblur_on="echoes", qblur_reach=0.0, qblur_style="rx", qblur_mode="blur", qblur_scale=0.5,
            qblur_width=0.3, qblur_size=128, qblur_shots=0)
PRESETS = {
    "faithful": {**_OFF, "negative_mode": "invert", "mix": 0.5, "decay": 0.82, "master_s": 2.0,
                 "spatial_width": 1.0, "grain_frames": 6},
    "trails":   {**_OFF, "negative_mode": "invert", "accumulate": "max", "sparsity": 6, "mix": 0.8,
                 "decay": 0.97, "master_s": 3.0, "spatial_width": 0.0, "grain_frames": 6},
    "tunnel":   {**_OFF, "negative_mode": "reverse", "blend": "screen", "accumulate": "max", "sparsity": 8,
                 "mix": 0.9, "decay": 0.9, "master_s": 1.5, "spatial_width": 0.0, "grain_frames": 6, "zoom": 0.6,
                 "feedback": 0.85, "feedback_source": "all", "fb_zoom": 0.08, "fb_spin": 4.0, "fb_crossfade": True,
                 "max_tail_s": 3.0},
    "shatter":  {**_OFF, "negative_mode": "negative", "blend": "difference", "punch": 0.8, "sparsity": 8,
                 "mix": 1.0, "decay": 0.9, "master_s": 1.2, "spatial_width": 1.0, "grain_frames": 6,
                 "spin": 0.6, "chroma_split": 0.5, "feedback": 0.5, "fb_crossfade": True, "max_tail_s": 3.0},
    "strobe":   {**_OFF, "negative_mode": "negative", "accumulate": "max", "sparsity": 6, "mix": 0.9,
                 "decay": 1.0, "master_s": 2.0, "spatial_width": 0.0, "grain_frames": 6, "bpm": 120.0,
                 "division": 0.25, "stutter": 0.7},
    "meltdown": {**_OFF, "negative_mode": "reverse", "blend": "screen", "accumulate": "max", "sparsity": 10,
                 "mix": 1.0, "decay": 0.95, "master_s": 1.5, "spatial_width": 0.0, "grain_frames": 8, "zoom": 0.6,
                 "spin": 0.8, "feedback": 0.85, "feedback_source": "all", "fb_zoom": 0.05, "fb_spin": 4.0,
                 "fb_hue": 0.5, "chroma_split": 0.7, "stutter": 0.4, "fb_crossfade": True, "max_tail_s": 3.0, "qblur": 0.3, "qblur_on": "loop", "qblur_reach": 0.3},
    "haze":     {**_OFF, "negative_mode": "invert", "accumulate": "max", "sparsity": 6, "mix": 0.7,
                 "decay": 0.97, "master_s": 3.0, "spatial_width": 0.0, "grain_frames": 6,
                 "qblur": 0.35, "qblur_on": "echoes", "qblur_reach": 0.0, "qblur_size": 128},
    "ghosts":   {**_OFF, "negative_mode": "invert", "accumulate": "max", "sparsity": 6, "mix": 0.9,
                 "decay": 0.97, "master_s": 2.5, "spatial_width": 0.0, "grain_frames": 6,
                 "qblur": 0.5, "qblur_on": "depth", "qblur_mode": "ghost", "qblur_scale": 0.6,
                 "qblur_width": 0.25, "qblur_size": 128, "qblur_shots": 0}
}
