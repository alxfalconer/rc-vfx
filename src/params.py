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
    feedback: float = Field(0.0, ge=0, lt=1)
    feedback_source: Literal["kick", "all"] = "kick"
    min_level: float = Field(0.02, ge=0, le=1)
    tail_s: Optional[float] = Field(None, ge=0, le=20)
    max_tail_s: float = Field(10.0, ge=0, le=20)
    max_height: int = Field(480, ge=64, le=MAX_HEIGHT)

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
                "feedback feedback_source min_level tail_s max_tail_s").split()
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
}
