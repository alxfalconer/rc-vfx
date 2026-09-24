"""Minimal EchoIR: the seam between the physics and any renderer.

In the real repo this is `from quantum_echo.ir import EchoIR`; this stand-in has the
same job (hold F[i,t] + geometry, round-trip a `trajectory` envelope) so render.py
never needs to know where F came from: local sim, otoc-echo-v1 job, or job:<id>/ir.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass(frozen=True)
class EchoIR:
    F: np.ndarray                 # complex, shape (n_sites, depth); column t-1 is depth t
    lattice: str = "chain"        # "chain" | "square"
    shape: tuple = ()             # (n,) for chain, (nx, ny) for square
    kick_site: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_sites(self): return self.F.shape[0]
    @property
    def depth(self): return self.F.shape[1]

    def site_xy(self, i):
        """Normalised (x, y) in [0,1] for a site. Chain: y = 0.5."""
        if self.lattice == "square":
            nx, ny = self.shape
            x, y = i % nx, i // nx
            return (x / max(nx - 1, 1), y / max(ny - 1, 1))
        return (i / max(self.n_sites - 1, 1), 0.5)

    def light_cone(self, tol=1e-9):
        return np.abs(self.F - 1.0) > tol          # True = a live tap

    def to_trajectory(self):
        return {
            "result_type": "trajectory",
            "data": {
                "observables": ["F_re", "F_im"],
                "sites": int(self.n_sites), "steps": int(self.depth),
                "times": list(range(1, self.depth + 1)),
                "series": {"F_re": self.F.real.tolist(), "F_im": self.F.imag.tolist()},
                "pairs": None, "error_bars": None,
            },
            "extras": {"lattice": self.lattice, "shape": list(self.shape),
                       "kick_site": int(self.kick_site),
                       "light_cone": self.light_cone().tolist(), **self.meta},
        }

    @classmethod
    def from_trajectory(cls, env):
        if isinstance(env, (str, bytes)):
            env = json.loads(env)
        d, ex = env["data"], env.get("extras", {})
        re = np.asarray(d["series"]["F_re"], float)
        im = np.asarray(d["series"].get("F_im", np.zeros_like(re)), float)
        n = int(d["sites"])
        if re.shape[0] != n and re.shape[1] == n:   # tolerate steps x sites
            re, im = re.T, im.T
        F = re + 1j * im
        # collapsed reference entries arrive as NaN: treat as "no tap", never as silence-by-accident
        F = np.where(np.isfinite(F), F, 1.0)
        meta = {k: v for k, v in ex.items() if k not in ("lattice", "shape", "kick_site", "light_cone")}
        return cls(F=F, lattice=ex.get("lattice", "chain"),
                   shape=tuple(ex.get("shape", [n])), kick_site=int(ex.get("kick_site", n // 2)),
                   meta=meta)
