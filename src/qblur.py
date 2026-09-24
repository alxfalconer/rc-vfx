"""QuantumBlur (moth-quantum/QuantumBlur, James Wootton) as vectorised numpy.

A transcription of the browser port in ~/Desktop/blurbooth/quantumblur.js, which is itself
verified bit-exact against the Python engine (the same maths the `blur-v1` Moth engine wraps):

    encode   brightness -> amplitude sqrt(h), laid out on a grid whose neighbouring cells
             differ in one bit (make_line / make_grid), so one-qubit rotations move light
             between neighbours
    rotate   one rotation per qubit: theta_j = pi * (locality * rate_j + 1 - locality) * xi
             (rate_j = brightness-weighted count of neighbour pairs that differ in bit j)
    readout  |amplitude|^2 (exact) or a seeded multinomial sample of `shots` measurements
    decode   back to the grid

`ghost` mode (blurbooth's ghostHeight) puts the rotation on qubits by displacement scale
instead of brightness rate: a few large-jump qubits give distinct displaced copies.

Gate conventions follow MicroMoth exactly: qubit j is bit j of the state index, rx is the
standard rotation, and ry is MicroMoth's composite rx(pi/2) . rz(theta) . rx(-pi/2) with
rz = h . rx . h.

Deliberate deviation for video: probs2height normalises every image to max = 1, which would
flicker frame to frame; `QBlurGrid.blur` rescales the output to the input's peak instead.
"""
from __future__ import annotations
import math
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------- grid (quantumblur.js:22-41)
def make_line(length: int) -> list[str]:
    n = int(math.ceil(math.log(length) / math.log(2))) if length > 1 else 0
    line = ["0", "1"]
    for _ in range(n - 1):
        line = line + line[::-1]
        half = len(line) // 2
        line = [s + "0" for s in line[:half]] + [s + "1" for s in line[half:]]
    return line


def make_grid(Lx: int, Ly: int | None = None):
    Ly = Ly or Lx
    lx, ly = make_line(Lx), make_line(Ly)
    grid = {lx[x] + ly[y]: (x, y) for x in range(Lx) for y in range(Ly)}
    return grid, len(lx[0] + ly[0])


# ---------------------------------------------------------------- gates (MicroMoth)
def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], np.complex128)


_H = np.array([[1, 1], [1, -1]], np.complex128) / math.sqrt(2)


def _ry(theta: float) -> np.ndarray:
    # MicroMoth: ry(t) = rx(pi/2); rz(t); rx(-pi/2), rz(t) = h; rx(t); h  (applied left to right)
    rz = _H @ _rx(theta) @ _H
    return _rx(-math.pi / 2) @ rz @ _rx(math.pi / 2)


def _apply_1q(state: np.ndarray, U: np.ndarray, j: int, n: int) -> np.ndarray:
    """Apply a 2x2 unitary to qubit j (bit j of the index) of a length-2^n statevector."""
    v = state.reshape(1 << (n - j - 1), 2, 1 << j)
    return np.einsum("ab,ibk->iak", U, v).reshape(-1)


# ---------------------------------------------------------------- one grid, cached
class QBlurGrid:
    def __init__(self, Lx: int, Ly: int):
        self.Lx, self.Ly = Lx, Ly
        grid, n = make_grid(Lx, Ly)
        self.n = n
        coord = {xy: bits for bits, xy in grid.items()}
        # pixel (y, x) row-major -> state index
        self.idx = np.array([int(coord[(x, y)], 2) for y in range(Ly) for x in range(Lx)], np.int64)
        # rates: for every pixel, which qubits its in-grid neighbours differ on (with multiplicity)
        inc = np.zeros((Ly * Lx, n), np.float64)
        for y in range(Ly):
            for x in range(Lx):
                s = coord[(x, y)]
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ns = coord.get((x + dx, y + dy))
                    if ns is None:
                        continue
                    for j in range(n):
                        if ns[j] != s[j]:
                            inc[y * Lx + x, n - j - 1] += 1
        self.inc = inc
        # ghost: displacement each qubit produces from the centre cell (quantumblur.js:195-211)
        cx, cy = Lx >> 1, Ly >> 1
        home = coord[(cx, cy)]
        self.dist = np.ones(n)
        for j in range(n):
            pos = n - 1 - j
            flip = home[:pos] + ("1" if home[pos] == "0" else "0") + home[pos + 1:]
            p = grid.get(flip)
            dx, dy = (p[0] - cx, p[1] - cy) if p else (0, 0)
            self.dist[j] = max(1.0, math.hypot(dx, dy))

    # -- thetas -----------------------------------------------------------------------------
    def blur_thetas(self, h: np.ndarray, xi: float, locality: float) -> np.ndarray:
        rates = np.einsum("p,pn->n", h.reshape(-1).astype(np.float64), self.inc)   # (no BLAS: avoids spurious FP warnings)
        m = rates.max()
        rates = rates / m if m > 0 else rates
        return math.pi * (locality * rates + (1 - locality)) * xi

    def ghost_thetas(self, xi: float, scale: float, width: float) -> np.ndarray:
        target = math.exp(scale * math.log(max(2.0, self.Lx / 2)))            # 1 px .. Lx/2
        w = np.exp(-((np.log(self.dist) - math.log(target)) ** 2) / (2 * max(width, 1e-3) ** 2))
        return math.pi * xi * w

    # -- the circuit ------------------------------------------------------------------------
    def probs(self, h: np.ndarray, thetas: np.ndarray, style: str = "rx") -> np.ndarray:
        """Exact |amp|^2 over the 2^n basis states for heights h[Ly, Lx] (>= 0)."""
        state = np.zeros(1 << self.n, np.complex128)
        state[self.idx] = np.sqrt(np.maximum(h.reshape(-1), 0))
        norm = np.sqrt((state.real ** 2).sum())
        if norm == 0:
            return np.zeros(1 << self.n)
        state /= norm
        gate = _rx if style == "rx" else _ry
        for j, t in enumerate(thetas):
            if abs(t) > 1e-12:
                state = _apply_1q(state, gate(float(t)), j, self.n)
        return state.real ** 2 + state.imag ** 2

    def blur(self, h: np.ndarray, xi: float, reach: float = 0.0, style: str = "rx", mode: str = "blur",
             scale: float = 0.5, width: float = 0.3, shots: int = 0, rng=None) -> np.ndarray:
        """Heights [Ly, Lx] in, heights out, peak kept (see module note)."""
        if xi <= 0:
            return h
        th = self.ghost_thetas(xi, scale, width) if mode == "ghost" else self.blur_thetas(h, xi, 1.0 - reach)
        p = self.probs(h, th, style)
        if shots:
            rng = rng or np.random.default_rng(0)
            s = p.sum()
            p = rng.multinomial(int(shots), p / s).astype(np.float64) / shots if s > 0 else p
        out = p[self.idx].reshape(self.Ly, self.Lx)
        peak_in, peak_out = float(h.max()), float(out.max())
        return out * (peak_in / peak_out) if peak_out > 0 else out


@lru_cache(maxsize=16)
def grid_for(Lx: int, Ly: int) -> QBlurGrid:
    return QBlurGrid(Lx, Ly)


def blur_frame(img: np.ndarray, xi: float, size: int = 128, **kw) -> np.ndarray:
    """Quantum-blur a float frame [H, W, C] >= 0 on a grid whose long side is `size` cells,
    per channel, and bring it back to full resolution (blur-v1's `downscale` strategy)."""
    import cv2
    if xi <= 0:
        return img
    H, W = img.shape[:2]
    s = size / max(H, W)
    gw, gh = max(2, int(round(W * s))), max(2, int(round(H * s)))
    small = cv2.resize(np.clip(img, 0, None).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = small[..., None]
    g = grid_for(gw, gh)
    out = np.stack([g.blur(small[..., c].astype(np.float64), xi, **kw) for c in range(small.shape[2])], -1)
    big = cv2.resize(out.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return big if img.ndim == 3 else big[..., 0]
