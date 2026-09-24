"""VideoEcho — a multi-tap frame delay whose tap map is a measured OTOC.

The visual twin of retrocausal-echo-v1's MultiTapDelay. Imports EchoIR and nothing
else from the quantum side (a test greps for it).

One measured entry F[i, t] -> one tap:

    delay_frames = round(t * step_frames + fine)
    level        = |F| * decay**(t - 1)
    window       = site position -> soft spatial window (partition of unity)
    sign / arg F -> how a negative tap is drawn (negative_mode)

Light can't go below zero, so a negative tap needs a reading:
    invert    literal: subtracted from the mix. Only visible against dry (mix < 1),
              exactly like a polarity-flipped audio tap.
    negative  photographic negative of the delayed frame. Visible at any mix.
    reverse   the delayed grain of `grain_frames` plays backwards.
    phase     chroma (Oklab a,b) multiplied by e^{i arg F}. The only mode that
              reads Im F; with theta_z = 0 it is a 0/180 degree hue flip.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import cv2
from echo_ir import EchoIR

MODES = ("invert", "negative", "reverse", "phase")


# ---------- colour -------------------------------------------------------------
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1i, _M2i = np.linalg.inv(_M1).astype(np.float32), np.linalg.inv(_M2).astype(np.float32)


def to_oklab(lin):
    return (np.cbrt(np.maximum(lin, 0) @ _M1.T) @ _M2.T).astype(np.float32)


def from_oklab(lab):
    return (((lab @ _M2i.T) ** 3) @ _M1i.T).astype(np.float32)


def rotate_chroma(lin, angle):
    """Multiply Oklab chroma (a + ib) by e^{i angle}; luminance untouched."""
    lab = np.cbrt(lin @ _M1.T) @ _M2.T
    c, s = math.cos(angle), math.sin(angle)
    a, b = lab[..., 1].copy(), lab[..., 2].copy()
    lab[..., 1], lab[..., 2] = c * a - s * b, s * a + c * b
    return ((lab @ _M2i.T) ** 3) @ _M1i.T


# ---------- params ---------------------------------------------------------------
@dataclass
class RenderParams:
    master_s: float = 2.0            # whole tap column spans this (ignored if bpm set)
    bpm: float | None = None
    division: float = 0.25           # fraction of a beat per depth step (0.25 = 1/16)
    decay: float = 0.82
    mix: float = 0.5
    spatial_width: float = 1.0       # 0 = every tap full-frame (mono); 1 = sites span the frame
    drift: float = 0.0               # echo displacement away from the kick site, fraction of width at t=T
    negative_mode: str = "invert"
    grain_frames: int = 6
    feedback: float = 0.0            # true loop gain (bus is gain-normalised)
    feedback_source: str = "kick"    # "kick" | "all"
    min_level: float = 0.02
    tail_s: float | None = None      # None = auto
    max_tail_s: float = 10.0

    def validate(self):
        if self.negative_mode not in MODES:
            raise ValueError(f"negative_mode must be one of {MODES}")
        if not 0 <= self.feedback < 1:
            raise ValueError("feedback must be in [0, 1)")
        if not 0 <= self.mix <= 1:
            raise ValueError("mix must be in [0, 1]")


@dataclass
class Tap:
    site: int
    depth: int
    delay: int
    level: float                     # |F| * decay^(t-1), unsigned
    sign: int                        # sign of Re F
    angle: float                     # arg F
    x: float
    y: float
    shift_px: tuple = (0.0, 0.0)


def _pou_weights(centres, n_px):
    """cos^2 partition of unity over pixel axis for sorted unique centres in [0,1]."""
    u = (np.arange(n_px) + 0.5) / n_px
    if len(centres) == 1:
        return {centres[0]: np.ones(n_px, np.float32)}
    pos = np.interp(u, centres, np.arange(len(centres)))    # clamps outside -> edge sites own margins
    return {c: (np.cos(np.pi / 2 * np.clip(pos - k, -1, 1)) ** 2 * (np.abs(pos - k) < 1)).astype(np.float32)
            for k, c in enumerate(centres)}


class VideoEcho:
    def __init__(self, ir: EchoIR, fps: float, size: tuple, params: RenderParams | None = None):
        self.ir, self.fps, (self.H, self.W) = ir, fps, size
        self.p = params or RenderParams()
        self.p.validate()
        self.warnings = []
        if self.p.negative_mode == "phase" and np.abs(ir.F.imag).max() < 1e-6:
            self.warnings.append("negative_mode=phase with real F (theta_z=0): arg F is 0 or pi, "
                                 "so phase reduces to a complementary-hue flip.")
        self.step = (60.0 / self.p.bpm * self.p.division * fps) if self.p.bpm else self.p.master_s * fps / ir.depth
        if self.step < 1:
            self.warnings.append(f"depth step is {self.step:.2f} frames; taps at adjacent depths will share frames.")
        self.taps = self._build_taps()
        if not self.taps:
            raise ValueError("no live taps: every F is ~1 (outside the light cone) or below min_level")
        self._build_windows()

    # -- tap map ------------------------------------------------------------------
    def _build_taps(self):
        ir, p = self.ir, self.p
        live = ir.light_cone()
        kx, ky = ir.site_xy(ir.kick_site)
        taps = []
        for i in range(ir.n_sites):
            x, y = ir.site_xy(i)
            # spatial_width squeezes site positions toward the centre, like stereo_width
            xw = 0.5 + (x - 0.5) * p.spatial_width
            yw = 0.5 + (y - 0.5) * p.spatial_width if ir.lattice == "square" else 0.5
            for t in range(1, ir.depth + 1):
                f = ir.F[i, t - 1]
                if not live[i, t - 1]:
                    continue
                level = abs(f) * p.decay ** (t - 1)
                if abs(f) < p.min_level:
                    continue
                d = max(1, int(round(t * self.step)))
                if p.negative_mode == "reverse" and f.real < 0:
                    d = max(d, p.grain_frames)        # causal: a reversed grain reads ahead inside itself
                frac = t / ir.depth
                shift = (p.drift * (x - kx) * frac * self.W,
                         p.drift * (y - ky) * frac * self.H if ir.lattice == "square" else 0.0)
                taps.append(Tap(i, t, d, float(level), 1 if f.real >= 0 else -1, float(np.angle(f)),
                                xw, yw, shift))
        return taps

    def _build_windows(self):
        p = self.p
        if p.spatial_width == 0:
            self.win = {id(t): np.ones((1, 1, 1), np.float32) for t in self.taps}
        else:
            wx = _pou_weights(sorted({t.x for t in self.taps}), self.W)
            wy = (_pou_weights(sorted({t.y for t in self.taps}), self.H) if self.ir.lattice == "square"
                  else None)
            self.win = {}
            for t in self.taps:
                col = wx[t.x][None, :]
                row = wy[t.y][:, None] if wy else np.ones((1, 1), np.float32)
                self.win[id(t)] = (row * col)[..., None]
        # normalise wet so the loudest pixel's total |level| is 1: no blow-out, sign structure kept
        total = sum(self.win[id(t)] * t.level for t in self.taps) + np.zeros((1, self.W, 1), np.float32)
        self.wet_gain = 1.0 / float(total.max())
        src = [t for t in self.taps if p.feedback_source == "all" or t.site == self.ir.kick_site]
        if p.feedback > 0 and not src:
            self.warnings.append("feedback_source has no live taps; feedback disabled.")
        s = sum(t.level for t in src) or 1.0
        self.fb_taps = [(t, t.level / s) for t in src] if p.feedback > 0 else []
        self.max_delay = max(t.delay for t in self.taps) + p.grain_frames

    # -- grouping: taps sharing a delay read the same frame -> one weight map each -------
    def _groups(self, taps_weights):
        """taps_weights: [(tap, weight_map_or_scalar)] -> dict key -> summed map.

        key = (delay, kind, shift). kind: 'pos' | 'neg'. For phase mode the map is complex
        (level * e^{i arg F} * window, i.e. the tap multiplies chroma by F itself).
        """
        g = {}
        phase = self.p.negative_mode == "phase"
        for t, w in taps_weights:
            shift = (round(t.shift_px[0], 1), round(t.shift_px[1], 1))
            if phase:                                   # (luminance map, chroma map)
                key = (t.delay, "lab", shift)
                Lm, Cm = g.get(key, (0, 0))
                g[key] = (Lm + t.level * w, Cm + t.level * np.exp(1j * t.angle) * w)
                continue
            elif t.sign >= 0:
                key, coef = (t.delay, "pos", shift), t.level
            else:
                key = (t.delay, "neg", shift)
                coef = -t.level if self.p.negative_mode == "invert" else t.level
            g[key] = g.get(key, 0) + coef * w
        return g

    def _build_groups(self):
        self.g_wet = self._groups([(t, self.win[id(t)]) for t in self.taps])
        self.g_fb = self._groups([(t, w) for t, w in self.fb_taps])

    def _src_index(self, n, delay, kind):
        k = n - delay
        if kind == "neg" and self.p.negative_mode == "reverse":
            gl = self.p.grain_frames
            ph = k % gl
            k = min(k - ph + (gl - 1 - ph), n - 1)
        return k

    def _shift(self, img, shift):
        dx, dy = shift
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return img
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    def _apply(self, groups, n, fetch):
        """Sum a group dict at output frame n. Linear RGB out."""
        if self.p.negative_mode == "phase":
            L = np.zeros((self.H, self.W), np.float32)
            Z = np.zeros((self.H, self.W), np.complex64)
            sq = lambda m: m[..., 0] if np.ndim(m) == 3 else m
            for (d, _, sh), (Lm, Cm) in groups.items():
                lab = fetch(n - d, "lab")
                if lab is None:
                    continue
                if sh != (0.0, 0.0):
                    lab = self._shift(lab, sh)
                L += sq(Lm) * lab[..., 0]                # luminance: sum of |F| levels, unrotated
                Z += sq(Cm) * (lab[..., 1] + 1j * lab[..., 2])   # chroma multiplied by F
            return L, Z
        acc = np.zeros((self.H, self.W, 3), np.float32)
        for (d, kind, sh), m in groups.items():
            k = self._src_index(n, d, kind)
            img = fetch(k, "negimg" if (kind == "neg" and self.p.negative_mode == "negative") else "lin")
            if img is None:
                continue
            acc += m * self._shift(img, sh)
        return acc

    # -- render ----------------------------------------------------------------------
    def tail_frames(self, n_in):
        p = self.p
        if p.tail_s is not None:
            return int(p.tail_s * self.fps)
        tail = self.max_delay
        if self.fb_taps:
            mean_d = sum(t.delay * w for t, w in self.fb_taps)
            tail += mean_d * math.log(1e-3) / math.log(p.feedback)
        return int(min(tail, p.max_tail_s * self.fps))

    def process(self, frames, echo_frames=None):
        """frames: iterable of float32 linear-RGB (H,W,3). Yields linear-RGB output frames,
        continuing past the input for the tail (input treated as black).

        echo_frames (cross-echo): a second clip, same size, that feeds the delay line instead of
        `frames`. `frames` stays the dry signal; every tap (and the feedback bus) reads the echo
        clip. Runs until both clips end, then the tail; a clip that ends early continues as black."""
        if not hasattr(self, "g_wet"):
            self._build_groups()
        p, ring_n = self.p, self.max_delay + 2
        ring = {}
        phase = p.negative_mode == "phase"

        def fetch(k, form):
            e = ring.get(k)
            if e is None:
                return None
            if form not in e:
                lin = e["lin"]
                if form == "negimg":
                    e[form] = srgb_to_lin(1.0 - lin_to_srgb(lin))
                elif form == "lab":
                    e[form] = to_oklab(lin)
            return e[form]

        def mix_lab(L, Z, gain=1.0):
            # normalise inside Oklab: it is cube-root nonlinear, so scaling after conversion is gain^3
            lab = gain * np.stack([L, Z.real, Z.imag], -1).astype(np.float32)
            return from_oklab(lab)

        black = np.zeros((self.H, self.W, 3), np.float32)
        it, n, tail_left = iter(frames), 0, None
        eit = iter(echo_frames) if echo_frames is not None else None
        while True:
            x = next(it, None) if tail_left is None else None
            e = next(eit, None) if eit is not None and tail_left is None else None
            if x is None and e is None:
                if tail_left is None:
                    tail_left = self.tail_frames(n)
                if tail_left <= 0:
                    return
                tail_left -= 1
            x = black if x is None else x
            u = x if eit is None else (black if e is None else e)
            if self.g_fb:
                bus = self._apply(self.g_fb, n, fetch)
                bus = mix_lab(*bus) if phase else bus
                u = u + p.feedback * bus
            ring[n] = {"lin": u.astype(np.float32)}
            ring.pop(n - ring_n, None)
            wet = self._apply(self.g_wet, n, fetch)
            wet = mix_lab(*wet, gain=self.wet_gain) if phase else self.wet_gain * wet
            yield (1 - p.mix) * x + p.mix * wet
            n += 1

    # -- sidecars --------------------------------------------------------------------
    def tap_map(self):
        return {
            "fps": self.fps, "size": [self.W, self.H], "step_frames": self.step,
            "wet_gain": self.wet_gain, "max_delay_frames": self.max_delay,
            "feedback_bus_gain": float(self.p.feedback * sum(w for _, w in self.fb_taps)),
            "params": asdict(self.p), "warnings": self.warnings,
            "taps": [{"site": t.site, "depth": t.depth, "delay_frames": t.delay,
                      "delay_s": t.delay / self.fps, "level": t.level, "sign": t.sign,
                      "arg": t.angle, "x": t.x, "y": t.y, "shift_px": list(t.shift_px)}
                     for t in self.taps],
        }
