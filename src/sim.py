"""Exact statevector first-order OTOC — local stand-in for quantum_echo.measure_ir(machine="aer").

|+>^n -> U(t) -> W_j -> U(t)^dagger -> <X_i> + i<Y_i>.  Layer = RZZ brickwork, then RX, then RZ.
c_ref = 1 exactly in this sim, so F = c_kick.
"""
from __future__ import annotations
import numpy as np
from echo_ir import EchoIR


def _bond_groups(lattice, shape):
    if lattice == "chain":
        n = shape[0]
        return [[(a, a + 1) for a in range(s, n - 1, 2)] for s in (0, 1)], n
    nx, ny = shape
    idx = lambda x, y: y * nx + x
    g = [[(idx(x, y), idx(x + 1, y)) for y in range(ny) for x in range(s, nx - 1, 2)] for s in (0, 1)]
    g += [[(idx(x, y), idx(x, y + 1)) for x in range(nx) for y in range(s, ny - 1, 2)] for s in (0, 1)]
    return g, nx * ny


class _Sim:
    def __init__(self, n):
        self.n = n
        b = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        self.z = 1 - 2 * b                       # z[k, q] = +-1, qubit q is bit q

    def rzz(self, psi, a, b, th):
        return psi * np.exp(-0.5j * th * self.z[:, a] * self.z[:, b])

    def rz_all(self, psi, th):
        return psi * np.exp(-0.5j * th * self.z.sum(1))

    def one_q(self, psi, q, M):
        s = psi.reshape([2] * self.n)            # axis n-1-q is qubit q (little-endian)
        ax = self.n - 1 - q
        s = np.moveaxis(np.tensordot(M, np.moveaxis(s, ax, 0), axes=1), 0, ax)
        return s.reshape(-1)


def _rx(th):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


PAULI = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1, -1]).astype(complex)}


def measure_ir(n_sites=10, depth=10, theta_zz=0.35 * np.pi, theta_x=0.3 * np.pi, theta_z=0.0,
               kick="Z", kick_site=None, lattice="chain", shape=None, disorder=0.0, seed=0):
    shape = tuple(shape) if shape else (n_sites,)
    groups, n = _bond_groups(lattice, shape)
    kick_site = n // 2 if kick_site is None else kick_site
    rng = np.random.default_rng(seed)
    S = _Sim(n)
    jit = lambda base, size: base * (1 + disorder * rng.standard_normal(size))
    layers = []
    for _ in range(depth):
        zz = [list(jit(theta_zz, len(g))) for g in groups]
        layers.append((zz, jit(theta_x, n), theta_z))

    def apply_layer(psi, L, sign):
        zz, xs, tz = L
        if sign > 0:
            for g, ths in zip(groups, zz):
                for (a, b), th in zip(g, ths): psi = S.rzz(psi, a, b, th)
            for q in range(n): psi = S.one_q(psi, q, _rx(xs[q]))
            if tz: psi = S.rz_all(psi, tz)
        else:                                    # exact adjoint: reversed order, negated angles
            if tz: psi = S.rz_all(psi, -tz)
            for q in range(n): psi = S.one_q(psi, q, _rx(-xs[q]))
            for g, ths in zip(reversed(groups), list(reversed(zz))):
                for (a, b), th in zip(reversed(g), reversed(ths)): psi = S.rzz(psi, a, b, -th)
        return psi

    plus = np.full(2 ** n, 2 ** (-n / 2), complex)
    F = np.zeros((n, depth), complex)
    fwd = plus
    for t in range(1, depth + 1):
        fwd = apply_layer(fwd, layers[t - 1], +1)
        psi = S.one_q(fwd, kick_site, PAULI[kick])
        for L in reversed(layers[:t]):
            psi = apply_layer(psi, L, -1)
        for i in range(n):
            ex = np.vdot(psi, S.one_q(psi, i, PAULI["X"])).real
            ey = np.vdot(psi, S.one_q(psi, i, PAULI["Y"])).real
            F[i, t - 1] = ex + 1j * ey
    return EchoIR(F=F, lattice=lattice, shape=shape, kick_site=kick_site,
                  meta={"spec": dict(n_sites=n, depth=depth, theta_zz=float(theta_zz),
                                     theta_x=float(theta_x), theta_z=float(theta_z), kick=kick,
                                     disorder=float(disorder), seed=seed, lattice=lattice),
                        "machine": "local-exact-sim"})
