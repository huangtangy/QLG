"""Random-state utilities and small statistics helpers."""

from __future__ import annotations

import numpy as np

def get_psi_fidelity(psi1, psi2):
    """Return fidelity |<psi1|psi2>|^2 for two state vectors."""
    return np.abs(np.vdot(psi1, psi2)) ** 2

def haar_random_state(N: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a Haar-random pure state vector in C^N."""
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if rng is None:
        rng = np.random.default_rng()
    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    x /= np.linalg.norm(x)
    return x

def get_DKl(fidelities, dim):
    """A simple discrepancy metric between empirical and Haar fidelity PDFs."""
    fidelities = np.array(fidelities)
    bbs = 40
    hist, _bins = np.histogram(fidelities, bins=bbs, density=False)
    probabilities = hist / len(fidelities)

    F = np.linspace(0, 1, len(probabilities))
    Q = (dim - 1) * (1 - F) ** (dim - 2)
    P = probabilities
    Q, P = np.array(Q), np.array(P)
    return np.average(np.abs(Q - P))

def get_final_state_viaQLG(states, Np, Nt, delta: float = 0):
    """Convenience wrapper used in the notebook."""
    from .state_prep import SP_bosonic
    _, _Fstate, _FU, _xopt, _F_opt, payload = SP_bosonic(states, int(Np), int(Nt), delta, logical=0)
    psi_fin = payload[0]
    return psi_fin.full()

def random_su2(seed=None):
    """Sample a random SU(2) matrix via normalized complex Gaussian."""
    rng = np.random.default_rng(seed)
    a = rng.normal() + 1j * rng.normal()
    b = rng.normal() + 1j * rng.normal()
    nrm = np.sqrt((abs(a) ** 2 + abs(b) ** 2))
    a /= nrm
    b /= nrm
    return np.array([[a, b], [-np.conj(b), np.conj(a)]], dtype=complex)
