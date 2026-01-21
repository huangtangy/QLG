"""Bosonic code state constructors and small helpers.

This module is adapted to match the user's original helper file (the one you
uploaded later), so that names/behaviors are consistent with the notebook's
expected API.

Key functions provided:
  - bionomial_state
  - fourleg_cat_state (via rho_cat/Nj)
  - finite_gkp_state (finite-energy GKP approximation)
  - Cubic_phase_state
  - get_cnm
  - get_fid / get_psi_fidelity
  - displace (QuTiP displacement operator)
"""

from __future__ import annotations

import numpy as np

from qutip import (
    basis,
    coherent,
    ket2dm,
    displace,
    squeeze,
    destroy,
    fock,
    identity,
)


def get_fid(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Fidelity between two state vectors (array-like), matching the original helper."""
    psi1 = np.asarray(psi1)
    psi2 = np.asarray(psi2)
    return float(np.abs(np.sum(psi1.conj().T @ psi2)) ** 2)


def get_psi_fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Alias of state fidelity using vdot (array-like)."""
    psi1 = np.asarray(psi1)
    psi2 = np.asarray(psi2)
    return float(np.abs(np.vdot(psi1, psi2)) ** 2)


def Nj(j: int, alpha: float) -> float:
    """Normalization factors used for the 4-legged cat states."""
    if j == 0 or j == 2:
        nj = 8 * np.exp(-alpha**2) * (np.cosh(alpha**2) + np.cos(alpha**2) * (-1) ** (j / 2))
    elif j == 1 or j == 3:
        nj = 8 * np.exp(-alpha**2) * (np.sinh(alpha**2) + np.sin(alpha**2) * (-1) ** ((j - 1) / 2))
    else:
        raise ValueError("j must be in {0,1,2,3}")
    return float(nj)


def rho_cat(Np: int, alpha: float):
    """Produce the four cat states (kets) and their density matrices.

    Returns:
        (kets, rhos)
        where kets = [psicatc0, psicatc1, psicate0, psicate1]
    """
    alpa1 = coherent(Np, alpha)
    alpa2 = coherent(Np, -alpha)
    alpa3 = coherent(Np, 1j * alpha)
    alpa4 = coherent(Np, -1j * alpha)
    psicatc0 = (1 / np.sqrt(Nj(0, alpha))) * (alpa1 + alpa2 + alpa3 + alpa4)
    psicatc1 = (1 / np.sqrt(Nj(2, alpha))) * (alpa1 + alpa2 - alpa3 - alpa4)
    psicate0 = (1 / np.sqrt(Nj(1, alpha))) * (alpa1 - alpa2 - 1j * alpa3 + 1j * alpa4)
    psicate1 = (1 / np.sqrt(Nj(3, alpha))) * (alpa1 - alpa2 + 1j * alpa3 - 1j * alpa4)
    rhocatc0, rhocatc1, rhocate0, rhocate1 = (
        ket2dm(psicatc0),
        ket2dm(psicatc1),
        ket2dm(psicate0),
        ket2dm(psicate1),
    )
    return [psicatc0, psicatc1, psicate0, psicate1], [rhocatc0, rhocatc1, rhocate0, rhocate1]


def bionomial_state(Np: int):
    """Binomial ("Bio") code logical kets.

    Matches the user's helper implementation.
    Returns a list [|0_L>, |1_L>].
    """
    psi0 = 0.5 * basis(Np, 0) + np.sqrt(3 / 4) * basis(Np, 4)
    psi1 = 0.5 * basis(Np, 6) + np.sqrt(3 / 4) * basis(Np, 2)
    return [psi0, psi1]


def finite_gkp_state(dim: int, sigma: float, mu: int, n_range: int):
    """Finite-energy GKP approximation.

    Implementation matches the uploaded helper file and the reference it cites
    (PRX Quantum 3, 030301) parameterization.
    """
    gkp = 0
    delta = np.sqrt(np.pi) / 2
    for n1 in range(-n_range, n_range + 1):
        for n2 in range(-n_range, n_range + 1):
            alpha = delta * ((2 * n1 + mu) + 1j * n2)
            gaussian_weight = np.exp(-sigma**2 * abs(alpha) ** 2)
            phase = np.exp(-1j * np.real(alpha) * np.imag(alpha))
            displaced_state = displace(dim, alpha) * basis(dim, 0)
            gkp += gaussian_weight * phase * displaced_state
    return gkp.unit()


def Cubic_phase_state(cubicity: float, xi: complex, n_truncate: int):
    """Cubic phase state: exp(i*gamma*q^3) S(xi)|0> in Fock truncation."""
    a = destroy(n_truncate)
    gamm = cubicity
    cubic_operator = (1j * gamm * (a + a.dag()) ** 3).expm()
    state = cubic_operator * squeeze(n_truncate, xi) * fock(n_truncate, 0)
    return state


def fourleg_cat_state(n_truncate: int, alpha: float):
    """Return the 4-legged cat states [c0, c1, e0, e1]."""
    kets, _rhos = rho_cat(n_truncate, alpha)
    return kets


def get_cnm(N: int, H: np.ndarray):
    """Get the cnm list from Hamiltonian matrix H = sum_{n,m} c_{n,m} |n><m|.

    Matches the original helper implementation (threshold 1e-8).
    Returns a list of [c_nm, n, m].
    """
    H = np.asarray(H, dtype=complex)
    if H.shape != (N, N):
        raise ValueError(f"Expected H shape {(N, N)}, got {H.shape}")

    # Keep the structure for compatibility (Hnm is unused but preserved conceptually).
    _Hnm = identity(N) * 0.0
    cnm_list = []
    for n in range(N):
        for m in range(N):
            cnm = complex(H[n, m])
            if np.abs(cnm) > 1e-8:
                _Hnm += cnm * basis(N, n) * basis(N, m).dag()
                cnm_list.append([cnm, n, m])
    return cnm_list
