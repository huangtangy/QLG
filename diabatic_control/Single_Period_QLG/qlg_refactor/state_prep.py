"""State preparation utilities used in the notebook.

This module provides:
- SP_bosonic: prepare a target state (random or bosonic-code logical) using QLG + GRAPE-like amplitude optimization.
- SP_bosonic_get_vlist: construct the baseline QLG Hamiltonian list (Vlist) for later reuse.
"""

from __future__ import annotations

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize
from joblib import Parallel, delayed

from qutip import Qobj, basis, fidelity

from .qlg import QLG
from .codes import bionomial_state, fourleg_cat_state, finite_gkp_state, displace, get_cnm
from .linalg_utils import unitary_from_two_states


def get_H_opt(t: float, cnm_list, amp: float, Np: int) -> np.ndarray:
    """Wrapper used across the notebook: H(t) = amp * V_Cat_t(t)."""
    q_lattice_gate = QLG(Np)
    return amp * q_lattice_gate.V_Cat_t(t, cnm_list)


def simulate_state(Vlist, Np: int, dt: float, lamd: float, H0, psi_in: Qobj, psi_target: Qobj, U_est: np.ndarray | None = None):
    """Evolve under the QLG Hamiltonian list and return fidelity trajectory."""
    Flist = []
    psit = psi_in.full()
    h0 = H0.full()
    U = np.identity(Np, dtype=complex)

    for h in Vlist:
        u = expm(-1j * h * dt / lamd) @ expm(-1j * h0 * dt / lamd)
        psit = u @ psit
        U = u @ U
        Flist.append(float(fidelity(Qobj(psit).unit(), psi_target)) ** 2)

    Fstate = Flist[-1] if Flist else float(fidelity(psi_in, psi_target)) ** 2
    FU = None
    if U_est is not None:
        FU = float(np.abs(np.trace(U @ U_est.conj().T)) / len(U))
    return psit, Fstate, FU, U, Flist


def SP_bosonic_get_vlist(states, Np: int, Nt: int, delta: float, logical: int = 0, n_jobs: int = -1):
    """Construct the baseline Hamiltonian list Vlist for the target (state) task."""
    qlg = QLG(Np)
    lamd, E, Dtau, H0, alpha = qlg.lamd, qlg.E, qlg.Dtau, qlg.H0, qlg.alpha

    # Build target state
    if isinstance(states, str):
        if states == 'Bio':
            logical0, logical1 = bionomial_state(Np)
        elif states == 'Cat':
            logical0, logical1, _psicate0, _psicate1 = fourleg_cat_state(Np, alpha)
        elif states == 'GKP':
            logical0 = finite_gkp_state(Np, 0.35, 0, 8)
            logical1 = displace(Np, np.sqrt(np.pi)) * logical0
        else:
            raise ValueError('wrong states')
        psi_target = logical0 if logical == 0 else logical1
    else:
        psi_target = Qobj(states).unit()

    psi_in = basis(Np, 0).unit()
    U_est = unitary_from_two_states(psi_in.full(), psi_target.full())
    H = scipy.linalg.logm(np.array(U_est)) * lamd / (-1j * E * Dtau)
    cnm_list = get_cnm(Np, H)

    tlist_finalT = np.linspace(0, Dtau, Nt + 1)
    Vlist = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(get_H_opt)(t, cnm_list, amp=1.0, Np=Np) for t in tlist_finalT[:]
    )
    return Vlist


def SP_bosonic(states, Np: int, Nt: int, delta: float, logical: int = 0, Vlist=None, n_jobs: int = -1):
    """Main entry point from the notebook.

    Args:
        states: 'Bio' | 'Cat' | 'GKP' or a custom state vector/array-like (Fock basis).
        Np: cutoff dimension
        Nt: Trotter steps
        delta: bound for control amplitudes (bounds are (1-delta, 1+delta))
        logical: which logical state to target for predefined codes
        Vlist: optional precomputed Hamiltonian list (from SP_bosonic_get_vlist)
        n_jobs: joblib parallelism for constructing Vlist

    Returns:
        H, Fstate, FU, xopt, F_opt, payload
    """
    qlg = QLG(Np)
    lamd, E, Dtau, H0, alpha = qlg.lamd, qlg.E, qlg.Dtau, qlg.H0, qlg.alpha

    # Build target state
    if isinstance(states, str):
        if states == 'Bio':
            logical0, logical1 = bionomial_state(Np)
        elif states == 'Cat':
            logical0, logical1, _psicate0, _psicate1 = fourleg_cat_state(Np, alpha)
        elif states == 'GKP':
            logical0 = finite_gkp_state(Np, 0.35, 0, 8)
            logical1 = displace(Np, np.sqrt(np.pi)) * logical0
        else:
            raise ValueError('wrong states')
        psi_target = logical0 if logical == 0 else logical1
    else:
        psi_target = Qobj(states).unit()

    psi_in = basis(Np, 0).unit()

    # Householder unitary mapping |0> -> |psi_target>
    U_est = unitary_from_two_states(psi_in.full(), psi_target.full())
    H = scipy.linalg.logm(np.array(U_est)) * lamd / (-1j * E * Dtau)
    cnm_list = get_cnm(Np, H)

    tlist_finalT = np.linspace(0, Dtau, Nt + 1)
    dt = tlist_finalT[1] - tlist_finalT[0]

    if Vlist is None:
        Vlist = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(get_H_opt)(t, cnm_list, amp=1.0, Np=Np) for t in tlist_finalT[:]
        )

    psif_before, Fstate, FU, U_before, Flist_before = simulate_state(
        Vlist, Np, dt, lamd, H0, psi_in, psi_target, U_est=U_est
    )

    # Optimize amplitude scalings on each time slice (GRAPE-like)
    def cost_function(params):
        Vlist1 = [params[i] * Vlist[i] for i in range(len(Vlist))]
        psit = psi_in.full()
        h0 = H0.full()
        U = np.identity(Np, dtype=complex)
        for h in Vlist1:
            u = expm(-1j * h * dt / lamd) @ expm(-1j * h0 * dt / lamd)
            psit = u @ psit
            U = u @ U
        Ffinal = float(fidelity(Qobj(psit).unit(), psi_target)) ** 2
        return 1 - Ffinal

    bounds = [(1 - delta, 1 + delta)] * len(tlist_finalT)
    initial_guess = np.random.random(len(tlist_finalT))
    result = minimize(cost_function, initial_guess, method='SLSQP', bounds=bounds, tol=1e-7)
    xopt = result.x

    Vlist_opt = [xopt[i] * Vlist[i] for i in range(len(Vlist))]
    psif_opt, F_opt, FUopt, U_opt, Flist_opt = simulate_state(
        Vlist_opt, Np, dt, lamd, H0, psi_in, psi_target, U_est=U_est
    )

    payload = [psi_target, psif_before, psif_opt, U_est, U_before, Flist_before, Flist_opt]
    return H, Fstate, FU, xopt, F_opt, payload
