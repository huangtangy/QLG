"""Compatibility shim.

Your original notebook referenced a standalone ``functions.py``.
After refactor, the implementations live under ``qlg_refactor/``.

This file re-exports the same names so you can keep old imports working:

  from functions import bionomial_state, plot_winger_2mod, get_cnm, ...
"""

from qlg_refactor.codes import (
    get_fid,
    get_psi_fidelity,
    Nj,
    rho_cat,
    bionomial_state,
    finite_gkp_state,
    Cubic_phase_state,
    fourleg_cat_state,
    get_cnm,
    displace,
)

from qlg_refactor.plotting import plot_winger_2mod
