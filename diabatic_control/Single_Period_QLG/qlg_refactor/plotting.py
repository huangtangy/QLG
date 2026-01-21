"""Plotting helpers used in the notebooks.

This module aligns with the user's original helper implementation
(`plot_winger_2mod`) while remaining backward compatible with the earlier
refactor API.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from qutip import Qobj, ket2dm, wigner


def plot_winger_2mod(psi, xvec: np.ndarray | None = None, title: str | None = None):
    """Plot Wigner + Fock PDF for a single-mode state.

    Notes:
      - Name kept as ``plot_winger_2mod`` (original spelling).
      - Accepts either a QuTiP ket/dm or an array-like vector.
      - Adds an optional ``title`` for compatibility with the refactored notebook.
    """

    if xvec is None:
        xvec = np.linspace(-5, 5, 200)

    # Normalize/convert
    if not isinstance(psi, Qobj):
        psi = Qobj(np.asarray(psi)).unit()
    else:
        psi = psi.unit() if psi.isket else psi

    rho = ket2dm(psi) if psi.isket else psi
    W0 = wigner(rho, xvec, xvec)

    coeffs = psi.full()
    coeffs = np.abs(coeffs) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.1))
    c0 = axes[0].contourf(xvec, xvec, W0, 100, cmap="RdBu_r")
    axes[0].set_title(title if title else r"$|\psi\rangle$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("p")
    plt.colorbar(c0, ax=axes[0])

    axes[1].plot(coeffs)
    axes[1].set_title("Fock PDF")
    axes[1].set_xlabel("Fock")
    axes[1].set_ylabel("Prob")

    plt.tight_layout()
    plt.show()
    return plt


# Backward-compatible alias (previous refactor name)
plot_wigner_2mod = plot_winger_2mod
