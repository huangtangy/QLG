"""State preparation demo (similar to the original notebook)."""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from qlg_refactor.state_prep import SP_bosonic
from qlg_refactor.plotting import plot_wigner_2mod

Nt, Np, delta = 32, 20, 1.0
states = 'Bio'

Hmat, Fstate, FU, xopt, F_opt, payload = SP_bosonic(states, Np, Nt, delta, logical=0)
psi_target, psif_before, psif_opt, U_est, U_before, Flist_before, Flist_opt = payload

print('Fidelity of QLG:', Fstate)
print('Fidelity of QLG with optimal control:', F_opt)

plot_wigner_2mod(Qobj(psif_before), xvec=np.linspace(-5, 5, 200), title='Before')
plot_wigner_2mod(Qobj(psif_opt), xvec=np.linspace(-5, 5, 200), title='After')
plot_wigner_2mod(Qobj(psi_target), xvec=np.linspace(-5, 5, 200), title='Target')

plt.figure(figsize=(5,2))
plt.subplot(1,2,1)
plt.plot(np.linspace(0, len(xopt), len(xopt)), xopt)
plt.xlabel('t/T')
plt.ylabel(r'$\beta(t)$')

plt.subplot(1,2,2)
plt.plot(1-np.array(Flist_before), label='before')
plt.plot(1-np.array(Flist_opt), label='opt')
plt.yscale('log')
plt.xlabel('t/T')
plt.ylabel('1-F')
plt.legend()
plt.tight_layout()
plt.show()
