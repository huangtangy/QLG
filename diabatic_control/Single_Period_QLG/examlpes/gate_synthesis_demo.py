"""Gate synthesis demo (logical gate -> QLG control optimization)."""

import numpy as np
import matplotlib.pyplot as plt

from qlg_refactor.single_qubit_gates import H
from qlg_refactor.gate_synthesis import gate_bosonic_gate_vlist, gate_bosonic_gate, global_phase

Nt, delta = 32, 1.0
states = 'Bio'
Np = 10

gate = H
Vlist, Q, cnm_list = gate_bosonic_gate_vlist(gate, Np, Nt, delta, states)
F_before, gate_before, F_opt, gate_opt, xopt = gate_bosonic_gate(gate, Np, Nt, delta, Vlist, Q)
print('Fgate_before:', F_before, 'Fgate_opt:', F_opt)

# Visualize matrices (phase-aligned)
gate_before1 = np.exp(1j*global_phase(gate_before, gate)) * gate_before
gate_opt1 = np.exp(1j*global_phase(gate_opt, gate)) * gate_opt

plt.figure(figsize=(8,3))
vmin, vmax = -1, 1
plt.subplot(2,3,1); plt.title('Re gate_before'); plt.imshow(np.real(gate_before1), vmin=vmin, vmax=vmax); plt.colorbar()
plt.subplot(2,3,2); plt.title('Re gate_opt');    plt.imshow(np.real(gate_opt1), vmin=vmin, vmax=vmax); plt.colorbar()
plt.subplot(2,3,3); plt.title('Re target');      plt.imshow(np.real(gate), vmin=vmin, vmax=vmax); plt.colorbar()
plt.subplot(2,3,4); plt.title('Im gate_before'); plt.imshow(np.imag(gate_before1), vmin=vmin, vmax=vmax); plt.colorbar()
plt.subplot(2,3,5); plt.title('Im gate_opt');    plt.imshow(np.imag(gate_opt1), vmin=vmin, vmax=vmax); plt.colorbar()
plt.subplot(2,3,6); plt.title('Im target');      plt.imshow(np.imag(gate), vmin=vmin, vmax=vmax); plt.colorbar()
plt.tight_layout()
plt.show()
