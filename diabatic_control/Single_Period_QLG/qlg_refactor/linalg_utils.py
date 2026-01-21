"""Linear-algebra helpers (Householder unitary, random states, etc.)."""
# Fix dtype usage in np.eye
import numpy as np
from numpy.linalg import norm

rng = np.random.default_rng(0)

def nz(x, tol=1e-12):
    return abs(x) < tol

def normalize(v):
    v = np.asarray(v, dtype=np.complex128).reshape(-1)
    n = norm(v)
    if n == 0:
        raise ValueError("Zero vector.")
    return v / n

def unitary_from_two_states(psi_i, psi_f):
    psi_i = normalize(psi_i)
    psi_f = normalize(psi_f)
    alpha = np.vdot(psi_f, psi_i)
    if np.isclose(abs(alpha), 1.0, rtol=0, atol=1e-12):
        phi = np.angle(alpha)
        N = psi_i.size
        return np.eye(N, dtype=np.complex128) + (np.exp(-1j*phi) - 1) * np.outer(psi_i, np.conjugate(psi_i))
    eiph = 1.0 + 0j if nz(abs(alpha)) else alpha/abs(alpha)
    u = normalize(psi_i - eiph*psi_f)
    H = np.eye(psi_i.size, dtype=np.complex128) - 2*np.outer(u, np.conjugate(u))
    P = np.eye(psi_i.size, dtype=np.complex128) + (np.conjugate(eiph) - 1)*np.outer(psi_f, np.conjugate(psi_f))
    return P @ H

def unitary_error(U):
    return norm(U.conj().T @ U - np.eye(U.shape[0], dtype=np.complex128))

def mapping_error(U, psi_i, psi_f):
    return norm(U @ normalize(psi_i) - normalize(psi_f))

def rand_state(N):
    v = rng.normal(size=N) + 1j*rng.normal(size=N)
    return normalize(v)
