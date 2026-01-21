"""Helpers to embed truncated Fock vectors/operators into a larger cutoff."""

from __future__ import annotations
import numpy as np

def embed_fock(obj: np.ndarray, M: int, keep_2d_state: bool = True) -> np.ndarray:
    """
    Embed a Fock-truncated wavefunction (state vector) or a unitary/operator
    from dimension N to M (M >= N).

    State input formats supported:
      - (N,)      1D ket
      - (N, 1)    column ket
      - (1, N)    row bra (treated as a vector, padded as row)

    Operator/unitary format:
      - (N, N)

    For states: zero-pad amplitudes for |N>,...,|M-1>.
    For unitaries/operators: direct-sum with identity, U_M = U_N ⊕ I_{M-N}.

    Args:
        obj: numpy array, wavefunction or square matrix
        M: target dimension
        keep_2d_state: if True, keep (N,1)/(1,N) shape for state inputs;
                       if False, always return 1D state (M,)

    Returns:
        embedded array in dimension M
    """
    obj = np.asarray(obj)

    # ---- State: (N,), (N,1), (1,N)
    if obj.ndim == 1:
        N = obj.shape[0]
        if M < N:
            raise ValueError(f"M must be >= N. Got M={M}, N={N}.")
        out = np.zeros((M,), dtype=np.result_type(obj, np.complex128))
        out[:N] = obj
        return out

    if obj.ndim == 2:
        r, c = obj.shape

        # column ket
        if c == 1 and r >= 1:
            N = r
            if M < N:
                raise ValueError(f"M must be >= N. Got M={M}, N={N}.")
            out = np.zeros((M, 1), dtype=np.result_type(obj, np.complex128))
            out[:N, 0] = obj[:, 0]
            if keep_2d_state:
                return out
            return out[:, 0]

        # row bra (or row vector)
        if r == 1 and c >= 1:
            N = c
            if M < N:
                raise ValueError(f"M must be >= N. Got M={M}, N={N}.")
            out = np.zeros((1, M), dtype=np.result_type(obj, np.complex128))
            out[0, :N] = obj[0, :]
            if keep_2d_state:
                return out
            return out[0, :]

        # ---- Operator/unitary: (N,N)
        if r == c:
            N = r
            if M < N:
                raise ValueError(f"M must be >= N. Got M={M}, N={N}.")
            out = np.eye(M, dtype=np.result_type(obj, np.complex128))
            out[:N, :N] = obj
            return out

        raise ValueError(
            f"Unsupported 2D shape {obj.shape}. "
            "State must be (N,1) or (1,N); operator must be (N,N)."
        )

    raise ValueError(f"obj must be a vector or a 2D array. Got ndim={obj.ndim}.")
