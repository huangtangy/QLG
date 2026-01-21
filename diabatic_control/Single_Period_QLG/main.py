#!/usr/bin/env python3
"""Minimal entry point (refactored from the notebook).

Usage examples:
  python main.py --mode state --states Bio --Np 20 --Nt 32 --delta 1
  python main.py --mode gate  --gate H    --states Bio --Np 10 --Nt 32 --delta 1

Notes:
- This project expects QuTiP and SciPy installed.
- See `examples/` for more detailed scripts.
"""

from __future__ import annotations

import argparse
import numpy as np

from qlg_refactor.state_prep import SP_bosonic
from qlg_refactor.plotting import plot_wigner_2mod
from qlg_refactor.gate_synthesis import gate_bosonic_gate_vlist, gate_bosonic_gate
from qlg_refactor.single_qubit_gates import H, S, T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['state', 'gate'], default='state')
    p.add_argument('--states', default='Bio', help="Bio | Cat | GKP (for predefined codes)")
    p.add_argument('--Np', type=int, default=20)
    p.add_argument('--Nt', type=int, default=32)
    p.add_argument('--delta', type=float, default=1.0)
    p.add_argument('--logical', type=int, default=0)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--gate', choices=['H', 'S', 'T'], default='H')
    p.add_argument('--n_jobs', type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'state':
        Hmat, Fstate, FU, xopt, F_opt, payload = SP_bosonic(
            args.states, args.Np, args.Nt, args.delta, logical=args.logical, n_jobs=args.n_jobs
        )
        psi_target, psif_before, psif_opt, U_est, U_before, Flist_before, Flist_opt = payload
        print('Fidelity of QLG:', Fstate)
        print('Fidelity of QLG with optimal control:', F_opt)

        if args.plot:
            plot_wigner_2mod(psif_before, title='Before optimization')
            plot_wigner_2mod(psif_opt, title='After optimization')
            plot_wigner_2mod(psi_target, title='Target state')

    else:
        gate = {'H': H, 'S': S, 'T': T}[args.gate]
        Vlist, Q, cnm_list = gate_bosonic_gate_vlist(
            gate, args.Np, args.Nt, args.delta, args.states, n_jobs=args.n_jobs
        )
        F_before, gate_before, F_opt, gate_opt, xopt = gate_bosonic_gate(
            gate, args.Np, args.Nt, args.delta, Vlist, Q
        )
        print('Gate fidelity before:', F_before)
        print('Gate fidelity after :', F_opt)


if __name__ == '__main__':
    main()
