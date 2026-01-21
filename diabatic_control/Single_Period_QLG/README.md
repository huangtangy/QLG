# Notebook Refactor Output

This folder is a refactor of `main.ipynb` into a small Python package so the *main file* stays simple.

## Structure

- `main.py` – minimal CLI entry point (state prep / gate synthesis)
- `qlg_refactor/` – extracted functions/classes
  - `qlg.py` – `QLG` class
  - `state_prep.py` – `SP_bosonic` + helpers
  - `gate_synthesis.py` – logical gate embedding + optimization
  - `linalg_utils.py` – Householder unitary + small linear algebra helpers
  - `embedding.py` – `embed_fock`
  - `codes.py` – bosonic code states + `get_cnm`
  - `plotting.py` – Wigner plotting helper
  - `single_qubit_gates.py` – H/S/T and a Clifford list
  - `random_utils.py` – Haar random state utilities
- `examples/` – scripts corresponding to the original notebook demos

## Quick start

```bash
python main.py --mode state --states Bio --Np 20 --Nt 32 --delta 1 --plot
python main.py --mode gate  --gate H   --states Bio --Np 10 --Nt 32 --delta 1
```

## Notes

- The original notebook relies on **QuTiP**, SciPy, joblib, matplotlib.
- For compatibility with your original notebook imports, a top-level `functions.py` is included, re-exporting
  the key helpers (e.g. `bionomial_state`, `fourleg_cat_state`, `finite_gkp_state`, `plot_winger_2mod`, `get_cnm`).
