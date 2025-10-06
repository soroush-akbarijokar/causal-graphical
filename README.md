# Causal & Graphical Models — Mini Project
**Course Project — Carnegie Mellon University, 10-701 Machine Learning**

This repository contains a small, self-contained project exploring the interface between **probabilistic graphical models** and **causal inference** using a simple chain DAG (A→B→C). It was developed to accompany course material on Bayesian networks, conditional independence, and causal identification.

## Contents
- `src/generate_data.py` — synthetic data generators for the chain DAG (A→B→C) with optional unobserved confounding `U` and a helper to mask the mediator.
- `src/estimators.py` — estimators for interventional targets: naïve association, back-door plug‑in, **front‑door**, **IPW**, and **AIPW**; plus small utilities.
- `src/em_missing_mediator.py` — EM routine for the case where the mediator `B` is partially missing (MAR).
- `experiments/run_chain_experiment.py` — end‑to‑end script that generates data, estimates the ATE, and saves a comparison figure.
- `experiments/run_sweep.py` — bias/variance sweeps across sample sizes and confounding settings; writes CSVs and plots.
- `experiments/interactive_demo.ipynb` — lightweight widget demo (n, confounding, missing‑B).
- `docs/` — figures produced by the experiments and a short `report.md`.
- `requirements.txt` — minimal dependencies.

## Learning objectives
- Distinguish **associational** quantities from **interventional** targets (do‑operator).
- Apply **identification** on a DAG (back‑door and front‑door) before estimation.
- Implement and compare classical estimators (plug‑in, IPW/AIPW) and handle missing mediators via **EM**.
- Practice reproducibility (scripts/figures, tests, and CI).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate    # or your preferred env
pip install -r requirements.txt
python experiments/run_chain_experiment.py           # saves docs/chain_compare.png
python experiments/run_sweep.py                      # saves sweep plots + CSVs
```
Optional interactive demo:
```bash
pip install ipywidgets
# Open experiments/interactive_demo.ipynb in Jupyter
```

## Notes
- The project is intentionally small and didactic: just enough code to connect **Bayesian-network** intuition with **causal** estimands on the same DAG.
- Plots and a brief summary live in `docs/`. Tests in `tests/` provide a quick sanity check.

MIT License © 2025
