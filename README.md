# Causal & Graphical Models — Mini Project

**Goal:** A compact, reproducible project that demonstrates PhD-level readiness in **causal modeling** using simple **graphical models** (DAGs). It bridges *probabilistic* and *causal* semantics via a small set of experiments and estimators.

## Contents
- `src/generate_data.py` — synthetic data generators for a chain DAG (A→B→C) with optional confounding U.
- `src/estimators.py` — identification-aware estimators: plug-in/back-door, IPW, AIPW, and front-door; plus utilities.
- `src/em_missing_mediator.py` — EM for a **partially observed mediator B** in the chain DAG.
- `experiments/run_chain_experiment.py` — end-to-end script to generate data, estimate effects, and save a figure.
- `docs/` — figures produced by the experiments.
- `requirements.txt` — lightweight dependencies.

## Highlights
- **Identification vs estimation**: we compute the true interventional target \( P(C\mid do(A)) \) and compare it to:
  - naive associational \( P(C\mid A) \) (biased under confounding),
  - **back-door** (when a sufficient adjustment set exists),
  - **front-door** (when confounding exists but a mediator opens identification),
  - **IPW/AIPW** estimators.
- **Graphical intuition**: the same DAG can be read probabilistically (Bayesian network) or causally; the project shows why the *do-operator* changes the answer.
- **Missing data**: EM recovers parameters for \( P(B\mid A) \) and \( P(C\mid B) \) when B is missing-at-random, then plugs them into the identification formula.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_chain_experiment.py
```
Outputs go to `docs/` (including `chain_compare.png`).

## What this demonstrates
- Ability to **specify a causal estimand** (ATE) on a DAG and derive **identification** formulas.
- Implementation of **classical estimators** (IPW/AIPW; front-door) and **EM** for missing mediators.
- Clear experiments with **reproducible** code and plots.

MIT License © 2025
