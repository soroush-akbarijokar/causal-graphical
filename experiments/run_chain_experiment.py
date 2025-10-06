
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.generate_data import gen_chain, mask_mediator_B
from src.estimators import naive_associational, plugin_backdoor_chain, frontdoor, ipw, aipw_constantprop, ate
from src.estimators import true_do_from_dgp_chain
from src.em_missing_mediator import em_chain_missing_B

def run(n=10000, confounded=True, seed=0, missing_frac=0.2, make_plot=True):
    df = gen_chain(n=n, seed=seed, confounded=confounded)

    # "Oracle" truth from empirical conditionals of the DGP
    params = dict(
        pB1_given_A0 = df[df["A"]==0]["B"].mean(),
        pB1_given_A1 = df[df["A"]==1]["B"].mean(),
        pC1_given_B0 = df[df["B"]==0]["C"].mean(),
        pC1_given_B1 = df[df["B"]==1]["C"].mean(),
    )

    true1 = true_do_from_dgp_chain(params, 1)
    true0 = true_do_from_dgp_chain(params, 0)
    true_ate = true1 - true0

    ate_naive = ate(naive_associational, df)
    ate_plugin = ate(plugin_backdoor_chain, df)
    ate_front  = ate(frontdoor, df)
    ate_ipw    = ate(ipw, df)
    ate_aipw   = ate(aipw_constantprop, df)

    df_miss = mask_mediator_B(df, frac_missing=missing_frac, seed=seed)
    params_em = em_chain_missing_B(df_miss, max_iter=200, tol=1e-7, seed=seed)

    def plugin_from_params(params, a):
        pB1 = params["pB1_given_A1"] if a==1 else params["pB1_given_A0"]
        pB0 = 1 - pB1
        return params["pC1_given_B1"]*pB1 + params["pC1_given_B0"]*pB0
    ate_em = plugin_from_params(params_em, 1) - plugin_from_params(params_em, 0)

    results = {
        "true_ate": true_ate,
        "naive_association": ate_naive,
        "plugin_backdoor": ate_plugin,
        "frontdoor": ate_front,
        "ipw": ate_ipw,
        "aipw": ate_aipw,
        "em_missing_B": ate_em,
    }

    if make_plot:
        labels = list(results.keys())
        values = [results[k] for k in labels]
        plt.figure(figsize=(7,4.5))
        plt.axhline(true_ate, linestyle="--", label="True ATE")
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.ylabel("ATE (Causal Risk Difference)")
        plt.title(f"ATE estimators vs. truth (confounded={confounded}, missing B={missing_frac*100:.0f}%)")
        plt.legend()
        import os
        os.makedirs("docs", exist_ok=True)
        out = "docs/chain_compare.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved plot to {out}")

    return results

if __name__ == "__main__":
    res = run(n=20000, confounded=True, seed=7, missing_frac=0.25, make_plot=True)
    for k,v in res.items():
        print(f"{k:>20s}: {v:+.4f}")
