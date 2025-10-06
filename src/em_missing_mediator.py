
import numpy as np
import pandas as pd

def em_chain_missing_B(df, max_iter=50, tol=1e-6, seed=0):
    """
    EM for A->B->C when some B are missing (MAR).
    Estimates params: pB1|A0, pB1|A1, pC1|B0, pC1|B1.
    Returns a dict of parameters.
    """
    rng = np.random.default_rng(seed)
    data = df.copy()
    obs = data["B"].notna().values
    A = data["A"].values.astype(int)
    B = data["B"].fillna(0).values.astype(int)  # placeholder for indexing
    C = data["C"].values.astype(int)

    # Initialize from complete cases
    cc = data[obs]
    def smean(s, default=0.5):
        v = s.mean()
        return default if np.isnan(v) else float(v)

    pB1_A0 = smean(cc[cc["A"]==0]["B"])
    pB1_A1 = smean(cc[cc["A"]==1]["B"])
    pC1_B0 = smean(cc[cc["B"]==0]["C"])
    pC1_B1 = smean(cc[cc["B"]==1]["C"])

    last = -1e18
    for _ in range(max_iter):
        # E-step: posteriors for missing B via Bayes in the chain A->B->C
        post = np.zeros(len(data), dtype=float)
        for i in range(len(data)):
            if obs[i]:
                post[i] = float(B[i]==1)
                continue
            pB1 = pB1_A1 if A[i]==1 else pB1_A0
            pB0 = 1 - pB1
            if C[i]==1:
                num = pB1 * pC1_B1
                den = pB1 * pC1_B1 + pB0 * pC1_B0
            else:
                num = pB1 * (1 - pC1_B1)
                den = pB1 * (1 - pC1_B1) + pB0 * (1 - pC1_B0)
            post[i] = num / max(den, 1e-12)

        # M-step: fractional counts
        # pB1|A
        for a in (0,1):
            m = (A==a)
            denom = m.sum()
            num = post[m].sum()
            if a==0:
                pB1_A0 = num / max(denom, 1e-12)
            else:
                pB1_A1 = num / max(denom, 1e-12)

        # pC1|B
        w0 = (1 - post)
        w1 = post
        pC1_B0 = (w0 * C).sum() / max(w0.sum(), 1e-12)
        pC1_B1 = (w1 * C).sum() / max(w1.sum(), 1e-12)

        # simple convergence check: pseudo-ll
        ll = 0.0
        for i in range(len(data)):
            pB1 = pB1_A1 if A[i]==1 else pB1_A0
            pBi = pB1 if post[i] else (1 - pB1)
            pC1 = pC1_B1 if post[i] else pC1_B0
            ll += np.log(max(pBi,1e-12)) + (C[i]*np.log(max(pC1,1e-12)) + (1-C[i])*np.log(max(1-pC1,1e-12)))
        if abs(ll - last) < tol:
            break
        last = ll

    return dict(
        pB1_given_A0=float(pB1_A0),
        pB1_given_A1=float(pB1_A1),
        pC1_given_B0=float(pC1_B0),
        pC1_given_B1=float(pC1_B1),
    )
