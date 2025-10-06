
import numpy as np
import pandas as pd

def true_do_from_dgp_chain(params, a):
    """
    If you know the DGP params (for simulation truth), compute P(C=1 | do(A=a)).
    `params` expects:
        pB1_given_A0, pB1_given_A1, pC1_given_B0, pC1_given_B1
    """
    pB1 = params["pB1_given_A1"] if a == 1 else params["pB1_given_A0"]
    pB0 = 1 - pB1
    return params["pC1_given_B1"] * pB1 + params["pC1_given_B0"] * pB0

def plugin_backdoor_chain(df, a):
    """
    In chain A->B->C with no confounding, identification is:
      P(C|do(A=a)) = sum_b P(C|b) P(b|A=a)
    Estimate via empirical conditionals.
    """
    sub = df[df["A"] == a]
    pB1 = sub["B"].mean()
    pB0 = 1 - pB1
    pC1_B1 = df[df["B"] == 1]["C"].mean()
    pC1_B0 = df[df["B"] == 0]["C"].mean()
    return pC1_B1 * pB1 + pC1_B0 * pB0

def naive_associational(df, a):
    """P(C=1 | A=a) from data (biased if confounded)."""
    return df[df["A"] == a]["C"].mean()

def ipw(df, a):
    """
    Binary A IPW for E[C | do(A=a)]: E[ 1[A=a]*C / e(A) ] / E[ 1[A=a] / e(A) ]
    Here we have no X; we estimate e = P(A=1) from data (toy).
    """
    e = df["A"].mean()
    w = np.where(a==1, df["A"]/e, (1 - df["A"]) / (1 - e))
    return np.average(df["C"] * (df["A"]==a), weights=w)

def aipw_constantprop(df, a):
    """
    A simple AIPW variant without covariates (no X). Uses E[C|A] and prop score.
    """
    e = df["A"].mean()
    mu = df.groupby("A")["C"].mean().to_dict()
    comp = ((df["A"]==a) * (df["C"] - mu[a]) / (e if a==1 else (1-e))) + mu[a]
    return comp.mean()

def frontdoor(df, a):
    """
    Front-door adjustment with mediator B:
      sum_b [ sum_{a'} P(C|b,a') P(a') ] * P(b|a)
    """
    pA1 = df["A"].mean()
    pA0 = 1 - pA1

    pB1_given_a = df[df["A"]==a]["B"].mean()
    pB0_given_a = 1 - pB1_given_a

    def safe_mean(s, default):
        v = s.mean()
        return default if np.isnan(v) else v

    pC1_b1_a1 = safe_mean(df[(df["B"]==1) & (df["A"]==1)]["C"], df[df["B"]==1]["C"].mean())
    pC1_b1_a0 = safe_mean(df[(df["B"]==1) & (df["A"]==0)]["C"], df[df["B"]==1]["C"].mean())
    pC1_b0_a1 = safe_mean(df[(df["B"]==0) & (df["A"]==1)]["C"], df[df["B"]==0]["C"].mean())
    pC1_b0_a0 = safe_mean(df[(df["B"]==0) & (df["A"]==0)]["C"], df[df["B"]==0]["C"].mean())

    m_b1 = pC1_b1_a1 * pA1 + pC1_b1_a0 * pA0
    m_b0 = pC1_b0_a1 * pA1 + pC1_b0_a0 * pA0

    return m_b1 * pB1_given_a + m_b0 * pB0_given_a

def ate(estimator, df):
    return estimator(df, 1) - estimator(df, 0)
