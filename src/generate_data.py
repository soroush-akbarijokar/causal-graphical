
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def gen_chain(n=5000, seed=0, confounded=False):
    """
    Generate data from A -> B -> C.
    Optionally introduce an unobserved confounder U -> {A, C}.
    Returns DataFrame with columns: A, B, C, (U if confounded).
    """
    rng = np.random.default_rng(seed)
    U = rng.integers(0, 2, size=n) if confounded else np.zeros(n, dtype=int)

    # A | U
    pA = sigmoid(-0.3 + 1.2*U)
    A = rng.binomial(1, pA, size=n)

    # B | A
    pB = sigmoid(-0.2 + 1.5*A)
    B = rng.binomial(1, pB, size=n)

    # C | B, U
    pC = sigmoid(-0.1 + 1.4*B + (1.0*U if confounded else 0.0))
    C = rng.binomial(1, pC, size=n)

    df = pd.DataFrame({"A": A, "B": B, "C": C})
    if confounded:
        df["U"] = U
    return df

def mask_mediator_B(df, frac_missing=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index, size=int(frac_missing*len(df)), replace=False)
    df_missing = df.copy()
    df_missing.loc[idx, "B"] = np.nan
    return df_missing
