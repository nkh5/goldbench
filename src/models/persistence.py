"""Persistence baseline for volatility: tomorrow's vol = recent realized vol.

Computed from our OWN log returns on the same scale as the target (NOT the
dataset's pre-computed volatility_30).
"""
import numpy as np
import pandas as pd
from src.features_vol import FWD_WINDOW


def fit_predict_persistence(X_train, y_train, X_test):
    """Predict test vol as the trailing realized vol (past FWD_WINDOW days),
    computed from log returns on the same annualized scale as the target."""
    # Reconstruct the chronological return series across train+test
    train = X_train.copy()
    test = X_test.copy()
    full = pd.concat([train, test], ignore_index=True)

    # ret_1 is the past-only daily log return already in the features
    logret = full["ret_1"].values
    n_train = len(train)

    preds = []
    for i in range(len(test)):
        # Trailing window ENDS at the current row (uses only past+current returns)
        end = n_train + i
        window = logret[max(0, end - FWD_WINDOW + 1): end + 1]
        if len(window) < 2 or np.all(np.isnan(window)):
            preds.append(y_train.mean())
        else:
            trailing_vol = np.nanstd(window) * np.sqrt(252)
            preds.append(trailing_vol)
    return np.array(preds)