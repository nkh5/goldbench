"""Shared evaluation utilities: run a model through CPCV and score it."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from src.cv import get_cpcv_splits


def run_cpcv(fit_predict_fn, X, y, n_blocks=8, n_test_blocks=2, embargo=5):
    """Run a model across all CPCV splits and collect out-of-sample predictions.

    fit_predict_fn(X_train, y_train, X_test) must return a 1D array of
    predicted P(up) for the test rows.

    Returns a long DataFrame with columns: split_id, row_idx, y_true, y_prob.
    """
    records = []
    splits = list(get_cpcv_splits(len(X), n_blocks, n_test_blocks, embargo))
    for split_id, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        proba = fit_predict_fn(X_train, y_train, X_test)
        for row_idx, p, yt in zip(test_idx, proba, y.iloc[test_idx]):
            records.append((split_id, int(row_idx), int(yt), float(p)))
    return pd.DataFrame(records, columns=["split_id", "row_idx", "y_true", "y_prob"])


def score_per_split(preds_df):
    """Compute AUC, accuracy, Brier per split. Returns one row per split."""
    rows = []
    for sid, g in preds_df.groupby("split_id"):
        yt, yp = g["y_true"].values, g["y_prob"].values
        pred_label = (yp > 0.5).astype(int)
        # AUC is undefined if a split's test set is all one class; guard it
        auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan
        rows.append({
            "split_id": sid,
            "auc": auc,
            "accuracy": accuracy_score(yt, pred_label),
            "brier": brier_score_loss(yt, yp),
        })
    return pd.DataFrame(rows)


def summarize(preds_df, model_name):
    """Collapse per-split scores into mean +/- std summary for one model."""
    s = score_per_split(preds_df)
    return {
        "model": model_name,
        "auc_mean": s["auc"].mean(),
        "auc_std": s["auc"].std(),
        "acc_mean": s["accuracy"].mean(),
        "acc_std": s["accuracy"].std(),
        "brier_mean": s["brier"].mean(),
    }

def regression_metrics(y_true, y_pred):
    """Volatility-forecasting metrics. Correlation and QLIKE are the headline
    metrics (R-squared is unreliable across heterogeneous CV folds for vol)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))

    # Correlation: does the forecast track when vol rises/falls? (headline)
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = np.nan

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    eps = 1e-8
    yp = np.clip(y_pred, eps, None)
    yt = np.clip(y_true, eps, None)
    qlike = np.mean(np.log(yp ** 2) + (yt ** 2) / (yp ** 2))

    return {"rmse": rmse, "mae": mae, "corr": corr, "r2": r2, "qlike": qlike}


def run_cpcv_regression(fit_predict_fn, X, y, n_blocks=8, n_test_blocks=2,
                        embargo=5, label_span=3):
    """Run a regression model across CPCV splits with the wider vol purge.

    Note label_span=3 by default, since the vol target looks 3 days forward.
    Returns a long DataFrame: split_id, row_idx, y_true, y_pred.
    """
    records = []
    splits = list(get_cpcv_splits(len(X), n_blocks, n_test_blocks, embargo, label_span))
    for split_id, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        pred = fit_predict_fn(X_train, y_train, X_test)
        for row_idx, p, yt in zip(test_idx, pred, y.iloc[test_idx]):
            records.append((split_id, int(row_idx), float(yt), float(p)))
    return pd.DataFrame(records, columns=["split_id", "row_idx", "y_true", "y_pred"])


def score_regression_per_split(preds_df):
    """Compute regression metrics per split."""
    rows = []
    for sid, g in preds_df.groupby("split_id"):
        m = regression_metrics(g["y_true"].values, g["y_pred"].values)
        m["split_id"] = sid
        rows.append(m)
    return pd.DataFrame(rows)

def summarize_regression(preds_df, model_name):
    """Collapse per-split regression scores into a summary row."""
    s = score_regression_per_split(preds_df)
    return {
        "model": model_name,
        "rmse_mean": s["rmse"].mean(),
        "corr_mean": s["corr"].mean(),
        "corr_std": s["corr"].std(),
        "qlike_mean": s["qlike"].mean(),
        #"r2_mean": s["r2"].mean(),
    }