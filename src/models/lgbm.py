"""LightGBM on the full engineered feature set."""
import numpy as np
import lightgbm as lgb

# Columns that are NOT features (identifiers, raw price, the label)
NON_FEATURES = ["date", "y", "open", "high", "low", "close", "volume",
                "macd", "macd_signal", "bb_upper", "bb_lower",
                "ma_7", "ma_30", "ma_90"]


def get_feature_cols(X):
    """Everything except identifiers and raw-price columns is a feature."""
    return [c for c in X.columns if c not in NON_FEATURES]


def fit_predict_lgbm(X_train, y_train, X_test):
    """Fit LightGBM with early stopping on a recent-data watch-list.

    The validation watch-list is the LAST 15% of the training fold (most
    recent rows), never random rows, to avoid scattering future-adjacent
    observations into the watch-list.
    """
    feats = get_feature_cols(X_train)

    # Carve the most-recent 15% of the training fold as the early-stop watch-list
    n_val = max(30, int(0.15 * len(X_train)))
    X_tr, y_tr = X_train.iloc[:-n_val][feats], y_train.iloc[:-n_val]
    X_val, y_val = X_train.iloc[-n_val:][feats], y_train.iloc[-n_val:]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model.predict_proba(X_test[feats])[:, 1]