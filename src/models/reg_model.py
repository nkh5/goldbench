"""Regression versions of the four ML models for volatility forecasting.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from src.models.logit import LOGIT_FEATURES
from src.models.lgbm import get_feature_cols
from src.nf_utils import to_nf_format, NF_CHANNELS

warnings.filterwarnings("ignore")
INPUT_SIZE = 60


# ---------- Ridge (linear baseline) ----------
def fit_predict_ridge(X_train, y_train, X_test):
    """Ridge regression on scaled scalar features (vol analog of logistic)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train[LOGIT_FEATURES], y_train)
    preds = pipe.predict(X_test[LOGIT_FEATURES])
    return np.clip(preds, 0.0, None)  # vol can't be negative


# ---------- LightGBM regressor ----------
def fit_predict_lgbm_reg(X_train, y_train, X_test):
    """LightGBM regressor on the wide engineered feature set."""
    feats = get_feature_cols(X_train)
    n_val = max(30, int(0.15 * len(X_train)))
    X_tr, y_tr = X_train.iloc[:-n_val][feats], y_train.iloc[:-n_val]
    X_val, y_val = X_train.iloc[-n_val:][feats], y_train.iloc[-n_val:]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, reg_lambda=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return np.clip(model.predict(X_test[feats]), 0.0, None)


# ---------- LSTM regression (no calibration needed) ----------
def fit_predict_lstm_reg(X_train, y_train, X_test):
    """LSTM regressing volatility, with a linear calibration layer.

    The raw LSTM captures volatility's SHAPE well (high correlation) but not
    its SCALE on limited noisy data. We fit a linear map (true ~ a*raw + b) on
    a held-out validation slice of the TRAINING fold, then apply it to test
    predictions. This is leakage-free (calibration uses only training data) and
    rescales the well-correlated raw output onto the correct magnitude.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM
    from neuralforecast.losses.pytorch import MAE
    from sklearn.linear_model import LinearRegression

    train_df = X_train.copy()
    train_df["y"] = y_train.values
    test_df = X_test.copy()

    # Hold out the last 20% of the training fold as the calibration slice
    n_cal = max(40, int(0.20 * len(train_df)))
    fit_df = train_df.iloc[:-n_cal]
    cal_df = train_df.iloc[-n_cal:]

    nf_fit = to_nf_format(fit_df, target_col="y")
    model = LSTM(
        h=1, input_size=INPUT_SIZE, hist_exog_list=NF_CHANNELS,
        encoder_n_layers=1, encoder_hidden_size=32,
        decoder_hidden_size=32, decoder_layers=1,
        loss=MAE(), max_steps=500, learning_rate=1e-3,
        scaler_type="robust", random_seed=42,
        enable_progress_bar=False, logger=False, accelerator="cpu",
    )
    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(df=nf_fit)

    # Raw predictions on the calibration slice (roll forward through it)
    cal_raw = []
    hist = fit_df.copy()
    for i in range(len(cal_df)):
        out = nf.predict(df=to_nf_format(hist))
        cal_raw.append(float(out["LSTM"].values[0]))
        hist = pd.concat([hist, cal_df.iloc[[i]]], ignore_index=True)
    cal_raw = np.array(cal_raw).reshape(-1, 1)

    # Fit the linear calibration: true_vol ~ a*raw + b
    calibrator = LinearRegression()
    calibrator.fit(cal_raw, cal_df["y"].values)

    # Raw predictions on the test set, then apply calibration
    test_raw = []
    hist = train_df.copy()
    for i in range(len(test_df)):
        out = nf.predict(df=to_nf_format(hist))
        test_raw.append(float(out["LSTM"].values[0]))
        row = test_df.iloc[[i]].copy()
        row["y"] = 0.0
        hist = pd.concat([hist, row], ignore_index=True)
    test_raw = np.array(test_raw).reshape(-1, 1)

    calibrated = calibrator.predict(test_raw)
    return np.clip(calibrated, 0.0, None)

