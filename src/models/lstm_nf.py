"""LSTM via neuralforecast, trained as regression then isotonically calibrated.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.nf_utils import to_nf_format, NF_CHANNELS

warnings.filterwarnings("ignore")  # silence neuralforecast/lightning chatter

INPUT_SIZE = 60  # lookback window in trading days


def fit_predict_lstm(X_train, y_train, X_test):
    """Train LSTM as regression on the training fold, calibrate, predict test.

    Returns calibrated P(up) for each test row.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM
    from neuralforecast.losses.pytorch import MAE

    # Reassemble train/test into single frames with the target attached
    train_df = X_train.copy()
    train_df["y"] = y_train.values
    test_df = X_test.copy()

    # Carve last 20% of training fold as the calibration slice (chronological)
    n_cal = max(40, int(0.20 * len(train_df)))
    fit_df = train_df.iloc[:-n_cal]
    cal_df = train_df.iloc[-n_cal:]

    # neuralforecast needs the target series to be continuous in time for the
    # context window, so we feed it the fitting portion then forecast forward.
    nf_fit = to_nf_format(fit_df, target_col="y")

    model = LSTM(
        h=1,                          # forecast horizon: 1 step ahead
        input_size=INPUT_SIZE,
        hist_exog_list=NF_CHANNELS,   # the 8 channels as historical exogenous
        encoder_n_layers=1,
        encoder_hidden_size=32,
        decoder_hidden_size=32,
        decoder_layers=1,
        loss=MAE(),
        max_steps=500,
        learning_rate=1e-3,
        scaler_type="robust",
        random_seed=42,
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
    )
    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(df=nf_fit)

    # Helper: get raw one-step predictions for an arbitrary continuation frame
    def raw_predict(history_df, future_df):
        # Concatenate history + the row we want to predict, roll one step at a time
        # For simplicity and CPCV compatibility, we predict each test row using
        # the rolling history up to it.
        full = pd.concat([history_df, future_df], ignore_index=True)
        nf_full = to_nf_format(full, target_col="y" if "y" in full else None) \
            if "y" in full.columns else to_nf_format(
                full.assign(y=0.0))  # dummy y for exog alignment
        preds = nf.predict(df=to_nf_format(history_df))
        return preds["LSTM"].values

    # Calibration: predict raw values on the calibration slice, fit isotonic
    cal_raw = []
    hist = fit_df.copy()
    for i in range(len(cal_df)):
        p = nf.predict(df=to_nf_format(hist))["LSTM"].values[0]
        cal_raw.append(p)
        # extend history by the true calibration row to roll forward
        hist = pd.concat([hist, cal_df.iloc[[i]]], ignore_index=True)
    cal_raw = np.array(cal_raw)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(cal_raw, cal_df["y"].values)

    # Test prediction: roll through the test set the same way
    test_raw = []
    hist = train_df.copy()  # full training history precedes the test block
    for i in range(len(test_df)):
        p = nf.predict(df=to_nf_format(hist))["LSTM"].values[0]
        test_raw.append(p)
        # extend with the actual test row (features known; y unknown but unused for exog)
        row = test_df.iloc[[i]].copy()
        row["y"] = 0.0  # placeholder; not used to predict its own row
        hist = pd.concat([hist, row], ignore_index=True)
    test_raw = np.array(test_raw)

    return iso.transform(test_raw)