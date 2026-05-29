"""Helpers for the Nixtla neuralforecast data contract and CPCV integration."""
import numpy as np
import pandas as pd

# The 8 channels we feed the sequence models (raw-ish, stationary inputs)
NF_CHANNELS = [
    "ret_1", "daily_return", "ma_7_minus_ma_30", "ma_30_minus_ma_90",
    "rsi", "macd_minus_signal", "bb_position", "volatility_7",
]


def to_nf_format(df, target_col="y"):
    """Convert our feature frame into Nixtla's (unique_id, ds, y, +exog) long format."""
    out = pd.DataFrame({
        "unique_id": "gold",
        "ds": pd.to_datetime(df["date"]).values,
        "y": df[target_col].values.astype(float),
    })
    for ch in NF_CHANNELS:
        out[ch] = df[ch].values.astype(float)
    return out.reset_index(drop=True)