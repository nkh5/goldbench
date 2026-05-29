"""Build aligned features and labels. ZERO lookahead tolerated."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_raw

PROCESSED_PATH = Path("data/processed/features.parquet")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the feature matrix and binary direction label.

    All features at time t use only information from rows <= t.
    The label at row t is the direction of return from t to t+1.
    """
    out = df.copy()
    close = out["close"]

    # Log returns at multiple horizons (using only past prices)
    out["ret_1"] = np.log(close / close.shift(1))
    out["ret_5"] = np.log(close / close.shift(5))
    out["ret_10"] = np.log(close / close.shift(10))
    out["ret_20"] = np.log(close / close.shift(20))

    # Moving average crossovers, normalized
    out["ma_7_minus_ma_30"] = (out["ma_7"] - out["ma_30"]) / close
    out["ma_30_minus_ma_90"] = (out["ma_30"] - out["ma_90"]) / close

    # MACD differential and Bollinger position
    out["macd_minus_signal"] = out["macd"] - out["macd_signal"]
    out["bb_position"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

    # Calendar features
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)
    out["is_quarter_end"] = out["date"].dt.is_quarter_end.astype(int)

    # Lagged returns and rolling stats for LightGBM tier
    for lag in [1, 2, 3, 5, 10, 20]:
        out[f"ret_lag_{lag}"] = out["ret_1"].shift(lag)
    for w in [5, 10, 20]:
        out[f"ret_rollmean_{w}"] = out["ret_1"].rolling(w).mean().shift(1)
        out[f"ret_rollstd_{w}"] = out["ret_1"].rolling(w).std().shift(1)
        out[f"ret_rollmin_{w}"] = out["ret_1"].rolling(w).min().shift(1)
        out[f"ret_rollmax_{w}"] = out["ret_1"].rolling(w).max().shift(1)
        out[f"ret_rollskew_{w}"] = out["ret_1"].rolling(w).skew().shift(1)
        out[f"ret_rollkurt_{w}"] = out["ret_1"].rolling(w).kurt().shift(1)

    # Interaction terms
    out["rsi_x_vol7"] = out["rsi"] * out["volatility_7"]
    out["macd_x_ret5"] = out["macd"] * out["ret_5"]
    out["bb_x_vol30"] = out["bb_position"] * out["volatility_30"]

    # Indicator deltas
    out["rsi_delta_5"] = out["rsi"] - out["rsi"].shift(5)
    out["macd_delta_5"] = out["macd"] - out["macd"].shift(5)

    # THE LABEL: direction of next day's close. This is the ONLY shift(-1).
    out["y"] = (close.shift(-1) > close).astype(int)

    # Drop rows with any NaN from rolling/lag windows or the final label
    out = out.dropna().reset_index(drop=True)
    return out


def main():
    raw = load_raw()
    feats = build_features(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(PROCESSED_PATH, index=False)
    print(f"Wrote {len(feats)} rows x {len(feats.columns)} cols to {PROCESSED_PATH}")
    print(f"Label balance: {feats['y'].mean():.4f} (target ~0.50)")
    print(f"Date range: {feats['date'].min().date()} to {feats['date'].max().date()}")


if __name__ == "__main__":
    main()