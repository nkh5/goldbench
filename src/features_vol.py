"""Build the volatility-forecasting target. Forward-looking, leakage-controlled.

Target: next-5-day annualized realized volatility = std of daily log returns
over days t+1..t+5, times sqrt(252). This depends on the FUTURE (like the
direction label), so it must be purged with a 5-day window in CV, wider than
the 1-day direction label.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_raw
from src.features import build_features

VOL_PROCESSED_PATH = Path("data/processed/features_vol.parquet")
VOL_HORIZON = 5  # trading days ahead


def build_vol_features(df):
    """Reuse the engineered features, swap in the forward-vol target.

    We keep all the same predictor columns from build_features (they use only
    past data), but replace the label with forward realized volatility.
    """
    feats = build_features(df)  # gives us all past-only features + the old 'y'

    # Recompute daily log returns on the full series for the forward window
    raw = df.copy().sort_values("date").reset_index(drop=True)
    raw["logret"] = np.log(raw["close"] / raw["close"].shift(1))

    # Forward realized vol: std of logret over the NEXT VOL_HORIZON days, annualized.
    # rolling() is backward-looking, so we reverse, roll, reverse back, then shift
    # so that row t sees days t+1..t+VOL_HORIZON (never day t itself).
    fwd_std = (
        raw["logret"][::-1]
        .rolling(VOL_HORIZON)
        .std()[::-1]
        .shift(-1)  # exclude day t, start at t+1
    )
    raw["y_vol"] = fwd_std * np.sqrt(252)

    # Merge the forward-vol target onto the feature frame by date
    vol_target = raw[["date", "y_vol"]]
    out = feats.merge(vol_target, on="date", how="left")

    # Drop the old direction label and any rows lacking a full forward window
    out = out.drop(columns=["y"]).rename(columns={"y_vol": "y"})
    out = out.dropna(subset=["y"]).reset_index(drop=True)
    return out


def main():
    raw = load_raw()
    vol = build_vol_features(raw)
    VOL_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    vol.to_parquet(VOL_PROCESSED_PATH, index=False)
    print(f"Wrote {len(vol)} rows to {VOL_PROCESSED_PATH}")
    print(f"Target (annualized vol) stats:")
    print(f"  mean {vol['y'].mean():.4f}, std {vol['y'].std():.4f}, "
          f"min {vol['y'].min():.4f}, max {vol['y'].max():.4f}")


if __name__ == "__main__":
    main()