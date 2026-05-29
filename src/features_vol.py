"""Build the volatility target: next-day close-to-close realized volatility.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_raw
from src.features import build_features

VOL_PROCESSED_PATH = Path("data/processed/features_vol.parquet")
FWD_WINDOW = 3  # short forward window to denoise the daily vol estimate


def build_vol_features(df):
    """Reuse engineered features, target = next-day close-to-close realized vol."""
    feats = build_features(df)

    raw = df.copy().sort_values("date").reset_index(drop=True)
    raw["logret"] = np.log(raw["close"] / raw["close"].shift(1))

    # Forward realized vol from close-to-close returns over the NEXT FWD_WINDOW
    # days, annualized. Reverse-roll-reverse to get a forward window, then
    # shift(-1) so row t sees days t+1..t+FWD_WINDOW (never day t).
    fwd_std = (
        raw["logret"][::-1]
        .rolling(FWD_WINDOW)
        .std()[::-1]
        .shift(-1)
    )
    raw["y_vol"] = fwd_std * np.sqrt(252)

    vol_target = raw[["date", "y_vol"]]
    out = feats.merge(vol_target, on="date", how="left")
    out = out.drop(columns=["y"]).rename(columns={"y_vol": "y"})
    out = out.dropna(subset=["y"]).reset_index(drop=True)
    return out


def main():
    raw = load_raw()
    vol = build_vol_features(raw)
    VOL_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    vol.to_parquet(VOL_PROCESSED_PATH, index=False)
    print(f"Wrote {len(vol)} rows to {VOL_PROCESSED_PATH}")
    print("Target (next-day close-to-close realized vol, annualized) stats:")
    print(f"  mean {vol['y'].mean():.4f}, std {vol['y'].std():.4f}, "
          f"min {vol['y'].min():.4f}, max {vol['y'].max():.4f}")


if __name__ == "__main__":
    main()