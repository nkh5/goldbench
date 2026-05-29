"""Prove that features at time t depend only on data from rows <= t."""
import numpy as np
import pandas as pd
import pytest
from src.data import load_raw
from src.features import build_features

# Columns we expect to be deterministic functions of past+current data.
# We exclude 'y' because it's defined to depend on t+1 (that's the whole point).
FEATURE_COLS_TO_CHECK = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "ma_7_minus_ma_30", "ma_30_minus_ma_90",
    "macd_minus_signal", "bb_position",
    "ret_lag_1", "ret_lag_5", "ret_lag_10",
    "ret_rollmean_5", "ret_rollstd_10", "ret_rollmean_20",
    "rsi_x_vol7", "macd_x_ret5",
    "rsi_delta_5", "macd_delta_5",
]


@pytest.fixture(scope="module")
def full_features():
    return build_features(load_raw())


def test_label_uses_future(full_features):
    """Sanity check: y MUST depend on t+1, otherwise we have no learning task."""
    df = full_features
    # Confirm y[t] equals direction from close[t] to close[t+1]
    expected = (df["close"].shift(-1) > df["close"]).astype(int).iloc[:-1]
    actual = df["y"].iloc[:-1]
    assert (expected.reset_index(drop=True) == actual.reset_index(drop=True)).all()


def test_no_lookahead_in_features(full_features):
    """The hard test: poison future rows, recompute, and assert past features unchanged."""
    raw = load_raw()
    full = build_features(raw)

    # Pick a cut point well inside the series
    cut = len(raw) // 2
    cut_date = raw.iloc[cut]["date"]

    # Poison everything strictly after cut_date with NaN, except 'date' itself
    poisoned_raw = raw.copy()
    cols_to_poison = [c for c in poisoned_raw.columns if c != "date"]
    poisoned_raw.loc[poisoned_raw["date"] > cut_date, cols_to_poison] = np.nan

    poisoned = build_features(poisoned_raw)

    # For every row in poisoned (which only contains rows <= cut_date after dropna),
    # the feature values must match the corresponding row in the full dataset
    for col in FEATURE_COLS_TO_CHECK:
        if col not in poisoned.columns:
            continue
        merged = poisoned[["date", col]].merge(
            full[["date", col]], on="date", suffixes=("_poisoned", "_full")
        )
        diffs = (merged[f"{col}_poisoned"] - merged[f"{col}_full"]).abs()
        max_diff = diffs.max()
        assert max_diff < 1e-9, (
            f"Column '{col}' shows lookahead: max diff between poisoned and full = {max_diff}. "
            f"This means feature[t] depended on data from rows > t."
        )


def test_label_balance_sane(full_features):
    """Gold should be roughly balanced between up and down days."""
    bal = full_features["y"].mean()
    assert 0.45 < bal < 0.58, f"Label balance {bal:.3f} is suspicious"


def test_no_nans_in_processed(full_features):
    assert not full_features.isna().any().any()

def test_vol_target_forward_window():
    """Target at row t must equal close-to-close vol over days t+1..t+FWD_WINDOW."""
    from src.features_vol import build_vol_features, FWD_WINDOW
    import numpy as np
    raw = load_raw()
    vol = build_vol_features(raw)

    r = raw.sort_values("date").reset_index(drop=True).copy()
    r["logret"] = np.log(r["close"] / r["close"].shift(1))

    for pos in [50, 200, 400, 600]:
        if pos + FWD_WINDOW >= len(r):
            continue
        target_date = r["date"].iloc[pos]
        future = r["logret"].iloc[pos + 1: pos + 1 + FWD_WINDOW]
        expected = future.std() * np.sqrt(252)
        actual_row = vol.loc[vol["date"] == target_date, "y"]
        if len(actual_row) == 0:
            continue
        actual = actual_row.values[0]
        assert abs(expected - actual) < 1e-6, (
            f"Vol target at pos {pos}: expected {expected:.6f}, got {actual:.6f}"
        )

def test_vol_purge_wider_than_direction():
    """A 5-day label span must purge more training rows than a 1-day span."""
    from src.cv import get_cpcv_splits
    n = 1000
    span1 = list(get_cpcv_splits(n, label_span=1))
    span5 = list(get_cpcv_splits(n, label_span=5))
    # Same number of splits, but span5 training sets should be smaller on average
    avg1 = np.mean([len(tr) for tr, _ in span1])
    avg5 = np.mean([len(tr) for tr, _ in span5])
    assert avg5 < avg1, (
        f"5-day span avg train ({avg5:.0f}) should be < 1-day span ({avg1:.0f}); "
        "the wider purge isn't removing extra rows."
    )