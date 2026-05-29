"""Load and validate the raw Kaggle gold futures dataset."""
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/gold_futures.csv")
EXPECTED_COLUMNS = {
    "date", "open", "high", "low", "close", "volume",
    "ma_7", "ma_30", "ma_90", "daily_return",
    "volatility_7", "volatility_30",
    "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw Kaggle CSV with light validation."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in raw data")

    return df


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.dtypes)