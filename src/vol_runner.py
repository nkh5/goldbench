"""Run volatility-forecasting models through CPCV with the wider purge."""
import pandas as pd
from pathlib import Path
from src.eval import run_cpcv_regression, summarize_regression
from src.models.garch import fit_predict_garch
from src.models.persistence import fit_predict_persistence
from src.models.persistence import fit_predict_persistence
from src.models.garch import fit_predict_garch
from src.models.reg_model import (
    fit_predict_ridge, fit_predict_lgbm_reg,
    fit_predict_lstm_reg, 
)

VOL_FEATURES_PATH = Path("data/processed/features_vol.parquet")
ARTIFACTS = Path("artifacts")


def main():
    df = pd.read_parquet(VOL_FEATURES_PATH)
    y = df["y"]
    X = df.drop(columns=["y"])

    ARTIFACTS.mkdir(exist_ok=True)
    summaries = []

    models = [
        ("persistence", fit_predict_persistence),
        ("garch", fit_predict_garch),
        ("ridge", fit_predict_ridge),
        ("lgbm_reg", fit_predict_lgbm_reg),
        ("lstm_reg", fit_predict_lstm_reg),
    ]
    for name, fn in models:
        print(f"Running {name}...")
        preds = run_cpcv_regression(fn, X, y)
        preds.to_parquet(ARTIFACTS / f"vol_preds_{name}.parquet", index=False)
        summaries.append(summarize_regression(preds, name))

    table = pd.DataFrame(summaries)
    table.to_parquet(ARTIFACTS / "summary_vol.parquet", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Volatility Results (mean across 28 CPCV splits) ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()