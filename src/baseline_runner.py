"""Run naive + logistic + LightGBM through CPCV, save predictions and summary."""
import pandas as pd
from pathlib import Path
from src.eval import run_cpcv, summarize
from src.models.naive import fit_predict_prior
from src.models.logit import fit_predict_logit
from src.models.lgbm import fit_predict_lgbm

FEATURES_PATH = Path("data/processed/features.parquet")
ARTIFACTS = Path("artifacts")


def main():
    df = pd.read_parquet(FEATURES_PATH)
    y = df["y"]
    X = df.drop(columns=["y"])

    ARTIFACTS.mkdir(exist_ok=True)
    summaries = []

    models = [
        ("naive", fit_predict_prior),
        ("logit", fit_predict_logit),
        ("lgbm", fit_predict_lgbm),
    ]
    for name, fn in models:
        print(f"Running {name}...")
        preds = run_cpcv(fn, X, y)
        preds.to_parquet(ARTIFACTS / f"preds_{name}.parquet", index=False)
        summaries.append(summarize(preds, name))

    table = pd.DataFrame(summaries)
    table.to_parquet(ARTIFACTS / "summary_baselines.parquet", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Results (mean across 28 CPCV splits) ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()