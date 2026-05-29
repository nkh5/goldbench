"""Train one LightGBM on the full series and compute SHAP values for the demo."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from src.models.lgbm import get_feature_cols

FEATURES_PATH = Path("data/processed/features.parquet")
ARTIFACTS = Path("artifacts")


def main():
    df = pd.read_parquet(FEATURES_PATH)
    feats = get_feature_cols(df)
    y = df["y"]

    # Use the first 85% to train, last 15% as the explained set (chronological)
    n = len(df)
    cut = int(0.85 * n)
    X_tr, y_tr = df.iloc[:cut][feats], y.iloc[:cut]
    X_explain = df.iloc[cut:][feats]
    dates_explain = df.iloc[cut:]["date"].values

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, reg_lambda=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    # LightGBM binary returns a single array in recent SHAP; handle both shapes
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    ARTIFACTS.mkdir(exist_ok=True)
    np.savez_compressed(
        ARTIFACTS / "shap.npz",
        shap_values=shap_values,
        feature_values=X_explain.values,
        feature_names=np.array(feats),
        dates=dates_explain,
        base_value=np.array([explainer.expected_value if np.isscalar(explainer.expected_value)
                             else explainer.expected_value[-1]]),
    )
    model.booster_.save_model(str(ARTIFACTS / "lgbm_shap_model.txt"))

    # Print the top features by mean absolute SHAP (global importance)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("Top 10 features by mean |SHAP|:")
    for i in order[:10]:
        print(f"  {feats[i]:25s} {mean_abs[i]:.5f}")


if __name__ == "__main__":
    main()