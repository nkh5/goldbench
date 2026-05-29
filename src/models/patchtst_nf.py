"""PatchTST via neuralforecast, trained directly with Bernoulli loss.

"""
import warnings
import numpy as np
import pandas as pd

from src.nf_utils import to_nf_format

warnings.filterwarnings("ignore")

INPUT_SIZE = 60


def fit_predict_patchtst(X_train, y_train, X_test):
    """Train PatchTST with Bernoulli loss on the target's own history only."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.losses.pytorch import DistributionLoss

    train_df = X_train.copy()
    train_df["y"] = y_train.values
    test_df = X_test.copy()

    nf_fit = to_nf_format(train_df, target_col="y")

    model = PatchTST(
        h=1,
        input_size=INPUT_SIZE,
        patch_len=8,
        stride=8,
        encoder_layers=2,
        n_heads=4,
        hidden_size=32,
        linear_hidden_size=64,
        dropout=0.3,
        fc_dropout=0.3,
        attn_dropout=0.1,
        revin=True,
        loss=DistributionLoss(distribution="Bernoulli"),
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

    preds = []
    hist = train_df.copy()
    for i in range(len(test_df)):
        out = nf.predict(df=to_nf_format(hist))
        col = "PatchTST" if "PatchTST" in out.columns else out.columns[-1]
        preds.append(float(out[col].values[0]))
        row = test_df.iloc[[i]].copy()
        row["y"] = 0.0
        hist = pd.concat([hist, row], ignore_index=True)

    return np.clip(np.array(preds), 0.0, 1.0)