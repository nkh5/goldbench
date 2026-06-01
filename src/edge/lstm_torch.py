"""A compact PyTorch LSTM for the edge-deployment study.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from src.nf_utils import NF_CHANNELS

FEATURES_PATH = Path("data/processed/features.parquet")
EDGE_DIR = Path("artifacts/edge")
INPUT_SIZE = 60
torch.manual_seed(42)
np.random.seed(42)


class SmallLSTM(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # take the final timestep
        return torch.sigmoid(self.head(last)).squeeze(-1)


def make_windows(df, channels, seq_len):
    """Build (n_windows, seq_len, n_channels) sequences and aligned labels."""
    feat = df[channels].values.astype(np.float32)
    y = df["y"].values.astype(np.float32)
    X, Y = [], []
    for i in range(seq_len, len(df)):
        X.append(feat[i - seq_len:i])
        Y.append(y[i])
    return np.array(X), np.array(Y)


def main():
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)

    X, Y = make_windows(df, NF_CHANNELS, INPUT_SIZE)
    # Chronological split: first 85% train, last 15% test
    cut = int(0.85 * len(X))
    Xtr, Ytr = X[:cut], Y[:cut]
    Xte, Yte = X[cut:], Y[cut:]

    # Per-feature standardization using TRAIN stats only (no leakage)
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    np.savez(EDGE_DIR / "lstm_scaler.npz", mean=mean, std=std)

    Xtr_t = torch.tensor(Xtr)
    Ytr_t = torch.tensor(Ytr)

    model = SmallLSTM(n_features=len(NF_CHANNELS), hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    n_epochs, batch = 30, 64
    for epoch in range(n_epochs):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, Ytr_t[idx])
            loss.backward()
            opt.step()

    # Save the trained model and the test set for the quantization benchmark
    torch.save(model.state_dict(), EDGE_DIR / "lstm_torch.pt")
    np.savez(EDGE_DIR / "lstm_test.npz", X=Xte, Y=Yte)

    # Quick sanity: test accuracy
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(Xte)).numpy()
    acc = ((p > 0.5).astype(int) == Yte).mean()
    print(f"Trained SmallLSTM. Test rows: {len(Xte)}, test accuracy: {acc:.4f}")
    print(f"Saved model + test set to {EDGE_DIR}")


if __name__ == "__main__":
    main()