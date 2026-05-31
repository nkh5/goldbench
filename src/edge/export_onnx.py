"""Export the best model to ONNX and benchmark baseline size + latency.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

from src.models.lgbm import get_feature_cols
import onnxruntime as ort
ort.set_default_logger_severity(3)  # 3 = ERROR only; hides the shape WARNINGs

FEATURES_PATH = Path("data/processed/features.parquet")
EDGE_DIR = Path("artifacts/edge")


def main():
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)
    feats = get_feature_cols(df)
    X = df[feats].astype(np.float32)
    y = df["y"]

    # Train on first 85% (chronological), hold out last 15% for benchmarking
    cut = int(0.85 * len(df))
    X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
    X_te = X.iloc[cut:]

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    # Save native LightGBM model + measure its size
    native_path = EDGE_DIR / "lgbm_native.txt"
    model.booster_.save_model(str(native_path))
    native_size = native_path.stat().st_size

    # Convert to ONNX
    initial_type = [("input", FloatTensorType([None, len(feats)]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type,
                                  zipmap=False)
    onnx_path = EDGE_DIR / "lgbm.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    onnx_size = onnx_path.stat().st_size

    # Benchmark inference latency: native LightGBM vs ONNX runtime
    X_te_np = X_te.values.astype(np.float32)

    # Native LightGBM timing
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.predict_proba(X_te_np)
    native_time = (time.perf_counter() - t0) / 100 * 1000  # ms per pass

    # ONNX runtime timing
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # The probability output is the one we want (skip the 'label' output that
    # triggers cosmetic shape warnings). Identify it by name.
    prob_output = [o.name for o in sess.get_outputs() if "prob" in o.name.lower()]
    out_names = prob_output if prob_output else [sess.get_outputs()[-1].name]

    t0 = time.perf_counter()
    for _ in range(100):
        _ = sess.run(out_names, {input_name: X_te_np})
    onnx_time = (time.perf_counter() - t0) / 100 * 1000

    native_pred = model.predict_proba(X_te_np)[:, 1]
    onnx_out = sess.run(out_names, {input_name: X_te_np})
    # Probability output is a list of dicts or a 2D array depending on version
    raw = onnx_out[0]
    if isinstance(raw, list):  # list of {0: p0, 1: p1} dicts
        onnx_pred = np.array([d[1] for d in raw])
    else:
        onnx_pred = np.array(raw)[:, 1] if np.array(raw).ndim == 2 else np.array(raw).ravel()
    max_diff = np.max(np.abs(native_pred - onnx_pred[:len(native_pred)]))
    
    # Verify ONNX predictions match native (sanity: export didn't corrupt model)
    native_pred = model.predict_proba(X_te_np)[:, 1]
    onnx_out = sess.run(None, {input_name: X_te_np})
    onnx_pred = np.array(onnx_out[1])[:, 1] if len(onnx_out) > 1 else np.array(onnx_out[0]).ravel()
    max_diff = np.max(np.abs(native_pred - onnx_pred[:len(native_pred)]))

    print("=== Edge Baseline Benchmark ===")
    print(f"Native LightGBM size: {native_size/1024:.1f} KB")
    print(f"ONNX size:            {onnx_size/1024:.1f} KB")
    print(f"Native latency:       {native_time:.3f} ms/batch")
    print(f"ONNX latency:         {onnx_time:.3f} ms/batch")
    print(f"Max pred difference (native vs ONNX): {max_diff:.6f}")
    print(f"Test rows: {len(X_te_np)}")


if __name__ == "__main__":
    main()