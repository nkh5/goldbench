"""Export the PyTorch LSTM to ONNX, quantize to INT8, benchmark the tradeoff.

"""
import time
import numpy as np
import torch
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import roc_auc_score, accuracy_score

from src.edge.lstm_torch import SmallLSTM, INPUT_SIZE
from src.nf_utils import NF_CHANNELS

ort.set_default_logger_severity(3)
EDGE_DIR = Path("artifacts/edge")


def _bench(sess, X, reps=100):
    name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {name: X})
    return (time.perf_counter() - t0) / reps * 1000  # ms


def _prob(sess, X):
    name = sess.get_inputs()[0].name
    return np.array(sess.run(None, {name: X})[0]).ravel()


def main():
    # Load the trained model
    model = SmallLSTM(n_features=len(NF_CHANNELS), hidden=32)
    model.load_state_dict(torch.load(EDGE_DIR / "lstm_torch.pt"))
    model.eval()

    # Load test set
    d = np.load(EDGE_DIR / "lstm_test.npz")
    Xte, Yte = d["X"].astype(np.float32), d["Y"]

    # Export to ONNX with a dynamic batch axis
    fp32_path = EDGE_DIR / "lstm_fp32.onnx"
    dummy = torch.tensor(Xte)  # full test batch, fixed shape
    torch.onnx.export(
        model, dummy, str(fp32_path),
        input_names=["input"], output_names=["prob"],
        opset_version=18,
        dynamo=False,
    )
    # Quantize to INT8
    int8_path = EDGE_DIR / "lstm_int8.onnx"
    prep_path = EDGE_DIR / "lstm_fp32_prep.onnx"
    quant_pre_process(str(fp32_path), str(prep_path))
    quantize_dynamic(str(prep_path), str(int8_path), weight_type=QuantType.QInt8)
    fp32_kb = fp32_path.stat().st_size / 1024
    int8_kb = int8_path.stat().st_size / 1024

    sess32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    lat32, lat8 = _bench(sess32, Xte), _bench(sess8, Xte)
    p32, p8 = _prob(sess32, Xte), _prob(sess8, Xte)

    auc32 = roc_auc_score(Yte, p32) if len(np.unique(Yte)) > 1 else float("nan")
    auc8 = roc_auc_score(Yte, p8) if len(np.unique(Yte)) > 1 else float("nan")
    acc32 = accuracy_score(Yte, (p32 > 0.5).astype(int))
    acc8 = accuracy_score(Yte, (p8 > 0.5).astype(int))
    shift = np.max(np.abs(p32 - p8))

    print("=== LSTM Quantization Tradeoff (FP32 ONNX vs INT8 ONNX) ===")
    print(f"{'Metric':<22}{'FP32':>12}{'INT8':>12}{'Change':>14}")
    print(f"{'Size (KB)':<22}{fp32_kb:>12.1f}{int8_kb:>12.1f}"
          f"{(int8_kb/fp32_kb - 1)*100:>13.1f}%")
    print(f"{'Latency (ms)':<22}{lat32:>12.3f}{lat8:>12.3f}"
          f"{(lat8/lat32 - 1)*100:>13.1f}%")
    print(f"{'AUC':<22}{auc32:>12.4f}{auc8:>12.4f}{auc8 - auc32:>14.4f}")
    print(f"{'Accuracy':<22}{acc32:>12.4f}{acc8:>12.4f}{acc8 - acc32:>14.4f}")
    print(f"\nMax probability shift (fp32 vs int8): {shift:.5f}")


if __name__ == "__main__":
    main()