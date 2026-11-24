"""Exports the deployed model to ONNX, quantises it to INT8 (static, calibrated on validation
images only), and measures what INT8 costs and gains: size, single-image latency, accuracy,
agreement with FP32. Writes models/clearscan_fp32.onnx, models/clearscan_int8.onnx and
results/quantization.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .data import splits
from .model import SkinNet, to_nchw

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


class _Reader(CalibrationDataReader):
    def __init__(self, x):
        self.it = iter([{"image": to_nchw(x[i:i + 1]).float().numpy()} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(sess, x_nhwc, batch=256):
    return np.concatenate([sess.run(None, {"image": to_nchw(x_nhwc[i:i + batch]).float().numpy()})[0]
                           for i in range(0, len(x_nhwc), batch)])


def latency_ms(sess, x_nhwc, n=300):
    xs = [to_nchw(x_nhwc[i:i + 1]).float().numpy() for i in range(50)]
    for i in range(30):
        sess.run(None, {"image": xs[i % 50]})
    t = []
    for i in range(n):
        t0 = time.perf_counter(); sess.run(None, {"image": xs[i % 50]}); t.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(t, 50)), float(np.percentile(t, 99))


def main(run_name="lesion_pretrained_s0"):
    MODELS.mkdir(exist_ok=True)
    model = SkinNet()
    model.load_state_dict(torch.load(ROOT / "results" / "runs" / run_name / "model.pt", map_location="cpu"))
    model.eval()
    fp32, pre, int8 = MODELS / "clearscan_fp32.onnx", MODELS / "_pre.onnx", MODELS / "clearscan_int8.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 28, 28), str(fp32), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    sp = splits("lesion")
    quant_pre_process(str(fp32), str(pre))
    calib = sp["val"].x[np.random.default_rng(0).choice(len(sp["val"].y), 300, replace=False)]
    quantize_static(str(pre), str(int8), _Reader(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    os.remove(pre)

    with torch.no_grad():
        torch_logits = model(to_nchw(sp["test"].x)).numpy()
    s32, s8 = session(fp32), session(int8)
    l32, l8 = run(s32, sp["test"].x), run(s8, sp["test"].x)
    y = sp["test"].y
    p50_32, p99_32 = latency_ms(s32, sp["test"].x)
    p50_8, p99_8 = latency_ms(s8, sp["test"].x)
    res = {
        "onnx_matches_pytorch_max_abs_logit_diff": float(np.abs(l32 - torch_logits).max()),
        "size_mb": {"fp32": fp32.stat().st_size / 1e6, "int8": int8.stat().st_size / 1e6},
        "test_accuracy": {"fp32": float((l32.argmax(1) == y).mean()), "int8": float((l8.argmax(1) == y).mean())},
        "int8_agrees_with_fp32": float((l32.argmax(1) == l8.argmax(1)).mean()),
        "latency_ms_1_image_1_thread": {"fp32_p50": p50_32, "fp32_p99": p99_32, "int8_p50": p50_8, "int8_p99": p99_8},
        "calibration_images": 300,
    }
    cfg_file = MODELS / "serving.json"
    if cfg_file.exists():  # does INT8 keep the melanoma referral behaviour chosen on FP32?
        cfg = json.load(open(cfg_file))
        mel = cfg["classes"].index("mel")

        def referral(logits):
            z = logits / cfg["temperature"]
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            flag = p[:, mel] >= cfg["melanoma_threshold"]
            return {"sensitivity": float(flag[y == mel].mean()), "specificity": float((~flag[y != mel]).mean())}
        res["melanoma_referral_on_test"] = {"fp32": referral(l32), "int8": referral(l8)}
    json.dump(res, open(ROOT / "results" / "quantization.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
