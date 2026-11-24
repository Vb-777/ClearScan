"""Writes models/serving.json from the evaluation: the deployed model file, its temperature and the
melanoma referral threshold (both chosen on validation data)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clearscan.data import CLASSES, CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
ev = json.load(open(ROOT / "results" / "evaluation.json"))
cfg = {"version": "1.0.0", "model_file": "clearscan_int8.onnx", "trained_run": ev["run"], "classes": CLASSES,
       "class_names": CLASS_NAMES, "temperature": round(ev["temperature"], 4),
       "melanoma_threshold": round(ev["melanoma_referral"]["threshold"], 4), "low_confidence_below": 0.5}
(ROOT / "models").mkdir(exist_ok=True)
json.dump(cfg, open(ROOT / "models" / "serving.json", "w"), indent=1)
print(cfg)
