"""Checks on the data handling, the evaluation maths, the exported models and the service."""
import io
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
HAVE_MODEL = (ROOT / "models" / "clearscan_int8.onnx").exists() and (ROOT / "models" / "serving.json").exists()


def test_labels_match_ham10000_metadata_and_lesion_split_has_no_leakage():
    from clearscan.data import leakage_report, load_all, splits
    x, y, t = load_all()  # raises if any label disagrees with the HAM10000 diagnosis
    assert len(y) == 10015 and t.lesion_id.notna().all()
    assert leakage_report("lesion")["test_images_with_lesion_in_train"] == 0
    assert leakage_report("original")["test_images_with_lesion_in_train"] == 693
    sp = splits("lesion")
    lesions = [set(sp[k].meta.lesion_id) for k in ("train", "val", "test")]
    assert not (lesions[0] & lesions[1]) and not (lesions[0] & lesions[2]) and not (lesions[1] & lesions[2])
    assert sum(len(sp[k].y) for k in sp) == 10015


def test_ece_and_temperature_behave():
    from clearscan.evaluate import ece, fit_temperature, softmax
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=2.0, size=(5000, 7))
    p = softmax(logits)
    y = np.array([rng.choice(7, p=row) for row in p])  # labels drawn from the model's own probabilities: calibrated
    over = logits * 4  # same ranking, far over-confident
    assert ece(softmax(over), y) > ece(softmax(logits), y)
    t = fit_temperature(over, y)
    assert 3.0 < t < 5.5  # recovers roughly the factor that was applied
    assert ece(softmax(over, t), y) < ece(softmax(over), y)


def test_cluster_bootstrap_resamples_whole_lesions():
    from clearscan.evaluate import cluster_bootstrap
    # two lesions, each with 50 identical photos: one always right, one always wrong.
    y = np.zeros(100, int)
    p = np.zeros((100, 7)); p[:50, 0] = 1; p[50:, 1] = 1
    groups = np.array(["a"] * 50 + ["b"] * 50)
    lo, hi = cluster_bootstrap(p, y, groups, lambda a, b: float((a.argmax(1) == b).mean()), n=400)
    assert lo == 0.0 and hi == 1.0  # with only 2 independent lesions the CI must be wide


@pytest.mark.skipif(not HAVE_MODEL, reason="run the pipeline first")
def test_exported_models_agree_with_recorded_results():
    q = json.load(open(ROOT / "results" / "quantization.json"))
    assert q["onnx_matches_pytorch_max_abs_logit_diff"] < 1e-3
    assert q["int8_agrees_with_fp32"] > 0.95 and q["size_mb"]["int8"] < q["size_mb"]["fp32"] / 3


@pytest.mark.skipif(not HAVE_MODEL, reason="run the pipeline first")
def test_service_predicts_and_rejects_bad_input():
    from fastapi.testclient import TestClient
    from PIL import Image

    from clearscan.data import CLASSES, splits
    from serve.app import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert "Intended use" in c.get("/model-card").text
    test = splits("lesion")["test"]
    i = int(np.where(test.y == CLASSES.index("nv"))[0][0])
    buf = io.BytesIO()
    Image.fromarray(test.x[i]).resize((224, 224)).save(buf, format="PNG")
    r = c.post("/predict", files={"image": ("x.png", buf.getvalue(), "image/png")})
    body = r.json()
    assert r.status_code == 200 and abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["prediction"] in CLASSES and "Not for diagnosis" in body["notice"]
    assert c.post("/predict", files={"image": ("x.txt", b"not an image", "text/plain")}).status_code == 400
    tiny = io.BytesIO(); Image.new("RGB", (10, 10)).save(tiny, format="PNG")
    assert c.post("/predict", files={"image": ("t.png", tiny.getvalue(), "image/png")}).status_code == 422


@pytest.mark.skipif(not HAVE_MODEL, reason="run the pipeline first")
def test_service_matches_offline_int8_model_on_test_images():
    """The service's own preprocessing path gives the same answer as the offline evaluation."""
    from clearscan.data import splits
    from clearscan.export import run, session
    from serve.app import classify, CFG
    test = splits("lesion")["test"]
    offline = run(session(ROOT / "models" / CFG["model_file"]), test.x[:200]).argmax(1)
    online = [CFG["classes"].index(classify(test.x[i].transpose(2, 0, 1)[None].astype(np.float32))["prediction"]) for i in range(200)]
    assert (offline == np.array(online)).mean() == 1.0
