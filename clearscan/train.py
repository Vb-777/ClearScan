"""Trains one model on one split and logs everything to MLflow.

    python -m clearscan.train --split lesion --seed 0 [--scratch] [--epochs 12]

Model selection uses validation macro-AUC only; the test split is scored once, at the end.
Writes results/runs/<run>/{model.pt, logits.npz, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .data import CLASSES, DATA, splits
from .model import SkinNet, to_nchw

ROOT = Path(__file__).resolve().parent.parent


def augment(x: torch.Tensor) -> torch.Tensor:
    """Dermoscopy has no 'up': random flips and 90-degree rotations, plus mild brightness jitter."""
    if random.random() < 0.5:
        x = x.flip(3)
    if random.random() < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, random.randint(0, 3), dims=(2, 3))
    scale = torch.empty(x.shape[0], 1, 1, 1).uniform_(0.85, 1.15)
    return (x.float() * scale).clamp(0, 255)


@torch.no_grad()
def predict_logits(model, x_nhwc, batch=512) -> np.ndarray:
    model.eval()
    out = [model(to_nchw(x_nhwc[i:i + batch])) for i in range(0, len(x_nhwc), batch)]
    return torch.cat(out).numpy()


def scores(logits: np.ndarray, y: np.ndarray) -> dict:
    p = torch.softmax(torch.as_tensor(logits), 1).numpy()
    pred = p.argmax(1)
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_auc": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
            "melanoma_auc": float(roc_auc_score(y == CLASSES.index("mel"), p[:, CLASSES.index("mel")]))}


def main(split="lesion", seed=0, epochs=12, scratch=False, lr=None, batch=128, threads=2):
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sp = splits(split)
    name = f"{split}_{'scratch' if scratch else 'pretrained'}_s{seed}"
    out = ROOT / "results" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    model = SkinNet(pretrained=None if scratch else DATA / "resnet18_a1.pth")
    lr = lr or (3e-3 if scratch else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * int(np.ceil(len(sp["train"].y) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    xtr, ytr = to_nchw(sp["train"].x), torch.as_tensor(sp["train"].y)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_registry_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("clearscan")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"split": split, "seed": seed, "epochs": epochs, "pretrained": not scratch, "lr": lr,
                           "batch": batch, "input": "28x28 RGB -> 64x64", "arch": "resnet18",
                           "n_train": len(ytr), "n_val": len(sp["val"].y), "n_test": len(sp["test"].y)})
        best, best_state = -1.0, None
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(ytr))
            t0, total = time.time(), 0.0
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                loss = F.cross_entropy(model(augment(xtr[idx])), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                total += loss.item() * len(idx)
            val = scores(predict_logits(model, sp["val"].x), sp["val"].y)
            mlflow.log_metrics({"train_loss": total / len(ytr), **{f"val_{k}": v for k, v in val.items()},
                                "epoch_seconds": time.time() - t0}, step=ep)
            print(f"{name} ep{ep} loss {total / len(ytr):.4f} val {val}", flush=True)
            if val["macro_auc"] > best:
                best, best_state = val["macro_auc"], {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        lv, lt = predict_logits(model, sp["val"].x), predict_logits(model, sp["test"].x)
        test = scores(lt, sp["test"].y)
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        torch.save(model.state_dict(), out / "model.pt")
        np.savez(out / "logits.npz", val=lv, test=lt, y_val=sp["val"].y, y_test=sp["test"].y)
        summary = {"run": name, "best_val_macro_auc": best, "test": test}
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
        mlflow.log_artifact(str(out / "summary.json"))
        mlflow.log_artifact(str(out / "model.pt"))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lesion", choices=["lesion", "original"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--scratch", action="store_true")
    a = ap.parse_args()
    print(json.dumps(main(a.split, a.seed, a.epochs, a.scratch)))
