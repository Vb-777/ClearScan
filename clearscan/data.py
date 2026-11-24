"""DermaMNIST (HAM10000 dermatoscopic images, 7 diagnoses, 28x28 RGB) joined to the HAM10000
metadata (lesion id, age, sex, body site), and two ways to split it.

* "original": the official DermaMNIST split. HAM10000 often has several photos of the same
  lesion, and this split assigns photos, not lesions, so a lesion can be in train and test.
* "lesion": every photo of a lesion goes to the same split (stratified by diagnosis, grouped by
  lesion, fixed seed). This is the split a clinical claim should rest on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # DermaMNIST label order (alphabetical)
CLASS_NAMES = {"akiec": "actinic keratosis / intraepithelial carcinoma", "bcc": "basal cell carcinoma",
               "bkl": "benign keratosis", "df": "dermatofibroma", "mel": "melanoma", "nv": "melanocytic nevus",
               "vasc": "vascular lesion"}
DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Split:
    x: np.ndarray        # N x 28 x 28 x 3 uint8
    y: np.ndarray        # N int64
    meta: pd.DataFrame   # image_id, lesion_id, age, sex, localization, dx


def load_all(data_dir: Path = DATA) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """All 10,015 images in one table, with each image's official split."""
    z = np.load(data_dir / "dermamnist_original.npz")
    info = pd.read_csv(data_dir / "dermamnist_split_info.csv")
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    xs, ys, rows = [], [], []
    for split in ("train", "val", "test"):
        part = info[info.split == split].sort_values("index")
        xs.append(z[f"{split}_images"])
        ys.append(z[f"{split}_labels"].ravel())
        rows.append(part)
    x, y = np.concatenate(xs), np.concatenate(ys).astype(np.int64)
    table = pd.concat(rows, ignore_index=True).merge(meta, on="image_id", how="left", validate="one_to_one")
    if not (table.dx.map(CLASSES.index).values == y).all():
        raise ValueError("labels in the npz do not match the HAM10000 diagnoses")
    return x, y, table.rename(columns={"split": "official_split"})


def lesion_split(table: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """train / val / test labels with every lesion in exactly one split (about 70 / 10 / 20)."""
    out = np.empty(len(table), dtype=object)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, test = next(outer.split(table, table.dx, groups=table.lesion_id))
    out[test] = "test"
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    sub = table.iloc[rest]
    tr, va = next(inner.split(sub, sub.dx, groups=sub.lesion_id))
    out[rest[tr]], out[rest[va]] = "train", "val"
    return out


def splits(kind: str, seed: int = 0) -> dict[str, Split]:
    x, y, t = load_all()
    assign = t.official_split.values if kind == "original" else lesion_split(t, seed)
    res = {}
    for s in ("train", "val", "test"):
        idx = np.where(assign == s)[0]
        res[s] = Split(x[idx], y[idx], t.iloc[idx].reset_index(drop=True))
    return res


def leakage_report(kind: str) -> dict:
    sp = splits(kind)
    train_lesions = set(sp["train"].meta.lesion_id)
    test = sp["test"].meta
    leaked = test.lesion_id.isin(train_lesions)
    return {"split": kind, "test_images": int(len(test)), "test_images_with_lesion_in_train": int(leaked.sum()),
            "share": round(float(leaked.mean()), 4)}
