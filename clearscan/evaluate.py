"""The study's evaluation, written to results/evaluation.json and results/figures/.

1. Leakage: the same recipe trained on the official DermaMNIST split vs a lesion-grouped split
   (3 seeds each), and, within the official test set, leaked vs clean images for the same model.
2. The deployed model (lesion split, seed 0): metrics with 95% CIs from a bootstrap that
   resamples lesions, not images (photos of one lesion are not independent).
3. Calibration before/after temperature scaling (temperature fitted on validation only).
4. A melanoma operating point chosen on validation (90% sensitivity), then measured on test.
5. Subgroups: sex, age band, body site.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .data import CLASSES, splits

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
MEL = CLASSES.index("mel")
rng = np.random.default_rng(0)


def softmax(z, t=1.0):
    return torch.softmax(torch.as_tensor(z) / t, 1).numpy()


def metrics(p, y):
    pred = p.argmax(1)
    out = {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    try:
        out["macro_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(7))))
    except ValueError:
        out["macro_auc"] = float("nan")
    if 0 < (y == MEL).sum() < len(y):
        out["melanoma_auc"] = float(roc_auc_score(y == MEL, p[:, MEL]))
    return out


def cluster_bootstrap(p, y, groups, fn, n=1000):
    """95% CI resampling whole lesions."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = fn(p[idx], y[idx])
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def fit_temperature(logits, y):
    res = minimize_scalar(lambda t: nll(softmax(logits, t), y), bounds=(0.05, 10), method="bounded")
    return float(res.x)


def load_run(name):
    z = np.load(RUNS / name / "logits.npz")
    return z["val"], z["test"], z["y_val"], z["y_test"]


def leakage_study():
    out = {}
    for split in ("original", "lesion"):
        rows = []
        for s in (0, 1, 2):
            f = RUNS / f"{split}_pretrained_s{s}" / "logits.npz"
            if f.exists():
                _, lt, _, yt = load_run(f"{split}_pretrained_s{s}")
                rows.append(metrics(softmax(lt), yt))
        if rows:
            out[split] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
                          for k in rows[0]} | {"seeds": len(rows)}
    # within the official test set: images whose lesion was seen in training vs not
    sp = splits("original")
    leaked = sp["test"].meta.lesion_id.isin(set(sp["train"].meta.lesion_id)).values
    within = []
    for s in (0, 1, 2):
        f = RUNS / f"original_pretrained_s{s}" / "logits.npz"
        if f.exists():
            _, lt, _, yt = load_run(f"original_pretrained_s{s}")
            pred = lt.argmax(1)
            within.append({"leaked_accuracy": float((pred[leaked] == yt[leaked]).mean()),
                           "clean_accuracy": float((pred[~leaked] == yt[~leaked]).mean()),
                           "leaked_balanced_accuracy": float(balanced_accuracy_score(yt[leaked], pred[leaked])),
                           "clean_balanced_accuracy": float(balanced_accuracy_score(yt[~leaked], pred[~leaked]))})
    # same comparison within each diagnosis, so a different class mix cannot explain the gap
    per_class = {}
    for c in ("mel", "nv", "bkl", "bcc"):
        ci_ = CLASSES.index(c)
        rec = {"leaked": [], "clean": []}
        for s in (0, 1, 2):
            f = RUNS / f"original_pretrained_s{s}" / "logits.npz"
            if f.exists():
                _, lt, _, yt = load_run(f"original_pretrained_s{s}")
                pred = lt.argmax(1)
                for key, m in (("leaked", leaked), ("clean", ~leaked)):
                    sel = m & (yt == ci_)
                    rec[key].append(float((pred[sel] == ci_).mean()))
        y_off = sp["test"].y
        per_class[c] = {"n_leaked": int((leaked & (y_off == ci_)).sum()), "n_clean": int((~leaked & (y_off == ci_)).sum()),
                        "recall_leaked": float(np.mean(rec["leaked"])), "recall_clean": float(np.mean(rec["clean"]))}
    out["official_test_leaked_vs_clean"] = {"n_leaked": int(leaked.sum()), "n_clean": int((~leaked).sum()),
                                            "per_seed": within, "per_class_mean_recall": per_class}
    return out


def main(run="lesion_pretrained_s0"):
    lv, lt, yv, yt = load_run(run)
    sp = splits("lesion")
    meta = sp["test"].meta
    assert (meta.dx.map(CLASSES.index).values == yt).all()
    groups = meta.lesion_id.values
    T = fit_temperature(lv, yv)
    p_raw, p = softmax(lt), softmax(lt, T)
    pv = softmax(lv, T)

    res = {"run": run, "test_images": int(len(yt)), "test_lesions": int(len(set(groups))), "temperature": T}
    m = metrics(p, yt)
    res["test"] = {k: {"value": v, "ci95": cluster_bootstrap(p, yt, groups, lambda a, b, k=k: metrics(a, b).get(k, float("nan")))}
                   for k, v in m.items()}
    res["per_class_recall"] = {c: float((p.argmax(1)[yt == i] == i).mean()) for i, c in enumerate(CLASSES)}
    res["calibration"] = {"before": {"ece": ece(p_raw, yt), "nll": nll(p_raw, yt)},
                          "after_temperature": {"ece": ece(p, yt), "nll": nll(p, yt)}}

    # melanoma referral threshold: highest threshold with >= 90% sensitivity on validation
    sv = pv[:, MEL]
    pos = np.sort(sv[yv == MEL])[::-1]
    thr = float(pos[int(np.ceil(0.9 * len(pos))) - 1])
    st = p[:, MEL]
    sens = lambda a, b: float(((a[:, MEL] >= thr) & (b == MEL)).sum() / max((b == MEL).sum(), 1))
    spec = lambda a, b: float(((a[:, MEL] < thr) & (b != MEL)).sum() / max((b != MEL).sum(), 1))
    refer = float((st >= thr).mean())
    res["melanoma_referral"] = {"threshold": thr, "chosen_on": "validation, 90% sensitivity target",
                                "test_sensitivity": {"value": sens(p, yt), "ci95": cluster_bootstrap(p, yt, groups, sens)},
                                "test_specificity": {"value": spec(p, yt), "ci95": cluster_bootstrap(p, yt, groups, spec)},
                                "share_of_images_referred": refer}

    # subgroups
    age = meta.age
    bands = np.where(age.isna(), "unknown", np.where(age < 40, "<40", np.where(age < 60, "40-59", "60+")))
    site_top = meta.localization.value_counts().index[:6]
    sub = {}
    for name, labels in (("sex", meta.sex.values), ("age_band", bands),
                         ("body_site", np.where(meta.localization.isin(site_top), meta.localization, "other").astype(str))):
        sub[name] = {}
        for g in sorted(set(labels)):
            idx = np.where(labels == g)[0]
            if len(idx) < 40:
                continue
            mm = metrics(p[idx], yt[idx])
            sub[name][str(g)] = {"n": int(len(idx)), "melanoma_n": int((yt[idx] == MEL).sum()),
                                 "accuracy": mm["accuracy"], "balanced_accuracy": mm["balanced_accuracy"],
                                 "melanoma_sensitivity": sens(p[idx], yt[idx]) if (yt[idx] == MEL).sum() >= 10 else None,
                                 "accuracy_ci95": cluster_bootstrap(p[idx], yt[idx], groups[idx], lambda a, b: float((a.argmax(1) == b).mean()), n=500)}
    res["subgroups"] = sub
    res["leakage"] = leakage_study()
    scr = RUNS / "lesion_scratch_s0" / "logits.npz"
    if scr.exists():
        _, st_, _, ys_ = load_run("lesion_scratch_s0")
        res["from_scratch_same_split"] = metrics(softmax(st_), ys_)
    out = ROOT / "results" / "evaluation.json"
    json.dump(res, open(out, "w"), indent=1)
    plots(p_raw, p, yt, res)
    return res


def plots(p_raw, p, y, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = ROOT / "results" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for probs, lab, col in ((p_raw, f"before (ECE {ece(p_raw, y):.3f})", "#9ca3af"), (p, f"after T={res['temperature']:.2f} (ECE {ece(p, y):.3f})", "#2563eb")):
        conf, acc = probs.max(1), probs.argmax(1) == y
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean()); ys.append(acc[m].mean())
        ax.plot(xs, ys, "o-", color=col, label=lab)
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability (test)"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(figs / "reliability.png", dpi=150); plt.close(fig)

    lk = res["leakage"]
    if "original" in lk and "lesion" in lk:
        fig, ax = plt.subplots(figsize=(5, 3))
        ks = ["accuracy", "balanced_accuracy", "macro_auc", "melanoma_auc"]
        for i, (split, col) in enumerate((("original", "#d97706"), ("lesion", "#2563eb"))):
            ax.bar(np.arange(4) + (i - 0.5) * 0.38, [lk[split][k]["mean"] for k in ks], 0.38,
                   yerr=[lk[split][k]["sd"] for k in ks], color=col, label={"original": "official split (lesions leak)", "lesion": "lesion-grouped split"}[split])
        ax.set_xticks(range(4), ["accuracy", "balanced acc.", "macro AUC", "melanoma AUC"], fontsize=8)
        ax.set_ylim(0.4, 1); ax.legend(fontsize=7, frameon=False); ax.set_title("Same recipe, 3 seeds each")
        fig.tight_layout(); fig.savefig(figs / "leakage.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("test", "calibration", "melanoma_referral", "leakage")}, indent=1))
