"""Grad-CAM on the deployed model, plus a check that the maps mean something.

Grad-CAM can be taken at any ResNet stage; for a 64x64 input layer2 gives an 8x8 map, layer3 4x4
and layer4 2x2. All three are tested below and the most faithful one is used for the figure.
A heatmap is only useful if the model really depends on the highlighted pixels, so we run a
deletion test: blank out the 25% of pixels Grad-CAM ranks highest and see how much the predicted
class's probability falls, compared with blanking a random 25% of pixels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASSES, splits
from .model import SkinNet, to_nchw

ROOT = Path(__file__).resolve().parent.parent


def gradcam(model: SkinNet, x_nchw: torch.Tensor, cls: torch.Tensor | None = None, layer: str = "layer3") -> tuple[np.ndarray, np.ndarray]:
    """Returns (cams N x 28 x 28 in [0, 1], predicted or given class)."""
    model.eval()
    feats = {}
    mod = getattr(model.net, layer)
    h1 = mod.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = mod.register_full_backward_hook(lambda m, gi, go: feats.__setitem__("g", go[0]))
    x = x_nchw.float().requires_grad_(False)
    logits = model(x)
    if cls is None:
        cls = logits.argmax(1)
    model.zero_grad()
    logits.gather(1, cls.view(-1, 1)).sum().backward()
    h1.remove(); h2.remove()
    w = feats["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.detach().numpy(), cls.numpy()


@torch.no_grad()
def _prob(model, x, cls):
    return torch.softmax(model(x), 1).gather(1, torch.as_tensor(cls).view(-1, 1)).ravel().numpy()


def deletion_test(model, x_nhwc, frac=0.25, seed=0, layer="layer3"):
    x = to_nchw(x_nhwc).float()
    cam, cls = gradcam(model, x, layer=layer)
    base = _prob(model, x, cls)
    fill = x.mean(dim=(2, 3), keepdim=True)  # blank = the image's own mean colour
    k = int(frac * 28 * 28)
    flat = cam.reshape(len(cam), -1)
    top = np.argsort(-flat, 1)[:, :k]
    bottom = np.argsort(flat, 1)[:, :k]
    rnd = np.stack([np.random.default_rng(seed + i).choice(28 * 28, k, replace=False) for i in range(len(cam))])

    def blank(idx):
        m = torch.zeros(len(x), 28 * 28)
        m.scatter_(1, torch.as_tensor(idx), 1.0)
        m = m.view(-1, 1, 28, 28)
        return x * (1 - m) + fill * m

    return {"images": int(len(x)), "blanked_share": frac,
            "mean_prob_drop_gradcam_pixels": float(np.mean(base - _prob(model, blank(top), cls))),
            "mean_prob_drop_random_pixels": float(np.mean(base - _prob(model, blank(rnd), cls))),
            "mean_prob_drop_lowest_gradcam_pixels": float(np.mean(base - _prob(model, blank(bottom), cls))),
            "layer": layer}


def main(run="lesion_pretrained_s0", n=400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = SkinNet()
    model.load_state_dict(torch.load(ROOT / "results" / "runs" / run / "model.pt", map_location="cpu"))
    test = splits("lesion")["test"]
    idx = np.random.default_rng(0).choice(len(test.y), n, replace=False)
    res = {layer: deletion_test(model, test.x[idx], layer=layer) for layer in ("layer2", "layer3", "layer4")}
    # most faithful = biggest margin over blanking the same number of random pixels
    best = max(res, key=lambda l: res[l]["mean_prob_drop_gradcam_pixels"] - res[l]["mean_prob_drop_random_pixels"])
    res = {"by_layer": res, "used_for_figure": best}
    # figure: one correctly classified example per class
    picks = []
    with torch.no_grad():
        pred = model(to_nchw(test.x)).argmax(1).numpy()
    for c in range(7):
        ok = np.where((test.y == c) & (pred == c))[0]
        if len(ok):
            picks.append(ok[0])
    cam, _ = gradcam(model, to_nchw(test.x[picks]), layer=best)
    fig, axes = plt.subplots(2, len(picks), figsize=(1.6 * len(picks), 3.4))
    for j, i in enumerate(picks):
        axes[0, j].imshow(test.x[i]); axes[0, j].set_title(CLASSES[test.y[i]], fontsize=8)
        axes[1, j].imshow(test.x[i]); axes[1, j].imshow(cam[j], cmap="jet", alpha=0.45)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM ({best}) on correctly classified test images, 28x28", fontsize=9)
    fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / "gradcam.png", dpi=150)
    json.dump(res, open(ROOT / "results" / "explainability.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
