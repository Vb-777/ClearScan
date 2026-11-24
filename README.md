# ClearScan: a skin-lesion classifier evaluated and served like a medical-AI product

ClearScan fine-tunes a ResNet-18 to recognise 7 diagnoses in dermatoscopic images (HAM10000 via
DermaMNIST). The goal is not the highest accuracy. It is to do everything around the model that a
healthcare team needs before trusting it:

* a test set that holds only lesions the model has never seen;
* confidence intervals;
* calibration;
* a melanoma-review threshold chosen without looking at the test set;
* subgroup results;
* explanations that are checked rather than just displayed;
* experiment tracking and a model card;
* a small INT8 model behind an API.

Every number below comes from `results/*.json`. The model card (`MODEL_CARD.md`) is generated from
those files, so no figure in it is typed by hand.

## Results (test set: 2,004 images of 1,495 lesions never seen in training)

| | value (95% CI, bootstrap over lesions) |
|---|---|
| accuracy | **77.9%** (75.6-80.2) |
| macro one-vs-rest AUC (7 classes) | **0.931** (0.920-0.941) |
| melanoma AUC | **0.871** (0.840-0.897) |
| balanced accuracy | 50.2% (45.7-54.4): the rare classes are hard |
| calibration error (ECE) | **0.021** (already calibrated; temperature scaling did not improve it) |
| melanoma review flag (threshold set on validation for 90% sensitivity) | sensitivity **75.3%** (68.4-82.2) at 79.5% specificity on test, so the 90% target did **not** transfer |
| INT8 vs FP32 ONNX | 44.7 → 11.3 MB, 3.1 → 1.0 ms per image (1 CPU thread), accuracy 77.9 → 77.2%, 97.9% of predictions unchanged |

Across 3 seeds the recipe gives 78.0 ± 0.5% accuracy and 0.933 ± 0.002 macro AUC. For reference,
the MedMNIST v2 paper reports 73.5% accuracy and 0.917 AUC for ResNet-18 at 28x28 on the official
split. That comparison is indicative only, because the splits differ.

## Findings worth reading

1. **Data leakage in the public split, tested rather than assumed.**
   * 693 of the 2,005 official test images (34.6%) show a lesion that is also in training, because
     the official split assigns photos, not lesions.
   * Training the same recipe on both splits (3 seeds each) gives **no measurable inflation**:
     77.9 ± 1.0% vs 78.0 ± 0.5%.
   * The leaked test photos are actually *harder*, and this holds within each diagnosis. For
     example, melanoma recall is 34.8% on seen lesions vs 44.0% on unseen ones. A likely reason is
     that lesions photographed several times are the followed-up, difficult cases.
   * The deployed model still uses the lesion split, because only that split measures what a new
     patient brings.
2. **A threshold tuned on ~110 validation melanomas does not hold on test.** The 90% sensitivity
   target came out at 75%. The model card says so, and a deployment would need a larger
   calibration set.
3. **Explanations were tested, not just drawn.**
   * Blanking the 25% of pixels that Grad-CAM (layer2) ranks highest lowers the predicted
     probability by 0.160. Blanking its lowest-ranked 25% lowers it by 0.070, and blanking random
     pixels by 0.142.
   * So the map points at relevant pixels, but its margin over random is small.
   * Grad-CAM at deeper layers is *less* faithful than random at this resolution (layer3: 0.123;
     layer4, a 2x2 map: 0.000).
4. **Subgroups.**
   * Accuracy varies by age (87% under 40, 65% at 60+) and body site (face 57%).
   * Balanced accuracy is much flatter, so most of that variation comes from older patients and
     faces having more malignant lesions.
   * HAM10000 records no skin type, so fairness across skin tones cannot be measured. This is the
     largest gap, and the model card states it.
5. **Pretraining adds little at 28x28.** The same network trained from scratch reaches 76.7%
   accuracy and 0.931 macro AUC.

## How it is built

```
clearscan/data.py      DermaMNIST joined to HAM10000 metadata; official vs lesion-grouped splits
clearscan/model.py     ResNet-18 (ImageNet weights from timm), 28x28 input resized to 64x64 inside the model
clearscan/train.py     training with augmentation, model selection on validation only, MLflow logging
clearscan/evaluate.py  lesion-cluster bootstrap CIs, calibration, temperature, melanoma threshold, subgroups, leakage
clearscan/explain.py   Grad-CAM at 3 layers + deletion (faithfulness) test
clearscan/export.py    ONNX export, static INT8 quantisation (onnxruntime), size/latency/accuracy/agreement
serve/app.py           FastAPI: /predict (image upload), /health, /model-card; INT8 ONNX, no PyTorch
serve/Dockerfile       slim inference image, non-root user
scripts/model_card.py  writes MODEL_CARD.md from the results
tests/                 data, maths, exported model and API tests (6, pytest)
```

**Experiment tracking.** All 7 training runs (2 splits × 3 seeds, plus 1 from-scratch run) are in
MLflow (`mlflow.db`), with parameters, per-epoch validation metrics and final test metrics.
Browse them with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Running it

```bash
pip install -r requirements.txt
./run_all.sh           # fetch data, 7 training runs (~70 min on 2 CPU cores), evaluate, export, explain, model card, tests
uvicorn serve.app:app --port 8000
curl -F image=@lesion.jpg localhost:8000/predict
docker build -f serve/Dockerfile -t clearscan . && docker run -p 8000:8000 clearscan
```


## Licensing

The project code is released under the MIT License. The trained model artifact and evaluation
outputs were produced from third-party datasets/weights and remain subject to their applicable
source licenses, including HAM10000's CC BY-NC 4.0 terms. See the data attribution below.

## Honest limits

* **Not a medical device and not for diagnosis.** The images are 28x28, and the data comes from one
  public dataset collected in Austria and Australia.
* The Docker image has not been built here (no Docker in the build environment). The API itself is
  tested in-process and was also run under uvicorn.
* Everything was trained and measured on CPU (2 cores). Larger images (MedMNIST+ 224x224, or the
  original HAM10000) would very likely do better, but they could not be downloaded in this environment.

Data: DermaMNIST (MedMNIST v2, CC BY 4.0) from HAM10000 (CC BY-NC 4.0). Image ids and metadata
come via Abhishek et al., "Investigating the Quality of DermaMNIST and Fitzpatrick17k" (2024),
github.com/kakumarabhishek/Corrected-Skin-Image-Datasets (Apache-2.0).
