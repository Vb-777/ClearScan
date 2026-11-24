# Model card: ClearScan 1.0.0

## Model details
* ResNet-18 (ImageNet-pretrained, timm A1 weights), fine-tuned to 7 dermatoscopic diagnoses.
  Input: 28x28 RGB (resized internally to 64x64). Deployed as INT8 ONNX (`clearscan_int8.onnx`).
* Trained run: `lesion_pretrained_s0`; every run is tracked in MLflow (`mlflow.db`).
* Research prototype by Vaibhav Barsaiyan, M.Tech CSE, IIT Hyderabad.

## Intended use
Research and teaching: studying how a skin-lesion classifier should be evaluated, calibrated,
explained and served. Output is a ranked list of diagnoses with calibrated probabilities and a
melanoma-review flag.

## Out-of-scope use
**Not for diagnosis, triage or any clinical decision.** It has not been validated prospectively,
on other cameras or clinics, or on darker skin types (HAM10000 is drawn mostly from fair-skinned
patients in Austria and Australia and records no skin type). 28x28 images discard detail a
dermatologist would use.

## Data
DermaMNIST (MedMNIST v2), 10,015 dermatoscopic images from HAM10000 (CC BY-NC 4.0), joined to the
HAM10000 metadata (lesion id, age, sex, body site). Split by lesion, stratified by diagnosis:
train / validation / test = 7,009 / 1,002 / 2,004 images; 1495 distinct lesions in test.
Classes are imbalanced: 67% are benign nevi.

## Performance (test set, lesion-grouped; 95% CIs by bootstrap over lesions)
| metric | value |
|---|---|
| accuracy | 77.9% (95% CI 75.6%-80.2%) |
| balanced accuracy (mean per-class recall) | 50.2% (95% CI 45.7%-54.4%) |
| macro one-vs-rest AUC | 0.931 (95% CI 0.920-0.941) |
| melanoma AUC | 0.871 (95% CI 0.840-0.897) |

Per-class recall: akiec 24.2%, bcc 56.3%, bkl 60.0%, df 8.7%, mel 37.2%, nv 93.3%, vasc 71.4%.
Rare classes (dermatofibroma, vascular, actinic keratosis) are recognised far less reliably than nevi.

## Melanoma review flag
Threshold 0.155 on the melanoma probability, chosen on validation data to reach 90%
sensitivity. On test it gives sensitivity 75.3% (95% CI 68.4%-82.2%) and specificity
79.5% (95% CI 77.3%-81.7%), flagging 26.6% of images.
**The 90% validation target was not reached on test**: with ~110 validation melanomas the threshold is
noisy. A deployment would need a larger calibration set and monitoring.

## Pretraining
A ResNet-18 trained from scratch on the same split reaches 76.7% accuracy and macro AUC
0.931 (melanoma AUC 0.895), so ImageNet pretraining adds little at 28x28.

## Calibration
ECE 0.021 before and 0.024 after temperature scaling (T = 1.11,
fitted on validation). The fine-tuned model was already well calibrated; scaling changed little.

## Subgroups (test set)
| attribute | group | images | accuracy (95% CI) | balanced acc. | melanoma sensitivity |
|---|---|---|---|---|---|
| sex | female | 892 | 79.5% (76.4%-82.3%) | 50.7% | 72.8% |
| sex | male | 1095 | 76.4% (73.2%-79.6%) | 49.3% | 76.8% |
| age_band | 40-59 | 899 | 83.9% (80.3%-86.8%) | 49.2% | 82.9% |
| age_band | 60+ | 710 | 65.1% (61.2%-69.0%) | 53.1% | 72.9% |
| age_band | <40 | 374 | 87.2% (82.5%-91.2%) | 52.5% | 61.5% |
| body_site | abdomen | 184 | 90.2% (84.3%-95.6%) | 54.2% | 63.6% |
| body_site | back | 437 | 74.4% (69.3%-79.4%) | 41.7% | 74.7% |
| body_site | face | 141 | 56.7% (47.8%-66.1%) | 49.9% | 66.7% |
| body_site | lower extremity | 409 | 80.4% (75.8%-84.9%) | 47.7% | 84.6% |
| body_site | other | 332 | 73.2% (67.2%-78.4%) | 41.5% | 74.4% |
| body_site | trunk | 278 | 90.6% (86.1%-95.3%) | 75.4% | n/a (<10 melanomas) |
| body_site | upper extremity | 223 | 74.9% (67.6%-82.0%) | 44.5% | 78.6% |

Accuracy differs a lot by age and body site (e.g. 60+ vs <40), but balanced accuracy is much closer:
older patients and the face have more malignant lesions, which are harder for every group. Melanoma
sensitivity per group rests on few cases (see CIs above for accuracy); treat gaps as leads, not facts.

## Data leakage in the public benchmark split

HAM10000 often holds several photos of the same lesion, and the official DermaMNIST split assigns
photos, not lesions: 693 of its 2,005 test images (34.6%) show a lesion that is also in training.
The same training recipe, 3 seeds per split:

| split | accuracy | balanced accuracy | macro AUC | melanoma AUC |
|---|---|---|---|---|
| official (lesions leak) | 77.9% ± 1.0 | 53.6% ± 2.6 | 0.936 ± 0.004 | 0.888 ± 0.005 |
| lesion-grouped (this model) | 78.0% ± 0.5 | 52.1% ± 2.2 | 0.933 ± 0.002 | 0.879 ± 0.006 |

**Finding: at this resolution the leak did not inflate the headline numbers.** Within the official
test set, models trained on the official split score 64.5% on photos of lesions they saw in
training and 84.9% on unseen lesions (mean of 3 seeds). The gap holds within a diagnosis
(mel: 34.8% vs 44.0%, nv: 85.2% vs 95.7%, bkl: 50.8% vs 50.4%, bcc: 58.5% vs 67.3% recall,
seen vs unseen lesion), so it is not just a different class mix; a likely reason is that lesions
photographed more than once are the harder, followed-up cases. All numbers in this card still use the
lesion-grouped split, because only it measures performance on lesions the model has never seen,
which is what a new patient brings.

## Explanations
Grad-CAM heatmaps are produced from `layer2`. Faithfulness check on 400 test images: blanking the
25% of pixels Grad-CAM ranks highest lowers the predicted class's probability by
0.160, versus 0.070 for its lowest-ranked pixels and
0.142 for random pixels. The margin over random is small, so the maps are a rough
guide, not proof of what the model uses. Deeper layers are worse at this resolution: layer3's top pixels
(0.123) matter less than random ones, and layer4 is a 2x2 map (0.000).

## Deployment
INT8 (static, per-channel, calibrated on 300 validation images) vs FP32 ONNX:
44.7 MB -> 11.3 MB; one-image latency p50 3.15 ms -> 0.97 ms (1 CPU thread);
test accuracy 77.9% -> 77.2%; 97.9% of predictions unchanged.
Melanoma flag on test: FP32 sensitivity 75.3%, INT8 74.0%.
Served by FastAPI (`serve/app.py`), packaged by `serve/Dockerfile`.

## Ethical considerations and limitations
* Skin type is not recorded, so fairness across skin tones cannot be measured; this is the largest gap.
* Labels: about half of HAM10000 diagnoses are histopathology-confirmed, the rest follow-up,
  expert consensus or confocal microscopy.
* Low-resolution input and a single public dataset: performance will drop on other devices and clinics.
