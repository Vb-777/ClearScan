#!/usr/bin/env bash
# Full study: train (7 runs) -> evaluate -> serving config -> ONNX + INT8 -> explanations -> model card -> tests.
set -euo pipefail
cd "$(dirname "$0")"
export MLFLOW_DISABLE_AGENT_HINT=1
scripts/fetch_data.sh
scripts/train_all.sh
python -m clearscan.evaluate > /dev/null
python scripts/serving_config.py
python -m clearscan.export > /dev/null
python -m clearscan.explain > /dev/null
python scripts/model_card.py
python -m pytest -q tests
