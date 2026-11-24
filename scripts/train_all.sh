#!/usr/bin/env bash
# All training runs in the study: 3 seeds on each split with ImageNet-pretrained ResNet-18,
# plus one from-scratch baseline. About 10 minutes per run on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
for seed in 0 1 2; do
  for split in lesion original; do
    [ -f results/runs/${split}_pretrained_s${seed}/summary.json ] || python -m clearscan.train --split $split --seed $seed
  done
done
[ -f results/runs/lesion_scratch_s0/summary.json ] || python -m clearscan.train --split lesion --seed 0 --scratch
echo ALL_DONE
