#!/usr/bin/env bash
# DermaMNIST 28x28 (MedMNIST v2, from HAM10000) with its image ids and the HAM10000 metadata, as
# published with Abhishek et al., "Investigating the Quality of DermaMNIST and Fitzpatrick17k" (2024),
# plus ImageNet ResNet-18 weights from timm's GitHub release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data"
cd "$ROOT/data"
R=https://raw.githubusercontent.com/kakumarabhishek/Corrected-Skin-Image-Datasets/main/DermaMNIST/DermaMNIST_Analysis
M=https://media.githubusercontent.com/media/kakumarabhishek/Corrected-Skin-Image-Datasets/main/DermaMNIST/DermaMNIST_Analysis
[ -f dermamnist_split_info.csv ] || curl -fsSL -o dermamnist_split_info.csv $R/dermamnist_split_info.csv
[ -f HAM10000_metadata.csv ] || curl -fsSL -o HAM10000_metadata.csv $R/HAM10000_metadata.csv
[ -f dermamnist_original.npz ] || curl -fsSL -o dermamnist_original.npz $R/NPZ_files/dermamnist_original.npz || curl -fsSL -o dermamnist_original.npz $M/NPZ_files/dermamnist_original.npz
[ -f resnet18_a1.pth ] || curl -fsSL -o resnet18_a1.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth
python -c "import numpy as np; z=np.load('dermamnist_original.npz'); assert z['train_images'].shape==(7007,28,28,3)"
