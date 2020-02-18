# Incorporating symmetries into deep dynamics models for improved generalization

## Prerequisites
- Linux or macOS
- Python 3
- Pytorch 1.1.0
- E2CNN (https://github.com/QUVA-Lab/e2cnn)

## Dataset
- Heat Diffusion
- Rayleigh-Benard convection

## Codebase
- Baseline_NonEqu: Standard Non-equivariant ResNet and U-net
- Equivariant-Models: ResNets and U-nets equipped with four different symmetries, including uniform motion, scaling, magnitude, rotation.
- Evaluation: Jupyter notebook & RMSE and Energy Spectrum functions.
