# DualAttnGAN
Partial Pytorch implementation of the DualAttnGAN model, described in https://ieeexplore.ieee.org/document/8930532, starting from the code in the original repository for AttnGAN.

The main additions to the code are in the `code/dualAttnModel.py` file.
At the time of this writing, the currently implemented features are:
- Channel Attention Module
- Spatial Attention Module
- Attention Embedding Module

Compared to the original model description, the Channel Attention Module includes two additional `conv2d` layers as a way to dimensionally reduce the data passing through the network; the original architecture described in the paper was too expensive memory-wise for the resources available for testing.

## Installation

Since the original code depends on Python 2.7 and an old (0.4.0) version of Pytorch, the new code was developed using a `conda` environment; other than Python2.7 and Pytorch0.4, the installation of `cudnn=7.1.2` and `cuda92` was needed. All the other dependecies were kept intact, and can be found in [README_old.md](README_old.md).

## Training and evaluation

The training and evaluation structure was kept from the original code, and the instructions can be found in [README_old.md](README_old.md)

Inception Score and Fréchet Inception Distance were calculated respectively using these implementations:
- IS: https://github.com/hanzhanggit/StackGAN-inception-model
- FID: https://github.com/bioinf-jku/TTUR
