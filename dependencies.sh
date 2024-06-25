#!/bin/bash

#Run this script as "bash dependencies.sh" to install all required packages.

# Install modules
pip install numpy scipy matplotlib opencv-python scikit-image Pillow bm3d
# Install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install torchsummary 
