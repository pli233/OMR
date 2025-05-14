#!/bin/bash

# Create conda environment with Python 3.10
conda create -n staff2sound python=3.10 -y

# Activate the newly created conda environment
conda activate staff2sound 

# Install PyTorch and other necessary libraries (please check CUDA version on PyTorch website for compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ultraytics, OpenCV, Pillow, pandas, scipy, jupyter, pretty_midi, mido, and pygame
pip install ultralytics
pip install matplotlib opencv-python Pillow pandas scipy jupyter
pip install pretty_midi mido pygame

echo "OMR environment setup is complete!"
