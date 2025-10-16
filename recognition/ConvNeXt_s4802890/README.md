# Alzheimer's Disease Classification using ConvNeXt

## Problem Description
Binary classification of brain MRIs from the ADNI dataset:
- **AD**: Alzheimer's Disease  
- **NC**: Normal Control (Cognitively Normal)
 
**Target accuracy**: ≥ 0.8 on test set

## Model Architecture
ConvNeXt built from scratch for binary classification.

ConvNeXt is a modernized CNN architecture that achieves competitive performance with Vision Transformers while maintaining the simplicity and efficiency of convolutional networks. Key features include:
- 7×7 depthwise convolutions for larger receptive fields
- Inverted bottleneck design (expand 4× then compress)
- GELU activation and LayerNorm
- Four-stage hierarchical structure

### ConvNeXtBlock (Building Block)
The basic building block implements:
- Depthwise 7×7 convolution (groups=dim for channel-wise processing)
- LayerNorm for channel-wise normalization
- Inverted bottleneck MLP (1× → 4× → 1×)
- GELU activation (smoother than ReLU)
- Residual connection (skip connection for gradient flow)

Each block processes features with large receptive field and efficient channel mixing.

### Full Model Architecture
Stem layer (4×4 conv stride 4: 224×224 → 56×56, 96 channels)

## Dataset
ADNI (Alzheimer's Disease Neuroimaging Initiative) preprocessed brain MRI data.

**Structure**:
- Training and test splits pre-defined
- Two classes: AD (Alzheimer's Disease) and NC (Normal Control)
- Grayscale brain MRI slices in JPEG/PNG format

**Preprocessing**:
- Resized to 224×224 for ConvNeXt input
- Normalized to mean=0.5, std=0.5
- Training augmentation: random horizontal flip, random rotation (±10°)

## Training

**Hyperparameters**:
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 50
- Optimizer: AdamW with weight decay 0.01
- Loss function: CrossEntropyLoss

**Training process**:
- Trains for 50 epochs with validation after each epoch
- Saves best model checkpoint based on validation accuracy
- Displays training and validation loss and accuracy per epoch


## Requirements
**Python packages:**
- `torch==2.7.1`
- `torchvision==0.22.1`
- `Pillow==11.0.0`
- `CUDA==11.8`

**Hardware:**
- GPU with CUDA 11.8 support
- Minimum 8GB RAM

Training performed on Rangpur HPC cluster.

## Model Architecture

### ConvNeXtBlock
- Depthwise 7x7 convolution implemented
- TODO: Add normalization and MLP layers