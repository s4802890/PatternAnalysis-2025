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

## Dataset
ADNI (Alzheimer's Disease Neuroimaging Initiative) preprocessed brain MRI data.

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