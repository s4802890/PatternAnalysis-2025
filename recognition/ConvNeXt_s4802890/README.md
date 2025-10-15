# Alzheimer's Disease Classification using ConvNeXt

## Problem Description
Binary classification of brain MRIs from the ADNI dataset:
- **AD**: Alzheimer's Disease  
- **NC**: Normal Control (Cognitively Normal)
 
**Target accuracy**: ≥ 0.8 on test set

## Model
ConvNeXt-Tiny with ImageNet pre-training, fine-tuned for binary classification.

ConvNeXt is a modernized CNN architecture that achieves competitive performance with Vision Transformers while maintaining the simplicity and efficiency of convolutional networks.

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
