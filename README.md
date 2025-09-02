# LungSeg
AuraViT: A Vision Transformer for Medical Image Segmentation
AuraViT is a deep learning project for medical image segmentation, featuring a hybrid architecture that leverages the strengths of Vision Transformers (ViT), Atrous Spatial Pyramid Pooling (ASPP), and residual convolutional networks. This repository documents the evolution of the model from an experimental concept (aura.py) to a stable, production-ready version (AuraV3.py).

The primary goal of this project is to create a robust and accurate model for segmenting tumors in medical scans, specifically demonstrated on the NSCLC (Non-Small Cell Lung Cancer) Radiomics dataset.

## Key Features
Hybrid Architecture: Combines a ViT encoder for capturing global context, an ASPP module for multi-scale feature extraction, and a CNN-based decoder with residual connections for precise localization.

Model Evolution: Includes two distinct versions:

AuraViT: An experimental model incorporating a Variational Autoencoder (VAE) for regularization.

StableEnhancedAuraViT: A refined, production-focused model that replaces the VAE with Attention Gates and integrates numerous stability enhancements.

Advanced Training Stability: The final version (AuraV3.py) is built for robustness, featuring gradient clipping, NaN detection, a stable learning rate scheduler, conservative hyperparameters, and advanced weight initialization techniques.

High Performance: Utilizes mixed-precision training (torch.cuda.amp) for faster performance and reduced memory usage.

Built with MONAI: Leverages the MONAI framework for data loading, transformations, and evaluation, ensuring best practices in medical imaging AI.

## Model Evolution
This project showcases a deliberate evolution from an experimental architecture to a stable, high-performance model.

1. AuraViT (aura.py) - The Experimental Hybrid
The initial version of AuraViT was designed as a novel combination of several state-of-the-art concepts.

ViT Encoder: Serves as the backbone, extracting powerful, context-aware features from image patches.

VAE Pathway: A VAE head was integrated to regularize the latent space, aiming to improve data efficiency and generalization. The model was trained with a composite loss function including segmentation loss (DiceCE), reconstruction loss (MSE), and KL-Divergence loss.

ASPP Module: Placed after the encoder to capture contextual information at multiple scales.

SegResNet-style Decoder: A convolutional decoder with residual blocks to reconstruct the final segmentation map.

While innovative, this architecture could be prone to training instabilities due to the complex interaction of its components and the sensitive nature of the VAE loss terms.

2. StableEnhancedAuraViT (AuraV3.py) - The Production-Ready Model
After identifying the training challenges of the original model, AuraV3.py was developed with an explicit focus on stability, reproducibility, and performance.

Key Architectural Changes:
Removal of VAE: The VAE pathway was removed to simplify the architecture and eliminate a major source of training instability.

Addition of Attention Gates: Attention Gates were integrated into the decoder's skip connections. This allows the model to learn to focus on relevant structures from the encoder feature maps, improving segmentation accuracy without the overhead of a VAE.

Stability Enhancements:
Robust Initializations: Implemented Kaiming (kaiming_normal_) and Xavier (xavier_uniform_) weight initializations in convolutional and attention layers to ensure a good starting point for training.

Pre-Norm Transformers: The ViT encoder layers use norm_first=True, a pre-normalization technique known to stabilize training for deep transformers.

Dropout Regularization: Added carefully placed dropout layers in residual blocks and the ASPP module to prevent overfitting.

Stable Loss Function: A wrapper around DiceCELoss that checks for and handles NaN/Inf values, preventing training crashes.

Advanced Training Loop: The main training script includes:

Aggressive Gradient Clipping: Prevents exploding gradients.

NaN/Inf Detection: Proactively checks for invalid values in inputs, model outputs, gradients, and loss.

Stable LR Scheduler: A custom learning rate scheduler with a long warmup phase and cosine annealing, bounded to prevent extreme learning rates.

Dynamic LR Reduction: Automatically reduces the learning rate if the training loss explodes, attempting to self-stabilize.

## Setup and Installation
Prerequisites
Python 3.9 or later

NVIDIA GPU with CUDA support (for optimal performance)

Anaconda or Miniconda (recommended for managing environments)


## Usage
1. Data Preparation

2. Training the Model

3. Modifying Configuration
All key hyperparameters (learning rate, batch size, epochs, etc.) and model configuration details are located in the StableTrainingConfig and model_config objects at the top of the main() function in AuraV3.py.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
This project was built using the powerful PyTorch deep learning framework.

Medical imaging components are heavily based on the excellent MONAI library.
