import os
import torch

class LightweightConfig:
    DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    MASK_DIR = os.path.join(DATA_DIR, "masks")
    CHECKPOINT_PATH = "lightweight_checkpoint_AuraViT.pth"
    BEST_MODEL_PATH = "lightweight_best_AuraViT_model.pth"
    PRUNED_MODEL_PATH = "lightweight_pruned_AuraViT_model.pth"
    QUANTIZED_MODEL_PATH = "lightweight_quantized_AuraViT_model.pth"
    
    # Training hyperparameters optimized for lightweight model
    BATCH_SIZE = 8  # Can use larger batch size with smaller model
    LEARNING_RATE = 3e-4  # Higher LR works better for smaller models
    MIN_LEARNING_RATE = 1e-6
    MAX_LR_MULTIPLIER = 1.0
    MAX_EPOCHS = 200
    WARMUP_EPOCHS = 10
    PATIENCE = 20
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 0.5
    
    # Pruning settings
    ENABLE_PRUNING = True
    PRUNING_RATE = 0.3  # Prune 30% of weights
    
    CHECK_LOSS_FREQUENCY = 5
    LOSS_EXPLOSION_THRESHOLD = 10.0
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LIGHTWEIGHT MODEL CONFIGURATIONS
# Choose one: 'small', 'tiny', or 'mobile'
MODEL_SIZE = 'small'  # Change this to 'tiny' or 'mobile' for even smaller models

configs = {
    'small': {
        "image_size": 256, 
        "num_layers": 8,        # Reduced from 12
        "hidden_dim": 512,      # Reduced from 768
        "mlp_dim": 2048,        # Reduced from 3072
        "num_heads": 8,         # Reduced from 12
        "dropout_rate": 0.1, 
        "block_dropout_rate": 0.05,
        "patch_size": 16, 
        "num_channels": 1,
    },
    'tiny': {
        "image_size": 256,
        "num_layers": 6,        # Even fewer layers
        "hidden_dim": 384,      # Smaller hidden dim
        "mlp_dim": 1536,
        "num_heads": 6,
        "dropout_rate": 0.1,
        "block_dropout_rate": 0.05,
        "patch_size": 16,
        "num_channels": 1,
    },
    'mobile': {
        "image_size": 224,      # Smaller image size
        "num_layers": 4,        # Minimal layers
        "hidden_dim": 256,
        "mlp_dim": 1024,
        "num_heads": 4,
        "dropout_rate": 0.1,
        "block_dropout_rate": 0.05,
        "patch_size": 16,
        "num_channels": 1,
    }
}

def get_config():
    config = LightweightConfig()
    model_config = configs[MODEL_SIZE]
    model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
    return config, model_config, MODEL_SIZE
