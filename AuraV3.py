import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandAffined,
    RandRotate90d,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    EnsureTyped,
    Activations,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandFlipd,
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils import set_determinism
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import math
import time
import json

# For reproducibility
set_determinism(seed=42)

def setup_logging(log_file="enhanced_auravit_training.log"):
    """Enhanced logging setup with both file and console output"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================================================================================
# ENHANCED BLOCKS WITH IMPROVED STABILITY
# ==================================================================================

class ResBlock(nn.Module):
    """Enhanced Residual Block with better numerical stability"""
    def __init__(self, in_c, out_c, stride=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1)
            )
        
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.layers(x)
        out = out + identity
        return self.relu(out)

class DeconvBlock(nn.Module):
    """Enhanced Deconvolution block with better upsampling"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        
        # Better initialization
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return self.relu(x)

class AtrousConv(nn.Module):
    """Atrous convolution with improved stability"""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.atrous_conv[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.atrous_conv(x)

class ASPP(nn.Module):
    """Enhanced ASPP with dropout and better initialization"""
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        self.atrous_blocks = nn.ModuleList([
            AtrousConv(in_channels, out_channels, rate) for rate in rates
        ])
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15)  # Slightly increased dropout
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.shape[2:]
        x_1x1 = self.conv_1x1(x)
        x_atrous = [block(x) for block in self.atrous_blocks]
        x_global = self.global_avg_pool(x)
        x_global = F.interpolate(x_global, size=size, mode='bilinear', align_corners=False)

        x_cat = torch.cat([x_1x1] + x_atrous + [x_global], dim=1)
        return self.output_conv(x_cat)

class AttentionGate(nn.Module):
    """Enhanced Attention Gate with numerical stability"""
    def __init__(self, gate_channels, in_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels, eps=1e-5, momentum=0.1)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels, eps=1e-5, momentum=0.1)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain for attention
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, gate, x):
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Prevent extreme attention values
        psi = torch.clamp(psi, min=0.01, max=1.0)
        return x * psi

# ==================================================================================
# ENHANCED AURAVIT MODEL
# ==================================================================================

class EnhancedAuraViT(nn.Module):
    """Enhanced AuraViT with improved stability and performance"""
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # Enhanced ViT Encoder
        self.patch_embed = nn.Sequential(
            nn.Linear(cf["patch_size"]*cf["patch_size"]*cf["num_channels"], cf["hidden_dim"]),
            nn.LayerNorm(cf["hidden_dim"], eps=1e-6),
            nn.Dropout(cf["dropout_rate"])
        )
        
        # Better position embedding initialization
        self.pos_embed = nn.Parameter(torch.zeros(1, cf["num_patches"], cf["hidden_dim"]))
        nn.init.trunc_normal_(self.pos_embed, std=0.01)  # Smaller std
        
        self.pos_dropout = nn.Dropout(cf["dropout_rate"])
        
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"], 
                nhead=cf["num_heads"], 
                dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"], 
                activation=F.gelu, 
                batch_first=True,
                norm_first=True
            ) for _ in range(cf["num_layers"])
        ])
        
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(cf["hidden_dim"], eps=1e-6) for _ in range(4)
        ])

        # ASPP Module
        self.aspp = ASPP(cf["hidden_dim"], cf["hidden_dim"], rates=[6, 12, 18])

        # Attention Gates
        self.att_gate_1 = AttentionGate(512, 512, 256)
        self.att_gate_2 = AttentionGate(256, 256, 128)
        self.att_gate_3 = AttentionGate(128, 128, 64)
        self.att_gate_4 = AttentionGate(64, 64, 32)

        # Enhanced Segmentation Decoder
        dropout_rate = cf.get("block_dropout_rate", 0.1)
        
        self.seg_d1 = DeconvBlock(cf["hidden_dim"], 512)
        self.seg_s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 512), 
            ResBlock(512, 512, dropout_rate=dropout_rate)
        )
        self.seg_c1 = nn.Sequential(
            ResBlock(1024, 512, dropout_rate=dropout_rate), 
            ResBlock(512, 512, dropout_rate=dropout_rate)
        )

        self.seg_d2 = DeconvBlock(512, 256)
        self.seg_s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256), ResBlock(256, 256, dropout_rate=dropout_rate), 
            DeconvBlock(256, 256), ResBlock(256, 256, dropout_rate=dropout_rate)
        )
        self.seg_c2 = nn.Sequential(
            ResBlock(512, 256, dropout_rate=dropout_rate), 
            ResBlock(256, 256, dropout_rate=dropout_rate)
        )

        self.seg_d3 = DeconvBlock(256, 128)
        self.seg_s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128), ResBlock(128, 128, dropout_rate=dropout_rate),
            DeconvBlock(128, 128), ResBlock(128, 128, dropout_rate=dropout_rate), 
            DeconvBlock(128, 128), ResBlock(128, 128, dropout_rate=dropout_rate)
        )
        self.seg_c3 = nn.Sequential(
            ResBlock(256, 128, dropout_rate=dropout_rate), 
            ResBlock(128, 128, dropout_rate=dropout_rate)
        )

        self.seg_d4 = DeconvBlock(128, 64)
        self.seg_s4 = nn.Sequential(
            ResBlock(cf["num_channels"], 64, dropout_rate=dropout_rate), 
            ResBlock(64, 64, dropout_rate=dropout_rate)
        )
        self.seg_c4 = nn.Sequential(
            ResBlock(128, 64, dropout_rate=dropout_rate), 
            ResBlock(64, 64, dropout_rate=dropout_rate)
        )

        self.seg_output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        # Careful output initialization
        nn.init.xavier_uniform_(self.seg_output.weight, gain=0.01)
        nn.init.constant_(self.seg_output.bias, 0)

    def forward(self, inputs):
        # Input validation
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logger.warning("Invalid input detected")
            return torch.zeros_like(inputs)[:, :1]  # Return zeros for segmentation
            
        # ViT Encoder
        p = self.cf["patch_size"]
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf["num_patches"], -1)
        
        patch_embed = self.patch_embed(patches)
        x = self.pos_dropout(patch_embed + self.pos_embed)

        skip_connection_index = [2, 5, 8, 11]
        skip_connections = []
        
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if torch.isnan(x).any():
                logger.warning(f"NaN detected after transformer layer {i}")
                return torch.zeros_like(inputs)[:, :1]
                
            if i in skip_connection_index:
                norm_idx = len(skip_connections)
                normalized_skip = self.skip_norms[norm_idx](x)
                skip_connections.append(normalized_skip)
        
        z3, z6, z9, z12_features = skip_connections

        # Reshape feature maps
        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        # ASPP Module
        aspp_out = self.aspp(z12_reshaped)

        # Segmentation Decoder
        x_seg = self.seg_d1(aspp_out)
        s = self.seg_s1(z9)
        s = self.att_gate_1(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c1(x_seg)

        x_seg = self.seg_d2(x_seg)
        s = self.seg_s2(z6)
        s = self.att_gate_2(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c2(x_seg)

        x_seg = self.seg_d3(x_seg)
        s = self.seg_s3(z3)
        s = self.att_gate_3(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c3(x_seg)

        x_seg = self.seg_d4(x_seg)
        s = self.seg_s4(z0)
        s = self.att_gate_4(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c4(x_seg)

        seg_output = self.seg_output(x_seg)
        
        # Final validation
        if torch.isnan(seg_output).any() or torch.isinf(seg_output).any():
            logger.warning("Invalid output detected")
            return torch.zeros_like(seg_output)

        return seg_output

# ==================================================================================
# ENHANCED LOSS FUNCTION
# ==================================================================================
class RobustLoss(nn.Module):
    """Robust loss function with stability improvements"""
    def __init__(self, dice_weight=0.7, ce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Updated DiceCELoss initialization for newer MONAI versions
        self.dice_loss = DiceCELoss(
            to_onehot_y=False, 
            sigmoid=True, 
            smooth_nr=1e-5, 
            smooth_dr=1e-5,
            # Use lambda parameter instead of separate ce_weight and dice_weight
            lambda_dice=dice_weight,
            lambda_ce=ce_weight
        )

    def forward(self, predictions, targets):
        # Input validation
        if torch.isnan(predictions).any() or torch.isnan(targets).any():
            logger.warning("NaN detected in loss inputs")
            return torch.tensor(100.0, requires_grad=True, device=predictions.device)
        
        # Clamp predictions to prevent extreme values
        predictions = torch.clamp(predictions, min=-15, max=15)
        
        try:
            loss = self.dice_loss(predictions, targets)
            
            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                logger.warning("Invalid loss computed, returning fallback")
                return torch.tensor(100.0, requires_grad=True, device=predictions.device)
                
            return loss
            
        except Exception as e:
            logger.warning(f"Loss computation failed: {e}")
            return torch.tensor(100.0, requires_grad=True, device=predictions.device)

# ==================================================================================
# ADAPTIVE LEARNING RATE SCHEDULER
# ==================================================================================

class AdaptiveLRScheduler:
    """Adaptive learning rate scheduler with plateau detection"""
    def __init__(self, optimizer, initial_lr, warmup_epochs=15, patience=8, factor=0.5, 
                 min_lr=1e-7, cooldown=5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.cooldown = cooldown
        
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.epochs_without_improvement = 0
        self.cooldown_counter = 0
        self.current_lr = initial_lr * 0.1  # Start low
        
        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def step(self, metric=None):
        """Update learning rate based on epoch and optional metric"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            progress = self.current_epoch / self.warmup_epochs
            self.current_lr = self.initial_lr * (0.1 + 0.9 * progress)
        else:
            # Plateau detection phase
            if metric is not None:
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Reduce LR on plateau
                if (self.epochs_without_improvement >= self.patience and 
                    self.cooldown_counter == 0):
                    old_lr = self.current_lr
                    self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                    if self.current_lr < old_lr:
                        logger.info(f"Reducing LR from {old_lr:.2e} to {self.current_lr:.2e}")
                        self.epochs_without_improvement = 0
                        self.cooldown_counter = self.cooldown
                
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        self.current_epoch += 1
        return self.current_lr

# ==================================================================================
# TRAINING UTILITIES
# ==================================================================================

def validate_model_state(model):
    """Check model for NaN parameters"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN detected in parameter: {name}")
            return False
    return True

def robust_save_checkpoint(state, filepath, max_retries=3):
    """Robust checkpoint saving with retries"""
    for attempt in range(max_retries):
        try:
            temp_path = filepath + '.tmp'
            torch.save(state, temp_path)
            os.replace(temp_path, filepath)
            logger.info(f"Checkpoint saved successfully: {filepath}")
            return True
        except Exception as e:
            logger.warning(f"Checkpoint save attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    
    logger.error(f"Failed to save checkpoint after {max_retries} attempts")
    return False

def load_checkpoint_safely(filepath, model, optimizer, scheduler):
    """Safe checkpoint loading with validation"""
    if not os.path.exists(filepath):
        return None
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Validate checkpoint structure
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'best_metric']
        if not all(key in checkpoint for key in required_keys):
            logger.warning("Checkpoint missing required keys")
            return None
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        if not validate_model_state(model):
            logger.warning("Loaded model contains NaN parameters")
            return None
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update scheduler
        if hasattr(scheduler, 'current_epoch'):
            scheduler.current_epoch = checkpoint.get('scheduler_epoch', 0)
            scheduler.best_metric = checkpoint.get('best_metric', -float('inf'))
        
        logger.info(f"Checkpoint loaded successfully from epoch {checkpoint['epoch']}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None

# ==================================================================================
# MAIN TRAINING FUNCTION
# ==================================================================================

def main():
    try:
        logger.info("Starting Enhanced AuraViT Training")
        
        # Configuration
        class EnhancedConfig:
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            CHECKPOINT_PATH = "enhanced_auravit_checkpoint.pth"
            BEST_MODEL_PATH = "enhanced_auravit_best.pth"
            
            # Improved hyperparameters
            BATCH_SIZE = 3  # Reduced for stability
            LEARNING_RATE = 1.5e-5  # Slightly reduced
            MIN_LEARNING_RATE = 1e-7
            MAX_EPOCHS = 300
            WARMUP_EPOCHS = 20  # Longer warmup
            EARLY_STOPPING_PATIENCE = 25  # More patience
            LR_PATIENCE = 8  # LR reduction patience
            WEIGHT_DECAY = 1e-6  # Reduced
            GRAD_CLIP_NORM = 0.3  # More aggressive clipping
            
            # Enhanced monitoring
            SAVE_FREQUENCY = 5
            LOG_FREQUENCY = 1
            VALIDATION_FREQUENCY = 1
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = EnhancedConfig()
        
        # Model configuration
        model_config = {
            "image_size": 256, "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072,
            "num_heads": 12, "dropout_rate": 0.12, "block_dropout_rate": 0.08,
            "patch_size": 16, "num_channels": 1,
        }
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2

        logger.info(f"Using device: {config.DEVICE}")
        logger.info(f"Model configuration: {model_config}")

        # Data preparation (same as before but with better error handling)
        try:
            image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
            mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
            
            if len(image_files) == 0 or len(mask_files) == 0:
                raise ValueError("No data files found")
            
            data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]
            train_files, test_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
            train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)

            logger.info(f"Training samples: {len(train_files)}")
            logger.info(f"Validation samples: {len(val_files)}")
            logger.info(f"Test samples: {len(test_files)}")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

        # Enhanced data transforms with less aggressive augmentation
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], 
                               spatial_size=(model_config["image_size"], model_config["image_size"])),
            # Reduced augmentation intensity
            RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.25),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.25),
            RandAffined(
                keys=['image', 'label'], prob=0.2, translate_range=(3, 3),
                rotate_range=(np.pi / 60, np.pi / 60), scale_range=(0.03, 0.03),
                mode=('bilinear', 'nearest'),
            ),
            RandGaussianNoised(keys=["image"], prob=0.08, mean=0.0, std=0.003),
            RandScaleIntensityd(keys=["image"], factors=0.03, prob=0.15),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ])

        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], 
                               spatial_size=(model_config["image_size"], model_config["image_size"])),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ])

        # Create datasets and loaders
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                                num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                              num_workers=2, pin_memory=True, persistent_workers=True)

        # Initialize model and training components
        model = EnhancedAuraViT(model_config).to(config.DEVICE)
        loss_function = RobustLoss()
        
        # Enhanced optimizer with lower weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Adaptive scheduler
        lr_scheduler = AdaptiveLRScheduler(
            optimizer=optimizer,
            initial_lr=config.LEARNING_RATE,
            warmup_epochs=config.WARMUP_EPOCHS,
            patience=config.LR_PATIENCE,
            factor=0.6,  # Less aggressive reduction
            min_lr=config.MIN_LEARNING_RATE,
            cooldown=5
        )
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=True)
        
        # Metrics
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # Training state initialization
        start_epoch = 0
        best_metric = -1
        best_metric_epoch = -1
        train_losses = []
        val_losses = []
        dice_scores = []
        learning_rates = []
        early_stopping_counter = 0

        # Load checkpoint if available
        checkpoint = load_checkpoint_safely(config.CHECKPOINT_PATH, model, optimizer, lr_scheduler)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            best_metric = checkpoint.get('best_metric', best_metric)
            best_metric_epoch = checkpoint.get('best_metric_epoch', best_metric_epoch)
            train_losses = checkpoint.get('train_losses', train_losses)
            val_losses = checkpoint.get('val_losses', val_losses)
            dice_scores = checkpoint.get('dice_scores', dice_scores)
            learning_rates = checkpoint.get('learning_rates', learning_rates)
            early_stopping_counter = checkpoint.get('early_stopping_counter', early_stopping_counter)

        logger.info("Starting Enhanced AuraViT Training Loop")
        logger.info(f"Training from epoch {start_epoch + 1} to {config.MAX_EPOCHS}")

        # Main training loop
        for epoch in range(start_epoch, config.MAX_EPOCHS):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            gradient_norms = []
            
            progress_bar = tqdm(train_loader, 
                              desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Train]", 
                              unit="batch")
            
            for batch_idx, batch_data in enumerate(progress_bar):
                try:
                    inputs = batch_data["image"].to(config.DEVICE, non_blocking=True)
                    labels = batch_data["label"].to(config.DEVICE, non_blocking=True)
                    
                    # Skip invalid batches
                    if torch.isnan(inputs).any() or torch.isnan(labels).any():
                        logger.warning(f"Invalid batch {batch_idx}, skipping")
                        continue
                    
                    optimizer.zero_grad()

                    # Forward pass with mixed precision
                    with autocast(enabled=True):
                        seg_outputs = model(inputs)
                        loss = loss_function(seg_outputs, labels)
                    
                    # Skip if loss is invalid
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 50:
                        logger.warning(f"Invalid loss {loss.item():.4f} in batch {batch_idx}, skipping")
                        continue

                    # Backward pass
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping and validation
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                               max_norm=config.GRAD_CLIP_NORM)
                    
                    if torch.isnan(grad_norm):
                        logger.warning(f"NaN gradients detected in batch {batch_idx}, skipping")
                        optimizer.zero_grad()
                        continue
                    
                    gradient_norms.append(grad_norm.item())
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()
                    train_batches += 1
                    
                    # Update progress bar
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "LR": f"{current_lr:.2e}",
                        "GradNorm": f"{grad_norm:.3f}"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"CUDA OOM in batch {batch_idx}, clearing cache")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        logger.error(f"Runtime error in training: {e}")
                        raise
                except Exception as e:
                    logger.warning(f"Error in training batch {batch_idx}: {e}")
                    continue

            # Calculate average training loss
            if train_batches == 0:
                logger.error("No valid training batches processed!")
                break
            
            avg_train_loss = train_loss / train_batches
            avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                dice_metric.reset()
                
                for val_batch_idx, val_data in enumerate(val_loader):
                    try:
                        val_inputs = val_data["image"].to(config.DEVICE, non_blocking=True)
                        val_labels = val_data["label"].to(config.DEVICE, non_blocking=True)
                        
                        # Skip invalid batches
                        if torch.isnan(val_inputs).any() or torch.isnan(val_labels).any():
                            continue
                        
                        val_seg_outputs = model(val_inputs)
                        
                        # Skip if outputs are invalid
                        if torch.isnan(val_seg_outputs).any():
                            continue
                        
                        val_batch_loss = loss_function(val_seg_outputs, val_labels)
                        
                        if not (torch.isnan(val_batch_loss) or torch.isinf(val_batch_loss)):
                            val_loss += val_batch_loss.item()
                            val_batches += 1
                        
                        # Compute metrics
                        val_outputs_post = [post_pred(i) for i in decollate_batch(val_seg_outputs)]
                        dice_metric(y_pred=val_outputs_post, y=val_labels)
                        
                    except Exception as e:
                        logger.warning(f"Error in validation batch {val_batch_idx}: {e}")
                        continue
                
                try:
                    dice_score = dice_metric.aggregate().item()
                except:
                    dice_score = dice_scores[-1] if dice_scores else 0.0
                    logger.warning("Failed to compute dice score, using previous value")

            # Calculate averages
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            
            # Update learning rate scheduler
            current_lr = lr_scheduler.step(metric=dice_score)
            
            # Record metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            dice_scores.append(dice_score)
            learning_rates.append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1:3d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Dice: {dice_score:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"GradNorm: {avg_grad_norm:.3f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Best Dice: {best_metric:.4f}"
            )
            
            # Save best model
            if dice_score > best_metric:
                best_metric = dice_score
                best_metric_epoch = epoch + 1
                early_stopping_counter = 0
                
                # Save best model
                try:
                    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                    logger.info(f"New best model saved! Dice: {best_metric:.4f}")
                except Exception as e:
                    logger.error(f"Failed to save best model: {e}")
            else:
                early_stopping_counter += 1

            # Save checkpoint periodically
            if (epoch + 1) % config.SAVE_FREQUENCY == 0 or early_stopping_counter == 0:
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_epoch': lr_scheduler.current_epoch,
                    'best_metric': best_metric,
                    'best_metric_epoch': best_metric_epoch,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'dice_scores': dice_scores,
                    'learning_rates': learning_rates,
                    'early_stopping_counter': early_stopping_counter
                }
                
                robust_save_checkpoint(checkpoint_state, config.CHECKPOINT_PATH)

            # Early stopping check
            if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}. No improvement for {config.EARLY_STOPPING_PATIENCE} epochs.")
                break
            
            # Check for training instability
            if len(train_losses) >= 5:
                recent_losses = train_losses[-5:]
                if any(loss > 10.0 for loss in recent_losses):
                    logger.warning("Training instability detected. Consider reducing learning rate.")

        # Training completed
        logger.info(f"Training completed! Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

        # Plot training curves
        logger.info("Generating training curves...")
        
        plt.figure(figsize=(20, 6))
        
        plt.subplot(1, 4, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 2)
        plt.plot(range(1, len(dice_scores) + 1), dice_scores, 'g-', label='Dice Score', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 3)
        plt.plot(range(1, len(learning_rates) + 1), learning_rates, 'm-', label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.subplot(1, 4, 4)
        if len(train_losses) >= 10:
            smooth_train = np.convolve(train_losses, np.ones(5)/5, mode='valid')
            smooth_val = np.convolve(val_losses, np.ones(5)/5, mode='valid')
            plt.plot(range(3, len(smooth_train) + 3), smooth_train, 'b-', label='Smoothed Train Loss', linewidth=2)
            plt.plot(range(3, len(smooth_val) + 3), smooth_val, 'r-', label='Smoothed Val Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Smoothed Loss Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("enhanced_auravit_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Final evaluation on test set
        logger.info("Running final test evaluation...")
        
        test_ds = Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

        # Load best model
        if os.path.exists(config.BEST_MODEL_PATH):
            model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
            logger.info("Best model loaded for testing")
        
        model.eval()
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for test_data in tqdm(test_loader, desc="Testing"):
                try:
                    test_inputs = test_data["image"].to(config.DEVICE, non_blocking=True)
                    test_labels = test_data["label"].to(config.DEVICE, non_blocking=True)
                    
                    if torch.isnan(test_inputs).any() or torch.isnan(test_labels).any():
                        continue
                    
                    test_seg_outputs = model(test_inputs)
                    
                    if torch.isnan(test_seg_outputs).any():
                        continue
                    
                    test_outputs_post = [post_pred(i) for i in decollate_batch(test_seg_outputs)]
                    dice_metric(y_pred=test_outputs_post, y=test_labels)
                    
                    # Store for additional metrics
                    test_predictions.extend([pred.cpu().numpy() for pred in test_outputs_post])
                    test_targets.extend([label.cpu().numpy() for label in decollate_batch(test_labels)])
                    
                except Exception as e:
                    logger.warning(f"Error in test batch: {e}")
                    continue

        # Compute final metrics
        try:
            final_dice = dice_metric.aggregate().item()
            final_iou = final_dice / (2 - final_dice) if final_dice > 0 else 0
            
            # Additional metrics computation
            if test_predictions and test_targets:
                test_predictions = np.array(test_predictions)
                test_targets = np.array(test_targets)
                
                # Flatten for pixel-wise metrics
                pred_flat = test_predictions.flatten()
                target_flat = test_targets.flatten()
                
                # Compute additional metrics
                tp = np.sum((pred_flat == 1) & (target_flat == 1))
                fp = np.sum((pred_flat == 1) & (target_flat == 0))
                fn = np.sum((pred_flat == 0) & (target_flat == 1))
                tn = np.sum((pred_flat == 0) & (target_flat == 0))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            else:
                sensitivity = specificity = precision = accuracy = f1_score = 0.0
                
        except Exception as e:
            logger.error(f"Failed to compute final metrics: {e}")
            final_dice = final_iou = sensitivity = specificity = precision = accuracy = f1_score = 0.0

        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'dice_scores': dice_scores,
            'learning_rates': learning_rates,
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
            'final_test_dice': final_dice,
            'final_test_iou': final_iou,
            'model_config': model_config,
            'training_config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'max_epochs': config.MAX_EPOCHS,
                'warmup_epochs': config.WARMUP_EPOCHS
            }
        }
        
        try:
            with open('enhanced_auravit_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            logger.info("Training history saved")
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")

        # Final results
        logger.info("="*80)
        logger.info("ENHANCED AURAVIT FINAL RESULTS")
        logger.info("="*80)
        logger.info(f"Best Validation Dice Score: {best_metric:.4f} (Epoch {best_metric_epoch})")
        logger.info(f"Final Test Dice Score: {final_dice:.4f}")
        logger.info(f"Final Test IoU: {final_iou:.4f}")
        logger.info(f"Test Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Test Specificity: {specificity:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-Score: {f1_score:.4f}")
        
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Model Size: {model_size_mb:.2f} MB")
        logger.info("="*80)
        
        logger.info("Enhanced AuraViT training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        logger.info("Training cleanup completed")

if __name__ == "__main__":
    main()
