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

# For reproducibility
set_determinism(seed=42)

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('auravit_training_stable.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

#BLOCKS 

class ResBlock(nn.Module):
    """
    A Residual Block with improved stability.
    Added initialization and dropout for better training stability.
    """
    def __init__(self, in_c, out_c, stride=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # Added dropout for stability
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Improved initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layers(x)
        x = x + shortcut
        return self.relu(x)

class DeconvBlock(nn.Module):
    """
    Standard Transposed Convolution block for upsampling with improved initialization.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        # Better initialization for transposed conv
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0)

    def forward(self, x):
        return self.deconv(x)

class AtrousConv(nn.Module):
    """
    A single Atrous Convolution block with improved stability.
    """
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Proper initialization
        nn.init.kaiming_normal_(self.atrous_conv[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.atrous_conv(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module with stability improvements.
    """
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_blocks = nn.ModuleList([
            AtrousConv(in_channels, out_channels, rate) for rate in rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Added dropout for stability
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
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
    """
    Attention Gate with improved numerical stability.
    """
    def __init__(self, gate_channels, in_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # Xavier for attention weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, gate, x):
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Clamp attention weights to prevent extreme values
        psi = torch.clamp(psi, min=1e-8, max=1.0)
        return x * psi

# ======================================================================================
# STABILITY-ENHANCED AURAVIT MODEL
# ======================================================================================

class StableEnhancedAuraViT(nn.Module):
    """
    Stability-Enhanced AuraViT with multiple improvements:
    1. Better weight initialization
    2. Gradient clipping safeguards
    3. Numerical stability improvements
    4. Loss checking mechanisms
    """
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # --- ENHANCED ViT ENCODER WITH STABILITY ---
        self.patch_embed = nn.Sequential(
            nn.Linear(cf["patch_size"]*cf["patch_size"]*cf["num_channels"], cf["hidden_dim"]),
            nn.LayerNorm(cf["hidden_dim"]),
            nn.Dropout(cf["dropout_rate"])
        )
        
        # Improved position embeddings initialization
        self.pos_embed = nn.Parameter(torch.zeros(1, cf["num_patches"], cf["hidden_dim"]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_dropout = nn.Dropout(cf["dropout_rate"])
        
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"], nhead=cf["num_heads"], dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"], activation=F.gelu, batch_first=True,
                norm_first=True  # Pre-norm for better stability
            ) for _ in range(cf["num_layers"])
        ])
        
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(cf["hidden_dim"]) for _ in range(4)
        ])

        # --- ASPP MODULE ---
        self.aspp = ASPP(cf["hidden_dim"], cf["hidden_dim"], rates=[6, 12, 18])

        # --- ATTENTION GATES ---
        self.att_gate_1 = AttentionGate(gate_channels=512, in_channels=512, inter_channels=256)
        self.att_gate_2 = AttentionGate(gate_channels=256, in_channels=256, inter_channels=128)
        self.att_gate_3 = AttentionGate(gate_channels=128, in_channels=128, inter_channels=64)
        self.att_gate_4 = AttentionGate(gate_channels=64, in_channels=64, inter_channels=32)

        # --- ENHANCED SEGMENTATION DECODER ---
        dropout_rate = cf.get("block_dropout_rate", 0.1)
        self.seg_d1 = DeconvBlock(cf["hidden_dim"], 512)
        self.seg_s1 = nn.Sequential(DeconvBlock(cf["hidden_dim"], 512), ResBlock(512, 512, dropout_rate=dropout_rate))
        self.seg_c1 = nn.Sequential(ResBlock(512+512, 512, dropout_rate=dropout_rate), ResBlock(512, 512, dropout_rate=dropout_rate))

        self.seg_d2 = DeconvBlock(512, 256)
        self.seg_s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256), ResBlock(256, 256, dropout_rate=dropout_rate), 
            DeconvBlock(256, 256), ResBlock(256, 256, dropout_rate=dropout_rate)
        )
        self.seg_c2 = nn.Sequential(ResBlock(256+256, 256, dropout_rate=dropout_rate), ResBlock(256, 256, dropout_rate=dropout_rate))

        self.seg_d3 = DeconvBlock(256, 128)
        self.seg_s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128), ResBlock(128, 128, dropout_rate=dropout_rate), DeconvBlock(128, 128),
            ResBlock(128, 128, dropout_rate=dropout_rate), DeconvBlock(128, 128), ResBlock(128, 128, dropout_rate=dropout_rate)
        )
        self.seg_c3 = nn.Sequential(ResBlock(128+128, 128, dropout_rate=dropout_rate), ResBlock(128, 128, dropout_rate=dropout_rate))

        self.seg_d4 = DeconvBlock(128, 64)
        self.seg_s4 = nn.Sequential(ResBlock(cf["num_channels"], 64, dropout_rate=dropout_rate), ResBlock(64, 64, dropout_rate=dropout_rate))
        self.seg_c4 = nn.Sequential(ResBlock(64+64, 64, dropout_rate=dropout_rate), ResBlock(64, 64, dropout_rate=dropout_rate))

        self.seg_output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.seg_output.weight, gain=0.1)
        nn.init.constant_(self.seg_output.bias, 0)

    def forward(self, inputs):
        # Check for NaN inputs
        if torch.isnan(inputs).any():
            raise ValueError("NaN detected in input tensor")
            
        # 1. Enhanced ViT Encoder
        p = self.cf["patch_size"]
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf["num_patches"], -1)
        patch_embed = self.patch_embed(patches)

        # Add position embeddings
        x = self.pos_dropout(patch_embed + self.pos_embed)

        skip_connection_index = [2, 5, 8, 11]
        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            # Check for NaN after each transformer layer
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after transformer layer {i}")
            if i in skip_connection_index:
                norm_idx = len(skip_connections)
                normalized_skip = self.skip_norms[norm_idx](x)
                skip_connections.append(normalized_skip)
        
        z3, z6, z9, z12_features = skip_connections

        # Reshape all feature maps
        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        # 2. ASPP Module
        aspp_out = self.aspp(z12_reshaped)

        # 3. Enhanced Segmentation Decoder Path
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
        
        # Final NaN check
        if torch.isnan(seg_output).any():
            raise ValueError("NaN detected in model output")

        return seg_output

# ======================================================================================
# STABLE LOSS FUNCTION WITH NaN CHECKING
# ======================================================================================
class StableLoss(nn.Module):
    """
    Stable loss function with NaN detection and gradient clipping.
    """
    def __init__(self):
        super().__init__()
        self.seg_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)

    def forward(self, seg_preds, seg_labels):
        # Check for NaN in predictions and labels
        if torch.isnan(seg_preds).any():
            logger.warning("NaN detected in predictions! Returning large finite loss.")
            return torch.tensor(1000.0, requires_grad=True, device=seg_preds.device)
        
        if torch.isnan(seg_labels).any():
            raise ValueError("NaN detected in ground truth labels")
        
        # Clamp predictions to prevent extreme values
        seg_preds = torch.clamp(seg_preds, min=-10, max=10)
        
        loss = self.seg_loss(seg_preds, seg_labels)
        
        # Check if loss is NaN or infinite
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or Inf loss detected! Returning fallback loss.")
            return torch.tensor(1000.0, requires_grad=True, device=seg_preds.device)
        
        return loss

# ======================================================================================
# STABLE LEARNING RATE SCHEDULER
# ======================================================================================
class StableLRScheduler:
    """
    Stable learning rate scheduler that prevents excessive LR values.
    """
    def __init__(self, optimizer, initial_lr, warmup_epochs, max_epochs, min_lr=1e-6, max_lr_multiplier=1.0):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = initial_lr * max_lr_multiplier  # Prevent LR from going too high
        self.current_epoch = 0
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr * 0.1  # Start with 10% of initial LR

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (0.1 + 0.9 * self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing from initial_lr to min_lr
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Ensure LR stays within bounds
        lr = max(self.min_lr, min(lr, self.max_lr))
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

def check_gradients(model, max_grad_norm=1.0):
    """
    Check and log gradient statistics to detect gradient explosion.
    """
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Check for NaN gradients
            if torch.isnan(param.grad).any():
                logger.error(f"NaN gradient detected in {name}")
                return False, float('inf')
    
    total_norm = total_norm ** (1. / 2)
    
    
    return True, total_norm

def main():
    try:
        # ======================================================================================
        # IMPROVED CONFIGURATION
        # ======================================================================================
        class StableTrainingConfig:
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            CHECKPOINT_PATH = "stable_checkpoint_Enhanced_AuraViT.pth"
            BEST_MODEL_PATH = "stable_best_Enhanced_AuraViT_model.pth"
            
            # CONSERVATIVE HYPERPARAMETERS FOR STABILITY
            BATCH_SIZE = 4
            LEARNING_RATE = 2e-5  # Reduced from 5e-5
            MIN_LEARNING_RATE = 1e-6  # Higher minimum
            MAX_LR_MULTIPLIER = 1.0  # Prevent LR from exceeding initial value
            MAX_EPOCHS = 200
            WARMUP_EPOCHS = 15  # Longer warmup
            PATIENCE = 20  # More patience
            WEIGHT_DECAY = 1e-5  # Reduced weight decay
            GRAD_CLIP_NORM = 0.5  # Aggressive gradient clipping
            
            # Stability checks
            CHECK_LOSS_FREQUENCY = 5  # Check for unstable training every N epochs
            LOSS_EXPLOSION_THRESHOLD = 10.0  # If loss > this, reduce LR
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Enhanced model configuration with stability improvements
        model_config = {
            "image_size": 256, "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072,
            "num_heads": 12, "dropout_rate": 0.1, "block_dropout_rate": 0.05,  # Reduced dropout
            "patch_size": 16, "num_channels": 1,
        }
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
        config = StableTrainingConfig()

        logger.info(f"Using device: {config.DEVICE}")
        logger.info("Starting stable training with conservative hyperparameters")

        # ======================================================================================
        # DATA PREPARATION (SAME AS BEFORE)
        # ======================================================================================
        image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
        mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
        data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

        train_files, test_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)

        logger.info(f"Total samples: {len(data_dicts)}")
        logger.info(f"Training samples: {len(train_files)}")
        logger.info(f"Validation samples: {len(val_files)}")
        logger.info(f"Testing samples: {len(test_files)}")

        # Transforms (same as before but with less aggressive augmentation)
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            # Reduced augmentation intensity for stability
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
            RandAffined(
                keys=['image', 'label'], prob=0.3, translate_range=(5, 5),
                rotate_range=(np.pi / 36, np.pi / 36), scale_range=(0.05, 0.05),
                mode=('bilinear', 'nearest'),
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.005),  # Reduced noise
            RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.2),  # Reduced intensity
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ])

        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ])

        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

        # ======================================================================================
        # STABLE MODEL SETUP
        # ======================================================================================
        model = StableEnhancedAuraViT(model_config).to(config.DEVICE)
        loss_function = StableLoss()
        
        # Conservative optimizer settings
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8  # Slightly larger epsilon for stability
        )
        
        # Stable learning rate scheduler
        lr_scheduler = StableLRScheduler(
            optimizer=optimizer,
            initial_lr=config.LEARNING_RATE,
            warmup_epochs=config.WARMUP_EPOCHS,
            max_epochs=config.MAX_EPOCHS,
            min_lr=config.MIN_LEARNING_RATE,
            max_lr_multiplier=config.MAX_LR_MULTIPLIER
        )
        
        # Mixed precision with stability
        scaler = GradScaler()
        
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # ======================================================================================
        # CHECKPOINT LOADING WITH STABILITY CHECKS
        # ======================================================================================
        start_epoch = 0
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        accuracy_values = []
        patience_counter = 0
        lr_reduction_counter = 0

        if os.path.exists(config.CHECKPOINT_PATH):
            try:
                checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_metric = checkpoint.get('best_metric', best_metric)
                epoch_loss_values = checkpoint.get('epoch_loss_values', epoch_loss_values)
                metric_values = checkpoint.get('metric_values', metric_values)
                accuracy_values = checkpoint.get('accuracy_values', accuracy_values)
                patience_counter = checkpoint.get('patience_counter', patience_counter)
                lr_reduction_counter = checkpoint.get('lr_reduction_counter', 0)
                
                # Update scheduler state
                if 'scheduler_epoch' in checkpoint:
                    lr_scheduler.current_epoch = checkpoint['scheduler_epoch']
                
                logger.info(f"Checkpoint found! Resuming training from epoch {start_epoch}.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
                start_epoch = 0
        else:
            logger.info("No checkpoint found. Starting training from scratch.")

        # ======================================================================================
        # STABLE TRAINING AND VALIDATION LOOP
        # ======================================================================================
        logger.info("Starting Stable Enhanced AuraViT Training")
        consecutive_bad_epochs = 0
        
        for epoch in range(start_epoch, config.MAX_EPOCHS):
            # Update learning rate
            current_lr = lr_scheduler.step()
            
            model.train()
            epoch_loss = 0
            batch_count = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Training]", unit="batch")
            
            for batch_data in progress_bar:
                try:
                    inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
                    
                    # Check for NaN in inputs
                    if torch.isnan(inputs).any() or torch.isnan(labels).any():
                        logger.warning("NaN detected in batch data. Skipping batch.")
                        continue
                    
                    optimizer.zero_grad()

                    # Mixed precision forward pass with error handling
                    with autocast():
                        try:
                            seg_outputs = model(inputs)
                            loss = loss_function(seg_outputs, labels)
                        except ValueError as e:
                            if "NaN" in str(e):
                                logger.warning(f"NaN detected during forward pass: {e}. Skipping batch.")
                                continue
                            else:
                                raise

                    # Check for invalid loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning("Invalid loss detected. Skipping batch.")
                        continue

                    # Mixed precision backward pass with gradient checking
                    scaler.scale(loss).backward()
                    
                    # Check gradients before clipping
                    grad_ok, grad_norm = check_gradients(model, config.GRAD_CLIP_NORM)
                    if not grad_ok:
                        logger.warning("Invalid gradients detected. Skipping batch.")
                        optimizer.zero_grad()
                        continue
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
                    
                    # Optimizer step with error handling
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "LR": f"{current_lr:.2e}",
                        "GradNorm": f"{grad_norm:.3f}"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error("CUDA out of memory. Clearing cache and skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        logger.error(f"Runtime error during training: {e}")
                        raise

            if batch_count == 0:
                logger.error("No valid batches processed in this epoch!")
                break
                
            avg_epoch_loss = epoch_loss / batch_count
            epoch_loss_values.append(avg_epoch_loss)

            # Validation
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                dice_metric.reset()
                cm_metric = ConfusionMatrixMetric(
                    include_background=False, 
                    metric_name=["accuracy"], 
                    reduction="mean"
                )
                
                for val_data in val_loader:
                    try:
                        val_inputs, val_labels = val_data["image"].to(config.DEVICE), val_data["label"].to(config.DEVICE)
                        
                        # Skip batch if NaN detected
                        if torch.isnan(val_inputs).any() or torch.isnan(val_labels).any():
                            continue
                        
                        seg_outputs = model(val_inputs)
                        
                        # Check for NaN outputs
                        if torch.isnan(seg_outputs).any():
                            continue
                        
                        val_batch_loss = loss_function(seg_outputs, val_labels)
                        if not (torch.isnan(val_batch_loss) or torch.isinf(val_batch_loss)):
                            val_loss += val_batch_loss.item()
                            val_batch_count += 1
                        
                        val_outputs_post = [post_pred(i) for i in decollate_batch(seg_outputs)]
                        dice_metric(y_pred=val_outputs_post, y=val_labels)
                        cm_metric(y_pred=val_outputs_post, y=val_labels)
                        
                    except Exception as e:
                        logger.warning(f"Error during validation: {e}. Skipping batch.")
                        continue
                
                try:
                    metric = dice_metric.aggregate().item()
                    accuracy = cm_metric.aggregate()[0].item()
                except:
                    logger.warning("Error computing metrics. Using previous values.")
                    metric = metric_values[-1] if metric_values else 0.0
                    accuracy = accuracy_values[-1] if accuracy_values else 0.0
                
                metric_values.append(metric)
                accuracy_values.append(accuracy)

            # Average validation loss
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {metric:.4f} | Accuracy: {accuracy:.4f} | LR: {current_lr:.2e} | Best Dice: {best_metric:.4f}")

            # Stability checks
            if avg_epoch_loss > config.LOSS_EXPLOSION_THRESHOLD:
                consecutive_bad_epochs += 1
                logger.warning(f"High loss detected ({avg_epoch_loss:.4f}). Bad epochs: {consecutive_bad_epochs}")
                
                if consecutive_bad_epochs >= 3:
                    logger.warning("Multiple consecutive high-loss epochs. Reducing learning rate.")
                    current_lr = current_lr * 0.5
                    lr_scheduler = StableLRScheduler(
                        optimizer=optimizer,
                        initial_lr=current_lr,
                        warmup_epochs=5,  # Short warmup after LR reduction
                        max_epochs=config.MAX_EPOCHS,
                        min_lr=config.MIN_LEARNING_RATE,
                        max_lr_multiplier=config.MAX_LR_MULTIPLIER
                    )
                    consecutive_bad_epochs = 0
                    lr_reduction_counter += 1
                    
                    if lr_reduction_counter >= 3:
                        logger.error("Learning rate reduced 3 times. Stopping training.")
                        break
            else:
                consecutive_bad_epochs = 0

            # Model saving and early stopping
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                patience_counter = 0
                try:
                    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                    logger.info(f"New best model saved! Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
                except Exception as e:
                    logger.error(f"Failed to save best model: {e}")
            else:
                patience_counter += 1

            # Save checkpoint with enhanced state
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_epoch': lr_scheduler.current_epoch,
                    'best_metric': best_metric,
                    'epoch_loss_values': epoch_loss_values,
                    'metric_values': metric_values,
                    'accuracy_values': accuracy_values,
                    'patience_counter': patience_counter,
                    'lr_reduction_counter': lr_reduction_counter,
                    'consecutive_bad_epochs': consecutive_bad_epochs
                }, config.CHECKPOINT_PATH)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

            # Early stopping check
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}. No improvement for {config.PATIENCE} epochs.")
                break

        logger.info(f"Training finished. Best Dice score of {best_metric:.4f} at epoch {best_metric_epoch}.")

        # ======================================================================================
        # PLOTTING AND FINAL EVALUATION
        # ======================================================================================
        logger.info("Plotting training curves...")
        plt.figure("train", (18, 6))
        
        plt.subplot(1, 3, 1)
        plt.title("Training Loss")
        plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, 'b-', label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.title("Validation Dice Score")
        plt.plot(range(1, len(metric_values) + 1), metric_values, 'g-', label='Validation Dice')
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.title("Validation Accuracy")
        plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, 'r-', label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("stable_training_curves_AuraViT.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Final evaluation on test set
        logger.info("Running final evaluation on the test set...")
        test_ds = Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

        if os.path.exists(config.BEST_MODEL_PATH):
            model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
        model.eval()
        
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        cm_metric = ConfusionMatrixMetric(
            include_background=False, metric_name=["sensitivity", "specificity", "accuracy"], reduction="mean"
        )

        with torch.no_grad():
            for test_data in tqdm(test_loader, desc="Testing"):
                try:
                    test_inputs, test_labels = test_data["image"].to(config.DEVICE), test_data["label"].to(config.DEVICE)
                    
                    if torch.isnan(test_inputs).any() or torch.isnan(test_labels).any():
                        continue
                    
                    test_seg_outputs = model(test_inputs)
                    
                    if torch.isnan(test_seg_outputs).any():
                        continue
                    
                    test_outputs_post = [post_pred(i) for i in decollate_batch(test_seg_outputs)]
                    
                    dice_metric(y_pred=test_outputs_post, y=test_labels)
                    cm_metric(y_pred=test_outputs_post, y=test_labels)
                except Exception as e:
                    logger.warning(f"Error during testing: {e}. Skipping sample.")
                    continue

        try:
            mean_dice_test = dice_metric.aggregate().item()
            cm_value = cm_metric.aggregate()
            sensitivity = cm_value[0].item()
            specificity = cm_value[1].item()
            accuracy = cm_value[2].item()
            iou_test = mean_dice_test / (2 - mean_dice_test) if mean_dice_test > 0 else 0
        except:
            logger.error("Failed to compute final metrics")
            mean_dice_test = sensitivity = specificity = accuracy = iou_test = 0.0
        
        logger.info("="*60)
        logger.info("FINAL TEST METRICS (Stable Enhanced AuraViT)")
        logger.info("="*60)
        logger.info(f"Mean Dice Score: {mean_dice_test:.4f}")
        logger.info(f"Intersection over Union (IoU): {iou_test:.4f}")
        logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("="*60)
        
        # Model summary
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        total_params = count_parameters(model)
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        model_size_mb = total_params * 4 / (1024 * 1024)
        logger.info(f"Approximate model size: {model_size_mb:.2f} MB")
        
        logger.info("Stable Enhanced AuraViT training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()