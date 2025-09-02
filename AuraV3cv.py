import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
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
import json

# For reproducibility
set_determinism(seed=42)

# Setup logging
def setup_logging(fold_num=None):
    log_filename = f'auravit_cv_fold_{fold_num}_stable.log' if fold_num is not None else 'auravit_cv_stable.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
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
    Fixed to handle small spatial dimensions that cause BatchNorm issues.
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
        # Fixed global average pooling branch - removed BatchNorm to avoid 1x1 spatial dim issues
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),  # Changed to bias=True since no BatchNorm
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

# ======================================================================================
# CROSS-VALIDATION STATE MANAGEMENT
# ======================================================================================
class CVStateManager:
    """
    Manages cross-validation state for resumable training.
    """
    def __init__(self, n_folds, cv_state_file='cv_state.json'):
        self.n_folds = n_folds
        self.cv_state_file = cv_state_file
        self.state = self._load_state()
    
    def _load_state(self):
        """Load CV state from file if exists."""
        default_state = {
            'completed_folds': [],
            'current_fold': 1,
            'fold_results': [],
            'interrupted': False,
            'total_folds': self.n_folds
        }
        
        if os.path.exists(self.cv_state_file):
            try:
                with open(self.cv_state_file, 'r') as f:
                    state = json.load(f)
                    # Validate state structure
                    for key in default_state:
                        if key not in state:
                            state[key] = default_state[key]
                    return state
            except Exception as e:
                logger.warning(f"Failed to load CV state: {e}. Starting fresh.")
                return default_state
        
        return default_state
    
    def save_state(self):
        """Save current CV state to file."""
        try:
            with open(self.cv_state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save CV state: {e}")
    
    def mark_fold_started(self, fold_num):
        """Mark that a fold has started training."""
        self.state['current_fold'] = fold_num
        self.state['interrupted'] = True
        self.save_state()
    
    def mark_fold_completed(self, fold_num, fold_result):
        """Mark that a fold has completed training."""
        if fold_num not in self.state['completed_folds']:
            self.state['completed_folds'].append(fold_num)
        
        # Update or add fold result
        updated = False
        for i, result in enumerate(self.state['fold_results']):
            if result['fold'] == fold_num:
                self.state['fold_results'][i] = fold_result
                updated = True
                break
        
        if not updated:
            self.state['fold_results'].append(fold_result)
        
        # Check if all folds are completed
        if len(self.state['completed_folds']) >= self.n_folds:
            self.state['interrupted'] = False
        
        self.save_state()
    
    def get_next_fold(self):
        """Get the next fold that needs to be trained."""
        for fold_num in range(1, self.n_folds + 1):
            if fold_num not in self.state['completed_folds']:
                return fold_num
        return None
    
    def is_completed(self):
        """Check if all folds are completed."""
        return len(self.state['completed_folds']) >= self.n_folds
    
    def reset(self):
        """Reset CV state (start fresh)."""
        self.state = {
            'completed_folds': [],
            'current_fold': 1,
            'fold_results': [],
            'interrupted': False,
            'total_folds': self.n_folds
        }
        self.save_state()

def save_fold_checkpoint(fold_num, epoch, model, optimizer, lr_scheduler, training_state, config):
    """
    Save comprehensive fold checkpoint for resumable training.
    """
    checkpoint_path = f"fold_{fold_num}_checkpoint.pth"
    
    checkpoint = {
        'fold': fold_num,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_epoch': lr_scheduler.current_epoch,
        'training_state': training_state,
        'config': {
            'LEARNING_RATE': config.LEARNING_RATE,
            'BATCH_SIZE': config.BATCH_SIZE,
            'MAX_EPOCHS': config.MAX_EPOCHS,
            'PATIENCE': config.PATIENCE,
            'WARMUP_EPOCHS': config.WARMUP_EPOCHS
        }
    }
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Fold {fold_num} checkpoint saved at epoch {epoch}")
        return True
    except Exception as e:
        logger.error(f"Failed to save fold {fold_num} checkpoint: {e}")
        return False

def load_fold_checkpoint(fold_num, model, optimizer, lr_scheduler, config):
    """
    Load fold checkpoint and restore training state.
    """
    checkpoint_path = f"fold_{fold_num}_checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found for fold {fold_num}. Starting fresh.")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        
        # Restore model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.current_epoch = checkpoint['scheduler_epoch']
        
        training_state = checkpoint['training_state']
        start_epoch = checkpoint['epoch']
        
        logger.info(f"Fold {fold_num} checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, training_state
        
    except Exception as e:
        logger.error(f"Failed to load fold {fold_num} checkpoint: {e}")
        return None

def train_fold(train_loader, val_loader, model, config, fold_num, logger, cv_manager):
    """
    Train a single fold with checkpoint resumption capability.
    """
    # Initialize components for this fold
    loss_function = StableLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    lr_scheduler = StableLRScheduler(
        optimizer=optimizer,
        initial_lr=config.LEARNING_RATE,
        warmup_epochs=config.WARMUP_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        min_lr=config.MIN_LEARNING_RATE,
        max_lr_multiplier=config.MAX_LR_MULTIPLIER
    )
    
    scaler = GradScaler()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Initialize training state
    training_state = {
        'best_metric': -1,
        'best_metric_epoch': -1,
        'epoch_loss_values': [],
        'metric_values': [],
        'accuracy_values': [],
        'patience_counter': 0,
        'consecutive_bad_epochs': 0,
        'lr_reduction_counter': 0
    }
    
    start_epoch = 0
    
    # Try to load checkpoint for this fold
    checkpoint_data = load_fold_checkpoint(fold_num, model, optimizer, lr_scheduler, config)
    if checkpoint_data:
        start_epoch, training_state = checkpoint_data
        logger.info(f"Resuming fold {fold_num} from epoch {start_epoch}")
    else:
        logger.info(f"Starting fold {fold_num} from epoch 0")
    
    # Mark fold as started
    cv_manager.mark_fold_started(fold_num)

    # Training loop for this fold
    for epoch in range(start_epoch, config.MAX_EPOCHS):
        current_lr = lr_scheduler.step()
        
        model.train()
        epoch_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold_num} - Epoch {epoch + 1}/{config.MAX_EPOCHS} [Training]", unit="batch")
        
        for batch_data in progress_bar:
            try:
                inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
                
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    logger.warning("NaN detected in batch data. Skipping batch.")
                    continue
                
                optimizer.zero_grad()

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

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Invalid loss detected. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                
                grad_ok, grad_norm = check_gradients(model, config.GRAD_CLIP_NORM)
                if not grad_ok:
                    logger.warning("Invalid gradients detected. Skipping batch.")
                    optimizer.zero_grad()
                    continue
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
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
        training_state['epoch_loss_values'].append(avg_epoch_loss)

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
                    
                    if torch.isnan(val_inputs).any() or torch.isnan(val_labels).any():
                        continue
                    
                    seg_outputs = model(val_inputs)
                    
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
                metric = training_state['metric_values'][-1] if training_state['metric_values'] else 0.0
                accuracy = training_state['accuracy_values'][-1] if training_state['accuracy_values'] else 0.0
            
            training_state['metric_values'].append(metric)
            training_state['accuracy_values'].append(accuracy)

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        logger.info(f"Fold {fold_num} - Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {metric:.4f} | Accuracy: {accuracy:.4f} | LR: {current_lr:.2e} | Best Dice: {training_state['best_metric']:.4f}")

        # Stability checks
        if avg_epoch_loss > config.LOSS_EXPLOSION_THRESHOLD:
            training_state['consecutive_bad_epochs'] += 1
            logger.warning(f"High loss detected ({avg_epoch_loss:.4f}). Bad epochs: {training_state['consecutive_bad_epochs']}")
            
            if training_state['consecutive_bad_epochs'] >= 3:
                logger.warning("Multiple consecutive high-loss epochs. Reducing learning rate.")
                current_lr = current_lr * 0.5
                lr_scheduler = StableLRScheduler(
                    optimizer=optimizer,
                    initial_lr=current_lr,
                    warmup_epochs=5,
                    max_epochs=config.MAX_EPOCHS,
                    min_lr=config.MIN_LEARNING_RATE,
                    max_lr_multiplier=config.MAX_LR_MULTIPLIER
                )
                training_state['consecutive_bad_epochs'] = 0
                training_state['lr_reduction_counter'] += 1
                
                if training_state['lr_reduction_counter'] >= 3:
                    logger.error("Learning rate reduced 3 times. Stopping training.")
                    break
        else:
            training_state['consecutive_bad_epochs'] = 0

        # Best model tracking
        if metric > training_state['best_metric']:
            training_state['best_metric'] = metric
            training_state['best_metric_epoch'] = epoch + 1
            training_state['patience_counter'] = 0
            # Save best model for this fold
            fold_best_path = f"fold_{fold_num}_best_Enhanced_AuraViT_model.pth"
            try:
                torch.save(model.state_dict(), fold_best_path)
                logger.info(f"New best model for fold {fold_num} saved! Best Dice: {training_state['best_metric']:.4f} at epoch {training_state['best_metric_epoch']}")
            except Exception as e:
                logger.error(f"Failed to save best model for fold {fold_num}: {e}")
        else:
            training_state['patience_counter'] += 1

        # Save checkpoint every 5 epochs and after each epoch with improvement
        if (epoch + 1) % 5 == 0 or training_state['patience_counter'] == 0:
            save_fold_checkpoint(fold_num, epoch + 1, model, optimizer, lr_scheduler, training_state, config)

        # Early stopping check
        if training_state['patience_counter'] >= config.PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch + 1} for fold {fold_num}. No improvement for {config.PATIENCE} epochs.")
            break

    logger.info(f"Fold {fold_num} training finished. Best Dice score of {training_state['best_metric']:.4f} at epoch {training_state['best_metric_epoch']}.")
    
    # Clean up checkpoint file after successful completion
    checkpoint_path = f"fold_{fold_num}_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Cleaned up checkpoint file for fold {fold_num}")
        except Exception as e:
            logger.warning(f"Failed to clean up checkpoint file for fold {fold_num}: {e}")
    
    # Return fold results
    fold_result = {
        'fold': fold_num,
        'best_metric': training_state['best_metric'],
        'best_epoch': training_state['best_metric_epoch'],
        'epoch_losses': training_state['epoch_loss_values'],
        'dice_scores': training_state['metric_values'],
        'accuracy_scores': training_state['accuracy_values'],
        'model_path': f"fold_{fold_num}_best_Enhanced_AuraViT_model.pth"
    }
    
    # Mark fold as completed in CV manager
    cv_manager.mark_fold_completed(fold_num, fold_result)
    
    return fold_result

def evaluate_fold(test_loader, model_path, config, fold_num, logger):
    """
    Evaluate a trained model on the test set for a specific fold.
    """
    model_config = {
        "image_size": 256, "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072,
        "num_heads": 12, "dropout_rate": 0.1, "block_dropout_rate": 0.05,
        "patch_size": 16, "num_channels": 1,
    }
    model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
    
    model = StableEnhancedAuraViT(model_config).to(config.DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        logger.info(f"Loaded model from {model_path} for fold {fold_num} evaluation")
    else:
        logger.warning(f"Model file {model_path} not found for fold {fold_num}")
        return None
    
    model.eval()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    cm_metric = ConfusionMatrixMetric(
        include_background=False, metric_name=["sensitivity", "specificity", "accuracy"], reduction="mean"
    )
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc=f"Evaluating Fold {fold_num}"):
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
                logger.warning(f"Error during fold {fold_num} evaluation: {e}. Skipping sample.")
                continue

    try:
        mean_dice_test = dice_metric.aggregate().item()
        cm_value = cm_metric.aggregate()
        sensitivity = cm_value[0].item()
        specificity = cm_value[1].item()
        accuracy = cm_value[2].item()
        iou_test = mean_dice_test / (2 - mean_dice_test) if mean_dice_test > 0 else 0
    except:
        logger.error(f"Failed to compute metrics for fold {fold_num}")
        return None
    
    return {
        'fold': fold_num,
        'dice': mean_dice_test,
        'iou': iou_test,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }

def main():
    try:
        # ======================================================================================
        # IMPROVED CONFIGURATION WITH CROSS-VALIDATION
        # ======================================================================================
        class StableTrainingConfig:
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            
            # Cross-validation configuration
            N_FOLDS = 4  # Number of folds for cross-validation
            RANDOM_SEED = 42  # Seed for reproducible splits
            
            # CONSERVATIVE HYPERPARAMETERS FOR STABILITY
            BATCH_SIZE = 4
            LEARNING_RATE = 2e-5
            MIN_LEARNING_RATE = 1e-6
            MAX_LR_MULTIPLIER = 1.0
            MAX_EPOCHS = 200  # Reduced for CV (will train N_FOLDS times)
            WARMUP_EPOCHS = 15
            PATIENCE = 20
            WEIGHT_DECAY = 1e-5
            GRAD_CLIP_NORM = 0.5
            
            # Stability checks
            CHECK_LOSS_FREQUENCY = 5
            LOSS_EXPLOSION_THRESHOLD = 10.0
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Enhanced model configuration with stability improvements
        model_config = {
            "image_size": 256, "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072,
            "num_heads": 12, "dropout_rate": 0.1, "block_dropout_rate": 0.05,
            "patch_size": 16, "num_channels": 1,
        }
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
        config = StableTrainingConfig()

        logger.info(f"Using device: {config.DEVICE}")
        logger.info(f"Starting {config.N_FOLDS}-fold cross-validation with resumable training")

        # Initialize CV state manager
        cv_manager = CVStateManager(config.N_FOLDS)
        
        # Check if we're resuming interrupted training
        if cv_manager.state['interrupted']:
            logger.info("Detected interrupted cross-validation training. Resuming...")
            logger.info(f"Completed folds: {cv_manager.state['completed_folds']}")
            logger.info(f"Current fold: {cv_manager.state['current_fold']}")
        else:
            logger.info("Starting fresh cross-validation training")

        # ======================================================================================
        # DATA PREPARATION FOR CROSS-VALIDATION
        # ======================================================================================
        image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
        mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
        data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

        logger.info(f"Total samples: {len(data_dicts)}")
        
        # First split: separate out test set (20% of total data)
        # Use same random seed to ensure consistent splits across runs
        train_val_files, test_files = train_test_split(
            data_dicts, test_size=0.2, random_state=config.RANDOM_SEED, shuffle=True
        )
        
        logger.info(f"Training+Validation samples: {len(train_val_files)}")
        logger.info(f"Testing samples: {len(test_files)}")

        # Initialize K-Fold cross-validation on the training+validation set
        # Use same random seed for consistent fold splits
        kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

        # Transforms
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
            RandAffined(
                keys=['image', 'label'], prob=0.3, translate_range=(5, 5),
                rotate_range=(np.pi / 36, np.pi / 36), scale_range=(0.05, 0.05),
                mode=('bilinear', 'nearest'),
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.005),
            RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.2),
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

        # ======================================================================================
        # RESUMABLE CROSS-VALIDATION TRAINING LOOP
        # ======================================================================================
        fold_histories = cv_manager.state['fold_results'].copy()  # Keep existing results
        
        logger.info("="*60)
        logger.info(f"CROSS-VALIDATION TRAINING ({config.N_FOLDS} FOLDS)")
        logger.info("="*60)

        # Create fold splits (consistent across runs due to same random seed)
        fold_splits = list(enumerate(kfold.split(train_val_files), 1))

        while not cv_manager.is_completed():
            next_fold = cv_manager.get_next_fold()
            if next_fold is None:
                logger.info("All folds completed!")
                break
            
            fold_num = next_fold
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING FOLD {fold_num}/{config.N_FOLDS}")
            logger.info(f"{'='*60}")
            
            # Setup fold-specific logging
            fold_logger = setup_logging(fold_num)
            
            # Get the correct train/val split for this fold
            train_idx, val_idx = fold_splits[fold_num - 1][1]  # -1 because fold_num is 1-indexed
            
            fold_train_files = [train_val_files[i] for i in train_idx]
            fold_val_files = [train_val_files[i] for i in val_idx]
            
            fold_logger.info(f"Fold {fold_num} - Training samples: {len(fold_train_files)}")
            fold_logger.info(f"Fold {fold_num} - Validation samples: {len(fold_val_files)}")
            
            # Create datasets and dataloaders for this fold
            fold_train_ds = Dataset(data=fold_train_files, transform=train_transforms)
            fold_val_ds = Dataset(data=fold_val_files, transform=val_transforms)
            fold_train_loader = DataLoader(fold_train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
            fold_val_loader = DataLoader(fold_val_ds, batch_size=1, shuffle=False, num_workers=2)
            
            # Initialize model for this fold
            fold_model = StableEnhancedAuraViT(model_config).to(config.DEVICE)
            
            # Train this fold
            try:
                fold_result = train_fold(
                    fold_train_loader, 
                    fold_val_loader, 
                    fold_model, 
                    config, 
                    fold_num, 
                    fold_logger,
                    cv_manager
                )
                
                # Update fold histories (replace if fold was rerun)
                updated = False
                for i, result in enumerate(fold_histories):
                    if result['fold'] == fold_num:
                        fold_histories[i] = fold_result
                        updated = True
                        break
                
                if not updated:
                    fold_histories.append(fold_result)
                
                fold_logger.info(f"Fold {fold_num} completed successfully!")
                fold_logger.info(f"Best Dice Score: {fold_result['best_metric']:.4f}")
                
            except Exception as e:
                fold_logger.error(f"Fold {fold_num} failed with error: {str(e)}")
                import traceback
                fold_logger.error(traceback.format_exc())
                
                # Don't mark as completed if failed
                logger.error(f"Fold {fold_num} training failed. Fix issues and restart to resume from this fold.")
                break
            
            # Clean up GPU memory after each fold
            del fold_model
            torch.cuda.empty_cache()

        # ======================================================================================
        # CROSS-VALIDATION RESULTS SUMMARY
        # ======================================================================================
        cv_results = cv_manager.state['fold_results']
        
        logger.info("\n" + "="*60)
        logger.info("CROSS-VALIDATION TRAINING RESULTS")
        logger.info("="*60)
        
        if cv_results and len(cv_results) == config.N_FOLDS:
            dice_scores = [result['best_metric'] for result in cv_results]
            mean_dice = np.mean(dice_scores)
            std_dice = np.std(dice_scores)
            
            logger.info(f"Individual fold results:")
            for result in cv_results:
                logger.info(f"  Fold {result['fold']}: Dice = {result['best_metric']:.4f} (Epoch {result['best_epoch']})")
            
            logger.info(f"\nCross-Validation Results:")
            logger.info(f"  Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
            logger.info(f"  Min Dice Score: {min(dice_scores):.4f}")
            logger.info(f"  Max Dice Score: {max(dice_scores):.4f}")
            
            # Save CV results
            cv_summary = {
                'mean_dice': mean_dice,
                'std_dice': std_dice,
                'min_dice': min(dice_scores),
                'max_dice': max(dice_scores),
                'individual_results': cv_results,
                'config': {
                    'n_folds': config.N_FOLDS,
                    'max_epochs': config.MAX_EPOCHS,
                    'batch_size': config.BATCH_SIZE,
                    'learning_rate': config.LEARNING_RATE
                }
            }
            
            with open('cv_results_summary.json', 'w') as f:
                json.dump(cv_summary, f, indent=2)
            
            logger.info("Cross-validation summary saved to cv_results_summary.json")
        else:
            if not cv_results:
                logger.error("No successful fold results to summarize!")
                return
            else:
                logger.warning(f"Only {len(cv_results)}/{config.N_FOLDS} folds completed. Partial results available.")
                cv_summary = {
                    'completed_folds': len(cv_results),
                    'total_folds': config.N_FOLDS,
                    'partial_results': cv_results
                }

        # ======================================================================================
        # TEST SET EVALUATION FOR ALL COMPLETED FOLDS
        # ======================================================================================
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ALL COMPLETED FOLDS ON TEST SET")
        logger.info("="*60)
        
        test_ds = Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
        
        test_results = []
        
        for result in cv_results:
            fold_num = result['fold']
            model_path = result['model_path']
            
            logger.info(f"Evaluating fold {fold_num} on test set...")
            fold_test_result = evaluate_fold(test_loader, model_path, config, fold_num, logger)
            
            if fold_test_result:
                test_results.append(fold_test_result)
                logger.info(f"Fold {fold_num} test results: Dice={fold_test_result['dice']:.4f}, IoU={fold_test_result['iou']:.4f}, Acc={fold_test_result['accuracy']:.4f}")

        # ======================================================================================
        # FINAL TEST RESULTS SUMMARY
        # ======================================================================================
        if test_results:
            logger.info("\n" + "="*60)
            logger.info("FINAL TEST SET RESULTS SUMMARY")
            logger.info("="*60)
            
            # Calculate statistics across all folds
            test_dice_scores = [result['dice'] for result in test_results]
            test_iou_scores = [result['iou'] for result in test_results]
            test_sensitivity_scores = [result['sensitivity'] for result in test_results]
            test_specificity_scores = [result['specificity'] for result in test_results]
            test_accuracy_scores = [result['accuracy'] for result in test_results]
            
            logger.info("Individual fold test results:")
            for result in test_results:
                logger.info(f"  Fold {result['fold']}: Dice={result['dice']:.4f}, IoU={result['iou']:.4f}, Sens={result['sensitivity']:.4f}, Spec={result['specificity']:.4f}, Acc={result['accuracy']:.4f}")
            
            logger.info(f"\nFinal Cross-Validation Test Results:")
            logger.info(f"  Mean Dice Score: {np.mean(test_dice_scores):.4f} ± {np.std(test_dice_scores):.4f}")
            logger.info(f"  Mean IoU Score: {np.mean(test_iou_scores):.4f} ± {np.std(test_iou_scores):.4f}")
            logger.info(f"  Mean Sensitivity: {np.mean(test_sensitivity_scores):.4f} ± {np.std(test_sensitivity_scores):.4f}")
            logger.info(f"  Mean Specificity: {np.mean(test_specificity_scores):.4f} ± {np.std(test_specificity_scores):.4f}")
            logger.info(f"  Mean Accuracy: {np.mean(test_accuracy_scores):.4f} ± {np.std(test_accuracy_scores):.4f}")
            
            # Save final test results
            final_results = {
                'test_statistics': {
                    'mean_dice': float(np.mean(test_dice_scores)),
                    'std_dice': float(np.std(test_dice_scores)),
                    'mean_iou': float(np.mean(test_iou_scores)),
                    'std_iou': float(np.std(test_iou_scores)),
                    'mean_sensitivity': float(np.mean(test_sensitivity_scores)),
                    'std_sensitivity': float(np.std(test_sensitivity_scores)),
                    'mean_specificity': float(np.mean(test_specificity_scores)),
                    'std_specificity': float(np.std(test_specificity_scores)),
                    'mean_accuracy': float(np.mean(test_accuracy_scores)),
                    'std_accuracy': float(np.std(test_accuracy_scores))
                },
                'individual_fold_results': test_results,
                'training_summary': cv_summary if 'cv_summary' in locals() else None
            }
            
            with open('final_cv_results.json', 'w') as f:
                json.dump(final_results, f, indent=2)
                
            logger.info("Final results saved to final_cv_results.json")

        # ======================================================================================
        # PLOTTING CROSS-VALIDATION RESULTS
        # ======================================================================================
        logger.info("Plotting cross-validation results...")
        
        if fold_histories:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot training losses for all folds
            axes[0, 0].set_title("Training Loss - All Folds")
            for i, fold_result in enumerate(fold_histories):
                axes[0, 0].plot(fold_result['epoch_losses'], label=f'Fold {fold_result["fold"]}', alpha=0.7)
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot validation dice scores for all folds
            axes[0, 1].set_title("Validation Dice Score - All Folds")
            for i, fold_result in enumerate(fold_histories):
                axes[0, 1].plot(fold_result['dice_scores'], label=f'Fold {fold_result["fold"]}', alpha=0.7)
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Dice Score")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot validation accuracy for all folds
            axes[1, 0].set_title("Validation Accuracy - All Folds")
            for i, fold_result in enumerate(fold_histories):
                axes[1, 0].plot(fold_result['accuracy_scores'], label=f'Fold {fold_result["fold"]}', alpha=0.7)
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot final test results comparison
            if test_results:
                fold_nums = [result['fold'] for result in test_results]
                test_dice = [result['dice'] for result in test_results]
                
                axes[1, 1].set_title("Final Test Dice Scores by Fold")
                bars = axes[1, 1].bar(fold_nums, test_dice, alpha=0.7)
                axes[1, 1].set_xlabel("Fold")
                axes[1, 1].set_ylabel("Test Dice Score")
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars, test_dice):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{score:.3f}', ha='center', va='bottom')
                
                # Add mean line
                mean_dice = np.mean(test_dice)
                axes[1, 1].axhline(y=mean_dice, color='red', linestyle='--', 
                                  label=f'Mean: {mean_dice:.3f}')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig("cv_training_results_AuraViT.png", dpi=300, bbox_inches='tight')
            plt.show()

        # Model summary
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Use one of the trained models for parameter counting
        if cv_results:
            sample_model = StableEnhancedAuraViT(model_config).to(config.DEVICE)
            total_params = count_parameters(sample_model)
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            logger.info(f"\nModel Statistics:")
            logger.info(f"Total trainable parameters: {total_params:,}")
            logger.info(f"Approximate model size: {model_size_mb:.2f} MB")
        
        # Clean up CV state file if all folds completed successfully
        if cv_manager.is_completed():
            logger.info("All folds completed successfully. Cleaning up CV state file.")
            cv_state_file = cv_manager.cv_state_file
            if os.path.exists(cv_state_file):
                try:
                    os.remove(cv_state_file)
                    logger.info("CV state file cleaned up.")
                except Exception as e:
                    logger.warning(f"Failed to clean up CV state file: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("RESUMABLE CROSS-VALIDATION TRAINING COMPLETED!")
        logger.info("="*60)
        
        # Final instructions for resumption
        logger.info("\nResumption Instructions:")
        logger.info("- If training is interrupted, simply restart the script")
        logger.info("- The script will automatically detect and resume from the last checkpoint")
        logger.info("- Individual fold checkpoints are saved every 5 epochs")
        logger.info("- CV state is tracked in 'cv_state.json'")
        logger.info("- To start completely fresh, delete 'cv_state.json' and all checkpoint files")

    except Exception as e:
        logger.error(f"Cross-validation training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Training can be resumed by restarting the script after fixing the issue")
        raise

if __name__ == "__main__":
    main()