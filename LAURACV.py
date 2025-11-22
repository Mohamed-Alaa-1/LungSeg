import argparse
import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
def setup_logging(fold=None):
    log_filename = f'lightweight_auravit_fold{fold}_training.log' if fold is not None else 'lightweight_auravit_training.log'
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

# ======================================================================================
# LIGHTWEIGHT BUILDING BLOCKS (same as before)
# ======================================================================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - reduces parameters by 70-80%"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class LightweightResBlock(nn.Module):
    """Lightweight Residual Block using depthwise separable convolutions"""
    def __init__(self, in_c, out_c, stride=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c, stride=stride),
            nn.Dropout2d(dropout_rate),
            DepthwiseSeparableConv(out_c, out_c),
        )
        
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
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
        return self.relu(self.layers(x) + self.shortcut(x))

class DeconvBlock(nn.Module):
    """Standard Transposed Convolution block for upsampling"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0)

    def forward(self, x):
        return self.deconv(x)

class LightweightAtrousConv(nn.Module):
    """Lightweight Atrous Convolution using depthwise separable convolutions"""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, 
                     dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.atrous_conv(x)

class LightweightASPP(nn.Module):
    """Lightweight ASPP module with reduced channels"""
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        intermediate_channels = out_channels // 2
        
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_blocks = nn.ModuleList([
            LightweightAtrousConv(in_channels, intermediate_channels, rate) for rate in rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(intermediate_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
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

class LightweightAttentionGate(nn.Module):
    """Lightweight Attention Gate with reduced channels"""
    def __init__(self, gate_channels, in_channels, inter_channels):
        super().__init__()
        inter_channels = inter_channels // 2
        
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, gate, x):
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = torch.clamp(psi, min=1e-8, max=1.0)
        return x * psi

# ======================================================================================
# LIGHTWEIGHT AURAVIT MODEL
# ======================================================================================

class LightweightAuraViT(nn.Module):
    """Lightweight version of AuraViT with reduced parameters"""
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # ViT ENCODER
        self.patch_embed = nn.Sequential(
            nn.Linear(cf['patch_size']*cf['patch_size']*cf['num_channels'], cf['hidden_dim']),
            nn.LayerNorm(cf['hidden_dim']),
            nn.Dropout(cf['dropout_rate'])
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, cf['num_patches'], cf['hidden_dim']))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_dropout = nn.Dropout(cf['dropout_rate'])
        
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf['hidden_dim'], 
                nhead=cf['num_heads'], 
                dim_feedforward=cf['mlp_dim'],
                dropout=cf['dropout_rate'], 
                activation='gelu', 
                batch_first=True,
                norm_first=True
            ) for _ in range(cf['num_layers'])
        ])
        
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(cf['hidden_dim']) for _ in range(4)
        ])

        # ASPP MODULE
        self.aspp = LightweightASPP(cf['hidden_dim'], cf['hidden_dim'], rates=[6, 12, 18])

        # ATTENTION GATES
        self.att_gate_1 = LightweightAttentionGate(gate_channels=256, in_channels=256, inter_channels=128)
        self.att_gate_2 = LightweightAttentionGate(gate_channels=128, in_channels=128, inter_channels=64)
        self.att_gate_3 = LightweightAttentionGate(gate_channels=64, in_channels=64, inter_channels=32)
        self.att_gate_4 = LightweightAttentionGate(gate_channels=32, in_channels=32, inter_channels=16)

        # SEGMENTATION DECODER
        dropout_rate = cf.get('block_dropout_rate', 0.1)
        
        self.seg_d1 = DeconvBlock(cf['hidden_dim'], 256)
        self.seg_s1 = nn.Sequential(
            DeconvBlock(cf['hidden_dim'], 256), 
            LightweightResBlock(256, 256, dropout_rate=dropout_rate)
        )
        self.seg_c1 = nn.Sequential(
            LightweightResBlock(256+256, 256, dropout_rate=dropout_rate), 
            LightweightResBlock(256, 256, dropout_rate=dropout_rate)
        )

        self.seg_d2 = DeconvBlock(256, 128)
        self.seg_s2 = nn.Sequential(
            DeconvBlock(cf['hidden_dim'], 128), 
            LightweightResBlock(128, 128, dropout_rate=dropout_rate), 
            DeconvBlock(128, 128), 
            LightweightResBlock(128, 128, dropout_rate=dropout_rate)
        )
        self.seg_c2 = nn.Sequential(
            LightweightResBlock(128+128, 128, dropout_rate=dropout_rate), 
            LightweightResBlock(128, 128, dropout_rate=dropout_rate)
        )

        self.seg_d3 = DeconvBlock(128, 64)
        self.seg_s3 = nn.Sequential(
            DeconvBlock(cf['hidden_dim'], 64), 
            LightweightResBlock(64, 64, dropout_rate=dropout_rate), 
            DeconvBlock(64, 64),
            LightweightResBlock(64, 64, dropout_rate=dropout_rate), 
            DeconvBlock(64, 64), 
            LightweightResBlock(64, 64, dropout_rate=dropout_rate)
        )
        self.seg_c3 = nn.Sequential(
            LightweightResBlock(64+64, 64, dropout_rate=dropout_rate), 
            LightweightResBlock(64, 64, dropout_rate=dropout_rate)
        )

        self.seg_d4 = DeconvBlock(64, 32)
        self.seg_s4 = nn.Sequential(
            LightweightResBlock(cf['num_channels'], 32, dropout_rate=dropout_rate), 
            LightweightResBlock(32, 32, dropout_rate=dropout_rate)
        )
        self.seg_c4 = nn.Sequential(
            LightweightResBlock(32+32, 32, dropout_rate=dropout_rate), 
            LightweightResBlock(32, 32, dropout_rate=dropout_rate)
        )

        self.seg_output = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        
        nn.init.xavier_uniform_(self.seg_output.weight, gain=0.1)
        nn.init.constant_(self.seg_output.bias, 0)

    def forward(self, inputs):
        if torch.isnan(inputs).any():
            raise ValueError("NaN detected in input tensor")
            
        # ViT Encoder
        p = self.cf['patch_size']
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf['num_patches'], -1)
        patch_embed = self.patch_embed(patches)

        x = self.pos_dropout(patch_embed + self.pos_embed)

        num_layers = len(self.trans_encoder_layers)
        if num_layers == 8:
            skip_connection_index = [1, 3, 5, 7]
        elif num_layers == 6:
            skip_connection_index = [1, 2, 4, 5]
        elif num_layers == 4:
            skip_connection_index = [0, 1, 2, 3]
        else:
            skip_connection_index = [2, 5, 8, 11]
            
        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after transformer layer {i}")
            if i in skip_connection_index:
                norm_idx = len(skip_connections)
                normalized_skip = self.skip_norms[norm_idx](x)
                skip_connections.append(normalized_skip)
        
        z3, z6, z9, z12_features = skip_connections

        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        aspp_out = self.aspp(z12_reshaped)

        # Decoder
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
        
        if torch.isnan(seg_output).any():
            raise ValueError("NaN detected in model output")

        return seg_output

# ======================================================================================
# STABLE LOSS FUNCTION
# ======================================================================================
class StableLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)

    def forward(self, seg_preds, seg_labels):
        if torch.isnan(seg_preds).any():
            logger.warning("NaN detected in predictions! Returning large finite loss.")
            return torch.tensor(1000.0, requires_grad=True, device=seg_preds.device)
        
        if torch.isnan(seg_labels).any():
            raise ValueError("NaN detected in ground truth labels")
        
        seg_preds = torch.clamp(seg_preds, min=-10, max=10)
        loss = self.seg_loss(seg_preds, seg_labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or Inf loss detected! Returning fallback loss.")
            return torch.tensor(1000.0, requires_grad=True, device=seg_preds.device)
        
        return loss

# ======================================================================================
# LEARNING RATE SCHEDULER
# ======================================================================================
class StableLRScheduler:
    def __init__(self, optimizer, initial_lr, warmup_epochs, max_epochs, min_lr=1e-6, max_lr_multiplier=1.0):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = initial_lr * max_lr_multiplier
        self.current_epoch = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr * 0.1

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr * (0.1 + 0.9 * self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        lr = max(self.min_lr, min(lr, self.max_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

def check_gradients(model, max_grad_norm=1.0):
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if torch.isnan(param.grad).any():
                logger.error(f"NaN gradient detected in {name}")
                return False, float('inf')
    
    total_norm = total_norm ** (1. / 2)
    return True, total_norm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ======================================================================================
# TRAINING FUNCTION FOR ONE FOLD
# ======================================================================================
def train_one_fold(fold, train_files, val_files, test_files, model_config, config):
    """Train and evaluate model for one fold"""
    
    fold_logger = setup_logging(fold)
    
    fold_logger.info("=" * 60)
    fold_logger.info(f"STARTING FOLD {fold + 1}/{config.N_FOLDS}")
    fold_logger.info("=" * 60)
    fold_logger.info(f"Training samples: {len(train_files)}")
    fold_logger.info(f"Validation samples: {len(val_files)}")
    fold_logger.info(f"Testing samples: {len(test_files)}")
    
    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=(model_config['image_size'], model_config['image_size'])),
        RandRotate90d(keys=['image', 'label'], prob=0.3, max_k=3),
        RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=0.3),
        RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.3),
        RandAffined(
            keys=['image', 'label'], prob=0.3, translate_range=(5, 5),
            rotate_range=(np.pi / 36, np.pi / 36), scale_range=(0.05, 0.05),
            mode=('bilinear', 'nearest'),
        ),
        RandGaussianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.005),
        RandScaleIntensityd(keys=['image'], factors=0.05, prob=0.2),
        EnsureTyped(keys=['image', 'label'], track_meta=False),
    ])

    val_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=(model_config['image_size'], model_config['image_size'])),
        EnsureTyped(keys=['image', 'label'], track_meta=False),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Model setup
    model = LightweightAuraViT(model_config).to(config.DEVICE)
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

    # Training variables
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    accuracy_values = []
    patience_counter = 0
    consecutive_bad_epochs = 0
    lr_reduction_counter = 0

    # Training loop
    fold_logger.info(f"Starting training for Fold {fold + 1}")
    
    for epoch in range(config.MAX_EPOCHS):
        current_lr = lr_scheduler.step()
        
        model.train()
        epoch_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {epoch + 1}/{config.MAX_EPOCHS}", unit="batch")
        
        for batch_data in progress_bar:
            try:
                inputs, labels = batch_data['image'].to(config.DEVICE), batch_data['label'].to(config.DEVICE)
                
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    fold_logger.warning("NaN detected in batch data. Skipping batch.")
                    continue
                
                optimizer.zero_grad()

                with autocast():
                    try:
                        seg_outputs = model(inputs)
                        loss = loss_function(seg_outputs, labels)
                    except ValueError as e:
                        if "NaN" in str(e):
                            fold_logger.warning(f"NaN detected during forward pass. Skipping batch.")
                            continue
                        else:
                            raise

                if torch.isnan(loss) or torch.isinf(loss):
                    fold_logger.warning("Invalid loss detected. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                
                grad_ok, grad_norm = check_gradients(model, config.GRAD_CLIP_NORM)
                if not grad_ok:
                    fold_logger.warning("Invalid gradients detected. Skipping batch.")
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
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    fold_logger.error("CUDA OOM. Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    fold_logger.error(f"Runtime error: {e}")
                    raise

        if batch_count == 0:
            fold_logger.error("No valid batches processed!")
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
                    val_inputs, val_labels = val_data['image'].to(config.DEVICE), val_data['label'].to(config.DEVICE)
                    
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
                    fold_logger.warning(f"Error during validation: {e}")
                    continue
            
            try:
                metric = dice_metric.aggregate().item()
                accuracy = cm_metric.aggregate()[0].item()
            except:
                fold_logger.warning("Error computing metrics. Using previous values.")
                metric = metric_values[-1] if metric_values else 0.0
                accuracy = accuracy_values[-1] if accuracy_values else 0.0
            
            metric_values.append(metric)
            accuracy_values.append(accuracy)

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        fold_logger.info(f"Fold {fold+1} - Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {metric:.4f} | Acc: {accuracy:.4f} | LR: {current_lr:.2e}")

        # Stability checks
        if avg_epoch_loss > config.LOSS_EXPLOSION_THRESHOLD:
            consecutive_bad_epochs += 1
            fold_logger.warning(f"High loss detected. Bad epochs: {consecutive_bad_epochs}")
            
            if consecutive_bad_epochs >= 3:
                fold_logger.warning("Reducing learning rate.")
                current_lr = current_lr * 0.5
                lr_scheduler = StableLRScheduler(
                    optimizer=optimizer,
                    initial_lr=current_lr,
                    warmup_epochs=5,
                    max_epochs=config.MAX_EPOCHS,
                    min_lr=config.MIN_LEARNING_RATE,
                    max_lr_multiplier=config.MAX_LR_MULTIPLIER
                )
                consecutive_bad_epochs = 0
                lr_reduction_counter += 1
                
                if lr_reduction_counter >= 3:
                    fold_logger.error("LR reduced 3 times. Stopping.")
                    break
        else:
            consecutive_bad_epochs = 0

        # Model saving
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            patience_counter = 0
            try:
                torch.save(model.state_dict(), f"{config.BEST_MODEL_PATH}_fold{fold}.pth")
                fold_logger.info(f"New best model saved! Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
            except Exception as e:
                fold_logger.error(f"Failed to save best model: {e}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            fold_logger.info(f"Early stopping at epoch {epoch + 1}. No improvement for {config.PATIENCE} epochs.")
            break

    fold_logger.info(f"Fold {fold + 1} training finished. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    # Test on this fold's test set
    fold_logger.info(f"Evaluating Fold {fold + 1} on test set...")
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    if os.path.exists(f"{config.BEST_MODEL_PATH}_fold{fold}.pth"):
        model.load_state_dict(torch.load(f"{config.BEST_MODEL_PATH}_fold{fold}.pth"))
    model.eval()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    cm_metric = ConfusionMatrixMetric(
        include_background=False, metric_name=["sensitivity", "specificity", "accuracy"], reduction="mean"
    )

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc=f"Testing Fold {fold+1}"):
            try:
                test_inputs, test_labels = test_data['image'].to(config.DEVICE), test_data['label'].to(config.DEVICE)
                
                if torch.isnan(test_inputs).any() or torch.isnan(test_labels).any():
                    continue
                
                test_seg_outputs = model(test_inputs)
                
                if torch.isnan(test_seg_outputs).any():
                    continue
                
                test_outputs_post = [post_pred(i) for i in decollate_batch(test_seg_outputs)]
                
                dice_metric(y_pred=test_outputs_post, y=test_labels)
                cm_metric(y_pred=test_outputs_post, y=test_labels)
            except Exception as e:
                fold_logger.warning(f"Error during testing: {e}")
                continue

    try:
        mean_dice_test = dice_metric.aggregate().item()
        cm_value = cm_metric.aggregate()
        sensitivity = cm_value[0].item()
        specificity = cm_value[1].item()
        accuracy = cm_value[2].item()
        iou_test = mean_dice_test / (2 - mean_dice_test) if mean_dice_test > 0 else 0
    except:
        fold_logger.error("Failed to compute final metrics")
        mean_dice_test = sensitivity = specificity = accuracy = iou_test = 0.0
    
    fold_logger.info("=" * 60)
    fold_logger.info(f"FOLD {fold + 1} TEST METRICS")
    fold_logger.info("=" * 60)
    fold_logger.info(f"Dice Score: {mean_dice_test:.4f}")
    fold_logger.info(f"IoU: {iou_test:.4f}")
    fold_logger.info(f"Sensitivity: {sensitivity:.4f}")
    fold_logger.info(f"Specificity: {specificity:.4f}")
    fold_logger.info(f"Accuracy: {accuracy:.4f}")
    fold_logger.info("=" * 60)
    
    # Return results for this fold
    return {
        'fold': fold,
        'best_metric': best_metric,
        'best_epoch': best_metric_epoch,
        'test_dice': mean_dice_test,
        'test_iou': iou_test,
        'test_sensitivity': sensitivity,
        'test_specificity': specificity,
        'test_accuracy': accuracy,
        'epoch_loss_values': epoch_loss_values,
        'metric_values': metric_values,
        'accuracy_values': accuracy_values
    }

# ======================================================================================
# MAIN FUNCTION WITH 3-FOLD CROSS VALIDATION
# ======================================================================================
def main():
    try:
        # Configuration
        # parser = argparse.ArgumentParser(description='Lightweight AuraViT Training with Cross-Validation')
        # parser.add_argument('--MODEL_SIZE', type=str, default='small', choices=['small', 'tiny', 'mobile'],
        #                     help='Size of the model to train (small, tiny, or mobile)')
        # parser.add_argument('--DATA_DIR', type=str, default='teamspacestudiosthis_studiolungprocessed_task6_lung',
        #                     help='Directory containing the data')
        # args = parser.parse_args()

        class LightweightConfig:
            DATA_DIR = '/teamspace/studios/this_studio/lung/processed_task6_lung'
            IMAGE_DIR = os.path.join(DATA_DIR, 'images')
            MASK_DIR = os.path.join(DATA_DIR, 'masks')
            BEST_MODEL_PATH = 'lightweight_best_AuraViT_model'
            
            # Cross-validation settings
            N_FOLDS = 3  # Change this to control number of folds
            TRAIN_RATIO = 0.70  # 70% for training
            VAL_RATIO = 0.10    # 10% for validation
            TEST_RATIO = 0.20   # 20% for testing
            
            BATCH_SIZE = 8
            LEARNING_RATE = 3e-4
            MIN_LEARNING_RATE = 1e-6
            MAX_LR_MULTIPLIER = 1.0
            MAX_EPOCHS = 200
            WARMUP_EPOCHS = 13
            PATIENCE = 50
            WEIGHT_DECAY = 1e-5
            GRAD_CLIP_NORM = 0.5
            
            CHECK_LOSS_FREQUENCY = 5
            LOSS_EXPLOSION_THRESHOLD = 10.0
            
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        MODEL_SIZE = 'small'  # Change this to 'small', 'tiny', or 'mobile'
        
        configs = {
            'small': {
                'image_size': 256, 
                'num_layers': 8,
                'hidden_dim': 512,
                'mlp_dim': 2048,
                'num_heads': 8,
                'dropout_rate': 0.1, 
                'block_dropout_rate': 0.05,
                'patch_size': 16, 
                'num_channels': 1,
            },
            'tiny': {
                'image_size': 256,
                'num_layers': 6,
                'hidden_dim': 384,
                'mlp_dim': 1536,
                'num_heads': 6,
                'dropout_rate': 0.1,
                'block_dropout_rate': 0.05,
                'patch_size': 16,
                'num_channels': 1,
            },
            'mobile': {
                'image_size': 224,
                'num_layers': 4,
                'hidden_dim': 256,
                'mlp_dim': 1024,
                'num_heads': 4,
                'dropout_rate': 0.1,
                'block_dropout_rate': 0.05,
                'patch_size': 16,
                'num_channels': 1,
            }
        }
        
        model_config = configs[MODEL_SIZE]
        model_config['num_patches'] = (model_config['image_size'] // model_config['patch_size']) ** 2
        config = LightweightConfig()

        logger.info('=' * 60)
        logger.info(f"LIGHTWEIGHT AURAVIT - {config.N_FOLDS}-FOLD CROSS VALIDATION")
        logger.info('=' * 60)
        logger.info(f"Using device: {config.DEVICE}")
        logger.info(f"Model configuration: {MODEL_SIZE}")
        logger.info(f"Image size: {model_config['image_size']}")
        logger.info(f"Layers: {model_config['num_layers']}")
        logger.info(f"Hidden dim: {model_config['hidden_dim']}")
        logger.info(f"Batch size: {config.BATCH_SIZE}")
        logger.info(f"Number of folds: {config.N_FOLDS}")
        logger.info(f"Split ratios - Train: {config.TRAIN_RATIO:.0%}, Val: {config.VAL_RATIO:.0%}, Test: {config.TEST_RATIO:.0%}")

        # Load all data files
        image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, '*.png')))
        data_dicts = [{'image': img, 'label': mask} for img, mask in zip(image_files, mask_files)]

        # Group data by patient ID
        patient_data = {}
        for item in data_dicts:
            patient_id = os.path.basename(item['image']).split('_')[1]
            if patient_id not in patient_data:
                patient_data[patient_id] = []
            patient_data[patient_id].append(item)

        patient_ids = list(patient_data.keys())
        logger.info(f"Total patients: {len(patient_ids)}")
        logger.info(f"Total samples: {len(data_dicts)}")
        
        # Store results from all folds
        all_fold_results = []
        
        # Perform N-fold cross validation with custom split ratios
        for fold in range(config.N_FOLDS):
            logger.info('=' * 60)
            logger.info(f"STARTING FOLD {fold + 1}/{config.N_FOLDS}")
            logger.info('=' * 60)
            
            # Shuffle patient IDs for each fold
            np.random.seed(42 + fold)
            shuffled_patient_ids = np.random.permutation(patient_ids)
            
            # Calculate split sizes for patients
            n_total_patients = len(shuffled_patient_ids)
            n_train_patients = int(n_total_patients * config.TRAIN_RATIO)
            n_val_patients = int(n_total_patients * config.VAL_RATIO)
            
            # Split patient IDs
            train_patient_ids = shuffled_patient_ids[:n_train_patients]
            val_patient_ids = shuffled_patient_ids[n_train_patients : n_train_patients + n_val_patients]
            test_patient_ids = shuffled_patient_ids[n_train_patients + n_val_patients:]
            
            # Create data splits
            train_files = [item for pid in train_patient_ids for item in patient_data[pid]]
            val_files = [item for pid in val_patient_ids for item in patient_data[pid]]
            test_files = [item for pid in test_patient_ids for item in patient_data[pid]]
            
            logger.info(f"Fold {fold + 1} - Train patients: {len(train_patient_ids)}, Val patients: {len(val_patient_ids)}, Test patients: {len(test_patient_ids)}")
            logger.info(f"Fold {fold + 1} - Train samples: {len(train_files)}, Val samples: {len(val_files)}, Test samples: {len(test_files)}")
            
            # Train this fold
            fold_results = train_one_fold(fold, train_files, val_files, test_files, model_config, config)
            all_fold_results.append(fold_results)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # ======================================================================================
        # AGGREGATE RESULTS ACROSS FOLDS
        # ======================================================================================
        logger.info("=" * 60)
        logger.info(f"{config.N_FOLDS}-FOLD CROSS VALIDATION RESULTS")
        logger.info("=" * 60)
        
        # Calculate mean and std for each metric
        test_dice_scores = [r['test_dice'] for r in all_fold_results]
        test_iou_scores = [r['test_iou'] for r in all_fold_results]
        test_sensitivities = [r['test_sensitivity'] for r in all_fold_results]
        test_specificities = [r['test_specificity'] for r in all_fold_results]
        test_accuracies = [r['test_accuracy'] for r in all_fold_results]
        
        logger.info("\nPer-Fold Results:")
        for i, result in enumerate(all_fold_results):
            logger.info(f"\nFold {i+1}:")
            logger.info(f"  Best Val Dice: {result['best_metric']:.4f} (Epoch {result['best_epoch']})")
            logger.info(f"  Test Dice: {result['test_dice']:.4f}")
            logger.info(f"  Test IoU: {result['test_iou']:.4f}")
            logger.info(f"  Test Sensitivity: {result['test_sensitivity']:.4f}")
            logger.info(f"  Test Specificity: {result['test_specificity']:.4f}")
            logger.info(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"AVERAGE METRICS ACROSS {config.N_FOLDS} FOLDS")
        logger.info("=" * 60)
        logger.info(f"Mean Dice Score: {np.mean(test_dice_scores):.4f} ± {np.std(test_dice_scores):.4f}")
        logger.info(f"Mean IoU: {np.mean(test_iou_scores):.4f} ± {np.std(test_iou_scores):.4f}")
        logger.info(f"Mean Sensitivity: {np.mean(test_sensitivities):.4f} ± {np.std(test_sensitivities):.4f}")
        logger.info(f"Mean Specificity: {np.mean(test_specificities):.4f} ± {np.std(test_specificities):.4f}")
        logger.info(f"Mean Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
        logger.info("=" * 60)
        
        # ======================================================================================
        # PLOT TRAINING CURVES FOR ALL FOLDS
        # ======================================================================================
        logger.info("Plotting training curves for all folds...")
        
        n_folds = config.N_FOLDS
        fig, axes = plt.subplots(n_folds, 3, figsize=(20, 5 * n_folds))
        
        # Handle case where n_folds = 1
        if n_folds == 1:
            axes = axes.reshape(1, -1)
        
        for fold_idx, result in enumerate(all_fold_results):
            # Plot training loss
            axes[fold_idx, 0].plot(range(1, len(result['epoch_loss_values']) + 1), 
                                   result['epoch_loss_values'], 'b-', label=f'Fold {fold_idx+1}')
            axes[fold_idx, 0].set_title(f"Fold {fold_idx+1} - Training Loss")
            axes[fold_idx, 0].set_xlabel("Epoch")
            axes[fold_idx, 0].set_ylabel("Loss")
            axes[fold_idx, 0].legend()
            axes[fold_idx, 0].grid(True)
            
            # Plot validation Dice
            axes[fold_idx, 1].plot(range(1, len(result['metric_values']) + 1), 
                                   result['metric_values'], 'g-', label=f'Fold {fold_idx+1}')
            axes[fold_idx, 1].set_title(f"Fold {fold_idx+1} - Validation Dice")
            axes[fold_idx, 1].set_xlabel("Epoch")
            axes[fold_idx, 1].set_ylabel("Dice Score")
            axes[fold_idx, 1].legend()
            axes[fold_idx, 1].grid(True)
            
            # Plot validation accuracy
            axes[fold_idx, 2].plot(range(1, len(result['accuracy_values']) + 1), 
                                   result['accuracy_values'], 'r-', label=f'Fold {fold_idx+1}')
            axes[fold_idx, 2].set_title(f"Fold {fold_idx+1} - Validation Accuracy")
            axes[fold_idx, 2].set_xlabel("Epoch")
            axes[fold_idx, 2].set_ylabel("Accuracy")
            axes[fold_idx, 2].legend()
            axes[fold_idx, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"lightweight_{MODEL_SIZE}_{n_folds}fold_cv_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # ======================================================================================
        # PLOT COMPARISON ACROSS FOLDS
        # ======================================================================================
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Box plot for Dice scores
        axes[0].boxplot([test_dice_scores], labels=['Dice Score'])
        axes[0].scatter([1]*len(test_dice_scores), test_dice_scores, c='red', alpha=0.6)
        axes[0].set_ylabel('Score')
        axes[0].set_title(f'Dice Score Distribution Across {config.N_FOLDS} Folds')
        axes[0].grid(True, axis='y')
        
        # Bar plot for all metrics
        metrics_names = ['Dice', 'IoU', 'Sensitivity', 'Specificity', 'Accuracy']
        metrics_means = [
            np.mean(test_dice_scores),
            np.mean(test_iou_scores),
            np.mean(test_sensitivities),
            np.mean(test_specificities),
            np.mean(test_accuracies)
        ]
        metrics_stds = [
            np.std(test_dice_scores),
            np.std(test_iou_scores),
            np.std(test_sensitivities),
            np.std(test_specificities),
            np.std(test_accuracies)
        ]
        
        x_pos = np.arange(len(metrics_names))
        axes[1].bar(x_pos, metrics_means, yerr=metrics_stds, capsize=5, alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(metrics_names, rotation=45)
        axes[1].set_ylabel('Score')
        axes[1].set_title(f'Average Metrics Across {config.N_FOLDS} Folds')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"lightweight_{MODEL_SIZE}_{config.N_FOLDS}fold_cv_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"{config.N_FOLDS}-Fold Cross Validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
