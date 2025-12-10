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
import json
from datetime import datetime
import pandas as pd

# For reproducibility
set_determinism(seed=42)

# ======================================================================================
# LOGGING SETUP
# ======================================================================================
def setup_logging(model_size):
    """Setup logging with model-specific log file"""
    log_filename = f'lightweight_auravit_{model_size}_training.log'
    
    # Remove any existing handlers
    logger = logging.getLogger(f'AuraViT_{model_size}')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ======================================================================================
# LIGHTWEIGHT BUILDING BLOCKS
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
    """Lightweight version of AuraViT optimized for binary box mask segmentation"""
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # --- LIGHTWEIGHT ViT ENCODER ---
        self.patch_embed = nn.Sequential(
            nn.Linear(cf["patch_size"]*cf["patch_size"]*cf["num_channels"], cf["hidden_dim"]),
            nn.LayerNorm(cf["hidden_dim"]),
            nn.Dropout(cf["dropout_rate"])
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, cf["num_patches"], cf["hidden_dim"]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
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
            nn.LayerNorm(cf["hidden_dim"]) for _ in range(4)
        ])

        # --- LIGHTWEIGHT ASPP MODULE ---
        self.aspp = LightweightASPP(cf["hidden_dim"], cf["hidden_dim"], rates=[6, 12, 18])

        # --- LIGHTWEIGHT ATTENTION GATES ---
        self.att_gate_1 = LightweightAttentionGate(gate_channels=256, in_channels=256, inter_channels=128)
        self.att_gate_2 = LightweightAttentionGate(gate_channels=128, in_channels=128, inter_channels=64)
        self.att_gate_3 = LightweightAttentionGate(gate_channels=64, in_channels=64, inter_channels=32)
        self.att_gate_4 = LightweightAttentionGate(gate_channels=32, in_channels=32, inter_channels=16)

        # --- LIGHTWEIGHT SEGMENTATION DECODER ---
        dropout_rate = cf.get("block_dropout_rate", 0.1)
        
        self.seg_d1 = DeconvBlock(cf["hidden_dim"], 256)
        self.seg_s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256), 
            LightweightResBlock(256, 256, dropout_rate=dropout_rate)
        )
        self.seg_c1 = nn.Sequential(
            LightweightResBlock(256+256, 256, dropout_rate=dropout_rate), 
            LightweightResBlock(256, 256, dropout_rate=dropout_rate)
        )

        self.seg_d2 = DeconvBlock(256, 128)
        self.seg_s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128), 
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
            DeconvBlock(cf["hidden_dim"], 64), 
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
            LightweightResBlock(cf["num_channels"], 32, dropout_rate=dropout_rate), 
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
            
        # 1. ViT Encoder
        p = self.cf["patch_size"]
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf["num_patches"], -1)
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

        # Reshape feature maps
        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        # 2. ASPP
        aspp_out = self.aspp(z12_reshaped)

        # 3. Decoder with Attention Gates
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
            return torch.tensor(1000.0, requires_grad=True, device=seg_preds.device)
        
        if torch.isnan(seg_labels).any():
            raise ValueError("NaN detected in ground truth labels")
        
        seg_preds = torch.clamp(seg_preds, min=-10, max=10)
        loss = self.seg_loss(seg_preds, seg_labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
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
                return False, float('inf')
    
    total_norm = total_norm ** (1. / 2)
    return True, total_norm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ======================================================================================
# TRAINING FUNCTION
# ======================================================================================
def train_model(model_size, model_config, config, train_loader, val_loader, test_loader, logger):
    """Train a single model configuration"""
    
    logger.info("="*80)
    logger.info(f"TRAINING LIGHTWEIGHT AURAVIT - {model_size.upper()} MODEL")
    logger.info("="*80)
    logger.info(f"Image size: {model_config['image_size']}")
    logger.info(f"Layers: {model_config['num_layers']}")
    logger.info(f"Hidden dim: {model_config['hidden_dim']}")
    logger.info(f"Batch size: {config['BATCH_SIZE']}")
    
    # Setup paths for this configuration
    checkpoint_path = f"checkpoint_AuraViT_{model_size}.pth"
    best_model_path = f"best_AuraViT_{model_size}.pth"
    
    # Initialize model
    model = LightweightAuraViT(model_config).to(config['DEVICE'])
    
    total_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024 * 1024)
    logger.info(f"Total trainable parameters: {total_params:,}")
    logger.info(f"Approximate model size: {model_size_mb:.2f} MB")
    
    # Setup training components
    loss_function = StableLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['LEARNING_RATE'], 
        weight_decay=config['WEIGHT_DECAY'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    lr_scheduler = StableLRScheduler(
        optimizer=optimizer,
        initial_lr=config['LEARNING_RATE'],
        warmup_epochs=config['WARMUP_EPOCHS'],
        max_epochs=config['MAX_EPOCHS'],
        min_lr=config['MIN_LEARNING_RATE'],
        max_lr_multiplier=config['MAX_LR_MULTIPLIER']
    )
    
    scaler = GradScaler()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Training state
    start_epoch = 0
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    accuracy_values = []
    patience_counter = 0
    lr_reduction_counter = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=config['DEVICE'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_metric = checkpoint.get('best_metric', best_metric)
            epoch_loss_values = checkpoint.get('epoch_loss_values', epoch_loss_values)
            metric_values = checkpoint.get('metric_values', metric_values)
            accuracy_values = checkpoint.get('accuracy_values', accuracy_values)
            patience_counter = checkpoint.get('patience_counter', patience_counter)
            lr_reduction_counter = checkpoint.get('lr_reduction_counter', 0)
            
            if 'scheduler_epoch' in checkpoint:
                lr_scheduler.current_epoch = checkpoint['scheduler_epoch']
            
            logger.info(f"Checkpoint found! Resuming from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    # Training loop
    logger.info("Starting Training...")
    consecutive_bad_epochs = 0
    
    for epoch in range(start_epoch, config['MAX_EPOCHS']):
        current_lr = lr_scheduler.step()
        
        model.train()
        epoch_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['MAX_EPOCHS']} [Training]", unit="batch")
        
        for batch_data in progress_bar:
            try:
                inputs, labels = batch_data["image"].to(config['DEVICE']), batch_data["label"].to(config['DEVICE'])
                
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
                            logger.warning(f"NaN detected during forward pass. Skipping batch.")
                            continue
                        else:
                            raise

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Invalid loss detected. Skipping batch.")
                    continue

                scaler.scale(loss).backward()
                
                grad_ok, grad_norm = check_gradients(model, config['GRAD_CLIP_NORM'])
                if not grad_ok:
                    logger.warning("Invalid gradients detected. Skipping batch.")
                    optimizer.zero_grad()
                    continue
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['GRAD_CLIP_NORM'])
                
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
                    logger.error("CUDA OOM. Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Runtime error: {e}")
                    raise

        if batch_count == 0:
            logger.error("No valid batches processed!")
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
                    val_inputs, val_labels = val_data["image"].to(config['DEVICE']), val_data["label"].to(config['DEVICE'])
                    
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
                    logger.warning(f"Error during validation: {e}")
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

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {metric:.4f} | Acc: {accuracy:.4f} | LR: {current_lr:.2e} | Best: {best_metric:.4f}")

        # Stability checks
        if avg_epoch_loss > config['LOSS_EXPLOSION_THRESHOLD']:
            consecutive_bad_epochs += 1
            logger.warning(f"High loss detected. Bad epochs: {consecutive_bad_epochs}")
            
            if consecutive_bad_epochs >= 3:
                logger.warning("Reducing learning rate.")
                current_lr = current_lr * 0.5
                lr_scheduler = StableLRScheduler(
                    optimizer=optimizer,
                    initial_lr=current_lr,
                    warmup_epochs=5,
                    max_epochs=config['MAX_EPOCHS'],
                    min_lr=config['MIN_LEARNING_RATE'],
                    max_lr_multiplier=config['MAX_LR_MULTIPLIER']
                )
                consecutive_bad_epochs = 0
                lr_reduction_counter += 1
                
                if lr_reduction_counter >= 3:
                    logger.error("LR reduced 3 times. Stopping.")
                    break
        else:
            consecutive_bad_epochs = 0

        # Model saving
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            patience_counter = 0
            try:
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved! Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")
        else:
            patience_counter += 1

        # Save checkpoint
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
            }, checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        # Early stopping
        if patience_counter >= config['PATIENCE']:
            logger.info(f"Early stopping at epoch {epoch + 1}. No improvement for {config['PATIENCE']} epochs.")
            break

    logger.info(f"Training finished. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    # Test evaluation
    logger.info("="*80)
    logger.info("RUNNING FINAL EVALUATION ON TEST SET")
    logger.info("="*80)
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    cm_metric = ConfusionMatrixMetric(
        include_background=False, metric_name=["sensitivity", "specificity", "accuracy"], reduction="mean"
    )

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Testing"):
            try:
                test_inputs, test_labels = test_data["image"].to(config['DEVICE']), test_data["label"].to(config['DEVICE'])
                
                if torch.isnan(test_inputs).any() or torch.isnan(test_labels).any():
                    continue
                
                test_seg_outputs = model(test_inputs)
                
                if torch.isnan(test_seg_outputs).any():
                    continue
                
                test_outputs_post = [post_pred(i) for i in decollate_batch(test_seg_outputs)]
                
                dice_metric(y_pred=test_outputs_post, y=test_labels)
                cm_metric(y_pred=test_outputs_post, y=test_labels)
            except Exception as e:
                logger.warning(f"Error during testing: {e}")
                continue

    try:
        mean_dice_test = dice_metric.aggregate().item()
        cm_value = cm_metric.aggregate()
        sensitivity = cm_value[0].item()
        specificity = cm_value[1].item()
        accuracy_test = cm_value[2].item()
        iou_test = mean_dice_test / (2 - mean_dice_test) if mean_dice_test > 0 else 0
    except:
        logger.error("Failed to compute final metrics")
        mean_dice_test = sensitivity = specificity = accuracy_test = iou_test = 0.0
    
    logger.info("="*80)
    logger.info(f"FINAL TEST METRICS - {model_size.upper()}")
    logger.info("="*80)
    logger.info(f"Mean Dice Score: {mean_dice_test:.4f}")
    logger.info(f"IoU: {iou_test:.4f}")
    logger.info(f"Sensitivity: {sensitivity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"Accuracy: {accuracy_test:.4f}")
    logger.info("="*80)
    
    # Save plots
    plt.figure("train", (18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Training Loss - {model_size.upper()}")
    plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, 'b-', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.title(f"Validation Dice Score - {model_size.upper()}")
    plt.plot(range(1, len(metric_values) + 1), metric_values, 'g-', label='Validation Dice')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.title(f"Validation Accuracy - {model_size.upper()}")
    plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, 'r-', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_curves_{model_size}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return results
    results = {
        'model_size': model_size,
        'parameters': total_params,
        'model_size_mb': model_size_mb,
        'best_dice': best_metric,
        'best_epoch': best_metric_epoch,
        'test_dice': mean_dice_test,
        'test_iou': iou_test,
        'test_sensitivity': sensitivity,
        'test_specificity': specificity,
        'test_accuracy': accuracy_test,
        'training_epochs': len(epoch_loss_values),
        'final_train_loss': epoch_loss_values[-1] if epoch_loss_values else 0,
        'epoch_loss_values': epoch_loss_values,
        'metric_values': metric_values,
        'accuracy_values': accuracy_values
    }
    
    return results

# ======================================================================================
# MAIN FUNCTION
# ======================================================================================
def main():
    try:
        # Configuration
        class Config:
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            
            # Training hyperparameters
            BATCH_SIZE = 8
            LEARNING_RATE = 3e-4
            MIN_LEARNING_RATE = 1e-6
            MAX_LR_MULTIPLIER = 1.0
            MAX_EPOCHS = 200
            WARMUP_EPOCHS = 10
            PATIENCE = 20
            WEIGHT_DECAY = 1e-5
            GRAD_CLIP_NORM = 0.5
            
            CHECK_LOSS_FREQUENCY = 5
            LOSS_EXPLOSION_THRESHOLD = 10.0
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_dict = {
            'BATCH_SIZE': Config.BATCH_SIZE,
            'LEARNING_RATE': Config.LEARNING_RATE,
            'MIN_LEARNING_RATE': Config.MIN_LEARNING_RATE,
            'MAX_LR_MULTIPLIER': Config.MAX_LR_MULTIPLIER,
            'MAX_EPOCHS': Config.MAX_EPOCHS,
            'WARMUP_EPOCHS': Config.WARMUP_EPOCHS,
            'PATIENCE': Config.PATIENCE,
            'WEIGHT_DECAY': Config.WEIGHT_DECAY,
            'GRAD_CLIP_NORM': Config.GRAD_CLIP_NORM,
            'LOSS_EXPLOSION_THRESHOLD': Config.LOSS_EXPLOSION_THRESHOLD,
            'DEVICE': Config.DEVICE
        }

        # Model configurations
        model_configs = {
            'small': {
                "image_size": 256, 
                "num_layers": 8,
                "hidden_dim": 512,
                "mlp_dim": 2048,
                "num_heads": 8,
                "dropout_rate": 0.1, 
                "block_dropout_rate": 0.05,
                "patch_size": 16, 
                "num_channels": 1,
            },
            'tiny': {
                "image_size": 256,
                "num_layers": 6,
                "hidden_dim": 384,
                "mlp_dim": 1536,
                "num_heads": 6,
                "dropout_rate": 0.1,
                "block_dropout_rate": 0.05,
                "patch_size": 16,
                "num_channels": 1,
            },
            'mobile': {
                "image_size": 224,
                "num_layers": 4,
                "hidden_dim": 256,
                "mlp_dim": 1024,
                "num_heads": 4,
                "dropout_rate": 0.1,
                "block_dropout_rate": 0.05,
                "patch_size": 16,
                "num_channels": 1,
            }
        }
        
        # Calculate num_patches for each config
        for config_name, model_config in model_configs.items():
            model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2

        print("="*80)
        print("MULTI-CONFIGURATION LIGHTWEIGHT AURAVIT TRAINING")
        print("Training on Binary Box Masks (Tumor White, Lung Black)")
        print("="*80)
        print(f"Using device: {config_dict['DEVICE']}")
        print(f"Configurations to train: {list(model_configs.keys())}")
        print("="*80)

        # Data preparation
        image_files = sorted(glob.glob(os.path.join(Config.IMAGE_DIR, "*.png")))
        mask_files = sorted(glob.glob(os.path.join(Config.MASK_DIR, "*.png")))
        
        if len(image_files) == 0 or len(mask_files) == 0:
            raise ValueError(f"No data found in {Config.IMAGE_DIR} or {Config.MASK_DIR}")
        
        data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

        train_files, test_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)

        print(f"Total samples: {len(data_dicts)}")
        print(f"Training samples: {len(train_files)}")
        print(f"Validation samples: {len(val_files)}")
        print(f"Testing samples: {len(test_files)}")
        print("="*80)

        # Store results for all configurations
        all_results = []
        
        # Train each configuration
        for model_size in ['small', 'tiny', 'mobile']:
            model_config = model_configs[model_size]
            
            # Setup logger for this configuration
            logger = setup_logging(model_size)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"STARTING TRAINING FOR {model_size.upper()} CONFIGURATION")
            logger.info(f"{'='*80}\n")
            
            # Create transforms specific to this configuration's image size
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

            # Create datasets
            train_ds = Dataset(data=train_files, transform=train_transforms)
            val_ds = Dataset(data=val_files, transform=val_transforms)
            test_ds = Dataset(data=test_files, transform=val_transforms)
            
            train_loader = DataLoader(train_ds, batch_size=config_dict['BATCH_SIZE'], shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
            
            # Train the model
            results = train_model(model_size, model_config, config_dict, train_loader, val_loader, test_loader, logger)
            all_results.append(results)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"COMPLETED TRAINING FOR {model_size.upper()} CONFIGURATION")
            logger.info(f"{'='*80}\n")
            
            # Clear GPU cache between configurations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ======================================================================================
        # COMPARE ALL RESULTS
        # ======================================================================================
        print("\n" + "="*80)
        print("FINAL COMPARISON OF ALL MODEL CONFIGURATIONS")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        # Display comparison table
        print("\nModel Performance Summary:")
        print("-"*80)
        print(f"{'Config':<10} {'Params':<15} {'Size(MB)':<12} {'Val Dice':<12} {'Test Dice':<12} {'Test IoU':<12}")
        print("-"*80)
        for result in all_results:
            print(f"{result['model_size']:<10} {result['parameters']:<15,} {result['model_size_mb']:<12.2f} "
                  f"{result['best_dice']:<12.4f} {result['test_dice']:<12.4f} {result['test_iou']:<12.4f}")
        print("-"*80)
        
        # Find best model
        best_model = max(all_results, key=lambda x: x['test_dice'])
        print(f"\n🏆 BEST MODEL: {best_model['model_size'].upper()}")
        print(f"   Test Dice Score: {best_model['test_dice']:.4f}")
        print(f"   Test IoU: {best_model['test_iou']:.4f}")
        print(f"   Parameters: {best_model['parameters']:,}")
        print(f"   Model Size: {best_model['model_size_mb']:.2f} MB")
        
        # Detailed metrics comparison
        print("\n" + "="*80)
        print("DETAILED METRICS COMPARISON")
        print("="*80)
        print(f"{'Config':<10} {'Sensitivity':<13} {'Specificity':<13} {'Accuracy':<12} {'Epochs':<10}")
        print("-"*80)
        for result in all_results:
            print(f"{result['model_size']:<10} {result['test_sensitivity']:<13.4f} "
                  f"{result['test_specificity']:<13.4f} {result['test_accuracy']:<12.4f} "
                  f"{result['training_epochs']:<10}")
        print("-"*80)
        
        # Save comparison to CSV
        comparison_csv = "model_comparison_results.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"\n✓ Comparison results saved to: {comparison_csv}")
        
        # Save detailed JSON report
        report = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': {
                'total_samples': len(data_dicts),
                'train_samples': len(train_files),
                'val_samples': len(val_files),
                'test_samples': len(test_files)
            },
            'training_config': {
                'batch_size': config_dict['BATCH_SIZE'],
                'learning_rate': config_dict['LEARNING_RATE'],
                'max_epochs': config_dict['MAX_EPOCHS'],
                'patience': config_dict['PATIENCE']
            },
            'results': all_results,
            'best_model': {
                'configuration': best_model['model_size'],
                'test_dice': best_model['test_dice'],
                'test_iou': best_model['test_iou'],
                'parameters': best_model['parameters'],
                'size_mb': best_model['model_size_mb']
            }
        }
        
        report_file = "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"✓ Detailed report saved to: {report_file}")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Configuration Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Parameters comparison
        ax = axes[0, 0]
        configs = [r['model_size'] for r in all_results]
        params = [r['parameters'] for r in all_results]
        bars = ax.bar(configs, params, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Parameters')
        ax.set_title('Model Size (Parameters)')
        ax.grid(axis='y', alpha=0.3)
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{param:,}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Model size in MB
        ax = axes[0, 1]
        sizes = [r['model_size_mb'] for r in all_results]
        bars = ax.bar(configs, sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Size (MB)')
        ax.set_title('Model Size (MB)')
        ax.grid(axis='y', alpha=0.3)
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{size:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Test Dice Score
        ax = axes[0, 2]
        test_dice = [r['test_dice'] for r in all_results]
        bars = ax.bar(configs, test_dice, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Dice Score')
        ax.set_title('Test Dice Score')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar, dice in zip(bars, test_dice):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dice:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Test IoU
        ax = axes[1, 0]
        test_iou = [r['test_iou'] for r in all_results]
        bars = ax.bar(configs, test_iou, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('IoU Score')
        ax.set_title('Test IoU')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar, iou in zip(bars, test_iou):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{iou:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 5: Test Accuracy
        ax = axes[1, 1]
        test_acc = [r['test_accuracy'] for r in all_results]
        bars = ax.bar(configs, test_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Test Accuracy')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, test_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 6: Training Epochs
        ax = axes[1, 2]
        epochs = [r['training_epochs'] for r in all_results]
        bars = ax.bar(configs, epochs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Epochs')
        ax.set_title('Training Epochs')
        ax.grid(axis='y', alpha=0.3)
        for bar, epoch in zip(bars, epochs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{epoch}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        comparison_plot_file = "model_comparison_plots.png"
        plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Comparison plots saved to: {comparison_plot_file}")
        
        # Create training curves comparison
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
        
        colors = {'small': '#1f77b4', 'tiny': '#ff7f0e', 'mobile': '#2ca02c'}
        
        # Loss curves
        ax = axes[0]
        for result in all_results:
            ax.plot(range(1, len(result['epoch_loss_values']) + 1), 
                   result['epoch_loss_values'], 
                   label=result['model_size'].upper(),
                   color=colors[result['model_size']])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation Dice curves
        ax = axes[1]
        for result in all_results:
            ax.plot(range(1, len(result['metric_values']) + 1), 
                   result['metric_values'], 
                   label=result['model_size'].upper(),
                   color=colors[result['model_size']])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.set_title('Validation Dice Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation Accuracy curves
        ax = axes[2]
        for result in all_results:
            ax.plot(range(1, len(result['accuracy_values']) + 1), 
                   result['accuracy_values'], 
                   label=result['model_size'].upper(),
                   color=colors[result['model_size']])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_comparison_file = "training_curves_comparison.png"
        plt.savefig(curves_comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training curves comparison saved to: {curves_comparison_file}")
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"✓ Trained {len(all_results)} model configurations")
        print(f"✓ Best model: {best_model['model_size'].upper()} (Test Dice: {best_model['test_dice']:.4f})")
        print(f"✓ All models saved with prefix: best_AuraViT_<config>.pth")
        print(f"✓ Logs saved: lightweight_auravit_<config>_training.log")
        print(f"✓ Individual plots: training_curves_<config>.png")
        print(f"✓ Comparison report: {report_file}")
        print(f"✓ Comparison CSV: {comparison_csv}")
        print(f"✓ Comparison plots: {comparison_plot_file}")
        print(f"✓ Training curves: {curves_comparison_file}")
        print("="*80)
        
        print("\n🎉 Multi-configuration training completed successfully!")
        print("\n📁 Output Files:")
        print(f"   - Best models: best_AuraViT_small.pth, best_AuraViT_tiny.pth, best_AuraViT_mobile.pth")
        print(f"   - Checkpoints: checkpoint_AuraViT_small.pth, checkpoint_AuraViT_tiny.pth, checkpoint_AuraViT_mobile.pth")
        print(f"   - Logs: lightweight_auravit_small_training.log, etc.")
        print(f"   - Individual plots: training_curves_small.png, etc.")
        print(f"   - Comparison files: {comparison_csv}, {report_file}, {comparison_plot_file}, {curves_comparison_file}")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()