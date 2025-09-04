"""
Enhanced AuraV3 with Federated Learning
Implements advanced federated learning with virtual client splitting,
FedProx, and Adaptive Averaging while maintaining stability features.
"""
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
from torch.utils.data import ConcatDataset
from monai.transforms import (
    AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, RandAffined,
    RandRotate90d, ResizeWithPadOrCropd, ScaleIntensityRanged, EnsureTyped,
    Activations, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandSpatialCropd, RandFlipd,
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils import set_determinism
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import math
from copy import deepcopy
from datetime import datetime
import json

# For reproducibility
set_determinism(seed=42)

# =====================================================================================
# LOGGING SETUP
# =====================================================================================
def setup_logging(log_file="federated_auravit_training.log"):
    """Setup comprehensive logging with both file and console output."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# =====================================================================================
# MODEL BLOCKS (Keep original architecture from AuraV3)
# =====================================================================================
class ResBlock(nn.Module):
    """Residual Block with improved stability."""
    def __init__(self, in_c, out_c, stride=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_c)
        )
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
        shortcut = self.shortcut(x)
        x = self.layers(x)
        x = x + shortcut
        return self.relu(x)

class DeconvBlock(nn.Module):
    """Transposed Convolution block for upsampling."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0)

    def forward(self, x):
        return self.deconv(x)

class AtrousConv(nn.Module):
    """Atrous Convolution block."""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.atrous_conv[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.atrous_conv(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
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

class AttentionGate(nn.Module):
    """Attention Gate with improved numerical stability."""
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

# =====================================================================================
# STABLE ENHANCED AURAVIT MODEL (Original from AuraV3)
# =====================================================================================
class StableEnhancedAuraViT(nn.Module):
    """Stability-Enhanced AuraViT - Keep original architecture."""
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # ViT Encoder
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
                d_model=cf["hidden_dim"], nhead=cf["num_heads"], dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"], activation=F.gelu, batch_first=True,
                norm_first=True
            ) for _ in range(cf["num_layers"])
        ])
        
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(cf["hidden_dim"]) for _ in range(4)
        ])

        # ASPP Module
        self.aspp = ASPP(cf["hidden_dim"], cf["hidden_dim"], rates=[6, 12, 18])

        # Attention Gates
        self.att_gate_1 = AttentionGate(gate_channels=512, in_channels=512, inter_channels=256)
        self.att_gate_2 = AttentionGate(gate_channels=256, in_channels=256, inter_channels=128)
        self.att_gate_3 = AttentionGate(gate_channels=128, in_channels=128, inter_channels=64)
        self.att_gate_4 = AttentionGate(gate_channels=64, in_channels=64, inter_channels=32)

        # Segmentation Decoder
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
        nn.init.xavier_uniform_(self.seg_output.weight, gain=0.1)
        nn.init.constant_(self.seg_output.bias, 0)

    def forward(self, inputs):
        if torch.isnan(inputs).any():
            raise ValueError("NaN detected in input tensor")
            
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

        # ASPP Module
        aspp_out = self.aspp(z12_reshaped)

        # Segmentation Decoder Path
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

# =====================================================================================
# STABLE LOSS FUNCTION
# =====================================================================================
class StableLoss(nn.Module):
    """Stable loss function with NaN detection."""
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

# =====================================================================================
# FEDERATED LEARNING COMPONENTS
# =====================================================================================
def client_update_fedprox(client_id, model, global_model, optimizer, train_loader, epochs, device, mu, scaler, loss_function, grad_clip_norm):
    """
    Enhanced FedProx client update with stability features and logging.
    """
    model.train()
    global_params = {name: param.clone() for name, param in global_model.named_parameters()}
    
    client_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Client {client_id} - Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        
        for batch_data in progress_bar:
            try:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                
                # Check for NaN in inputs
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    logger.warning(f"Client {client_id}: NaN detected in batch data. Skipping batch.")
                    continue
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(inputs)
                    primary_loss = loss_function(outputs, labels)
                    
                    # FedProx proximal term
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            proximal_term += torch.sum((param - global_params[name]) ** 2)
                    
                    total_loss = primary_loss + (mu / 2) * proximal_term
                
                # Check for invalid loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning(f"Client {client_id}: Invalid loss detected. Skipping batch.")
                    continue
                
                # Backward pass with gradient scaling
                scaler.scale(total_loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                progress_bar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "Prox": f"{(mu / 2) * proximal_term.item():.4f}",
                    "GradNorm": f"{grad_norm:.3f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"Client {client_id}: CUDA out of memory. Clearing cache.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            client_losses.append(avg_loss)
            logger.info(f"Client {client_id} - Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    return model, client_losses

def adaptive_weighted_average(w_states, s, global_model, local_models, alpha):
    """
    Enhanced adaptive weighted averaging with stability improvements.
    """
    total_samples = sum(s)
    w_avg = deepcopy(w_states[0])
    
    # Calculate similarity weights
    similarity_weights = []
    for local_model in local_models:
        similarity = 0.0
        param_count = 0
        
        for (name1, param1), (name2, param2) in zip(global_model.named_parameters(), local_model.named_parameters()):
            if param1.dtype.is_floating_point:
                # Ensure tensors are on the same device
                param1_flat = param1.detach().view(-1)
                param2_flat = param2.detach().view(-1)
                
                # Handle potential NaN values
                if not (torch.isnan(param1_flat).any() or torch.isnan(param2_flat).any()):
                    cos_sim = F.cosine_similarity(param1_flat, param2_flat, dim=0)
                    similarity += cos_sim.item()
                    param_count += 1
        
        avg_similarity = max(0.0, similarity / param_count) if param_count > 0 else 0.0
        similarity_weights.append(avg_similarity)
    
    # Normalize similarity weights
    sim_sum = sum(similarity_weights)
    if sim_sum > 0:
        similarity_weights = [weight / sim_sum for weight in similarity_weights]
    else:
        similarity_weights = [1.0 / len(w_states)] * len(w_states)
    
    # Combine data and similarity weights
    final_weights = []
    for i in range(len(w_states)):
        data_weight = s[i] / total_samples
        combined_weight = alpha * similarity_weights[i] + (1 - alpha) * data_weight
        final_weights.append(combined_weight)
    
    # Normalize final weights
    final_sum = sum(final_weights)
    normalized_final_weights = [fw / final_sum for fw in final_weights]
    
    # Weighted averaging of model parameters
    for key in w_avg.keys():
        if w_avg[key].dtype.is_floating_point:
            w_avg[key] = torch.zeros_like(w_avg[key])
            for i in range(len(w_states)):
                w_avg[key] += w_states[i][key] * normalized_final_weights[i]
    
    logger.info(f"Aggregation weights: {[f'{w:.3f}' for w in normalized_final_weights]}")
    
    return w_avg

def create_virtual_clients(data_dicts, num_clients=5, test_size=0.2, val_size=0.125, random_state=42):
    """
    Create virtual clients by splitting the dataset.
    Returns train_loaders, val_datasets, test_datasets, and sample sizes for each client.
    """
    logger.info(f"Creating {num_clients} virtual clients from {len(data_dicts)} samples...")
    
    # Shuffle and split data into virtual clients
    np.random.seed(random_state)
    np.random.shuffle(data_dicts)
    
    # Calculate sizes for uneven splits (to simulate heterogeneity)
    total_samples = len(data_dicts)
    
    if num_clients == 5:
        # Create imbalanced splits: 30%, 25%, 20%, 15%, 10%
        split_percentages = [0.30, 0.25, 0.20, 0.15, 0.10]
    else:
        # Equal splits for other numbers
        split_percentages = [1.0 / num_clients] * num_clients
    
    client_splits = []
    start_idx = 0
    
    for i, percentage in enumerate(split_percentages):
        if i == len(split_percentages) - 1:
            # Last client gets remaining samples
            client_data = data_dicts[start_idx:]
        else:
            end_idx = start_idx + int(total_samples * percentage)
            client_data = data_dicts[start_idx:end_idx]
            start_idx = end_idx
        
        client_splits.append(client_data)
    
    # Log client data distribution
    for i, split in enumerate(client_splits):
        logger.info(f"  Client {i+1}: {len(split)} samples ({len(split)/total_samples*100:.1f}%)")
    
    return client_splits

def save_federated_checkpoint(checkpoint_data, filepath):
    """Save federated learning checkpoint with error handling."""
    try:
        torch.save(checkpoint_data, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False

def load_federated_checkpoint(filepath, device):
    """Load federated learning checkpoint with error handling."""
    if os.path.exists(filepath):
        try:
            checkpoint = torch.load(filepath, map_location=device)
            logger.info(f"Checkpoint loaded from {filepath}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    return None

# =====================================================================================
# STABLE LEARNING RATE SCHEDULER
# =====================================================================================
class StableLRScheduler:
    """Stable learning rate scheduler for federated learning."""
    def __init__(self, optimizer, initial_lr, warmup_rounds, max_rounds, min_lr=1e-6, max_lr_multiplier=1.0):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_rounds = warmup_rounds
        self.max_rounds = max_rounds
        self.min_lr = min_lr
        self.max_lr = initial_lr * max_lr_multiplier
        self.current_round = 0
        
        # Start with lower LR for warmup
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr * 0.1

    def step(self):
        if self.current_round < self.warmup_rounds:
            # Linear warmup
            lr = self.initial_lr * (0.1 + 0.9 * self.current_round / self.warmup_rounds)
        else:
            # Cosine annealing
            progress = (self.current_round - self.warmup_rounds) / (self.max_rounds - self.warmup_rounds)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        lr = max(self.min_lr, min(lr, self.max_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_round += 1
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# =====================================================================================
# MAIN FEDERATED TRAINING FUNCTION
# =====================================================================================
def main():
    try:
        # =====================================================================================
        # CONFIGURATION
        # =====================================================================================
        class FederatedConfig:
            # Data paths
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            
            # Additional data source (if available)
            ADDITIONAL_DATA_DIR = "/teamspace/studios/this_studio/lung/processed_task6_lung"
            USE_ADDITIONAL_DATA = os.path.exists(ADDITIONAL_DATA_DIR)
            
            # Model paths
            CHECKPOINT_PATH = "federated_auravit_checkpoint.pth"
            BEST_MODEL_PATH = "best_federated_auravit_model.pth"
            LAST_MODEL_PATH = "last_federated_auravit_model.pth"
            
            # Federated learning parameters
            NUM_CLIENTS = 5
            COMM_ROUNDS = 100
            CLIENT_EPOCHS = [10, 8, 8, 6, 6]  # Different epochs for different clients
            FEDPROX_MU = 0.01  # FedProx regularization parameter
            ADAPTIVE_ALPHA = 0.3  # Weight for similarity-based aggregation
            
            # Training parameters
            BATCH_SIZE = 4
            LEARNING_RATE = 2e-5
            MIN_LEARNING_RATE = 1e-6
            MAX_LR_MULTIPLIER = 1.0
            WARMUP_ROUNDS = 10
            WEIGHT_DECAY = 1e-5
            GRAD_CLIP_NORM = 0.5
            
            # Early stopping and stability
            PATIENCE = 20
            CHECK_LOSS_FREQUENCY = 5
            LOSS_EXPLOSION_THRESHOLD = 10.0
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration
        model_config = {
            "image_size": 256,
            "num_layers": 12,
            "hidden_dim": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "dropout_rate": 0.1,
            "block_dropout_rate": 0.05,
            "patch_size": 16,
            "num_channels": 1,
        }
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
        
        config = FederatedConfig()
        
        logger.info("="*80)
        logger.info("FEDERATED LEARNING CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Number of clients: {config.NUM_CLIENTS}")
        logger.info(f"Communication rounds: {config.COMM_ROUNDS}")
        logger.info(f"FedProx mu: {config.FEDPROX_MU}")
        logger.info(f"Adaptive alpha: {config.ADAPTIVE_ALPHA}")
        logger.info(f"Learning rate: {config.LEARNING_RATE}")
        logger.info("="*80)

        # =====================================================================================
        # DATA PREPARATION
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION")
        logger.info("="*80)
        
        # Load primary dataset
        image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
        mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
        data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]
        
        logger.info(f"Primary dataset: {len(data_dicts)} samples from {config.DATA_DIR}")
        
        # Load additional dataset if available
        if config.USE_ADDITIONAL_DATA:
            add_image_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "image", "*.png")))
            add_mask_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "masks", "*.png")))
            add_data_dicts = [{"image": img, "label": mask} for img, mask in zip(add_image_files, add_mask_files)]
            logger.info(f"Additional dataset: {len(add_data_dicts)} samples from {config.ADDITIONAL_DATA_DIR}")
            
            # Combine datasets
            all_data_dicts = data_dicts + add_data_dicts
            logger.info(f"Total combined samples: {len(all_data_dicts)}")
        else:
            all_data_dicts = data_dicts
            logger.info(f"Total samples: {len(all_data_dicts)}")
        
        # Create virtual clients
        client_data_splits = create_virtual_clients(
            all_data_dicts,
            num_clients=config.NUM_CLIENTS,
            test_size=0.2,
            val_size=0.125,
            random_state=42
        )
        
        # Define transforms
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
                keys=['image', 'label'], prob=0.3,
                translate_range=(5, 5),
                rotate_range=(np.pi / 36, np.pi / 36),
                scale_range=(0.05, 0.05),
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
        
        # Create data loaders for each client
        client_train_loaders = []
        client_val_datasets = []
        client_test_datasets = []
        client_sample_sizes = []
        
        for i, client_data in enumerate(client_data_splits):
            # Split client data into train/val/test
            train_files, test_files = train_test_split(client_data, test_size=0.2, random_state=42)
            train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)
            
            # Create datasets
            train_ds = Dataset(data=train_files, transform=train_transforms)
            val_ds = Dataset(data=val_files, transform=val_transforms)
            test_ds = Dataset(data=test_files, transform=val_transforms)
            
            # Create data loader
            train_loader = DataLoader(
                train_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            client_train_loaders.append(train_loader)
            client_val_datasets.append(val_ds)
            client_test_datasets.append(test_ds)
            client_sample_sizes.append(len(train_files))
            
            logger.info(f"Client {i+1}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Create global validation and test sets
        global_val_ds = ConcatDataset(client_val_datasets)
        global_test_ds = ConcatDataset(client_test_datasets)
        global_val_loader = DataLoader(global_val_ds, batch_size=1, shuffle=False, num_workers=2)
        global_test_loader = DataLoader(global_test_ds, batch_size=1, shuffle=False, num_workers=2)
        
        logger.info(f"\nGlobal validation samples: {len(global_val_ds)}")
        logger.info(f"Global testing samples: {len(global_test_ds)}")

        # =====================================================================================
        # MODEL INITIALIZATION
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("MODEL INITIALIZATION")
        logger.info("="*80)
        
        global_model = StableEnhancedAuraViT(model_config).to(config.DEVICE)
        loss_function = StableLoss()
        
        # Count parameters
        total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")
        logger.info(f"Approximate model size: {total_params * 4 / (1024 * 1024):.2f} MB")
        
        # =====================================================================================
        # CHECKPOINT LOADING
        # =====================================================================================
        start_round = 0
        best_metric = -1
        best_metric_round = -1
        metric_values = []
        round_loss_values = []
        patience_counter = 0
        lr_reduction_counter = 0
        
        checkpoint = load_federated_checkpoint(config.CHECKPOINT_PATH, config.DEVICE)
        if checkpoint:
            global_model.load_state_dict(checkpoint['model_state_dict'])
            start_round = checkpoint.get('round', 0)
            best_metric = checkpoint.get('best_metric', -1)
            best_metric_round = checkpoint.get('best_metric_round', -1)
            metric_values = checkpoint.get('metric_values', [])
            round_loss_values = checkpoint.get('round_loss_values', [])
            patience_counter = checkpoint.get('patience_counter', 0)
            lr_reduction_counter = checkpoint.get('lr_reduction_counter', 0)
            logger.info(f"Resuming from round {start_round + 1}, Best metric: {best_metric:.4f}")
        else:
            logger.info("Starting training from scratch")
        
        # Initialize metrics
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
        # =====================================================================================
        # FEDERATED TRAINING LOOP
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("STARTING FEDERATED TRAINING")
        logger.info("="*80)
        
        for round_num in range(start_round, config.COMM_ROUNDS):
            round_start_time = datetime.now()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"COMMUNICATION ROUND {round_num + 1}/{config.COMM_ROUNDS}")
            logger.info(f"{'='*60}")
            
            local_models = []
            local_losses = []
            
            # Train each client
            for client_id, train_loader in enumerate(client_train_loaders):
                logger.info(f"\nTraining Client {client_id + 1}/{config.NUM_CLIENTS}")
                
                # Create local model copy
                local_model = deepcopy(global_model).to(config.DEVICE)
                
                # Adjust learning rate for different clients
                lr_multiplier = 1.2 if client_id == 0 else 1.0  # Give more weight to first client
                client_lr = config.LEARNING_RATE * lr_multiplier
                
                # Create optimizer for local model
                local_optimizer = torch.optim.AdamW(
                    local_model.parameters(),
                    lr=client_lr,
                    weight_decay=config.WEIGHT_DECAY,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
                
                # Create scaler for mixed precision
                scaler = GradScaler()
                
                # Get client-specific epochs
                client_epochs = config.CLIENT_EPOCHS[client_id]
                
                # Train local model with FedProx
                trained_model, client_loss_history = client_update_fedprox(
                    client_id=client_id + 1,
                    model=local_model,
                    global_model=global_model,
                    optimizer=local_optimizer,
                    train_loader=train_loader,
                    epochs=client_epochs,
                    device=config.DEVICE,
                    mu=config.FEDPROX_MU,
                    scaler=scaler,
                    loss_function=loss_function,
                    grad_clip_norm=config.GRAD_CLIP_NORM
                )
                
                local_models.append(trained_model)
                local_losses.extend(client_loss_history)
            
            # Aggregate local models
            logger.info(f"\nAggregating {len(local_models)} client models...")
            local_weights_states = [model.state_dict() for model in local_models]
            
            global_weights = adaptive_weighted_average(
                w_states=local_weights_states,
                s=client_sample_sizes,
                global_model=global_model,
                local_models=local_models,
                alpha=config.ADAPTIVE_ALPHA
            )
            
            # Update global model
            global_model.load_state_dict(global_weights)
            
            # Calculate round metrics
            avg_round_loss = np.mean(local_losses) if local_losses else float('inf')
            round_loss_values.append(avg_round_loss)
            
            # Global validation
            logger.info("\nEvaluating global model...")
            global_model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                dice_metric.reset()
                cm_metric = ConfusionMatrixMetric(
                    include_background=False,
                    metric_name=["accuracy"],
                    reduction="mean"
                )
                
                for val_data in tqdm(global_val_loader, desc="Validation", leave=False):
                    val_inputs = val_data["image"].to(config.DEVICE)
                    val_labels = val_data["label"].to(config.DEVICE)
                    
                    val_outputs = global_model(val_inputs)
                    batch_loss = loss_function(val_outputs, val_labels)
                    
                    if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                        val_loss += batch_loss.item()
                        val_batch_count += 1
                    
                    val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs_post, y=val_labels)
                    cm_metric(y_pred=val_outputs_post, y=val_labels)
                
                metric = dice_metric.aggregate().item()
                accuracy = cm_metric.aggregate()[0].item()
                avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            
            metric_values.append(metric)
            
            # Round summary
            round_time = (datetime.now() - round_start_time).total_seconds()
            logger.info(f"\n{'='*40}")
            logger.info(f"Round {round_num + 1} Summary:")
            logger.info(f"  Training Loss: {avg_round_loss:.4f}")
            logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"  Validation Dice: {metric:.4f}")
            logger.info(f"  Validation Accuracy: {accuracy:.4f}")
            logger.info(f"  Best Dice: {best_metric:.4f} (Round {best_metric_round})")
            logger.info(f"  Round Time: {round_time:.1f}s")
            logger.info(f"{'='*40}")
            
            # Save best model
            if metric > best_metric:
                best_metric = metric
                best_metric_round = round_num + 1
                patience_counter = 0
                
                torch.save(global_model.state_dict(), config.BEST_MODEL_PATH)
                logger.info(f"🏆 New best model saved! Dice: {best_metric:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            
            # Save checkpoint
            checkpoint_data = {
                'round': round_num + 1,
                'model_state_dict': global_model.state_dict(),
                'best_metric': best_metric,
                'best_metric_round': best_metric_round,
                'metric_values': metric_values,
                'round_loss_values': round_loss_values,
                'patience_counter': patience_counter,
                'lr_reduction_counter': lr_reduction_counter,
                'client_sample_sizes': client_sample_sizes
            }
            save_federated_checkpoint(checkpoint_data, config.CHECKPOINT_PATH)
            
            # Save last model
            torch.save(global_model.state_dict(), config.LAST_MODEL_PATH)
            
            # Early stopping
            if patience_counter >= config.PATIENCE:
                logger.info(f"\n⚠️ Early stopping triggered after {config.PATIENCE} rounds without improvement")
                break
            
            # Check for training instability
            if avg_round_loss > config.LOSS_EXPLOSION_THRESHOLD:
                lr_reduction_counter += 1
                logger.warning(f"High loss detected! Reduction counter: {lr_reduction_counter}")
                
                if lr_reduction_counter >= 3:
                    logger.error("Training appears unstable. Stopping.")
                    break
        
        # =====================================================================================
        # TRAINING COMPLETE
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("FEDERATED TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best Dice Score: {best_metric:.4f} at round {best_metric_round}")
        
        # =====================================================================================
        # PLOTTING RESULTS
        # =====================================================================================
        logger.info("\nGenerating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training loss
        axes[0].plot(range(1, len(round_loss_values) + 1), round_loss_values, 'b-', linewidth=2)
        axes[0].set_xlabel('Communication Round')
        axes[0].set_ylabel('Average Training Loss')
        axes[0].set_title('Federated Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Plot validation dice
        axes[1].plot(range(1, len(metric_values) + 1), metric_values, 'g-', linewidth=2)
        axes[1].axhline(y=best_metric, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_metric:.4f}')
        axes[1].set_xlabel('Communication Round')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Global Validation Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('federated_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # =====================================================================================
        # FINAL EVALUATION
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("="*80)
        
        # Load best model
        if os.path.exists(config.BEST_MODEL_PATH):
            global_model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
            logger.info("Loaded best model for final evaluation")
        
        global_model.eval()
        
        # Initialize metrics
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        cm_metric = ConfusionMatrixMetric(
            include_background=False,
            metric_name=["sensitivity", "specificity", "accuracy"],
            reduction="mean"
        )
        
        # Evaluate on test set
        with torch.no_grad():
            for test_data in tqdm(global_test_loader, desc="Testing"):
                test_inputs = test_data["image"].to(config.DEVICE)
                test_labels = test_data["label"].to(config.DEVICE)
                
                test_outputs = global_model(test_inputs)
                test_outputs_post = [post_pred(i) for i in decollate_batch(test_outputs)]
                
                dice_metric(y_pred=test_outputs_post, y=test_labels)
                cm_metric(y_pred=test_outputs_post, y=test_labels)
        
        # Compute final metrics
        mean_dice_test = dice_metric.aggregate().item()
        cm_value = cm_metric.aggregate()
        sensitivity = cm_value[0].item()
        specificity = cm_value[1].item()
        accuracy = cm_value[2].item()
        iou_test = mean_dice_test / (2 - mean_dice_test) if mean_dice_test > 0 else 0
        
        # Print final results
        logger.info("\n" + "="*60)
        logger.info("FINAL TEST METRICS (Federated Enhanced AuraViT)")
        logger.info("="*60)
        logger.info(f"📊 Mean Dice Score: {mean_dice_test:.4f}")
        logger.info(f"📊 Intersection over Union (IoU): {iou_test:.4f}")
        logger.info(f"📊 Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"📊 Specificity: {specificity:.4f}")
        logger.info(f"📊 Accuracy: {accuracy:.4f}")
        logger.info("="*60)
        
        # Save final metrics to file
        final_metrics = {
            "mean_dice": mean_dice_test,
            "iou": iou_test,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
            "best_validation_dice": best_metric,
            "best_round": best_metric_round,
            "total_rounds": round_num + 1
        }
        
        with open("federated_final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=4)
        
        logger.info("\n✅ Federated training completed successfully!")
        logger.info(f"📁 Results saved:")
        logger.info(f"   - Best model: {config.BEST_MODEL_PATH}")
        logger.info(f"   - Last model: {config.LAST_MODEL_PATH}")
        logger.info(f"   - Training curves: federated_training_curves.png")
        logger.info(f"   - Final metrics: federated_final_metrics.json")
        logger.info(f"   - Training log: federated_auravit_training.log")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()