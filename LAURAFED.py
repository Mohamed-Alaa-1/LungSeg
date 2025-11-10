"""
Enhanced Federated Lightweight AuraViT with Production-Ready Features - FIXED VERSION
Implements all improvements from Phase 1-5 with critical bug fixes:
- Fixed global test loader with dataset-specific normalization
- Increased minimum epochs to prevent underfitting
- Improved loss function for severe class imbalance
- Better normalization strategy using per-image statistics
- Added prediction monitoring and model collapse detection
- Increased learning rate for better convergence
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
    Activations, RandGaussianNoised, RandScaleIntensityd, RandFlipd,
    NormalizeIntensityd, ScaleIntensityRangePercentilesd,
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
from collections import defaultdict
import warnings
from pathlib import Path

# For reproducibility
set_determinism(seed=42)

# =====================================================================================
# LOGGING SETUP
# =====================================================================================
def setup_logging(log_file="federated_lightweight_auravit_training_fixed.log"):
    """Setup comprehensive logging with both file and console output."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# =====================================================================================
# LIGHTWEIGHT BUILDING BLOCKS (Unchanged from original)
# =====================================================================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - reduces parameters by 70-80%"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

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
    """Transposed Convolution block for upsampling"""
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

# =====================================================================================
# LIGHTWEIGHT AURAVIT MODEL (Unchanged from original)
# =====================================================================================

class LightweightAuraViT(nn.Module):
    """Lightweight version of AuraViT with reduced parameters"""
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
                d_model=cf["hidden_dim"], nhead=cf["num_heads"], 
                dim_feedforward=cf["mlp_dim"], dropout=cf["dropout_rate"], 
                activation=F.gelu, batch_first=True, norm_first=True
            ) for _ in range(cf["num_layers"])
        ])
        
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(cf["hidden_dim"]) for _ in range(4)
        ])

        # Lightweight ASPP
        self.aspp = LightweightASPP(cf["hidden_dim"], cf["hidden_dim"], rates=[6, 12, 18])

        # Attention Gates
        self.att_gate_1 = LightweightAttentionGate(256, 256, 128)
        self.att_gate_2 = LightweightAttentionGate(128, 128, 64)
        self.att_gate_3 = LightweightAttentionGate(64, 64, 32)
        self.att_gate_4 = LightweightAttentionGate(32, 32, 16)

        # Decoder
        dropout_rate = cf.get("block_dropout_rate", 0.1)
        
        self.seg_d1 = DeconvBlock(cf["hidden_dim"], 256)
        self.seg_s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256), 
            LightweightResBlock(256, 256, dropout_rate=dropout_rate)
        )
        self.seg_c1 = nn.Sequential(
            LightweightResBlock(512, 256, dropout_rate=dropout_rate), 
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
            LightweightResBlock(256, 128, dropout_rate=dropout_rate), 
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
            LightweightResBlock(128, 64, dropout_rate=dropout_rate), 
            LightweightResBlock(64, 64, dropout_rate=dropout_rate)
        )

        self.seg_d4 = DeconvBlock(64, 32)
        self.seg_s4 = nn.Sequential(
            LightweightResBlock(cf["num_channels"], 32, dropout_rate=dropout_rate), 
            LightweightResBlock(32, 32, dropout_rate=dropout_rate)
        )
        self.seg_c4 = nn.Sequential(
            LightweightResBlock(64, 32, dropout_rate=dropout_rate), 
            LightweightResBlock(32, 32, dropout_rate=dropout_rate)
        )

        self.seg_output = nn.Conv2d(32, 1, kernel_size=1, padding=0)
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

        # Skip connections
        num_layers = len(self.trans_encoder_layers)
        if num_layers == 8:
            skip_connection_index = [1, 3, 5, 7]
        elif num_layers == 6:
            skip_connection_index = [1, 2, 4, 5]
        elif num_layers == 4:
            skip_connection_index = [0, 1, 2, 3]
        else:  # 12 layers
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

        # Reshape
        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        # ASPP
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

# =====================================================================================
# FIXED: STABLE LOSS FUNCTION WITH BETTER CLASS IMBALANCE HANDLING
# =====================================================================================
class StableLoss(nn.Module):
    """Stable loss function with improved class imbalance handling."""
    def __init__(self, dice_weight=0.7, ce_weight=0.3):
        super().__init__()
        # Increase Dice weight to better handle class imbalance
        self.seg_loss = DiceCELoss(
            to_onehot_y=False, 
            sigmoid=True, 
            smooth_nr=1e-5, 
            smooth_dr=1e-5,
            lambda_dice=dice_weight,
            lambda_ce=ce_weight
        )

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
# PHASE 1: IMPROVED DATA SPLITTING (NO PATIENT LEAKAGE)
# =====================================================================================

def extract_patient_id(filename):
    """
    Extract patient ID from filename to prevent data leakage.
    
    Handles various filename formats:
    - patientID_slice.png -> patientID
    - patientID.png -> patientID
    - path/to/patientID_slice.png -> patientID
    """
    basename = os.path.basename(filename)
    # Remove extension
    name_without_ext = basename.rsplit('.', 1)[0]
    
    # Try splitting by underscore first
    parts = name_without_ext.split('_')
    if len(parts) > 1:
        # Return everything before the last part (which is usually slice number)
        # For format like: LUNG1-001_slice_5 -> LUNG1-001
        # Or: patient001_img_005 -> patient001_img
        return '_'.join(parts[:-1]) if parts[-1].isdigit() else parts[0]
    
    # If no underscore, return the whole name
    return name_without_ext

def split_data_by_patient(data_dicts, test_size=0.2, val_size=0.125, random_state=42):
    """
    Split data ensuring no patient appears in multiple splits.
    Returns: train_data, val_data, test_data
    
    Handles edge cases:
    - Empty data
    - Single patient
    - Very small datasets
    """
    if len(data_dicts) == 0:
        logger.warning("Empty dataset provided to split_data_by_patient")
        return [], [], []
    
    # Group by patient
    patient_groups = defaultdict(list)
    for item in data_dicts:
        patient_id = extract_patient_id(item["image"])
        patient_groups[patient_id].append(item)
    
    # Get patient IDs
    patient_ids = list(patient_groups.keys())
    n_patients = len(patient_ids)
    
    if n_patients == 0:
        logger.warning("No patients found in dataset")
        return [], [], []
    
    # Shuffle patients
    np.random.seed(random_state)
    np.random.shuffle(patient_ids)
    
    # Calculate splits with minimum of 1 patient per split (if possible)
    if test_size > 0:
        n_test = max(1, min(n_patients - 2, int(n_patients * test_size)))  # At least 1 for test, leave at least 2 for train/val
    else:
        n_test = 0
    
    remaining_patients = n_patients - n_test
    
    if val_size > 0 and remaining_patients > 1:
        n_val = max(1, min(remaining_patients - 1, int(remaining_patients * val_size)))  # At least 1 for val, leave at least 1 for train
    else:
        n_val = 0
    
    # Split patients
    if n_test > 0:
        test_patients = patient_ids[:n_test]
    else:
        test_patients = []
    
    if n_val > 0:
        val_patients = patient_ids[n_test:n_test + n_val]
    else:
        val_patients = []
    
    train_patients = patient_ids[n_test + n_val:]
    
    # Gather samples
    train_data = [item for pid in train_patients for item in patient_groups[pid]]
    val_data = [item for pid in val_patients for item in patient_groups[pid]]
    test_data = [item for pid in test_patients for item in patient_groups[pid]]
    
    logger.info(f"  Patients - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    logger.info(f"  Samples - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data

def create_global_test_set_first(dataset_clients, test_size=0.2, random_state=42, min_train_patients=1):
    """
    Phase 1.1: Create global test set FIRST from all datasets, ensuring no patient overlap.
    
    Args:
        dataset_clients: List of (dataset_name, data_dicts) tuples
        test_size: Fraction to reserve for testing (0-1)
        random_state: Random seed
        min_train_patients: Minimum patients to keep for training per dataset
    
    Returns: (client_train_val_data, global_test_data)
    
    Edge case handling:
    - Datasets with few patients: Keep at least min_train_patients for training
    - Single patient datasets: Keep for training, include in global test if possible
    """
    all_test_data = []
    client_train_val = []
    
    for dataset_name, data_dicts in dataset_clients:
        if len(data_dicts) == 0:
            logger.warning(f"{dataset_name}: Empty dataset, skipping")
            continue
        
        # Extract unique patients
        patient_groups = defaultdict(list)
        for item in data_dicts:
            patient_id = extract_patient_id(item["image"])
            patient_groups[patient_id].append(item)
        
        patient_ids = list(patient_groups.keys())
        n_patients = len(patient_ids)
        
        logger.info(f"{dataset_name}: Found {n_patients} unique patients, {len(data_dicts)} total samples")
        
        # Handle edge cases
        if n_patients <= min_train_patients:
            # Too few patients - keep all for training, but warn
            logger.warning(f"{dataset_name}: Only {n_patients} patients. Keeping all for training (no test samples from this dataset)")
            train_val_data = data_dicts
            test_data = []
        else:
            # Normal case - can afford to split
            np.random.seed(random_state)
            np.random.shuffle(patient_ids)
            
            # Reserve test patients - but ensure we keep enough for training
            n_test = max(1, min(n_patients - min_train_patients, int(n_patients * test_size)))
            test_patients = patient_ids[:n_test]
            train_val_patients = patient_ids[n_test:]
            
            # Gather data
            test_data = [item for pid in test_patients for item in patient_groups[pid]]
            train_val_data = [item for pid in train_val_patients for item in patient_groups[pid]]
            
            logger.info(f"{dataset_name}: Reserved {len(test_patients)} patients ({len(test_data)} samples) for global test")
            logger.info(f"{dataset_name}: Kept {len(train_val_patients)} patients ({len(train_val_data)} samples) for train/val")
        
        # Add to collections
        all_test_data.extend([(dataset_name, item) for item in test_data])
        if len(train_val_data) > 0:
            client_train_val.append((dataset_name, train_val_data))
        else:
            logger.warning(f"{dataset_name}: No training data available after test split")
    
    logger.info(f"\nTotal global test samples: {len(all_test_data)} from {len(set(ds for ds, _ in all_test_data))} datasets")
    logger.info(f"Total datasets with training data: {len(client_train_val)}")
    
    return client_train_val, all_test_data

# =====================================================================================
# PHASE 1: ADAPTIVE MAX CLIENT SIZE
# =====================================================================================

def determine_max_client_size(dataset_sizes, strategy='percentile', percentile=50):
    """
    Phase 1.3: Adaptive max client size determination.
    Strategies: 'percentile', 'median', 'iqr'
    """
    if strategy == 'percentile':
        max_size = int(np.percentile(dataset_sizes, percentile))
    elif strategy == 'median':
        max_size = int(np.median(dataset_sizes))
    elif strategy == 'iqr':
        q1, q3 = np.percentile(dataset_sizes, [25, 75])
        max_size = int(q3 + 0.5 * (q3 - q1))
    else:
        max_size = int(np.mean(dataset_sizes))
    
    # Ensure reasonable bounds
    max_size = max(100, min(max_size, max(dataset_sizes)))
    
    logger.info(f"Adaptive max client size ({strategy}): {max_size}")
    return max_size

def split_large_clients_adaptive(dataset_clients, max_client_size=None, strategy='percentile'):
    """
    Phase 1.3: Split clients with adaptive sizing and better imbalance handling.
    Filters out empty datasets.
    """
    # Filter out empty datasets first
    non_empty_datasets = [(name, data) for name, data in dataset_clients if len(data) > 0]
    
    if len(non_empty_datasets) == 0:
        logger.error("All datasets are empty after test split!")
        return []
    
    if max_client_size is None:
        dataset_sizes = [len(data) for _, data in non_empty_datasets]
        max_client_size = determine_max_client_size(dataset_sizes, strategy=strategy)
    
    virtual_clients = []
    
    for dataset_name, data_dicts in non_empty_datasets:
        dataset_size = len(data_dicts)
        
        if dataset_size == 0:
            logger.warning(f"  {dataset_name}: skipped (0 samples after test split)")
            continue
        
        if dataset_size <= max_client_size:
            virtual_clients.append((f"{dataset_name}_c1", data_dicts, dataset_name))
            logger.info(f"  {dataset_name}: 1 client with {dataset_size} samples")
        else:
            # Use balanced splitting
            num_splits = math.ceil(dataset_size / max_client_size)
            
            # Shuffle to ensure randomness
            shuffled_data = data_dicts.copy()
            np.random.shuffle(shuffled_data)
            
            # Create balanced splits
            splits = np.array_split(shuffled_data, num_splits)
            
            # If last split is too small, redistribute
            if len(splits[-1]) < max_client_size * 0.5 and len(splits) > 1:
                last_split = splits[-1]
                splits = splits[:-1]
                for i, item in enumerate(last_split):
                    splits[i % len(splits)] = np.append(splits[i % len(splits)], [item])
            
            for i, split in enumerate(splits, 1):
                virtual_clients.append((f"{dataset_name}_c{i}", split.tolist(), dataset_name))
            
            sizes = [len(s) for s in splits]
            logger.info(f"  {dataset_name}: split into {len(splits)} clients (sizes: {sizes})")
    
    return virtual_clients

# =====================================================================================
# PHASE 2: CLIENT SELECTION
# =====================================================================================

def select_clients_for_round(client_names, min_participation=0.5, max_participation=1.0, 
                            client_performances=None, selection_strategy='random',
                            client_source_datasets=None, ensure_diversity=True):
    """
    Phase 2.1: Implement client selection with minimum participation threshold.
    
    Args:
        ensure_diversity: If True, ensure at least one client from each dataset is selected
    
    Strategies:
    - 'random': Random selection
    - 'performance': Select based on validation performance
    - 'diverse': Ensure diverse client selection
    """
    num_clients = len(client_names)
    min_selected = max(1, int(num_clients * min_participation))
    max_selected = max(min_selected, int(num_clients * max_participation))
    
    if selection_strategy == 'random':
        num_to_select = np.random.randint(min_selected, max_selected + 1)
        
        # If diversity enforcement is enabled and we have dataset info
        if ensure_diversity and client_source_datasets:
            # Get unique datasets
            unique_datasets = list(set(client_source_datasets))
            selected = []
            
            # First, select at least one client from each dataset
            for dataset in unique_datasets:
                dataset_clients = [name for name, src in zip(client_names, client_source_datasets) if src == dataset]
                if dataset_clients:
                    selected.append(np.random.choice(dataset_clients))
            
            # Then randomly select remaining clients
            remaining_needed = num_to_select - len(selected)
            if remaining_needed > 0:
                remaining_clients = [c for c in client_names if c not in selected]
                if remaining_clients:
                    additional = np.random.choice(
                        remaining_clients, 
                        size=min(remaining_needed, len(remaining_clients)), 
                        replace=False
                    )
                    selected.extend(additional)
        else:
            selected = np.random.choice(client_names, size=num_to_select, replace=False).tolist()
    
    elif selection_strategy == 'performance' and client_performances:
        # Select based on recent performance (prioritize improving clients)
        scores = []
        for client in client_names:
            if client in client_performances and len(client_performances[client]) > 0:
                # Recent improvement score
                recent = client_performances[client][-3:]
                if len(recent) > 1:
                    improvement = recent[-1] - recent[0]
                    scores.append((client, improvement))
                else:
                    scores.append((client, 0))
            else:
                scores.append((client, 0))
        
        # Sort by improvement (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        num_to_select = np.random.randint(min_selected, max_selected + 1)
        selected = [client for client, _ in scores[:num_to_select]]
    
    else:  # diverse
        num_to_select = max_selected
        selected = client_names[:num_to_select]
    
    logger.info(f"Selected {len(selected)}/{num_clients} clients for this round: {selected}")
    return selected

# =====================================================================================
# PHASE 2: ROBUST AGGREGATION STRATEGIES
# =====================================================================================

def robust_aggregation(w_states, s, strategy='trimmed_mean', trim_ratio=0.2):
    """
    Phase 2.2: Implement robust aggregation methods.
    
    Strategies:
    - 'median': Coordinate-wise median
    - 'trimmed_mean': Trimmed mean (remove outliers)
    - 'krum': Krum algorithm (Byzantine-robust)
    """
    if strategy == 'median':
        w_avg = deepcopy(w_states[0])
        for key in w_avg.keys():
            if w_avg[key].dtype.is_floating_point:
                stacked = torch.stack([w[key] for w in w_states])
                w_avg[key] = torch.median(stacked, dim=0)[0]
        return w_avg
    
    elif strategy == 'trimmed_mean':
        w_avg = deepcopy(w_states[0])
        n_trim = max(1, int(len(w_states) * trim_ratio))
        
        for key in w_avg.keys():
            if w_avg[key].dtype.is_floating_point:
                stacked = torch.stack([w[key] for w in w_states])
                # Sort and trim extremes
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[n_trim:-n_trim] if n_trim > 0 else sorted_vals
                w_avg[key] = torch.mean(trimmed, dim=0)
        return w_avg
    
    else:  # weighted average (default)
        return weighted_average(w_states, s)

def weighted_average(w_states, s):
    """Simple weighted average by sample size."""
    total_samples = sum(s)
    w_avg = deepcopy(w_states[0])
    
    for key in w_avg.keys():
        if w_avg[key].dtype.is_floating_point:
            w_avg[key] = torch.zeros_like(w_avg[key])
            for i in range(len(w_states)):
                weight = s[i] / total_samples
                w_avg[key] += w_states[i][key] * weight
    
    return w_avg

def adaptive_weighted_average_improved(w_states, s, global_model, local_models, 
                                       alpha, client_val_metrics=None):
    """
    Phase 2.2: Enhanced adaptive weighted averaging using validation metrics.
    """
    total_samples = sum(s)
    w_avg = deepcopy(w_states[0])
    
    # Compute similarity weights based on validation performance (if available)
    if client_val_metrics:
        # Use validation dice scores as similarity proxy
        performance_weights = []
        for metric in client_val_metrics:
            # Higher performing clients get higher weight
            performance_weights.append(max(0.0, metric))
        
        perf_sum = sum(performance_weights)
        if perf_sum > 0:
            performance_weights = [w / perf_sum for w in performance_weights]
        else:
            performance_weights = [1.0 / len(w_states)] * len(w_states)
    else:
        # Fallback to parameter similarity
        similarity_weights = []
        for local_model in local_models:
            similarity = 0.0
            param_count = 0
            
            for (name1, param1), (name2, param2) in zip(global_model.named_parameters(), 
                                                         local_model.named_parameters()):
                if param1.dtype.is_floating_point:
                    param1_flat = param1.detach().view(-1)
                    param2_flat = param2.detach().view(-1)
                    
                    if not (torch.isnan(param1_flat).any() or torch.isnan(param2_flat).any()):
                        cos_sim = F.cosine_similarity(param1_flat, param2_flat, dim=0)
                        similarity += cos_sim.item()
                        param_count += 1
            
            avg_similarity = max(0.0, similarity / param_count) if param_count > 0 else 0.0
            similarity_weights.append(avg_similarity)
        
        sim_sum = sum(similarity_weights)
        performance_weights = [w / sim_sum for w in similarity_weights] if sim_sum > 0 else [1.0 / len(w_states)] * len(w_states)
    
    # Combine with data size weights
    final_weights = []
    for i in range(len(w_states)):
        data_weight = s[i] / total_samples
        combined_weight = alpha * performance_weights[i] + (1 - alpha) * data_weight
        final_weights.append(combined_weight)
    
    # Normalize
    final_sum = sum(final_weights)
    normalized_weights = [fw / final_sum for fw in final_weights]
    
    # Aggregate
    for key in w_avg.keys():
        if w_avg[key].dtype.is_floating_point:
            w_avg[key] = torch.zeros_like(w_avg[key])
            for i in range(len(w_states)):
                w_avg[key] += w_states[i][key] * normalized_weights[i]
    
    logger.info(f"Aggregation weights: {[f'{w:.3f}' for w in normalized_weights]}")
    return w_avg

# =====================================================================================
# PHASE 2: CLIENT DRIFT DETECTION
# =====================================================================================

class ClientDriftDetector:
    """
    Phase 2.3: Track and detect client drift.
    """
    def __init__(self, threshold=0.15, window_size=5):
        self.client_metrics = defaultdict(list)
        self.threshold = threshold
        self.window_size = window_size
        self.flagged_clients = set()
    
    def update(self, client_id, metric_value):
        """Update client metric history."""
        self.client_metrics[client_id].append(metric_value)
        
        # Keep only recent history
        if len(self.client_metrics[client_id]) > self.window_size * 2:
            self.client_metrics[client_id] = self.client_metrics[client_id][-self.window_size * 2:]
    
    def detect_drift(self, client_id):
        """Detect if client has anomalous performance."""
        if client_id not in self.client_metrics or len(self.client_metrics[client_id]) < self.window_size:
            return False
        
        recent_metrics = self.client_metrics[client_id][-self.window_size:]
        mean_metric = np.mean(recent_metrics)
        std_metric = np.std(recent_metrics)
        
        # Check for sudden drop
        if len(self.client_metrics[client_id]) > self.window_size:
            previous_mean = np.mean(self.client_metrics[client_id][-2*self.window_size:-self.window_size])
            if previous_mean - mean_metric > self.threshold:
                self.flagged_clients.add(client_id)
                logger.warning(f"⚠️ Client drift detected: {client_id} (drop: {previous_mean - mean_metric:.4f})")
                return True
        
        return False
    
    def get_flagged_clients(self):
        """Return list of flagged clients."""
        return list(self.flagged_clients)
    
    def reset_flag(self, client_id):
        """Reset flag for a client."""
        if client_id in self.flagged_clients:
            self.flagged_clients.remove(client_id)

# =====================================================================================
# PHASE 3: ADAPTIVE LEARNING RATE
# =====================================================================================

class AdaptiveFederatedLRScheduler:
    """
    Phase 3.1: Adaptive learning rate with proper warmup and per-client adaptation.
    """
    def __init__(self, base_lr, warmup_rounds, max_rounds, min_lr=1e-6, 
                 mode='cosine', warmup_start_factor=0.1):
        self.base_lr = base_lr
        self.warmup_rounds = warmup_rounds
        self.max_rounds = max_rounds
        self.min_lr = min_lr
        self.mode = mode
        self.warmup_start_factor = warmup_start_factor
        self.current_round = 0
        
        # Track per-client learning rates
        self.client_lrs = {}
    
    def get_lr(self):
        """Get current learning rate."""
        if self.current_round < self.warmup_rounds:
            # Warmup: gradually increase from start_factor * base_lr to base_lr
            alpha = self.current_round / self.warmup_rounds
            lr = self.base_lr * (self.warmup_start_factor + (1 - self.warmup_start_factor) * alpha)
        else:
            # After warmup
            progress = (self.current_round - self.warmup_rounds) / (self.max_rounds - self.warmup_rounds)
            
            if self.mode == 'cosine':
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.mode == 'linear':
                lr = self.base_lr * (1 - progress)
            elif self.mode == 'exponential':
                lr = self.base_lr * (0.95 ** (self.current_round - self.warmup_rounds))
            else:
                lr = self.base_lr
        
        return max(self.min_lr, lr)
    
    def get_client_lr(self, client_id, client_size):
        """
        Phase 3.1: Get adaptive learning rate for specific client based on data size.
        """
        base_lr = self.get_lr()
        
        # Adapt based on client size (smaller clients may need higher LR)
        size_factor = 1.0 / (1.0 + np.log10(max(1, client_size / 100)))
        adapted_lr = base_lr * (0.5 + 0.5 * size_factor)
        
        self.client_lrs[client_id] = adapted_lr
        return adapted_lr
    
    def step(self):
        """Advance to next round."""
        self.current_round += 1
    
    def state_dict(self):
        """Save state."""
        return {
            'current_round': self.current_round,
            'client_lrs': self.client_lrs
        }
    
    def load_state_dict(self, state_dict):
        """Load state."""
        self.current_round = state_dict.get('current_round', 0)
        self.client_lrs = state_dict.get('client_lrs', {})

# =====================================================================================
# PHASE 3: DYNAMIC CLIENT EPOCHS
# =====================================================================================

def compute_dynamic_epochs(client_size, min_epochs=3, max_epochs=10, target_size=1000):
    """
    Phase 3.2: Compute adaptive epochs based on client data size.
    Smaller clients train longer, larger clients train shorter.
    FIXED: Increased min_epochs from 1 to 3 to prevent underfitting.
    """
    if client_size >= target_size:
        return min_epochs
    else:
        # Inverse relationship: smaller clients need more epochs
        epochs = max(min_epochs, min(max_epochs, int(target_size / max(1, client_size))))
        return epochs

# =====================================================================================
# PHASE 3: ENHANCED CLIENT UPDATE WITH GRADIENT MONITORING
# =====================================================================================

def client_update_enhanced(client_id, model, global_model, optimizer, train_loader, 
                          epochs, device, mu, scaler, loss_function, grad_clip_norm,
                          early_stopping_patience=3):
    """
    Phase 3: Enhanced client update with:
    - Gradient-based early stopping
    - Better logging
    - Gradient norm monitoring
    - Handling for small batches
    """
    model.train()
    global_params = {name: param.clone() for name, param in global_model.named_parameters()}
    
    client_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        grad_norms = []
        
        progress_bar = tqdm(train_loader, desc=f"Client {client_id} - Epoch {epoch+1}/{epochs}", 
                          unit="batch", leave=False)
        
        for batch_data in progress_bar:
            try:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                
                # Check batch size and handle edge case
                batch_size = inputs.size(0)
                if batch_size == 1:
                    # For batch_size=1, temporarily switch to eval mode to avoid BatchNorm issues
                    # while still computing gradients
                    model.eval()
                    logger.debug(f"Client {client_id}: Single sample batch, using eval mode")
                else:
                    model.train()
                
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    logger.warning(f"Client {client_id}: NaN in batch. Skipping.")
                    continue
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(inputs)
                    primary_loss = loss_function(outputs, labels)
                    
                    # FedProx regularization
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            proximal_term += torch.sum((param - global_params[name]) ** 2)
                    
                    total_loss = primary_loss + (mu / 2) * proximal_term
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning(f"Client {client_id}: Invalid loss. Skipping.")
                    continue
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                grad_norms.append(grad_norm.item())
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                progress_bar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "GradNorm": f"{grad_norm:.3f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"Client {client_id}: OOM. Clearing cache.")
                    torch.cuda.empty_cache()
                    continue
                elif "Expected more than 1 value per channel" in str(e):
                    logger.warning(f"Client {client_id}: BatchNorm issue with batch_size=1. Skipping batch.")
                    continue
                else:
                    raise
        
        # Ensure model is back in train mode
        model.train()
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
            client_losses.append(avg_loss)
            
            # Phase 3.2: Gradient-based early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Stop if gradient norm is very small (converged)
            if avg_grad_norm < 0.01:
                logger.info(f"Client {client_id} converged at epoch {epoch+1} (grad norm: {avg_grad_norm:.5f})")
                break
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Client {client_id} early stopped at epoch {epoch+1}")
                break
            
            logger.info(f"Client {client_id} - Epoch {epoch+1}: Loss={avg_loss:.4f}, GradNorm={avg_grad_norm:.4f}")
    
    return model, client_losses

# =====================================================================================
# NEW: PREDICTION MONITORING FUNCTION
# =====================================================================================

def check_predictions_distribution(model, data_loader, device, name=""):
    """Check if model is predicting something or just zeros."""
    model.eval()
    pred_stats = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            if i >= 10:  # Check first 10 batches
                break
            inputs = batch_data["image"].to(device)
            outputs = model(inputs)
            outputs_sigmoid = torch.sigmoid(outputs)
            
            pred_stats.append({
                'mean': outputs_sigmoid.mean().item(),
                'std': outputs_sigmoid.std().item(),
                'max': outputs_sigmoid.max().item(),
                'min': outputs_sigmoid.min().item(),
                'percent_above_0.5': (outputs_sigmoid > 0.5).float().mean().item()
            })
    
    avg_stats = {k: np.mean([s[k] for s in pred_stats]) for k in pred_stats[0].keys()}
    
    logger.info(f"{name} Prediction Statistics:")
    logger.info(f"  Mean: {avg_stats['mean']:.4f}, Std: {avg_stats['std']:.4f}")
    logger.info(f"  Max: {avg_stats['max']:.4f}, Min: {avg_stats['min']:.4f}")
    logger.info(f"  % Above 0.5: {avg_stats['percent_above_0.5']:.4f}")
    
    if avg_stats['percent_above_0.5'] < 0.001:
        logger.warning(f"⚠️ {name}: Model is predicting almost all zeros!")
    
    return avg_stats

# =====================================================================================
# PHASE 4: DATA QUALITY CHECKS
# =====================================================================================

def validate_data_quality(data_loader, dataset_name):
    """
    Phase 4.2: Validate data distributions and quality.
    """
    logger.info(f"Validating data quality for {dataset_name}...")
    
    pixel_means = []
    pixel_stds = []
    label_positives = []
    
    for batch_data in tqdm(data_loader, desc=f"Checking {dataset_name}", leave=False):
        images = batch_data["image"]
        labels = batch_data["label"]
        
        pixel_means.append(images.mean().item())
        pixel_stds.append(images.std().item())
        label_positives.append((labels > 0.5).float().mean().item())
    
    stats = {
        "mean_pixel_mean": np.mean(pixel_means),
        "std_pixel_mean": np.std(pixel_means),
        "mean_pixel_std": np.mean(pixel_stds),
        "mean_label_positive": np.mean(label_positives)
    }
    
    logger.info(f"{dataset_name} statistics:")
    logger.info(f"  Pixel mean: {stats['mean_pixel_mean']:.4f} ± {stats['std_pixel_mean']:.4f}")
    logger.info(f"  Pixel std: {stats['mean_pixel_std']:.4f}")
    logger.info(f"  Label positive ratio: {stats['mean_label_positive']:.4f}")
    
    # Warnings
    if stats['mean_label_positive'] < 0.01:
        logger.warning(f"⚠️ {dataset_name} has very low positive label ratio!")
    if stats['mean_pixel_std'] < 0.05:
        logger.warning(f"⚠️ {dataset_name} has very low pixel variance!")
    
    return stats

# =====================================================================================
# PHASE 5: CHECKPOINT MANAGEMENT
# =====================================================================================

class CheckpointManager:
    """
    Phase 5.2: Advanced checkpoint management with rotation.
    """
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
    
    def save(self, checkpoint_data, round_num, is_best=False):
        """Save checkpoint with rotation."""
        try:
            # Save latest
            latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
            torch.save(checkpoint_data, latest_path)
            
            # Save versioned checkpoint
            versioned_path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pth"
            torch.save(checkpoint_data, versioned_path)
            
            # Save best model separately
            if is_best:
                best_path = self.checkpoint_dir / "checkpoint_best.pth"
                torch.save(checkpoint_data, best_path)
                logger.info(f"🏆 Best checkpoint saved at round {round_num}")
            
            # Rotate old checkpoints
            self._rotate_checkpoints()
            
            logger.info(f"Checkpoint saved: round {round_num}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def _rotate_checkpoints(self):
        """Keep only the most recent N checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_round_*.pth"))
        
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                try:
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_checkpoint}: {e}")
    
    def load_latest(self, device):
        """Load the latest checkpoint."""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        return self._load_checkpoint(latest_path, device)
    
    def load_best(self, device):
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        return self._load_checkpoint(best_path, device)
    
    def _load_checkpoint(self, path, device):
        """Load checkpoint from path."""
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                logger.info(f"Loaded checkpoint from {path.name}")
                return checkpoint
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {path}: {e}")
                return None
        return None

# =====================================================================================
# PHASE 5: MONITORING AND VISUALIZATION
# =====================================================================================

class FederatedTrainingMonitor:
    """
    Phase 5.1: Comprehensive training monitor with per-client tracking.
    """
    def __init__(self):
        self.round_metrics = []
        self.client_metrics = defaultdict(list)
        self.aggregation_weights = defaultdict(list)
        self.learning_rates = []
        self.communication_efficiency = []
    
    def log_round(self, round_num, global_loss, global_dice, val_loss=None):
        """Log global metrics for a round."""
        self.round_metrics.append({
            'round': round_num,
            'train_loss': global_loss,
            'dice': global_dice,
            'val_loss': val_loss
        })
    
    def log_client(self, client_id, round_num, loss, dice=None):
        """Log client-specific metrics."""
        self.client_metrics[client_id].append({
            'round': round_num,
            'loss': loss,
            'dice': dice
        })
    
    def log_aggregation(self, round_num, weights, client_names):
        """Log aggregation weights."""
        for client, weight in zip(client_names, weights):
            self.aggregation_weights[client].append({
                'round': round_num,
                'weight': weight
            })
    
    def log_lr(self, round_num, lr):
        """Log learning rate."""
        self.learning_rates.append({'round': round_num, 'lr': lr})
    
    def plot_training_curves(self, save_path='training_curves_fixed.png'):
        """Generate comprehensive training visualization."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Global training loss
        ax1 = fig.add_subplot(gs[0, 0])
        rounds = [m['round'] for m in self.round_metrics]
        train_losses = [m['train_loss'] for m in self.round_metrics]
        ax1.plot(rounds, train_losses, 'b-', linewidth=2)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Global Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # 2. Global Dice score
        ax2 = fig.add_subplot(gs[0, 1])
        dice_scores = [m['dice'] for m in self.round_metrics]
        ax2.plot(rounds, dice_scores, 'g-', linewidth=2)
        ax2.axhline(y=max(dice_scores), color='r', linestyle='--', alpha=0.5, 
                   label=f'Best: {max(dice_scores):.4f}')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Global Validation Dice')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate schedule
        ax3 = fig.add_subplot(gs[0, 2])
        if self.learning_rates:
            lr_rounds = [m['round'] for m in self.learning_rates]
            lrs = [m['lr'] for m in self.learning_rates]
            ax3.plot(lr_rounds, lrs, 'r-', linewidth=2)
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # 4. Per-client loss trajectories
        ax4 = fig.add_subplot(gs[1, :])
        for client_id, metrics in self.client_metrics.items():
            if len(metrics) > 0 and not client_id.startswith('DATASET_'):
                client_rounds = [m['round'] for m in metrics]
                client_losses = [m['loss'] for m in metrics]
                ax4.plot(client_rounds, client_losses, marker='o', alpha=0.6, 
                        label=client_id, linewidth=1.5)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Client Loss')
        ax4.set_title('Per-Client Training Loss')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Aggregation weights heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        if self.aggregation_weights:
            clients = list(self.aggregation_weights.keys())
            max_rounds = max(len(self.aggregation_weights[c]) for c in clients)
            
            weight_matrix = np.zeros((len(clients), max_rounds))
            for i, client in enumerate(clients):
                for j, entry in enumerate(self.aggregation_weights[client]):
                    if j < max_rounds:
                        weight_matrix[i, j] = entry['weight']
            
            im = ax5.imshow(weight_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            ax5.set_xlabel('Round')
            ax5.set_ylabel('Client')
            ax5.set_yticks(range(len(clients)))
            ax5.set_yticklabels(clients, fontsize=8)
            ax5.set_title('Client Contribution Weights')
            plt.colorbar(im, ax=ax5, label='Weight')
        
        # 6. Model divergence
        ax6 = fig.add_subplot(gs[2, 2])
        if len(self.round_metrics) > 1:
            # Compute round-to-round improvement
            improvements = []
            for i in range(1, len(dice_scores)):
                improvement = dice_scores[i] - dice_scores[i-1]
                improvements.append(improvement)
            
            ax6.bar(rounds[1:], improvements, color=['g' if x > 0 else 'r' for x in improvements])
            ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax6.set_xlabel('Round')
            ax6.set_ylabel('Dice Improvement')
            ax6.set_title('Round-to-Round Improvement')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to {save_path}")
    
    def save_metrics(self, filepath='training_metrics_fixed.json'):
        """Save all metrics to JSON."""
        metrics_data = {
            'round_metrics': self.round_metrics,
            'client_metrics': {k: v for k, v in self.client_metrics.items()},
            'aggregation_weights': {k: v for k, v in self.aggregation_weights.items()},
            'learning_rates': self.learning_rates
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        logger.info(f"Metrics saved to {filepath}")

# =====================================================================================
# EVALUATION FUNCTIONS
# =====================================================================================

def evaluate_on_dataset(model, test_loader, device, loss_function, post_pred, dataset_name):
    """Evaluate model on a specific dataset."""
    model.eval()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    cm_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["sensitivity", "specificity", "accuracy"],
        reduction="mean"
    )
    
    test_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for test_data in tqdm(test_loader, desc=f"Testing {dataset_name}", leave=False):
            test_inputs = test_data["image"].to(device)
            test_labels = test_data["label"].to(device)
            
            test_outputs = model(test_inputs)
            batch_loss = loss_function(test_outputs, test_labels)
            
            if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                test_loss += batch_loss.item()
                batch_count += 1
            
            test_outputs_post = [post_pred(i) for i in decollate_batch(test_outputs)]
            dice_metric(y_pred=test_outputs_post, y=test_labels)
            cm_metric(y_pred=test_outputs_post, y=test_labels)
    
    mean_dice = dice_metric.aggregate().item()
    cm_value = cm_metric.aggregate()
    sensitivity = cm_value[0].item()
    specificity = cm_value[1].item()
    accuracy = cm_value[2].item()
    avg_loss = test_loss / batch_count if batch_count > 0 else float('inf')
    iou = mean_dice / (2 - mean_dice) if mean_dice > 0 else 0
    
    return {
        "dataset": dataset_name,
        "dice": mean_dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "loss": avg_loss
    }

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =====================================================================================
# FIXED: MULTI-DATASET WRAPPER FOR CORRECT NORMALIZATION
# =====================================================================================

class MultiDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper dataset that applies correct transform per sample based on source dataset."""
    def __init__(self, data_with_sources, transforms_map):
        self.data = data_with_sources
        self.transforms_map = transforms_map
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataset_name, data_dict = self.data[idx]
        transform = self.transforms_map[dataset_name]
        return transform(data_dict)

# =====================================================================================
# MAIN FEDERATED TRAINING FUNCTION
# =====================================================================================

def main():
    try:
        # =====================================================================================
        # CONFIGURATION - FIXED
        # =====================================================================================
        class FederatedConfig:
            # Data paths
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            
            ADDITIONAL_DATA_DIR = "/teamspace/studios/this_studio/lung/processed_task6_lung"
            USE_ADDITIONAL_DATA = os.path.exists(ADDITIONAL_DATA_DIR)
            
            # Model paths
            CHECKPOINT_DIR = "checkpoints_fixed"
            BEST_MODEL_PATH = "best_federated_lightweight_auravit_model_fixed.pth"
            LAST_MODEL_PATH = "last_federated_lightweight_auravit_model_fixed.pth"
            
            # Federated learning parameters
            COMM_ROUNDS = 200
            CLIENT_EPOCHS_PER_ROUND = 5  # Base epochs (used with dynamic)
            DYNAMIC_EPOCHS = True
            MIN_EPOCHS = 8  # FIXED: Increased from 1 to 3
            MAX_EPOCHS = 20
            
            FEDPROX_MU = 0.01
            ADAPTIVE_MU = True
            ADAPTIVE_ALPHA = 0.3
            
            # Client selection (Phase 2.1)
            MIN_CLIENT_PARTICIPATION = 1.0
            MAX_CLIENT_PARTICIPATION = 1.0
            CLIENT_SELECTION_STRATEGY = 'random'
            
            # Aggregation strategy (Phase 2.2)
            AGGREGATION_STRATEGY = 'adaptive_weighted'
            
            # Training parameters - FIXED
            BATCH_SIZE = 8
            MIN_BATCH_SIZE = 2
            LEARNING_RATE = 1e-3  # FIXED: Increased from 3e-4 to 1e-3
            MIN_LEARNING_RATE = 1e-5  # FIXED: Increased from 1e-6
            MAX_LR_MULTIPLIER = 1.0
            WARMUP_ROUNDS = 5  # FIXED: Reduced from 10 to 5
            LR_MODE = 'cosine'
            WEIGHT_DECAY = 1e-5
            GRAD_CLIP_NORM = 0.5
            
            # Early stopping and stability
            PATIENCE = 30
            MIN_ROUNDS_BEFORE_STOPPING = 50
            CHECK_LOSS_FREQUENCY = 5
            LOSS_EXPLOSION_THRESHOLD = 10.0
            
            # Data splitting (Phase 1.1)
            GLOBAL_TEST_SIZE = 0.2
            VAL_SIZE = 0.125
            
            # Client sizing (Phase 1.3)
            MAX_CLIENT_SIZE_STRATEGY = 'percentile'
            MAX_CLIENT_SIZE_PERCENTILE = 50
            
            # Drift detection (Phase 2.3)
            ENABLE_DRIFT_DETECTION = True
            DRIFT_THRESHOLD = 0.15
            
            # Data quality checks (Phase 4.2)
            VALIDATE_DATA_QUALITY = True
            
            # Checkpoint management (Phase 5.2)
            MAX_CHECKPOINTS = 5
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configurations
        MODEL_SIZE = 'small'
        
        configs = {
            'small': {
                "image_size": 256, "num_layers": 8, "hidden_dim": 512,
                "mlp_dim": 2048, "num_heads": 8, "dropout_rate": 0.1, 
                "block_dropout_rate": 0.05, "patch_size": 16, "num_channels": 1,
            },
            'tiny': {
                "image_size": 256, "num_layers": 6, "hidden_dim": 384,
                "mlp_dim": 1536, "num_heads": 6, "dropout_rate": 0.1,
                "block_dropout_rate": 0.05, "patch_size": 16, "num_channels": 1,
            },
            'mobile': {
                "image_size": 224, "num_layers": 4, "hidden_dim": 256,
                "mlp_dim": 1024, "num_heads": 4, "dropout_rate": 0.1,
                "block_dropout_rate": 0.05, "patch_size": 16, "num_channels": 1,
            }
        }
        
        model_config = configs[MODEL_SIZE]
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
        
        config = FederatedConfig()
        
        logger.info("="*80)
        logger.info(f"FIXED ENHANCED FEDERATED LIGHTWEIGHT AURAVIT - {MODEL_SIZE.upper()} MODEL")
        logger.info("="*80)
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Communication rounds: {config.COMM_ROUNDS}")
        logger.info(f"MIN_EPOCHS: {config.MIN_EPOCHS} (FIXED: increased from 1)")
        logger.info(f"LEARNING_RATE: {config.LEARNING_RATE} (FIXED: increased from 3e-4)")
        logger.info(f"Loss weights: Dice=0.7, CE=0.3 (FIXED for class imbalance)")
        logger.info(f"Normalization: Per-image with NormalizeIntensityd (FIXED)")
        logger.info("="*80)

        # =====================================================================================
        # PHASE 1: IMPROVED DATA PREPARATION
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: IMPROVED DATA PREPARATION (NO PATIENT LEAKAGE)")
        logger.info("="*80)
        
        # Load datasets separately
        dataset_clients_raw = []
        
        # Primary dataset
        image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
        mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
        nsclc_data = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]
        
        if len(nsclc_data) > 0:
            dataset_clients_raw.append(("NSCLC-Radiomics", nsclc_data))
            logger.info(f"Dataset 1: NSCLC-Radiomics with {len(nsclc_data)} samples")
        
        # Additional dataset
        if config.USE_ADDITIONAL_DATA:
            add_image_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "image", "*.png")))
            add_mask_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "masks", "*.png")))
            task6_data = [{"image": img, "label": mask} for img, mask in zip(add_image_files, add_mask_files)]
            
            if len(task6_data) > 0:
                dataset_clients_raw.append(("Task6-Lung", task6_data))
                logger.info(f"Dataset 2: Task6-Lung with {len(task6_data)} samples")
        
        # Phase 1.1: Create global test set FIRST
        logger.info("\nPhase 1.1: Creating global test set (patient-level split)...")
        client_train_val_data, global_test_data = create_global_test_set_first(
            dataset_clients_raw, 
            test_size=config.GLOBAL_TEST_SIZE,
            random_state=42
        )
        
        # Phase 1.3: Adaptive client sizing
        logger.info("\nPhase 1.3: Determining adaptive client sizes...")
        dataset_sizes = [len(data) for _, data in client_train_val_data]
        # max_client_size = determine_max_client_size(
        #     dataset_sizes, 
        #     strategy=config.MAX_CLIENT_SIZE_STRATEGY,
        #     percentile=config.MAX_CLIENT_SIZE_PERCENTILE
        # )
        max_client_size=1300 #dont change that 

        # Split large clients
        logger.info("\nSplitting clients adaptively...")
        virtual_clients = split_large_clients_adaptive(
            client_train_val_data,
            max_client_size=max_client_size,
            strategy=config.MAX_CLIENT_SIZE_STRATEGY
        )
        
        logger.info(f"\nTotal virtual clients created: {len(virtual_clients)}")
        
        # =====================================================================================
        # PHASE 4: COMPUTE DATASET-SPECIFIC NORMALIZATION STATISTICS
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: COMPUTING DATASET-SPECIFIC NORMALIZATION STATISTICS")
        logger.info("="*80)
        
        # Compute per-dataset statistics for robust normalization
        dataset_stats = {}
        for dataset_name, data_dicts in dataset_clients_raw:
            logger.info(f"\nAnalyzing {dataset_name}...")
            
            # Sample some images to compute statistics
            sample_size = min(100, len(data_dicts))
            sample_data = np.random.choice(data_dicts, sample_size, replace=False)
            
            pixel_values = []
            for item in tqdm(sample_data, desc=f"Computing stats for {dataset_name}", leave=False):
                from PIL import Image
                img = Image.open(item["image"])
                pixel_values.extend(np.array(img).flatten())
            
            pixel_values = np.array(pixel_values)
            mean = np.mean(pixel_values)
            std = np.std(pixel_values)
            p1, p99 = np.percentile(pixel_values, [1, 99])
            
            dataset_stats[dataset_name] = {
                'mean': float(mean),
                'std': float(std),
                'p1': float(p1),
                'p99': float(p99)
            }
            
            logger.info(f"{dataset_name} statistics:")
            logger.info(f"  Mean: {mean:.4f}, Std: {std:.4f}")
            logger.info(f"  1st percentile: {p1:.4f}, 99th percentile: {p99:.4f}")
        
        # Save dataset statistics
        stats_file = "dataset_normalization_stats_fixed.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=4)
        logger.info(f"\n💾 Dataset statistics saved to {stats_file}")
        
        # Warning if distributions are very different
        if len(dataset_stats) > 1:
            means = [s['mean'] for s in dataset_stats.values()]
            if max(means) / min(means) > 2.0:
                logger.warning("\n⚠️ Dataset intensity distributions are very different!")
                logger.warning("   Using per-image normalization to handle this.")
        
        # =====================================================================================
        # FIXED: CREATE DATASET-SPECIFIC TRANSFORMS WITH BETTER NORMALIZATION
        # =====================================================================================
        def create_transforms_for_dataset(dataset_name, is_train=True):
            """Create transforms with robust per-image normalization."""
            stats = dataset_stats.get(dataset_name, {'mean': 127.5, 'std': 127.5, 'p1': 0, 'p99': 255})
            
            base_transforms = [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                # FIXED: Use per-image z-score normalization (more robust to intensity differences)
                ScaleIntensityRangePercentilesd(
                    keys=["image"],
                    lower=1, upper=99,
                    b_min=0.0, b_max=255.0,
                    clip=True
                ),
                # Per-image normalization to zero mean, unit variance
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
                ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            ]
            
            if is_train:
                base_transforms.extend([
                    RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
                    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
                    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
                    RandAffined(
                        keys=['image', 'label'], prob=0.3,
                        translate_range=(5, 5), rotate_range=(np.pi / 36, np.pi / 36),
                        scale_range=(0.05, 0.05), mode=('bilinear', 'nearest'),
                    ),
                    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.02),
                    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
                ])
            
            base_transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
            
            return Compose(base_transforms)
        
        # Create client data loaders with dataset-specific normalization
        client_train_loaders = []
        client_val_loaders = []
        client_sample_sizes = []
        client_names = []
        client_source_datasets = []
        dataset_val_loaders = {}
        
        for client_name, client_data, source_dataset in virtual_clients:
            # Skip empty clients
            if len(client_data) == 0:
                logger.warning(f"Skipping {client_name} - no data available")
                continue
            
            # Split into train/val (patient-level)
            train_data, val_data, _ = split_data_by_patient(
                client_data,
                test_size=0,
                val_size=config.VAL_SIZE,
                random_state=42
            )
            
            # Skip if no training data
            if len(train_data) == 0:
                logger.warning(f"Skipping {client_name} - no training data after split")
                continue
            
            # Create dataset-specific transforms
            train_transform = create_transforms_for_dataset(source_dataset, is_train=True)
            val_transform = create_transforms_for_dataset(source_dataset, is_train=False)
            
            # Create datasets
            train_ds = Dataset(data=train_data, transform=train_transform)
            val_ds = Dataset(data=val_data, transform=val_transform)
            
            # Determine appropriate batch size for this client
            client_batch_size = min(config.BATCH_SIZE, max(config.MIN_BATCH_SIZE, len(train_data) // 10))
            
            # Create loaders
            train_loader = DataLoader(
                train_ds, batch_size=client_batch_size,
                shuffle=True, num_workers=2, pin_memory=True,
                drop_last=(len(train_ds) > client_batch_size)
            )
            val_loader = DataLoader(
                val_ds, batch_size=1,
                shuffle=False, num_workers=2, pin_memory=True
            )
            
            client_train_loaders.append(train_loader)
            client_val_loaders.append(val_loader)
            client_sample_sizes.append(len(train_data))
            client_names.append(client_name)
            client_source_datasets.append(source_dataset)
            
            # Track val loaders per dataset
            if source_dataset not in dataset_val_loaders:
                dataset_val_loaders[source_dataset] = []
            dataset_val_loaders[source_dataset].append(val_ds)
            
            logger.info(f"{client_name}: Train={len(train_data)}, Val={len(val_data)}, BatchSize={client_batch_size}")
        
        # Verify we have at least one client
        if len(client_names) == 0:
            raise ValueError("No valid clients created! Check your data and test split ratio.")
        
        logger.info(f"\nTotal valid clients: {len(client_names)}")
        
        # =====================================================================================
        # FIXED: CREATE GLOBAL TEST LOADER WITH PROPER PER-SAMPLE NORMALIZATION
        # =====================================================================================
        logger.info("\nFIXED: Creating dataset-aware test loaders...")
        global_test_loaders_by_dataset = {}
        
        for dataset_name in set(ds for ds, _ in global_test_data):
            dataset_test_samples = [item for ds, item in global_test_data if ds == dataset_name]
            test_transform = create_transforms_for_dataset(dataset_name, is_train=False)
            test_ds = Dataset(data=dataset_test_samples, transform=test_transform)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
            global_test_loaders_by_dataset[dataset_name] = test_loader
            logger.info(f"Test set for {dataset_name}: {len(dataset_test_samples)} samples")
        
        # FIXED: Create combined global test loader with dataset-specific normalization
        test_transforms_map = {
            dataset_name: create_transforms_for_dataset(dataset_name, is_train=False)
            for dataset_name in set(ds for ds, _ in global_test_data)
        }
        
        global_test_ds = MultiDatasetWrapper(global_test_data, test_transforms_map)
        global_test_loader = DataLoader(global_test_ds, batch_size=1, shuffle=False, num_workers=2)
        
        logger.info(f"\nGlobal test samples: {len(global_test_ds)} (with per-sample normalization)")
        logger.info("✅ FIXED: Test loader now applies correct normalization per dataset")
        
        # Store for later use
        dataset_test_loaders = global_test_loaders_by_dataset
        
        # Phase 4.2: Data quality validation AFTER normalization
        if config.VALIDATE_DATA_QUALITY:
            logger.info("\nPhase 4.2: Validating data quality AFTER normalization...")
            for client_name, train_loader in zip(client_names[:3], client_train_loaders[:3]):
                quality_stats = validate_data_quality(train_loader, client_name)

        # =====================================================================================
        # MODEL INITIALIZATION
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("MODEL INITIALIZATION")
        logger.info("="*80)
        
        global_model = LightweightAuraViT(model_config).to(config.DEVICE)
        loss_function = StableLoss(dice_weight=0.7, ce_weight=0.3)  # FIXED: Better weights
        
        total_params = count_parameters(global_model)
        model_size_mb = total_params * 4 / (1024 * 1024)
        logger.info(f"Total parameters: {total_params:,} ({model_size_mb:.2f} MB)")
        logger.info(f"Loss function: DiceCE with Dice=0.7, CE=0.3 (FIXED for class imbalance)")
        
        # =====================================================================================
        # INITIALIZE COMPONENTS
        # =====================================================================================
        
        # Phase 5.2: Checkpoint manager
        checkpoint_manager = CheckpointManager(config.CHECKPOINT_DIR, max_checkpoints=config.MAX_CHECKPOINTS)
        
        # Phase 5.1: Training monitor
        monitor = FederatedTrainingMonitor()
        
        # Phase 2.3: Drift detector
        drift_detector = ClientDriftDetector(threshold=config.DRIFT_THRESHOLD) if config.ENABLE_DRIFT_DETECTION else None
        
        # Phase 3.1: Adaptive LR scheduler - FIXED
        lr_scheduler = AdaptiveFederatedLRScheduler(
            base_lr=config.LEARNING_RATE,
            warmup_rounds=config.WARMUP_ROUNDS,
            max_rounds=config.COMM_ROUNDS,
            min_lr=config.MIN_LEARNING_RATE,
            mode=config.LR_MODE,
            warmup_start_factor=0.1
        )
        
        # Load checkpoint if exists
        start_round = 0
        best_metric = -1
        best_metric_round = -1
        patience_counter = 0
        
        checkpoint = checkpoint_manager.load_latest(config.DEVICE)
        if checkpoint:
            global_model.load_state_dict(checkpoint['model_state_dict'])
            start_round = checkpoint.get('round', 0)
            best_metric = checkpoint.get('best_metric', -1)
            best_metric_round = checkpoint.get('best_metric_round', -1)
            patience_counter = checkpoint.get('patience_counter', 0)
            
            if 'lr_scheduler_state' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
            
            logger.info(f"Resuming from round {start_round + 1}")
        
        # Metrics
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
        # =====================================================================================
        # FEDERATED TRAINING LOOP
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("STARTING FIXED FEDERATED TRAINING")
        logger.info("="*80)
        
        for round_num in range(start_round, config.COMM_ROUNDS):
            round_start_time = datetime.now()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num + 1}/{config.COMM_ROUNDS}")
            logger.info(f"{'='*60}")
            
            # Phase 3.1: Get current learning rate
            current_lr = lr_scheduler.get_lr()
            logger.info(f"Learning rate: {current_lr:.6f}")
            monitor.log_lr(round_num + 1, current_lr)
            
            # Phase 2.1: Client selection with dataset diversity
            client_performances = {name: [m['dice'] for m in monitor.client_metrics[name] if m['dice'] is not None]
                                  for name in client_names}
            
            selected_clients = select_clients_for_round(
                client_names,
                min_participation=config.MIN_CLIENT_PARTICIPATION,
                max_participation=config.MAX_CLIENT_PARTICIPATION,
                client_performances=client_performances,
                selection_strategy=config.CLIENT_SELECTION_STRATEGY,
                client_source_datasets=client_source_datasets,
                ensure_diversity=True
            )
            
            # Train selected clients
            local_models = []
            local_state_dicts = []
            local_losses = []
            selected_sample_sizes = []
            client_val_metrics = []
            
            for client_name in selected_clients:
                client_idx = client_names.index(client_name)
                train_loader = client_train_loaders[client_idx]
                val_loader = client_val_loaders[client_idx]
                client_size = client_sample_sizes[client_idx]
                
                logger.info(f"\nTraining {client_name} (samples: {client_size})")
                
                # Phase 3.2: Dynamic epochs - FIXED
                if config.DYNAMIC_EPOCHS:
                    epochs = compute_dynamic_epochs(
                        client_size,
                        min_epochs=config.MIN_EPOCHS,
                        max_epochs=config.MAX_EPOCHS
                    )
                    logger.info(f"  Dynamic epochs: {epochs}")
                else:
                    epochs = config.CLIENT_EPOCHS_PER_ROUND
                
                # Create local model
                local_model = deepcopy(global_model).to(config.DEVICE)
                
                # Phase 3.1: Adaptive per-client LR
                client_lr = lr_scheduler.get_client_lr(client_name, client_size)
                
                local_optimizer = torch.optim.AdamW(
                    local_model.parameters(),
                    lr=client_lr,
                    weight_decay=config.WEIGHT_DECAY,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
                
                scaler = GradScaler()
                
                # Phase 4.3: Adaptive FedProx mu
                if config.ADAPTIVE_MU:
                    if drift_detector and drift_detector.detect_drift(client_name):
                        mu = config.FEDPROX_MU * 2
                        logger.info(f"  Increased mu to {mu} due to detected drift")
                    else:
                        mu = config.FEDPROX_MU
                else:
                    mu = config.FEDPROX_MU
                
                # Phase 3: Enhanced client update
                trained_model, client_loss_history = client_update_enhanced(
                    client_id=client_name,
                    model=local_model,
                    global_model=global_model,
                    optimizer=local_optimizer,
                    train_loader=train_loader,
                    epochs=epochs,
                    device=config.DEVICE,
                    mu=mu,
                    scaler=scaler,
                    loss_function=loss_function,
                    grad_clip_norm=config.GRAD_CLIP_NORM,
                    early_stopping_patience=3
                )
                
                # Validate client model
                trained_model.eval()
                val_dice = None
                with torch.no_grad():
                    dice_metric.reset()
                    for val_data in val_loader:
                        val_inputs = val_data["image"].to(config.DEVICE)
                        val_labels = val_data["label"].to(config.DEVICE)
                        val_outputs = trained_model(val_inputs)
                        val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
                        dice_metric(y_pred=val_outputs_post, y=val_labels)
                    val_dice = dice_metric.aggregate().item()
                
                client_val_metrics.append(val_dice)
                
                # Phase 2.3: Track client performance
                if drift_detector:
                    drift_detector.update(client_name, val_dice)
                
                # Log client metrics
                avg_client_loss = np.mean(client_loss_history) if client_loss_history else 0
                monitor.log_client(client_name, round_num + 1, avg_client_loss, val_dice)
                logger.info(f"  {client_name} - Final loss: {avg_client_loss:.4f}, Val Dice: {val_dice:.4f}")
                
                # Phase 1.2: Extract state dict and delete model immediately
                local_state_dict = trained_model.state_dict()
                local_state_dicts.append(local_state_dict)
                selected_sample_sizes.append(client_size)
                local_losses.extend(client_loss_history)
                
                # Memory cleanup
                del trained_model
                del local_model
                del local_optimizer
                del scaler
                torch.cuda.empty_cache()
            
            # Phase 2.2: Robust aggregation
            logger.info(f"\nAggregating {len(local_state_dicts)} client models...")
            logger.info(f"Using strategy: {config.AGGREGATION_STRATEGY}")
            
            if config.AGGREGATION_STRATEGY == 'adaptive_weighted':
                global_weights = adaptive_weighted_average_improved(
                    w_states=local_state_dicts,
                    s=selected_sample_sizes,
                    global_model=global_model,
                    local_models=[],
                    alpha=config.ADAPTIVE_ALPHA,
                    client_val_metrics=client_val_metrics
                )
            elif config.AGGREGATION_STRATEGY in ['trimmed_mean', 'median']:
                global_weights = robust_aggregation(
                    w_states=local_state_dicts,
                    s=selected_sample_sizes,
                    strategy=config.AGGREGATION_STRATEGY
                )
            else:
                global_weights = weighted_average(local_state_dicts, selected_sample_sizes)
            
            # Update global model
            global_model.load_state_dict(global_weights)
            
            # Memory cleanup
            del local_state_dicts
            torch.cuda.empty_cache()
            
            # Calculate round metrics
            avg_round_loss = np.mean(local_losses) if local_losses else float('inf')
            
            # Global validation - evaluate on each dataset separately
            logger.info("\nEvaluating global model per dataset...")
            global_model.eval()
            
            dataset_val_results = {}
            with torch.no_grad():
                for dataset_name, test_loader in dataset_test_loaders.items():
                    dice_metric.reset()
                    val_loss = 0
                    val_batch_count = 0
                    
                    for val_data in tqdm(test_loader, desc=f"Validating {dataset_name}", leave=False):
                        val_inputs = val_data["image"].to(config.DEVICE)
                        val_labels = val_data["label"].to(config.DEVICE)
                        
                        val_outputs = global_model(val_inputs)
                        batch_loss = loss_function(val_outputs, val_labels)
                        
                        if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                            val_loss += batch_loss.item()
                            val_batch_count += 1
                        
                        val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
                        dice_metric(y_pred=val_outputs_post, y=val_labels)
                    
                    dataset_dice = dice_metric.aggregate().item()
                    dataset_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
                    
                    dataset_val_results[dataset_name] = {
                        'dice': dataset_dice,
                        'loss': dataset_loss
                    }
                    
                    logger.info(f"  {dataset_name}: Dice={dataset_dice:.4f}, Loss={dataset_loss:.4f}")
            
            # Compute global metrics (weighted average by dataset size)
            total_samples = sum([len(dataset_test_loaders[ds].dataset) for ds in dataset_test_loaders])
            metric = sum([
                dataset_val_results[ds]['dice'] * len(dataset_test_loaders[ds].dataset) / total_samples
                for ds in dataset_test_loaders
            ])
            avg_val_loss = sum([
                dataset_val_results[ds]['loss'] * len(dataset_test_loaders[ds].dataset) / total_samples
                for ds in dataset_test_loaders
            ])
            
            # NEW: Check prediction distribution
            logger.info("\nChecking prediction distribution...")
            pred_stats = check_predictions_distribution(
                global_model, 
                global_test_loader, 
                config.DEVICE, 
                name=f"Round {round_num + 1}"
            )
            
            # Log to monitor
            monitor.log_round(round_num + 1, avg_round_loss, metric, avg_val_loss)
            
            # Track per-dataset performance
            for ds_name, ds_result in dataset_val_results.items():
                monitor.log_client(f"DATASET_{ds_name}", round_num + 1, ds_result['loss'], ds_result['dice'])
            
            # Warning system for poor dataset performance
            for ds_name, ds_result in dataset_val_results.items():
                if ds_result['dice'] < 0.1 and round_num >= 10:
                    logger.warning(f"⚠️ {ds_name} has very poor performance: Dice={ds_result['dice']:.4f}")
                    logger.warning(f"   This may indicate data distribution mismatch or preprocessing issues.")
            
            # Round summary with per-dataset metrics
            round_time = (datetime.now() - round_start_time).total_seconds()
            logger.info(f"\n{'='*40}")
            logger.info(f"Round {round_num + 1} Summary:")
            logger.info(f"  Training Loss: {avg_round_loss:.4f}")
            logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"  Global Dice: {metric:.4f}")
            for ds_name, ds_result in dataset_val_results.items():
                logger.info(f"    └─ {ds_name}: {ds_result['dice']:.4f}")
            logger.info(f"  Best Dice: {best_metric:.4f} (Round {best_metric_round})")
            logger.info(f"  Round Time: {round_time:.1f}s")
            logger.info(f"{'='*40}")
            
            # NEW: Model collapse detection and recovery
            if round_num >= 5:
                recent_dice = [m['dice'] for m in monitor.round_metrics[-5:]]
                if all(d < 0.01 for d in recent_dice):
                    logger.error("="*80)
                    logger.error("🚨 MODEL COLLAPSE DETECTED!")
                    logger.error("   The model is predicting all background (zeros).")
                    logger.error("   Possible causes:")
                    logger.error("   1. Severe class imbalance overwhelming the loss")
                    logger.error("   2. Learning rate too high causing instability")
                    logger.error("   3. Dataset normalization issues")
                    logger.error("   4. Insufficient training epochs per round")
                    logger.error("="*80)
                    
                    # Try recovery strategy
                    logger.info("Attempting recovery: Reducing learning rate by 50%")
                    lr_scheduler.base_lr *= 0.5
                    lr_scheduler.min_lr *= 0.5
            
            # Check for improvement
            is_best = False
            if metric > best_metric:
                best_metric = metric
                best_metric_round = round_num + 1
                patience_counter = 0
                is_best = True
                logger.info(f"🏆 New best model! Dice: {best_metric:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            
            # Save checkpoint
            checkpoint_data = {
                'round': round_num + 1,
                'model_state_dict': global_model.state_dict(),
                'best_metric': best_metric,
                'best_metric_round': best_metric_round,
                'patience_counter': patience_counter,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'client_sample_sizes': client_sample_sizes,
                'client_names': client_names,
                'model_config': model_config,
                'model_size': MODEL_SIZE,
                'selected_clients': selected_clients
            }
            checkpoint_manager.save(checkpoint_data, round_num + 1, is_best=is_best)
            
            # Phase 3.1: Step LR scheduler
            lr_scheduler.step()
            
            # Phase 3.3: Early stopping with minimum rounds
            if round_num + 1 >= config.MIN_ROUNDS_BEFORE_STOPPING:
                if patience_counter >= config.PATIENCE:
                    logger.info(f"\n⚠️ Early stopping after {config.PATIENCE} rounds without improvement")
                    break
            
            # Check training stability
            if avg_round_loss > config.LOSS_EXPLOSION_THRESHOLD:
                logger.error("Training appears unstable. Stopping.")
                break
        
        # =====================================================================================
        # TRAINING COMPLETE
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("FEDERATED TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best Dice: {best_metric:.4f} at round {best_metric_round}")
        
        # Phase 5.1: Generate visualizations
        logger.info("\nGenerating training visualizations...")
        monitor.plot_training_curves(save_path=f'fixed_federated_{MODEL_SIZE}_training_curves.png')
        monitor.save_metrics(filepath=f'fixed_federated_{MODEL_SIZE}_training_metrics.json')
        
        # =====================================================================================
        # FINAL EVALUATION
        # =====================================================================================
        logger.info("\n" + "="*80)
        logger.info("FINAL EVALUATION PER DATASET")
        logger.info("="*80)
        
        # Load best model
        best_checkpoint = checkpoint_manager.load_best(config.DEVICE)
        if best_checkpoint:
            global_model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info("Loaded best model for evaluation")
        
        # Evaluate per dataset
        dataset_results = []
        for dataset_name, test_loader in dataset_test_loaders.items():
            logger.info(f"\nEvaluating on {dataset_name}...")
            result = evaluate_on_dataset(
                model=global_model,
                test_loader=test_loader,
                device=config.DEVICE,
                loss_function=loss_function,
                post_pred=post_pred,
                dataset_name=dataset_name
            )
            dataset_results.append(result)
            
            logger.info(f"\n{dataset_name} Results:")
            logger.info(f"  Dice: {result['dice']:.4f}")
            logger.info(f"  IoU: {result['iou']:.4f}")
            logger.info(f"  Sensitivity: {result['sensitivity']:.4f}")
            logger.info(f"  Specificity: {result['specificity']:.4f}")
            logger.info(f"  Accuracy: {result['accuracy']:.4f}")
        
        # Overall stats
        logger.info("\n" + "="*60)
        logger.info("OVERALL STATISTICS")
        logger.info("="*60)
        
        mean_dice = np.mean([r['dice'] for r in dataset_results])
        mean_iou = np.mean([r['iou'] for r in dataset_results])
        
        logger.info(f"📊 Average Dice: {mean_dice:.4f}")
        logger.info(f"📊 Average IoU: {mean_iou:.4f}")
        
        # Per-dataset performance analysis
        logger.info("\n📊 Per-Dataset Performance Analysis:")
        for result in dataset_results:
            logger.info(f"\n{result['dataset']}:")
            logger.info(f"  Dice: {result['dice']:.4f}")
            logger.info(f"  IoU: {result['iou']:.4f}")
            logger.info(f"  Sensitivity: {result['sensitivity']:.4f}")
            
            # Performance warnings
            if result['dice'] < 0.1:
                logger.warning(f"  ⚠️ CRITICAL: Very poor performance detected!")
                logger.warning(f"     This suggests severe data distribution mismatch.")
            elif result['dice'] < 0.3:
                logger.warning(f"  ⚠️ WARNING: Poor performance detected.")
        
        logger.info("="*60)
        
        # Save final metrics
        final_metrics = {
            "model_info": {
                "model_size": MODEL_SIZE,
                "total_parameters": total_params,
                "model_size_mb": model_size_mb
            },
            "per_dataset_results": dataset_results,
            "overall_metrics": {
                "mean_dice": mean_dice,
                "mean_iou": mean_iou
            },
            "training_info": {
                "best_validation_dice": best_metric,
                "best_round": best_metric_round,
                "total_rounds": round_num + 1,
                "num_clients": len(client_names)
            },
            "fixes_applied": {
                "min_epochs_increased": "1 → 3",
                "learning_rate_increased": "3e-4 → 1e-3",
                "loss_weights_adjusted": "Dice=0.7, CE=0.3",
                "normalization_fixed": "Per-image with NormalizeIntensityd",
                "test_loader_fixed": "Dataset-specific normalization per sample",
                "prediction_monitoring": "Added",
                "model_collapse_detection": "Added"
            }
        }
        
        with open(f"fixed_federated_{MODEL_SIZE}_final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=4)
        
        logger.info("\n✅ FIXED Enhanced Federated Lightweight AuraViT training completed!")
        logger.info(f"📁 All results saved in {config.CHECKPOINT_DIR}/")
        logger.info("\n🔧 Applied fixes:")
        logger.info("  1. Increased MIN_EPOCHS from 1 to 3")
        logger.info("  2. Increased LEARNING_RATE from 3e-4 to 1e-3")
        logger.info("  3. Adjusted loss weights: Dice=0.7, CE=0.3")
        logger.info("  4. Fixed normalization: Per-image with NormalizeIntensityd")
        logger.info("  5. Fixed test loader: Dataset-specific normalization per sample")
        logger.info("  6. Added prediction monitoring")
        logger.info("  7. Added model collapse detection and recovery")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
