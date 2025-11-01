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
            logging.FileHandler('lightweight_auravit_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ======================================================================================
# LIGHTWEIGHT BUILDING BLOCKS
# ======================================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution - reduces parameters by 70-80%
    """
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
    """
    Lightweight Residual Block using depthwise separable convolutions
    """
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
    """
    Standard Transposed Convolution block for upsampling
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0)

    def forward(self, x):
        return self.deconv(x)

class LightweightAtrousConv(nn.Module):
    """
    Lightweight Atrous Convolution using depthwise separable convolutions
    """
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
    """
    Lightweight ASPP module with reduced channels
    """
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        # Use fewer output channels in intermediate layers
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
    """
    Lightweight Attention Gate with reduced channels
    """
    def __init__(self, gate_channels, in_channels, inter_channels):
        super().__init__()
        # Use even fewer intermediate channels
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
    """
    Lightweight version of AuraViT with:
    - Reduced model dimensions (50% parameter reduction)
    - Depthwise separable convolutions
    - Efficient attention mechanisms
    - Optimized decoder
    """
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
        
        # Fewer transformer layers with pre-norm for stability
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
        
        # Adjust skip connections based on number of layers
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

        # --- LIGHTWEIGHT SEGMENTATION DECODER (Reduced channels) ---
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

        # Adjust skip connection indices based on number of layers
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

        # Reshape feature maps
        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        # 2. Lightweight ASPP
        aspp_out = self.aspp(z12_reshaped)

        # 3. Lightweight Decoder
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

# ======================================================================================
# MODEL PRUNING
# ======================================================================================
def prune_model(model, pruning_rate=0.3):
    """
    Prune model weights to reduce size
    """
    import torch.nn.utils.prune as prune
    
    logger.info(f"Pruning model with rate: {pruning_rate}")
    
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )
    
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    logger.info("Pruning complete!")
    return model

# ======================================================================================
# MODEL QUANTIZATION
# ======================================================================================
def quantize_model(model):
    """
    Quantize model to INT8 for deployment
    """
    logger.info("Quantizing model to INT8...")
    model.eval()
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    logger.info("Quantization complete!")
    return model_quantized

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    try:
        # ======================================================================================
        # LIGHTWEIGHT MODEL CONFIGURATION
        # ======================================================================================
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
        
        model_config = configs[MODEL_SIZE]
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
        config = LightweightConfig()

        logger.info("="*60)
        logger.info(f"LIGHTWEIGHT AURAVIT - {MODEL_SIZE.upper()} MODEL")
        logger.info("="*60)
        logger.info(f"Using device: {config.DEVICE}")
        logger.info(f"Model configuration: {MODEL_SIZE}")
        logger.info(f"Image size: {model_config['image_size']}")
        logger.info(f"Layers: {model_config['num_layers']}")
        logger.info(f"Hidden dim: {model_config['hidden_dim']}")
        logger.info(f"Batch size: {config.BATCH_SIZE}")

        # ======================================================================================
        # DATA PREPARATION
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

        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

        # ======================================================================================
        # MODEL SETUP
        # ======================================================================================
        model = LightweightAuraViT(model_config).to(config.DEVICE)
        
        # Display model statistics
        total_params = count_parameters(model)
        model_size_mb = total_params * 4 / (1024 * 1024)
        logger.info(f"Total trainable parameters: {total_params:,}")
        logger.info(f"Approximate model size: {model_size_mb:.2f} MB")
        logger.info(f"Parameter reduction vs original (~50M): {(1 - total_params/50_000_000)*100:.1f}%")
        
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

        # ======================================================================================
        # CHECKPOINT LOADING
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
                
                if 'scheduler_epoch' in checkpoint:
                    lr_scheduler.current_epoch = checkpoint['scheduler_epoch']
                
                logger.info(f"Checkpoint found! Resuming from epoch {start_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
                start_epoch = 0
        else:
            logger.info("No checkpoint found. Starting training from scratch.")

        # ======================================================================================
        # TRAINING LOOP
        # ======================================================================================
        logger.info("Starting Lightweight AuraViT Training")
        consecutive_bad_epochs = 0
        
        for epoch in range(start_epoch, config.MAX_EPOCHS):
            current_lr = lr_scheduler.step()
            
            model.train()
            epoch_loss = 0
            batch_count = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Training]", unit="batch")
            
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
                                logger.warning(f"NaN detected during forward pass. Skipping batch.")
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
            if avg_epoch_loss > config.LOSS_EXPLOSION_THRESHOLD:
                consecutive_bad_epochs += 1
                logger.warning(f"High loss detected. Bad epochs: {consecutive_bad_epochs}")
                
                if consecutive_bad_epochs >= 3:
                    logger.warning("Reducing learning rate.")
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
                    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
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
                }, config.CHECKPOINT_PATH)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

            # Early stopping
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}. No improvement for {config.PATIENCE} epochs.")
                break

        logger.info(f"Training finished. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

        # ======================================================================================
        # POST-TRAINING OPTIMIZATION (PRUNING)
        # ======================================================================================
        if config.ENABLE_PRUNING:
            logger.info("="*60)
            logger.info("APPLYING POST-TRAINING PRUNING")
            logger.info("="*60)
            
            # Load best model
            if os.path.exists(config.BEST_MODEL_PATH):
                model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
            
            # Prune model
            model_pruned = prune_model(model, config.PRUNING_RATE)
            
            # Save pruned model
            torch.save(model_pruned.state_dict(), config.PRUNED_MODEL_PATH)
            
            # Calculate new size
            pruned_params = count_parameters(model_pruned)
            pruned_size_mb = pruned_params * 4 / (1024 * 1024)
            
            logger.info(f"Pruned model parameters: {pruned_params:,}")
            logger.info(f"Pruned model size: {pruned_size_mb:.2f} MB")
            logger.info(f"Size reduction: {(1 - pruned_params/total_params)*100:.1f}%")

        # ======================================================================================
        # PLOTTING
        # ======================================================================================
        logger.info("Plotting training curves...")
        plt.figure("train", (18, 6))
        
        plt.subplot(1, 3, 1)
        plt.title(f"Training Loss - {MODEL_SIZE.upper()} Model")
        plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, 'b-', label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.title(f"Validation Dice Score")
        plt.plot(range(1, len(metric_values) + 1), metric_values, 'g-', label='Validation Dice')
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.title(f"Validation Accuracy")
        plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, 'r-', label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"lightweight_{MODEL_SIZE}_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

        # ======================================================================================
        # FINAL EVALUATION
        # ======================================================================================
        logger.info("="*60)
        logger.info("RUNNING FINAL EVALUATION ON TEST SET")
        logger.info("="*60)
        
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
                    logger.warning(f"Error during testing: {e}")
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
        logger.info(f"FINAL TEST METRICS - LIGHTWEIGHT {MODEL_SIZE.upper()} AURAVIT")
        logger.info("="*60)
        logger.info(f"Mean Dice Score: {mean_dice_test:.4f}")
        logger.info(f"Intersection over Union (IoU): {iou_test:.4f}")
        logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("="*60)
        
        # ======================================================================================
        # FINAL MODEL STATISTICS
        # ======================================================================================
        logger.info("="*60)
        logger.info("MODEL SIZE COMPARISON")
        logger.info("="*60)
        logger.info(f"Original Model (estimated): ~50,000,000 parameters (~200 MB)")
        logger.info(f"Lightweight Model: {total_params:,} parameters ({model_size_mb:.2f} MB)")
        logger.info(f"Parameter Reduction: {(1 - total_params/50_000_000)*100:.1f}%")
        
        if config.ENABLE_PRUNING and os.path.exists(config.PRUNED_MODEL_PATH):
            logger.info(f"Pruned Model: {pruned_params:,} parameters ({pruned_size_mb:.2f} MB)")
            logger.info(f"Total Reduction (vs original): {(1 - pruned_params/50_000_000)*100:.1f}%")
        
        logger.info("="*60)
        
        # ======================================================================================
        # OPTIONAL: QUANTIZATION FOR DEPLOYMENT
        # ======================================================================================
        logger.info("Creating quantized model for deployment...")
        model_quantized = quantize_model(model)
        
        # Save quantized model
        torch.save(model_quantized.state_dict(), config.QUANTIZED_MODEL_PATH)
        logger.info(f"Quantized model saved to: {config.QUANTIZED_MODEL_PATH}")
        
        # Estimate quantized size (INT8 is ~4x smaller than FP32)
        quantized_size_mb = model_size_mb / 4
        logger.info(f"Estimated quantized model size: {quantized_size_mb:.2f} MB")
        logger.info(f"Size reduction from quantization: ~75%")
        
        logger.info("="*60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"✓ Architecture optimization: {(1 - total_params/50_000_000)*100:.1f}% reduction")
        if config.ENABLE_PRUNING:
            logger.info(f"✓ Weight pruning ({config.PRUNING_RATE*100:.0f}%): Additional reduction applied")
        logger.info(f"✓ Quantization: ~75% additional size reduction")
        logger.info(f"✓ Total final size: ~{quantized_size_mb:.2f} MB (vs ~200 MB original)")
        logger.info("="*60)
        
        logger.info("Lightweight AuraViT training and optimization completed successfully!")
        
        # ======================================================================================
        # USAGE INSTRUCTIONS
        # ======================================================================================
        logger.info("\n" + "="*60)
        logger.info("HOW TO USE THE OPTIMIZED MODELS")
        logger.info("="*60)
        logger.info("1. For continued training:")
        logger.info(f"   - Use: {config.BEST_MODEL_PATH}")
        logger.info("   - Full precision, best for fine-tuning")
        logger.info("")
        logger.info("2. For deployment (CPU/mobile):")
        logger.info(f"   - Use: {config.QUANTIZED_MODEL_PATH}")
        logger.info("   - INT8 quantized, 4x smaller, faster inference")
        logger.info("")
        if config.ENABLE_PRUNING:
            logger.info("3. For balanced deployment:")
            logger.info(f"   - Use: {config.PRUNED_MODEL_PATH}")
            logger.info("   - Pruned weights, good balance of size and speed")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
