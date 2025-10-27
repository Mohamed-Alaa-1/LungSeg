"""

Enhanced AuraV3 with TRUE Federated Learning

- Data stays at clients (privacy-preserving)

- Per-client validation and testing

- Server only receives model weights and metrics

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

from monai.transforms import (

    AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, RandAffined,

    RandRotate90d, ResizeWithPadOrCropd, ScaleIntensityRanged, EnsureTyped,

    Activations, RandGaussianNoised, RandScaleIntensityd, RandFlipd,

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



set_determinism(seed=42)



# =====================================================================================

# LOGGING SETUP

# =====================================================================================

def setup_logging(log_file="federated_auravit_training.log"):

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

# MODEL BLOCKS

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

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),

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

# STABLE ENHANCED AURAVIT MODEL

# =====================================================================================

class StableEnhancedAuraViT(nn.Module):

    """Stability-Enhanced AuraViT."""

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

    """FedProx client update - data stays at client."""

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

                

                if torch.isnan(inputs).any() or torch.isnan(labels).any():

                    logger.warning(f"Client {client_id}: NaN detected in batch data. Skipping batch.")

                    continue

                

                optimizer.zero_grad()

                

                with autocast():

                    outputs = model(inputs)

                    primary_loss = loss_function(outputs, labels)

                    

                    proximal_term = 0.0

                    for name, param in model.named_parameters():

                        if param.requires_grad:

                            proximal_term += torch.sum((param - global_params[name]) ** 2)

                    

                    total_loss = primary_loss + (mu / 2) * proximal_term

                

                if torch.isnan(total_loss) or torch.isinf(total_loss):

                    logger.warning(f"Client {client_id}: Invalid loss detected. Skipping batch.")

                    continue

                

                scaler.scale(total_loss).backward()

                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

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



def client_validation(client_id, model, val_loader, device, loss_function, post_pred):

    """

    Client validates locally and returns only metrics.

    Raw data never leaves the client.

    """

    model.eval()

    

    dice_metric = DiceMetric(include_background=True, reduction="mean")

    cm_metric = ConfusionMatrixMetric(

        include_background=False,

        metric_name=["accuracy"],

        reduction="mean"

    )

    

    val_loss = 0

    batch_count = 0

    

    with torch.no_grad():

        for val_data in val_loader:

            val_inputs = val_data["image"].to(device)

            val_labels = val_data["label"].to(device)

            

            val_outputs = model(val_inputs)

            batch_loss = loss_function(val_outputs, val_labels)

            

            if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):

                val_loss += batch_loss.item()

                batch_count += 1

            

            val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]

            dice_metric(y_pred=val_outputs_post, y=val_labels)

            cm_metric(y_pred=val_outputs_post, y=val_labels)

    

    mean_dice = dice_metric.aggregate().item()

    accuracy = cm_metric.aggregate()[0].item()

    avg_loss = val_loss / batch_count if batch_count > 0 else float('inf')

    

    # Only metrics leave the client

    return {

        'client_id': client_id,

        'dice': mean_dice,

        'accuracy': accuracy,

        'loss': avg_loss,

        'num_samples': batch_count

    }



def client_test_evaluation(client_id, model, test_loader, device, loss_function, post_pred):

    """

    Client evaluates on local test set and returns only metrics.

    Raw data never leaves the client.

    """

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

        for test_data in tqdm(test_loader, desc=f"Testing {client_id}", leave=False):

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

    

    # Only metrics leave the client

    return {

        "client_id": client_id,

        "num_samples": batch_count,

        "dice": mean_dice,

        "iou": mean_dice / (2 - mean_dice) if mean_dice > 0 else 0,

        "sensitivity": cm_value[0].item(),

        "specificity": cm_value[1].item(),

        "accuracy": cm_value[2].item(),

        "loss": test_loss / batch_count if batch_count > 0 else float('inf')

    }



def adaptive_weighted_average(w_states, s, global_model, local_models, alpha):

    """Enhanced adaptive weighted averaging with stability improvements."""

    total_samples = sum(s)

    w_avg = deepcopy(w_states[0])

    

    similarity_weights = []

    for local_model in local_models:

        similarity = 0.0

        param_count = 0

        

        for (name1, param1), (name2, param2) in zip(global_model.named_parameters(), local_model.named_parameters()):

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

    if sim_sum > 0:

        similarity_weights = [weight / sim_sum for weight in similarity_weights]

    else:

        similarity_weights = [1.0 / len(w_states)] * len(w_states)

    

    final_weights = []

    for i in range(len(w_states)):

        data_weight = s[i] / total_samples

        combined_weight = alpha * similarity_weights[i] + (1 - alpha) * data_weight

        final_weights.append(combined_weight)

    

    final_sum = sum(final_weights)

    normalized_final_weights = [fw / final_sum for fw in final_weights]

    

    for key in w_avg.keys():

        if w_avg[key].dtype.is_floating_point:

            w_avg[key] = torch.zeros_like(w_avg[key])

            for i in range(len(w_states)):

                w_avg[key] += w_states[i][key] * normalized_final_weights[i]

    

    logger.info(f"Aggregation weights: {[f'{w:.3f}' for w in normalized_final_weights]}")

    

    return w_avg



def split_large_clients(dataset_clients, max_client_size):

    """

    Split large clients (datasets) into smaller virtual clients.

    """

    virtual_clients = []

    

    for dataset_name, data_dicts in dataset_clients:

        dataset_size = len(data_dicts)

        

        if dataset_size <= max_client_size:

            virtual_clients.append((f"{dataset_name}_c1", data_dicts, dataset_name))

            logger.info(f"  {dataset_name}: 1 client with {dataset_size} samples")

        else:

            num_splits = math.ceil(dataset_size / max_client_size)

            np.random.shuffle(data_dicts)

            

            splits = np.array_split(data_dicts, num_splits)

            for i, split in enumerate(splits, 1):

                virtual_clients.append((f"{dataset_name}_c{i}", split.tolist(), dataset_name))

            

            logger.info(f"  {dataset_name}: split into {num_splits} clients " + 

                       f"({[len(s) for s in splits]} samples each)")

    

    return virtual_clients



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

            checkpoint = torch.load(filepath, map_location=device, weights_only=False)

            logger.info(f"Checkpoint loaded from {filepath}")

            return checkpoint

        except Exception as e:

            logger.error(f"Failed to load checkpoint: {e}")

            return None

    return None



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

            DATA_DIR = "NSCLC"

            IMAGE_DIR = os.path.join(DATA_DIR, "images")

            MASK_DIR = os.path.join(DATA_DIR, "masks")

            

            ADDITIONAL_DATA_DIR = "MSD"

            USE_ADDITIONAL_DATA = os.path.exists(ADDITIONAL_DATA_DIR)

            

            # Model paths

            CHECKPOINT_PATH = "federated_auravit_checkpoint.pth"

            BEST_MODEL_PATH = "best_federated_auravit_model.pth"

            LAST_MODEL_PATH = "last_federated_auravit_model.pth"

            

            # Federated learning parameters

            COMM_ROUNDS = 100

            CLIENT_EPOCHS_PER_ROUND = 8

            FEDPROX_MU = 0.01

            ADAPTIVE_ALPHA = 0.3

            

            # Training parameters

            BATCH_SIZE = 4

            LEARNING_RATE = 2e-5

            MIN_LEARNING_RATE = 1e-6

            WEIGHT_DECAY = 1e-5

            GRAD_CLIP_NORM = 0.5

            

            # Early stopping

            PATIENCE = 20

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

        logger.info("PRIVACY-PRESERVING FEDERATED LEARNING")

        logger.info("="*80)

        logger.info(f"Device: {config.DEVICE}")

        logger.info(f"Communication rounds: {config.COMM_ROUNDS}")

        logger.info(f"FedProx mu: {config.FEDPROX_MU}")

        logger.info(f"Adaptive alpha: {config.ADAPTIVE_ALPHA}")

        logger.info("="*80)



        # =====================================================================================

        # DATA PREPARATION - SPLIT FIRST, THEN CREATE CLIENTS

        # =====================================================================================

        logger.info("\n" + "="*80)

        logger.info("DATA PREPARATION - PRIVACY-PRESERVING APPROACH")

        logger.info("="*80)

        

        # Load datasets separately

        all_datasets = []

        

        # Primary dataset (NSCLC-Radiomics)

        image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))

        mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))

        nsclc_data = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

        

        if len(nsclc_data) > 0:

            all_datasets.append(("NSCLC-Radiomics", nsclc_data))

            logger.info(f"Loaded NSCLC-Radiomics: {len(nsclc_data)} samples")

        

        # Additional dataset (Task6 Lung)

        if config.USE_ADDITIONAL_DATA:

            add_image_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "image", "*.png")))

            add_mask_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "masks", "*.png")))

            task6_data = [{"image": img, "label": mask} for img, mask in zip(add_image_files, add_mask_files)]

            

            if len(task6_data) > 0:

                all_datasets.append(("Task6-Lung", task6_data))

                logger.info(f"Loaded Task6-Lung: {len(task6_data)} samples")

        

        # =====================================================================================

        # STEP 1: SPLIT EACH DATASET INTO TRAIN/VAL/TEST FIRST

        # =====================================================================================

        logger.info("\n" + "="*80)

        logger.info("STEP 1: Creating Global Train/Val/Test Splits")

        logger.info("="*80)

        

        dataset_train_data = []

        dataset_val_data = {}

        dataset_test_data = {}

        

        for dataset_name, all_data in all_datasets:

            # Split: 75% train, 15% val, 10% test

            train_data, test_data = train_test_split(all_data, test_size=0.10, random_state=42)

            train_data, val_data = train_test_split(train_data, test_size=0.15/0.90, random_state=42)

            

            dataset_train_data.append((dataset_name, train_data))

            dataset_val_data[dataset_name] = val_data

            dataset_test_data[dataset_name] = test_data

            

            logger.info(f"{dataset_name}:")

            logger.info(f"  Train: {len(train_data)} samples (75%)")

            logger.info(f"  Val:   {len(val_data)} samples (15%)")

            logger.info(f"  Test:  {len(test_data)} samples (10%)")

        

        # =====================================================================================

        # STEP 2: SPLIT TRAINING DATA INTO VIRTUAL CLIENTS

        # =====================================================================================

        logger.info("\n" + "="*80)

        logger.info("STEP 2: Creating Virtual Clients from Training Data")

        logger.info("="*80)

        

        # Calculate median size for splitting

        train_sizes = [len(data) for _, data in dataset_train_data]

        max_client_size = 1375

        logger.info(f"Training data sizes: {train_sizes}")

        logger.info(f"Max client size (median): {max_client_size}")

        

        # Split large clients

        virtual_clients = split_large_clients(dataset_train_data, max_client_size)

        logger.info(f"\nTotal virtual clients: {len(virtual_clients)}")

        

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

        

        # =====================================================================================

        # STEP 3: CREATE CLIENT DATA LOADERS

        # =====================================================================================

        logger.info("\n" + "="*80)

        logger.info("STEP 3: Creating Client Data Loaders")

        logger.info("="*80)

        

        client_train_loaders = []

        client_val_loaders = []

        client_test_loaders = []

        client_sample_sizes = []

        client_names = []

        client_source_datasets = []

        

        for client_name, train_data, source_dataset in virtual_clients:

            # Distribute validation and test data proportionally

            source_val_data = dataset_val_data[source_dataset]

            source_test_data = dataset_test_data[source_dataset]

            

            # Calculate proportion for this client

            total_train_in_dataset = sum(len(data) for name, data, src in virtual_clients if src == source_dataset)

            proportion = len(train_data) / total_train_in_dataset

            

            # Assign proportional validation and test samples

            val_size = max(1, int(len(source_val_data) * proportion))

            test_size = max(1, int(len(source_test_data) * proportion))

            

            client_val_subset = source_val_data[:val_size]

            client_test_subset = source_test_data[:test_size]

            

            # Create datasets

            train_ds = Dataset(data=train_data, transform=train_transforms)

            val_ds = Dataset(data=client_val_subset, transform=val_transforms)

            test_ds = Dataset(data=client_test_subset, transform=val_transforms)

            

            # Create loaders

            train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

            

            client_train_loaders.append(train_loader)

            client_val_loaders.append(val_loader)

            client_test_loaders.append(test_loader)

            client_sample_sizes.append(len(train_data))

            client_names.append(client_name)

            client_source_datasets.append(source_dataset)

            

            logger.info(f"{client_name} (from {source_dataset}):")

            logger.info(f"  Train: {len(train_data)}, Val: {len(client_val_subset)}, Test: {len(client_test_subset)}")



        # =====================================================================================

        # MODEL INITIALIZATION

        # =====================================================================================

        logger.info("\n" + "="*80)

        logger.info("MODEL INITIALIZATION")

        logger.info("="*80)

        

        global_model = StableEnhancedAuraViT(model_config).to(config.DEVICE)

        loss_function = StableLoss()

        

        total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)

        logger.info(f"Total trainable parameters: {total_params:,}")

        

        # =====================================================================================

        # CHECKPOINT LOADING

        # =====================================================================================

        start_round = 0

        best_metric = -1

        best_metric_round = -1

        metric_values = []

        round_loss_values = []

        patience_counter = 0

        

        checkpoint = load_federated_checkpoint(config.CHECKPOINT_PATH, config.DEVICE)

        if checkpoint:

            global_model.load_state_dict(checkpoint['model_state_dict'])

            start_round = checkpoint.get('round', 0)

            best_metric = checkpoint.get('best_metric', -1)

            best_metric_round = checkpoint.get('best_metric_round', -1)

            metric_values = checkpoint.get('metric_values', [])

            round_loss_values = checkpoint.get('round_loss_values', [])

            patience_counter = checkpoint.get('patience_counter', 0)

            logger.info(f"Resuming from round {start_round + 1}")

        

        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        

        # =====================================================================================

        # FEDERATED TRAINING LOOP

        # =====================================================================================

        logger.info("\n" + "="*80)

        logger.info("STARTING PRIVACY-PRESERVING FEDERATED TRAINING")

        logger.info("="*80)

        

        for round_num in range(start_round, config.COMM_ROUNDS):

            round_start_time = datetime.now()

            

            logger.info(f"\n{'='*60}")

            logger.info(f"COMMUNICATION ROUND {round_num + 1}/{config.COMM_ROUNDS}")

            logger.info(f"{'='*60}")

            

            local_models = []

            local_losses = []

            

            # ============================================================

            # TRAINING PHASE: Each client trains locally

            # ============================================================

            for idx, (client_name, train_loader) in enumerate(zip(client_names, client_train_loaders)):

                logger.info(f"\nTraining {client_name} ({idx + 1}/{len(client_names)})")

                

                # Server sends global model to client

                local_model = deepcopy(global_model).to(config.DEVICE)

                

                local_optimizer = torch.optim.AdamW(

                    local_model.parameters(),

                    lr=config.LEARNING_RATE,

                    weight_decay=config.WEIGHT_DECAY,

                    betas=(0.9, 0.999),

                    eps=1e-8

                )

                

                scaler = GradScaler()

                

                # Client trains on local data

                trained_model, client_loss_history = client_update_fedprox(

                    client_id=client_name,

                    model=local_model,

                    global_model=global_model,

                    optimizer=local_optimizer,

                    train_loader=train_loader,

                    epochs=config.CLIENT_EPOCHS_PER_ROUND,

                    device=config.DEVICE,

                    mu=config.FEDPROX_MU,

                    scaler=scaler,

                    loss_function=loss_function,

                    grad_clip_norm=config.GRAD_CLIP_NORM

                )

                

                # Client sends back trained model weights (not data!)

                local_models.append(trained_model)

                local_losses.extend(client_loss_history)

            

            # ============================================================

            # AGGREGATION: Server combines model updates

            # ============================================================

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

            

            avg_round_loss = np.mean(local_losses) if local_losses else float('inf')

            round_loss_values.append(avg_round_loss)

            

            # ============================================================

            # VALIDATION PHASE: Each client validates locally

            # ============================================================

            logger.info("\nFederated validation (per-client)...")

            client_val_metrics = []

            

            for idx, (client_name, val_loader) in enumerate(zip(client_names, client_val_loaders)):

                # Server sends global model to client for validation

                # Client validates locally and returns only metrics

                metrics = client_validation(

                    client_id=client_name,

                    model=global_model,

                    val_loader=val_loader,

                    device=config.DEVICE,

                    loss_function=loss_function,

                    post_pred=post_pred

                )

                client_val_metrics.append(metrics)

                

                logger.info(f"  {client_name}: Dice={metrics['dice']:.4f}, Loss={metrics['loss']:.4f}")

            

            # Server aggregates metrics (weighted by sample size)

            total_samples = sum(m['num_samples'] for m in client_val_metrics)

            weighted_avg_dice = sum(m['dice'] * m['num_samples'] for m in client_val_metrics) / total_samples

            weighted_avg_loss = sum(m['loss'] * m['num_samples'] for m in client_val_metrics) / total_samples

            

            metric_values.append(weighted_avg_dice)

            

            # ============================================================

            # Round Summary

            # ============================================================

            round_time = (datetime.now() - round_start_time).total_seconds()

            logger.info(f"\n{'='*40}")

            logger.info(f"Round {round_num + 1} Summary:")

            logger.info(f"  Training Loss: {avg_round_loss:.4f}")

            logger.info(f"  Weighted Val Dice: {weighted_avg_dice:.4f}")

            logger.info(f"  Weighted Val Loss: {weighted_avg_loss:.4f}")

            logger.info(f"  Best Dice: {best_metric:.4f} (Round {best_metric_round})")

            logger.info(f"  Round Time: {round_time:.1f}s")

            logger.info(f"{'='*40}")

            

            # Save best model

            if weighted_avg_dice > best_metric:

                best_metric = weighted_avg_dice

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

                'client_sample_sizes': client_sample_sizes,

                'client_names': client_names

            }

            save_federated_checkpoint(checkpoint_data, config.CHECKPOINT_PATH)

            

            # Save last model

            torch.save(global_model.state_dict(), config.LAST_MODEL_PATH)

            

            # Early stopping

            if patience_counter >= config.PATIENCE:

                logger.info(f"\n⚠️ Early stopping triggered after {config.PATIENCE} rounds")

                break

            

            # Check for training instability

            if avg_round_loss > config.LOSS_EXPLOSION_THRESHOLD:

                logger.warning("High loss detected - potential training instability")

        

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

        

        axes[0].plot(range(1, len(round_loss_values) + 1), round_loss_values, 'b-', linewidth=2)

        axes[0].set_xlabel('Communication Round')

        axes[0].set_ylabel('Average Training Loss')

        axes[0].set_title('Federated Training Loss')

        axes[0].grid(True, alpha=0.3)

        

        axes[1].plot(range(1, len(metric_values) + 1), metric_values, 'g-', linewidth=2)

        axes[1].axhline(y=best_metric, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_metric:.4f}')

        axes[1].set_xlabel('Communication Round')

        axes[1].set_ylabel('Dice Score')

        axes[1].set_title('Federated Validation Dice Score')

        axes[1].legend()

        axes[1].grid(True, alpha=0.3)

        

        plt.tight_layout()

        plt.savefig('federated_training_curves.png', dpi=300, bbox_inches='tight')

        plt.show()

        

        # =====================================================================================

        # FINAL EVALUATION - PRIVACY-PRESERVING

        # =====================================================================================

        logger.info("\n" + "="*80)

        logger.info("FINAL FEDERATED EVALUATION")

        logger.info("="*80)

        

        # Load best model

        if os.path.exists(config.BEST_MODEL_PATH):

            global_model.load_state_dict(torch.load(config.BEST_MODEL_PATH))

            logger.info("Loaded best model for final evaluation")

        

        # ============================================================

        # Each client tests locally and returns only metrics

        # ============================================================

        client_test_results = []

        

        for idx, (client_name, test_loader) in enumerate(zip(client_names, client_test_loaders)):

            logger.info(f"\n{client_name} evaluating on local test set...")

            

            # Server sends model to client

            # Client evaluates locally and returns only metrics

            result = client_test_evaluation(

                client_id=client_name,

                model=global_model,

                test_loader=test_loader,

                device=config.DEVICE,

                loss_function=loss_function,

                post_pred=post_pred

            )

            

            client_test_results.append(result)

            

            logger.info(f"  Dice: {result['dice']:.4f}")

            logger.info(f"  IoU: {result['iou']:.4f}")

            logger.info(f"  Sensitivity: {result['sensitivity']:.4f}")

            logger.info(f"  Specificity: {result['specificity']:.4f}")

            logger.info(f"  Accuracy: {result['accuracy']:.4f}")

        

        # ============================================================

        # Server aggregates results per original dataset

        # ============================================================

        logger.info("\n" + "="*80)

        logger.info("AGGREGATED RESULTS PER DATASET")

        logger.info("="*80)

        

        # Group results by source dataset

        dataset_results = {}

        for result, source in zip(client_test_results, client_source_datasets):

            if source not in dataset_results:

                dataset_results[source] = []

            dataset_results[source].append(result)

        

        # Compute weighted averages per dataset

        final_dataset_metrics = []

        

        for dataset_name, results in dataset_results.items():

            total_samples = sum(r['num_samples'] for r in results)

            

            weighted_metrics = {

                "dataset": dataset_name,

                "num_clients": len(results),

                "total_samples": total_samples,

                "dice": sum(r['dice'] * r['num_samples'] for r in results) / total_samples,

                "iou": sum(r['iou'] * r['num_samples'] for r in results) / total_samples,

                "sensitivity": sum(r['sensitivity'] * r['num_samples'] for r in results) / total_samples,

                "specificity": sum(r['specificity'] * r['num_samples'] for r in results) / total_samples,

                "accuracy": sum(r['accuracy'] * r['num_samples'] for r in results) / total_samples,

                "loss": sum(r['loss'] * r['num_samples'] for r in results) / total_samples

            }

            

            final_dataset_metrics.append(weighted_metrics)

            

            logger.info(f"\n{dataset_name} (aggregated from {len(results)} clients):")

            logger.info(f"  Total samples: {total_samples}")

            logger.info(f"  Weighted Dice: {weighted_metrics['dice']:.4f}")

            logger.info(f"  Weighted IoU: {weighted_metrics['iou']:.4f}")

            logger.info(f"  Weighted Sensitivity: {weighted_metrics['sensitivity']:.4f}")

            logger.info(f"  Weighted Specificity: {weighted_metrics['specificity']:.4f}")

            logger.info(f"  Weighted Accuracy: {weighted_metrics['accuracy']:.4f}")

        

        # ============================================================

        # Overall global statistics

        # ============================================================

        logger.info("\n" + "="*80)

        logger.info("OVERALL GLOBAL STATISTICS")

        logger.info("="*80)

        

        total_global_samples = sum(m['total_samples'] for m in final_dataset_metrics)

        

        global_metrics = {

            "weighted_dice": sum(m['dice'] * m['total_samples'] for m in final_dataset_metrics) / total_global_samples,

            "weighted_iou": sum(m['iou'] * m['total_samples'] for m in final_dataset_metrics) / total_global_samples,

            "weighted_sensitivity": sum(m['sensitivity'] * m['total_samples'] for m in final_dataset_metrics) / total_global_samples,

            "weighted_specificity": sum(m['specificity'] * m['total_samples'] for m in final_dataset_metrics) / total_global_samples,

            "weighted_accuracy": sum(m['accuracy'] * m['total_samples'] for m in final_dataset_metrics) / total_global_samples,

        }

        

        logger.info(f"📊 Global Weighted Dice: {global_metrics['weighted_dice']:.4f}")

        logger.info(f"📊 Global Weighted IoU: {global_metrics['weighted_iou']:.4f}")

        logger.info(f"📊 Global Weighted Sensitivity: {global_metrics['weighted_sensitivity']:.4f}")

        logger.info(f"📊 Global Weighted Specificity: {global_metrics['weighted_specificity']:.4f}")

        logger.info(f"📊 Global Weighted Accuracy: {global_metrics['weighted_accuracy']:.4f}")

        logger.info(f"📊 Total test samples: {total_global_samples}")

        

        # ============================================================

        # Save results

        # ============================================================

        final_results = {

            "per_client_results": client_test_results,

            "per_dataset_results": final_dataset_metrics,

            "global_metrics": global_metrics,

            "training_info": {

                "best_validation_dice": best_metric,

                "best_round": best_metric_round,

                "total_rounds": len(metric_values),

                "num_clients": len(client_names),

                "client_names": client_names

            }

        }

        

        with open("federated_final_results.json", "w") as f:

            json.dump(final_results, f, indent=4)

        

        logger.info("\n✅ Privacy-preserving federated training completed!")

        logger.info(f"📁 Results saved:")

        logger.info(f"   - Best model: {config.BEST_MODEL_PATH}")

        logger.info(f"   - Last model: {config.LAST_MODEL_PATH}")

        logger.info(f"   - Training curves: federated_training_curves.png")

        logger.info(f"   - Final results: federated_final_results.json")

        logger.info(f"   - Training log: federated_auravit_training.log")

        

    except Exception as e:

        logger.error(f"Training failed with error: {str(e)}")

        import traceback

        logger.error(traceback.format_exc())

        raise



if __name__ == "__main__":

    main()