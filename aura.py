import os
import glob
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
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils import set_determinism
from tqdm import tqdm

# For reproducibility
set_determinism(seed=42)

#BLOCKS 

class ResBlock(nn.Module):
    """
    A Residual Block, the core of SegResNet.
    This block helps in training deeper networks by allowing gradients to flow more easily.
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layers(x)
        x = x + shortcut
        return self.relu(x)

class DeconvBlock(nn.Module):
    """
    Standard Transposed Convolution block for upsampling.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)

class AtrousConv(nn.Module):
    """
    A single Atrous Convolution block, used within the ASPP module.
    """
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.atrous_conv(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module from DeepLabV3+.
    This captures multi-scale context by using parallel atrous convolutions with different rates.
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        x_1x1 = self.conv_1x1(x)
        x_atrous = [block(x) for block in self.atrous_blocks]
        x_global = self.global_avg_pool(x)
        x_global = F.interpolate(x_global, size=size, mode='bilinear', align_corners=False)

        x_cat = torch.cat([x_1x1] + x_atrous + [x_global], dim=1)
        return self.output_conv(x_cat)

# ======================================================================================
# THE NEW HYBRID MODEL: AuraViT
# ======================================================================================

class AuraViT(nn.Module):
    """
    AuraViT: Atrous UNETR-style Residual VAE with Vision Transformer.
    This model combines:
    1. A Vision Transformer (ViT) encoder for global context (from UNETR).
    2. An ASPP module for multi-scale feature extraction (from DeepLabV3+).
    3. A Segmentation Decoder with Residual Blocks (from SegResNet).
    4. A VAE pathway for regularization and data-efficient learning (from SegResNetVAE).
    """
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # --- ViT ENCODER (from UNETR) ---
        self.patch_embed = nn.Linear(cf["patch_size"]*cf["patch_size"]*cf["num_channels"], cf["hidden_dim"])
        self.positions = torch.arange(start=0, end=cf["num_patches"], step=1, dtype=torch.long)
        self.pos_embed = nn.Embedding(cf["num_patches"], cf["hidden_dim"])
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"], nhead=cf["num_heads"], dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"], activation=F.gelu, batch_first=True
            ) for _ in range(cf["num_layers"])
        ])

        # --- VAE HEAD ---
        self.fc_mu = nn.Linear(cf["hidden_dim"], cf["hidden_dim"])
        self.fc_log_var = nn.Linear(cf["hidden_dim"], cf["hidden_dim"])

        # --- ASPP MODULE (from DeepLabV3+) ---
        self.aspp = ASPP(cf["hidden_dim"], cf["hidden_dim"], rates=[6, 12, 18])

        # --- SEGMENTATION DECODER (with ResBlocks) ---
        self.seg_d1 = DeconvBlock(cf["hidden_dim"], 512)
        self.seg_s1 = nn.Sequential(DeconvBlock(cf["hidden_dim"], 512), ResBlock(512, 512))
        self.seg_c1 = nn.Sequential(ResBlock(512+512, 512), ResBlock(512, 512))

        self.seg_d2 = DeconvBlock(512, 256)
        self.seg_s2 = nn.Sequential(DeconvBlock(cf["hidden_dim"], 256), ResBlock(256, 256), DeconvBlock(256, 256), ResBlock(256, 256))
        self.seg_c2 = nn.Sequential(ResBlock(256+256, 256), ResBlock(256, 256))

        self.seg_d3 = DeconvBlock(256, 128)
        self.seg_s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128), ResBlock(128, 128), DeconvBlock(128, 128),
            ResBlock(128, 128), DeconvBlock(128, 128), ResBlock(128, 128)
        )
        self.seg_c3 = nn.Sequential(ResBlock(128+128, 128), ResBlock(128, 128))

        self.seg_d4 = DeconvBlock(128, 64)
        self.seg_s4 = nn.Sequential(ResBlock(cf["num_channels"], 64), ResBlock(64, 64))
        self.seg_c4 = nn.Sequential(ResBlock(64+64, 64), ResBlock(64, 64))

        self.seg_output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        # --- VAE DECODER ---
        self.vae_decoder = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 512),
            ResBlock(512, 256),
            DeconvBlock(256, 256),
            ResBlock(256, 128),
            DeconvBlock(128, 128),
            ResBlock(128, 64),
            DeconvBlock(64, 64),
            nn.Conv2d(64, cf["num_channels"], kernel_size=1, padding=0) # Reconstruct original channels
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs):
        # 1. ViT Encoder
        p = self.cf["patch_size"]
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf["num_patches"], -1)
        patch_embed = self.patch_embed(patches)

        positions = self.positions.to(inputs.device)
        pos_embed = self.pos_embed(positions)
        x = patch_embed + pos_embed

        skip_connection_index = [2, 5, 8, 11]
        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if i in skip_connection_index:
                skip_connections.append(x)
        z3, z6, z9, z12_features = skip_connections

        # 2. VAE Head and Reparameterization
        mu = self.fc_mu(z12_features)
        log_var = self.fc_log_var(z12_features)
        z12_latent = self.reparameterize(mu, log_var)

        # Reshape all feature maps
        batch, num_patches, hidden_dim = z12_latent.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_latent.permute(0, 2, 1).contiguous().view(shape)

        # 3. ASPP Module
        aspp_out = self.aspp(z12_reshaped)

        # 4. Segmentation Decoder Path
        x_seg = self.seg_d1(aspp_out); s = self.seg_s1(z9); x_seg = torch.cat([x_seg, s], dim=1); x_seg = self.seg_c1(x_seg)
        x_seg = self.seg_d2(x_seg); s = self.seg_s2(z6); x_seg = torch.cat([x_seg, s], dim=1); x_seg = self.seg_c2(x_seg)
        x_seg = self.seg_d3(x_seg); s = self.seg_s3(z3); x_seg = torch.cat([x_seg, s], dim=1); x_seg = self.seg_c3(x_seg)
        x_seg = self.seg_d4(x_seg); s = self.seg_s4(z0); x_seg = torch.cat([x_seg, s], dim=1); x_seg = self.seg_c4(x_seg)
        seg_output = self.seg_output(x_seg)

        # 5. VAE Decoder Path
        recon_output = self.vae_decoder(z12_reshaped)

        return seg_output, recon_output, mu, log_var

# ======================================================================================
# HYBRID LOSS FUNCTION FOR AuraViT
# ======================================================================================
class AuraViTLoss(nn.Module):
    """
    A composite loss function for the AuraViT model.
    It combines:
    1. DiceCELoss for the segmentation task.
    2. MSELoss for the VAE reconstruction task.
    3. KL Divergence loss to regularize the VAE's latent space.
    """
    def __init__(self, seg_weight=1.0, recon_weight=0.1, kl_weight=0.01):
        super().__init__()
        self.seg_loss = DiceCELoss(to_onehot_y=False, sigmoid=True)
        self.recon_loss = nn.MSELoss()
        self.seg_weight = seg_weight
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def forward(self, seg_preds, seg_labels, recon_preds, recon_labels, mu, log_var):
        # Segmentation Loss
        segmentation_loss = self.seg_loss(seg_preds, seg_labels)

        # Reconstruction Loss
        reconstruction_loss = self.recon_loss(recon_preds, recon_labels)

        # KL Divergence Loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / seg_preds.shape[0] # Normalize by batch size

        # Total Weighted Loss
        total_loss = (self.seg_weight * segmentation_loss +
                      self.recon_weight * reconstruction_loss +
                      self.kl_weight * kl_loss)

        return total_loss, segmentation_loss, reconstruction_loss, kl_loss


def main():
    report_file = open("training_report_AuraViT_NSCLC.txt", "w", buffering=1)
    try:
        # ======================================================================================
        # CONFIGURATION
        # ======================================================================================
        class TrainingConfig:
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            CHECKPOINT_PATH = "training_checkpoint_AuraViT_NSCLC.pth"
            BEST_MODEL_PATH = "best_AuraViT_model_NSCLC.pth"
            BATCH_SIZE = 8
            LEARNING_RATE = 1e-6
            MAX_EPOCHS = 125
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_config = {
            "image_size": 256, "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072,
            "num_heads": 12, "dropout_rate": 0.1, "patch_size": 16, "num_channels": 1,
        }
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
        config = TrainingConfig()

        print(f"Using device: {config.DEVICE}")
        report_file.write(f"Using device: {config.DEVICE}\n")

        # ======================================================================================
        # DATA PREPARATION (No changes needed here)
        # ======================================================================================
        image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
        mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
        data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

        train_files, test_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)

        data_summary = (
            f"Total samples: {len(data_dicts)}\n"
            f"Training samples: {len(train_files)}\n"
            f"Validation samples: {len(val_files)}\n"
            f"Testing samples: {len(test_files)}\n"
        )
        print(data_summary)
        report_file.write(data_summary)

        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandAffined(
                keys=['image', 'label'], prob=0.5, translate_range=(10, 10),
                rotate_range=(np.pi / 36, np.pi / 36), scale_range=(0.05, 0.05),
                mode=('bilinear', 'nearest'),
            ),
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
        # UPDATED MODEL, LOSS, OPTIMIZER
        # ======================================================================================
        model = AuraViT(model_config).to(config.DEVICE)
        loss_function = AuraViTLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=4)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # ======================================================================================
        # CHECKPOINT LOADING & RESUME LOGIC (No changes needed here)
        # ======================================================================================
        start_epoch = 0; best_metric = -1; best_metric_epoch = -1
        epoch_loss_values = []; metric_values = []

        if os.path.exists(config.CHECKPOINT_PATH):
            checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_metric = checkpoint.get('best_metric', best_metric)
            epoch_loss_values = checkpoint.get('epoch_loss_values', epoch_loss_values)
            metric_values = checkpoint.get('metric_values', metric_values)
            resume_msg = f"\n✅ Checkpoint found! Resuming training from epoch {start_epoch}.\n"
            print(resume_msg); report_file.write(resume_msg)
        else:
            start_msg = "\nℹ️ No checkpoint found. Starting training from scratch.\n"
            print(start_msg); report_file.write(start_msg)

        # ======================================================================================
        # UPDATED TRAINING AND VALIDATION LOOP
        # ======================================================================================
        report_file.write("="*50 + "\n" + "AuraViT TRAINING LOG\n" + "="*50 + "\n")
        for epoch in range(start_epoch, config.MAX_EPOCHS):
            model.train(); epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Training]", unit="batch")
            for batch_data in progress_bar:
                inputs, labels = batch_data["image"].to(config.DEVICE), batch_data["label"].to(config.DEVICE)
                optimizer.zero_grad()

                # --- NEW MODEL FORWARD PASS ---
                seg_outputs, recon_outputs, mu, log_var = model(inputs)
                total_loss, seg_loss, recon_loss, kl_loss = loss_function(
                    seg_outputs, labels, recon_outputs, inputs, mu, log_var
                )

                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                progress_bar.set_postfix({
                    "Total Loss": f"{total_loss.item():.4f}",
                    "Seg": f"{seg_loss.item():.4f}",
                    "Recon": f"{recon_loss.item():.4f}",
                    "KL": f"{kl_loss.item():.4f}"
                })

            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_loss_values.append(avg_epoch_loss)

            model.eval()
            with torch.no_grad():
                dice_metric.reset()
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(config.DEVICE), val_data["label"].to(config.DEVICE)
                    
                    # --- VALIDATION FORWARD PASS (only need seg_outputs) ---
                    seg_outputs, _, _, _ = model(val_inputs)
                    
                    val_outputs_post = [post_pred(i) for i in decollate_batch(seg_outputs)]
                    dice_metric(y_pred=val_outputs_post, y=val_labels)
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)

            scheduler.step(metric)
            summary = f"Epoch {epoch + 1} Summary | Avg Total Loss: {avg_epoch_loss:.4f} | Val Dice: {metric:.4f}"
            tqdm.write(summary); report_file.write(summary + "\n")

            if metric > best_metric:
                best_metric = metric; best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                save_msg = f"🏆 New best model saved! Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}"
                tqdm.write(save_msg); report_file.write(save_msg + "\n")

            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'best_metric': best_metric,
                'epoch_loss_values': epoch_loss_values, 'metric_values': metric_values
            }, config.CHECKPOINT_PATH)

        training_summary = f"\n🏁 Training finished. Best Dice score of {best_metric:.4f} at epoch {best_metric_epoch}."
        print(training_summary); report_file.write(training_summary + "\n\n")

        # ======================================================================================
        # PLOTTING (No changes needed here, but will plot total loss)
        # ======================================================================================
        print("\n📈 Plotting training curves...")
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1); plt.title("Epoch Average Total Loss")
        plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.subplot(1, 2, 2); plt.title("Validation Mean Dice")
        plt.plot(range(1, len(metric_values) + 1), metric_values); plt.xlabel("Epoch"); plt.ylabel("Mean Dice")
        plt.savefig("training_validation_curves_AuraViT.png"); plt.show()

        # ======================================================================================
        # UPDATED FINAL EVALUATION ON TEST SET
        # ======================================================================================
        print("\n🧪 Running final evaluation on the test set...")
        test_ds = Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

        model.load_state_dict(torch.load(config.BEST_MODEL_PATH)); model.eval()
        
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        cm_metric = ConfusionMatrixMetric(
            include_background=False, metric_name=["sensitivity", "specificity", "accuracy"], reduction="mean"
        )

        with torch.no_grad():
            for test_data in tqdm(test_loader, desc="Testing"):
                test_inputs, test_labels = test_data["image"].to(config.DEVICE), test_data["label"].to(config.DEVICE)
                
                # --- TEST FORWARD PASS ---
                test_seg_outputs, _, _, _ = model(test_inputs)
                
                test_outputs_post = [post_pred(i) for i in decollate_batch(test_seg_outputs)]
                
                dice_metric(y_pred=test_outputs_post, y=test_labels)
                cm_metric(y_pred=test_outputs_post, y=test_labels)

        mean_dice_test = dice_metric.aggregate().item()
        cm_value = cm_metric.aggregate()

        sensitivity = cm_value[0].item()
        specificity = cm_value[1].item()
        accuracy = cm_value[2].item()
        iou_test = mean_dice_test / (2 - mean_dice_test) if mean_dice_test > 0 else 0
        
        metrics_summary = (
            f"\n📊 Final Test Metrics (AuraViT):\n"
            f"  - Mean Dice Score: {mean_dice_test:.4f}\n"
            f"  - Intersection over Union (IoU): {iou_test:.4f}\n"
            f"  - Sensitivity (Recall): {sensitivity:.4f}\n"
            f"  - Specificity: {specificity:.4f}\n"
            f"  - Accuracy: {accuracy:.4f}\n"
        )
        print(metrics_summary); report_file.write(metrics_summary)
        
        # ======================================================================================
        # UPDATED VISUALIZATION
        # ======================================================================================
        print("\n🎨 Visualizing predictions on test samples...")
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
        model.to(config.DEVICE)
        model.eval()

        num_visualizations = 10
        fig, axes = plt.subplots(num_visualizations, 3, figsize=(12, 4 * num_visualizations))
        plt.suptitle("AuraViT Predictions vs. Ground Truth", fontsize=16)

        with torch.no_grad():
            for i in range(num_visualizations):
                if i >= len(test_ds): break
                test_sample = test_ds[i]
                image_tensor = test_sample["image"].unsqueeze(0).to(config.DEVICE)
                label_np = test_sample["label"].cpu().numpy().squeeze()

                pred_seg, _, _, _ = model(image_tensor)
                pred_mask = torch.sigmoid(pred_seg).detach().cpu().numpy().squeeze()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

                axes[i, 0].imshow(test_sample["image"].cpu().numpy().squeeze(), cmap="gray")
                axes[i, 0].set_title(f"Input Image #{i+1}"); axes[i, 0].axis("off")

                axes[i, 1].imshow(label_np, cmap="hot")
                axes[i, 1].set_title("Ground Truth Mask"); axes[i, 1].axis("off")

                axes[i, 2].imshow(test_sample["image"].cpu().numpy().squeeze(), cmap="gray")
                axes[i, 2].imshow(pred_mask, cmap="Blues", alpha=0.4)
                axes[i, 2].set_title("Predicted Mask"); axes[i, 2].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("test_set_visualizations_AuraViT.png")
        plt.show()

    finally:
        print(f"\nTraining report saved to {report_file.name}")
        report_file.close()

if __name__ == "__main__":
    main()
