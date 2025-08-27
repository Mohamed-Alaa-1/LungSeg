"""
Advanced Federated Learning for Lung Segmentation
Implements a 5-client system by splitting the larger dataset into four virtual clients.
Uses FedProx and Adaptive Averaging to combat client drift.
"""
import os
import glob
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
    RandRotate90d, ResizeWithPadOrCropd, ScaleIntensityRanged, EnsureTyped, Activations
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils import set_determinism
from tqdm import tqdm
from copy import deepcopy

# For reproducibility
set_determinism(seed=42)

# --- MODEL DEFINITION (UNETR_2D and Blocks) ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        return self.deconv(x)

class UNETR_2D(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.cf = cf
        self.patch_embed = nn.Linear(cf["patch_size"]*cf["patch_size"]*cf["num_channels"], cf["hidden_dim"])
        self.register_buffer('positions', torch.arange(start=0, end=cf["num_patches"], step=1, dtype=torch.long))
        self.pos_embed = nn.Embedding(cf["num_patches"], cf["hidden_dim"])
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"], nhead=cf["num_heads"], dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"], activation=F.gelu, batch_first=True
            ) for _ in range(cf["num_layers"])
        ])
        self.d1 = DeconvBlock(cf["hidden_dim"], 512)
        self.s1 = nn.Sequential(DeconvBlock(cf["hidden_dim"], 512), ConvBlock(512, 512))
        self.c1 = nn.Sequential(ConvBlock(512+512, 512), ConvBlock(512, 512))
        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(DeconvBlock(cf["hidden_dim"], 256), ConvBlock(256, 256), DeconvBlock(256, 256), ConvBlock(256, 256))
        self.c2 = nn.Sequential(ConvBlock(256+256, 256), ConvBlock(256, 256))
        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128), ConvBlock(128, 128), DeconvBlock(128, 128),
            ConvBlock(128, 128), DeconvBlock(128, 128), ConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(ConvBlock(128+128, 128), ConvBlock(128, 128))
        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(ConvBlock(cf["num_channels"], 64), ConvBlock(64, 64))
        self.c4 = nn.Sequential(ConvBlock(64+64, 64), ConvBlock(64, 64))
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        p = self.cf["patch_size"]
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf["num_patches"], -1)
        patch_embed = self.patch_embed(patches)
        pos_embed = self.pos_embed(self.positions)
        x = patch_embed + pos_embed
        skip_connection_index = [2, 5, 8, 11]
        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if i in skip_connection_index:
                skip_connections.append(x)
        z3, z6, z9, z12 = skip_connections
        batch, num_patches, hidden_dim = z12.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)
        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12 = z12.permute(0, 2, 1).contiguous().view(shape)
        x = self.d1(z12); s = self.s1(z9); x = torch.cat([x, s], dim=1); x = self.c1(x)
        x = self.d2(x); s = self.s2(z6); x = torch.cat([x, s], dim=1); x = self.c2(x)
        x = self.d3(x); s = self.s3(z3); x = torch.cat([x, s], dim=1); x = self.c3(x)
        x = self.d4(x); s = self.s4(z0); x = torch.cat([x, s], dim=1); x = self.c4(x)
        return self.output(x)

# --- ADVANCED FEDERATED LEARNING FUNCTIONS ---
def client_update_fedprox(client_id, model, global_model, optimizer, train_loader, epochs, device, mu):
    model.train()
    global_params = {name: param.clone() for name, param in global_model.named_parameters()}
    
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Client {client_id} - Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for batch_data in progress_bar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            primary_loss = DiceCELoss(to_onehot_y=False, sigmoid=True)(outputs, labels)
            
            proximal_term = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    proximal_term += torch.sum((param - global_params[name]) ** 2)
            
            total_loss = primary_loss + (mu / 2) * proximal_term
            total_loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Prox": f"{(mu / 2) * proximal_term.item():.4f}"})
    
    return model

def adaptive_weighted_average(w_states, s, global_model, local_models, alpha):
    total_samples = sum(s)
    w_avg = deepcopy(w_states[0])
    
    similarity_weights = []
    for local_model in local_models:
        similarity = 0.0
        param_count = 0
        for (name1, param1), (name2, param2) in zip(global_model.named_parameters(), local_model.named_parameters()):
            if param1.dtype.is_floating_point:
                cos_sim = F.cosine_similarity(param1.detach().view(-1), param2.detach().view(-1), dim=0)
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
    
    return w_avg

# --- MAIN EXECUTION ---
def main():
    report_file = open("federated_training_5_clients.txt", "w", buffering=1)
    try:
        # --- CONFIGURATION ---
        class TrainingConfig:
            CLIENT1_DATA_DIR = "/teamspace/studios/this_studio/lung/processed_task6_lung"
            CLIENT2_DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            PRETRAINED_MODEL_PATH = "/teamspace/studios/this_studio/lung/best_custom_unetr_modelv1dontuse.pth"
            CHECKPOINT_PATH = "federated_5_clients_checkpoint.pth"
            BEST_MODEL_PATH = "best_federated_5_clients_model.pth"
            LAST_MODEL_PATH = "last_federated_5_clients_model.pth"
            BATCH_SIZE = 8
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Parameters for 5 clients
            COMM_ROUNDS = 100
            LEARNING_RATE = 5e-5
            # Client 1 (unique data) gets more epochs, others get fewer.
            CLIENT_EPOCHS = [15, 8, 8, 8, 8] 
            FEDPROX_MU = 0.01
            ADAPTIVE_ALPHA = 0.3

        config = TrainingConfig()
        model_config = {
            "image_size": 256, "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072,
            "num_heads": 12, "dropout_rate": 0.1, "patch_size": 16, "num_channels": 1,
        }
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2
        
        print(f"Using device: {config.DEVICE}")
        report_file.write(f"Using device: {config.DEVICE}\n")
        
        # --- DATA SETUP FOR 5 CLIENTS ---
        print("\nSetting up data for 5 clients...")
        client_train_loaders = []
        all_val_ds = []
        all_test_ds = []
        client_sample_sizes = []

        # Define transforms once
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandAffined(
                keys=['image', 'label'], prob=0.5, translate_range=(10, 10),
                rotate_range=(np.pi / 36, np.pi / 36), scale_range=(0.05, 0.05),
                mode=('bilinear', 'nearest')),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ])
        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ])

        # Client 1 (task6_lung)
        c1_image_files = sorted(glob.glob(os.path.join(config.CLIENT1_DATA_DIR, "image", "*.png")))
        c1_mask_files = sorted(glob.glob(os.path.join(config.CLIENT1_DATA_DIR, "masks", "*.png")))
        c1_data_dicts = [{"image": img, "label": mask} for img, mask in zip(c1_image_files, c1_mask_files)]
        c1_train_files, c1_test_files = train_test_split(c1_data_dicts, test_size=0.2, random_state=42)
        c1_train_files, c1_val_files = train_test_split(c1_train_files, test_size=0.125, random_state=42)
        
        client_train_loaders.append(DataLoader(Dataset(data=c1_train_files, transform=train_transforms), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2))
        all_val_ds.append(Dataset(data=c1_val_files, transform=val_transforms))
        all_test_ds.append(Dataset(data=c1_test_files, transform=val_transforms))
        client_sample_sizes.append(len(c1_train_files))
        
        # Clients 2-5 (Split from NSCLC)
        nsclc_image_files = sorted(glob.glob(os.path.join(config.CLIENT2_DATA_DIR, "images", "*.png")))
        nsclc_mask_files = sorted(glob.glob(os.path.join(config.CLIENT2_DATA_DIR, "masks", "*.png")))
        nsclc_data_dicts = [{"image": img, "label": mask} for img, mask in zip(nsclc_image_files, nsclc_mask_files)]
        
        # Shuffle before splitting to ensure random distribution
        np.random.shuffle(nsclc_data_dicts)
        
        # Split into 4 chunks
        nsclc_splits = np.array_split(nsclc_data_dicts, 4)

        for split in nsclc_splits:
            client_data_dicts = split.tolist()
            train_files, test_files = train_test_split(client_data_dicts, test_size=0.2, random_state=42)
            train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)
            
            client_train_loaders.append(DataLoader(Dataset(data=train_files, transform=train_transforms), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2))
            all_val_ds.append(Dataset(data=val_files, transform=val_transforms))
            all_test_ds.append(Dataset(data=test_files, transform=val_transforms))
            client_sample_sizes.append(len(train_files))
            
        # Create global validation and test sets
        val_ds = ConcatDataset(all_val_ds); test_ds = ConcatDataset(all_test_ds)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
        
        data_summary = f"Total Clients: {len(client_train_loaders)}\n"
        for i, size in enumerate(client_sample_sizes):
            data_summary += f"  - Client {i+1} samples: {size}\n"
        data_summary += (
            f"Total validation samples: {len(val_ds)}\n"
            f"Total testing samples: {len(test_ds)}\n")
        print(data_summary); report_file.write(data_summary + "\n")

        # --- MODEL INITIALIZATION AND LOADING ---
        global_model = UNETR_2D(model_config).to(config.DEVICE)
        
        start_round = 0; best_metric = -1; best_metric_round = -1; metric_values = []
        if os.path.exists(config.CHECKPOINT_PATH):
            checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
            global_model.load_state_dict(checkpoint['model_state_dict'])
            start_round = checkpoint.get('round', 0)
            best_metric = checkpoint.get('best_metric', -1)
            metric_values = checkpoint.get('metric_values', [])
            print(f"\n✅ Checkpoint found! Resuming from round {start_round + 1}.\n")
        elif os.path.exists(config.PRETRAINED_MODEL_PATH):
            state_dict = torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE)
            global_model.load_state_dict(state_dict, strict=False)
            print(f"\n✅ Pre-trained model found! Starting with weights from {config.PRETRAINED_MODEL_PATH}.\n")
        else:
            print("\nℹ️ No checkpoint or pre-trained model. Starting from scratch.\n")

        # --- FEDERATED TRAINING LOOP ---
        report_file.write("="*50 + "\nFEDERATED TRAINING LOG\n" + "="*50 + "\n")
        for round_num in range(start_round, config.COMM_ROUNDS):
            print(f"\n--- Communication Round {round_num + 1}/{config.COMM_ROUNDS} ---")
            local_models = []
            
            for client_id, train_loader in enumerate(client_train_loaders):
                local_model = deepcopy(global_model).to(config.DEVICE)
                
                # Use different LRs for the unique client vs the others
                lr_multiplier = 1.2 if client_id == 0 else 1.0
                client_lr = config.LEARNING_RATE * lr_multiplier
                local_optimizer = torch.optim.AdamW(local_model.parameters(), lr=client_lr, weight_decay=1e-5)
                client_epochs = config.CLIENT_EPOCHS[client_id]
                
                trained_local_model = client_update_fedprox(
                    client_id + 1, local_model, global_model, local_optimizer, 
                    train_loader, client_epochs, config.DEVICE, config.FEDPROX_MU)
                local_models.append(trained_local_model)

            local_weights_states = [model.state_dict() for model in local_models]
            
            global_weights = adaptive_weighted_average(
                local_weights_states, client_sample_sizes, global_model, local_models, config.ADAPTIVE_ALPHA)
            global_model.load_state_dict(global_weights)

            # --- Global Model Validation ---
            global_model.eval()
            with torch.no_grad():
                dice_metric = DiceMetric(include_background=True, reduction="mean")
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(config.DEVICE), val_data["label"].to(config.DEVICE)
                    val_outputs = global_model(val_inputs)
                    val_outputs = [Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)

            summary = f"Round {round_num + 1} Summary | Global Val Dice: {metric:.4f}"
            tqdm.write(summary); report_file.write(summary + "\n")

            # --- Save Best Model and Checkpoint ---
            if metric > best_metric:
                best_metric = metric; best_metric_round = round_num + 1
                torch.save(global_model.state_dict(), config.BEST_MODEL_PATH)
                save_msg = f"🏆 New best model! Dice: {best_metric:.4f} at round {best_metric_round}"
                tqdm.write(save_msg); report_file.write(save_msg + "\n")

            torch.save({
                'round': round_num + 1, 'model_state_dict': global_model.state_dict(), 
                'best_metric': best_metric, 'metric_values': metric_values
            }, config.CHECKPOINT_PATH)
            
        # --- FINAL ACTIONS ---
        training_summary = f"\n🏁 Training finished. Best Dice: {best_metric:.4f} at round {best_metric_round}."
        print(training_summary); report_file.write(training_summary + "\n")
        torch.save(global_model.state_dict(), config.LAST_MODEL_PATH)
        print(f"Saved last model state to {config.LAST_MODEL_PATH}")

        # --- PLOTTING ---
        print("\n📈 Plotting training curves...")
        plt.figure("train", (12, 6))
        plt.title("Global Validation Mean Dice per Round")
        plt.plot(range(1, len(metric_values) + 1), metric_values)
        plt.xlabel("Communication Round")
        plt.ylabel("Mean Dice")
        plt.grid(True)
        plt.savefig("federated_5_clients_training_curves.png")
        plt.show()

        # --- FINAL EVALUATION ---
        print("\n🧪 Running final evaluation on the test set...")
        global_model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
        global_model.eval()

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        cm_metric = ConfusionMatrixMetric(
            include_background=False, metric_name=["sensitivity", "specificity", "accuracy"], reduction="mean"
        )
        with torch.no_grad():
            for test_data in tqdm(test_loader, desc="Testing Global Model"):
                test_inputs, test_labels = test_data["image"].to(config.DEVICE), test_data["label"].to(config.DEVICE)
                test_outputs = global_model(test_inputs)
                test_outputs = [Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])(i) for i in decollate_batch(test_outputs)]
                dice_metric(y_pred=test_outputs, y=test_labels)
                cm_metric(y_pred=test_outputs, y=test_labels)

        mean_dice_test = dice_metric.aggregate().item()
        cm_value = cm_metric.aggregate()
        sensitivity = cm_value[0].item(); specificity = cm_value[1].item(); accuracy = cm_value[2].item()
        iou_test = mean_dice_test / (2 - mean_dice_test) if mean_dice_test > 0 else 0
        
        metrics_summary = (
            f"\n📊 Final Global Model Test Metrics:\n"
            f"  - Mean Dice Score: {mean_dice_test:.4f}\n"
            f"  - Intersection over Union (IoU): {iou_test:.4f}\n"
            f"  - Sensitivity (Recall): {sensitivity:.4f}\n"
            f"  - Specificity: {specificity:.4f}\n"
            f"  - Accuracy: {accuracy:.4f}\n"
        )
        print(metrics_summary); report_file.write(metrics_summary)
        
        # --- VISUALIZATION ---
        print("\n📸 Visualizing predictions on combined test samples...")
        global_model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
        global_model.to(config.DEVICE)
        global_model.eval()

        num_visualizations = min(10, len(test_ds))
        fig, axes = plt.subplots(num_visualizations, 3, figsize=(12, 4 * num_visualizations))
        plt.suptitle("Global Model Predictions vs. Ground Truth", fontsize=16)

        with torch.no_grad():
            for i in range(num_visualizations):
                test_sample = test_ds[i]
                image = test_sample["image"].unsqueeze(0).to(config.DEVICE)
                label = test_sample["label"].cpu().numpy().squeeze()

                pred = global_model(image)
                pred_mask = torch.sigmoid(pred).detach().cpu().numpy().squeeze()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

                axes[i, 0].imshow(test_sample["image"].cpu().numpy().squeeze(), cmap="gray")
                axes[i, 0].set_title(f"Input Image #{i+1}"); axes[i, 0].axis("off")
                axes[i, 1].imshow(label, cmap="hot")
                axes[i, 1].set_title("Ground Truth Mask"); axes[i, 1].axis("off")
                axes[i, 2].imshow(test_sample["image"].cpu().numpy().squeeze(), cmap="gray")
                axes[i, 2].imshow(pred_mask, cmap="Blues", alpha=0.4)
                axes[i, 2].set_title("Predicted Mask"); axes[i, 2].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("federated_5_clients_test_visualizations.png")
        plt.show()

    finally:
        if 'report_file' in locals() and not report_file.closed:
            print(f"\nTraining report saved to {report_file.name}")
            report_file.close()

if __name__ == "__main__":
    main()