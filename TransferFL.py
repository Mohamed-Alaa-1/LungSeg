
import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from datetime import datetime
import json
from tqdm import tqdm
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, RandAffined,
    RandRotate90d, ResizeWithPadOrCropd, ScaleIntensityRanged, EnsureTyped,
    Activations, RandGaussianNoised, RandScaleIntensityd, RandFlipd,
    NormalizeIntensityd, ScaleIntensityRangePercentilesd,
)
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils import set_determinism

# Import building blocks and helpers from LAURAFED
try:
    from LAURAFED import (
        LightweightAuraViT, StableLoss, MultiDatasetWrapper,
        split_data_by_patient, create_global_test_set_first,
        split_large_clients_adaptive, select_clients_for_round,
        adaptive_weighted_average_improved, weighted_average, robust_aggregation,
        client_update_enhanced, check_predictions_distribution,
        validate_data_quality, count_parameters, extract_patient_id,
        AdaptiveFederatedLRScheduler, ClientDriftDetector, determine_max_client_size,
        compute_dynamic_epochs, setup_logging, CheckpointManager, FederatedTrainingMonitor
    )
except ImportError:
    print("Error importing from LAURAFED.py. Make sure it exists in the same directory.")
    raise

# Setup logging
logger = setup_logging("transfer_federated_training.log")

# =====================================================================================
# TRANSFER LEARNING HELPERS
# =====================================================================================

def load_pretrained_model(model, path, device, strict=False):
    """Load pre-trained weights with robust error handling."""
    if not os.path.exists(path):
        logger.warning(f"Pre-trained model not found at {path}. initializing randomly.")
        return model, False

    try:
        logger.info(f"Loading pre-trained weights from {path}")
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Try loading
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Successfully loaded weights (strict=True)")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}. Retrying with strict=False")
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded weights (strict=False)")
            
        return model, True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return model, False

def freeze_encoder(model, freeze_bn=True):
    """
    Freeze the encoder part of LightweightAuraViT.
    Target layers: patch_embed, pos_embed, trans_encoder_layers
    """
    logger.info("Freezing Encoder layers (ViT backbone)...")
    
    frozen_layers = 0
    total_layers = 0
    
    # Layers to freeze
    encoder_prefixes = ['patch_embed', 'pos_embed', 'trans_encoder_layers', 'skip_norms']
    
    for name, param in model.named_parameters():
        total_layers += 1
        if any(prefix in name for prefix in encoder_prefixes):
            param.requires_grad = False
            frozen_layers += 1
            
    # Also optionally freeze BatchNorm stats in encoder
    if freeze_bn:
        for name, module in model.named_modules():
            if any(prefix in name for prefix in encoder_prefixes):
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    module.eval()
    
    logger.info(f"Frozen {frozen_layers}/{total_layers} parameters")
    return model

def unfreeze_all(model):
    """Unfreeze all parameters for fine-tuning."""
    logger.info("Unfreezing all layers...")
    for param in model.parameters():
        param.requires_grad = True
    return model

# =====================================================================================
# MAIN TRANSFER FEDERATED TRAINING
# =====================================================================================

def main():
    try:
        class TransferConfig:
            # Paths
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            ADDITIONAL_DATA_DIR = "/teamspace/studios/this_studio/lung/processed_task6_lung"
            USE_ADDITIONAL_DATA = os.path.exists(ADDITIONAL_DATA_DIR)
            
            # TRANSFER LEARNING PARAMS
            PRETRAINED_PATH = "lightweight_best_AuraViT_model.pth" # Path to your source model
            FREEZE_ENCODER = True      # Freeze ViT encoder initially?
            UNFREEZE_ROUND = 50        # Round to unfreeze encoder (set > COMM_ROUNDS to never unfreeze)
            
            CHECKPOINT_DIR = "checkpoints_transfer"
            
            # FL Params
            COMM_ROUNDS = 200
            MIN_EPOCHS = 5             # Higher epochs for fine-tuning
            MAX_EPOCHS = 25
            DYNAMIC_EPOCHS = True
            
            # Training
            BATCH_SIZE = 8
            MIN_BATCH_SIZE = 2
            LEARNING_RATE = 5e-4       # Slightly lower LR for fine-tuning
            MIN_LEARNING_RATE = 1e-6
            WARMUP_ROUNDS = 5
            
            # Client Selection
            MIN_CLIENT_PARTICIPATION = 1.0
            MAX_CLIENT_PARTICIPATION = 1.0
            
            # Aggregation
            AGGREGATION_STRATEGY = 'adaptive_weighted'
            ADAPTIVE_ALPHA = 0.4       # Higher alpha: trust performance more
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Reuse from LAURAFED
            GLOBAL_TEST_SIZE = 0.2
            VAL_SIZE = 0.125
            MAX_CLIENT_SIZE_STRATEGY = 'percentile'
            MAX_CLIENT_SIZE_PERCENTILE = 50
            FEDPROX_MU = 0.05          # Higher MU for Transfer Learning to stay close to global
            ADAPTIVE_MU = True
            WEIGHT_DECAY = 1e-4        # Higher decay for regularization
            GRAD_CLIP_NORM = 1.0
            LR_MODE = 'cosine'
            PATIENCE = 40
            MIN_ROUNDS_BEFORE_STOPPING = 30
            LOSS_EXPLOSION_THRESHOLD = 10.0
            VALIDATE_DATA_QUALITY = True
            MAX_CHECKPOINTS = 5

        config = TransferConfig()
        
        # Model configurations
        MODEL_SIZE = 'small'
        configs = {
            'small': {
                "image_size": 256, "num_layers": 8, "hidden_dim": 512,
                "mlp_dim": 2048, "num_heads": 8, "dropout_rate": 0.1, 
                "block_dropout_rate": 0.05, "patch_size": 16, "num_channels": 1,
            }
        }
        model_config = configs[MODEL_SIZE]
        model_config["num_patches"] = (model_config["image_size"] // model_config["patch_size"]) ** 2

        logger.info("="*80)
        logger.info(f"FEDERATED TRANSFER LEARNING - {MODEL_SIZE.upper()} MODEL")
        logger.info("="*80)
        logger.info(f"Source Model: {config.PRETRAINED_PATH}")
        logger.info(f"Freeze Encoder: {config.FREEZE_ENCODER}")
        if config.FREEZE_ENCODER:
            logger.info(f"Unfreeze at Round: {config.UNFREEZE_ROUND}")
        
        # =====================================================================================
        # DATA PREPARATION
        # =====================================================================================
        # (Reusing robust logic from LAURAFED/EnsembleFL)
        dataset_clients_raw = []
        
        if os.path.exists(config.IMAGE_DIR):
            image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
            mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
            nsclc_data = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]
            if nsclc_data:
                dataset_clients_raw.append(("NSCLC-Radiomics", nsclc_data))
        
        if config.USE_ADDITIONAL_DATA:
            add_image_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "image", "*.png")))
            add_mask_files = sorted(glob.glob(os.path.join(config.ADDITIONAL_DATA_DIR, "masks", "*.png")))
            task6_data = [{"image": img, "label": mask} for img, mask in zip(add_image_files, add_mask_files)]
            if task6_data:
                dataset_clients_raw.append(("Task6-Lung", task6_data))

        # Split Data
        client_train_val_data, global_test_data = create_global_test_set_first(
            dataset_clients_raw, test_size=config.GLOBAL_TEST_SIZE
        )
        
        # Client Sizing
        dataset_sizes = [len(data) for _, data in client_train_val_data]
        max_client_size = 1300
        
        virtual_clients = split_large_clients_adaptive(
            client_train_val_data, max_client_size=max_client_size
        )
        
        # Compute Stats for Normalization
        dataset_stats = {}
        for dataset_name, data_dicts in dataset_clients_raw:
            sample_size = min(100, len(data_dicts))
            sample_data = np.random.choice(data_dicts, sample_size, replace=False)
            pixel_values = []
            for item in sample_data:
                from PIL import Image
                img = Image.open(item["image"])
                pixel_values.extend(np.array(img).flatten())
            dataset_stats[dataset_name] = {
                'mean': float(np.mean(pixel_values)),
                'std': float(np.std(pixel_values)),
                'p1': float(np.percentile(pixel_values, 1)),
                'p99': float(np.percentile(pixel_values, 99))
            }

        # Transforms
        def create_transforms_for_dataset(dataset_name, is_train=True):
            stats = dataset_stats.get(dataset_name, {'mean': 127.5, 'std': 127.5, 'p1': 0, 'p99': 255})
            base_transforms = [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=255.0, clip=True),
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
                ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(model_config["image_size"], model_config["image_size"])),
            ]
            if is_train:
                base_transforms.extend([
                    RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
                    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
                    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
                    RandAffined(keys=['image', 'label'], prob=0.3, translate_range=(5, 5), rotate_range=(np.pi/36, np.pi/36), scale_range=(0.05, 0.05), mode=("bilinear", "nearest")),
                    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.02),
                    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
                ])
            base_transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
            return Compose(base_transforms)

        # Create Loaders
        client_train_loaders, client_val_loaders = [], []
        client_sample_sizes, client_names, client_source_datasets = [], [], []
        
        for client_name, client_data, source_dataset in virtual_clients:
            if not client_data: continue
            train_data, val_data, _ = split_data_by_patient(client_data, test_size=0, val_size=config.VAL_SIZE)
            if not train_data: continue
            
            train_ds = Dataset(data=train_data, transform=create_transforms_for_dataset(source_dataset, True))
            val_ds = Dataset(data=val_data, transform=create_transforms_for_dataset(source_dataset, False))
            
            client_batch_size = min(config.BATCH_SIZE, max(config.MIN_BATCH_SIZE, len(train_data) // 10))
            
            client_train_loaders.append(DataLoader(train_ds, batch_size=client_batch_size, shuffle=True, num_workers=0))
            client_val_loaders.append(DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0))
            client_sample_sizes.append(len(train_data))
            client_names.append(client_name)
            client_source_datasets.append(source_dataset)

        # Global Test Loaders
        dataset_test_loaders = {}
        for dataset_name in set(ds for ds, _ in global_test_data):
            data = [item for ds, item in global_test_data if ds == dataset_name]
            ds = Dataset(data=data, transform=create_transforms_for_dataset(dataset_name, False))
            dataset_test_loaders[dataset_name] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        # =====================================================================================
        # INITIALIZE MODEL & TRANSFER LEARNING SETUP
        # =====================================================================================
        
        global_model = LightweightAuraViT(model_config).to(config.DEVICE)
        
        # Load Pre-trained Weights
        global_model, loaded_success = load_pretrained_model(global_model, config.PRETRAINED_PATH, config.DEVICE)
        
        if not loaded_success:
            logger.warning("Starting with RANDOM initialization (standard FL, not transfer learning)")
        else:
            logger.info("Starting with PRE-TRAINED initialization (Federated Transfer Learning)")
            
        # Freeze Encoder if requested
        if config.FREEZE_ENCODER:
            global_model = freeze_encoder(global_model)
            
        loss_function = StableLoss(dice_weight=0.7, ce_weight=0.3)
        
        # Components
        checkpoint_manager = CheckpointManager(config.CHECKPOINT_DIR, max_checkpoints=config.MAX_CHECKPOINTS)
        monitor = FederatedTrainingMonitor()
        drift_detector = ClientDriftDetector(threshold=0.15)
        
        # LR Scheduler
        lr_scheduler = AdaptiveFederatedLRScheduler(
            config.LEARNING_RATE, config.WARMUP_ROUNDS, config.COMM_ROUNDS, min_lr=config.MIN_LEARNING_RATE
        )
        
        best_metric = -1
        
        # =====================================================================================
        # TRAINING LOOP
        # =====================================================================================
        
        for round_num in range(config.COMM_ROUNDS):
            logger.info(f"\n--- Round {round_num+1}/{config.COMM_ROUNDS} ---")
            
            # Check if we need to unfreeze
            if config.FREEZE_ENCODER and (round_num + 1) == config.UNFREEZE_ROUND:
                logger.info("SCHEDULED EVENT: Unfreezing Encoder for fine-tuning!")
                global_model = unfreeze_all(global_model)
                # Lower LR for fine-tuning entire model
                lr_scheduler.base_lr *= 0.5
                lr_scheduler.min_lr *= 0.5
            
            # Log LR
            current_lr = lr_scheduler.get_lr()
            monitor.log_lr(round_num + 1, current_lr)
            
            selected_clients = select_clients_for_round(
                client_names, 
                min_participation=config.MIN_CLIENT_PARTICIPATION,
                max_participation=config.MAX_CLIENT_PARTICIPATION,
                ensure_diversity=True,
                client_source_datasets=client_source_datasets
            )
            
            local_weights = []
            local_sizes = []
            val_metrics = []
            local_losses = []
            
            dice_metric = DiceMetric(include_background=True, reduction="mean")
            post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            
            for client_name in selected_clients:
                idx = client_names.index(client_name)
                train_loader = client_train_loaders[idx]
                val_loader = client_val_loaders[idx]
                size = client_sample_sizes[idx]
                
                epochs = compute_dynamic_epochs(size, min_epochs=config.MIN_EPOCHS, max_epochs=config.MAX_EPOCHS) if config.DYNAMIC_EPOCHS else config.MIN_EPOCHS
                
                # Adjust mu based on drift
                mu = config.FEDPROX_MU
                if config.ADAPTIVE_MU and drift_detector.detect_drift(client_name):
                    mu *= 2.0
                    logger.info(f"  Client {client_name}: Drift detected, increased mu to {mu}")
                
                # Training
                logger.info(f"Client {client_name}: Fine-tuning")
                local_model = deepcopy(global_model)
                
                # IMPORTANT: Filter parameters that require grad!
                # If encoder is frozen, optimizer should not touch it.
                trainable_params = [p for p in local_model.parameters() if p.requires_grad]
                
                optimizer = torch.optim.AdamW(
                    trainable_params, 
                    lr=lr_scheduler.get_client_lr(client_name, size), 
                    weight_decay=config.WEIGHT_DECAY
                )
                
                trained_model, losses = client_update_enhanced(
                    client_name, local_model, global_model, optimizer, train_loader, epochs, 
                    config.DEVICE, mu, torch.cuda.amp.GradScaler(), 
                    loss_function, config.GRAD_CLIP_NORM
                )
                
                # Validation
                trained_model.eval()
                with torch.no_grad():
                    dice_metric.reset()
                    for val_data in val_loader:
                        val_inputs = val_data["image"].to(config.DEVICE)
                        val_labels = val_data["label"].to(config.DEVICE)
                        val_outputs = trained_model(val_inputs)
                        val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
                        dice_metric(y_pred=val_outputs_post, y=val_labels)
                    val_dice = dice_metric.aggregate().item()
                
                # Collect results
                local_weights.append(trained_model.state_dict())
                local_sizes.append(size)
                val_metrics.append(val_dice)
                local_losses.extend(losses)
                
                drift_detector.update(client_name, val_dice)
                monitor.log_client(client_name, round_num + 1, np.mean(losses) if losses else 0, val_dice)
                
                del local_model, optimizer, trained_model
                torch.cuda.empty_cache()
            
            # Aggregation
            logger.info("Aggregating updates...")
            new_global_weights = adaptive_weighted_average_improved(
                [{k: v for k, v in w.items()} for w in local_weights], # Convert to list of dicts if needed, or just pass list of state_dicts
                local_sizes, global_model, [], config.ADAPTIVE_ALPHA, val_metrics
            )
            global_model.load_state_dict(new_global_weights)
            
            lr_scheduler.step()
            
            # Global Evaluation
            logger.info("Global Evaluation...")
            global_dice_scores = []
            global_losses = []
            
            for name, loader in dataset_test_loaders.items():
                global_model.eval()
                dice_metric.reset()
                test_loss = 0
                batch_cnt = 0
                with torch.no_grad():
                    for test_data in loader:
                        inputs = test_data["image"].to(config.DEVICE)
                        labels = test_data["label"].to(config.DEVICE)
                        outputs = global_model(inputs)
                        loss = loss_function(outputs, labels)
                        if not torch.isnan(loss):
                            test_loss += loss.item()
                            batch_cnt += 1
                        
                        outputs_post = [post_pred(i) for i in decollate_batch(outputs)]
                        dice_metric(y_pred=outputs_post, y=labels)
                    
                d = dice_metric.aggregate().item()
                l = test_loss / batch_cnt if batch_cnt > 0 else 0
                global_dice_scores.append(d)
                global_losses.append(l)
                logger.info(f"  {name}: Dice={d:.4f}")
            
            avg_global_dice = np.mean(global_dice_scores)
            avg_global_loss = np.mean(global_losses)
            
            monitor.log_round(round_num + 1, np.mean(local_losses), avg_global_dice, avg_global_loss)
            
            logger.info(f"Round {round_num+1} Global Dice: {avg_global_dice:.4f}")
            
            # Save
            is_best = False
            if avg_global_dice > best_metric:
                best_metric = avg_global_dice
                is_best = True
                logger.info("New Best Transfer Model!")
            
            checkpoint_data = {
                'round': round_num + 1,
                'model_state_dict': global_model.state_dict(),
                'best_metric': best_metric,
                'lr_scheduler': lr_scheduler.state_dict()
            }
            checkpoint_manager.save(checkpoint_data, round_num + 1, is_best=is_best)
            
            if (round_num + 1) % 5 == 0:
                monitor.plot_training_curves(save_path=os.path.join(config.CHECKPOINT_DIR, 'training_curves.png'))

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
