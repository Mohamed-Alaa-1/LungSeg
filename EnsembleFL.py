
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
# This assumes LAURAFED.py is in the same directory and is importable
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
    # Fallback if import fails (e.g. running in a way where relative imports fail)
    # In a real scenario, we might want to copy the classes here.
    # For now, we assume the file exists.
    print("Error importing from LAURAFED.py. Make sure it exists in the same directory.")
    raise

# Setup logging
logger = setup_logging("ensemble_federated_training.log")

# =====================================================================================
# ENSEMBLE EVALUATION
# =====================================================================================

def evaluate_ensemble_on_dataset(models, test_loader, device, loss_function, post_pred, dataset_name):
    """
    Evaluate ensemble of models on a specific dataset.
    Predictions are averaged (soft voting).
    """
    for model in models:
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
        for test_data in tqdm(test_loader, desc=f"Testing Ensemble {dataset_name}", leave=False):
            test_inputs = test_data["image"].to(device)
            test_labels = test_data["label"].to(device)
            
            # Ensemble Prediction
            ensemble_outputs = None
            batch_loss_sum = 0
            
            for model in models:
                outputs = model(test_inputs)
                
                # Accumulate loss for tracking (average loss of individual models)
                loss = loss_function(outputs, test_labels)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss_sum += loss.item()
                
                # Soft voting: average probabilities (after sigmoid)
                probs = torch.sigmoid(outputs)
                if ensemble_outputs is None:
                    ensemble_outputs = probs
                else:
                    ensemble_outputs += probs
            
            # Average probabilities
            ensemble_outputs /= len(models)
            
            # Average loss
            avg_batch_loss = batch_loss_sum / len(models)
            test_loss += avg_batch_loss
            batch_count += 1
            
            # Metric calculation using ensemble probabilities
            # Note: post_pred usually applies sigmoid + threshold. 
            # Since we already applied sigmoid, we just need threshold.
            # But post_pred in main includes Activations(sigmoid=True). 
            # So we should pass the raw logit average? No, averaging logits is different.
            # Averaging probabilities is standard.
            # Let's just threshold the averaged probabilities.
            
            test_outputs_post = [AsDiscrete(threshold=0.5)(i) for i in decollate_batch(ensemble_outputs)]
            
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

# =====================================================================================
# MAIN ENSEMBLE FEDERATED TRAINING
# =====================================================================================

def main():
    try:
        class EnsembleConfig:
            # Paths
            DATA_DIR = "/teamspace/studios/this_studio/lung/processed-NSCLC-Radiomics"
            IMAGE_DIR = os.path.join(DATA_DIR, "images")
            MASK_DIR = os.path.join(DATA_DIR, "masks")
            ADDITIONAL_DATA_DIR = "/teamspace/studios/this_studio/lung/processed_task6_lung"
            USE_ADDITIONAL_DATA = os.path.exists(ADDITIONAL_DATA_DIR)
            
            # PRE-TRAINED MODELS (Update these paths)
            # Using placeholders or assuming they are in current dir
            PRETRAINED_MODEL_1 = "/teamspace/studios/this_studio/newplan/small/NSCLC/lightweight_best_AuraViT_model_fold0.pth" 
            PRETRAINED_MODEL_2 = "/teamspace/studios/this_studio/newplan/small/msd/lightweight_best_AuraViT_model_fold0_MSD.pth" 
            
            CHECKPOINT_DIR = "checkpoints_ensemble"
            
            # FL Params
            COMM_ROUNDS = 200
            MIN_EPOCHS = 3
            MAX_EPOCHS = 20
            DYNAMIC_EPOCHS = True
            
            # Training
            BATCH_SIZE = 8
            MIN_BATCH_SIZE = 2
            LEARNING_RATE = 1e-3
            MIN_LEARNING_RATE = 1e-5
            WARMUP_ROUNDS = 5
            
            # Client Selection
            MIN_CLIENT_PARTICIPATION = 1.0
            MAX_CLIENT_PARTICIPATION = 1.0
            
            # Aggregation
            AGGREGATION_STRATEGY = 'adaptive_weighted'
            ADAPTIVE_ALPHA = 0.3
            
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Config from LAURAFED
            GLOBAL_TEST_SIZE = 0.2
            VAL_SIZE = 0.125
            MAX_CLIENT_SIZE_STRATEGY = 'percentile'
            MAX_CLIENT_SIZE_PERCENTILE = 50
            FEDPROX_MU = 0.01
            ADAPTIVE_MU = True
            WEIGHT_DECAY = 1e-5
            GRAD_CLIP_NORM = 0.5
            LR_MODE = 'cosine'
            PATIENCE = 30
            MIN_ROUNDS_BEFORE_STOPPING = 50
            LOSS_EXPLOSION_THRESHOLD = 10.0
            VALIDATE_DATA_QUALITY = True

        config = EnsembleConfig()
        
        # Model configurations (Same as LAURAFED)
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
        logger.info(f"ENSEMBLE FEDERATED LEARNING - {MODEL_SIZE.upper()} MODEL")
        logger.info("="*80)
        
        # =====================================================================================
        # DATA PREPARATION (Copied logic from LAURAFED)
        # =====================================================================================
        dataset_clients_raw = []
        
        # Dataset 1
        if os.path.exists(config.IMAGE_DIR):
            image_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*.png")))
            mask_files = sorted(glob.glob(os.path.join(config.MASK_DIR, "*.png")))
            nsclc_data = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]
            if nsclc_data:
                dataset_clients_raw.append(("NSCLC-Radiomics", nsclc_data))
        
        # Dataset 2
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
        max_client_size = 1300 # Fixed as per LAURAFED
        
        virtual_clients = split_large_clients_adaptive(
            client_train_val_data, max_client_size=max_client_size
        )
        
        # Compute Stats
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
            
        # Transform Creator
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
        # ENSEMBLE MODEL INITIALIZATION
        # =====================================================================================
        logger.info("Initializing Ensemble Models...")
        
        model1 = LightweightAuraViT(model_config).to(config.DEVICE)
        model2 = LightweightAuraViT(model_config).to(config.DEVICE)
        
        # Load pre-trained weights
        if os.path.exists(config.PRETRAINED_MODEL_1):
            logger.info(f"Loading Model 1 from {config.PRETRAINED_MODEL_1}")
            ckpt = torch.load(config.PRETRAINED_MODEL_1, map_location=config.DEVICE)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            try:
                model1.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                logger.warning(f"Model 1 loading warning (using strict=False): {e}")
                model1.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"Pre-trained Model 1 not found at {config.PRETRAINED_MODEL_1}. Initializing randomly.")

        if os.path.exists(config.PRETRAINED_MODEL_2):
            logger.info(f"Loading Model 2 from {config.PRETRAINED_MODEL_2}")
            ckpt = torch.load(config.PRETRAINED_MODEL_2, map_location=config.DEVICE)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            try:
                model2.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                logger.warning(f"Model 2 loading warning (using strict=False): {e}")
                model2.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"Pre-trained Model 2 not found at {config.PRETRAINED_MODEL_2}. Initializing randomly.")
            # If Model 1 was found but Model 2 wasn't, maybe copy Model 1 or perturb it?
            # For now, random initialization provides diversity.
        
        models = [model1, model2]
        loss_function = StableLoss(dice_weight=0.7, ce_weight=0.3)
        
        # =====================================================================================
        # INITIALIZE COMPONENTS
        # =====================================================================================
        
        # Checkpoint manager
        checkpoint_manager = CheckpointManager(config.CHECKPOINT_DIR, max_checkpoints=5)
        
        # Training monitor
        monitor = FederatedTrainingMonitor()
        
        # Drift detector
        drift_detector = ClientDriftDetector(threshold=0.15)
        
        # LR Schedulers (one for each model stream)
        lr_scheduler1 = AdaptiveFederatedLRScheduler(config.LEARNING_RATE, config.WARMUP_ROUNDS, config.COMM_ROUNDS)
        lr_scheduler2 = AdaptiveFederatedLRScheduler(config.LEARNING_RATE, config.WARMUP_ROUNDS, config.COMM_ROUNDS)
        
        best_metric = -1
        
        # =====================================================================================
        # TRAINING LOOP
        # =====================================================================================
        
        for round_num in range(config.COMM_ROUNDS):
            logger.info(f"\n--- Round {round_num+1}/{config.COMM_ROUNDS} ---")
            
            # Log LR
            current_lr = lr_scheduler1.get_lr()
            monitor.log_lr(round_num + 1, current_lr)
            
            # --- FEDERATED BAGGING: Independent Client Sampling ---
            # Select two independent sets of clients to ensure diversity
            clients_m1 = select_clients_for_round(
                client_names, 
                min_participation=config.MIN_CLIENT_PARTICIPATION,
                max_participation=config.MAX_CLIENT_PARTICIPATION,
                ensure_diversity=True,
                client_source_datasets=client_source_datasets
            )
            
            clients_m2 = select_clients_for_round(
                client_names, 
                min_participation=config.MIN_CLIENT_PARTICIPATION,
                max_participation=config.MAX_CLIENT_PARTICIPATION,
                ensure_diversity=True,
                client_source_datasets=client_source_datasets
            )
            
            # Create a union of all unique clients involved this round
            active_clients = list(set(clients_m1 + clients_m2))
            
            logger.info(f"Bagging Selection: {len(clients_m1)} clients for M1, {len(clients_m2)} clients for M2")
            logger.info(f"Total active clients this round: {len(active_clients)}")
            
            # Storage for updates
            updates_m1 = []
            updates_m2 = []
            sizes_m1 = []
            sizes_m2 = []
            
            # Metrics for adaptive aggregation
            val_metrics_m1 = []
            val_metrics_m2 = []
            local_losses_m1 = []
            local_losses_m2 = []
            
            dice_metric = DiceMetric(include_background=True, reduction="mean")
            post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            
            for client_name in active_clients:
                idx = client_names.index(client_name)
                train_loader = client_train_loaders[idx]
                val_loader = client_val_loaders[idx]
                size = client_sample_sizes[idx]
                
                epochs = compute_dynamic_epochs(size, min_epochs=config.MIN_EPOCHS, max_epochs=config.MAX_EPOCHS) if config.DYNAMIC_EPOCHS else config.MIN_EPOCHS
                
                # Check drift to adjust mu
                mu = config.FEDPROX_MU
                if config.ADAPTIVE_MU and drift_detector.detect_drift(client_name):
                    mu *= 2.0
                    logger.info(f"  Client {client_name}: Drift detected, increased mu to {mu}")
                
                # --- Train Model 1 (If selected for M1) ---
                val_dice1 = 0.0
                loss1 = 0.0
                if client_name in clients_m1:
                    logger.info(f"Client {client_name}: Training Model 1")
                    local_m1 = deepcopy(model1)
                    opt1 = torch.optim.AdamW(local_m1.parameters(), lr=lr_scheduler1.get_client_lr(client_name, size), weight_decay=config.WEIGHT_DECAY)
                    trained_m1, losses1 = client_update_enhanced(
                        client_name, local_m1, model1, opt1, train_loader, epochs, 
                        config.DEVICE, mu, torch.cuda.amp.GradScaler(), 
                        loss_function, config.GRAD_CLIP_NORM
                    )
                    updates_m1.append(trained_m1.state_dict())
                    local_losses_m1.extend(losses1)
                    sizes_m1.append(size)
                    loss1 = np.mean(losses1)
                    
                    # Validate Model 1
                    trained_m1.eval()
                    with torch.no_grad():
                        dice_metric.reset()
                        for val_data in val_loader:
                            val_inputs = val_data["image"].to(config.DEVICE)
                            val_labels = val_data["label"].to(config.DEVICE)
                            val_outputs = trained_m1(val_inputs)
                            val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
                            dice_metric(y_pred=val_outputs_post, y=val_labels)
                        val_dice1 = dice_metric.aggregate().item()
                    val_metrics_m1.append(val_dice1)
                    del local_m1, opt1, trained_m1
                
                # --- Train Model 2 (If selected for M2) ---
                val_dice2 = 0.0
                loss2 = 0.0
                if client_name in clients_m2:
                    logger.info(f"Client {client_name}: Training Model 2")
                    local_m2 = deepcopy(model2)
                    opt2 = torch.optim.AdamW(local_m2.parameters(), lr=lr_scheduler2.get_client_lr(client_name, size), weight_decay=config.WEIGHT_DECAY)
                    trained_m2, losses2 = client_update_enhanced(
                        client_name, local_m2, model2, opt2, train_loader, epochs, 
                        config.DEVICE, mu, torch.cuda.amp.GradScaler(), 
                        loss_function, config.GRAD_CLIP_NORM
                    )
                    updates_m2.append(trained_m2.state_dict())
                    local_losses_m2.extend(losses2)
                    sizes_m2.append(size)
                    loss2 = np.mean(losses2)

                    # Validate Model 2
                    trained_m2.eval()
                    with torch.no_grad():
                        dice_metric.reset()
                        for val_data in val_loader:
                            val_inputs = val_data["image"].to(config.DEVICE)
                            val_labels = val_data["label"].to(config.DEVICE)
                            val_outputs = trained_m2(val_inputs)
                            val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]
                            dice_metric(y_pred=val_outputs_post, y=val_labels)
                        val_dice2 = dice_metric.aggregate().item()
                    val_metrics_m2.append(val_dice2)
                    del local_m2, opt2, trained_m2
                
                # Update drift detector & monitor (Use average of valid metrics)
                valid_dices = []
                if client_name in clients_m1: valid_dices.append(val_dice1)
                if client_name in clients_m2: valid_dices.append(val_dice2)
                
                if valid_dices:
                    avg_val_dice = sum(valid_dices) / len(valid_dices)
                    avg_loss = (loss1 + loss2) / len(valid_dices) # approx
                    drift_detector.update(client_name, avg_val_dice)
                    monitor.log_client(client_name, round_num + 1, avg_loss, avg_val_dice)
                
                torch.cuda.empty_cache()
            
            # --- Aggregation (Adaptive) ---
            logger.info("Aggregating updates (Adaptive Weighted)...")
            
            if updates_m1:
                new_weights_m1 = adaptive_weighted_average_improved(
                    updates_m1, sizes_m1, model1, [], config.ADAPTIVE_ALPHA, val_metrics_m1
                )
                model1.load_state_dict(new_weights_m1)
            
            if updates_m2:
                new_weights_m2 = adaptive_weighted_average_improved(
                    updates_m2, sizes_m2, model2, [], config.ADAPTIVE_ALPHA, val_metrics_m2
                )
                model2.load_state_dict(new_weights_m2)
            
            lr_scheduler1.step()
            lr_scheduler2.step()
            
            # --- Ensemble Evaluation ---
            logger.info("Evaluating Ensemble...")
            dataset_metrics = []
            for name, loader in dataset_test_loaders.items():
                res = evaluate_ensemble_on_dataset([model1, model2], loader, config.DEVICE, loss_function, None, name)
                dataset_metrics.append(res)
                logger.info(f"  {name}: Dice={res['dice']:.4f}")
            
            avg_dice = np.mean([r['dice'] for r in dataset_metrics])
            avg_loss = np.mean([r['loss'] for r in dataset_metrics])
            logger.info(f"Round {round_num+1} Ensemble Dice: {avg_dice:.4f}")
            
            # Log round metrics
            monitor.log_round(round_num + 1, np.mean(local_losses_m1 + local_losses_m2), avg_dice, avg_loss)
            
            # Save Checkpoint
            is_best = False
            if avg_dice > best_metric:
                best_metric = avg_dice
                is_best = True
                logger.info("New Best Ensemble Model!")
            
            checkpoint_data = {
                'round': round_num + 1,
                'model1_state_dict': model1.state_dict(),
                'model2_state_dict': model2.state_dict(),
                'best_metric': best_metric,
                'lr_scheduler1': lr_scheduler1.state_dict(),
                'lr_scheduler2': lr_scheduler2.state_dict()
            }
            checkpoint_manager.save(checkpoint_data, round_num + 1, is_best=is_best)
            
            # Plot curves periodically
            if (round_num + 1) % 5 == 0:
                monitor.plot_training_curves(save_path=os.path.join(config.CHECKPOINT_DIR, 'training_curves.png'))
                monitor.save_metrics(filepath=os.path.join(config.CHECKPOINT_DIR, 'metrics.json'))
                
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
