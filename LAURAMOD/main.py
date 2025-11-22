import os
import torch
import matplotlib.pyplot as plt
from monai.utils import set_determinism

from config.config import get_config
from data.dataset import get_dataloaders
from models.model import LightweightAuraViT
from utils.logger import setup_logging
from utils.losses import StableLoss
from utils.optimizers import StableLRScheduler
from utils.utils import count_parameters, prune_model, quantize_model
from engine.trainer import train_epoch, validate
from engine.evaluator import evaluate

# For reproducibility
set_determinism(seed=42)

def main():
    logger = setup_logging()
    
    try:
        config, model_config, model_size = get_config()

        logger.info("="*60)
        logger.info(f"LIGHTWEIGHT AURAVIT - {model_size.upper()} MODEL")
        logger.info("="*60)
        logger.info(f"Using device: {config.DEVICE}")
        logger.info(f"Model configuration: {model_size}")
        logger.info(f"Image size: {model_config['image_size']}")
        logger.info(f"Layers: {model_config['num_layers']}")
        logger.info(f"Hidden dim: {model_config['hidden_dim']}")
        logger.info(f"Batch size: {config.BATCH_SIZE}")

        train_loader, val_loader, train_files, val_files, test_files = get_dataloaders(config, model_config)
        
        logger.info(f"Total training samples: {len(train_files)}")
        logger.info(f"Total validation samples: {len(val_files)}")
        logger.info(f"Total testing samples: {len(test_files)}")

        model = LightweightAuraViT(model_config).to(config.DEVICE)
        
        total_params = count_parameters(model)
        model_size_mb = total_params * 4 / (1024 * 1024)
        logger.info(f"Total trainable parameters: {total_params:,}")
        logger.info(f"Approximate model size: {model_size_mb:.2f} MB")
        
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
        
        scaler = torch.cuda.amp.GradScaler()

        start_epoch = 0
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        accuracy_values = []
        patience_counter = 0

        if os.path.exists(config.CHECKPOINT_PATH):
            try:
                checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_metric = checkpoint.get('best_metric', best_metric)
                epoch_loss_values = checkpoint.get('epoch_loss_values', [])
                metric_values = checkpoint.get('metric_values', [])
                patience_counter = checkpoint.get('patience_counter', 0)
                if 'scheduler_epoch' in checkpoint:
                    lr_scheduler.current_epoch = checkpoint['scheduler_epoch']
                
                logger.info(f"Checkpoint found! Resuming from epoch {start_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
                start_epoch = 0

        logger.info("Starting Lightweight AuraViT Training")
        
        for epoch in range(start_epoch, config.MAX_EPOCHS):
            avg_epoch_loss, current_lr = train_epoch(model, train_loader, optimizer, loss_function, scaler, lr_scheduler, config, epoch)
            if avg_epoch_loss is None:
                break
                
            epoch_loss_values.append(avg_epoch_loss)

            avg_val_loss, metric, accuracy = validate(model, val_loader, loss_function, config)
            metric_values.append(metric)
            accuracy_values.append(accuracy)

            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {metric:.4f} | Acc: {accuracy:.4f} | LR: {current_lr:.2e} | Best: {best_metric:.4f}")

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

            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_epoch': lr_scheduler.current_epoch,
                    'best_metric': best_metric,
                    'epoch_loss_values': epoch_loss_values,
                    'metric_values': metric_values,
                    'patience_counter': patience_counter,
                }, config.CHECKPOINT_PATH)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}. No improvement for {config.PATIENCE} epochs.")
                break

        logger.info(f"Training finished. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

        if config.ENABLE_PRUNING:
            logger.info("="*60)
            logger.info("APPLYING POST-TRAINING PRUNING")
            logger.info("="*60)
            
            if os.path.exists(config.BEST_MODEL_PATH):
                model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
            
            model_pruned = prune_model(model, config.PRUNING_RATE)
            torch.save(model_pruned.state_dict(), config.PRUNED_MODEL_PATH)
            
            pruned_params = count_parameters(model_pruned)
            pruned_size_mb = pruned_params * 4 / (1024 * 1024)
            
            logger.info(f"Pruned model parameters: {pruned_params:,}")
            logger.info(f"Pruned model size: {pruned_size_mb:.2f} MB")

        logger.info("Plotting training curves...")
        plt.figure("train", (18, 6))
        
        plt.subplot(1, 3, 1)
        plt.title(f"Training Loss - {model_size.upper()} Model")
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
        plt.savefig(f"lightweight_{model_size}_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("="*60)
        logger.info("RUNNING FINAL EVALUATION ON TEST SET")
        logger.info("="*60)
        
        if os.path.exists(config.BEST_MODEL_PATH):
            model.load_state_dict(torch.load(config.BEST_MODEL_PATH))

        mean_dice_test, iou_test, sensitivity, specificity, accuracy = evaluate(model, test_files, config, model_config)
        
        logger.info("="*60)
        logger.info(f"FINAL TEST METRICS - LIGHTWEIGHT {model_size.upper()} AURAVIT")
        logger.info("="*60)
        logger.info(f"Mean Dice Score: {mean_dice_test:.4f}")
        logger.info(f"Intersection over Union (IoU): {iou_test:.4f}")
        logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("="*60)

        logger.info("Creating quantized model for deployment...")
        model_quantized = quantize_model(model)
        
        torch.save(model_quantized.state_dict(), config.QUANTIZED_MODEL_PATH)
        logger.info(f"Quantized model saved to: {config.QUANTIZED_MODEL_PATH}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
