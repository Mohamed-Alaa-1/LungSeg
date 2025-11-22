import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.data import decollate_batch
from ..utils.utils import check_gradients
import logging

logger = logging.getLogger(__name__)

def train_epoch(model, loader, optimizer, loss_fn, scaler, lr_scheduler, config, epoch):
    model.train()
    epoch_loss = 0
    batch_count = 0
    current_lr = lr_scheduler.step()
    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Training]", unit="batch")

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
                    loss = loss_fn(seg_outputs, labels)
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
        return None, current_lr

    return epoch_loss / batch_count, current_lr


def validate(model, loader, loss_fn, config):
    model.eval()
    val_loss = 0
    val_batch_count = 0
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    cm_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["accuracy"], 
        reduction="mean"
    )

    with torch.no_grad():
        for val_data in loader:
            try:
                val_inputs, val_labels = val_data["image"].to(config.DEVICE), val_data["label"].to(config.DEVICE)
                
                if torch.isnan(val_inputs).any() or torch.isnan(val_labels).any():
                    continue
                
                seg_outputs = model(val_inputs)
                
                if torch.isnan(seg_outputs).any():
                    continue
                
                val_batch_loss = loss_fn(seg_outputs, val_labels)
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
        metric = 0.0
        accuracy = 0.0
    
    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
    
    return avg_val_loss, metric, accuracy
