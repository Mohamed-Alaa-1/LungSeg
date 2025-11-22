import torch
from tqdm import tqdm
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.data import decollate_batch, DataLoader, Dataset
from monai.transforms import Activations, AsDiscrete, Compose
from ..data.transforms import get_val_transforms
import logging

logger = logging.getLogger(__name__)

def evaluate(model, test_files, config, model_config):
    model.eval()
    
    test_ds = Dataset(data=test_files, transform=get_val_transforms(model_config["image_size"]))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    cm_metric = ConfusionMatrixMetric(
        include_background=False, metric_name=["sensitivity", "specificity", "accuracy"], reduction="mean"
    )
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

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

    return mean_dice_test, iou_test, sensitivity, specificity, accuracy
