import torch
import torch.nn as nn
import logging
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prune_model(model, pruning_rate=0.3):
    """
    Prune model weights to reduce size
    """
    
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
