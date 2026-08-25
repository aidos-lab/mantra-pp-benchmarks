import torch
import os

def is_blackwell_gpu(cfg):
    if cfg.trainer.device == 'cpu':
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10

def is_gpu(cfg):
    if cfg.trainer.device == 'cpu':
        return False
    if not torch.cuda.is_available():
        return False
    return True

def set_gpu_configs(cfg):
    # Only set for CPU
    if not is_gpu(cfg):
        torch.set_num_threads(os.cpu_count())

    # Set config GPU
    if is_gpu(cfg):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if is_blackwell_gpu(cfg):
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
