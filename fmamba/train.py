# utils
import gc
import math
import sys
from pathlib import Path

import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from time import time
from tqdm import tqdm


import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DDP
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# VAE
from diffusers.models import AutoencoderKL

from create_model import create_model
from datasets_prep import get_dataset
from fmamba_modules import interpolate_pos_embed

# dataset scaffolding
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from flow import Sampler, create_flow

# for easy imports from eval toolbox
eval_import_path = (Path(__file__).parent.parent / "eval_toolbox").resolve().as_posix()
sys.path.append(eval_import_path)
# import dnnlib from evaltoolbox for convenience classes like EasyDict
import dnnlib
# for FID evaluations
from pytorch_fid import metric_main, metric_utils


# -------------------------------------------------------------------------------------------------------------------------
#  Training Helper Fucntions
# -------------------------------------------------------------------------------------------------------------------------

# exponential moving average for weight update
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    
    # can use list comprehension alternative instead as well dict = {n:p for n,p in ...}
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: consider applying only to params that require_grad to avoid small numerical changes in pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha = 1 - decay)

def requires_grad (model, flag = True):
    """
    Set requires_grad flag for all parameters in a model.
    """

    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger (log_dir):
    """
    Create a logger that writes to a log file and stdout.
    """

    if dist.get_rank() == 0: # master process
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else: # dummy logger (does nothing) for non master processes
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

# TODO: might want to remove this if not used anywhere
def center_crop_arr (pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX) # average pixels in 2x2 BOX for downsampling
    
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x : crop_x + image_size])


# TODO : Ablation linear warmup that starts at 1 / warmup, change if doesnt work out
def adjust_learning_rate (optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after linear warmup"""
    if epoch < args.warmup_epochs:
        lr = args.max_lr * (epoch+1) / args.warmup_epochs
    else:
        decay_ratio = (epoch - args.warmup_epochs) / (args.max_epochs - args.warmup_epochs)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = args.min_lr + (args.max_lr - args.min_lr) * coeff
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
    return lr

# ---------------------------------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------------------------------


def main (args):
    """
    Core training loop
    """

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # setup DDP:
    dist.init_process_group("nccl")
    # if "SLURM_PROCID" in os.environ:
    #   rank = int(os.environ["SLURM_PROCID"])
    #   gpu = rank % torch.cuda.device_count()
    #   world_size = int(os.environ["WORLD_SIZE"], 1)
    # else:
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size"



    


