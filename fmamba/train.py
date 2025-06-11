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

    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha = 1 - decay)