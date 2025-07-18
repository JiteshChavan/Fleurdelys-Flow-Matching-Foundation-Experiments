import math
from functools import partial
from typing import Optional

from huggingface_hub import PyTorchModelHubMixin


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import CondMamba, Mamba
from pe.cpe import AdaInPosCNN
from pe.my_rotary import apply_rotary, get_2d_sincos_rotary_embed
from rope import *
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch import Tensor


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from CrossAttentionFusion import CrossAttentionFusion
from dct_layer import init_dct_kernel, init_idct_kernel
from einops import rearrange
from mlp import GatedMLP
from switch_mlp import SwitchMLP
from wavelet_layer import DWT_2D, IDWT_2D

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    embed_dim : int dimensions of pos embed vector representation
    grid_size : int, height and width of the grid
    
    Returns:
    pos_embed : [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (/w or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h) # width first
    grid = np.stack(grid, axis=0) # stack the two elements in the list in a single list 0 indexes into abscissae nad 1 into ordinates

    # rearrange the grid so that 0 and 1 explicitly index into abscissae and ordinates respectively
    grid = grid.reshape([2, 1, grid_size, grid_size])
    # get pos embedding from the constructed grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed # ([grid_size*grid_size, embed_dim]) includes prefix of extra tokens and cls tokens if specified

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0, f"embed_dim is required to be divisible by 2,\nhalf of the dimensions represent abscissae\nthe other half ordinates"

    emb_h = get_1d_sincos_pos_embed_from_grid(grid[0], embed_dim // 2) # (H*W, D/2) representation of ordinates
    emb_w = get_1d_sincos_pos_embed_from_grid(grid[1], embed_dim // 2) # (H*W, D/2) representation of abscissae

    emb = np.concatenate([emb_h, emb_w], axis=1) #(H*W, D) concatenate the representations along channels
    return emb

def get_1d_sincos_pos_embed_from_grid(pos_grid, embed_dim):
    """
    Takes a grid that specifies ordinates or abscissae of a co-ordinate grid, returns (H*W, embed_dim) representation
    of the grid.
    Args:
        pos_grid: np array specifies ordinates or abscissae of a co-ordinate grid.
        embed_dim: number of dimensions for vector representation of each position
    """

    assert embed_dim % 2 == 0, f"embed_dim:{embed_dim} must be divisible by 2\nhalf dimensions for sine components\nthe other half for cosine"

    # frequencies linearly spaced in log scale, will be exponentially spaced in normal scale
    log_freq = np.arange(embed_dim // 2, dtype=np.float64) # (0 through D//2 -1 )
    log_freq = log_freq / (embed_dim // 2) # normalize to be between 0 and 1 so that we dont have numerical instability while exponentiating
    freq = 1 / 1000.0 ** log_freq # (D/2)

    pos = pos_grid.reshape(-1) # (T,) (flatten H,W)
    # we want to infuse the frequencies with position
    # two ways
    #pos_modulated_freq = pos.reshape(-1, 1)  * freq # (T, 1) * (D/2) -> (T, D/2)
    # or
    pos_modulated_freq = np.einsum("T,D->TD", pos, freq) # (T) @ (D/2) -> (T, D/2) outer product

    sin_embd = np.sin(pos_modulated_freq)
    cos_embd = np.cos(pos_modulated_freq)

    emb = np.concatenate([sin_embd, cos_embd], axis=1) # (T, D)
    return emb


