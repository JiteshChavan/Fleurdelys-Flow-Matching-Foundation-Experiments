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
# inspiredfrom
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

# --------------------------------------------------------------------------------
# Interpolate position embeddings for high-resolution
# Regerences:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1] # channels
        num_patches = model.x_embedder.num_patches # patches in new resolution
        num_extra_tokens = (model.pos_embed.shape[-2] - num_patches) # same number of extra tokens, only resolution is different
        # original height (== width) from the checkpoint
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
        # new height (== width) for the new resolution
        new_size = int(num_patches**0.5) # num patches are derived from the model.x_embedder so that T corresponds to the new resolutions, also it doesnt coutn extra tokens
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Interpolating positional embedding from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:,:num_extra_tokens] # pos_embed is usually (1, T, C) for ease of broadcasting x = x + pos_embed
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2) # (B, C, H, W)
            pos_tokens = F.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            ) # (B, C, H, W)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2) #(B, T, C)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1) # concatenate extra tokens along T not along B hence dim = 1
            checkpoint_model["pos_embed"] = new_pos_embed


# ----------------------------------------------------------------------------------------------
# Embedding layers for Timesteps and Class Labels
# ----------------------------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        Args:
            t : Float, a 1-D tensor of B indices, one per batch element.
            dim : the dimension of the output vector representation.
            max_period: controls the minimum frequency of the embeddings.
            retuns (B, dim=C) tensor representation corresponding to scalar timesteps t (B,) 
        """

        half = dim // 2
        # interpolate linearly between 0 and -log(max_period) [0, log(f_min)] then exponentiate (decay 1 -> f_min)
        # gives us linearly spaced frequencies in logspace
        # exponentiation results in exponential decay between [1, 1/max_period] i.e [1, f_min]
        # (C/2)
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        # t (B,)
        # outer product to get (B, C/2)
        # three ways
        #args = torch.einsum("B,C->BC", t, freqs) # outer product < (B), (C/2)> -> (B, C/2)
        #args = t.unsqueeze(1) * freqs #(B, 1) * (C/2) -> (B, C/2)
        args = t[:, None].float() * freqs # (B, 1) * (C/2)
        
        embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)

        if dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    