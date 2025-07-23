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
from scanning_orders import SCAN_ZOO, local_reverse, local_scan, reverse_permut_np
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

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg = dropout_prob > 0.0 # 1
        self.in_channels = num_classes + 1 if use_cfg else num_classes # 1001 or 1000
        self.embedding_table = nn.Embedding(self.in_channels, hidden_size)
        self.num_classes = num_classes # 1000
        self.dropout_prob = dropout_prob
    
    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """

        if force_drop_ids is None:
            # drop labels where p < drop prob
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob # setup boolean tensor via elementwise comparison
        else:
            drop_ids = force_drop_ids == 1 # elementwise comparison with 1, drop labels where force_drop_ids is 1
        
        # for each index in labels, set labels to be num_classes where drop_ids is True
        labels = torch.where(drop_ids, self.num_classes, labels) # 1000 or labels (table[1000] = null label)

        return labels
    
    def forward (self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def get_in_channels(self):
        """Returns in_channels or number of classes in the embedding table matrix"""
        return self.in_channels # 1001 if dropout_prob > 0.0 else 1000

class FinalLayer (nn.Module):
    """
    Final layer of the backbone
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
    
    def forward(self, x, y):
        scale, shift = self.adaLN_modulation(y).chunk(2, dim=-1) # 2x (B, C)
        x = modulate(self.norm_final(x), shift, scale) # x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) -> (B, T, C)
        x = self.linear(x) # (B, T, C) - > (B, T, patch_size * patch_size * out_channels)
        return x
    
class FlBlock(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32 = False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            skip=False,
            use_gated_mlp=True,
    ):
        """
        Simple block wrapping a mixer class (attention/mamba) with LayerNorm/RMSNorm and residual connection.

        This block has different structure compared to a regular pre layer norm transformer block.
        The standard is layer norm -> MHA/MLP -> proj (add)
        [Ref: https://arxiv.org/abs/2002.04745]

        Here we have a residual input (x) from previous block, aggregated x (processed by mixer attention/mamba).
        First this block adds residual with aggregated signal (x + proj(att(x))) which becomes residual for this block -> LN -> Mixer (out)
        then this block returns both out and the new residual
        returns out, previous residual(x + proj(att(x)))

        basically returns both output of the mixer projection and the residual.
        This is purely for performance reasons, as we can fuse add and layer norm.
        The residual needs to be provided (except for the very first block)
        """

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity

        # TODO: check how to initiate a cond mamba here
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # w/o FFN
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import failed"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), f"Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.norm_2 = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        mlp_hidden_dim = int (dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0) # dim -> 4*dim -> dim
    
    def forward (
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            y: Optional[Tensor] = None,
            inference_params=None,
    ):
        r"""Pass the input signal through the encoder layer.
        Args:
            hidden_states: input signal (B,T,C), the sequence to the encoder layer (required)
            residual: from previous block ? TODO: think, got it; hidden_states = Mixer(LN(residual))
            we fuse add and norm before branching residual and mixer
            y: label embedding (B, C)
        """
        if not self.fused_add_norm:
            # manual addition then norm which then becomes new residual then mix it to get hidden states and forward to next block
            if residual is None:
                # block 0
                residual = hidden_states
            else:
                # ADD
                residual = residual + self.drop_path(hidden_states)
            
            # we have residual path for this block now compute the mixer branch
            hidden_states = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # case of block 0
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else: 
                # we do have a residual and hidden_state = mixer(LN(residual)) and we add and normalize them
                hidden_states, residual= fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        
        T = hidden_states.shape[1] # (B, T, C)
        h = w = int(np.sqrt(T))

        # Prepare the signal for computation
        if self.transpose:
            hidden_states = rearrange(hidden_states, "n (h w) c -> n (w h) c", h=h, w=w)
        
        # Reduce jumps in locality by scanning the image in zig-zag order
        # left-right then right-left then left-right
        # seems like top to bottom then bottom-top then top-bottom so on
        # note that zig zag order depends on how the image is stored in B, T, C and w,h are just semantic and thusly interchangeable labels
        # id assume its a raster scan storage, so B, C, H, W if transpose true it becomes B, C, W, H which implies flipping along H
        # hence vertical zig zag
        if self.scanning_continuity:
            # we have (B, T, C)
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w) # (B, c, w, h)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1) # flip every alternate column along height axis so we index by zig zag scan (vertical zig zag in this case)
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w) # back to (B, T, C) from (B, C, W, H)
        
        # NOTE: Zigzag flip happens inside 2D space for maintaining spatial continuity; the final flip.(1) reverses the entire token sequence
        # lets the model see the sequence in reverse order
        # both the flips serve different goals
        if self.reverse:
            hidden_states = hidden_states.flip(1)
        
        scale_ssm, gate_ssm, shift_ssm, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1) #(B, 6C)-> 6x(B, C)
        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
            modulate(hidden_states, shift_ssm, scale_ssm), y, inference_params=inference_params
        )
        
        # forgot to pre normalize before mlp branch :P
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp))

        # transform the signal back
        if self.reverse:
            hidden_states = hidden_states.flip(1) # revert the flip across T
        
        if self.scanning_continuity:
            # (B,T, C) -> (B, C, H, W)
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w) # (B, C, W, H)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1) # revert flip every odd row
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w)
        
        if self.transpose:
            hidden_states = rearrange(hidden_states, "n (w h) c -> n (h w) c", h=h, w=w)
        
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
class FlBlockWindow(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            skip=False,
            shift_window=False,
    ):
        """
        Simple block around a mixer class with LayerNorm/RMSNorm and residual connection.

        Just like before, its still pre norm residual block with slightly different structure that allows
        to fuse addition back into residual pathway and then normalization.
        Purely for performance reasons, as we can fuse addition into residual path and LayerNorm.
        Residual needs to be provided except for the first block
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity # not really relevant here since the local non overlapping windows enforce continuity
        self.shift_window = shift_window

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        
        #w/o FFN
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"Only LayerNorm and RMSNorm are supported for fused_add_norm"
        
        self.norm_2 = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        mlp_hidden_dim = int (4 * dim)
        # Function to instantiate GELU object upon calling
        approx_gelu = lambda:nn.GELU(approximate="tanh")
        self.mlp = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    
    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            y: Optional[Tensor] = None, #(B, C)
            inference_params=None,
    ):
        r"""Pass the input signal through the encoder layer.
        
        Args:
            hidden_states: input signal (B, T, C) to the encoder layer (required)
            residual:  from previous block hidden_states = Mixer (LN(residual)) if residual is not None
            First we fuse add and norm before branching residual and hidden_states
            y: label embeddings (B, C)
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual) if isinstance (self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # I have residual <- supposed to be returned and hidden_state = norm(residual) <- this is supposed to go into mixer
        # shape (B, T, C)
        # transpose, scanning continuity, token reversal (for bidrectional bias)
        
        T = hidden_states.shape[1]
        w = h = int (np.sqrt(T))

        # unrelated but Usually raster store is B, C, H, W -> transpose B, C, W, H zigzag along dim=-1 gives vertical zig zag scan
        # intenally reconstructs B, C, H, W, orders in column major or row major scan in non overlapping windows of specified size
        column_first = True if self.transpose else False
        hidden_states = local_scan (hidden_states, w=4, H=h, W=w, column_first=column_first).contiguous() #(B, T, C) make sure to have contiguous layout so its safer to view/reshape without losing order

        if self.shift_window:
            # move content up-left, to break alignment patterns and break positional aliasing
            # (B, T, C)-> (B, H, W, C)
            hidden_states = rearrange(hidden_states, "b (h w) c -> b h w c", h=h)
            hidden_states = torch.roll(hidden_states, shifts=(-1, -1), dims=(1, 2)).reshape(-1, h * w, hidden_states.size(-1))
        
        if self.reverse:
            hidden_states = hidden_states.flip(1) # (B, T, C)
        
        # now we have hidden states in desired scan orders
        scale_ssm, gate_ssm, shift_ssm, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1) # (B, 6C) -> 6(B, C)

        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
            modulate(hidden_states, shift_ssm, scale_ssm), y, inference_params=inference_params
        )

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
        )

        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1) # (B, T, C)
        
        if self.shift_window:
            # convert to (B, H, W, C) undo the shift (induced to break positional aliasing and alignment patterns)
            hidden_states = rearrange(hidden_states, "b (h w) c -> b h w c", h=h) # (B, H, W, C)
            hidden_states = torch.roll (hidden_states, shifts=(1, 1), dims=(1, 2)).reshape(-1, h * w, hidden_states.size(-1))
        
        hidden_states = local_reverse(hidden_states, w=4, H=h, W=w, column_first=column_first)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class WaveFlBlock(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            skip=False,
            no_ffn=False,
            y_dim=None, # label embedding size
            window_scan=True,
            num_wavelet_lv=2,
    ):
        """
        Simple block wrapping a mixer with LayerNorm/RMSNorm and a residual connection.

        Just as before still pre norm residual block with slightly different structure than standard pre norm block.
        Solely for performance reasons, as we can fuse addition, back into residual pathway, and normalization.
        Residual is required unless its the first block.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        self.no_ffn = no_ffn
        self.window_scan = window_scan
        self.num_wavelet_lv = num_wavelet_lv
        y_dim = dim if y_dim is None else y_dim

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance (self.norm, (RMSNorm, nn.LayerNorm)), f"fused_add_norm is only supported for LayerNorm and RMSNorm"
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, 6 * dim if not self.no_ffn else 3 * dim, bias=True))

        if not self.no_ffn:
            self.norm_2 = norm_cls(dim)

            mlp_hidden_dim = int (4 * dim)
            gelu_approx = lambda:nn.GELU(approximate="tanh")
            self.mlp = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=gelu_approx, drop=0)
    
    # if output of DWT (B, T, C) is transformed to (B, C, H, W)
    # its a tiled representation of different frequency components
    # first row being all LLs and last row being all HHs because the column major indexing into concatenation trick
    # DWT(0 index) is always LL
    def _dwt_fast(self, x):
        # implementation supports only two consecutive DWT transformations
        T = x.size(1) # x(B, T, C)
        h = w = int(np.sqrt(T))
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        subbands = self.dwt(x) # xll, xlh, xhl, xhh where each has shape of [B, 4C, h/2, W/2]
        scale = 2**self.num_wavelet_lv
        patch_size = scale # receptive patch size DWT
        if self.num_wavelet_lv > 1:
            out = (self.dwt(subbands) / scale).chunk(patch_size * patch_size, dim=1) # (B, 16C, h/4, w/4) -> 16x(B, C, h/4, w/4)
            indices = []
            for i in range(patch_size * patch_size):
                # normally: value = row * 4 + column
                # here we want to have all LL components first, -> column wise vertical scan
                # same indices, values transpose
                # hence value = column(i) * 4 + row
                # indices [0, 4, 8, 12, 1, 5...]
                indices.append(i % 4 * patch_size + i // 4)
            out = torch.cat([out[i] for i in indices], dim=1) # 16x(B, C, h/4, w/4) -> (B, 16C, h/4, w/4) but ordered from LL to HH along dim=1
        else:
            out = subbands / scale
        
        return rearrange(out, "b (c p1 p2) h w -> b (h p1 w p2) c", p1=patch_size, p2=patch_size) # (B, 16C, h/4, w/4) -> (B, HxW, C)
    
    # We refrain from using IDWT
    def _idwt_fast(self, x):
        scale = 2**self.num_wavelet_lv
        patch_size = scale
        lowest_size = int(np.sqrt(x.size(1))) // patch_size
        subbands = rearrange(
            x * scale, "b (h p1 w p2) c -> b (c p1 p2) h w", p1=patch_size, p2=patch_size, h=lowest_size
        ).chunk(patch_size * patch_size, dim=1)
        if self.num_wavelet_lv > 1:
            indices = []
            for i in range(patch_size * patch_size):
                indices.append(i % 4 * patch_size + i // 4)
            subbands = torch.cat([subbands[i] for i in indices], dim=1)
            out = self.idwt(subbands)
            out = self.idwt(out)
        else:
            out = self.idwt(torch.cat(subbands, dim=1))
        return rearrange(out, "b c h w -> b (h w) c")  # [b, c, h, w]
    

    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            y: Optional[Tensor] = None,
            inference_params = None,
    ):
        r"""Pass the input signal (B, T, C) through the block
        
        Args:
        hidden_states: (B, T, C) input signal to the block (required)
        residual: residual from previous block, hidden_states = Mixer(LN(residual)) in previous block
        we fuse add and then norm before branching residual and mixer
        y: tensor (B, C) label embeddings. 
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # not none here
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # now I have residual <- supposed to be retured and hidden_states = norm(residual) <- supposed to go into mixer
        # shape (B, T, C)
        # transpose, scanning continuity, token reversal (for directional bias)
        
        # DWT block, processes frequeny domain representation of the signal
        # hidden_states <- signal (B, T, C) implicit (B, C, H, W)

        T = hidden_states.shape[1]
        h = w = int (np.sqrt(T)) # Original image size
        hidden_states = self._dwt_fast(hidden_states).contiguous() # (B, T, C) implicitly flattened in LL x4 LH x4 ... order
        patch_size = int(2**self.num_wavelet_lv)
        
        if self.window_scan:
            # perform a non overlaping window scan over each subband / freq component
            column_first = True if self.transpose else False # (column first scans LL LH HL HH of same component (LL0) tiled in the first column)
            # Internally Constructs tiled representation of frequency components, the same resolution as original signal (B, C, H, W) from (B, T, C)
            # Performs row_wise or column_wise scans in non overlapping windows.
            # each non overlapping window corresponds to a frequency component (subband) first being LL(LL(x)) = LL0
            hidden_states = local_scan(hidden_states, w=w//patch_size, H=h, W=w, column_first=column_first).contiguous()
        else:
            if self.transpose:
                hidden_states = rearrange(hidden_states, "b (h w) c -> b (w h) c", h=h, w=w)
        
        if self.scanning_continuity:
            # does not integrate with window scan over non overlapping windows (each corresponding to a subband (h=w=H/patchsize))
            # we'll set this pos False
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h-> b (w h) c", h=h, w=w) # (B, T, C)
        
        if self.reverse:
            hidden_states = hidden_states.flip(1)
        
        if not self.no_ffn:
            # there is ffn
            scale_ssm, gate_ssm, shift_ssm, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1) #(B,C)->(B,6C)->6x(B,C)
            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), y, inference_params=inference_params
            )
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
            )
        else:
            scale_ssm, gate_ssm, shift_ssm = self.adaLN_modulation(y).chunk(3, dim=-1) # (B,C)->(B,3C)->3x(B,C)
            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), y, inference_params=inference_params
            )
        
        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)
        
        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h -> b (w h) c", h=h, w=w)
        
        if self.window_scan:
            hidden_states = local_reverse(hidden_states, w=w // patch_size, H=h, W=w, column_first=column_first)
        else:
            if self.transpose:
                hidden_states = rearrange (hidden_states, "b (w h) c -> b (h w) c", h=h, w=w)
        
        # TODO: we refrain from IDWT for reasons explained in the paper
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    

class MoEBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"fused_add_norm only supports LayerNorm and RMSNorm"
        
    def forward(self, hidden_states:Tensor, residual: Optional[Tensor]=None, inference_params=None):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=None, **kwargs)


class FlBlockCombined(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            use_gated_mlp=True,
    ):
        """
        Block wrapping a parallel spatial and frequency ssm blocks with residual and hidden states from previous block.
        Still the standard prenorm block, just slightly different structure so we can fuse addition, back into residual pathway,
        and then normalization to branch towards the mixer.
        Purely for performance reasons as we can fuse add->norm
        Residual has to be specified unless its the very first block.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity

        self.norm = norm_cls(dim)

        # TODO: Ablate here, each branch having access to only half the channels and then concatenation
        self.spatial_mamba = FlBlockRaw(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            drop_path=0.0,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            transpose=transpose,
            scanning_continuity=scanning_continuity,
            y_dim=dim,
        )
    
        # TODo: Ablate they dont setup an MLP in freq mamba block maybe because relying on IDWT
        self.freq_mamba = WaveFlBlock(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            drop_path=0.0,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            reverse=False,
            transpose=reverse, # tranpose, # disable if only left to right scanning is used
            scanning_continuity=scanning_continuity,
            no_ffn=True,
            y_dim=dim,
            num_wavelet_lv=2,
        )

        # TODO: Ablate
        self.proj = CrossAttentionFusion(dim, num_heads=8, qkv_bias=True, swap_k=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"fused_add_norm is only supported for layer norm and rms norm"
        
        self.norm_2 = norm_cls(dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3*dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda:nn.GELU(approximate="tanh")
        if use_gated_mlp:
            self.mlp = GatedMLP (fan_in=dim, mlp_hidden_dim=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    
    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor],
            y: Optional[Tensor],
            inference_params=None,
    ):
        """
            Pass the input signal (B, T, C) through the FlBlock (spatial, freq ssm, ssm fusion)
            
            Args:
                hidden_states: input sequence to the FlBlock (required)
                residual: residual from previous block. hidden_states = Mixer(LN(residual))
                
            We fuse add (residual + hidden_states) then norm, before branching residual mixer
        """

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, # no drop path since residual is None (very first block)
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states), # drop path since residual is present
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # Not None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # Now we have hidden_states (B, T, C) = LN(residual) and residual
        
        x1, x2 = hidden_states.chunk(2, dim=-1) #2x(B, T, C/2)
        # TODO: Ablation, use the residual instead of discarding it
        x1, _ = self.spatial_mamba(x1, None, y, inference_params) # 2x(B, T, C/2)
        x2, _ = self.freq_mamba(x2, None, y, inference_params) # 2x(B, T, C/2)
        
        # TODO: Ablation, residual connection over fusion layer
        # TODO: Ablation, give options for fusion layer
        if isinstance(self.proj, CrossAttentionFusion):
            x = self.proj(x1, x2) # (B, T, C)
        else:
            x = torch.cat((x1, x2), dim=-1)
            x = self.proj(x) # (B, T, C/2)
        
        hidden_states = hidden_states + x
        # FFN
        scale_mp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(3, dim=-1) #3x(B, C)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mp)
        )

        return hidden_states, residual

class FlBlockRaw(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            y_dim=None,
            no_ffn=False,
    ):
        """
        
        A block around SSM mixer block with residual and LayerNorm/RMSNorm.
        Takes two inputs hidden_states and residua. Residual being from the preivous block.

        Still the standard prenorm block with slightly different structure, solely for performance reasons
        as we can fuse addition, back into residual path, and norm.

        Residual has to be specified except for very first block.

        We add hidden_states from mixer of previous block, into residual pathway and then normalize to branch residual and mixer.
        """

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        y_dim = dim if y_dim is None else y_dim
        
        self.no_ffn = no_ffn

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"fused_add_norm only supported for RMSNorm or LayerNorm"
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim if self.no_ffn else 6 * dim, bias=True))

        if not self.no_ffn:
            self.norm_2 = norm_cls(dim)

            mlp_hidden_dim = int (4 * dim)
            approx_gelu = lambda:nn.GELU(approximate="tanh")
            self.mlp = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    
    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            y: Optional[Tensor] = None,
            inference_params=None,
    ):
        """
        Pass the input signal (B, T, C) through the spatial SSM block.

        Args:
            hidden_states: input signal (B, T, C) to the block
            residual: residual from the previous block. hidden_states = Mixer(Norm(residual))
            We fuse add (residual, hidden_states) and then norm before branching residual and mixer

            y: class label embedding, guidance embedding
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # Now I have residual (<- To be returned (B, T, C)) and hidden_states after norm, to be processed by mixer
        
        T = hidden_states.shape[1]
        h = w = int (np.sqrt(T))
        if self.transpose:
            hidden_states = rearrange(hidden_states, "b (h w) c -> b (w h) c", h=h, w=w)
        
        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h -> b (w h) c", h=h, w=w) # (B, T, C)
        
        if self.reverse:
            hidden_states = hidden_states.flip(1) # flip along tokens

        if not self.no_ffn:
            scale_ssm, gate_ssm, shift_ssm, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1) #(B, 6C) -> 6x(B,C)

            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer (
                modulate(hidden_states, shift_ssm, scale_ssm), y, inference_params=inference_params
            )
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
            )
        else:
            scale_ssm, gate_ssm, shift_ssm = self.adaLN_modulation(y).chunk(3, dim=-1) #(B, 3C) -> 3x(B, C)

            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), y, inference_params=inference_params
            )
        
        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)
        
        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h -> b (w h) c", h=h, w=w)
        
        if self.transpose:
            hidden_states = rearrange (hidden_states, "b (w h) c -> b (h w) c", h=h, w=w)
        

        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
















        



        

def drop_path (x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """
    Randomly kill all activations for some examples and scale up survivors to retain same expected value, so we dont have to upscale
    during inference when dropout is disabled

    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)

    This is the same as the DropConnect implementation for EfficientNet, however, the original name is misleading
    as 'Drop Connect' is a different form of droupout in a separate paper.
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    This implementation opts to changing the layer and the argument names to 'drop path' and 'drop_prob' rather than
    DropConnect and survival_rate respectively
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (B, T, C) -> (B, 1, 1) so that works with diff dim tensors, not just 2D convnets
    # new tensor same device and dtype as x
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob) # A tensor of shape (B, [1]*(x.ndim-1)) containing 0/1 with keep_prob; broad castable
    
    # if we dont scale_by_keep, during training mask * x scales output down by keep_prob on average
    # to rectify we have to multiply by keep_prob during inference
    # with scaling (inverted dropout) mask = mask / keep_prob, it automatically rectifies by scaling surviving units up by keep_prob
    # so the expected value stays the same during train and test time
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob) 
    return x * random_tensor



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample/ example level dropout (when applied in main path of residual blocks)
    Kills all activations across all tokens/channels for some random examples in a batch
    """

    def __init__(self, drop_prob: float=0.0, scale_by_keep:bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"

