import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint
from timm.layers import use_fused_attn
from torch.jit import Final

class CrossAttentionFusion(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            n_embd: int,
            n_head: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm,
            swap_k = False,
    )-> None:
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // 2 // n_head # n_embd is split into 2 half sent for freq transform
        self.scale = self.head_size**(-0.5)
        self.fused_attn = use_fused_attn()
        self.swap_k = swap_k
        
        self.qkv1 = nn.Linear(n_embd // 2, 3 * n_embd // 2, bias=qkv_bias)
        self.norm_q1 = norm_layer(self.head_size) if qk_norm else nn.Identity()
        self.norm_k1 = norm_layer(self.head_size) if qk_norm else nn.Identity()

        self.qkv2 = nn.Linear(n_embd // 2, 3 * n_embd // 2, bias=qkv_bias)
        self.norm_q2 = norm_layer(self.head_size) if qk_norm else nn.Identity()
        self.norm_k2 = norm_layer(self.head_size) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(n_embd, n_embd) # concat(C/2 spatial, C/2 frequency) -> C project C back onto backbone
        self.proj_drop = nn.Dropout(proj_drop)

    def _compute_attention(self, q, k, v):
        if self.fused_attn:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            # B, nh, T, hs
            att = q @ k.transpose(-2, -1) * self.scale # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T) attention matrix
            att = F.softmax(att, dim=-1) # normalize aggregated contribution from all tokens for a specific token
            att = self.attn_drop(att)
            y = att @ v # (B, nh, T, hs)

        return y
    
    def forward(self)
