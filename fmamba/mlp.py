import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MLP (nn.Module):
    def __init__(
            self,
            fan_in,
            add_bias_linear: bool = False,
            gated_linear_unit: bool = True,
            is_expert: bool = False,
            layer_idx=None,
            device=None,
    ):
        super().__init__()

        self.layer = layer_idx
        fan_h1 = 4 * fan_in
        fan_h2 = 4 * fan_in

        # if this is gated lear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if gated_linear_unit:
            fan_h1 = fan_h1 * 2
        
        self.linear_fc1 = nn.Linear(fan_in, fan_h1, bias=add_bias_linear, device=device)
        self.linear_fc1.is_expert = is_expert

        if gated_linear_unit:
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.gelu(x[0]) * x[1]
            self.activation = glu
        else:
            self.activation = F.gelu
        
        self.linear_fc2 = nn.Linear(fan_h2, fan_in, bias=add_bias_linear, device=device)

    
    def forward (self, hidden_states, inference_params=None):
        intermediate = self.linear_fc1(hidden_states)
        intermediate = self.activation(intermediate)
        output = self.linear_fc2(intermediate)
        return output

class GatedMLP(nn.Module):
    def __init__(
            self,
            fan_in: int,
            fan_h: int = None,
            fan_out: int = None,
            act_layer = lambda:nn.GELU(approximate="tanh"),
            drop: float = 0.0,
            bias: bool = True,
    )-> None:
        super().__init__()
        fan_out = fan_out or fan_in # stores first truth value
        fan_h = fan_h or fan_in
        self.fc1 = nn.Linear(fan_in, 2*fan_h, bias=bias)
        self.fc2 = nn.Linear(fan_h, fan_out, bias=bias)
        self.act_layer = act_layer()
    
    def forward(self, x:Tensor)-> Tensor:
        x = self.fc1(x)
        x, scale = x.chunk(2, dim=-1)
        x = self.act_layer(x) * scale
        x = self.fc2(x)
        return x
        
