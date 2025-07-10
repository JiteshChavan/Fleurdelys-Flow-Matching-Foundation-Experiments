import torch.nn as nn

def modulate (x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # to broadcast along tokens, modulators might be pooled across tokens

class PosCNN(nn.Module):
    def __init__(self, in_channels, n_embd=768, stride=1):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, n_embd, kernel_size=3, stride=stride, padding=1, bias=True, groups=n_embd))
        self.stride = stride

    def forward(self, x, H, W):
        B, T, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.stride == 1: # same spatial resolution, res connection works
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat) # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)

        return x
    
    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]

class AdaInPosCNN(nn.Module):
    def __init__(self, in_channels, n_embd=768, stride=1):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, n_embd, 3, stride, 1, bias=True, groups=n_embd)) # sequential container for the no weight decay fucntion compatibility
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, 2*n_embd, bias=True))
        self.norm = nn.LayerNorm(n_embd)
        self.stride = stride
    
    def forward(self, x, y, H, W):
        B, T, C = x.shape
        x = x.transpose(1,2).view(B, C, H, W)
        shift, scale = self.adaLN_modulation(y).chunk (2, dim=1)

        if self.stride == 1:
            x = self.proj(x) + x
        else:
            x = self.proj(x)
        
        x = x.flatten(2).transpose(1, 2)
        x = modulate(self.norm(x), shift, scale)

        return x
    
    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]
    