import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x): #x: 4,79,1024
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"

        b = x.size(0)
        running_mean = self.running_mean.repeat(b) #4096 #tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')  
        running_var = self.running_var.repeat(b) #4096 #tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')

        # Apply instance norm
        x_reshaped = rearrange(x.contiguous(), "B N D -> 1 (B D) N")
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias,True, self.momentum, self.eps)
        
        out = rearrange(out, "1 (B D) N -> B N D", B=b)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

# classes
class PreInstanceNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = AdaptiveInstanceNorm2d(dim, eps=1e-06)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # mask: [B, N] -> [B, H, N, N]
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask * mask.transpose(-1, -2)
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.heads, 1, 1)
            assert mask.dim() == 4
            dots = dots.masked_fill(mask==0, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreInstanceNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreInstanceNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x

