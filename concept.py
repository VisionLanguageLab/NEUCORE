import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed, VisionTransformer


class MILAttention(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        A = self.attention(x).transpose(2, 1)
        A = F.softmax(A, dim=-1)
        M = torch.bmm(A, x).squeeze(1)

        return M

class MyVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, init_values=None, class_token=True, no_embed_class=False, pre_norm=False, fc_norm=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        super().__init__(img_size, patch_size, in_chans, num_classes, global_pool, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, init_values, class_token, no_embed_class, pre_norm, fc_norm, drop_rate, attn_drop_rate, drop_path_rate, weight_init, embed_layer, norm_layer, act_layer, block_fn)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed[:, x.shape[1]]
        return self.pos_drop(x)

    def forward(self, x):
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)

        tokens = self.fc_norm(x[:, self.num_prefix_tokens:])
        return tokens

class ConceptNet(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()

        self.encoder = MyVisionTransformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.fg_net = MILAttention(embed_dim, embed_dim//2)
        self.bg_net = MILAttention(embed_dim, embed_dim//2)

    def forward(self, r, t=None):
        r_token = self.encoder(r)
        r_fg = self.fg_net(r_token)
        r_bg = self.bg_net(r_token)

        if t is not None:
            t_token = self.encoder(t)
            t_fg = self.fg_net(t_token)
            t_bg = self.bg_net(t_token)

            rt = torch.cat((r, t), dim=1)
            rt_token = self.encoder(rt)
            rt_fg = self.fg_net(rt_token)

            return rt_fg, r_token, r_fg, r_bg, t_token, t_fg, t_bg
        return r_token, r_fg, r_bg
