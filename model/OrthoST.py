# -*- coding: utf-8 -*-
'''
@Time    : 2025/3/3 9:48
@Author  : Linjie Wang
@FileName: OrthoST.py
@Software: PyCharm
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class GeneAttention(nn.Module):
    """Gene-wise self-attention module."""

    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.apply(init_weights)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv

        # Gene attention
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2).unsqueeze(1),
            k.transpose(1, 2).unsqueeze(1),
            v.transpose(1, 2).unsqueeze(1),
            dropout_p=0.0, is_causal=False
        )

        x = x.squeeze(1).transpose(1, 2)
        return self.proj(x)

    def get_attn(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, _ = qkv

        q = q * (N ** -0.5)
        attn = F.softmax(torch.einsum('bnc,bnd->bcd', q, k), dim=-1)

        return self.qkv.weight, attn

# class ChannelAttention(nn.Module):
#     r""" Channel based self attention.
#
#     Args:
#         dim (int): Number of input channels.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#     """
#
#     def __init__(self, dim, qkv_bias=False):
#         super().__init__()
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#
#         self.qkv.apply(init_weights)
#         self.proj.apply(init_weights)
#
#     def forward(self, x):
#         B, N, C = x.shape
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         x = F.scaled_dot_product_attention(q.transpose(1, 2).unsqueeze(1), k.transpose(1, 2).unsqueeze(1),
#                                                v.transpose(1, 2).unsqueeze(1),
#                                                dropout_p=0.0, is_causal=False)
#         x = x.squeeze(1).transpose(1, 2)  # back to (B, N, C)
#         x = self.proj(x)
#         return x
#
#     def get_attn(self, x):
#         B, N, C = x.shape
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         q = q * (N ** -0.5)
#         attention = F.softmax(torch.einsum('bnc,bnd->bcd', q, k), dim=-1)  # 保持原get_attn用einsum
#
#         return self.qkv.weight, attention

class WGABlock(nn.Module):
    """Windows gene-wise Transformer block."""

    def __init__(self, dim, mlp_ratio=4.0, qkv_bias=True, dropout=0.0,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm, use_ffn=True):
        super().__init__()

        self.use_ffn = use_ffn
        self.norm_1 = norm_layer(dim)
        self.attn = GeneAttention(dim, qkv_bias)
        self.drop_1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_ffn:
            hidden_dim = int(dim * mlp_ratio)
            self.norm_2 = norm_layer(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, dim))
            self.drop_2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.apply(init_weights)

    def forward(self, x):
        x = x + self.drop_1(self.attn(self.norm_1(x)))
        if self.use_ffn:
            x = x + self.drop_2(self.mlp(self.norm_2(x)))
        return x

    def get_attn(self, x):
        x_norm = self.norm_1(x)
        return self.attn.get_attn(x_norm)

# class ChannelBlock(nn.Module):
#     r""" Channel-wise Local Transformer Block.
#
#     Args:
#         dim (int): Number of input channels.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         ffn (bool): If False, pure attention network without FFNs
#     """
#     def __init__(self, dim, mlp_ratio=4., qkv_bias=True,
#                  dropout=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
#                  ffn=True):
#         super().__init__()
#
#         self.ffn = ffn
#         self.norm1 = norm_layer(dim)
#         self.attn = ChannelAttention(dim, qkv_bias=qkv_bias)
#         self.dropout_1 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#         self.dropout_2 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#
#         if self.ffn:
#             self.norm2 = norm_layer(dim)
#             mlp_hidden_dim = int(dim * mlp_ratio)
#             self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim),
#                                      act_layer(),
#                                      nn.Dropout(dropout),
#                                      nn.Linear(mlp_hidden_dim, dim))
#
#     def forward(self, x):
#         x = x + self.dropout_1(self.attn(self.norm1(x)))  # Pre norm
#
#         if self.ffn:
#             x = x + self.dropout_2(self.mlp(self.norm2(x)))
#         return x
#
#     def get_attn(self, x):
#         cur = self.norm1(x)
#         qkv_weight, attn = self.attn.get_attn(cur)
#         return qkv_weight, attn

class Encoder(nn.Module):
    def __init__(self, dims=None):
        super().__init__()

        if dims is None:
            dims = [3000, 128, 30]

        in_dim, hidden_dim, latent_dim = dims

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

        # WGA blocks
        self.WGA_blocks = nn.ModuleList([
            WGABlock(hidden_dim, mlp_ratio=2.0, dropout=0.1),
            WGABlock(hidden_dim, mlp_ratio=2.0, dropout=0.1)
        ])

        # WSA blocks
        def WSABlock():
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation="relu",
                batch_first=True,
                norm_first=True
            )
            return nn.TransformerEncoder(layer, num_layers=1)

        self.WSA_blocks = nn.ModuleList([
            WSABlock(),
            WSABlock()
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.dropout(x)

        # Interleaved spatial + channel attention
        for spatial, gene in zip(self.WSA_blocks, self.WGA_blocks):
            x = spatial(x)
            x = gene(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling

        return self.mlp(x)

    def get_attn(self, x):
        x = self.input_proj(x)
        linear_weight = self.input_proj.weight

        # First block
        x = self.WSA_blocks[0](x)
        qkv1, attn1 = self.WGA_blocks[0].get_attn(x)
        x = self.WGA_blocks[0](x)

        # Second block
        x = self.WSA_blocks[1](x)
        qkv2, attn2 = self.WGA_blocks[1].get_attn(x)

        return linear_weight, qkv1, attn1, qkv2, attn2
# class Encoder(nn.Module):
#     def __init__(self, cfg=None):
#         super(Encoder, self).__init__()
#
#         if cfg is None:
#             cfg = [3000, 128, 30]
#         self.x_enc = nn.Linear(cfg[0], cfg[1], bias=False)
#         self.dropout = nn.Dropout(0.1)
#
#         # gene attn
#         layer = WGABlock(cfg[1], mlp_ratio=2.0, dropout=0.1)
#         self.add_module("WGA_1", layer)
#         layer = WGABlock(cfg[1], mlp_ratio=2.0, dropout=0.1)
#         self.add_module("WGA_2", layer)
#
#         FormerBlock = nn.TransformerEncoderLayer(
#             d_model=cfg[1],
#             nhead=4,
#             dim_feedforward=cfg[1]*2,
#             dropout=0.1,
#             activation='relu',
#             batch_first=True,
#             norm_first=True
#         )
#         layer = nn.TransformerEncoder(FormerBlock, num_layers=1)
#         self.add_module("WSA_1", layer)
#
#         FormerBlock = nn.TransformerEncoderLayer(
#             d_model=cfg[1],
#             nhead=4,
#             dim_feedforward=cfg[1]*2,
#             dropout=0.1,
#             activation='relu',
#             batch_first=True,
#             norm_first=True
#         )
#         layer = nn.TransformerEncoder(FormerBlock, num_layers=1)
#         self.add_module("WSA_2", layer)
#
#         self.norm = nn.LayerNorm(cfg[1])
#
#         layers = []
#         layers.append(nn.Linear(cfg[1], cfg[2]))
#         layers.append(nn.BatchNorm1d(cfg[2]))
#         layers.append(nn.LeakyReLU())
#         layers.append(nn.Linear(cfg[2], cfg[2]))
#         self.mlp = nn.Sequential(*layers)
#
#         self.apply(init_weights)
#         self.mlp.apply(init_weights)
#         self.x_enc.apply(init_weights)
#
#     def forward(self, x_emb, mask=None):
#         x_emb = self.x_enc(x_emb)
#         x_emb = self.dropout(x_emb)
#
#         layer = self.__getattr__("WSA_1")
#         x_emb = layer(x_emb, src_key_padding_mask=mask)
#
#         layer = self.__getattr__("WGA_1")
#         x_emb = layer(x_emb)
#
#         layer = self.__getattr__("WSA_1")
#         x_emb = layer(x_emb, src_key_padding_mask=mask)
#
#         layer = self.__getattr__("WGA_1")
#         x_emb = layer(x_emb)
#
#         x_emb = self.norm(x_emb)
#         x_emb = x_emb.mean(dim=1)  # avg pool
#         x_emb = self.mlp(x_emb)
#
#         return x_emb
#
#     def get_attn(self, x_emb):
#         x_emb = self.x_enc(x_emb)
#         linear_attn = self.x_enc.weight
#
#         layer = self.__getattr__("WSA_1")
#         x_emb = layer(x_emb)
#
#         layer = self.__getattr__("WGA_1")
#         qkv_1, attn_1 = layer.get_attn(x_emb)
#         x_emb = layer(x_emb)
#
#         layer = self.__getattr__("WSA_2")
#         x_emb = layer(x_emb)
#
#         layer = self.__getattr__("WGA_2")
#         qkv_2, attn_2 = layer.get_attn(x_emb)
#
#         return linear_attn, qkv_1, attn_1, qkv_2, attn_2

class Decoder(nn.Module):
    def __init__(self, dims=[30, 256]):
        super().__init__()

        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU()
            ])

        self.mlp = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.mlp(x)
# class Decoder(nn.Module):
#     def __init__(self, cfg=[30, 256]):
#         super(Decoder, self).__init__()
#
#         self.num_layer = len(cfg)
#         layers = []
#         for i in range(self.num_layer - 1):
#             layers.append(nn.Linear(cfg[i], cfg[i + 1]))
#             layers.append(nn.BatchNorm1d(cfg[i + 1]))
#             layers.append(nn.LeakyReLU())
#
#         self.mlp = nn.Sequential(*layers)
#         self.mlp.apply(init_weights)  # initialize weights
#
#     def forward(self, x):
#         x = self.mlp(x)
#         return x

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super().__init__()

        self.layer = nn.Linear(in_dim, out_dim)
        self.activation = activation

        self.apply(init_weights)

    def forward(self, x):
        x = self.layer(x)

        if self.activation is None:
            return x

        act_map = {
            "relu": F.relu,
            "elu": F.elu,
            "gelu": F.gelu,
            "leaky_relu": F.leaky_relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "softplus": F.softplus,
            "softmax": lambda t: F.softmax(t, dim=-1),
        }
        if self.activation not in act_map:
            raise ValueError(f"Unsupported activation: {self.activation}")

        return act_map[self.activation](x)
# class Porj_NN(nn.Module):
#     def __init__(self, input_dim, output_dim, activation=None):
#         super(Porj_NN, self).__init__()
#         self.activation = activation
#         self.layer = nn.Linear(input_dim, output_dim)
#         self.apply(init_weights)
#
#     def forward(self, x):
#         x = self.layer(x)
#         if self.activation == "relu":
#             x = F.relu(x, inplace=True)
#         elif self.activation == "elu":
#             x = F.elu(x, inplace=True)
#         elif self.activation == "gelu":
#             x = F.gelu(x)
#         elif self.activation == "leaky_relu":
#             x = F.leaky_relu(x, inplace=True)
#         elif self.activation == "sigmoid":
#             x = torch.sigmoid(x)
#         elif self.activation == "exp_norm":
#             x = torch.exp(x - x.max(dim=1)[0].unsqueeze(1))
#         elif self.activation == "tanh":
#             x = torch.tanh(x)
#         elif self.activation == "softmax":
#             softmax = torch.nn.Softmax(dim=-1)
#             x = softmax(x)
#         elif self.activation == "softplus":
#             x = F.softplus(x)
#         elif self.activation == "gumbel":
#             x = F.gumbel_softmax(x, hard=True)#
#         elif self.activation is None:
#             pass
#         else:
#             assert TypeError
#
#         return x


class OrthoST(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = Encoder([input_dim, hidden_dim, latent_dim])

        # Contrastive projection
        self.cl_proj = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.ELU()
        )

        self.decoder = Decoder([latent_dim, hidden_dim])
        self.reconstruct = ProjectionHead(hidden_dim, input_dim, activation="softplus")

        self.apply(init_weights)

    def forward_cl(self, x_anchor, x_pos, x_neg):
        z0 = self.cl_proj(self.encoder(x_anchor))
        z1 = self.cl_proj(self.encoder(x_pos))
        z2 = self.cl_proj(self.encoder(x_neg))
        return z0, z1, z2

    def forward_indiv(self, x):
        h = self.encoder(x)
        x_recon = self.reconstruct(self.decoder(h))
        z = self.cl_proj(h)
        return h, z, x_recon

    def get_emb(self, x):
        h = self.encoder(x)
        return h, self.cl_proj(h)

# class OrthoST(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(OrthoST, self).__init__()
#
#         self.encoder = Encoder([input_dim, hidden_dim, latent_dim])
#
#         # 对比学习投影头
#         self.cl_proj = nn.Sequential(
#             nn.BatchNorm1d(latent_dim),
#             nn.ELU(),
#         )
#         self.cl_proj.apply(init_weights)
#
#         self.decoder = Decoder([latent_dim, hidden_dim])
#         self.proj_x = Porj_NN(hidden_dim, input_dim, activation="softplus")
#
#     def forward_CL(self, x_group, x_pos, x_neg):
#
#         h0 = self.cl_proj(self.encoder(x_group))
#         h1 = self.cl_proj(self.encoder(x_pos))
#         h2 = self.cl_proj(self.encoder(x_neg))
#
#         return h0, h1, h2
#
#     def forward_indiv(self, x_group):
#         latent_group = self.encoder(x_group)
#         x_recon = self.proj_x(self.decoder(latent_group))
#         h0 = self.cl_proj(latent_group)
#         return latent_group, h0, x_recon
#
#
#     def get_emb(self, x_group):
#         latent_group = self.encoder(x_group)
#         h0 = self.cl_proj(latent_group)
#         return latent_group, h0
#
#     def get_recon(self, x_group):
#         return self.proj_x(self.decoder(self.encoder(x_group)))

