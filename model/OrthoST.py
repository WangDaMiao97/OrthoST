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
