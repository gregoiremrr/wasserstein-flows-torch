"""LightningDiT generator (PyTorch port of the JAX `models/generator.py`).

A DiT-style one-step generator following "Generative Modeling via Drifting"
(Appendix A.2) and the LightningDiT recipe (Yao et al., 2025): SwiGLU MLPs,
RoPE, RMSNorm, QK-Norm, adaLN-zero conditioning, learnable register
("in-context") tokens, and StyleGAN-like random style tokens. The network maps
Gaussian noise (plus class / CFG / style conditioning) directly to an image.

Tensors use the PyTorch NCHW convention. The transformer operates on a token
sequence of [register tokens | image patches].
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence

#----------------------------------------------------------------------------
# 2D sin-cos positional embedding (DiT/MAE style).

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

#----------------------------------------------------------------------------
# Building blocks.

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.to(dtype)


def modulate(x, shift, scale):
    # x: [B, N, C]; shift/scale: [B, C].
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(q, k):
    # q, k: [B, N, H, D]; rotate along the head dimension.
    B, N, H, D = q.shape
    half = D // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / half))
    t = torch.arange(N, device=q.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                       # [N, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)             # [N, D]
    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]

    def rotate_half(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    qf, kf = q.float(), k.float()
    q_out = qf * cos + rotate_half(qf) * sin
    k_out = kf * cos + rotate_half(kf) * sin
    return q_out.to(q.dtype), k_out.to(k.dtype)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=False, use_rope=False, attn_fp32=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.attn_fp32 = attn_fp32
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, N, H, D]
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, k)

        if self.attn_fp32:
            q, k, v = q.float(), k.float(), v.float()
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C).to(self.proj.weight.dtype)
        return self.proj(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim, bias=True)
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='none'))


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, cond_dim, mlp_ratio=4.0,
                 use_qknorm=False, use_swiglu=False, use_rmsnorm=False,
                 use_rope=False, attn_fp32=True):
        super().__init__()
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              qk_norm=use_qknorm, use_rope=use_rope, attn_fp32=attn_fp32)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            hid = int(2 / 3 * mlp_hidden_dim)
            hid = (hid + 31) // 32 * 32
            self.mlp = SwiGLUFFN(hidden_size, hid)
        else:
            self.mlp = MLP(hidden_size, mlp_hidden_dim)

        self.adaLN = nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, c):
        chunks = self.adaLN(F.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks.chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cond_dim, use_rmsnorm=False):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.adaLN = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(F.silu(c)).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

#----------------------------------------------------------------------------
# Timestep / CFG-scale embedder.

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)

#----------------------------------------------------------------------------
# Main generator.

@persistence.persistent_class
class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        out_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        cond_dim=384,
        num_classes=10,
        n_cls_tokens=16,            # register / in-context tokens
        noise_classes=64,           # style-token codebook size (0 disables)
        noise_coords=32,            # number of style tokens
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        attn_fp32=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.n_cls_tokens = n_cls_tokens
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords

        grid = input_size // patch_size
        self.grid = grid
        self.num_patches = grid * grid

        # Patch embedding (linear on flattened patches).
        patch_dim = patch_size * patch_size * in_channels
        self.patch_embed = nn.Linear(patch_dim, hidden_size, bias=True)
        # Learnable positional embedding, sincos-initialized (matches the official
        # repo, which trains pos_embed as an nn.Parameter rather than freezing it).
        pos = get_2d_sincos_pos_embed(hidden_size, grid)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos).float()[None])

        # Register tokens.
        if n_cls_tokens > 0:
            self.cls_proj = nn.Linear(cond_dim, hidden_size, bias=True)
            self.cls_embed = nn.Parameter(torch.randn(1, n_cls_tokens, hidden_size) * 0.02)

        # Conditioning embeddings.
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        if noise_classes > 0 and noise_coords > 0:
            self.noise_embeds = nn.ModuleList([
                nn.Embedding(noise_classes, cond_dim) for _ in range(noise_coords)
            ])
            for emb in self.noise_embeds:
                nn.init.normal_(emb.weight, std=0.02)
        self.cfg_embedder = TimestepEmbedder(cond_dim)
        self.cfg_norm = RMSNorm(cond_dim)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, num_heads, cond_dim, mlp_ratio=mlp_ratio,
                              use_qknorm=use_qknorm, use_swiglu=use_swiglu,
                              use_rmsnorm=use_rmsnorm, use_rope=use_rope, attn_fp32=attn_fp32)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, cond_dim, use_rmsnorm=use_rmsnorm)

        self._init_weights()

    # -- weight init ---------------------------------------------------------

    def _init_weights(self):
        # Xavier-uniform on every Linear (weights) with zero bias, matching the
        # official repo's TorchLinear default. Embeddings keep their normal(0,0.02)
        # init; RMSNorm scales stay at 1. The adaLN modulation and the final
        # projection are then re-zeroed (adaLN-zero: blocks start as identity).
        def _xavier(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_xavier)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN.weight)
            nn.init.zeros_(block.adaLN.bias)
        nn.init.zeros_(self.final_layer.adaLN.weight)
        nn.init.zeros_(self.final_layer.adaLN.bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # -- conditioning --------------------------------------------------------

    def make_cond(self, class_idx, cfg_scale, style_idx):
        B = class_idx.shape[0]
        cond = self.class_embed(class_idx)
        if self.noise_classes > 0 and self.noise_coords > 0:
            for i, emb in enumerate(self.noise_embeds):
                cond = cond + emb(style_idx[:, i])
        if not torch.is_tensor(cfg_scale):
            cfg_scale = torch.full((B,), float(cfg_scale), device=class_idx.device)
        cfg_emb = self.cfg_norm(self.cfg_embedder(cfg_scale))
        cond = cond + cfg_emb * 0.02
        return cond

    # -- patch helpers -------------------------------------------------------

    def patchify(self, x):
        N, C, H, W = x.shape
        p, g = self.patch_size, self.grid
        x = x.reshape(N, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(N, g * g, p * p * C)
        return x

    def unpatchify(self, x):
        N = x.shape[0]
        p, g, c = self.patch_size, self.grid, self.out_channels
        x = x.reshape(N, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(N, c, g * p, g * p)
        return x

    # -- forward -------------------------------------------------------------

    def forward(self, noise, cond):
        """noise: [B, C, H, W] Gaussian. cond: [B, cond_dim]. Returns [B, out, H, W]."""
        x = self.patch_embed(self.patchify(noise))
        x = x + self.pos_embed.to(x.dtype)
        if self.n_cls_tokens > 0:
            c_tokens = self.cls_proj(cond).unsqueeze(1).expand(-1, self.n_cls_tokens, -1)
            c_tokens = c_tokens + self.cls_embed
            x = torch.cat([c_tokens, x], dim=1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_layer(x, cond)
        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens:]
        return self.unpatchify(x)

#----------------------------------------------------------------------------
