"""ResNet-style MAE feature encoder (PyTorch port of JAX `models/mae_model.py`).

Appendix A.3 of "Generative Modeling via Drifting" describes a customized MAE:
a convolutional ResNet encoder paired with a U-Net-style deconvolutional
decoder, trained to reconstruct randomly (2x2-patch) masked inputs. Only the
encoder is used at drift-loss time, where it provides rich multi-scale,
multi-location features via :meth:`MAEResNet.get_activations`.

GroupNorm replaces BatchNorm; all residual blocks are "basic" (two 3x3 convs).
For latent-space generation the ResNet runs directly on the input; for
pixel-space high-res inputs an ``input_patch_size`` first patchifies the image
into channels so the ResNet always operates on a 32x32 grid. For CIFAR-10
pixel space the input is already 32x32, so ``input_patch_size=1``.

Tensors use the PyTorch NCHW convention.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


def _gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(_choose_gn_groups(num_channels, max_groups), num_channels)


def safe_std(x, dim, eps: float = 1e-6, keepdim: bool = False):
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(var.clamp(min=0.0) + eps)

#----------------------------------------------------------------------------
# Encoder.

class _BasicBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(filters)
        self.drop = nn.Dropout(dropout_prob)
        self.needs_proj = (stride != 1) or (in_channels != filters)
        if self.needs_proj:
            self.proj_conv = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
            self.proj_gn = _gn(filters)

    def forward(self, x):
        residual = x
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.drop(y)
        y = self.gn2(self.conv2(y))
        if self.needs_proj:
            residual = self.proj_gn(self.proj_conv(residual))
        return F.relu(residual + y)


class _ResNetEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=64, layers=(2, 2, 2, 2), dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False)
        self.gn1 = _gn(base_channels)

        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        ch = base_channels
        for stage_idx, num_blocks in enumerate(layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = ch * (2 ** stage_idx) if stage_idx > 0 else ch
            in_ch = ch if stage_idx == 0 else ch * (2 ** (stage_idx - 1))
            blocks = [_BasicBlock(in_ch, out_ch, stride=stride, dropout_prob=dropout_prob)]
            for _ in range(1, num_blocks):
                blocks.append(_BasicBlock(out_ch, out_ch, stride=1, dropout_prob=dropout_prob))
            self.stages.append(nn.ModuleList(blocks))
            self.stage_norms.append(_gn(out_ch))

    def forward(self, x, return_block_outputs: bool = False):
        feats: Dict[str, torch.Tensor] = {}
        block_outputs: Dict[str, List[torch.Tensor]] = {}
        x = F.relu(self.gn1(self.conv1(x)))
        feats['conv1'] = x
        for i, (blocks, norm) in enumerate(zip(self.stages, self.stage_norms)):
            outs = []
            for block in blocks:
                x = block(x)
                outs.append(x)
            block_outputs[f'layer{i + 1}'] = outs
            feats[f'layer{i + 1}'] = norm(x)
        if return_block_outputs:
            return feats, block_outputs
        return feats

#----------------------------------------------------------------------------
# Decoder (U-Net style).

class _ConvGNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False)
        self.gn = _gn(out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        concat_channels = in_channels + skip_channels
        self.concat_norm = _gn(concat_channels)
        self.proj = _ConvGNReLU(concat_channels, out_channels, kernel=3)
        self.refine = _ConvGNReLU(out_channels, out_channels, kernel=3)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm(x)
        return self.refine(self.proj(x))


class _UNetDecoder(nn.Module):
    def __init__(self, base_channels, out_channels):
        super().__init__()
        c1 = base_channels
        c2 = base_channels
        c3 = base_channels * 2
        c4 = base_channels * 4
        c5 = base_channels * 8
        self.bridge = _ConvGNReLU(c5, c5)
        self.up43 = _UpBlock(c5, c4, c4)
        self.up32 = _UpBlock(c4, c3, c3)
        self.up21 = _UpBlock(c3, c2, c2)
        self.up10 = _UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, feats):
        x = self.bridge(feats['layer4'])
        x = self.up43(x, feats['layer3'])
        x = self.up32(x, feats['layer2'])
        x = self.up21(x, feats['layer1'])
        x = self.up10(x, feats['conv1'])
        return self.head(x)

#----------------------------------------------------------------------------
# Patch / mask helpers.

def patch_input(x, p: int):
    if p == 1:
        return x
    N, C, H, W = x.shape
    x = x.reshape(N, C, H // p, p, W // p, p)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(N, C * p * p, H // p, W // p)


def make_patch_mask(x, mask_ratio, patch_size: int):
    # x: [N, C, H, W] -> per-2x2-patch boolean mask [N, 1, H, W].
    N, _, H, W = x.shape
    nh, nw = H // patch_size, W // patch_size
    noise = torch.rand(N, 1, nh, nw, device=x.device, dtype=x.dtype)
    mask = (noise < mask_ratio.view(N, 1, 1, 1)).to(x.dtype)
    mask = mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return mask

#----------------------------------------------------------------------------
# MAE model.

@persistence.persistent_class
class MAEResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        base_channels=128,
        patch_size=2,               # masking patch size (zero out 2x2 patches)
        input_patch_size=1,         # patchify high-res input into channels
        layers=(3, 4, 6, 3),
        dropout_prob=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.patch_size = patch_size
        self.input_patch_size = input_patch_size
        enc_in = in_channels * input_patch_size * input_patch_size
        self.encoder = _ResNetEncoder(enc_in, base_channels=base_channels, layers=layers, dropout_prob=dropout_prob)
        self.decoder = _UNetDecoder(base_channels=base_channels, out_channels=enc_in)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x, labels, lambda_cls=0.0, mask_ratio_min=0.5, mask_ratio_max=0.5):
        """Reconstruct masked input; returns (loss, metrics)."""
        x = patch_input(x, self.input_patch_size)
        b = x.shape[0]
        mask_ratio = torch.rand(b, device=x.device, dtype=x.dtype) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask = make_patch_mask(x, mask_ratio, self.patch_size)
        x_in = x * (1.0 - mask)

        feats = self.encoder(x_in)
        pooled = feats['layer4'].mean(dim=(2, 3))
        logits = self.fc(pooled)
        recon = self.decoder(feats)

        cls_loss = F.cross_entropy(logits, labels, reduction='none')
        mse = (recon - x) ** 2
        denom = mask.expand_as(mse).sum(dim=(1, 2, 3)) + 1e-8
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / denom
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss

        metrics = dict(
            loss=loss.mean().detach(),
            cls_loss=cls_loss.mean().detach(),
            recon_loss=recon_loss.mean().detach(),
            accuracy=(logits.argmax(dim=-1) == labels).float().mean().detach(),
            mask_ratio=mask.mean().detach(),
        )
        return loss.mean(), metrics

    def _process_feat(self, out, name, feat, patch_mean_size, patch_std_size, use_mean, use_std):
        B, C, H, W = feat.shape
        out[name] = feat.reshape(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        if use_mean:
            out[f'{name}_mean'] = feat.mean(dim=(2, 3))[:, None, :]
        if use_std:
            out[f'{name}_std'] = safe_std(feat, dim=(2, 3))[:, None, :]
        for size in patch_mean_size:
            if H % size == 0 and W % size == 0:
                r = feat.reshape(B, C, H // size, size, W // size, size)
                r = r.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // size) * (W // size), size * size, C)
                out[f'{name}_mean_{size}'] = r.mean(dim=2)
        for size in patch_std_size:
            if H % size == 0 and W % size == 0:
                r = feat.reshape(B, C, H // size, size, W // size, size)
                r = r.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // size) * (W // size), size * size, C)
                out[f'{name}_std_{size}'] = safe_std(r, dim=2)

    def get_activations(
        self,
        x,
        patch_mean_size: Optional[List[int]] = (2, 4),
        patch_std_size: Optional[List[int]] = (2, 4),
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> Dict[str, torch.Tensor]:
        """Multi-scale / multi-location features used by the drift loss.

        Each returned value has shape ``[B, T, D]`` (T tokens, D channels).
        Gradients flow through this method (the generator is trained through it).
        """
        patch_mean_size = list(patch_mean_size or [])
        patch_std_size = list(patch_std_size or [])
        x = patch_input(x, self.input_patch_size)
        need_blocks = isinstance(every_k_block, (int, float)) and not math.isinf(float(every_k_block)) and every_k_block >= 1
        if need_blocks:
            feats, block_outputs = self.encoder(x, return_block_outputs=True)
        else:
            feats = self.encoder(x)
            block_outputs = {}

        out: Dict[str, torch.Tensor] = {}
        out['norm_x'] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[:, None, :]
        for name, feat in feats.items():
            self._process_feat(out, name, feat, patch_mean_size, patch_std_size, use_mean, use_std)
        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                blocks = block_outputs.get(f'layer{i}', [])
                for blk_idx, feat_i in enumerate(blocks, start=1):
                    if blk_idx % k == 0:
                        self._process_feat(out, f'layer{i}_blk{blk_idx}', feat_i,
                                           patch_mean_size, patch_std_size, use_mean, use_std)
        return out

#----------------------------------------------------------------------------
