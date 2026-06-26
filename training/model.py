"""Drifting generator model and one-step sampler.

`DriftingModel` wraps the LightningDiT network and exposes a single-pass
("1-NFE") generation interface. Generation draws Gaussian noise (and discrete
style indices) internally and maps them, together with class / CFG
conditioning, to an image. Unlike diffusion/flow models there is no iterative
sampler: inference is one forward pass.
"""

import torch
from torch_utils import persistence
from training.networks_dit import LightningDiT

#----------------------------------------------------------------------------

@persistence.persistent_class
class DriftingModel(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        patch_size=2,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        cond_dim=None,
        n_cls_tokens=16,
        noise_classes=64,
        noise_coords=32,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        attn_fp32=True,
        use_fp16=False,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.num_classes = max(label_dim, 1)
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords
        self.use_fp16 = use_fp16
        cond_dim = cond_dim if cond_dim is not None else hidden_size

        self.net = LightningDiT(
            input_size=img_resolution,
            patch_size=patch_size,
            in_channels=img_channels,
            out_channels=img_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            cond_dim=cond_dim,
            num_classes=self.num_classes,
            n_cls_tokens=n_cls_tokens,
            noise_classes=noise_classes,
            noise_coords=noise_coords,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            attn_fp32=attn_fp32,
        )

    @property
    def device(self):
        return self.net.class_embed.weight.device

    def generate(self, class_idx, cfg_scale=1.0, noise=None, style_idx=None, generator=None):
        """Single-pass generation. ``class_idx``: [B] long. Returns [B, C, H, W]."""
        device = self.device
        class_idx = class_idx.to(device).long()
        B = class_idx.shape[0]
        if noise is None:
            noise = torch.randn(B, self.img_channels, self.img_resolution, self.img_resolution,
                                device=device, generator=generator)
        noise = noise.to(device)
        if style_idx is None and self.noise_classes > 0 and self.noise_coords > 0:
            style_idx = torch.randint(0, self.noise_classes, (B, self.noise_coords),
                                      device=device, generator=generator)
        cond = self.net.make_cond(class_idx, cfg_scale, style_idx)
        return self.net(noise, cond)

    def forward(self, class_idx, cfg_scale=1.0, noise=None, style_idx=None):
        return self.generate(class_idx, cfg_scale=cfg_scale, noise=noise, style_idx=style_idx)

#----------------------------------------------------------------------------
# One-step sampler used by monitoring / metrics / generate_images.
# `n_steps` is accepted but ignored (drifting models are 1-NFE).

@torch.no_grad()
def sample(model, labels=None, n_samples=None, guidance=1.0, noise=None, n_steps=None, generator=None, **kwargs):
    device = model.device
    if noise is not None:
        n = noise.shape[0]
    elif labels is not None:
        n = labels.shape[0]
    else:
        n = n_samples
    assert n is not None, 'sample() needs labels, noise, or n_samples'

    if labels is not None:
        class_idx = labels.argmax(dim=1)
    else:
        class_idx = torch.zeros(n, dtype=torch.long, device=device)

    return model.generate(class_idx, cfg_scale=guidance, noise=noise, generator=generator)

#----------------------------------------------------------------------------
