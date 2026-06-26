"""W&B monitoring utilities.

Logged values are split into the following categories:
  - main_metrics / main_eval_metrics / metrics : scalars, plotted against
                             three x-axes (training_steps, nimgs, time).
  - main_plots   / plots   : media (images, etc.), plotted against
                             training_steps only.

Anything tagged "main_*" is intended to land on the run's main W&B page;
the rest live in their own panels. `main_eval_metrics` is reserved for
the (expensive) evaluation metrics (FID / FD-DINOv2 / MIND) so they get
their own section separate from the per-tick training metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import misc

#----------------------------------------------------------------------------

_METRIC_CATEGORIES = ('main_metrics', 'main_eval_metrics', 'metrics')
_PLOT_CATEGORIES = ('main_plots', 'plots')
_X_AXES = ('training_steps', 'nimgs', 'time')

#----------------------------------------------------------------------------
# W&B metric setup. Call once after wandb.init().

def setup_wandb_metrics(wandb):
    for axis in _X_AXES:
        wandb.define_metric(f'trainer/{axis}')
    for cat in _METRIC_CATEGORIES:
        for axis in _X_AXES:
            wandb.define_metric(f'{cat}/by_{axis}/*', step_metric=f'trainer/{axis}')
    for cat in _PLOT_CATEGORIES:
        wandb.define_metric(f'{cat}/*', step_metric='trainer/training_steps')

#----------------------------------------------------------------------------
# Push a categorized batch of values to W&B.

def log_to_wandb(
    wandb,
    cur_step,
    cur_nimg,
    elapsed_time,
    main_metrics=None,
    main_eval_metrics=None,
    metrics=None,
    main_plots=None,
    plots=None,
):
    log_dict = {
        'trainer/training_steps': cur_step,
        'trainer/nimgs': cur_nimg,
        # `elapsed_time` is in seconds; expose the W&B time x-axis in hours.
        'trainer/time': elapsed_time / 3600.0,
    }
    for k, v in (main_metrics or {}).items():
        for axis in _X_AXES:
            log_dict[f'main_metrics/by_{axis}/{k}'] = v
    for k, v in (main_eval_metrics or {}).items():
        for axis in _X_AXES:
            log_dict[f'main_eval_metrics/by_{axis}/{k}'] = v
    for k, v in (metrics or {}).items():
        for axis in _X_AXES:
            log_dict[f'metrics/by_{axis}/{k}'] = v
    for k, v in (main_plots or {}).items():
        log_dict[f'main_plots/{k}'] = v
    for k, v in (plots or {}).items():
        log_dict[f'plots/{k}'] = v
    wandb.log(log_dict)

#----------------------------------------------------------------------------
# Generate a square grid of samples using the sampler from the dataset preset.
# For class-conditional models the first `label_dim` samples cycle through the
# classes, the rest are random (seeded by `seed`).

def generate_sample_grid(
    model, encoder, sampler_kwargs,
    n_samples=16, label_dim=0, seed=0, device=torch.device('cuda'),
):
    h = w = int(np.sqrt(n_samples))
    assert h * w == n_samples, 'n_samples must be a perfect square'

    gen = torch.Generator(device=device).manual_seed(int(seed))
    if label_dim > 0:
        n_fixed = min(label_dim, n_samples)
        fixed = torch.arange(n_fixed, device=device)
        rand = torch.randint(0, label_dim, (n_samples - n_fixed,), device=device, generator=gen)
        idx = torch.cat([fixed, rand])
        labels = F.one_hot(idx, num_classes=label_dim).float()
    else:
        labels = None

    noise = torch.randn(
        (n_samples, model.img_channels, model.img_resolution, model.img_resolution),
        device=device, generator=gen,
    )

    latents = dnnlib.util.call_func_by_name(
        **sampler_kwargs,
        model=model,
        labels=labels,
        n_samples=n_samples,
        noise=noise,
    )
    images = encoder.decode(latents)
    grid = misc.tile_images(images, w=w, h=h)
    return grid

#----------------------------------------------------------------------------
