"""Distributed metric (FID / FD-DINOv2 / MIND) computation during training.

Reuses the building blocks in `calculate_metrics.py` (feature detector,
distributed accumulation of mu/sigma OR raw features, scoring against a
reference) but plugs in an *online* image iterator that samples from the
current EMA model on each rank, instead of reading images from disk.

MIND-style metrics ('mind', 'mind_dinov2') optionally use a different
sample count (``mind_num_samples``) than Fréchet metrics ('fid',
'fd_dinov2'). In that case we generate ``max(num_samples,
mind_num_samples)`` images once per eval, and each metric consumes the
prefix of features it needs -- so the extra sampling cost is zero when
``mind_num_samples <= num_samples``.
"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import distributed as dist
import calculate_metrics

#----------------------------------------------------------------------------
# Build the rank-local iterable of generated uint8 image batches.
#
# We mirror the partitioning logic of `calculate_metrics.calculate_stats_for_files`
# so that `calculate_stats_for_iterable` sees exactly `num_batches // world_size`
# batches per rank and finalizes the all_reduce on the last batch correctly.

def _build_image_iter(
    model,
    encoder,
    sampler_kwargs,
    num_samples,
    max_batch_size,
    seed,
    device,
):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if num_samples < 2 * world_size:
        raise ValueError(f'metric_num_samples ({num_samples}) must be at least 2 * world_size ({world_size}).')

    # Match calculate_stats_for_files: num_batches is a multiple of world_size,
    # so every rank gets the same number of batches.
    num_batches = max((num_samples - 1) // (max_batch_size * world_size) + 1, 1) * world_size
    # Ensure num_batches <= num_samples so no split (and therefore no rank-batch) is empty,
    # which would otherwise crash the feature detector on an empty input.
    num_batches = min(num_batches, (num_samples // world_size) * world_size)
    splits = np.array_split(np.arange(num_samples), num_batches)
    rank_batches = splits[rank::world_size]

    img_channels = model.img_channels
    img_resolution = model.img_resolution
    label_dim = int(getattr(model, 'label_dim', 0) or 0)

    class GeneratedImageIter:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            for batch_idx, indices in enumerate(rank_batches):
                bsz = len(indices)
                if bsz == 0:
                    # Yield a 0-sized uint8 batch; the stats accumulator handles it.
                    yield torch.zeros((0, 3, img_resolution, img_resolution),
                                      dtype=torch.uint8, device=device)
                    continue

                # Per-batch generator => deterministic given (seed, first index).
                gen = torch.Generator(device=device).manual_seed(
                    int(seed) * (10 ** 9) + int(indices[0])
                )
                noise = torch.randn(
                    (bsz, img_channels, img_resolution, img_resolution),
                    device=device, generator=gen,
                )
                labels = None
                if label_dim > 0:
                    idx = torch.randint(0, label_dim, (bsz,), device=device, generator=gen)
                    labels = F.one_hot(idx, num_classes=label_dim).float()

                latents = dnnlib.util.call_func_by_name(
                    **sampler_kwargs,
                    model=model,
                    noise=noise,
                    labels=labels,
                    n_samples=bsz,
                )
                images = encoder.decode(latents)  # NCHW uint8, 3 channels
                yield images

                # Help peak-memory: free inter-batch tensors.
                del noise, labels, latents, images

    return GeneratedImageIter()

#----------------------------------------------------------------------------
# Compute the requested metrics (e.g. 'fid') against a reference .pkl.
# Returns a dict {metric: float} on rank 0, None on other ranks.

@torch.inference_mode()
def compute_metrics(
    model,                                  # EMA model (eval mode, on device).
    encoder,                                # Encoder used to decode latents into raw pixels.
    sampler_kwargs,                         # Sampler config (same as monitoring grid).
    ref_path,                               # Path to a reference statistics .pkl/.npz.
    num_samples         = 10_000,           # # images for FID / FD-DINOv2.
    mind_num_samples    = 5_000,            # # images for MIND / MIND-DINOv2 (paper recommends 5k).
    metrics             = ('fid',),         # Which metrics to compute.
    max_batch_size      = 64,               # Per-rank batch size for sampling and feature extraction.
    seed                = 0,                # Seed for noise/labels.
    mind_n_projections  = None,             # None = use calculate_metrics.MIND_DEFAULT_N_PROJECTIONS.
    mind_seed           = None,             # None = use calculate_metrics.MIND_DEFAULT_SEED.
    device              = torch.device('cuda'),
):
    metrics = list(metrics)
    if not metrics:
        return {} if dist.get_rank() == 0 else None

    # Per-metric sample budgets. We always generate max(budgets) images and
    # let each metric truncate its own feature stream so that, with default
    # settings (num_samples=10k, mind_num_samples=5k), MIND adds zero extra
    # sampling cost on top of FID.
    metric_num_samples = {}
    for m in metrics:
        spec = calculate_metrics.metric_specs.get(m)
        if spec is not None and spec.stat_type == 'features':
            metric_num_samples[m] = mind_num_samples
        else:
            metric_num_samples[m] = num_samples
    total_num_samples = max(metric_num_samples.values())

    was_training = model.training
    model.eval()

    image_iter = _build_image_iter(
        model=model,
        encoder=encoder,
        sampler_kwargs=sampler_kwargs,
        num_samples=total_num_samples,
        max_batch_size=max_batch_size,
        seed=seed,
        device=device,
    )

    stats_iter = calculate_metrics.calculate_stats_for_iterable(
        image_iter=image_iter,
        metrics=metrics,
        verbose=True,
        device=device,
        metric_num_samples=metric_num_samples,
    )

    final_r = None
    for r in tqdm(stats_iter):
        final_r = r

    results = None
    if dist.get_rank() == 0:
        ref = calculate_metrics.load_stats(ref_path, verbose=True)
        kw = {}
        if mind_n_projections is not None:
            kw['mind_n_projections'] = mind_n_projections
        if mind_seed is not None:
            kw['mind_seed'] = mind_seed
        results = calculate_metrics.calculate_metrics_from_stats(
            stats=final_r.stats,
            ref=ref,
            metrics=metrics,
            verbose=True,
            **kw,
        )

    if dist.get_world_size() > 1:
        torch.distributed.barrier()

    if was_training:
        model.train()
    return results

#----------------------------------------------------------------------------
