# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Calculate evaluation metrics (FID and FD_DINOv2)."""

import os
import json
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset

#----------------------------------------------------------------------------
# Abstract base class for feature detectors.

class Detector:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def __call__(self, x): # NCHW, uint8, 3 channels => NC, float32
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# InceptionV3 feature detector.
# This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

class InceptionV3Detector(Detector):
    def __init__(self):
        super().__init__(feature_dim=2048)
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        return self.model.to(x.device)(x, return_features=True)

#----------------------------------------------------------------------------
# DINOv2 feature detector.
# Modeled after https://github.com/layer6ai-labs/dgm-eval

class DINOv2Detector(Detector):
    def __init__(self, resize_mode='torch'):
        super().__init__(feature_dim=1024)
        self.resize_mode = resize_mode
        import warnings
        warnings.filterwarnings('ignore', 'xFormers is not available')
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False, skip_validation=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        # Resize images.
        if self.resize_mode == 'pil': # Slow reference implementation that matches the original dgm-eval codebase exactly.
            device = x.device
            x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            x = np.stack([np.uint8(PIL.Image.fromarray(xx, 'RGB').resize((224, 224), PIL.Image.Resampling.BICUBIC)) for xx in x])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        elif self.resize_mode == 'torch': # Fast practical implementation that yields almost the same results.
            x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
        else:
            raise ValueError(f'Invalid resize mode "{self.resize_mode}"')

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        # Run DINOv2 model.
        return self.model.to(x.device)(x)

#----------------------------------------------------------------------------
# Metric specifications.
#
# `stat_type` controls how features are aggregated:
#   * 'moments' -- accumulate per-batch (mu, Σ) sums and reduce them across
#     ranks. Used by Fréchet-style metrics (FID, FD-DINOv2).
#   * 'features' -- collect raw per-sample feature vectors on each rank, then
#     concat + all-gather them at the end. Used by MIND (sliced Wasserstein
#     needs the actual sample-level embeddings, not just first two moments).

metric_specs = {
    'fid':          dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector), stat_type='moments'),
    'fd_dinov2':    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector),     stat_type='moments'),
    'mind':         dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector), stat_type='features'),
    'mind_dinov2':  dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector),     stat_type='features'),
}

# MIND default knobs (Monge Inception Distance, Berthet et al. 2026,
# arXiv:2605.06797). Paper recommends M in [100, 1000] random projections;
# we default to 1024. The α = 3d scaling is applied inside `compute_mind`.
MIND_DEFAULT_N_PROJECTIONS = 1024
MIND_DEFAULT_SEED = 0

#----------------------------------------------------------------------------
# Get feature detector for the given metric. Detectors are cached by detector
# *class* so two metrics that share a backbone (e.g. `fid` + `mind`, both
# Inception-v3) only instantiate the network once.

_detector_cache = dict()

def get_detector(metric, verbose=True):
    kwargs = metric_specs[metric].detector_kwargs
    cache_key = kwargs.class_name if not isinstance(kwargs.class_name, str) else kwargs.class_name
    if cache_key in _detector_cache:
        return _detector_cache[cache_key]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if verbose:
        name = kwargs.class_name.split('.')[-1] if isinstance(kwargs.class_name, str) else kwargs.class_name.__name__
        dist.print0(f'Setting up {name}...')
    detector = dnnlib.util.construct_class_by_name(**kwargs)
    _detector_cache[cache_key] = detector

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    return detector

#----------------------------------------------------------------------------
# MIND (Monge Inception Distance) -- arXiv:2605.06797.
#
# Sliced 2-Wasserstein in feature space: project both samples along M random
# unit directions, sort each 1-D projection, and average the squared paired
# differences. The metric is scaled by α = 3·d so its order of magnitude
# matches that of FID (paper, Section 4.2). x, y must have the same number
# of rows. We accept arbitrary float dtypes and run the computation in the
# native dtype of the inputs.

def compute_mind(
    x_features,             # (n, d) torch.Tensor or np.ndarray of generated features.
    y_features,             # (n, d) torch.Tensor or np.ndarray of reference features.
    n_projections   = MIND_DEFAULT_N_PROJECTIONS,
    seed            = MIND_DEFAULT_SEED,
    device          = None,
):
    x = torch.as_tensor(x_features)
    y = torch.as_tensor(y_features)
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    elif x.device != y.device:
        y = y.to(x.device)
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    assert x.ndim == 2 and y.ndim == 2, f'compute_mind expects 2-D features, got {x.shape}, {y.shape}'
    assert x.shape == y.shape, (
        f'compute_mind expects x and y to have the same shape; got x={tuple(x.shape)}, y={tuple(y.shape)}'
    )
    n, d = x.shape
    alpha = 3.0 * d

    # Same projection draws every eval iff `seed` is fixed -> bit-stable
    # values across training runs at the same checkpoint count.
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    u = torch.randn((n_projections, d), generator=gen, dtype=x.dtype, device=x.device)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)

    # (M, n) projections per side. topk(.., k=n) returns the sorted view in
    # descending order -- since ((sort_desc(x) - sort_desc(y))**2) ==
    # ((sort_asc(x) - sort_asc(y))**2), this matches Eq. (3.1) of the paper.
    x_proj = u @ x.T
    y_proj = u @ y.T
    x_sorted = torch.topk(x_proj, n, dim=-1).values
    y_sorted = torch.topk(y_proj, n, dim=-1).values

    per_proj = ((x_sorted - y_sorted) ** 2).mean(dim=1)  # (M,)
    return float(alpha * per_proj.mean().item())

#----------------------------------------------------------------------------
# Load feature statistics from the given .pkl or .npz file.

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

#----------------------------------------------------------------------------
# Save feature statistics to the given .pkl file. If `merge_existing=True`
# and the file already exists, only the keys present in `stats` are
# overwritten -- this lets the `ref` subcommand add MIND features to an
# existing FID/FD-DINOv2 ref pkl without recomputing the FID entries.

def save_stats(stats, path, verbose=True, merge_existing=False):
    if verbose:
        print(f'Saving feature statistics to {path} ...')
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    out = dict(stats)
    if merge_existing and os.path.isfile(path):
        try:
            with open(path, 'rb') as f:
                existing = pickle.load(f)
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(out)
                out = merged
        except Exception as e:
            if verbose:
                print(f'Warning: could not merge into existing {path}: {e}; overwriting.')
    with open(path, 'wb') as f:
        pickle.dump(out, f)

#----------------------------------------------------------------------------
# Calculate feature statistics for the given image batches
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_iterable(
    image_iter,                         # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics     = ['fid', 'fd_dinov2'], # Metrics to compute the statistics for.
    verbose     = True,                 # Enable status prints?
    dest_path   = None,                 # Where to save the statistics. None = do not save.
    device      = torch.device('cuda'), # Which compute device to use.
    metric_num_samples = None,          # Optional dict {metric: int} capping how many samples each
                                        # metric collects (others may run for longer). None = no cap.
):
    # Initialize.
    num_batches = len(image_iter)
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Group metrics by detector class so we only forward each backbone once
    # per batch (e.g. `fid` and `mind` both use Inception-v3).
    detectors = {}
    metric_to_detector_key = {}
    for m in metrics:
        kwargs = metric_specs[m].detector_kwargs
        key = kwargs.class_name
        if key not in detectors:
            detectors[key] = get_detector(m, verbose=verbose)
        metric_to_detector_key[m] = key

    metric_num_samples = metric_num_samples or {}

    # Convenience wrapper for torch.distributed.all_reduce().
    def all_reduce(x):
        x = x.clone()
        torch.distributed.all_reduce(x)
        return x

    # All-gather a per-rank 2-D feature tensor (shape may differ per rank)
    # into a single concatenated tensor on every rank.
    def all_gather_features(local_feats):
        world_size = dist.get_world_size()
        if world_size == 1:
            return local_feats
        local_n = torch.tensor([local_feats.shape[0]], dtype=torch.long, device=local_feats.device)
        sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
        torch.distributed.all_gather(sizes, local_n)
        sizes_list = [int(s.item()) for s in sizes]
        max_n = max(sizes_list)
        # Pad to max for the all_gather, then trim per-rank afterwards.
        if local_feats.shape[0] < max_n:
            pad = torch.zeros((max_n - local_feats.shape[0], local_feats.shape[1]),
                              dtype=local_feats.dtype, device=local_feats.device)
            local_padded = torch.cat([local_feats, pad], dim=0)
        else:
            local_padded = local_feats
        gathered = [torch.empty_like(local_padded) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, local_padded)
        return torch.cat([g[:n] for g, n in zip(gathered, sizes_list)], dim=0)

    # Return an iterable over the batches.
    class StatsIterable:
        def __len__(self):
            return num_batches

        def __iter__(self):
            state = []
            for m in metrics:
                spec = metric_specs[m]
                detector = detectors[metric_to_detector_key[m]]
                s = dnnlib.EasyDict(
                    metric=m,
                    detector_key=metric_to_detector_key[m],
                    detector=detector,
                    stat_type=spec.stat_type,
                    target_count=metric_num_samples.get(m, None),
                    cum_count=0,
                )
                if s.stat_type == 'moments':
                    s.cum_mu = torch.zeros([detector.feature_dim], dtype=torch.float64, device=device)
                    s.cum_sigma = torch.zeros([detector.feature_dim, detector.feature_dim], dtype=torch.float64, device=device)
                elif s.stat_type == 'features':
                    s.feature_chunks = []  # list of (k_i, d) float32 tensors on `device`
                else:
                    raise ValueError(f'Unknown stat_type {s.stat_type!r} for metric {m!r}')
                state.append(s)
            cum_images = torch.zeros([], dtype=torch.int64, device=device)

            with torch.inference_mode():
                # Loop over batches.
                for batch_idx, images in enumerate(image_iter):
                    if isinstance(images, dict) and 'images' in images: # dict(images)
                        images = images['images']
                    elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                        images = images[0]
                    images = torch.as_tensor(images).to(device)

                    if images is not None and images.shape[0] > 0:
                        # Run each backbone once per batch and share its
                        # features across all metrics that consume it.
                        features_by_key = {key: det(images) for key, det in detectors.items()}
                        for s in state:
                            features = features_by_key[s.detector_key]
                            if s.stat_type == 'moments':
                                if s.target_count is not None and s.cum_count >= s.target_count:
                                    continue
                                take = features.shape[0]
                                if s.target_count is not None:
                                    take = min(take, s.target_count - s.cum_count)
                                f = features[:take].to(torch.float64)
                                s.cum_mu += f.sum(0)
                                s.cum_sigma += f.T @ f
                                s.cum_count += take
                            else:  # 'features'
                                if s.target_count is not None and s.cum_count >= s.target_count:
                                    continue
                                take = features.shape[0]
                                if s.target_count is not None:
                                    take = min(take, s.target_count - s.cum_count)
                                # Keep feature collection in fp32 -- the
                                # eventual MIND computation runs in fp32
                                # and we'd lose precision if we packed to
                                # fp16 here. Storage is small (k * d * 4 B
                                # per rank, capped by mind_num_samples).
                                s.feature_chunks.append(features[:take].detach().to(torch.float32))
                                s.cum_count += take
                        cum_images += images.shape[0]

                    # Output results.
                    r = dnnlib.EasyDict(stats=None, images=images, batch_idx=batch_idx, num_batches=num_batches)
                    r.num_images = int(all_reduce(cum_images).cpu())
                    if batch_idx == num_batches - 1:
                        assert r.num_images >= 2
                        r.stats = dict(num_images=r.num_images)
                        for s in state:
                            if s.stat_type == 'moments':
                                # Per-metric effective sample count = sum
                                # of per-rank cum_counts (which equal
                                # min(target, what each rank saw)).
                                n_eff = int(all_reduce(torch.tensor(s.cum_count, dtype=torch.int64, device=device)).cpu())
                                assert n_eff >= 2, f'metric {s.metric}: only {n_eff} samples accumulated'
                                mu = all_reduce(s.cum_mu) / n_eff
                                sigma = (all_reduce(s.cum_sigma) - mu.ger(mu) * n_eff) / (n_eff - 1)
                                r.stats[s.metric] = dict(
                                    mu=mu.cpu().numpy(),
                                    sigma=sigma.cpu().numpy(),
                                    num_images=n_eff,
                                )
                            else:  # 'features'
                                local_feats = (torch.cat(s.feature_chunks, dim=0)
                                               if s.feature_chunks
                                               else torch.zeros((0, s.detector.feature_dim),
                                                                dtype=torch.float32, device=device))
                                gathered = all_gather_features(local_feats)
                                r.stats[s.metric] = dict(
                                    features=gathered.cpu().numpy(),
                                    num_images=int(gathered.shape[0]),
                                )
                        if dest_path is not None and dist.get_rank() == 0:
                            save_stats(stats=r.stats, path=dest_path, verbose=False)
                    yield r

    return StatsIterable()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_files(
    image_path,             # Path to a directory or ZIP file containing the images.
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0,    # Random seed for selecting the images.
    max_batch_size  = 64,   # Maximum batch size.
    num_workers     = 2,    # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    **stats_kwargs,         # Arguments for calculate_stats_for_iterable().
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    # Return an interable for calculating the statistics.
    return calculate_stats_for_iterable(image_iter=data_loader, verbose=verbose, **stats_kwargs)

#----------------------------------------------------------------------------
# Calculate metrics based on the given feature statistics.

def calculate_metrics_from_stats(
    stats,                          # Feature statistics of the generated images.
    ref,                            # Reference statistics of the dataset. Can be a path or URL.
    metrics = ['fid', 'fd_dinov2'], # List of metrics to compute.
    verbose = True,                 # Enable status prints?
    mind_n_projections = MIND_DEFAULT_N_PROJECTIONS,
    mind_seed          = MIND_DEFAULT_SEED,
    mind_device        = None,      # Where to run the MIND projection+sort. Defaults to CPU.
):
    if isinstance(ref, str):
        ref = load_stats(ref, verbose=verbose)
    results = dict()
    for metric in metrics:
        if metric not in stats or metric not in ref:
            if verbose:
                print(f'No statistics computed for {metric} -- skipping.')
            continue
        if verbose:
            print(f'Calculating {metric}...')
        spec = metric_specs.get(metric)
        stat_type = spec.stat_type if spec is not None else 'moments'

        if stat_type == 'moments':
            m = np.square(stats[metric]['mu'] - ref[metric]['mu']).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(stats[metric]['sigma'], ref[metric]['sigma']), disp=False)
            value = float(np.real(m + np.trace(stats[metric]['sigma'] + ref[metric]['sigma'] - s * 2)))
        elif stat_type == 'features':
            x_feats = stats[metric]['features']
            y_feats = ref[metric]['features']
            # MIND requires equal sample counts -- truncate the longer one.
            n = min(x_feats.shape[0], y_feats.shape[0])
            if x_feats.shape[0] != y_feats.shape[0] and verbose:
                print(f'  {metric}: truncating to n={n} (gen={x_feats.shape[0]}, ref={y_feats.shape[0]})')
            value = compute_mind(
                x_feats[:n], y_feats[:n],
                n_projections=mind_n_projections,
                seed=mind_seed,
                device=mind_device,
            )
        else:
            raise ValueError(f'Unknown stat_type {stat_type!r} for metric {metric!r}')

        results[metric] = value
        if verbose:
            print(f'{metric} = {value:g}')
    return results

#----------------------------------------------------------------------------
# Parse a comma separated list of strings.

def parse_metric_list(s):
    metrics = s if isinstance(s, list) else s.split(',')
    for metric in metrics:
        if metric not in metric_specs:
            raise click.ClickException(f'Invalid metric "{metric}"')
    return metrics

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """Calculate evaluation metrics (FID and FD_DINOv2).

    Examples:

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img512-xxl-guid-fid --outdir=out --subdirs --seeds=0-49999

    \b
    # Calculate metrics for a random subset of 50000 images in out/
    python calculate_metrics.py calc --images=out \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl

    \b
    # Compute dataset reference statistics
    python calculate_metrics.py ref --data=datasets/my-dataset.zip \\
        --dest=fid-refs/my-dataset.pkl
    """

#----------------------------------------------------------------------------
# 'calc' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num-images', 'num_images', help='Number of images to use', metavar='INT',                  type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--max-batch-size',           help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)

def calc(ref_path, metrics, **opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    stats_iter = calculate_stats_for_files(metrics=metrics, **opts)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        final_scores = calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)

        save_path = os.path.join(opts['image_path'], 'metrics_log.json')
        with open(save_path, 'w') as f:
            json.dump(final_scores, f, indent=4)

    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'ref' subcommand.

@cmdline.command()
@click.option('--data', 'image_path',       help='Path to the dataset', metavar='PATH|ZIP',             type=str, required=True)
@click.option('--dest', 'dest_path',        help='Destination file', metavar='PKL',                     type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',          type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num-images', 'num_images', help='Number of images to use (None = all)', metavar='INT', type=click.IntRange(min=2), default=None, show_default=True)
@click.option('--merge/--overwrite',        help='Merge into existing --dest pkl (preserving other metrics already in it) instead of overwriting.',
                                            default=True, show_default=True)
@click.option('--max-batch-size',           help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT', type=click.IntRange(min=0), default=2, show_default=True)

def ref(dest_path, merge, **opts):
    """Calculate dataset reference statistics for 'calc' and 'gen'.

    Typical workflow to extend an existing FID/FD-DINOv2 ref pkl with MIND
    reference features (5,000 samples by default):

    \b
    torchrun --standalone --nproc_per_node=2 calculate_metrics.py ref \\
        --data=datasets/cifar10.zip \\
        --dest=fid-refs/cifar10.pkl \\
        --metrics=mind,mind_dinov2 --num-images=5000

    With `--merge` (default), the existing `fid` and `fd_dinov2` entries are
    preserved; only the keys for the metrics passed via `--metrics` are
    written.
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    # We don't pass dest_path into the stats iterator so we can control the
    # merge semantics ourselves (the iterator's own save_stats call would
    # always overwrite the file).
    stats_iter = calculate_stats_for_files(dest_path=None, **opts)
    final_r = None
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        final_r = r
    if dist.get_rank() == 0 and final_r is not None and final_r.stats is not None:
        save_stats(stats=final_r.stats, path=dest_path, verbose=True, merge_existing=merge)
    if dist.get_world_size() > 1:
        torch.distributed.barrier()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
