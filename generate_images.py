# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')

#----------------------------------------------------------------------------
# Configuration presets.

# Drifting models are one-step (1-NFE) generators, so `n_sampling_steps` is
# ignored; only `guidance` (the training-time-style CFG scale fed to the
# generator) matters. Point `--model` at a `model-snapshot-*.pkl` produced by
# `train.py`, or define a preset here.
config_presets = {
    'wflow-cifar-10': dnnlib.EasyDict(
        model='training-runs/cifar10/wflow/model-snapshot-latest.pkl',
        sampler_fn="training.model.sample",
        n_sampling_steps=1,
        guidance=1.0,
    ),
    'wflow-cifar-10-guid': dnnlib.EasyDict(
        model='training-runs/cifar10/wflow/model-snapshot-latest.pkl',
        sampler_fn="training.model.sample",
        n_sampling_steps=1,
        guidance=2.0,
    ),
}

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

@torch.inference_mode()
def generate_images(
    # OS and seeds.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.

    # Sampling parameters.
    model               = None,                 # Main model. Path, URL, or torch.nn.Module.
    sampler_fn          = None,                 # Sampler function for the model.
    n_sampling_steps    = 30,                   # Number of steps during sampling
    guidance            = 1.0,                  # CFG coef
    class_idx           = None,                 # Class label. None = select randomly.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from model pickle.

    # Performance options and verbose. 
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    device              = torch.device('cuda'), # Which compute device to use.
    verbose             = True,                 # Enable status prints?
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load model.
    if isinstance(model, str):
        if verbose:
            dist.print0(f'Loading main model from {model} ...')
        with dnnlib.util.open_url(model, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        model = data['ema'].to(device)

        model.eval()
        model.requires_grad_(False)

        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert model is not None

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), model.img_channels, model.img_resolution, model.img_resolution], device=device)
                    r.labels = None
                    if model.label_dim > 0:
                        r.labels = torch.eye(model.label_dim, device=device)[rnd.randint(model.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(
                        func_name=sampler_fn,
                        model=model,
                        noise=r.noise,
                        labels=r.labels,
                        n_steps=n_sampling_steps,
                        n_samples=len(r.seeds),
                        guidance=guidance,
                    )
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)

@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--model',                    help='Main model pickle filename', metavar='PATH|URL',                  type=str, default=None)
@click.option('--sampler-fn',               help='Sampler function for the model', metavar='FUNC',                  type=str, default=None)
@click.option('--n-sampling-steps',         help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=None)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)

@click.option('--max-batch-size',           help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--encoder-batch-size',       help='Maximum batch size for the encoder', metavar='INT',               type=click.IntRange(min=1), default=None, show_default=True)


def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.model is None:
        raise click.ClickException('Please specify either --preset or --model')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        del _r

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
