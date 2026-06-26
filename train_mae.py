"""Build a config and launch ResNet-MAE pretraining.

The MAE is the feature encoder consumed by the drift loss. Train it first, then
point `train.py --mae-pkl` at one of its `model-snapshot-*.pkl` files.
"""

import os
import time
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop_mae
import datetime

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------

def _wait_for_path(path, timeout=300, interval=0.1):
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            raise TimeoutError(f'Timed out after {timeout}s waiting for {path}')
        time.sleep(interval)

#----------------------------------------------------------------------------

dataset_presets = {
    'cifar10': dnnlib.EasyDict(
        lr_scheduler_kwargs=dnnlib.EasyDict(func_name='training.schedulers.warmup_const_lr'),
        net_kwargs=dnnlib.EasyDict(
            base_channels=128,
            patch_size=2,           # zero out 2x2 patches
            input_patch_size=1,     # input is already 32x32
            layers=[3, 4, 6, 3],
            dropout_prob=0.0,
        ),
    ),
}

config_presets = {
    'mae-cifar10': dnnlib.EasyDict(
        dataset='cifar10',
        total_steps=100_000,
        batch_size=512,
        lr=2e-3,
        warmup_steps=1_000,
        weight_decay=0.01,
        adam_betas=(0.9, 0.95),
        max_clip_norm=2.0,
        mask_ratio=0.5,
        ema_decay=0.9995,     # Single exponential EMA decay (constant per-step beta), matching the JAX reference.
        finetune_last_steps=3_000,
        warmup_finetune_steps=1_000,
        finetune_cls=0.1,
    ),
    'mae-cifar10-debug': dnnlib.EasyDict(
        dataset='cifar10',
        total_steps=50,
        batch_size=64,
        lr=2e-3,
        warmup_steps=10,
        weight_decay=0.01,
        adam_betas=(0.9, 0.95),
        max_clip_norm=2.0,
        mask_ratio=0.5,
        ema_decay=0.999,
        finetune_last_steps=10,
        warmup_finetune_steps=5,
        finetune_cls=0.1,
    ),
}

#----------------------------------------------------------------------------

def setup_training_config(preset='mae-cifar10', **opts):
    opts = dnnlib.EasyDict(opts)
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    config_preset = config_presets[preset]
    dataset_name = config_preset['dataset']
    dataset_preset = dataset_presets[dataset_name]
    overlap = set(config_preset).intersection(dataset_preset)
    assert not overlap, f'shared keys: {sorted(overlap)}'

    merged = {**dataset_preset, **config_preset}
    for key, value in merged.items():
        if opts.get(key, None) is None:
            opts[key] = value

    batch_size = opts.batch_size
    total_nimg = opts.total_steps * batch_size
    warmup_nimg = opts.warmup_steps * batch_size

    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=True, xflip=True)
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    c.data_loader_kwargs = dict(class_name='torch.utils.data.DataLoader',
                                pin_memory=opts.pin_memory, num_workers=opts.num_workers,
                                prefetch_factor=opts.prefetch_factor)
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    else:
        raise click.ClickException(f'--data: expected 3-channel pixel data, got {dataset_channels}')

    c.model_kwargs = dnnlib.EasyDict(class_name='training.networks_mae.MAEResNet', **opts.net_kwargs)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=opts.lr,
                                         betas=tuple(opts.adam_betas), eps=1e-8, weight_decay=opts.weight_decay)
    c.lr_kwargs = dnnlib.EasyDict(**opts.lr_scheduler_kwargs, base_lr=opts.lr,
                                  total_nimg=total_nimg, warmup_nimg=warmup_nimg)
    c.ema_kwargs = dict(class_name='training.phema.FixedDecayEMA', decay=opts.ema_decay)
    c.max_clip_norm = opts.max_clip_norm

    c.batch_size = batch_size
    c.max_batch_gpu = opts.max_batch_gpu or None
    c.total_nimg = total_nimg
    c.mask_ratio_min = opts.mask_ratio
    c.mask_ratio_max = opts.mask_ratio
    c.finetune_last_nimg = opts.finetune_last_steps * batch_size
    c.warmup_finetune_nimg = opts.warmup_finetune_steps * batch_size
    c.finetune_cls = opts.finetune_cls

    c.loss_scaling = opts.ls
    c.cudnn_benchmark = opts.bench
    c.force_finite = opts.force_finite
    c.status_nimg = opts.status or None
    c.snapshot_nimg = opts.snapshot or None
    c.checkpoint_nimg = opts.checkpoint or None
    c.seed = opts.seed
    return c

#----------------------------------------------------------------------------

def parse_nimg(s):
    if isinstance(s, int):
        return s
    for suf, sh in (('Ki', 10), ('Mi', 20), ('Gi', 30)):
        if s.endswith(suf):
            return int(s[:-2]) << sh
    return int(s)

#----------------------------------------------------------------------------

@click.command()
@click.option('--outdir',           help='Output directory', metavar='DIR', type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR', type=str, required=True)
@click.option('--preset',           help='Configuration preset', metavar='STR', type=str, default='mae-cifar10', show_default=True)

@click.option('--total-steps',      help='Number of training steps', metavar='INT', type=int, default=None)
@click.option('--batch-size',       help='Total batch size', metavar='INT', type=int, default=None)
@click.option('--lr',               help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--mask-ratio',       help='Masking ratio', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=None)
@click.option('--ema-decay',        help='MAE EMA decay (constant per-step beta)', metavar='FLOAT', type=click.FloatRange(min=0, max=1, min_open=True), default=None)

@click.option('--max-batch-gpu',    help='Limit per-GPU batch', metavar='INT', type=int, default=None)
@click.option('--pin-memory',       help='Pinned dataloader memory', metavar='BOOL', default=True, show_default=True)
@click.option('--num-workers',      help='Dataloader workers', metavar='INT', type=int, default=2, show_default=True)
@click.option('--prefetch_factor',  help='Dataloader prefetch', metavar='INT', type=int, default=2, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='cuDNN benchmarking', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--force-finite',     help='Zero NaN/Inf gradients', metavar='BOOL', type=bool, default=True, show_default=True)

@click.option('--status',           help='Status print interval', metavar='NIMG', type=parse_nimg, default='1Mi', show_default=True)
@click.option('--snapshot',         help='Snapshot interval', metavar='NIMG', type=parse_nimg, default='16Mi', show_default=True)
@click.option('--checkpoint',       help='Checkpoint interval', metavar='NIMG', type=parse_nimg, default='64Mi', show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT', type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print config and exit', is_flag=True)
def cmdline(outdir, dry_run, **opts):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up MAE training config...')
    c = setup_training_config(**opts)

    if os.path.isdir(outdir) and any(f.startswith('training-state-') for f in os.listdir(outdir)):
        run_dir = outdir
        dist.print0(f'Resuming from {run_dir}')
    else:
        preset_name = opts.get('preset', 'run')
        os.makedirs(outdir, exist_ok=True)
        run_id = os.environ.get('TORCHELASTIC_RUN_ID', os.environ.get('MASTER_PORT', 'default'))
        marker_path = os.path.join(outdir, f'.run_dir.{run_id}')
        if dist.get_rank() == 0:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            run_dir = os.path.join(outdir, f'{now}_{preset_name}')
            with open(marker_path, 'wt') as f:
                f.write(run_dir)
        else:
            _wait_for_path(marker_path)
            with open(marker_path, 'rt') as f:
                run_dir = f.read().strip()

    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0(f'Output directory: {run_dir}')
    dist.print0(f'Number of GPUs:   {dist.get_world_size()}')
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        options_path = os.path.join(run_dir, 'training_options.json')
        if dist.get_rank() == 0:
            os.makedirs(run_dir, exist_ok=True)
            with open(options_path, 'wt') as f:
                json.dump(c, f, indent=2)
        else:
            _wait_for_path(options_path)
        dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
        training.training_loop_mae.training_loop_mae(run_dir=run_dir, **c)
    torch.distributed.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
