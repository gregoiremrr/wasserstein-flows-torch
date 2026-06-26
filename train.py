"""Build a config and launch W-Flow (one-step generator) training.

Mirrors the two-preset layout of the base repo: `dataset_presets` holds
everything intrinsic to the data (generator architecture, monitoring sampler,
LR schedule family, EMA decay); `config_presets` describes a particular run
(sample-allocation, CFG, Sinkhorn knobs, optimization budget). The generator is
trained with the Sinkhorn (Wasserstein gradient flow) drift loss in the feature
space of a pretrained ResNet-MAE (`--mae-pkl`), which must be trained first via
`train_mae.py`.
"""

import os
import time
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop
from calculate_metrics import parse_metric_list
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
# Dataset presets: generator architecture + monitoring sampler + LR family.

dataset_presets = {
    'cifar10': dnnlib.EasyDict(
        resolution=32,
        ema_decay=0.999,      # Single exponential EMA decay (constant per-step beta).
        net_kwargs=dnnlib.EasyDict(
            patch_size=2,
            hidden_size=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            cond_dim=384,
            n_cls_tokens=16,
            noise_classes=64,
            noise_coords=32,
            use_qknorm=True,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            attn_fp32=True,
        ),
        sampler_kwargs=dnnlib.EasyDict(
            func_name='training.model.sample',
            guidance=1.0,
        ),
        lr_scheduler_kwargs=dnnlib.EasyDict(
            func_name='training.schedulers.warmup_const_lr',
        ),
    ),
}

#----------------------------------------------------------------------------
# Configuration presets.

_common_cifar = dict(
    dataset='cifar10',
    cond=True,
    cfg_min=1.0,
    cfg_max=4.0,
    neg_cfg_pw=3.0,        # p(alpha) ~ alpha^-3.
    no_cfg_frac=0.0,
    # Sinkhorn (debiased entropic-OT) drift knobs.
    R_list=[0.05],         # Entropic-regularisation eps (single value; W-Flow is robust to it).
    sinkhorn_num_iter=10,
    use_quadratic_cost=True,
    disable_diag_mask=True,  # Two-batch self-transport => no diagonal masking.
    lr=2e-4,
    warmup_steps=1_000,
    weight_decay=0.01,
    max_clip_norm=2.0,
    adam_betas=(0.9, 0.95),
)

config_presets = {
    'wflow-cifar10': dnnlib.EasyDict(
        **_common_cifar,
        total_steps=10_000,
        labels_per_step=8,        # Nc per rank
        gen_per_label=64,         # generated samples per label (main batch)
        self_gen_per_label=64,    # second (independent) generated batch for self-transport
        pos_per_sample=256,       # real positives per label
        neg_per_sample=64,        # unconditional reals per label (velocity-CFG)
        positive_bank_size=1024,
        negative_bank_size=1000,
        push_per_step=256,
        push_at_resume=8,
        loss_microbatch_labels=2,
    ),
    # Tiny preset for smoke tests / single-GPU debugging.
    'wflow-cifar10-debug': dnnlib.EasyDict(
        **_common_cifar,
        total_steps=100,
        labels_per_step=8,
        gen_per_label=8,
        self_gen_per_label=8,
        pos_per_sample=8,
        neg_per_sample=4,
        positive_bank_size=64,
        negative_bank_size=128,
        push_per_step=64,
        push_at_resume=2,
        loss_microbatch_labels=2,
    ),
}

#----------------------------------------------------------------------------

def setup_training_config(preset='wflow-cifar10', **opts):
    opts = dnnlib.EasyDict(opts)

    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    config_preset = config_presets[preset]
    dataset_name = config_preset['dataset']
    if dataset_name not in dataset_presets:
        raise click.ClickException(f'Invalid dataset preset "{dataset_name}"')
    dataset_preset = dataset_presets[dataset_name]

    overlap = set(config_preset).intersection(dataset_preset)
    assert not overlap, f'config_preset and dataset_preset share keys: {sorted(overlap)}'

    merged = {**dataset_preset, **config_preset}
    for key, value in merged.items():
        if opts.get(key, None) is None:
            opts[key] = value

    world_size = dist.get_world_size()
    batch_size = opts.labels_per_step * opts.gen_per_label * world_size
    total_nimg = opts.total_steps * batch_size
    warmup_nimg = opts.warmup_steps * batch_size

    c = dnnlib.EasyDict()

    # Dataset / dataloader.
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=True, xflip=True)
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if not dataset_obj.has_labels:
            raise click.ClickException('W-Flow models are class-conditional; dataset has no labels')
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    c.data_loader_kwargs = dict(class_name='torch.utils.data.DataLoader',
                                pin_memory=opts.pin_memory, num_workers=opts.num_workers,
                                prefetch_factor=opts.prefetch_factor)

    # Encoder (pixel space, no VAE).
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    else:
        raise click.ClickException(f'--data: expected 3-channel pixel data, got {dataset_channels}')

    # Model.
    c.model_kwargs = dnnlib.EasyDict(class_name='training.model.DriftingModel', use_fp16=opts.fp16, **opts.net_kwargs)

    # Loss (Sinkhorn debiased entropic-OT drift).
    c.loss_kwargs = dnnlib.EasyDict(
        class_name='training.loss.DriftLoss',
        gen_per_label=opts.gen_per_label,
        self_gen_per_label=opts.self_gen_per_label,
        cfg_min=opts.cfg_min, cfg_max=opts.cfg_max,
        neg_cfg_pw=opts.neg_cfg_pw, no_cfg_frac=opts.no_cfg_frac,
        R_list=list(opts.R_list),
        sinkhorn_num_iter=opts.sinkhorn_num_iter,
        use_quadratic_cost=bool(opts.use_quadratic_cost),
        disable_diag_mask=bool(opts.disable_diag_mask),
    )

    # Optimizer / LR / EMA.
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=opts.lr,
                                         betas=tuple(opts.adam_betas), eps=1e-8, weight_decay=opts.weight_decay)
    c.lr_kwargs = dnnlib.EasyDict(**opts.lr_scheduler_kwargs, base_lr=opts.lr,
                                  total_nimg=total_nimg, warmup_nimg=warmup_nimg)
    c.ema_kwargs = dict(class_name='training.phema.FixedDecayEMA', decay=opts.ema_decay)
    c.sampler_kwargs = dnnlib.EasyDict(**opts.sampler_kwargs)
    c.max_clip_norm = opts.max_clip_norm

    # Drift-specific.
    c.batch_size = batch_size
    c.labels_per_step = opts.labels_per_step
    c.loss_microbatch_labels = opts.loss_microbatch_labels
    c.pos_per_sample = opts.pos_per_sample
    c.neg_per_sample = opts.neg_per_sample
    c.positive_bank_size = opts.positive_bank_size
    c.negative_bank_size = opts.negative_bank_size
    c.push_per_step = opts.push_per_step
    c.push_at_resume = opts.push_at_resume
    c.total_nimg = total_nimg
    c.mae_pkl = opts.mae_pkl

    # Performance.
    c.loss_scaling = opts.ls
    c.cudnn_benchmark = opts.bench
    c.force_finite = opts.force_finite

    # I/O intervals are specified in optimizer steps; the training loop works in
    # image counts, so convert here (1 step == batch_size images).
    c.status_nimg = opts.status * batch_size if opts.status else None
    c.snapshot_nimg = opts.snapshot * batch_size if opts.snapshot else None
    c.checkpoint_nimg = opts.checkpoint * batch_size if opts.checkpoint else None

    # Metrics. (Interval is given in optimizer steps.)
    c.metrics_nimg = opts.metrics * batch_size if opts.metrics else None
    if c.metrics_nimg is not None:
        if not opts.metric_ref:
            raise click.ClickException('--metrics requires --metric-ref')
        if '://' not in opts.metric_ref and not os.path.isfile(opts.metric_ref):
            raise click.ClickException(f'--metric-ref: file not found: {opts.metric_ref}')
        c.metrics_kwargs = dnnlib.EasyDict(
            metrics=parse_metric_list(opts.metric_names), ref_path=opts.metric_ref,
            num_samples=opts.metric_num_samples, mind_num_samples=opts.mind_num_samples,
            max_batch_size=opts.metric_batch_size)
    else:
        c.metrics_kwargs = None

    c.seed = opts.seed
    return c

#----------------------------------------------------------------------------

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'MAE feature encoder:     {c.mae_pkl}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Effective batch size:    {c.batch_size}')
    dist.print0()

#----------------------------------------------------------------------------

def launch_training(run_dir, c):
    options_path = os.path.join(run_dir, 'training_options.json')
    if dist.get_rank() == 0:
        if not os.path.isdir(run_dir):
            dist.print0('Creating output directory...')
            os.makedirs(run_dir)
        with open(options_path, 'wt') as f:
            json.dump(c, f, indent=2)
    else:
        _wait_for_path(options_path)
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------

def parse_count(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------

@click.command()
@click.option('--outdir',           help='Output directory', metavar='DIR', type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR', type=str, required=True)
@click.option('--mae-pkl',          help='Pretrained ResNet-MAE snapshot pickle', metavar='PATH', type=str, default=None)
@click.option('--preset',           help='Configuration preset', metavar='STR', type=str, default='wflow-cifar10', show_default=True)

@click.option('--total-steps',      help='Number of training steps', metavar='INT', type=int, default=None)
@click.option('--labels-per-step',  help='Class labels (Nc) per rank per step', metavar='INT', type=int, default=None)
@click.option('--gen-per-label',    help='Generated samples per label (main batch)', metavar='INT', type=int, default=None)
@click.option('--self-gen-per-label', help='Second (independent) generated batch per label for self-transport', metavar='INT', type=int, default=None)
@click.option('--pos-per-sample',   help='Positive samples per label (Npos)', metavar='INT', type=int, default=None)
@click.option('--neg-per-sample',   help='Unconditional reals per label (velocity-CFG)', metavar='INT', type=int, default=None)
@click.option('--loss-microbatch-labels', help='Process Nc labels in chunks of this size (grad accumulation) to cap VRAM; 0/unset = no split', metavar='INT', type=int, default=None)
@click.option('--eps',              help='Sinkhorn entropic-regularisation value (overrides single-value R_list)', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--sinkhorn-num-iter', help='Sinkhorn-Knopp iterations per OT problem', metavar='INT', type=click.IntRange(min=1), default=None)
@click.option('--lr',               help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--ema-decay',        help='Generator EMA decay (constant per-step beta)', metavar='FLOAT', type=click.FloatRange(min=0, max=1, min_open=True), default=None)
@click.option('--max-clip-norm',    help='Max gradient norm', metavar='FLOAT', type=click.FloatRange(min=0), default=None)

@click.option('--pin-memory',       help='Pinned dataloader memory', metavar='BOOL', default=True, show_default=True)
@click.option('--num-workers',      help='Dataloader workers', metavar='INT', type=int, default=2, show_default=True)
@click.option('--prefetch_factor',  help='Dataloader prefetch', metavar='INT', type=int, default=2, show_default=True)
@click.option('--fp16/--no-fp16',   help='Mixed-precision generator', metavar='BOOL', default=False, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='cuDNN benchmarking', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--force-finite',     help='Zero NaN/Inf gradients', metavar='BOOL', type=bool, default=True, show_default=True)

@click.option('--status',           help='Interval of status prints (optimizer steps)', metavar='STEPS', type=parse_count, default='512', show_default=True)
@click.option('--snapshot',         help='Interval of model snapshots (optimizer steps)', metavar='STEPS', type=parse_count, default='32Ki', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints (optimizer steps)', metavar='STEPS', type=parse_count, default='512Ki', show_default=True)

@click.option('--metrics',          help='Interval of FID/FD-DINOv2/MIND evaluation (optimizer steps). Disabled by default.', metavar='STEPS', type=parse_count, default=None, show_default=True)
@click.option('--metric-names',     help='Metrics to compute', metavar='LIST', type=str, default='fid', show_default=True)
@click.option('--metric-num-samples', help='# samples for Frechet metrics', metavar='INT', type=click.IntRange(min=2), default=10000, show_default=True)
@click.option('--mind-num-samples', help='# samples for MIND metrics', metavar='INT', type=click.IntRange(min=2), default=5000, show_default=True)
@click.option('--metric-ref',       help='Reference statistics', metavar='PATH', type=str, default='fid-refs/cifar10.pkl', show_default=True)
@click.option('--metric-batch-size',help='Per-rank metric batch', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--seed',             help='Random seed', metavar='INT', type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print config and exit', is_flag=True)
def cmdline(outdir, dry_run, eps, sinkhorn_num_iter, **opts):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    # --eps is a convenience override for the (single-value) entropic reg list.
    if eps is not None:
        opts['R_list'] = [eps]
    if sinkhorn_num_iter is not None:
        opts['sinkhorn_num_iter'] = sinkhorn_num_iter
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

    print_training_config(run_dir=run_dir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=run_dir, c=c)
    torch.distributed.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
