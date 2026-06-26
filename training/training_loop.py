"""W-Flow one-step generator training loop.

EDM2-style infrastructure (status / snapshot / checkpoint / metric / W&B
machinery) wrapped around the Sinkhorn drift training step:

  1. Push freshly loaded reals into per-class (positive) and global
     (unconditional) memory-bank queues.
  2. Sample a batch of class labels; draw positives and unconditional negatives
     from the queues.
  3. Sample a CFG scale per label; the loss generates `gen_per_label` samples
     per label (plus an independent second batch for the debiased
     self-transport) and computes the multi-feature Sinkhorn drift loss.
  4. Update the generator.

The drift loss is computed in the feature space of a frozen, pretrained
ResNet-MAE (loaded from a snapshot pickle, ``mae_pkl``).
"""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import wandb
from training import monitoring
from training import evaluation
from training.memory_bank import ArrayMemoryBank

#----------------------------------------------------------------------------

def _load_feature_encoder(mae_pkl, device):
    if mae_pkl is None:
        dist.print0('No MAE feature encoder (raw-pixel drift loss only).')
        return None
    dist.print0(f'Loading MAE feature encoder from {mae_pkl}...')
    with open(mae_pkl, 'rb') as f:
        data = pickle.load(f)
    mae = data['ema'] if isinstance(data, dict) else data.ema
    # torch_utils.persistence pickles the MAE's source code, so the unpickled
    # object's methods (notably get_activations) run the code as it was AT SAVE
    # TIME. Rebuild from the *current* MAEResNet source and copy the trained
    # weights over, so source edits (e.g. gradient flow through MAE features)
    # actually take effect.
    from training.networks_mae import MAEResNet
    fresh = MAEResNet(*mae.init_args, **mae.init_kwargs)
    misc.copy_params_and_buffers(mae, fresh, require_all=True)
    mae = fresh.to(device).eval().requires_grad_(False)
    return mae

#----------------------------------------------------------------------------

def training_loop(
    dataset_kwargs,
    encoder_kwargs,
    data_loader_kwargs,
    model_kwargs,
    loss_kwargs,
    optimizer_kwargs,
    lr_kwargs,
    ema_kwargs,
    sampler_kwargs,
    mae_pkl,                 # Pretrained ResNet-MAE snapshot pickle (or None).
    max_clip_norm,

    run_dir,
    seed,
    batch_size,             # Effective global batch (labels_per_step * gen_per_label * world_size).
    labels_per_step,        # Class labels (Nc) per rank per step.
    pos_per_sample,         # Positives drawn per label.
    neg_per_sample,         # Unconditional negatives drawn per label.
    positive_bank_size,     # Per-class positive queue size.
    negative_bank_size,     # Global unconditional queue size.
    push_per_step,          # Reals pushed into the queues per step (per rank).
    push_at_resume,         # Extra queue-fill multiplier on resume.

    total_nimg,
    status_nimg,
    snapshot_nimg,
    checkpoint_nimg,
    metrics_nimg,
    metrics_kwargs,

    loss_scaling,
    cudnn_benchmark,
    force_finite,

    loss_microbatch_labels=None,   # Split the Nc labels into chunks of this size and accumulate grads. None/0 => no split.
):
    device = torch.device('cuda')

    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    assert total_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or snapshot_nimg % batch_size == 0
    assert checkpoint_nimg is None or checkpoint_nimg % batch_size == 0
    assert metrics_nimg is None or metrics_nimg % batch_size == 0
    if metrics_nimg is not None:
        assert metrics_kwargs is not None and metrics_kwargs.get('ref_path'), \
            '--metrics requires --metric-ref to be set'

    # Report the batch composition (mirrors the base repo). The drift loss treats
    # each class label as an independent group, so per-rank gradient accumulation
    # is over chunks of `loss_microbatch_labels` labels rather than image
    # micro-batches.
    world_size = dist.get_world_size()
    assert batch_size % (labels_per_step * world_size) == 0, \
        'batch_size must equal labels_per_step * gen_per_label * world_size'
    batch_gpu_total = batch_size // world_size
    gen_per_label = batch_gpu_total // labels_per_step
    mb_labels = loss_microbatch_labels if (loss_microbatch_labels and loss_microbatch_labels > 0) else labels_per_step
    num_accumulation_rounds = (labels_per_step + mb_labels - 1) // mb_labels
    dist.print0(f'Batch size: total {batch_size}, per-GPU {batch_gpu_total} '
                f'(micro-batch {mb_labels * gen_per_label} x {num_accumulation_rounds} accumulation rounds, '
                f'{mb_labels}/{labels_per_step} labels per chunk), '
                f'GPUs {world_size}')

    # Dataset and encoder.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    num_classes = ref_label.shape[-1]
    assert num_classes > 0, 'W-Flow models are class-conditional; use a labeled dataset.'

    # Model.
    dist.print0('Constructing model...')
    interface_kwargs = dict(
        img_resolution=ref_image.shape[-1],
        img_channels=ref_image.shape[1],
        label_dim=num_classes,
    )
    model = dnnlib.util.construct_class_by_name(**model_kwargs, **interface_kwargs)
    model.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        n_params = sum(p.numel() for p in model.parameters())
        dist.print0(f'Generator parameters: {n_params/1e6:.2f}M')

    # Frozen feature encoder.
    feature_encoder = _load_feature_encoder(mae_pkl, device)

    # Training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, cur_step=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=model.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(model=model, **ema_kwargs) if ema_kwargs is not None else None

    checkpoint = dist.CheckpointIO(state=state, model=model, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    assert total_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {total_nimg // 1000} kimg:')
    dist.print0()

    # WandB.
    wandb_run = None
    if dist.get_rank() == 0:
        if not state.get('wandb_run_id', None):
            state.wandb_run_id = wandb.util.generate_id()
        wandb_run = wandb.init(project='wflow', name=os.path.basename(run_dir),
                               dir=run_dir, id=state.wandb_run_id, resume='allow')
        monitoring.setup_wandb_metrics(wandb)

    # Data loader.
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(),
        seed=seed, start_idx=state.cur_nimg,
    )
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(
        dataset=dataset_obj, sampler=dataset_sampler, batch_size=push_per_step, **data_loader_kwargs))

    # Memory-bank queues for real positives + unconditional negatives.
    bank_pos = ArrayMemoryBank(num_classes=num_classes, max_size=positive_bank_size)
    bank_neg = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)

    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    start_step = state.cur_step
    stats_jsonl = None
    step_stats = dnnlib.EasyDict()

    while True:
        done = (state.cur_nimg >= total_nimg)

        # ---- Report status. ----
        first_step_report = (state.cur_step == start_step + 1)
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0 or first_step_report) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
                'loss',         f"{step_stats.get('loss', float('nan')):<6.4f}",
            ]))
            sec_per_tick = cur_time - prev_status_time
            sec_per_kimg = cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            if wandb_run is not None:
                grid_np = None
                if ema is not None:
                    ema_model = ema.get()
                    if isinstance(ema_model, list):
                        ema_model = ema_model[0][0]
                    ema_model.eval()
                    grid = monitoring.generate_sample_grid(
                        ema_model, encoder, sampler_kwargs,
                        n_samples=16, label_dim=model.label_dim,
                        seed=(state.cur_nimg // status_nimg) % 20, device=device,
                    )
                    grid_np = grid.permute(1, 2, 0).cpu().numpy()

                main_metrics = {
                    'loss': step_stats.get('loss', float('nan')),
                    'lr': step_stats.get('lr', float('nan')),
                    'grad_norm': step_stats.get('grad_norm', float('nan')),
                }
                metrics = {
                    'cfg': step_stats.get('cfg', float('nan')),
                    'clip_coef': step_stats.get('clip_coef', float('nan')),
                    'sec_per_tick': sec_per_tick,
                    'sec_per_kimg': sec_per_kimg,
                }
                plot_caption = (f"nimg: {state.cur_nimg}, nstep: {state.cur_step}, "
                                f"ntime: {dnnlib.util.format_time(state.total_elapsed_time)} "
                                f"({int(state.total_elapsed_time)}s)")
                main_plots = {'samples': wandb.Image(grid_np, caption=plot_caption)} if grid_np is not None else None
                monitoring.log_to_wandb(
                    wandb, cur_step=state.cur_step, cur_nimg=state.cur_nimg,
                    elapsed_time=state.total_elapsed_time, main_metrics=main_metrics,
                    metrics=metrics, main_plots=main_plots, plots=None,
                )

            dist.update_progress(state.cur_nimg // 1000, total_nimg // 1000)
            if dist.should_stop() or dist.should_suspend():
                done = True

        # ---- Eval metrics. ----
        if (metrics_nimg is not None and (done or state.cur_nimg % metrics_nimg == 0) and state.cur_nimg != start_nimg):
            if ema is not None:
                ema_model = ema.get()
                if isinstance(ema_model, list):
                    ema_model = ema_model[0][0]
                ema_model.eval()
                metric_start = time.time()
                dist.print0(f"Computing metrics ({', '.join(metrics_kwargs['metrics'])})...")
                metric_results = evaluation.compute_metrics(
                    model=ema_model, encoder=encoder, sampler_kwargs=sampler_kwargs,
                    ref_path=metrics_kwargs['ref_path'], num_samples=metrics_kwargs['num_samples'],
                    mind_num_samples=metrics_kwargs.get('mind_num_samples', 5000),
                    metrics=metrics_kwargs['metrics'], max_batch_size=metrics_kwargs['max_batch_size'],
                    seed=0, device=device,
                )
                metric_elapsed = time.time() - metric_start
                if dist.get_rank() == 0 and metric_results is not None:
                    msg = ', '.join(f'{k}={v:g}' for k, v in metric_results.items())
                    dist.print0(f'Metrics @ kimg {state.cur_nimg/1e3:.1f}: {msg} (took {metric_elapsed:.1f}s)')
                    if wandb_run is not None:
                        monitoring.log_to_wandb(
                            wandb, cur_step=state.cur_step, cur_nimg=state.cur_nimg,
                            elapsed_time=state.total_elapsed_time, main_eval_metrics=metric_results,
                            metrics={'metric_eval_sec': metric_elapsed},
                        )
                prev_status_time = time.time()

        # ---- Save snapshot. ----
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and state.cur_nimg != start_nimg and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else model
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_model, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_model).cpu().eval().requires_grad_(False)
                fname = f'model-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data

        # ---- Save checkpoint. ----
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(model)

        if done:
            break

        # ---- Training step. ----
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_step)

        # 1. Fill the memory-bank queues with fresh reals.
        goal = push_per_step
        if start_step > 0 and state.cur_step == start_step:
            goal = push_at_resume * push_per_step
        pushed = 0
        pushed_labels = []
        while pushed < goal:
            images, labels = next(dataset_iterator)
            images = encoder.encode_latents(images.to(device))
            label_idx = labels.argmax(dim=1).cpu().numpy()
            img_np = images.detach().cpu().numpy()
            bank_pos.add(img_np, label_idx)
            bank_neg.add(img_np, np.zeros_like(label_idx))
            pushed += images.shape[0]
            pushed_labels.append(label_idx)
        pushed_labels = np.concatenate(pushed_labels)

        # 2. Select Nc labels and draw positives / unconditional negatives.
        sel = np.random.choice(pushed_labels, labels_per_step, replace=(len(pushed_labels) < labels_per_step))
        sel_labels = torch.as_tensor(sel, dtype=torch.long, device=device)
        pos_images = bank_pos.sample(sel, pos_per_sample).to(device).float()
        if neg_per_sample > 0:
            uncond_images = bank_neg.sample(np.zeros_like(sel), neg_per_sample).to(device).float()
        else:
            uncond_images = pos_images[:, :0]

        # 3. Sample CFG scale per label.
        cfg = loss_fn.sample_cfg(labels_per_step, device)

        # 4. Drift loss + update. Each label is an independent group in the
        # drift loss, so the Nc labels can be processed in micro-batches with
        # gradient accumulation to cap peak memory. The per-chunk loss is
        # weighted by (chunk_labels / Nc) so the accumulated gradient matches
        # the full-batch mean; DDP all-reduce is deferred to the final chunk.
        optimizer.zero_grad(set_to_none=True)
        Nc = sel_labels.shape[0]
        mb = loss_microbatch_labels if (loss_microbatch_labels and loss_microbatch_labels > 0) else Nc
        loss_accum = 0.0
        cfg_accum = 0.0
        for i in range(0, Nc, mb):
            sl = slice(i, min(i + mb, Nc))
            chunk_w = sel_labels[sl].shape[0] / Nc
            is_last = (i + mb >= Nc)
            with misc.ddp_sync(ddp, is_last):
                chunk_loss, chunk_stats = loss_fn(
                    model=ddp, feature_encoder=feature_encoder,
                    labels=sel_labels[sl], pos_images=pos_images[sl],
                    uncond_images=uncond_images[sl], cfg=cfg[sl],
                )
                (chunk_loss * loss_scaling * chunk_w).backward()
            loss_accum += chunk_loss.item() * chunk_w
            cfg_accum += chunk_stats['cfg'].item() * chunk_w
        training_stats.report('Loss/loss', loss_accum)

        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        inv_scale = 1 / loss_scaling
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(inv_scale)
                if force_finite:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

        clip_norm = max_clip_norm if (max_clip_norm is not None and max_clip_norm > 0) else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        clip_coef = min(1.0, clip_norm / (grad_norm.item() + 1e-12))
        optimizer.step()

        step_stats.loss = loss_accum
        step_stats.lr = lr
        step_stats.grad_norm = grad_norm.item()
        step_stats.clip_coef = clip_coef
        step_stats.cfg = cfg_accum

        state.cur_nimg += batch_size
        state.cur_step += 1
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

    if dist.get_rank() == 0 and wandb_run is not None:
        wandb.finish()

#----------------------------------------------------------------------------
