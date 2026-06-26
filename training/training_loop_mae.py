"""ResNet-MAE pretraining loop (PyTorch port of JAX `train_mae.py`).

Trains the feature encoder used by the drift loss: a ResNet-MAE that
reconstructs randomly (2x2-patch) masked inputs with an L2 loss on the masked
region (Appendix A.3). Optionally fine-tunes with a linear classifier head over
the last `finetune_last_nimg` images, ramping `lambda_cls` up via a warmup.

Reuses the EDM2-style status / snapshot / checkpoint / EMA / W&B machinery so
MAE snapshots (`model-snapshot-*.pkl`) can be consumed directly by the drift
generator training loop.
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

#----------------------------------------------------------------------------

def training_loop_mae(
    dataset_kwargs,
    encoder_kwargs,
    data_loader_kwargs,
    model_kwargs,
    optimizer_kwargs,
    lr_kwargs,
    ema_kwargs,
    max_clip_norm,

    run_dir,
    seed,
    batch_size,
    max_batch_gpu,
    total_nimg,
    status_nimg,
    snapshot_nimg,
    checkpoint_nimg,

    mask_ratio_min,
    mask_ratio_max,
    finetune_last_nimg,      # Enable cls finetune over the last N images.
    warmup_finetune_nimg,    # lambda_cls warmup duration.
    finetune_cls,            # Target lambda_cls.

    loss_scaling,
    cudnn_benchmark,
    force_finite,
):
    device = torch.device('cuda')

    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    batch_gpu_total = batch_size // dist.get_world_size()
    if max_batch_gpu is None or max_batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    else:
        batch_gpu = max_batch_gpu
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0

    # Dataset and encoder.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    num_classes = max(ref_label.shape[-1], 1)

    # Model.
    dist.print0('Constructing MAE...')
    model = dnnlib.util.construct_class_by_name(
        in_channels=ref_image.shape[1], num_classes=num_classes, **model_kwargs)
    model.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        n_params = sum(p.numel() for p in model.parameters())
        dist.print0(f'MAE parameters: {n_params/1e6:.2f}M')

    # Training state.
    state = dnnlib.EasyDict(cur_nimg=0, cur_step=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    optimizer = dnnlib.util.construct_class_by_name(params=model.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(model=model, **ema_kwargs) if ema_kwargs is not None else None

    checkpoint = dist.CheckpointIO(state=state, model=model, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    assert total_nimg > state.cur_nimg
    dist.print0(f'Training MAE from {state.cur_nimg // 1000} kimg to {total_nimg // 1000} kimg:')

    wandb_run = None
    if dist.get_rank() == 0:
        if not state.get('wandb_run_id', None):
            state.wandb_run_id = wandb.util.generate_id()
        wandb_run = wandb.init(project='drifting-mae', name=os.path.basename(run_dir),
                               dir=run_dir, id=state.wandb_run_id, resume='allow')
        monitoring.setup_wandb_metrics(wandb)

    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(),
        seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(
        dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    start_step = state.cur_step
    stats_jsonl = None
    step_stats = dnnlib.EasyDict()
    start_finetune_nimg = total_nimg - finetune_last_nimg

    while True:
        done = (state.cur_nimg >= total_nimg)

        # ---- Status. ----
        first_step_report = (state.cur_step == start_step + 1)
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0 or first_step_report) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',     f"{training_stats.report0('Progress/kimg', state.cur_nimg / 1e3):<9.1f}",
                'time',     f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec', state.total_elapsed_time)):<12s}",
                'sec/tick', f"{cur_time - prev_status_time:<8.2f}",
                'recon',    f"{step_stats.get('recon_loss', float('nan')):<7.4f}",
                'acc',      f"{step_stats.get('accuracy', float('nan')):<6.3f}",
                'gpumem',   f"{training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
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
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + ('%g' % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            if wandb_run is not None:
                monitoring.log_to_wandb(
                    wandb, cur_step=state.cur_step, cur_nimg=state.cur_nimg,
                    elapsed_time=state.total_elapsed_time,
                    main_metrics={
                        'recon_loss': step_stats.get('recon_loss', float('nan')),
                        'cls_loss': step_stats.get('cls_loss', float('nan')),
                        'accuracy': step_stats.get('accuracy', float('nan')),
                        'lr': step_stats.get('lr', float('nan')),
                    },
                    metrics={'lambda_cls': step_stats.get('lambda_cls', 0.0),
                             'grad_norm': step_stats.get('grad_norm', float('nan')),
                             'sec_per_tick': sec_per_tick, 'sec_per_kimg': sec_per_kimg},
                )
            dist.update_progress(state.cur_nimg // 1000, total_nimg // 1000)
            if dist.should_stop() or dist.should_suspend():
                done = True

        # ---- Snapshot. ----
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else model
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_model, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs)
                data.ema = copy.deepcopy(ema_model).cpu().eval().requires_grad_(False)
                fname = f'model-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data

        # ---- Checkpoint. ----
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(model)

        if done:
            break

        # ---- Training step. ----
        lambda_cls = 0.0
        if finetune_last_nimg > 0 and state.cur_nimg >= start_finetune_nimg:
            prog = (state.cur_nimg - start_finetune_nimg) / max(1, warmup_finetune_nimg)
            lambda_cls = finetune_cls * min(1.0, prog)

        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_step)
        optimizer.zero_grad(set_to_none=True)
        acc = dnnlib.EasyDict(recon_loss=0.0, cls_loss=0.0, accuracy=0.0)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                labels_idx = labels.argmax(dim=1).to(device)
                loss, metrics = ddp(images, labels_idx, lambda_cls=lambda_cls,
                                    mask_ratio_min=mask_ratio_min, mask_ratio_max=mask_ratio_max)
                acc.recon_loss += metrics['recon_loss'].item() / num_accumulation_rounds
                acc.cls_loss += metrics['cls_loss'].item() / num_accumulation_rounds
                acc.accuracy += metrics['accuracy'].item() / num_accumulation_rounds
                (loss * (loss_scaling / num_accumulation_rounds)).backward()

        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
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
        optimizer.step()

        step_stats.recon_loss = acc.recon_loss
        step_stats.cls_loss = acc.cls_loss
        step_stats.accuracy = acc.accuracy
        step_stats.lr = lr
        step_stats.grad_norm = grad_norm.item()
        step_stats.lambda_cls = lambda_cls

        state.cur_nimg += batch_size
        state.cur_step += 1
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

    if dist.get_rank() == 0 and wandb_run is not None:
        wandb.finish()

#----------------------------------------------------------------------------
