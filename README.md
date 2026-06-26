# W-Flow (PyTorch)

A multi-GPU PyTorch implementation of **"One-Step Generative Modeling via
Wasserstein Gradient Flows"** (W-Flow, arXiv:2605.11755), adapted to
**CIFAR-10 in pixel space (no VAE)**.

W-Flow is a *one-step* (1-NFE) generator: a LightningDiT transformer maps
Gaussian noise directly to an image in a single forward pass. There is no
iterative sampler. Training does not use a fixed regression target; instead the
generated distribution `q` evolves along the steepest-descent direction of the
**(debiased) Sinkhorn divergence** `S_eps(q, p)` in Wasserstein space. The
induced velocity field is a difference of two entropic-OT barycentric maps,

```
V(x) = T_{q,p}(x) - T_{q,q'}(x)        (attraction toward reals - debiased self-transport)
```

and the loss regresses each generated sample onto its frozen drifted target
`stopgrad(x + V)`. Two design choices distinguish this from a plain mean-shift
drift and are the *only* conceptual changes from the softmax drifting field:

- **Two-batch self-transport.** `T_{q,q'}` is estimated against an independent
  second generated batch `q'` rather than diagonal-masking the batch against
  itself — a cleaner debiasing of the self term (Sec. 3.3).
- **Velocity-guidance CFG.** Guidance is injected at the velocity level via
  `w * (T_{q,p} - T_{q,uncond})` (Eq. 16) instead of re-weighting unconditional
  negatives.

The drift is computed in the feature space of a frozen, pretrained
**ResNet-MAE**, so you train the MAE first and the generator second.

The surrounding infrastructure (`dnnlib`, `torch_utils`, EMA, `dataset_tool.py`,
FID/FD-DINOv2 metrics, persistence-based pickling) is borrowed from NVIDIA's
[EDM2](https://github.com/NVlabs/edm2). The generator, MAE, Sinkhorn drift loss,
memory bank, samplers, and configuration layout sit on top.

## Layout

```
.
├── train.py                    # Build a config and launch generator (W-Flow) training.
├── train_mae.py                # Build a config and launch ResNet-MAE pretraining.
├── generate_images.py          # Sample (1-NFE) from a saved snapshot pickle.
├── calculate_metrics.py        # FID and FD-DINOv2 against a reference dataset.
├── reconstruct_phema.py        # Post-hoc EMA reconstruction (EDM2).
├── dataset_tool.py             # Pack a folder of images into a zip dataset.
├── training/
│   ├── training_loop.py        # Generator (W-Flow drift) training loop.
│   ├── training_loop_mae.py    # ResNet-MAE pretraining loop.
│   ├── model.py                # DriftingModel wrapper + 1-NFE sample().
│   ├── loss.py                 # DriftLoss (multi-feature, two-batch, velocity-CFG).
│   ├── drift_field.py          # Sinkhorn debiased velocity + per-feature drift loss.
│   ├── memory_bank.py          # Class-wise sample queues (real positives + unconditional).
│   ├── networks_dit.py         # LightningDiT generator (SwiGLU/RoPE/RMSNorm/QK-norm/adaLN-zero).
│   ├── networks_mae.py         # ResNet-MAE encoder + U-Net decoder + get_activations().
│   ├── encoders.py             # StandardRGB ([-1,1] pixels) and Stability VAE encoders.
│   ├── schedulers.py           # LR schedules (incl. warmup_const_lr).
│   ├── phema.py                # Power-function, fixed-decay, and traditional EMA.
│   ├── monitoring.py           # W&B logging helpers.
│   ├── evaluation.py           # Distributed FID/MIND computation.
│   └── dataset.py              # Streaming image dataset (zip or folder).
├── torch_utils/                # Distributed, persistence, training stats (EDM).
├── dnnlib/                     # EasyDict, class/func construction by name (EDM).
├── scripts/                    # Shell scripts: env setup, training, metrics.
├── datasets/                   # Place your packed datasets here.
├── training-runs/              # Output runs (one timestamped subdir per launch).
├── fid-refs/                   # Reference statistics for FID/FD-DINOv2.
└── out/                        # Generated images.
```

## Setup

```bash
# Install the Python environment (CUDA 12.x wheel; adjust as needed).
bash scripts/module.sh
```

A `Dockerfile` is also provided.

## End-to-end workflow

### 1. Pack the dataset

```bash
python dataset_tool.py convert \
    --source=raw_cifar/ \
    --dest=datasets/cifar10.zip \
    --resolution=32x32
```

W-Flow is class-conditional, so the dataset **must** have labels.

### 2. Pretrain the ResNet-MAE feature encoder

```bash
torchrun --standalone --nproc_per_node=4 train_mae.py \
    --outdir=training-runs/cifar10-mae \
    --data=datasets/cifar10.zip \
    --preset=mae-cifar10
```

This produces `model-snapshot-*.pkl` files containing the EMA encoder. The MAE
reconstructs randomly (2x2-patch) masked inputs and (over the final images)
fine-tunes a small linear classifier head. The encoder is architecturally
identical to the Drifting-model MAE, so an existing drifting MAE snapshot can be
reused directly via `--mae-pkl`.

### 3. Train the generator with the Sinkhorn drift loss

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10 \
    --data=datasets/cifar10.zip \
    --mae-pkl=training-runs/cifar10-mae/<run-dir>/model-snapshot-<latest>.pkl \
    --preset=wflow-cifar10
```

Each launch creates a timestamped subdirectory inside `--outdir`. Pointing
`--outdir` at an existing run that contains a `training-state-*.pt` resumes from
the latest checkpoint. Use the `*-debug` presets (`mae-cifar10-debug`,
`wflow-cifar10-debug`) for fast single-GPU smoke tests. The entropic
regularisation can be overridden with `--eps` and the Sinkhorn iteration count
with `--sinkhorn-num-iter`.

### 4. Generate images (1-NFE)

```bash
python generate_images.py \
    --model=training-runs/cifar10/<run-dir>/model-snapshot-<latest>.pkl \
    --sampler-fn=training.model.sample \
    --outdir=out --seeds=0-63 --guidance=1.0
```

`guidance` is the training-time-style CFG scale fed to the generator as a
conditioning input; `--n-sampling-steps` is accepted but ignored (the model is
one-step).

### 5. Compute reference statistics and FID

```bash
bash scripts/metrics/ref50k.sh      # build fid-refs/cifar10.pkl
bash scripts/metrics/fid50k.sh      # FID / FD-DINOv2
```

FID can also be computed inline during generator training via `--metrics` /
`--metric-ref`.

## How the W-Flow training step works

Each generator step (`training/training_loop.py`):

1. **Fill the queues.** Freshly loaded reals are pushed into per-class
   (positive) and a global (unconditional) ring buffer (`memory_bank.py`) — a
   MoCo-style queue rather than a bespoke data loader.
2. **Draw a batch.** Sample `labels_per_step` class labels; draw real positives
   and unconditional negatives from the queues.
3. **Generate + featurize.** Sample a CFG scale per label, generate
   `gen_per_label` samples per label (and an independent `self_gen_per_label`
   second batch for the self-transport term), and extract multi-scale /
   multi-location features for reals and fakes through the frozen MAE.
4. **Drift.** For each feature compute the debiased Sinkhorn velocity
   (`drift_field.py`): normalize features, run Sinkhorn-Knopp to obtain the
   `q->p` and `q->q'` transport plans, form `V = T_{q,p} - T_{q,q'}` (plus the
   velocity-guidance CFG term), normalize per `eps`, and regress the generator
   output onto its frozen drifted target.

## Configuration

Both launchers split configuration into two preset dictionaries:

- `dataset_presets` holds everything intrinsic to the data: the network
  architecture (`net_kwargs`), the monitoring sampler (`sampler_kwargs`), EMA
  decay, and the LR scheduler family (`lr_scheduler_kwargs`).
- `config_presets` describes a particular run: sample allocation
  (`labels_per_step`, `gen_per_label`, `self_gen_per_label`, `pos_per_sample`,
  `neg_per_sample`), queue sizes, CFG range (`cfg_min`/`cfg_max`/`neg_cfg_pw`),
  Sinkhorn knobs (`R_list` = entropic-reg eps values, `sinkhorn_num_iter`,
  `use_quadratic_cost`, `disable_diag_mask`), optimization budget, LR, warmup,
  and gradient clipping.

The two dictionaries must have disjoint keys (asserted at startup). Per-run
overrides are exposed as CLI flags.

## Monitoring

Loss, learning rate, gradient norm, gradient-clip coefficient, mean CFG scale,
and timing counters are pushed to W&B at every `--status` interval, alongside a
grid of 1-NFE samples from the EMA generator. The MAE loop logs reconstruction
loss, classification loss/accuracy, and `lambda_cls`.

## Credits

W-Flow method: "One-Step Generative Modeling via Wasserstein Gradient Flows"
(arXiv:2605.11755), building on "Generative Modeling via Drifting"
(arXiv:2602.04770). Infrastructure built on NVIDIA's
[EDM](https://github.com/NVlabs/edm) and [EDM2](https://github.com/NVlabs/edm2).
