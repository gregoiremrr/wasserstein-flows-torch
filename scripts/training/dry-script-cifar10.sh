# Quick single-GPU smoke test of the W-Flow generator config (no real run).
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs/cifar10 \
    --data=datasets/cifar10.zip \
    --mae-pkl=training-runs/cifar10-mae/<run-dir>/model-snapshot-<latest>.pkl \
    --preset=wflow-cifar10-debug \
    --status=128 \
    --snapshot=1280 \
    --checkpoint=2560 \
    --dry-run
