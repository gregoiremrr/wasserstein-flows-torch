export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume an existing W-Flow run: point --outdir at the timestamped run dir that
# already contains training-state-*.pt checkpoints (the preset/mae-pkl are read
# back from training_options.json on resume).
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10/<run-dir> \
    --data=datasets/cifar10.zip \
    --mae-pkl=training-runs/cifar10-mae/<run-dir>/model-snapshot-<latest>.pkl \
    --preset=wflow-cifar10 \
    --no-fp16 \
    --status=1000 \
    --snapshot=10000 \
    --checkpoint=30000 \
    --metrics=10000 \
    --metric-names=fid,fd_dinov2,mind,mind_dinov2 \
    --metric-num-samples=20000 \
    --mind-num-samples=5000 \
    --metric-ref=../fid-refs/cifar10.pkl \
    --metric-batch-size=32
