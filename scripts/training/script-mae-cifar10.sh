export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# Pretrain the ResNet-MAE feature encoder consumed by the W-Flow drift loss.
# Point train.py --mae-pkl at one of the resulting model-snapshot-*.pkl files.
torchrun --standalone --nproc_per_node=4 train_mae.py \
    --outdir=training-runs/cifar10-mae \
    --data=datasets/cifar10.zip \
    --preset=mae-cifar10 \
    --max-batch-gpu=128 \
    --status=512 \
    --snapshot=16Ki \
    --checkpoint=64Ki
