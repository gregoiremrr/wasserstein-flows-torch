# /etc/nccl.conf on GCE Deep Learning VMs forces NCCL_NET=gIB, which is
# Google's GPUDirect-RDMA networking stack for A3/H100 instances. On A2/A100
# (or any single-node run) gIB has no fabric to talk to and fails to init.
# Override here to use the built-in Socket transport for the control plane;
# NCCL still uses NVLink/P2P for actual GPU<->GPU traffic.
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train the one-step W-Flow generator. Requires a pretrained ResNet-MAE
# snapshot (see script-mae-cifar10.sh) passed via --mae-pkl.
torchrun --standalone --nproc_per_node=4 train.py \
    --outdir=training-runs/cifar10 \
    --data=../datasets/cifar10.zip \
    --mae-pkl=../drifting-models-torch/training-runs/cifar10-mae/260605_155827_mae-cifar10/model-snapshot-0050331.pkl \
    --preset=wflow-cifar10 \
    --no-fp16 \
    --status=100 \
    --snapshot=1000 \
    --checkpoint=3000 \
    --metrics=1000 \
    --metric-names=fid,fd_dinov2,mind,mind_dinov2 \
    --metric-num-samples=20000 \
    --mind-num-samples=5000 \
    --metric-ref=../fid-refs/cifar10.pkl \
    --metric-batch-size=32
