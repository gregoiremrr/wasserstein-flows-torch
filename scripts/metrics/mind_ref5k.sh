# One-time setup: extend an existing FID/FD-DINOv2 ref pkl with MIND
# reference features (5k samples, both Inception-v3 and DINOv2 backbones).
# `--merge` (default) preserves the existing `fid` and `fd_dinov2` entries
# already in the pkl.
#
# Usage:
#   sh scripts/metrics/mind_ref5k.sh
#
# Same NCCL env tweaks as scripts/training/script-cifar10.sh so the run
# works on single-node A100/H100 boxes where /etc/nccl.conf might force a
# fabric (e.g. NCCL_NET=gIB on GCE Deep Learning VMs) with no peer.

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

torchrun --standalone --nproc_per_node=2 calculate_metrics.py ref \
    --data=datasets/cifar10.zip \
    --dest=fid-refs/cifar10.pkl \
    --metrics=mind,mind_dinov2 \
    --num-images=5000
