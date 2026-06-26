python calculate_metrics.py calc \
    --images=out/classic-cifar-10-nocfg \
    --ref=fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --max-batch-size=512
