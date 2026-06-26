# 30 steps
torchrun --standalone --nproc_per_node=1 generate_images.py \
	--outdir=out/classic-cifar-10-nocfg-30steps \
	--subdirs \
	--seeds=0-49999 \
	--preset=classic-cifar-10-30steps \
	--max-batch-size=2048 \
	--encoder-batch-size=2048

python calculate_metrics.py calc \
    --images=out/classic-cifar-10-nocfg-30steps \
    --ref=fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --max-batch-size=512

# 20 steps
torchrun --standalone --nproc_per_node=1 generate_images.py \
	--outdir=out/classic-cifar-10-nocfg-20steps \
	--subdirs \
	--seeds=0-49999 \
	--preset=classic-cifar-10-20steps \
	--max-batch-size=2048 \
	--encoder-batch-size=2048

python calculate_metrics.py calc \
    --images=out/classic-cifar-10-nocfg-20steps \
    --ref=fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --max-batch-size=512

# 10 steps
torchrun --standalone --nproc_per_node=1 generate_images.py \
	--outdir=out/classic-cifar-10-nocfg-10steps \
	--subdirs \
	--seeds=0-49999 \
	--preset=classic-cifar-10-10steps \
	--max-batch-size=2048 \
	--encoder-batch-size=2048

python calculate_metrics.py calc \
    --images=out/classic-cifar-10-nocfg-10steps \
    --ref=fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --max-batch-size=512

# 5 steps
torchrun --standalone --nproc_per_node=1 generate_images.py \
	--outdir=out/classic-cifar-10-nocfg-5steps \
	--subdirs \
	--seeds=0-49999 \
	--preset=classic-cifar-10-5steps \
	--max-batch-size=2048 \
	--encoder-batch-size=2048

python calculate_metrics.py calc \
    --images=out/classic-cifar-10-nocfg-5steps \
    --ref=fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --max-batch-size=512

# 2 steps
torchrun --standalone --nproc_per_node=1 generate_images.py \
	--outdir=out/classic-cifar-10-nocfg-2steps \
	--subdirs \
	--seeds=0-49999 \
	--preset=classic-cifar-10-2steps \
	--max-batch-size=2048 \
	--encoder-batch-size=2048

python calculate_metrics.py calc \
    --images=out/classic-cifar-10-nocfg-2steps \
    --ref=fid-refs/cifar10.pkl \
    --metrics=fid,fd_dinov2 \
    --max-batch-size=512
