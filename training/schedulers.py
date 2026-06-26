import numpy as np

#----------------------------------------------------------------------------
# Schedulers

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

def cosine_lr(cur_nimg, batch_size, base_lr=1e-3, total_nimg=200_000*512, warmup_nimg=5_000*512):
    if cur_nimg < warmup_nimg:
        return base_lr * cur_nimg / warmup_nimg
    progress = (cur_nimg - warmup_nimg) / (total_nimg - warmup_nimg)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

def warmup_const_lr(cur_nimg, batch_size, base_lr=2e-4, total_nimg=None, warmup_nimg=5_000*256):
    # Linear warmup then constant. Matches the W-Flow / Drifting default schedule.
    if warmup_nimg > 0 and cur_nimg < warmup_nimg:
        return base_lr * cur_nimg / warmup_nimg
    return base_lr

#----------------------------------------------------------------------------
