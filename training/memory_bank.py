"""Class-wise sample queue (PyTorch port of the JAX `memory_bank.py`).

Following the MoCo-style queue described in Appendix A.8 of "Generative
Modeling via Drifting", real (positive / unconditional) samples are cached
in per-class ring buffers instead of using a specialized data loader. At each
training step new reals are pushed in (oldest dequeued), and positives /
unconditional negatives are drawn from the queues.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ArrayMemoryBank:
    """Per-class ring buffer for image/feature samples."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int64)
        self.count = np.zeros(self.num_classes, dtype=np.int64)

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype)

    def add(self, samples, labels) -> None:
        """Insert ``samples`` (``[N, *feature_shape]``) into per-class buffers."""
        samples = np.asarray(samples, dtype=self.dtype)
        labels = np.asarray(labels)
        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def sample(self, labels, n_samples: int) -> torch.Tensor:
        """Draw ``n_samples`` per label.

        Returns a tensor of shape ``(len(labels), n_samples, *feature_shape)``.
        Sampling is without replacement when enough valid entries exist.
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError('MemoryBank is empty. Call add() before sample().')

        labels = np.asarray(labels)
        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int64)
        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int64)
            else:
                sample_indices[i] = np.random.choice(valid, n_samples, replace=(valid < n_samples))

        out = self.bank[labels[:, None], sample_indices]
        return torch.from_numpy(out)

    @property
    def ready(self) -> bool:
        return self.bank is not None and bool((self.count > 0).any())
