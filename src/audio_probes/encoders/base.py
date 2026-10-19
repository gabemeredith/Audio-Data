"""The Encoder interface: audio in, one pooled vector per layer out."""

from __future__ import annotations

import numpy as np


class Encoder:
    name: str = "base"

    def layers(self, audio: np.ndarray, sr: int) -> list[np.ndarray]:
        """One pooled vector per layer for a single clip."""
        return [layer[0] for layer in self.layers_batch([audio], sr)]

    def layers_batch(self, audios: list[np.ndarray], sr: int) -> list[np.ndarray]:
        """Batched version: returns a list over layers of arrays shaped (batch, dim)."""
        per_clip = [self.layers(a, sr) for a in audios]
        return [np.stack([clip[i] for clip in per_clip]) for i in range(len(per_clip[0]))]
