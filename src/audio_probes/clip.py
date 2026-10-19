"""The Clip record passed between stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Clip:
    """One labelled audio example.

    meta always contains `room_id` and `source_id`; the train/test split groups on them.
    """

    id: str
    audio: np.ndarray  # float32 mono, shape (n_samples,)
    sr: int
    label: float
    meta: dict = field(default_factory=dict)
