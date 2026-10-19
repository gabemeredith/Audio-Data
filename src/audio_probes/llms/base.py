"""The AudioLLM interface: audio and a text prompt in, the model's raw text answer out."""

from __future__ import annotations

import numpy as np


class LLMUnavailable(RuntimeError):
    """Raised when a backend cannot run here (weights not downloaded, no API key, ...)."""


class AudioLLM:
    name: str = "base"

    def ask(self, audio: np.ndarray, sr: int, prompt: str) -> str:
        raise NotImplementedError
