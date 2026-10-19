"""A fake audio-LLM for tests and dry runs. It never listens: it answers from a hash of the
audio, sometimes with junk, so the parsing, logging and scoring code can be exercised
without a 16 GB download. Its scores should sit at chance."""

from __future__ import annotations

import hashlib

import numpy as np

from audio_probes.llms.base import AudioLLM


class MockLLM(AudioLLM):
    name = "mock"

    def ask(self, audio: np.ndarray, sr: int, prompt: str) -> str:
        digest = hashlib.sha256(audio.tobytes() + prompt.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        if rng.random() < 0.1:
            return "I cannot determine that from the audio."
        if "letter" in prompt:
            return f"({'ABCD'[int(rng.integers(4))]})"
        return f"About {rng.uniform(0.5, 10):.1f}"
