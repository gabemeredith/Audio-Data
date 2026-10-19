"""The Property interface. A property is a plug-in: one file that knows how to
generate labelled clips, how to group them for the split, and how to ask an LLM about them."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np

from audio_probes.clip import Clip

CHOICE_LETTERS = "ABCD"


class Property:
    name: str = "base"
    label_kind: Literal["regression", "classification"] = "regression"
    label_unit: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_cfg = cfg["data"]

    # ---- required ------------------------------------------------------------------
    def generate(self, n: int, seed: int) -> list[Clip]:
        raise NotImplementedError

    # ---- sensible defaults ---------------------------------------------------------
    def split_key(self, clip: Clip) -> str:
        """Group key for the train/test split. Clips that share any '|'-separated part
        (a room or a source) always land on the same side."""
        return f"room:{clip.meta['room_id']}|source:{clip.meta['source_id']}"

    def llm_prompts(self) -> dict[str, str]:
        return dict(self.cfg["llm"]["prompts"])

    def parse_llm(self, prompt_kind: str, text: str) -> float | int | None:
        """numeric -> first number in the answer; choice -> bin index 0..3 from a letter."""
        if prompt_kind == "numeric":
            return parse_number(text)
        if prompt_kind == "choice":
            return parse_choice(text)
        raise ValueError(f"unknown prompt kind {prompt_kind!r}")

    def baseline_features(self, clip: Clip) -> np.ndarray:
        """Hand-crafted features fed to the same probe as a reference line."""
        return np.array([rms_dbfs(clip.audio)])

    def baseline_description(self) -> str:
        return "RMS level (dBFS)"


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS level in dB relative to full scale: 20 * log10(sqrt(mean(x^2)))."""
    return float(20 * np.log10(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + 1e-12))


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """First number in the text, e.g. 'About 3.5 meters' -> 3.5. Ranges '2-3' -> midpoint."""
    if text is None:
        return None
    cleaned = text.replace(",", "")
    rng = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)", cleaned)
    if rng:
        return (float(rng.group(1)) + float(rng.group(2))) / 2
    match = _NUMBER.search(cleaned)
    return float(match.group()) if match else None


def parse_choice(text: str) -> int | None:
    """Letter answer A-D -> 0-3. Accepts 'B', '(B)', 'B)', 'Answer: B'. Anything else -> None."""
    if text is None:
        return None
    stripped = text.strip()
    match = re.match(r"^\(?([ABCD])\b", stripped)
    if not match:
        match = re.search(r"(?:answer(?: is)?:?\s*|\()([ABCD])\b", stripped, flags=re.IGNORECASE)
    return CHOICE_LETTERS.index(match.group(1).upper()) if match else None
