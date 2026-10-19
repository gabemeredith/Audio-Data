"""Whisper encoder (Hugging Face transformers) -> one pooled vector per layer."""

from __future__ import annotations

import math

import numpy as np
import torch

from audio_probes.encoders.base import Encoder

ENCODER_FRAMES_PER_SECOND = 50  # 100 mel frames/s, then a stride-2 convolution
MAX_FRAMES = 1500  # Whisper always sees 30 s


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def n_real_frames(n_samples: int, sr: int) -> int:
    """Encoder frames that cover real audio: ceil(seconds * 50), capped at 1500 (30 s).
    A 5 s clip -> 250. Everything after that is padding (silence Whisper added)."""
    return min(MAX_FRAMES, math.ceil(n_samples / sr * ENCODER_FRAMES_PER_SECOND))


def pool(hidden: np.ndarray, n_frames: int, method: str = "mean", first_n: int = 50) -> np.ndarray:
    """Collapse one layer's (frames, dim) output to a single (dim,) vector, using ONLY the
    first n_frames rows (real audio), never the padding.

    mean:    v = (1/T) * sum_{t<T} h_t
    max:     v_d = max_{t<T} h_{t,d}
    first_n: mean over the first min(first_n, T) frames
    """
    real = hidden[:n_frames]
    if method == "mean":
        return real.mean(axis=0)
    if method == "max":
        return real.max(axis=0)
    if method == "first_n":
        return real[: min(first_n, n_frames)].mean(axis=0)
    raise ValueError(f"unknown pooling {method!r}")


class WhisperEncoder(Encoder):
    name = "whisper"

    def __init__(self, size: str = "small", layers="all", pooling: str = "mean",
                 first_n_frames: int = 50, device: str = "auto"):
        from transformers import WhisperFeatureExtractor, WhisperModel

        model_id = f"openai/whisper-{size}"
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        self.device = pick_device(device)
        self.model = WhisperModel.from_pretrained(model_id).encoder.to(self.device).eval()
        self.n_layers = self.model.config.encoder_layers + 1  # input embeddings + each block
        self.layer_ids = list(range(self.n_layers)) if layers == "all" else list(layers)
        self.pooling, self.first_n = pooling, first_n_frames

    @torch.no_grad()
    def layers_batch(self, audios: list[np.ndarray], sr: int) -> list[np.ndarray]:
        if sr != 16000:
            raise ValueError("Whisper expects 16 kHz audio")
        feats = self.feature_extractor(audios, sampling_rate=sr, return_tensors="pt").input_features
        out = self.model(feats.to(self.device), output_hidden_states=True)
        hidden = [h.float().cpu().numpy() for h in out.hidden_states]  # each (batch, 1500, dim)
        frames = [n_real_frames(len(a), sr) for a in audios]
        return [
            np.stack([pool(hidden[layer][b], frames[b], self.pooling, self.first_n) for b in range(len(audios))])
            for layer in self.layer_ids
        ]


def build_encoder(cfg: dict) -> Encoder:
    enc = cfg["encoder"]
    if enc["name"] != "whisper":
        raise ValueError(f"unknown encoder {enc['name']!r}")
    return WhisperEncoder(enc["size"], enc["layers"], enc["pooling"], enc["first_n_frames"], enc["device"])
