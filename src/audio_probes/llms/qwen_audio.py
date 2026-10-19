"""Qwen2-Audio-7B-Instruct via Hugging Face transformers (runs locally on CUDA, Apple MPS or CPU).

The weights are about 16 GB. With llm.allow_download: false (the default) this backend only
runs if the weights are already in the Hugging Face cache; otherwise it raises LLMUnavailable
and the llm stage is skipped. See DECISIONS.md.
"""

from __future__ import annotations

import numpy as np

from audio_probes.llms.base import AudioLLM, LLMUnavailable


def ensure_local(model_id: str, allow_download: bool) -> None:
    """Raise LLMUnavailable unless the model is cached (or we are allowed to fetch it)."""
    from huggingface_hub import snapshot_download

    if allow_download:
        return
    try:
        snapshot_download(model_id, local_files_only=True)
    except Exception as err:  # LocalEntryNotFoundError and friends
        raise LLMUnavailable(f"{model_id} is not downloaded (about 16 GB). Re-run with "
                             "llm.allow_download=true to fetch it.") from err


class QwenAudio(AudioLLM):
    name = "qwen2_audio"

    def __init__(self, model_id: str, device: str = "auto", temperature: float = 0.0,
                 max_new_tokens: int = 24, allow_download: bool = False):
        ensure_local(model_id, allow_download)
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        from audio_probes.encoders.whisper import pick_device

        self.device = pick_device(device)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.model_id, self.temperature, self.max_new_tokens = model_id, temperature, max_new_tokens

    def ask(self, audio: np.ndarray, sr: int, prompt: str) -> str:
        import torch

        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"},
                                                     {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        sampling = {"do_sample": True, "temperature": self.temperature} if self.temperature > 0 else {"do_sample": False}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **sampling)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
