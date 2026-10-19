"""AudioLLM backends. Adding one = write a module with an AudioLLM subclass and add it here."""

from __future__ import annotations

from audio_probes.llms.base import AudioLLM, LLMUnavailable


def build_llm(cfg: dict) -> AudioLLM:
    llm = cfg["llm"]
    if llm["backend"] == "qwen2_audio":
        from audio_probes.llms.qwen_audio import QwenAudio
        return QwenAudio(llm["model"], llm["device"], llm["temperature"], llm["max_new_tokens"], llm["allow_download"])
    if llm["backend"] == "mock":
        from audio_probes.llms.mock import MockLLM
        return MockLLM()
    raise ValueError(f"unknown llm backend {llm['backend']!r}")


__all__ = ["AudioLLM", "LLMUnavailable", "build_llm"]
