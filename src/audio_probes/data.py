"""Stage 1: generate clips and write per-clip artifacts.

Output, under data/clips/<property>/<data_hash>/:
    wav/<id>.wav        32-bit float audio
    spec/<id>.png       log-mel spectrogram
    meta.jsonl          one line per clip: id, label, rms_dbfs, and the property's metadata
    config.yaml         the resolved config that produced it
    DONE                marker; if present the stage is skipped
"""

from __future__ import annotations

import json
import time

import numpy as np

from audio_probes.audio import read_wav, set_rms_dbfs, spectrogram_png, write_wav
from audio_probes.clip import Clip
from audio_probes.config import clips_dir, save_resolved_config
from audio_probes.properties import get_property
from audio_probes.properties.base import rms_dbfs


def run(cfg: dict) -> None:
    out = clips_dir(cfg)
    if (out / "DONE").exists():
        print(f"[data] {cfg['property']}: cached at {out}")
        return
    prop = get_property(cfg)
    t0 = time.time()
    clips = prop.generate(cfg["data"]["n_clips"], cfg["seed"])
    print(f"[data] {cfg['property']}: generated {len(clips)} clips in {time.time() - t0:.0f}s")
    write_clips(clips, out, spectrograms=cfg["artifacts"]["spectrograms"])
    save_resolved_config(cfg, out / "config.yaml")
    (out / "DONE").write_text("ok\n")
    print(f"[data] {cfg['property']}: wrote {out}")


def write_clips(clips: list[Clip], out, spectrograms: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "meta.jsonl", "w") as f:
        for clip in clips:
            write_wav(out / "wav" / f"{clip.id}.wav", clip.audio, clip.sr)
            if spectrograms:
                spectrogram_png(clip.audio, clip.sr, out / "spec" / f"{clip.id}.png")
            record = {"id": clip.id, "label": clip.label, "rms_dbfs": rms_dbfs(clip.audio), **clip.meta}
            f.write(json.dumps(record, default=_to_json) + "\n")


def _to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def load_meta(cfg: dict) -> list[dict]:
    path = clips_dir(cfg) / "meta.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No clips for {cfg['property']}; run `make data PROPERTY={cfg['property']}` first")
    return [json.loads(line) for line in path.read_text().splitlines()]


def apply_condition(audio: np.ndarray, condition: str, cfg: dict) -> np.ndarray:
    """raw: unchanged. loudness_normalized: RMS-normalise to data.normalize_dbfs."""
    if condition == "raw":
        return audio
    if condition == "loudness_normalized":
        return set_rms_dbfs(audio, cfg["data"]["normalize_dbfs"])
    raise ValueError(f"unknown condition {condition!r}")


def load_clip(cfg: dict, record: dict, condition: str = "raw") -> Clip:
    audio, sr = read_wav(clips_dir(cfg) / "wav" / f"{record['id']}.wav")
    meta = {k: v for k, v in record.items() if k not in ("id", "label")}
    return Clip(record["id"], apply_condition(audio, condition, cfg), sr, record["label"], meta)


def load_clips(cfg: dict, condition: str = "raw") -> list[Clip]:
    return [load_clip(cfg, r, condition) for r in load_meta(cfg)]
