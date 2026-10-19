"""`predict` and `play`: try the trained probes on any sound file on your Mac, no microphone needed.

    make predict PROPERTY=pitch FILE=/System/Library/Sounds/Glass.aiff
    make play PROPERTY=pitch CLIP=pitch_00042

`predict` decodes the file with ffmpeg (so .aiff, .wav, .m4a, .mp3 all work), resamples it to
16 kHz the same way the pipeline does, keeps the first `data.duration_s` seconds, applies the
condition, runs Whisper, and prints what each layer's probe thinks the label is.
Sounds that ship with macOS live in /System/Library/Sounds, and `say -o out.aiff "hello"`
makes speech from text.
"""

from __future__ import annotations

import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from audio_probes.audio import WHISPER_SR, write_wav
from audio_probes.config import clips_dir, probe_dir
from audio_probes.data import apply_condition, load_meta
from audio_probes.probes import inverse_label


def decode_file(path: str | Path, sr: int = WHISPER_SR) -> np.ndarray:
    """Any audio file -> mono float32 at `sr`, via ffmpeg (as openai-whisper's load_audio does)."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
           "-ar", str(sr), "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def predict_file(cfg: dict, condition: str, path: str) -> list[dict]:
    from audio_probes.encoders.whisper import build_encoder

    models_path = probe_dir(cfg, condition) / "models.pkl"
    if not models_path.exists():
        raise FileNotFoundError(f"No trained probes; run `make probe PROPERTY={cfg['property']}` first")
    with open(models_path, "rb") as f:
        models = pickle.load(f)
    audio = decode_file(path)[: int(cfg["data"]["duration_s"] * WHISPER_SR)]
    audio = apply_condition(audio, condition, cfg)
    pooled = build_encoder(cfg).layers_batch([audio], WHISPER_SR)
    layer_ids = sorted(models)
    rows = []
    for layer_id, vec in zip(layer_ids, pooled):
        reg, clf = models[layer_id]
        z = reg.predict(vec)
        rows.append({"layer": layer_id, "prediction": float(inverse_label(z, cfg["probe"]["label_transform"])[0]),
                     "bin": "ABCD"[int(clf.predict(vec)[0])]})
    return rows


def run_predict(cfg: dict, condition: str, path: str) -> None:
    from audio_probes.properties import get_property

    rows = predict_file(cfg, condition, path)
    unit = get_property(cfg).label_unit
    edges = cfg["probe"]["bin_edges"]
    print(f"\n{cfg['property']} probes ({condition}) on {path}")
    print(f"bins: A < {edges[0]} <= B < {edges[1]} <= C < {edges[2]} <= D  ({unit})")
    print(f"{'layer':>5}  {'prediction':>12}  bin")
    for r in rows:
        print(f"{r['layer']:>5}  {r['prediction']:>10.4g} {unit:<3} {r['bin']}")
    print("Note: the probes were trained on synthetic clips, so real-world sounds are out of distribution.")


def play_clip(cfg: dict, condition: str, clip_id: str, boost_db: float = 0.0) -> None:
    """Play one generated clip through the Mac's speakers or headphones with afplay."""
    if shutil.which("afplay") is None:
        raise RuntimeError("afplay not found (it ships with macOS)")
    if clip_id not in {r["id"] for r in load_meta(cfg)}:
        raise KeyError(f"No clip {clip_id!r} for {cfg['property']}")
    from audio_probes.audio import read_wav

    audio, sr = read_wav(clips_dir(cfg) / "wav" / f"{clip_id}.wav")
    audio = apply_condition(audio, condition, cfg) * 10 ** (boost_db / 20)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / f"{clip_id}.wav"
        write_wav(wav, np.clip(audio, -1, 1), sr)
        print(f"[play] {clip_id} ({condition}, boost {boost_db:+.0f} dB)")
        subprocess.run(["afplay", str(wav)], check=True)
