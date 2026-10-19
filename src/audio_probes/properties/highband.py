"""Negative control: level of a 9-15 kHz noise band that Whisper's 16 kHz input cannot contain.

Each clip is built at 48 kHz from two independent parts:
    base  = a harmonic tone + noise, all below 7 kHz, at a random level
    band  = white noise restricted to 9-15 kHz, at a random level (this level is the label)
Then the mix is resampled to 16 kHz exactly as Whisper's loader does. Resampling to 16 kHz
keeps only content below 8 kHz, so the band is gone before the encoder sees anything.
If a probe can still predict the label, information leaked (through the resampler, or a bug).
"""

from __future__ import annotations

import numpy as np

from audio_probes.audio import (band_level_dbfs, band_limited_noise, db_to_amp, fade, resample_like_whisper,
                                set_rms_dbfs)
from audio_probes.clip import Clip
from audio_probes.properties.base import Property
from audio_probes.properties.pitch import harmonic_tone, random_timbre


class HighBand(Property):
    name = "highband"
    label_unit = "dBFS"

    def generate(self, n: int, seed: int) -> list[Clip]:
        d = self.data_cfg
        src_sr, n_src = d["source_sr"], int(d["source_sr"] * d["duration_s"])
        lo, hi = d["band_hz"]
        clips = []
        for i in range(n):
            rng = np.random.default_rng([seed, i])
            base = self._base(rng, src_sr, n_src)
            band_level = float(rng.uniform(*d["band_level_dbfs_range"]))
            band = set_rms_dbfs(band_limited_noise(rng, n_src, src_sr, lo, hi), band_level)
            mix48 = fade(base + band, src_sr)
            audio = resample_like_whisper(mix48, src_sr, d["resampler"])[: int(d["sr"] * d["duration_s"])]
            audio = audio + db_to_amp(d["noise_floor_dbfs"]) * rng.standard_normal(len(audio)).astype(np.float32)
            clip_id = f"highband_{i:05d}"
            clips.append(Clip(clip_id, audio.astype(np.float32), d["sr"], band_level, {
                "room_id": "none", "source_id": clip_id,
                "band_level_dbfs": band_level,
                "measured_band_level_48k_dbfs": band_level_dbfs(mix48, src_sr, lo, hi),
                "residual_7k5_8k_dbfs_16k": band_level_dbfs(audio, d["sr"], 7500, 8000),
            }))
        return clips

    def _base(self, rng, sr, n) -> np.ndarray:
        """Low-band content: a random harmonic tone plus noise, both below base_max_hz."""
        d = self.data_cfg
        f0 = float(np.exp(rng.uniform(np.log(100), np.log(1000))))
        weights, phases = random_timbre(rng, f0, d["base_max_hz"], [0.0, 2.0])
        tone = harmonic_tone(f0, weights, phases, sr, n)
        noise = band_limited_noise(rng, n, sr, 50.0, d["base_max_hz"])
        mix = set_rms_dbfs(tone, 0) + set_rms_dbfs(noise, float(rng.uniform(-20, -5)))
        return set_rms_dbfs(mix, float(rng.uniform(*d["base_level_dbfs_range"])))

    def split_key(self, clip: Clip) -> str:
        return f"source:{clip.meta['source_id']}"

    def baseline_features(self, clip: Clip) -> np.ndarray:
        """Overall level, plus the level of the top of the 16 kHz band (7.5-8 kHz), where any
        resampler leakage would show up."""
        return np.array([band_level_dbfs(clip.audio, clip.sr, 0, 8000),
                         band_level_dbfs(clip.audio, clip.sr, 7500, 8000)])

    def baseline_description(self) -> str:
        return "overall level + 7.5-8 kHz level"
