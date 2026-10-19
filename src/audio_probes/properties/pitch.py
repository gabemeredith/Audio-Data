"""Positive control: fundamental frequency (f0) of a harmonic tone.

Each clip is a sum of harmonics k * f0 with random weights and phases, so two clips with
the same f0 still have different waveforms (timbres). A probe must read pitch, not a template.
"""

from __future__ import annotations

import numpy as np

from audio_probes.audio import db_to_amp, fade, set_rms_dbfs
from audio_probes.clip import Clip
from audio_probes.properties.base import Property


def harmonic_tone(f0: float, weights: np.ndarray, phases: np.ndarray, sr: int, n: int) -> np.ndarray:
    """x(t) = sum_k w_k * sin(2*pi*k*f0*t + phi_k), for k = 1..K."""
    t = np.arange(n) / sr
    k = np.arange(1, len(weights) + 1)[:, None]
    return np.sum(weights[:, None] * np.sin(2 * np.pi * k * f0 * t + phases[:, None]), axis=0)


def random_timbre(rng: np.random.Generator, f0: float, max_hz: float, decay_range) -> tuple[np.ndarray, np.ndarray]:
    """Random harmonic weights w_k = u_k * k^-decay (u_k ~ U(0,1)), with the fundamental
    kept audible (w_1 >= 0.3) so the pitch is not ambiguous. Returns (weights, phases)."""
    n_harm = max(1, int(max_hz // f0))
    k = np.arange(1, n_harm + 1)
    decay = rng.uniform(*decay_range)
    weights = rng.uniform(0, 1, n_harm) * k ** (-decay)
    weights[0] = max(weights[0], 0.3 * weights.max())
    phases = rng.uniform(0, 2 * np.pi, n_harm)
    return weights, phases


class Pitch(Property):
    name = "pitch"
    label_unit = "Hz"

    def generate(self, n: int, seed: int) -> list[Clip]:
        d = self.data_cfg
        sr, n_samples = d["sr"], int(d["sr"] * d["duration_s"])
        lo, hi = np.log(d["f0_range_hz"][0]), np.log(d["f0_range_hz"][1])
        clips = []
        for i in range(n):
            rng = np.random.default_rng([seed, i])
            f0 = float(np.exp(rng.uniform(lo, hi)))
            weights, phases = random_timbre(rng, f0, d["max_harmonic_hz"], d["harmonic_decay_range"])
            level = float(rng.uniform(*d["level_dbfs_range"]))
            audio = set_rms_dbfs(harmonic_tone(f0, weights, phases, sr, n_samples), level)
            audio = audio + db_to_amp(d["noise_floor_dbfs"]) * rng.standard_normal(n_samples).astype(np.float32)
            clip_id = f"pitch_{i:05d}"
            clips.append(Clip(clip_id, fade(audio, sr), sr, f0, {
                "room_id": "none", "source_id": clip_id, "f0_hz": f0, "n_harmonics": len(weights),
                "level_dbfs": level,
            }))
        return clips

    def split_key(self, clip: Clip) -> str:
        # No rooms here; every tone has its own random timbre, so each clip is its own group.
        return f"source:{clip.meta['source_id']}"

    def baseline_features(self, clip: Clip) -> np.ndarray:
        """log of an autocorrelation pitch estimate: a classic hand-made pitch tracker."""
        return np.array([np.log(autocorr_f0(clip.audio, clip.sr))])

    def baseline_description(self) -> str:
        return "autocorrelation pitch estimate (log Hz)"


def autocorr_f0(audio: np.ndarray, sr: int, fmin: float = 80.0, fmax: float = 1100.0) -> float:
    """f0 = sr / lag*, where r(lag) = sum_n x[n] x[n+lag] and lag* is the SHORTEST lag in
    [sr/fmax, sr/fmin] whose r is within 10% of the best. (A periodic signal also peaks at
    2x, 3x... the period; taking the shortest avoids those octave errors.)
    The signal is low-passed at 1.5 kHz first so the autocorrelation peaks are broad enough
    to sample at integer lags; the lag is then refined with a parabola through its neighbours."""
    x = audio[: sr].astype(np.float64)  # first second is plenty for a stationary tone
    spectrum = np.fft.rfft(x, 2 * len(x))
    spectrum[np.fft.rfftfreq(2 * len(x), 1 / sr) > 1500.0] = 0
    r = np.fft.irfft(np.abs(spectrum) ** 2)[: len(x)]
    lo, hi = int(sr / fmax), int(sr / fmin)
    window = r[lo:hi]
    is_peak = (window[1:-1] > window[:-2]) & (window[1:-1] >= window[2:])
    peaks = np.flatnonzero(is_peak) + 1
    if len(peaks) == 0:
        return sr / (lo + int(np.argmax(window)))
    good = peaks[window[peaks] >= 0.9 * window[peaks].max()]
    i = lo + int(good[0])
    a, b, c = r[i - 1], r[i], r[i + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return sr / (i + shift)
