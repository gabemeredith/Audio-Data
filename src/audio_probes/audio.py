"""Small numpy DSP helpers shared by the properties, the data stage and the inspect view."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Whisper's front end: 25 ms Hann window, 10 ms hop, 80 mel bands, 16 kHz audio.
WHISPER_SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80


def db_to_amp(db: float) -> float:
    """Amplitude ratio for a level in dB: 10^(dB / 20)."""
    return float(10 ** (db / 20))


def rms(audio: np.ndarray) -> float:
    """Root mean square: sqrt(mean(x^2))."""
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def set_rms_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale x so that 20*log10(rms(x)) = target_dbfs."""
    current = rms(audio)
    if current == 0:
        return audio.astype(np.float32)
    return (audio * (db_to_amp(target_dbfs) / current)).astype(np.float32)


def fade(audio: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    """Linear fade in and out, so clips do not start or end with a click."""
    n = min(int(sr * ms / 1000), len(audio) // 2)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = audio.astype(np.float32).copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def band_limited_noise(rng: np.random.Generator, n: int, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """White noise with every FFT bin outside [lo_hz, hi_hz] set to zero (a brick-wall band)."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spectrum[(freqs < lo_hz) | (freqs > hi_hz)] = 0
    return np.fft.irfft(spectrum, n).astype(np.float32)


def band_level_dbfs(audio: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    """RMS level (dBFS) of the part of x inside [lo_hz, hi_hz], via Parseval on the FFT:
    mean(x_band^2) = (sum over band bins of |X_k|^2, counted twice for k != 0) / n^2."""
    n = len(audio)
    spectrum = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    power = 2 * np.sum(np.abs(spectrum[mask]) ** 2) / n**2
    return float(10 * np.log10(power + 1e-24))


def resample_like_whisper(audio: np.ndarray, sr: int, method: str = "ffmpeg") -> np.ndarray:
    """Resample to 16 kHz the way openai-whisper's load_audio does (ffmpeg -ar 16000).

    `method="scipy"` uses a polyphase filter (resample_poly) instead, for machines without ffmpeg.
    """
    if sr == WHISPER_SR:
        return audio.astype(np.float32)
    if method == "ffmpeg":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found; set data.resampler=scipy")
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1",
               "-i", "pipe:0", "-f", "f32le", "-ar", str(WHISPER_SR), "-ac", "1", "pipe:1"]
        out = subprocess.run(cmd, input=audio.astype(np.float32).tobytes(), capture_output=True, check=True)
        return np.frombuffer(out.stdout, dtype=np.float32).copy()
    if method == "scipy":
        g = np.gcd(sr, WHISPER_SR)
        return resample_poly(audio, WHISPER_SR // g, sr // g).astype(np.float32)
    raise ValueError(f"unknown resampler {method!r}")


# ---- log-mel spectrogram (same recipe as Whisper's feature extractor) ---------------

def _hz_to_mel(hz):
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    hz = np.asarray(hz, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(hz >= min_log_hz, min_log_mel + np.log(np.maximum(hz, 1e-9) / min_log_hz) / logstep, hz / f_sp)


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(mel >= min_log_mel, min_log_hz * np.exp(logstep * (mel - min_log_mel)), mel * f_sp)


@lru_cache(maxsize=4)
def mel_filterbank(sr: int = WHISPER_SR, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Triangular filters, evenly spaced on the mel scale from 0 Hz to sr/2, area-normalised
    (Slaney). Shape (n_mels, n_fft//2 + 1). Row m sums the power spectrum into mel band m."""
    fft_freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    mel_points = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    lower, center, upper = mel_points[:-2, None], mel_points[1:-1, None], mel_points[2:, None]
    rising = (fft_freqs - lower) / (center - lower)
    falling = (upper - fft_freqs) / (upper - center)
    weights = np.maximum(0, np.minimum(rising, falling))
    weights *= (2.0 / (upper - lower))  # equal area per filter
    return weights


def log_mel(audio: np.ndarray, sr: int = WHISPER_SR) -> np.ndarray:
    """Whisper-style log-mel spectrogram, shape (80, n_frames).

    1. STFT: X[t, k] = sum_n x[n + t*hop] * w[n] * exp(-2*pi*i*k*n / N), Hann window w, N = 400.
    2. Power: P = |X|^2 (phase is discarded).
    3. Mel: M = F @ P with the triangular filterbank F (fine frequency detail is discarded).
    4. Log: L = log10(max(M, 1e-10)), floor at max(L) - 8 (anything 80 dB below the peak
       is discarded), then (L + 4) / 4 to land roughly in [-1, 1].
    """
    x = np.pad(audio.astype(np.float64), N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    window = np.hanning(N_FFT + 1)[:-1]  # periodic Hann, as in torch.hann_window
    power = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2
    mel = mel_filterbank(sr) @ power[:-1].T  # Whisper drops the last frame
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def spectrogram_png(audio: np.ndarray, sr: int, path: Path) -> None:
    """Save the log-mel spectrogram as a PNG (low frequencies at the bottom)."""
    from matplotlib import colormaps
    from PIL import Image

    spec = log_mel(audio, sr)
    lo, hi = spec.min(), spec.max()
    norm = (spec - lo) / (hi - lo + 1e-9)
    rgb = (colormaps["magma"](norm[::-1])[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((spec.shape[1], spec.shape[0] * 3), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """32-bit float WAV, so very quiet signals (e.g. leakage checks) are not lost to 16-bit rounding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sr, subtype="FLOAT")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    return audio, sr
