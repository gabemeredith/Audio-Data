import numpy as np

from audio_probes.audio import band_level_dbfs
from audio_probes.config import load_config
from audio_probes.properties.highband import HighBand


def _clips(n=12, **over):
    cfg = load_config("highband", [f"data.{k}={v}" for k, v in over.items()])
    return HighBand(cfg).generate(n, seed=0)


def test_label_is_the_measured_band_level_at_48k():
    for clip in _clips():
        assert abs(clip.meta["measured_band_level_48k_dbfs"] - clip.label) < 0.5


def test_output_is_16k_five_seconds():
    for clip in _clips(3):
        assert clip.sr == 16000 and len(clip.audio) == 80000 and clip.audio.dtype == np.float32


def test_resampler_attenuates_the_band_by_at_least_40_db():
    """Same base, loud (-10 dBFS) vs quiet (-60 dBFS) band. After resampling, the difference
    between the two 16 kHz signals is what leaked. Real resamplers are not perfect brick walls:
    ffmpeg's default (what openai-whisper uses) lets about -46 dB through, so we assert 40 dB
    of attenuation and record the measured value in DECISIONS.md rather than pretending it is zero."""
    from audio_probes.audio import band_limited_noise, resample_like_whisper, set_rms_dbfs

    rng = np.random.default_rng(0)
    n = 48000 * 2
    base = set_rms_dbfs(band_limited_noise(rng, n, 48000, 50, 7000), -25)
    band = band_limited_noise(rng, n, 48000, 9000, 15000)
    quiet = resample_like_whisper(base + set_rms_dbfs(band, -60), 48000)
    loud = resample_like_whisper(base + set_rms_dbfs(band, -10), 48000)
    diff_db = band_level_dbfs(loud - quiet, 16000, 0, 8000)
    assert diff_db < -10 - 40, f"band leaked through the resampler at {diff_db:.1f} dBFS"
