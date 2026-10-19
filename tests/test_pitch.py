import numpy as np

from audio_probes.audio import log_mel
from audio_probes.config import load_config
from audio_probes.properties.pitch import Pitch, autocorr_f0


def _clips(n=20, seed=0):
    return Pitch(load_config("pitch")).generate(n, seed)


def test_shape_rate_and_label_range():
    for clip in _clips():
        assert clip.sr == 16000 and clip.audio.shape == (80000,) and clip.audio.dtype == np.float32
        assert 100 <= clip.label <= 1000
        assert np.abs(clip.audio).max() < 1.0


def test_energy_sits_on_harmonics_of_the_label():
    """Almost all spectral energy lies within 3 Hz of an integer multiple of the labelled f0
    (not 100%: the fades and the noise floor smear a little energy between harmonics)."""
    for clip in _clips():
        power = np.abs(np.fft.rfft(clip.audio)) ** 2
        freqs = np.fft.rfftfreq(len(clip.audio), 1 / clip.sr)
        k = np.round(freqs / clip.label)
        near = (k >= 1) & (np.abs(freqs - k * clip.label) < 3.0)
        assert power[near].sum() / power.sum() > 0.97


def test_autocorrelation_baseline_recovers_f0():
    for clip in _clips():
        est = autocorr_f0(clip.audio, clip.sr)
        assert abs(est - clip.label) / clip.label < 0.02, (est, clip.label)


def test_same_seed_is_deterministic_and_timbres_differ():
    a, b = _clips(5, seed=1), _clips(5, seed=1)
    assert all(np.array_equal(x.audio, y.audio) for x, y in zip(a, b))
    c = _clips(5, seed=2)
    assert not np.allclose(a[0].audio, c[0].audio)


def test_no_energy_above_max_harmonic():
    clip = _clips(1)[0]
    spectrum = np.abs(np.fft.rfft(clip.audio)) ** 2
    freqs = np.fft.rfftfreq(len(clip.audio), 1 / clip.sr)
    assert spectrum[freqs > 7600].sum() / spectrum.sum() < 1e-4


def test_log_mel_matches_whisper_feature_extractor():
    from transformers import WhisperFeatureExtractor
    clip = _clips(1)[0]
    ours = log_mel(clip.audio)
    fe = WhisperFeatureExtractor(feature_size=80, sampling_rate=16000)
    theirs = fe(clip.audio, sampling_rate=16000, return_tensors="np").input_features[0][:, : ours.shape[1]]
    # the last frame differs only because Whisper pads with zeros to 30 s and we reflect-pad
    assert np.abs(ours[:, :-1] - theirs[:, :-1]).max() < 1e-3
