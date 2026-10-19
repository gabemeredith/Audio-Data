import numpy as np
import pytest

from audio_probes.encoders.whisper import n_real_frames, pool


def test_real_frame_count():
    assert n_real_frames(80000, 16000) == 250  # 5 s
    assert n_real_frames(16000, 16000) == 50  # 1 s
    assert n_real_frames(16001, 16000) == 51  # a partial frame still counts
    assert n_real_frames(16000 * 40, 16000) == 1500  # never beyond Whisper's 30 s window


def _hidden_with_padding(n_real=250, dim=8):
    """Real frames hold 1.0, padding frames hold a huge value that would wreck the mean."""
    hidden = np.full((1500, dim), 1000.0, dtype=np.float32)
    hidden[:n_real] = 1.0
    hidden[:n_real, 0] = np.arange(n_real)  # something non-constant to check the mean/max
    return hidden


@pytest.mark.parametrize("method", ["mean", "max", "first_n"])
def test_pooling_ignores_padding(method):
    pooled = pool(_hidden_with_padding(), 250, method, first_n=50)
    assert np.all(pooled < 1000), "padding leaked into the pooled vector"


def test_pooling_values():
    hidden = _hidden_with_padding()
    assert pool(hidden, 250, "mean")[0] == pytest.approx(np.arange(250).mean())
    assert pool(hidden, 250, "max")[0] == 249
    assert pool(hidden, 250, "first_n", first_n=50)[0] == pytest.approx(np.arange(50).mean())
    assert pool(hidden, 250, "mean")[1] == 1.0


def test_whisper_hidden_states_trim_matches_manual():
    """End to end on whisper-tiny: our pooled layer equals a manual mean over the first 250 frames,
    and differs from a mean over all 1500 frames (so the trimming matters)."""
    torch = pytest.importorskip("torch")
    from transformers import WhisperFeatureExtractor, WhisperModel

    from audio_probes.encoders.whisper import WhisperEncoder

    try:
        enc = WhisperEncoder(size="tiny", device="cpu")
    except OSError:
        pytest.skip("whisper-tiny weights not available offline")
    audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(80000) / 16000)).astype(np.float32)
    ours = enc.layers_batch([audio], 16000)
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    model = WhisperModel.from_pretrained("openai/whisper-tiny").encoder.eval()
    with torch.no_grad():
        hs = model(fe(audio, sampling_rate=16000, return_tensors="pt").input_features,
                   output_hidden_states=True).hidden_states
    last = hs[-1][0].numpy()
    assert len(ours) == len(hs)
    np.testing.assert_allclose(ours[-1][0], last[:250].mean(0), rtol=1e-4, atol=1e-4)
    assert not np.allclose(ours[-1][0], last.mean(0), atol=1e-3)
