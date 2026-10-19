import numpy as np
from scipy.stats import spearmanr

from audio_probes.config import load_config
from audio_probes.properties.distance import (Distance, direct_index, direct_to_reverb_db, place_pair,
                                              simulate_rir)

SR = 16000


def test_drr_decreases_with_distance_in_one_room():
    """Generator sanity: in one room, moving the source further away lowers the DRR.
    A single position can buck the trend (a wall right behind the source adds an early
    reflection), so average over 12 random mic/source placements per distance."""
    dims = [16.0, 12.0, 4.0]
    rng = np.random.default_rng(0)
    distances = [0.5, 1, 2, 4, 6, 9]
    mean_drr = []
    for d in distances:
        values = []
        for _ in range(12):
            mic, src = place_pair(rng, dims, d, 0.5)
            values.append(direct_to_reverb_db(simulate_rir(dims, 0.3, mic, src, SR, 10), d, SR))
        mean_drr.append(np.mean(values))
    assert all(a > b for a, b in zip(mean_drr, mean_drr[1:])), mean_drr


def test_drr_decreases_with_distance_across_random_rooms():
    cfg = load_config("distance", ["data.source_types=[noise]", "data.max_order=8", "data.workers=1"])
    clips = Distance(cfg).generate(40, seed=3)
    rho = spearmanr([c.label for c in clips], [c.meta["drr_db"] for c in clips]).statistic
    assert rho < -0.6, rho


def test_direct_path_arrives_where_expected():
    dims, mic = [6.0, 5.0, 3.0], np.array([1.0, 1.0, 1.5])
    rir = simulate_rir(dims, 0.5, mic, mic + [3.0, 0, 0], SR, 5)
    assert abs(int(np.argmax(np.abs(rir))) - direct_index(3.0, SR)) <= 1


def test_loudness_falls_with_distance_in_raw_audio():
    cfg = load_config("distance", ["data.source_types=[noise]", "data.max_order=8", "data.workers=1"])
    clips = Distance(cfg).generate(30, seed=4)
    rms = [np.sqrt(np.mean(c.audio.astype(float) ** 2)) for c in clips]
    assert spearmanr([c.label for c in clips], rms).statistic < -0.6


def test_placement_respects_distance_and_walls():
    rng = np.random.default_rng(0)
    for d in [0.5, 3.0, 9.0]:
        mic, src = place_pair(rng, [20.0, 15.0, 4.0], d, 0.3)
        assert abs(np.linalg.norm(src - mic) - d) < 1e-9
        assert np.all(src >= 0.3) and np.all(src <= np.array([20, 15, 4]) - 0.3)


def test_labels_ids_and_no_clipping():
    cfg = load_config("distance", ["data.source_types=[noise]", "data.max_order=6", "data.workers=1"])
    clips = Distance(cfg).generate(20, seed=0)
    for c in clips:
        assert 0.5 <= c.label <= 10 and c.audio.shape == (80000,)
        assert not c.meta["clipped"]
        assert c.meta["room_id"].startswith("b") and c.meta["source_id"].startswith("noise_b")
