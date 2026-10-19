import numpy as np

from audio_probes.split import connected_components, grouped_split


def _tags(keys, idx, prefix):
    return {part for i in idx for part in keys[i].split("|") if part.startswith(prefix)}


def test_no_room_or_source_on_both_sides():
    rng = np.random.default_rng(0)
    # 10 blocks; inside a block, rooms and sources mix freely (like the distance generator)
    keys = []
    for i in range(1000):
        block = i % 10
        keys.append(f"room:b{block}r{rng.integers(20)}|source:b{block}s{rng.integers(4)}")
    train, test, _ = grouped_split(keys, 0.2, seed=0)
    assert len(set(train) & set(test)) == 0 and len(train) + len(test) == 1000
    assert not _tags(keys, train, "room:") & _tags(keys, test, "room:")
    assert not _tags(keys, train, "source:") & _tags(keys, test, "source:")
    assert 0.1 < len(test) / 1000 < 0.35


def test_shared_tag_joins_components():
    comp = connected_components(["room:a|source:x", "room:b|source:x", "room:c|source:y"])
    assert comp[0] == comp[1] != comp[2]


def test_single_tag_keys_split_randomly_by_clip():
    keys = [f"source:s{i}" for i in range(100)]
    train, test, _ = grouped_split(keys, 0.2, seed=1)
    assert len(test) == 20


def test_real_property_splits_are_clean():
    """The actual split of every property's metadata keeps rooms and sources apart."""
    from audio_probes.clip import Clip
    from audio_probes.config import load_config
    from audio_probes.properties import get_property

    for name in ["pitch"]:
        cfg = load_config(name, ["data.n_clips=60", "data.source_types=[noise]", "data.max_order=3"])
        prop = get_property(cfg)
        clips = prop.generate(60, seed=0)
        keys = [prop.split_key(c) for c in clips]
        train, test, _ = grouped_split(keys, 0.2, seed=0)
        for prefix in ("room:", "source:"):
            both = _tags(keys, train, prefix) & _tags(keys, test, prefix) - {"room:none"}
            assert not both, (name, both)
