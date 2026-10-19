from audio_probes.config import config_hash, deep_merge, load_config, parse_override


def test_parse_override_types():
    assert parse_override("probe.alpha=10") == {"probe": {"alpha": 10}}
    assert parse_override("encoder.layers=[0, 6]") == {"encoder": {"layers": [0, 6]}}
    assert parse_override("artifacts.spectrograms=false") == {"artifacts": {"spectrograms": False}}
    assert parse_override("encoder.size=tiny") == {"encoder": {"size": "tiny"}}


def test_cli_overrides_win_over_yaml():
    cfg = load_config("distance", ["probe.alpha=10", "data.n_clips=50"])
    assert cfg["probe"]["alpha"] == 10
    assert cfg["data"]["n_clips"] == 50
    assert cfg["probe"]["label_transform"] == "log"  # from distance.yaml
    assert cfg["encoder"]["size"] == "small"  # from defaults.yaml


def test_deep_merge_keeps_siblings():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}


def test_hash_ignores_key_order_and_changes_with_values():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
