"""Grouped train/test split.

Clips that share a room OR a source must land on the same side, otherwise the probe can
score well by memorising a room's echo or a speaker's voice. Each clip's split key is a
'|'-separated list of group tags (e.g. 'room:r12|source:spk84'). Tags that co-occur are
merged into one connected component (union-find), and whole components are assigned to
train or test.
"""

from __future__ import annotations

import json

import numpy as np

from audio_probes.config import DATA_DIR, save_resolved_config, split_hash


def connected_components(keys: list[str]) -> np.ndarray:
    """Component id per clip: two clips are connected if their keys share any tag."""
    parent: dict[str, str] = {}

    def find(tag: str) -> str:
        while parent.setdefault(tag, tag) != tag:
            parent[tag] = parent[parent[tag]]  # path halving
            tag = parent[tag]
        return tag

    for key in keys:
        tags = key.split("|")
        for tag in tags[1:]:
            parent[find(tag)] = find(tags[0])
    roots = [find(key.split("|")[0]) for key in keys]
    ids = {root: i for i, root in enumerate(dict.fromkeys(roots))}
    return np.array([ids[root] for root in roots])


def grouped_split(keys: list[str], test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle components, then move whole components into test until it holds ~test_frac
    of the clips. Returns (train_idx, test_idx, component_id_per_clip)."""
    comp = connected_components(keys)
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(comp))
    sizes = np.bincount(comp)
    target, taken, test_comps = test_frac * len(keys), 0, set()
    for c in order:
        if taken >= target:
            break
        test_comps.add(int(c))
        taken += sizes[c]
    is_test = np.isin(comp, list(test_comps))
    if is_test.all() or not is_test.any():
        raise ValueError(f"Split failed: {len(np.unique(comp))} component(s) cannot be split; "
                         "the groups are too connected (check data.n_groups)")
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), comp


def get_split(cfg: dict, records: list[dict]) -> dict:
    """Compute (or load the cached) split for this property's clips."""
    from audio_probes.properties import get_property
    from audio_probes.clip import Clip

    out = DATA_DIR / "splits" / cfg["property"] / split_hash(cfg)
    if (out / "split.json").exists():
        return json.loads((out / "split.json").read_text())
    prop = get_property(cfg)
    keys = [prop.split_key(Clip(r["id"], None, 0, r["label"], r)) for r in records]
    train, test, comp = grouped_split(keys, cfg["split"]["test_frac"], cfg["seed"])
    split = {"ids": [r["id"] for r in records], "train": train.tolist(), "test": test.tolist(),
             "groups": comp.tolist(), "n_components": int(comp.max() + 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split))
    save_resolved_config(cfg, out / "config.yaml")
    return split
