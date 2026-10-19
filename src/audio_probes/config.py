"""Config loading, command-line overrides, and config hashing.

Every knob lives in YAML. A run's config is built in three layers, later layers win:

    configs/defaults.yaml  ->  configs/<property>.yaml  ->  CLI overrides (key.sub=value)

Each pipeline stage caches its output under a hash of only the config sections that
affect it, so changing a probe knob never forces re-generating audio.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in recursively (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> dict:
    """Turn 'probe.alpha=10' into {'probe': {'alpha': 10}}.

    The value is parsed as YAML, so numbers, booleans, null and lists ('[1, 2]') work.
    """
    if "=" not in text:
        raise ValueError(f"Override must look like key.sub=value, got {text!r}")
    dotted, raw_value = text.split("=", 1)
    value = yaml.safe_load(raw_value) if raw_value != "" else None
    keys = dotted.strip().split(".")
    nested: dict = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        nested = {key: nested}
    return nested


def load_config(property_name: str, overrides: list[str] | None = None) -> dict:
    """Load defaults, then the property's YAML, then CLI overrides."""
    cfg = yaml.safe_load((CONFIG_DIR / "defaults.yaml").read_text())
    prop_path = CONFIG_DIR / f"{property_name}.yaml"
    if not prop_path.exists():
        raise FileNotFoundError(f"No config for property {property_name!r} at {prop_path}")
    cfg = deep_merge(cfg, yaml.safe_load(prop_path.read_text()) or {})
    for text in overrides or []:
        cfg = deep_merge(cfg, parse_override(text))
    cfg["property"] = property_name
    return cfg


def config_hash(obj: Any, length: int = 10) -> str:
    """Short, stable hash of any JSON-serialisable object (key order does not matter)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


# ---- per-stage hashes: each one includes everything upstream of it ----------------

def data_hash(cfg: dict) -> str:
    return config_hash({"property": cfg["property"], "seed": cfg["seed"], "data": cfg["data"]})


def embed_hash(cfg: dict, condition: str) -> str:
    return config_hash({"data": data_hash(cfg), "condition": condition, "encoder": cfg["encoder"]})


def split_hash(cfg: dict) -> str:
    return config_hash({"data": data_hash(cfg), "split": cfg["split"], "seed": cfg["seed"]})


def probe_hash(cfg: dict, condition: str) -> str:
    return config_hash({"embed": embed_hash(cfg, condition), "split": split_hash(cfg), "probe": cfg["probe"]})


def llm_hash(cfg: dict, condition: str) -> str:
    return config_hash({"data": data_hash(cfg), "split": split_hash(cfg), "condition": condition, "llm": cfg["llm"]})


def run_hash(cfg: dict, condition: str) -> str:
    """Hash naming the results folder: covers every knob that can change a result."""
    return config_hash({"probe": probe_hash(cfg, condition), "llm": llm_hash(cfg, condition)})


# ---- paths -------------------------------------------------------------------------

def clips_dir(cfg: dict) -> Path:
    return DATA_DIR / "clips" / cfg["property"] / data_hash(cfg)


def embed_dir(cfg: dict, condition: str) -> Path:
    return DATA_DIR / "embeddings" / cfg["property"] / condition / embed_hash(cfg, condition)


def probe_dir(cfg: dict, condition: str) -> Path:
    return DATA_DIR / "probes" / cfg["property"] / condition / probe_hash(cfg, condition)


def llm_dir(cfg: dict, condition: str) -> Path:
    return DATA_DIR / "llm" / cfg["property"] / condition / llm_hash(cfg, condition)


def run_dir(cfg: dict, condition: str) -> Path:
    return RESULTS_DIR / cfg["property"] / condition / run_hash(cfg, condition)


def save_resolved_config(cfg: dict, path: Path) -> None:
    """Write the fully resolved config next to a stage's output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
