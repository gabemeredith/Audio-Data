"""Copy a stage's human-facing artifacts from its cache into the results run folder.

Caches live under data/ (keyed by the stage's own hash). The results folder
results/<property>/<condition>/<run_hash>/ collects every stage's plots for one full config,
and results/<property>/<condition>/latest points at the most recent run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from audio_probes.config import run_dir, save_resolved_config


def publish(cfg: dict, condition: str, src: Path, name: str) -> Path:
    """Copy folder `src` to <run_dir>/<name>, save the resolved config, and update `latest`."""
    dest_root = run_dir(cfg, condition)
    dest = dest_root / name
    if src.exists():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    save_resolved_config({**cfg, "condition": condition}, dest_root / "config.yaml")
    point_latest(dest_root)
    return dest


def point_latest(dest_root: Path) -> None:
    latest = dest_root.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(dest_root.name, latest)
