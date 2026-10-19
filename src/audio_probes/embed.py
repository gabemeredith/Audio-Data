"""Stage 2: run every clip through the encoder and cache one pooled vector per layer.

Output, under data/embeddings/<property>/<condition>/<embed_hash>/:
    layers.npy          float32 array (n_layers, n_clips, dim)
    ids.json            clip ids, in row order
    plots/pca_layer_XX.png   2D PCA of all clips, coloured by label
    plots/pca_grid.png       all layers side by side
    plots/norm_stats.json    per-layer vector-norm mean / std / min / max
    plots/pca_coords.npz     the 2D coordinates (used by the inspect view)
"""

from __future__ import annotations

import json
import time

import numpy as np

from audio_probes.config import embed_dir, save_resolved_config
from audio_probes.data import load_clip, load_meta
from audio_probes.publish import publish


def run(cfg: dict, condition: str) -> None:
    out = embed_dir(cfg, condition)
    if (out / "DONE").exists():
        print(f"[embed] {cfg['property']}/{condition}: cached at {out}")
    else:
        extract(cfg, condition, out)
    publish(cfg, condition, out / "plots", "embeddings")


def extract(cfg: dict, condition: str, out) -> None:
    from audio_probes.encoders.whisper import build_encoder

    records = load_meta(cfg)
    encoder = build_encoder(cfg)
    batch_size = cfg["encoder"]["batch_size"]
    per_layer: list[list[np.ndarray]] | None = None
    t0 = time.time()
    for start in range(0, len(records), batch_size):
        batch = [load_clip(cfg, r, condition) for r in records[start:start + batch_size]]
        pooled = encoder.layers_batch([c.audio for c in batch], batch[0].sr)
        per_layer = [[] for _ in pooled] if per_layer is None else per_layer
        for layer, vecs in zip(per_layer, pooled):
            layer.append(vecs)
        if start // batch_size % 20 == 0:
            print(f"[embed] {cfg['property']}/{condition}: {start + len(batch)}/{len(records)} "
                  f"({time.time() - t0:.0f}s, device={encoder.device})")
    elapsed = time.time() - t0
    layers = np.stack([np.concatenate(chunks) for chunks in per_layer]).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "layers.npy", layers)
    (out / "ids.json").write_text(json.dumps([r["id"] for r in records]))
    (out / "timing.json").write_text(json.dumps({
        "device": encoder.device, "seconds": elapsed, "n_clips": len(records),
        "seconds_per_clip": elapsed / len(records), "model": f"whisper-{cfg['encoder']['size']}",
    }, indent=2))
    labels = np.array([r["label"] for r in records])
    write_plots(layers, labels, cfg, out / "plots", encoder.layer_ids)
    save_resolved_config({**cfg, "condition": condition}, out / "config.yaml")
    (out / "DONE").write_text("ok\n")
    print(f"[embed] {cfg['property']}/{condition}: {layers.shape} in {elapsed:.0f}s -> {out}")


def load_embeddings(cfg: dict, condition: str) -> tuple[np.ndarray, list[str]]:
    out = embed_dir(cfg, condition)
    if not (out / "DONE").exists():
        raise FileNotFoundError(f"No embeddings; run `make embed PROPERTY={cfg['property']}` first")
    return np.load(out / "layers.npy"), json.loads((out / "ids.json").read_text())


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the top-2 principal components: center X, take the SVD X = U S V^T,
    return X V[:, :2] and the fraction of variance each component explains (S^2 / sum S^2)."""
    centered = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, (s[:2] ** 2) / np.sum(s**2)


def write_plots(layers: np.ndarray, labels: np.ndarray, cfg: dict, plot_dir, layer_ids) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    color = np.log(labels) if cfg["probe"]["label_transform"] == "log" else labels
    clabel = f"{'log ' if cfg['probe']['label_transform'] == 'log' else ''}{cfg['property']} label"
    coords, stats = [], {}
    n = len(layers)
    cols = min(n, 5)
    grid, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(3.2 * cols, 3 * ((n + cols - 1) // cols)))
    axes = np.atleast_1d(axes).ravel()
    for i, (layer_id, x) in enumerate(zip(layer_ids, layers)):
        xy, var = pca_2d(x)
        coords.append(xy)
        norms = np.linalg.norm(x, axis=1)
        stats[f"layer_{layer_id:02d}"] = {"norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
                                          "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                                          "pc1_var": float(var[0]), "pc2_var": float(var[1])}
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=color, s=6, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=clabel)
        ax.set(title=f"{cfg['property']} layer {layer_id}: PCA", xlabel=f"PC1 ({var[0]:.0%} var)",
               ylabel=f"PC2 ({var[1]:.0%} var)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"pca_layer_{layer_id:02d}.png", dpi=110)
        plt.close(fig)
        axes[i].scatter(xy[:, 0], xy[:, 1], c=color, s=3, cmap="viridis")
        axes[i].set_title(f"layer {layer_id}", fontsize=9)
        axes[i].set_xticks([]), axes[i].set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    grid.suptitle(f"{cfg['property']}: PCA of pooled embeddings per layer (colour = {clabel})")
    grid.tight_layout()
    grid.savefig(plot_dir / "pca_grid.png", dpi=100)
    plt.close(grid)
    np.savez(plot_dir / "pca_coords.npz", coords=np.stack(coords), layer_ids=np.array(layer_ids))
    (plot_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
