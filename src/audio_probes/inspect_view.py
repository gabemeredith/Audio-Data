"""`inspect`: follow one clip through every stage on a single local HTML page.

    make inspect PROPERTY=distance CONDITION=raw CLIP=distance_00042 [RUN=<run_hash>]

The page shows: audio player, log-mel spectrogram, metadata, the clip's position in each
layer's PCA plot, each layer's probe prediction, and the LLM's answers (if that stage ran).
It reads the run's saved config.yaml, so it works for any past run, not just the latest.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from audio_probes.audio import log_mel
from audio_probes.config import RESULTS_DIR, llm_dir, probe_dir
from audio_probes.data import load_clip, load_meta
from audio_probes.probes import inverse_label


def resolve_run(property_name: str, condition: str, run_hash: str | None) -> Path:
    base = RESULTS_DIR / property_name / condition
    path = base / (run_hash or "latest")
    if not path.exists():
        known = sorted(p.name for p in base.glob("*")) if base.exists() else []
        raise FileNotFoundError(f"No run {path.name!r} for {property_name}/{condition}. Known: {known}")
    return path.resolve()


def run(cfg: dict, condition: str, clip_id: str, run_hash: str | None = None) -> Path:
    run_path = resolve_run(cfg["property"], condition, run_hash)
    run_cfg = yaml.safe_load((run_path / "config.yaml").read_text())  # the config that made this run
    records = {r["id"]: r for r in load_meta(run_cfg)}
    if clip_id not in records:
        raise KeyError(f"No clip {clip_id!r} in this run (ids look like {next(iter(records))})")
    record = records[clip_id]
    clip = load_clip(run_cfg, record, condition)

    sections = [
        _audio_section(clip, condition),
        _meta_section(record),
        _probe_section(run_cfg, condition, clip_id, record),
        _pca_section(run_path, run_cfg, list(records), clip_id),
        _llm_section(run_cfg, condition, clip_id),
    ]
    page = _PAGE.format(title=html.escape(f"{clip_id} ({cfg['property']}/{condition})"),
                        run=html.escape(run_path.name), body="\n".join(sections))
    out = run_path / "inspect" / f"{clip_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"[inspect] wrote {out}")
    return out


# ---- sections -----------------------------------------------------------------------

def _audio_section(clip, condition: str) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(clip.audio, -1, 1), clip.sr, format="WAV", subtype="PCM_16")
    wav64 = base64.b64encode(buf.getvalue()).decode()
    png64 = _spectrogram_png64(clip.audio, clip.sr)
    peak = float(np.max(np.abs(clip.audio)))
    gain_note = ""
    if peak < 0.05:
        gain_note = (f"<p class=note>This clip is quiet (peak {20 * np.log10(peak + 1e-12):.0f} dBFS): that is the "
                     "real level the model sees. Use the boost button to hear it.</p>")
    return f"""<section><h2>1. Audio ({html.escape(condition)})</h2>
<audio id=player controls src="data:audio/wav;base64,{wav64}"></audio>
<button onclick="boost()">boost x10 (playback only)</button>{gain_note}
<h3>Log-mel spectrogram (what Whisper's encoder receives)</h3>
<img class=spec src="data:image/png;base64,{png64}">
<p class=note>Horizontal: time (5 s). Vertical: 80 mel bands, 0 Hz at the bottom to 8 kHz at the top.</p></section>"""


def _meta_section(record: dict) -> str:
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(_fmt(v))}</td></tr>" for k, v in record.items())
    return (f"<section><h2>2. Metadata</h2><p class=note>Measured on the clip as generated (raw), "
            f"before any condition such as loudness normalisation.</p><table>{rows}</table></section>")


def _probe_section(cfg: dict, condition: str, clip_id: str, record: dict) -> str:
    path = probe_dir(cfg, condition) / "predictions.npz"
    if not path.exists():
        return "<section><h2>3. Probe predictions</h2><p>Probe stage has not run for this config.</p></section>"
    p = np.load(path)
    i = int(np.flatnonzero(p["ids"] == clip_id)[0])
    kind, edges = cfg["probe"]["label_transform"], cfg["probe"]["bin_edges"]
    side = "TEST (held out)" if p["is_test"][i] else "TRAIN (in-sample, so optimistic)"
    true_bin = int(np.digitize(record["label"], edges))
    rows = []
    for layer_id, z, b in zip(p["layer_ids"], p["z_pred"][:, i], p["bin_pred"][:, i]):
        pred = float(inverse_label(np.array(z), kind))
        ok = "yes" if int(b) == true_bin else "no"
        rows.append(f"<tr><td>{layer_id}</td><td>{pred:.4g}</td><td>{pred - record['label']:+.3g}</td>"
                    f"<td>{'ABCD'[int(b)]}</td><td>{ok}</td></tr>")
    return f"""<section><h2>3. Probe predictions per layer</h2>
<p>True label: <b>{record['label']:.4g}</b> (bin {'ABCD'[true_bin]}). This clip was in the <b>{side}</b> set.</p>
<table><tr><th>layer</th><th>ridge prediction</th><th>error</th><th>4-bin guess</th><th>bin correct?</th></tr>
{''.join(rows)}</table></section>"""


def _pca_section(run_path: Path, cfg: dict, ids: list[str], clip_id: str) -> str:
    path = run_path / "embeddings" / "pca_coords.npz"
    if not path.exists():
        return "<section><h2>4. PCA</h2><p>Embedding stage has not run for this config.</p></section>"
    data = np.load(path)
    coords, layer_ids = data["coords"], data["layer_ids"]
    labels = np.array([r["label"] for r in load_meta(cfg)])
    color = np.log(labels) if cfg["probe"]["label_transform"] == "log" else labels
    i = ids.index(clip_id)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(ids), size=min(600, len(ids)), replace=False)
    svgs = "".join(_scatter_svg(coords[k], color, sample, i, f"layer {layer_ids[k]}") for k in range(len(coords)))
    return f"""<section><h2>4. Where this clip sits in each layer (2D PCA)</h2>
<p class=note>Each dot is a clip, coloured dark (low label) to yellow (high label). The red ring is this clip.</p>
<div class=grid>{svgs}</div></section>"""


def _llm_section(cfg: dict, condition: str, clip_id: str) -> str:
    path = llm_dir(cfg, condition) / "answers.jsonl"
    if not path.exists():
        return ("<section><h2>5. Audio-LLM answers</h2><p>The LLM stage has not run for this config "
                "(the model download is held back; see DECISIONS.md).</p></section>")
    rows = []
    for line in path.read_text().splitlines():
        a = json.loads(line)
        if a["clip_id"] == clip_id:
            rows.append(f"<tr><td>{html.escape(a['prompt_kind'])}</td><td>{html.escape(a['prompt'])}</td>"
                        f"<td>{html.escape(a['raw_answer'])}</td><td>{html.escape(_fmt(a['parsed']))}</td></tr>")
    if not rows:
        return "<section><h2>5. Audio-LLM answers</h2><p>This clip was not among the clips sent to the LLM.</p></section>"
    return f"""<section><h2>5. Audio-LLM answers</h2><table><tr><th>kind</th><th>prompt</th><th>raw answer</th>
<th>parsed</th></tr>{''.join(rows)}</table></section>"""


# ---- helpers ------------------------------------------------------------------------

def _spectrogram_png64(audio: np.ndarray, sr: int) -> str:
    from matplotlib import colormaps
    from PIL import Image

    spec = log_mel(audio, sr)
    norm = (spec - spec.min()) / (np.ptp(spec) + 1e-9)
    rgb = (colormaps["magma"](norm[::-1])[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((spec.shape[1] * 2, spec.shape[0] * 3), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _scatter_svg(xy: np.ndarray, color: np.ndarray, sample: np.ndarray, focus: int, title: str) -> str:
    from matplotlib import colormaps

    size, pad = 220, 12
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    scale = (size - 2 * pad) / np.maximum(hi - lo, 1e-9)

    def px(p):
        return pad + (p[0] - lo[0]) * scale[0], size - pad - (p[1] - lo[1]) * scale[1]

    cnorm = (color - color.min()) / (np.ptp(color) + 1e-9)
    cmap = colormaps["viridis"]
    dots = []
    for j in sample:
        x, y = px(xy[j])
        r, g, b, _ = (np.array(cmap(cnorm[j])) * 255).astype(int)
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="rgb({r},{g},{b})" opacity="0.7"/>')
    fx, fy = px(xy[focus])
    dots.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="6" fill="none" stroke="red" stroke-width="2.5"/>')
    return (f'<figure><svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<rect width="{size}" height="{size}" fill="white" stroke="#ccc"/>{"".join(dots)}</svg>'
            f"<figcaption>{html.escape(title)}</figcaption></figure>")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px auto; max-width: 1100px; padding: 0 16px;
       color: #1d1d1f; background: #fafafa; }}
section {{ background: white; border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px 18px; margin: 14px 0; }}
table {{ border-collapse: collapse; font-size: 13px; }} td, th {{ border: 1px solid #ddd; padding: 3px 8px; text-align: left; }}
.spec {{ width: 100%; image-rendering: pixelated; border: 1px solid #ddd; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 10px; }} figure {{ margin: 0; text-align: center; font-size: 12px; }}
.note {{ color: #666; font-size: 13px; }} audio {{ width: 100%; }}
</style></head><body>
<h1>{title}</h1><p class=note>run {run}</p>
{body}
<script>
let ctx, gainNode;
function boost() {{
  const el = document.getElementById('player');
  if (!ctx) {{ ctx = new AudioContext(); const src = ctx.createMediaElementSource(el);
    gainNode = ctx.createGain(); src.connect(gainNode).connect(ctx.destination); }}
  gainNode.gain.value = gainNode.gain.value > 1 ? 1 : 10; el.play();
}}
</script></body></html>"""
