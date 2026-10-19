"""Stage 5: compare probes, baselines and the audio-LLM on the same test set.

Writes into results/<property>/<condition>/<run_hash>/:
    layer_curve.png   probe score by layer + LLM, chance and hand-crafted baseline lines
    metrics.json      everything in one file
    report.md         a few generated sentences: best layer, best score, LLM score, the gap
"""

from __future__ import annotations

import json

from audio_probes.config import llm_dir, probe_dir, run_dir
from audio_probes.publish import publish


def load_llm_results(cfg: dict, condition: str) -> dict | None:
    path = llm_dir(cfg, condition) / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def run(cfg: dict, condition: str) -> None:
    path = probe_dir(cfg, condition) / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"No probe results; run `make probe PROPERTY={cfg['property']}` first")
    probes = json.loads(path.read_text())
    llm = load_llm_results(cfg, condition)
    out = run_dir(cfg, condition)
    publish(cfg, condition, probe_dir(cfg, condition) / "plots", "probes")
    metrics = {**probes, "llm": llm}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_layer_curve(metrics, cfg, out / "layer_curve.png")
    (out / "report.md").write_text(summary_markdown(metrics, cfg))
    print(f"[report] {cfg['property']}/{condition}: {out}")
    print((out / "report.md").read_text())


def best_layer(metrics: dict, key: str) -> tuple[str, float]:
    name = max(metrics["layers"], key=lambda k: metrics["layers"][k][key])
    return name, metrics["layers"][name][key]


def plot_layer_curve(metrics: dict, cfg: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics["layers"])
    x = [int(n.split("_")[1]) for n in names]
    llm = metrics.get("llm")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("r2", f"R² ({metrics['label_transform']} label)", "numeric"), ("acc4", "4-bin accuracy", "choice")]
    for ax, (key, ylabel, prompt_kind) in zip(axes, panels):
        ax.plot(x, [metrics["layers"][n][key] for n in names], "o-", color="C0", label="linear probe (test)")
        ax.axhline(metrics["chance"][key], color="grey", ls=":", label="chance")
        ax.axhline(metrics["baseline"][key], color="C2", ls="--", label=f"hand-crafted: {metrics['baseline']['features']}")
        if llm and prompt_kind in llm:
            ax.axhline(llm[prompt_kind][key], color="C3", ls="-", lw=2, label=f"audio-LLM ({llm['model']})")
        ax.set(xlabel="Whisper encoder layer (0 = input embeddings)", ylabel=ylabel, xticks=x)
        if key == "r2":
            lo = min(-0.1, min(metrics["layers"][n][key] for n in names), metrics["baseline"][key])
            ax.set_ylim(max(lo, -1.0) - 0.05, 1.05)
        else:
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    llm_note = "" if llm else "  (audio-LLM not run yet; see DECISIONS.md)"
    fig.suptitle(f"{cfg['property']} / {metrics['condition']}: how much survives in each layer{llm_note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_markdown(metrics: dict, cfg: dict) -> str:
    layer, r2 = best_layer(metrics, "r2")
    acc_layer, acc = best_layer(metrics, "acc4")
    m = metrics["layers"][layer]
    from audio_probes.properties import get_property
    unit = " " + get_property(cfg).label_unit
    lines = [
        f"# {cfg['property']} / {metrics['condition']}",
        "",
        f"- Test set: {metrics['n_test']} clips ({metrics['n_train']} train), split by {metrics['n_groups']} room/source groups.",
        f"- Best layer for regression: **{layer.replace('_', ' ')}**, R² = **{r2:.3f}** "
        f"(Spearman {m['spearman']:.3f}, MAE {m['mae']:.3g}{unit}).",
        f"- Best layer for 4-bin classification: **{acc_layer.replace('_', ' ')}**, accuracy **{acc:.3f}** "
        f"(chance {metrics['chance']['acc4']:.3f}).",
        f"- Hand-crafted baseline ({metrics['baseline']['features']}): R² {metrics['baseline']['r2']:.3f}, "
        f"accuracy {metrics['baseline']['acc4']:.3f}.",
    ]
    llm = metrics.get("llm")
    if llm:
        lines += [
            f"- Audio-LLM ({llm['model']}, {llm['numeric']['n']} clips): numeric R² {llm['numeric']['r2']:.3f} "
            f"(unparseable {llm['numeric']['unparseable_rate']:.0%}), multiple-choice accuracy "
            f"{llm['choice']['acc4']:.3f} (unparseable {llm['choice']['unparseable_rate']:.0%}).",
            f"- **Richness lost** (best probe minus LLM): R² gap **{r2 - llm['numeric']['r2']:.3f}**, "
            f"accuracy gap **{acc - llm['choice']['acc4']:.3f}**.",
        ]
    else:
        lines.append("- Audio-LLM: not run yet (model download held back; see DECISIONS.md), so the gap is not computed.")
    return "\n".join(lines) + "\n"


def all_metrics_table(rows: list[tuple[str, str, dict]]) -> str:
    """Markdown table across properties/conditions, used by the final README summary."""
    out = ["| property | condition | best layer | best R² | best acc4 | baseline R² | LLM R² | LLM acc4 |",
           "|---|---|---|---|---|---|---|---|"]
    for prop, cond, m in rows:
        layer, r2 = best_layer(m, "r2")
        _, acc = best_layer(m, "acc4")
        llm = m.get("llm")
        llm_r2 = f"{llm['numeric']['r2']:.3f}" if llm else "not run"
        llm_acc = f"{llm['choice']['acc4']:.3f}" if llm else "not run"
        out.append(f"| {prop} | {cond} | {layer} | {r2:.3f} | {acc:.3f} | {m['baseline']['r2']:.3f} | {llm_r2} | {llm_acc} |")
    return "\n".join(out)

