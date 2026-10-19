"""Stage 4: ask an audio-LLM about each test clip, log every exchange, and score the answers
with the same metrics as the probes.

Output, under data/llm/<property>/<condition>/<llm_hash>/:
    answers.jsonl   one line per (clip, prompt): prompt, raw answer, parsed value, label
    results.json    metrics for the numeric prompt (R^2, Spearman, MAE) and the choice prompt
                    (4-bin accuracy), each with its unparseable-answer rate
    SKIPPED.txt     written instead when the backend cannot run here (e.g. weights not downloaded)

Unparseable answers count as wrong: a choice answer that is not A-D scores 0, and a numeric
answer that is missing (or not positive, for a log-scale label) is replaced by the training
median, i.e. a chance-level guess.
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np

from audio_probes.config import llm_dir, run_dir, save_resolved_config
from audio_probes.data import load_clip, load_meta
from audio_probes.probes import accuracy, regression_metrics, to_bins, transform_label
from audio_probes.properties import get_property
from audio_probes.split import get_split


def run(cfg: dict, condition: str) -> None:
    from audio_probes.llms import LLMUnavailable, build_llm

    out = llm_dir(cfg, condition)
    out.mkdir(parents=True, exist_ok=True)
    records = load_meta(cfg)
    split = get_split(cfg, records)
    clips = choose_test_clips(cfg, records, split)
    try:
        llm = build_llm(cfg)
    except LLMUnavailable as err:
        (out / "SKIPPED.txt").write_text(f"{err}\n")
        print(f"[llm] {cfg['property']}/{condition}: SKIPPED. {err}")
        return
    (out / "SKIPPED.txt").unlink(missing_ok=True)
    ask_all(cfg, condition, llm, clips, out / "answers.jsonl")
    train_labels = np.array([records[i]["label"] for i in split["train"]])
    results = score_answers(cfg, read_answers(out / "answers.jsonl"), train_labels)
    results.update({"model": f"{cfg['llm']['backend']}:{cfg['llm']['model']}" if cfg["llm"]["backend"] != "mock"
                    else "mock", "condition": condition})
    (out / "results.json").write_text(json.dumps(results, indent=2))
    save_resolved_config({**cfg, "condition": condition}, out / "config.yaml")
    dest = run_dir(cfg, condition) / "llm"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "answers.jsonl", dest / "answers.jsonl")
    print(f"[llm] {cfg['property']}/{condition}: numeric R2={results['numeric']['r2']:.3f} "
          f"(unparseable {results['numeric']['unparseable_rate']:.0%}), "
          f"choice acc={results['choice']['acc4']:.3f} (unparseable {results['choice']['unparseable_rate']:.0%})")


def choose_test_clips(cfg: dict, records: list[dict], split: dict) -> list[dict]:
    """The first n_test_clips test clips after a seeded shuffle (all of them if null)."""
    test = [records[i] for i in split["test"]]
    order = np.random.default_rng(cfg["seed"]).permutation(len(test))
    n = cfg["llm"]["n_test_clips"] or len(test)
    return [test[i] for i in order[:n]]


def read_answers(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def ask_all(cfg: dict, condition: str, llm, clips: list[dict], log_path) -> None:
    """Ask every (clip, prompt) pair not already in the log. Appends as it goes, so an
    interrupted run resumes where it stopped."""
    prop = get_property(cfg)
    prompts = prop.llm_prompts()
    done = {(a["clip_id"], a["prompt_kind"]) for a in read_answers(log_path)}
    t0, asked = time.time(), 0
    with open(log_path, "a") as log:
        for record in clips:
            todo = [kind for kind in prompts if (record["id"], kind) not in done]
            if not todo:
                continue
            clip = load_clip(cfg, record, condition)
            for kind in todo:
                raw = llm.ask(clip.audio, clip.sr, prompts[kind])
                entry = {"clip_id": clip.id, "prompt_kind": kind, "prompt": prompts[kind], "raw_answer": raw,
                         "parsed": prop.parse_llm(kind, raw), "label": clip.label, "backend": llm.name,
                         "temperature": cfg["llm"]["temperature"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                asked += 1
            if asked and asked % 50 == 0:
                print(f"[llm] {cfg['property']}/{condition}: {asked} answers ({time.time() - t0:.0f}s)")


def score_answers(cfg: dict, answers: list[dict], train_labels: np.ndarray) -> dict:
    kind, edges = cfg["probe"]["label_transform"], cfg["probe"]["bin_edges"]
    fallback = float(np.median(train_labels))
    numeric = [a for a in answers if a["prompt_kind"] == "numeric"]
    y = np.array([a["label"] for a in numeric])
    guesses = [a["parsed"] for a in numeric]
    valid = [g is not None and (kind != "log" or g > 0) for g in guesses]
    pred = np.array([g if ok else fallback for g, ok in zip(guesses, valid)], dtype=float)
    num = regression_metrics(y, transform_label(pred, kind), kind) if len(y) > 1 else {}
    num.update({"n": len(y), "unparseable_rate": float(1 - np.mean(valid)) if valid else 0.0})
    num["acc4"] = accuracy(to_bins(y, edges), to_bins(pred, edges)) if len(y) else float("nan")

    choice = [a for a in answers if a["prompt_kind"] == "choice"]
    yc = np.array([a["label"] for a in choice])
    picks = np.array([a["parsed"] if a["parsed"] is not None else -1 for a in choice])
    cho = {"n": len(choice), "acc4": accuracy(to_bins(yc, edges), picks) if len(choice) else float("nan"),
           "unparseable_rate": float(np.mean(picks == -1)) if len(choice) else 0.0}
    # The plot draws the numeric R^2 and the choice accuracy as the LLM lines.
    cho["r2"] = num.get("r2", float("nan"))
    return {"numeric": num, "choice": cho}
