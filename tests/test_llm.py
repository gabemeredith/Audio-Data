import json

import numpy as np

from audio_probes.config import load_config
from audio_probes.llm_eval import score_answers
from audio_probes.llms import LLMUnavailable, build_llm
from audio_probes.properties.base import parse_choice, parse_number


def test_parse_number():
    assert parse_number("3.5") == 3.5
    assert parse_number("About 2 meters away.") == 2.0
    assert parse_number("2-3 m") == 2.5
    assert parse_number("between 1 and 2") == 1.0
    assert parse_number("-35 dBFS") == -35.0
    assert parse_number("-20 to -10 dB") == -15.0
    assert parse_number("1,200 Hz") == 1200.0
    assert parse_number("I can't tell") is None
    assert parse_number(None) is None


def test_parse_choice():
    for text, want in [("B", 1), ("(C)", 2), ("D) more than 6 meters", 3), ("Answer: A", 0),
                       ("The answer is (B).", 1), ("a", 0)]:
        assert parse_choice(text) == want, text
    for text in ["", "I am not sure", "Both", "Definitely"]:
        assert parse_choice(text) is None, text


def test_unparseable_answers_count_as_wrong():
    cfg = load_config("distance")
    answers = [
        {"prompt_kind": "numeric", "label": 0.7, "parsed": 0.7},
        {"prompt_kind": "numeric", "label": 8.0, "parsed": None},  # -> train median
        {"prompt_kind": "numeric", "label": 4.0, "parsed": -1.0},  # not positive -> invalid on log scale
        {"prompt_kind": "choice", "label": 0.7, "parsed": 0},
        {"prompt_kind": "choice", "label": 8.0, "parsed": None},
    ]
    res = score_answers(cfg, answers, train_labels=np.array([2.0, 2.0, 2.0]))
    assert abs(res["numeric"]["unparseable_rate"] - 2 / 3) < 1e-9
    assert res["choice"]["acc4"] == 0.5 and res["choice"]["unparseable_rate"] == 0.5


def test_qwen_unavailable_without_download():
    cfg = load_config("distance", ["llm.model=nonexistent-org/definitely-not-a-model"])
    try:
        build_llm(cfg)
    except LLMUnavailable:
        return
    raise AssertionError("expected LLMUnavailable")


def test_mock_backend_end_to_end_logs_jsonl(tmp_path, monkeypatch):
    """Run the llm stage with the mock backend on a tiny generated dataset and check the log."""
    import audio_probes.config as config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    import audio_probes.data as data
    import audio_probes.llm_eval as llm_eval
    import audio_probes.split as split

    for mod in (data, llm_eval, split):
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data", raising=False)
    cfg = load_config("pitch", ["data.n_clips=30", "llm.backend=mock", "llm.n_test_clips=5",
                                "artifacts.spectrograms=false"])
    data.run(cfg)
    llm_eval.run(cfg, "raw")
    out = config.llm_dir(cfg, "raw")
    lines = [json.loads(x) for x in (out / "answers.jsonl").read_text().splitlines()]
    assert len(lines) == 10 and {x["prompt_kind"] for x in lines} == {"numeric", "choice"}
    assert all({"prompt", "raw_answer", "parsed", "label"} <= set(x) for x in lines)
    results = json.loads((out / "results.json").read_text())
    assert results["numeric"]["n"] == 5 and results["choice"]["n"] == 5
    llm_eval.run(cfg, "raw")  # resumes: nothing new is asked
    assert len((out / "answers.jsonl").read_text().splitlines()) == 10
