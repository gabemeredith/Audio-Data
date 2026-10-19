# Decisions

Places where the spec was ambiguous (or where I had to deviate), what I picked, and why.
Simpler option wins by default. Newest at the bottom.

## Process

- **No stops between milestones.** The spec asks to stop after each milestone for review.
  Gabe asked for one unattended run to the finish, so I ran straight through and recorded
  each milestone's result in its commit message and in this file instead.
- **Repo root is the project root.** The spec shows an `audio-probes/` folder; the GitHub
  repo already is that folder, so `pyproject.toml`, `Makefile`, `src/` etc. live at the root.
- **Python 3.12 managed by uv.** All dependencies pinned with `==` in `pyproject.toml`,
  with the full lock in `uv.lock`.

## Downloads held back (Gabe will do these later)

- **The audio-LLM (Qwen2-Audio-7B-Instruct, about 16 GB) is NOT downloaded.** The spec says
  to ask before any download over 2 GB, and Gabe said to hold off on heavy downloads.
  `llm.allow_download: false` makes the `llm` stage skip with a clear message instead of
  downloading. To run it later:
  `make llm report llm.allow_download=true` (needs ~16 GB disk, ~20 GB RAM; runs on Apple
  MPS or CUDA). Paid API backends are also not called, for the same reason.
- Small downloads I did make: `openai/whisper-small` weights (about 1 GB, from Hugging
  Face) and LibriSpeech dev-clean (about 337 MB, from openslr.org). Both are under the
  2 GB limit.

## Config and caching

- **CLI overrides use `key.sub=value`**, parsed as YAML, so `probe.alpha=10`,
  `encoder.layers=[0,6,12]`, `artifacts.spectrograms=false` all work. `make` forwards any
  such argument through `$(MAKEOVERRIDES)`.
- **Each stage caches under a hash of just the knobs that affect it** (data, embedding,
  split, probe, LLM). The results folder is `results/<property>/<condition>/<run_hash>/`
  and `results/<property>/<condition>/latest` points at the newest run. The fully
  resolved config is saved as `config.yaml` in every cache and results folder.
