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

## Mac audio helpers (added on request, mid-run)

- Gabe asked to be able to use macOS audio instead of speaking into the mic (working in a
  library). I read that as "let me feed in sounds already on my Mac and listen on
  headphones". Added:
  - `make predict PROPERTY=pitch FILE=/System/Library/Sounds/Glass.aiff` runs any audio
    file (decoded by ffmpeg, so aiff/wav/m4a/mp3) through Whisper and the trained probes
    and prints each layer's guess. Probes are saved as `models.pkl` for this.
  - `make play PROPERTY=... CLIP=... BOOST=20` plays a generated clip with `afplay`.
  - The inspect page already has an in-browser player with a x10 boost button.
  - The distance property can use macOS text-to-speech (`say`) voices as dry sources
    (`data.source_types=[say]`), so it runs with no LibriSpeech download.

## High-band control (milestone 2)

- **Resampler = ffmpeg's default**, invoked the same way openai-whisper's `load_audio`
  does (`ffmpeg -ar 16000`), because the spec says "exactly as Whisper's preprocessing
  does". `data.resampler=scipy` switches to `scipy.signal.resample_poly`.
- **The resampler is not a perfect wall.** Measured: a -10 dBFS band at 9-15 kHz leaves a
  residue of about -56 dBFS below 8 kHz with ffmpeg (-57 with scipy), i.e. about 46 dB of
  attenuation, spread fairly evenly across 0-8 kHz. The test asserts at least 40 dB.
- **Result: no leakage reached the probes.** Every layer R^2 is between -0.005 and 0.000,
  and 4-bin accuracy is 0.22 to 0.27 against a chance rate of 0.23. The hand-crafted
  baseline (overall level and 7.5-8 kHz level) also fails (R^2 -0.001). The residue sits
  20 to 50 dB under the low-band content and near the -70 dBFS noise floor, so it is
  not usable. No debugging was needed.
- The label is the level we set; the clip metadata also stores the level measured at
  48 kHz (they agree within 0.5 dB, tested) and the 7.5-8 kHz level after resampling.
