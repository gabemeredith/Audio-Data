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

## Distance generator (milestone 3)

- **pyroomacoustics RIRs are scaled 1/r, not 1/(4 pi r).** Measured: the direct peak times
  d is about 0.8 at 1, 2 and 4 m. The fixed gain is set so the direct sound of a 1 m source
  is at `level_at_1m_dbfs` = -35 dBFS. With -30 or louder, close noise bursts in live rooms
  clipped; at -35 the loudest of 2000 clips peaks at 0.37 and none clip.
- **"SNR" knob is an absolute noise floor** (`noise_floor_dbfs`, -85 dBFS), not an SNR
  relative to each clip. A relative SNR would hand the probe the clip's loudness for free in
  the raw condition and blur the loudness cue. Far, quiet clips do approach the floor, as
  they would with a real mic.
- **Rooms that cannot fit the distance.** If a sampled distance does not fit in the block's
  current room after 300 placement tries, that clip gets its own fresh room (same block).
  Large distances therefore lean slightly toward large rooms, which is physical.
- **Grouping** uses 10 blocks; each block owns 4 LibriSpeech speakers and 4 noise-burst
  patterns, plus its own rooms (4 clips per room). Source id for speech = speaker, stricter
  than per-utterance. Result: 546 rooms, 80 sources, 10 connected components, test = 2
  whole blocks (400 clips).
- **DRR sanity test** averages 12 random placements per distance in one room. Single
  placements are not strictly monotonic (a wall just behind the source adds an early
  reflection inside the 2.5 ms direct window), which is real acoustics, not a bug.
- **Source types:** default `[speech, noise]` (50/50). `say` (macOS text-to-speech) and
  `tone` are also available, e.g. `data.source_types=[say,noise]` needs no download.
- **Hand-crafted baseline uses the oracle DRR** from the true RIR plus RMS. Blind DRR
  estimation from audio is a research problem of its own; the spec did not say which, and
  the oracle is the simpler option. It is labelled "oracle" everywhere it appears.

## Distance results and two artifact checks (milestone 3)

Main run (results `raw/5d3ae2fa5e`, `loudness_normalized/097b2bb322`), test = 400 clips
from 2 held-out blocks of rooms and speakers:

| condition | best layer | R^2 | Spearman | MAE | best 4-bin acc | baseline R^2 (RMS + oracle DRR) |
|---|---|---|---|---|---|---|
| raw | 4 | 0.928 | 0.963 | 0.66 m | 0.770 (layer 2) | 0.894 |
| loudness_normalized | 4 | 0.851 | 0.927 | 0.89 m | 0.677 (layer 3) | 0.666 |

Both curves rise to a peak at layers 2 to 5 and fall toward layer 12 (raw 0.79,
normalized 0.68 at the last layer): the encoder keeps distance most linearly readable in
its early-middle layers and partially discards it as it moves toward speech content.

Before trusting the normalized number I checked two simulation artifacts:

1. **Time of flight.** The simulated clip started with d / c seconds of near-silence
   (up to 29 ms), which encodes distance in a way no real microphone could know.
   Added `data.align_direct_path: true` (default) to trim it. Effect: tiny (raw 0.931 ->
   0.928, normalized 0.855 -> 0.851), so Whisper was not leaning on it. Kept the fix anyway.
2. **Noise floor smuggling loudness back in.** The mic noise floor is fixed in absolute
   terms, so after loudness normalisation a far (originally quiet) clip has a relatively
   louder hiss. Ablation with `data.noise_floor_dbfs=-200` (effectively no noise; results
   `raw/593273a37b`, `loudness_normalized/382211f6a1`): raw 0.925 (unchanged), normalized
   **0.778** (best layer 5). So about 0.07 R^2 of the normalized score came through the
   noise floor. The remaining 0.78 is non-loudness information (reverberation, spectral
   colouring) and still beats the oracle-DRR baseline (0.666). The main run keeps the -85
   dBFS floor because real microphones have one; the ablation is the cleaner answer to "is
   anything beyond loudness encoded?" and is reported next to it.

Also: `batch_size`, `device` and `workers` no longer enter cache hashes (they change speed
and memory, not results), and the default batch size is now 4 (~1.5 GB peak instead of ~3).
Pre-alignment result folders were deleted to avoid confusion.
