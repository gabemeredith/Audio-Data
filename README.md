# Audio property probes

**The question:** when Whisper listens to a sound, how much of a physical fact about that
sound survives in each layer of its encoder? And how much of it does an audio-LLM
actually tell you when you ask?

For each property we train a simple "probe" (a weighted sum) to read the fact out of
each encoder layer, then plot the probe's score by layer. An audio-LLM's score on the same
test clips is drawn as a horizontal line. The gap between the best layer and the LLM line
is the "richness lost": information that is there inside the model but does not come out
in words.

Three properties:

| property | role | label | what should happen |
|---|---|---|---|
| pitch | positive control | fundamental frequency, 100-1000 Hz | probe succeeds (R² > 0.9), or the setup is broken |
| highband | negative control | level of a 9-15 kHz noise band | probe fails, because Whisper's 16 kHz input cannot contain it |
| distance | the experiment | source-to-mic distance, 0.5-10 m, in simulated rooms | unknown |

Distance runs twice: **raw** (far sounds are quieter) and **loudness_normalized** (every
clip rescaled to the same level, so only echo and tone cues are left).

## Results

See [`RESULTS.md`](RESULTS.md) for the numbers and charts, and
[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) for what each stage does, in plain words
and then in math. [`DECISIONS.md`](DECISIONS.md) lists every judgement call.

## Running it

Needs macOS or Linux, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv sync                        # Python 3.12 + pinned dependencies
make download-librispeech      # ~337 MB of dry speech for the distance property
make all                       # data -> embed -> probe -> llm -> report, all properties
make test
```

Whisper-small (~1 GB) downloads on first use. The audio-LLM stage is **skipped by
default** because Qwen2-Audio-7B is ~16 GB; turn it on with
`make llm report llm.allow_download=true`.

### Stages and knobs

```bash
make data    PROPERTY=distance data.n_clips=500 data.source_types=[say,noise]
make embed   PROPERTY=pitch encoder.size=tiny encoder.pooling=max
make probe   PROPERTY=distance probe.alpha=10 probe.type=mlp
make llm     PROPERTY=distance llm.n_test_clips=200 llm.temperature=0.2
make report  PROPERTY=distance CONDITION=loudness_normalized
make config  PROPERTY=distance            # print the fully resolved config
```

Every knob lives in [`configs/`](configs) and any `key.sub=value` on the command line
overrides it. Each stage caches under a hash of only the knobs that affect it, so changing
a probe setting reuses the embeddings. Results land in
`results/<property>/<condition>/<run_hash>/`, with `latest` pointing at the newest run and
the full resolved config saved alongside.

### Looking at intermediate results

- Per clip: `data/clips/<property>/<hash>/{wav,spec}/` and `meta.jsonl`.
- Per layer: `results/.../embeddings/pca_layer_XX.png`, `results/.../probes/scatter_layer_XX.png`.
- LLM: `results/.../llm/answers.jsonl` (every prompt, raw answer and parsed value).
- One clip through every stage:
  `make inspect PROPERTY=distance CONDITION=raw CLIP=distance_00042` writes an HTML page
  with the audio, spectrogram, per-layer predictions, PCA positions and LLM answers.

### Interactive lab (localhost)

```bash
make demo        # opens http://127.0.0.1:8770 ; Ctrl+C to stop
```

Needs `make all` first (it loads the trained probes). Keeps Whisper loaded, so it uses about
1.5 GB of memory while it runs. Four tabs:

- **Room lab:** sliders for distance, room size and wall absorption, a sound source
  (LibriSpeech speaker, noise bursts, tone, or a macOS `say` voice), and a loudness-normalize
  toggle. Every change re-simulates the room, plays the clip, and plots what each of the 13
  layers predicts against the true distance. **Sweep** moves the source from 0.5 m to 10 m
  in the same room and plots predicted vs true for any layer.
- **Pitch lab** and **High-band lab:** the two controls, live. In the high-band lab the
  layers should not follow the slider.
- **Your Mac's sounds:** any `/System/Library/Sounds` file or an uploaded audio file.

The server binds to 127.0.0.1 only and uses the Python standard library (no extra deps).

### No microphone needed

```bash
make predict PROPERTY=pitch FILE=/System/Library/Sounds/Glass.aiff   # any audio file
make play PROPERTY=distance CONDITION=raw CLIP=distance_00042 BOOST=20  # headphones
say -o /tmp/hi.aiff "testing one two" && make predict PROPERTY=distance FILE=/tmp/hi.aiff
```

## Layout

```
configs/                    YAML knobs (defaults + one file per property)
src/audio_probes/
  properties/               one plug-in file per property (generate, split key, prompts, parsing)
  encoders/                 Encoder interface + Whisper
  llms/                     AudioLLM interface + Qwen2-Audio + a mock
  data.py embed.py probes.py llm_eval.py report.py     the five stages
  split.py                  grouped train/test split (rooms and sources never straddle it)
  inspect_view.py predict.py                           inspection helpers
tests/                      generators, split, frame trimming, parsing, config
```

Adding a property means writing one file in `properties/`, adding a YAML config, and
registering it in `properties/__init__.py`.
