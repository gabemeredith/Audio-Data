Spec: Audio Property Probe Harness
Oct 19, 2026 · @gabe
Goal
Build a small repo that answers one question: how much of a physical audio property survives at each layer of Whisper's encoder, and how much of it does an audio-LLM actually report?
The deliverable is one chart per property: probe score by encoder layer, with the audio-LLM's score drawn as a horizontal line. The gap between the best layer and the LLM line is "richness lost."
v1 property: source distance (how far the sound source is from the mic).
Scope
v1 runs three properties: the target, plus two controls that prove the harness works.
Property
Role
Label
Expected result
Why
Source distance
Target
Distance in meters, 0.5–10 m
Unknown; this is the experiment
Physical fact with no natural word for it
Pitch
Positive control
Fundamental frequency in Hz, 100–1000 Hz
Probe should succeed (R² > 0.9)
If it fails, the probe setup is broken
Energy above 8 kHz
Negative control
Level of the 9–15 kHz band in dB
Probe should fail (near chance)
Whisper resamples to 16 kHz, so this info is destroyed. Success means label leakage
Distance runs in two conditions:
• Raw: loudness varies naturally with distance.
• Loudness-normalized: every clip RMS-normalized to -20 dBFS. Tests whether anything beyond loudness (direct-to-reverb ratio, high-frequency loss) is encoded.
Non-goals for v1: fine-tuning any model, real recordings, encoders other than Whisper, any UI. The interfaces should allow adding these later.
Pipeline
Five stages. Each one reads the previous stage's cached output from disk, so any stage can be rerun alone.
1. Generate data.
    ◦ Distance: take dry source clips (LibriSpeech dev-clean speech plus synthetic noise bursts) and simulate shoebox rooms with pyroomacoustics. Randomize room size, wall absorption, and mic position. Place the source at distance d. Output 5 s clips at 16 kHz.
    ◦ Pitch: harmonic tones with random timbre (random harmonic weights) so the probe can't cheat on one fixed waveform.
    ◦ High-band control: generate at 48 kHz, add band-limited noise at 9–15 kHz with a random level, then resample to 16 kHz exactly as Whisper's preprocessing does.
    ◦ About 2,000 clips per property. Split train/test by room and by source clip, never at random, or the probe memorizes rooms.
2. Extract embeddings. Run openai/whisper-small (Hugging Face transformers) with output_hidden_states=True. Whisper pads every input to 30 s, so mean-pool only the frames covering real audio (the encoder outputs 50 frames per second; a 5 s clip = first 250 frames). Pooling over the padding would dilute every feature. Cache one array per layer.
3. Train probes. Per layer: standardize features, then fit RidgeCV on the label (log-distance for distance). Also fit a 4-bin logistic-regression classifier for distance (bins: <1 m, 1–3 m, 3–6 m, >6 m) so the score is comparable to the LLM's multiple-choice answer.
4. Query the audio-LLM. Make the model configurable. Default: an open model such as Qwen2-Audio-7B-Instruct if a GPU is available, otherwise an API audio model. Ask each test clip two prompts: a number ("Estimate the distance in meters between the sound source and the microphone. Answer with a number only.") and a 4-way multiple choice matching the bins. Parse answers; count unparseable ones as wrong and log them.
5. Compare. Same test set, same metrics for probe and LLM. Plot probe score per layer with the LLM score as a horizontal line, plus a chance baseline and a hand-crafted baseline (RMS and a direct-to-reverb estimate fed to the same ridge regression).
Repo structure
A property is a plug-in; the encoder, probe, LLM, and plotting code is shared. Adding a property should mean writing one file.
audio-probes/
  pyproject.toml          # pinned versions
  Makefile                # make data / embed / probe / llm / report / all
  configs/
    distance.yaml         # n_clips, seed, ranges, conditions
    pitch.yaml
    highband.yaml
  src/audio_probes/
    properties/
      base.py             # Property interface
      distance.py
      pitch.py
      highband.py
    encoders/
      base.py             # Encoder interface
      whisper.py
    llms/
      base.py             # AudioLLM interface
      qwen_audio.py
    probes.py             # ridge + logistic, per layer
    report.py             # plots + metrics.json + report.md
  tests/
  data/                   # gitignored: clips, embeddings, LLM answers
  results/<property>/<condition>/
Interfaces (Python, roughly):
class Property:
    name: str
    label_kind: Literal["regression", "classification"]
    def generate(self, n: int, seed: int) -> list[Clip]: ...   # Clip = audio, sr, label, meta (room_id, source_id)
    def split_key(self, clip: Clip) -> str: ...               # group key for train/test split
    def llm_prompts(self) -> dict[str, str]: ...              # "numeric", "choice"
    def parse_llm(self, prompt_kind: str, text: str) -> float | int | None: ...

class Encoder:
    def layers(self, audio: np.ndarray, sr: int) -> list[np.ndarray]: ...  # one pooled vector per layer

class AudioLLM:
    def ask(self, audio: np.ndarray, sr: int, prompt: str) -> str: ...
Outputs and acceptance criteria
The run is done when make all reproduces every result from a fixed seed and the controls behave as expected.
Per property and condition, results/<property>/<condition>/ holds:
• layer_curve.png: probe score by layer, LLM line, chance line, hand-crafted baseline line.
• metrics.json: per layer R², Spearman, and MAE (in meters for distance); 4-bin accuracy; same for the LLM, plus its unparseable-answer rate.
• report.md: a few auto-generated lines stating best layer, best score, LLM score, and the gap.
Acceptance:
[ ] Pitch control: best-layer R² > 0.9.
[ ] High-band control: every layer R² < 0.1 (near chance).
[ ] Distance: results for both raw and loudness-normalized conditions.
[ ] LLM evaluated on at least 200 test clips per prompt type.
[ ] Train/test split is grouped by room and source; a test asserts no group appears in both.
[ ] Generator sanity test: in simulated rooms, direct-to-reverb ratio decreases as distance increases.
[ ] Embedding extraction for all three properties runs on a laptop without a GPU (target: under ~1 hour for whisper-small; a guess, adjust if wrong).
Knobs, observability, and explainability
I need to change any setting without editing code, see every intermediate result, and understand each stage conceptually and mathematically.
Knobs. Every setting lives in the YAML configs and can be overridden from the command line (e.g. make probe PROPERTY=distance probe.alpha=10).
Stage
Knobs
Data
n_clips, seed, distance range, room size range, wall absorption, source type (speech / noise / tone), SNR, loudness normalization on/off
Encoder
model size (tiny / base / small), which layers, pooling (mean / max / first N frames)
Probe
type (ridge / logistic / small MLP for comparison), regularization strength, label transform (linear / log), bin edges
LLM
model, prompt text, temperature, number of test clips
Each run saves its fully resolved config next to its results, and results folders are named by config hash, so two runs can always be compared.
Observability. Every stage writes artifacts I can open:
• Data: the wav, a log-mel spectrogram PNG, and metadata (room size, distance, measured direct-to-reverb ratio, RMS) per clip.
• Embeddings: per layer, a 2D PCA plot of all clips colored by label, plus vector-norm stats.
• Probes: per layer, a predicted-vs-true scatter plot and residuals.
• LLM: the JSONL log of prompt, raw answer, and parsed value.
• An inspect command (a small local HTML page is fine) that follows one clip through every stage: play audio, view spectrogram, see its position in each layer's PCA plot, each layer's probe prediction, and the LLM's answer.
Explainability. Write docs/HOW_IT_WORKS.md: one short section per stage, plain words first, then the math, then the function that implements it. It must cover:
• Room simulation: image-source method, inverse-square loudness falloff, direct-to-reverb ratio.
• Log-mel features: STFT, mel filterbank, log, and what each step discards.
• Pooling: mean over real (non-padded) frames.
• Ridge regression: the closed-form solution and what the regularization term does.
• Logistic regression and the R², Spearman, and MAE metrics.
Code should be readable over clever: small functions, numpy/sklearn rather than custom frameworks, and a docstring naming the equation each function computes.
Build the knobs, inspect view, and HOW_IT_WORKS.md section for each stage in the same milestone as the stage itself, not at the end.
[ ] Any knob above can be changed from the CLI without code edits.
[ ] inspect works on any clip ID from any run.
[ ] HOW_IT_WORKS.md covers every stage, with each formula linked to its function.
Instructions for Claude Code
Build in milestones and stop after each one so I can look at the result before you continue.
1. Milestone 1: pitch control end to end. Scaffold the repo and interfaces, then implement pitch generation, Whisper embeddings, probes, and the plot. No LLM yet. Stop and show me the layer curve.
2. Milestone 2: high-band control. If any layer scores well, stop and debug leakage before going further.
3. Milestone 3: distance in both conditions, probes only.
4. Milestone 4: audio-LLM evaluation on all three properties, then the final report.
Constraints:
• Ask before any download over 2 GB or any paid API call.
• Pin all dependency versions. Core logic lives in src/, not notebooks.
• Cache every stage to data/ so reruns skip finished work.
• Write tests for generators, the grouped split, and frame-trimming before pooling.
• Log every LLM prompt and raw answer to a JSONL file.
• Commit after every step in the commit plan below, and only when tests pass. Each message states what changed and, if the step produced results, what they showed.
• When something is ambiguous, pick the simpler option, note it in DECISIONS.md, and keep going.
Commit plan (~10 commits):
1. Scaffold: repo layout, pinned dependencies, Makefile, config loading with CLI overrides, the three interfaces.
2. Pitch generator, its tests, and per-clip artifacts (wav, spectrogram, metadata).
3. Whisper encoder: frame trimming, pooling, caching, and a test for trimming.
4. Probes, metrics, grouped split with its test, and the layer-curve plot.
5. inspect view and HOW_IT_WORKS.md for the stages so far. End of milestone 1.
6. High-band control generator and its results. End of milestone 2.
7. Distance generator (pyroomacoustics) with the direct-to-reverb sanity test.
8. Distance probes in both conditions, plus the hand-crafted baseline. End of milestone 3.
9. AudioLLM interface, one backend, answer parsing, and JSONL logging.
10. LLM evaluation on all three properties, final report, and completed HOW_IT_WORKS.md. End of milestone 4.