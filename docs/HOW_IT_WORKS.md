# How it works

The question: when Whisper "listens" to a sound, how much of a physical fact about that
sound (its pitch, how far away it was) is still in its internal numbers at each layer? And
how much of that does an audio-LLM actually tell you when you ask?

The pipeline has five stages. Each section below says what the stage does in plain words,
then gives the math, then names the function that does it. Every stage caches its output in
`data/`, so you can rerun any one of them alone.

```
 generate clips  ->  Whisper encoder  ->  linear probes  ->  ask an audio-LLM  ->  compare
   (data.py)          (embed.py)          (probes.py)       (llm_eval.py)        (report.py)
```

---

## 1. Generating labelled audio

**Plain words.** We need thousands of short sounds where we know the exact answer (the
"label"). Recording them for real is slow and noisy, so we synthesise them. Each property
is a plug-in in `src/audio_probes/properties/` that knows how to make its clips.

### Pitch (positive control)

A musical note is a stack of sine waves at whole-number multiples of a base frequency, the
**fundamental** f0. We pick f0 between 100 and 1000 Hz, then give every clip a different
random recipe of how loud each multiple ("harmonic") is, so no two clips look alike even at
the same pitch. If Whisper's layers did not encode pitch, something in our setup would be broken.

**Math.**

    x(t) = sum_{k=1..K} w_k * sin(2*pi*k*f0*t + phi_k),   K = floor(7500 / f0)
    w_k  = u_k * k^(-decay),  u_k ~ Uniform(0, 1),  decay ~ Uniform(0, 2),  phi_k ~ Uniform(0, 2*pi)

f0 is drawn log-uniformly (equal chance per octave). All harmonics stay below 7.5 kHz so
nothing is lost when Whisper looks only up to 8 kHz.

**Code.** [`harmonic_tone`, `random_timbre`, `Pitch.generate`](../src/audio_probes/properties/pitch.py).

**Hand-crafted baseline.** A classic pitch tracker: the autocorrelation
r(lag) = sum_n x[n] x[n + lag] peaks when the lag equals one period, so
f0 = sample_rate / best_lag. [`autocorr_f0`](../src/audio_probes/properties/pitch.py).

### Source distance (the target): simulated rooms

**Plain words.** Imagine clapping once in a room. The microphone hears the clap arrive
directly, then a smear of echoes off the walls. That "click response" is the room impulse
response (RIR). Any sound played at that spot in that room reaches the mic as the sound
blended with its echoes, which is a convolution of the dry sound with the RIR. We build
random box-shaped rooms, place a mic and a source d metres apart, compute the RIR, and
convolve a dry sound (LibriSpeech speech or synthetic noise bursts; macOS `say` voices or
tones as options) with it.

Two things change as the source moves away:

1. **Loudness falls.** Sound power spreads over a sphere of area 4 pi d^2, so power falls as
   1 / d^2 and amplitude as 1 / d: every doubling of distance is 6 dB quieter. We keep
   one fixed gain for all clips, so the raw clips really do get quieter with distance.
2. **The room takes over.** The direct sound weakens with distance but the diffuse reverb
   in a room is roughly the same everywhere, so far sources sound more echoey. The
   **direct-to-reverb ratio** (DRR) measures this.

The `loudness_normalized` condition rescales every clip to -20 dBFS, which deletes cue 1.
Whatever the probe still finds there must come from cue 2 or other subtler changes.

**Math.**

- *Image-source method.* A reflection off a wall sounds exactly like a second source
  mirrored behind that wall. Mirroring the mirrors gives every multi-bounce path. The RIR
  is a sum of delayed, scaled clicks, one per image source i at distance r_i that bounced
  n_i times off walls with energy absorption a:
  `h(t) = sum_i (1 - a)^(n_i / 2) / r_i * delta(t - r_i / c)`, with c = 343 m/s. We keep
  images up to `max_order` bounces.
- *Inverse-square law.* The 1 / r_i factor above (amplitude), i.e. 1 / r^2 in power.
- *DRR.* With n_d the sample where the direct sound arrives and a +-2.5 ms window w:
  `DRR = 10 log10( sum_{|n - n_d| <= w} h[n]^2 / sum_{n > n_d + w} h[n]^2 )` dB.
- *Reverb time* (stored as metadata). Sabine: `RT60 = 0.161 V / (S a)` seconds.
- *Mic signal.* `y = g * (x conv h) + noise`, with one fixed g for every clip and a fixed
  -85 dBFS noise floor (so a far, quiet source sinks toward the noise).

**Code.** [`simulate_rir`, `direct_to_reverb_db`, `place_pair`, `sabine_rt60`, `render_clip`,
`Distance.plan`](../src/audio_probes/properties/distance.py). Sanity tests in
[`tests/test_distance.py`](../tests/test_distance.py) check that DRR falls with distance
(averaged over placements in one room, and as a rank correlation across random rooms),
that the direct sound arrives at sample n_d, and that raw loudness falls with distance.

**Grouping.** Rooms and sources are dealt into 10 blocks and a clip only pairs a room and
source from the same block, so the grouped split can hold out whole blocks: test rooms and
test speakers are never heard in training.

**Hand-crafted baseline.** RMS level plus the DRR measured from the true RIR. The DRR is an
*oracle* (a real device would have to estimate it from the audio alone), so this line is
a generous reference, not a fair competitor.

### Energy above 8 kHz (negative control)

**Plain words.** Whisper only ever sees audio at 16 kHz, which can only describe frequencies
up to 8 kHz. So we hide a fact *above* 8 kHz: we build each clip at 48 kHz, add a hiss
between 9 and 15 kHz at a random loudness (the label), then shrink the clip to 16 kHz the
same way Whisper's loader does. The hiss should be gone. If a probe could still guess its
loudness, information is leaking somewhere, either through an imperfect resampler or a bug
that lets the label sneak into the features.

**Math.** Band-limited noise: take white noise, FFT it, zero every bin outside 9 to 15 kHz,
inverse FFT. Its level is measured with Parseval's theorem, mean(x_band^2) =
2 * sum_{k in band} |X_k|^2 / n^2. Resampling 48 kHz to 16 kHz low-pass filters at 8 kHz
and keeps every third sample; a real filter only *attenuates* the band (we measured about
46 dB), it does not remove it completely.

**Code.** [`HighBand.generate`](../src/audio_probes/properties/highband.py),
[`band_limited_noise`, `band_level_dbfs`, `resample_like_whisper`](../src/audio_probes/audio.py).

### Levels in dB

Loudness is measured as RMS (root mean square, the typical size of the wave):
rms = sqrt(mean(x^2)), and expressed in decibels relative to the loudest possible digital
signal: dBFS = 20 * log10(rms). Every 6 dB is a factor of two in amplitude.
[`rms`, `set_rms_dbfs`, `db_to_amp`](../src/audio_probes/audio.py).

### Per-clip artifacts

For every clip the data stage saves a 32-bit float `.wav` (float, so very quiet details
survive), a spectrogram picture, and a line of metadata in `meta.jsonl`.
[`data.run`, `write_clips`](../src/audio_probes/data.py).

---

## 2. From waveform to log-mel features (what Whisper actually sees)

**Plain words.** Whisper never looks at the raw wave. It first turns the sound into a
picture: time left to right, pitch bottom to top, brightness = how much energy. Each step of
making that picture throws something away, and anything thrown away here can never be
recovered by any later layer.

**Math, step by step** (16 kHz audio):

1. **STFT** (short-time Fourier transform). Cut the wave into 25 ms windows (N = 400
   samples) every 10 ms (hop 160), taper each with a Hann window w[n], and measure how
   much of each frequency it contains:
   `X[t, k] = sum_n x[n + 160 t] * w[n] * exp(-2 pi i k n / 400)`.
   *Discards:* anything above 8 kHz (half the 16 kHz sample rate, the Nyquist limit) and
   timing finer than about 10 ms.
2. **Power.** `P[t, k] = |X[t, k]|^2`. *Discards:* phase (where in its cycle each wave is).
3. **Mel filterbank.** Sum the 201 frequency bins into 80 overlapping triangular bands
   that are narrow at low frequencies and wide at high ones, like the ear:
   `M = F @ P`. *Discards:* fine frequency detail, especially at high frequencies.
4. **Log and clamp.** `L = log10(max(M, 1e-10))`, then `L = max(L, max(L) - 8)` and
   `(L + 4) / 4`. *Discards:* everything more than 80 dB quieter than the loudest point.
   Note the log makes overall loudness an *additive offset*, which a linear probe can read.

**Code.** [`log_mel`, `mel_filterbank`](../src/audio_probes/audio.py). A test checks it
matches Hugging Face's `WhisperFeatureExtractor` to within 0.001
([`test_log_mel_matches_whisper_feature_extractor`](../tests/test_pitch.py)).

---

## 3. The Whisper encoder and pooling

**Plain words.** Whisper's encoder is a stack of 12 transformer blocks (for whisper-small).
Each block takes a sequence of vectors, one per 20 ms of audio, and rewrites them. We save
the sequence after the input layer (layer 0) and after every block (layers 1 to 12). To
get *one* vector per clip per layer we average over time. The catch: Whisper always pads
input to 30 seconds with silence. Our clips are 5 seconds, so 25 of the 30 seconds are
padding. Averaging over the padding would drown the real signal, so we only average the
frames that cover real audio.

**Math.** The encoder outputs 50 frames per second (100 mel frames per second, halved by a
stride-2 convolution). For a clip of S samples at rate sr:

    T_real = ceil(S / sr * 50)        (5 s clip -> 250 of 1500 frames)
    v_layer = (1 / T_real) * sum_{t < T_real} h_layer[t]          (mean pooling)

Alternatives behind the `encoder.pooling` knob: `max` (v_d = max_t h[t, d]) and `first_n`.

**Code.** [`n_real_frames`, `pool`, `WhisperEncoder.layers_batch`](../src/audio_probes/encoders/whisper.py),
stage driver [`embed.extract`](../src/audio_probes/embed.py). Tests in
[`tests/test_pooling.py`](../tests/test_pooling.py) put huge values in the padding and
check they never reach the pooled vector.

**Observability.** For each layer we plot all clips in 2D using PCA (principal component
analysis: center the vectors, take the SVD X = U S V^T, and project onto the two directions
with the most spread). If the colours (labels) form a smooth gradient, the property is
easy to read from that layer. [`pca_2d`](../src/audio_probes/embed.py).

---

## 4. Linear probes

**Plain words.** A probe is a deliberately simple model (a weighted sum) trained to read the
label out of one layer's vectors. If even a weighted sum can recover the label, the
information is there and easily accessible. We use simple probes on purpose: a big
model could compute the answer itself instead of finding it.

### The train/test split must be grouped

If the same room (or the same speaker) appears in both training and test clips, a probe
can "cheat" by recognising the room rather than measuring distance. So clips that share a
room or a source are tied together, and whole groups go to one side. Concretely, each clip
gets a key like `room:r12|source:spk84`; clips sharing any tag are merged into one
connected component (union-find), and components are shuffled and moved into the test set
until it holds about 20% of clips.
[`connected_components`, `grouped_split`](../src/audio_probes/split.py), tested in
[`tests/test_split.py`](../tests/test_split.py).

### Ridge regression (the number probe)

**Plain words.** Find weights so that "weights times features" is as close as possible to
the label, while keeping the weights small so the probe cannot memorise noise.

**Math.** Standardise each feature (subtract its mean, divide by its standard deviation,
both computed on train). Then

    w = argmin_w ||z - X w||^2 + alpha ||w||^2
      = (X^T X + alpha I)^(-1) X^T z          (closed form)

where z is the label (or log-label: distance and pitch use log, because "twice as far"
matters more than "one metre further"). The `alpha ||w||^2` term is the regulariser: adding
alpha to the diagonal of X^T X makes the inverse stable when features are correlated and
shrinks weights toward zero. Bigger alpha = simpler, smoother probe. alpha is chosen from a
grid by grouped 5-fold cross-validation inside the training set only.
[`make_regressor`, `grouped_cv_pick`, `fit_probe_pair`](../src/audio_probes/probes.py).

### Logistic regression (the multiple-choice probe)

**Plain words.** Same idea, but the probe picks one of 4 bins (for distance: under 1 m,
1 to 3 m, 3 to 6 m, over 6 m), so its score is directly comparable to an LLM answering
a multiple-choice question.

**Math.** Scores s_k = w_k . x + b_k for each bin k, turned into probabilities by softmax
p(k | x) = exp(s_k) / sum_j exp(s_j). Training minimises cross-entropy
-sum_i log p(y_i | x_i) plus ||W||^2 / (2C); smaller C means stronger regularisation.
[`make_classifier`, `to_bins`](../src/audio_probes/probes.py).

### Metrics

- **R^2** = 1 - sum (z - z_hat)^2 / sum (z - mean z)^2, on the probe's (possibly log)
  scale. 1 is perfect, 0 is "no better than always guessing the average", negative is worse
  than that. [`r2_score`](../src/audio_probes/probes.py).
- **Spearman rho**: correlation between the *rank orders* of truth and prediction. It asks
  "does it at least sort clips correctly?", ignoring scale. [`spearman`](../src/audio_probes/probes.py).
- **MAE** (mean absolute error) = mean |y - y_hat| in real units (Hz, metres, dB), after
  undoing the log. [`mae`](../src/audio_probes/probes.py).
- **4-bin accuracy**: fraction of test clips put in the right bin.
  [`accuracy`](../src/audio_probes/probes.py).
- **Chance**: always predict the training average (R^2 about 0) or the most common
  training bin. [`chance_scores`](../src/audio_probes/probes.py).
