"""Target property: distance from the sound source to the microphone, in simulated rooms.

For each clip we build a random shoebox (box-shaped) room, put a microphone and a source
d metres apart inside it, compute the room impulse response (RIR: what a single click at the
source sounds like at the mic, echoes included) with the image-source method, and convolve a
dry source sound with it.

Two cues change with distance:
    loudness:              the direct sound falls off as 1/d (inverse-square law for power)
    direct-to-reverb ratio (DRR): the direct sound weakens with d while the room's reverb
                           stays roughly constant, so far sources sound more "roomy"
The loudness_normalized condition removes the first cue, leaving only the second (and
subtler ones such as air and wall high-frequency loss).

Train/test grouping: rooms and sources are partitioned into `n_groups` blocks and a clip only
combines a room and a source from the same block, so whole blocks can be held out.
"""

from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from audio_probes.audio import db_to_amp, fade, resample_like_whisper, set_rms_dbfs
from audio_probes.clip import Clip
from audio_probes.config import REPO_ROOT
from audio_probes.properties.base import Property, rms_dbfs
from audio_probes.properties.pitch import harmonic_tone, random_timbre

SPEED_OF_SOUND = 343.0  # m/s
SAY_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the quiet river bank.",
    "Please remember to bring the blue notebook to the meeting tomorrow morning.",
    "A gentle breeze moved through the tall grass as the sun began to set.",
    "Turn left at the second traffic light and the library is on your right.",
]


# ---- room acoustics -----------------------------------------------------------------

def direct_index(distance_m: float, sr: int) -> int:
    """Sample where the direct sound arrives in a pyroomacoustics RIR:
    its fractional-delay filter adds a fixed offset, then the sound travels d / c seconds."""
    import pyroomacoustics as pra
    return int(round(pra.constants.get("frac_delay_length") // 2 + distance_m / SPEED_OF_SOUND * sr))


def direct_to_reverb_db(rir: np.ndarray, distance_m: float, sr: int, window_ms: float = 2.5) -> float:
    """DRR = 10 log10( sum_{|n - n_d| <= w} h[n]^2  /  sum_{n > n_d + w} h[n]^2 ),
    where n_d is the direct-path sample and w = 2.5 ms."""
    n_d, w = direct_index(distance_m, sr), int(sr * window_ms / 1000)
    direct = np.sum(rir[max(0, n_d - w): n_d + w + 1] ** 2)
    reverb = np.sum(rir[n_d + w + 1:] ** 2)
    return float(10 * np.log10(direct / (reverb + 1e-20) + 1e-20))


def sabine_rt60(dims, absorption: float) -> float:
    """Sabine's formula: RT60 = 0.161 * V / (S * a) seconds (time for reverb to drop 60 dB)."""
    lx, ly, lz = dims
    volume, surface = lx * ly * lz, 2 * (lx * ly + lx * lz + ly * lz)
    return 0.161 * volume / (surface * absorption)


def simulate_rir(dims, absorption: float, mic, src, sr: int, max_order: int) -> np.ndarray:
    """Image-source method: each wall reflection is replaced by a mirror-image source behind
    the wall; the RIR is the sum of delayed, attenuated impulses
        h(t) = sum_images  (1 - a)^(n_reflections / 2) / r_i * delta(t - r_i / c).
    The 1 / r factor is the inverse-square law written for amplitude (power falls as 1 / r^2).
    (pyroomacoustics leaves out the constant 1 / (4 pi), so a source 1 m away arrives at gain ~1.)"""
    import pyroomacoustics as pra

    room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(float(absorption)), max_order=max_order)
    room.add_source(list(src))
    room.add_microphone(list(mic))
    room.compute_rir()
    return np.asarray(room.rir[0][0], dtype=np.float64)


def place_pair(rng: np.random.Generator, dims, distance_m: float, margin: float, tries: int = 300):
    """Random mic position and direction such that the source, d metres away, is inside the
    room with `margin` to every wall. Elevation is kept within +-30 degrees. None if impossible."""
    dims = np.asarray(dims)
    lo, hi = np.full(3, margin), dims - margin
    for _ in range(tries):
        mic = rng.uniform(lo, hi)
        mic[2] = rng.uniform(max(margin, 1.0), min(hi[2], 2.0))  # ear / desk height
        az, el = rng.uniform(0, 2 * np.pi), rng.uniform(-np.pi / 6, np.pi / 6)
        src = mic + distance_m * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        if np.all(src >= lo) and np.all(src <= hi):
            return mic, src
    return None


# ---- dry sources --------------------------------------------------------------------

@lru_cache(maxsize=1)
def librispeech_index(root: str) -> dict[str, list[str]]:
    """speaker id -> sorted list of utterance files at least 5.5 s long."""
    base = (REPO_ROOT / root) if not Path(root).is_absolute() else Path(root)
    if not base.exists():
        raise FileNotFoundError(f"LibriSpeech not found at {base}. Run `make download-librispeech`, "
                                "or use data.source_types=[noise,say] to skip speech.")
    index = {}
    for spk in sorted(p.name for p in base.iterdir() if p.is_dir()):
        files = sorted(str(f) for f in (base / spk).rglob("*.flac") if sf.info(str(f)).duration >= 5.5)
        if files:
            index[spk] = files
    return index


@lru_cache(maxsize=1)
def say_voices() -> list[str]:
    """English macOS text-to-speech voices (empty off macOS)."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    voices = []
    for line in out.splitlines():
        name, _, rest = line.partition("#")[0].rstrip().rpartition(" ")
        if rest.startswith("en_") and name.strip():
            voices.append(name.strip())
    return voices


def say_audio(voice: str, text: str, sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "say.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(path), text], check=True)
        audio, file_sr = sf.read(path, dtype="float32")
    return resample_like_whisper(audio if audio.ndim == 1 else audio.mean(1), file_sr, "scipy") if file_sr != sr else audio


def noise_bursts(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """Bursts of coloured noise: white noise shaped by a random spectral tilt, gated on and off
    by a random envelope (bursts of 50-400 ms), so there are onsets and decays to hear the room in."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spectrum *= (np.maximum(freqs, 50) / 1000.0) ** rng.uniform(-1.0, 0.0)
    noise = np.fft.irfft(spectrum, n)
    env, t = np.zeros(n), 0
    while t < n:
        on, off = int(rng.uniform(0.05, 0.4) * sr), int(rng.uniform(0.05, 0.5) * sr)
        env[t:t + on] = 1.0
        t += on + off
    return noise * np.convolve(env, np.hanning(int(0.01 * sr)) / np.sum(np.hanning(int(0.01 * sr))), "same")


def pick_segment(audio: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) <= n:
        return np.pad(audio, (0, n - len(audio)))
    start = int(rng.integers(0, len(audio) - n))
    return audio[start:start + n]


# ---- the property -------------------------------------------------------------------

class Distance(Property):
    name = "distance"
    label_unit = "m"

    def generate(self, n: int, seed: int) -> list[Clip]:
        specs = self.plan(n, seed)
        workers = self.data_cfg.get("workers", 1)
        if workers > 1 and n > 8:
            with ProcessPoolExecutor(workers) as pool:
                return list(pool.map(render_clip, specs, chunksize=8))
        return [render_clip(s) for s in specs]

    def plan(self, n: int, seed: int) -> list[dict]:
        """Decide every clip's room, positions and source up front (cheap), so rendering can
        run in parallel and is fully determined by the seed."""
        d = self.data_cfg
        n_groups, per_room = d["n_groups"], d["clips_per_room"]
        pools = self._source_pools(n_groups)
        rooms: dict[str, dict] = {}
        specs = []
        for i in range(n):
            rng = np.random.default_rng([seed, i])
            block = i % n_groups
            room_id = f"b{block}_r{(i // n_groups) // per_room}"
            distance = float(np.exp(rng.uniform(*np.log(d["distance_range_m"]))))
            room = rooms.get(room_id) or self._new_room(np.random.default_rng([seed, 10_000 + block, len(rooms)]))
            pair = place_pair(rng, room["dims"], distance, d["wall_margin_m"])
            attempt = 0
            while pair is None:  # this distance does not fit; give the clip its own new room
                attempt += 1
                room_id = f"b{block}_r{(i // n_groups) // per_room}_x{i}_{attempt}"
                room = self._new_room(rng)
                pair = place_pair(rng, room["dims"], distance, d["wall_margin_m"])
            rooms.setdefault(room_id, room)
            source_type = d["source_types"][int(rng.integers(len(d["source_types"])))]
            source = pools[source_type][block][int(rng.integers(len(pools[source_type][block])))]
            specs.append({"id": f"distance_{i:05d}", "seed": [seed, i], "distance": distance, "room_id": room_id,
                          **room, "mic": pair[0].tolist(), "src": pair[1].tolist(), "source_type": source_type,
                          "source": source, "cfg": d})
        return specs

    def _new_room(self, rng: np.random.Generator) -> dict:
        d = self.data_cfg
        dims = [float(rng.uniform(lo, hi)) for lo, hi in d["room_size_range_m"]]
        return {"dims": dims, "absorption": float(rng.uniform(*d["absorption_range"]))}

    def _source_pools(self, n_groups: int) -> dict[str, list[list[dict]]]:
        """For each source type, a list (one per block) of the sources that block may use.
        Speakers / voices are dealt round-robin into blocks so no source spans two blocks."""
        d = self.data_cfg
        pools: dict[str, list[list[dict]]] = {}
        types = set(d["source_types"])
        if "speech" in types:
            index = librispeech_index(d["librispeech_dir"])
            pools["speech"] = [[] for _ in range(n_groups)]
            for k, spk in enumerate(index):
                pools["speech"][k % n_groups].append({"source_id": f"spk{spk}", "files": index[spk]})
        if "say" in types:
            voices = say_voices()
            if not voices:
                raise RuntimeError("source type 'say' needs macOS text-to-speech voices")
            pools["say"] = [[] for _ in range(n_groups)]
            for k, voice in enumerate(voices):  # fewer voices than blocks -> blocks share voices
                pools["say"][k % n_groups].append({"source_id": f"say_{voice}", "voice": voice})
            for b in range(n_groups):
                pools["say"][b] = pools["say"][b] or pools["say"][b % len(voices)]
        for kind in ("noise", "tone"):
            if kind in types:
                pools[kind] = [[{"source_id": f"{kind}_b{b}_k{k}", "pattern_seed": [b, k]} for k in range(4)]
                               for b in range(n_groups)]
        return pools

    def split_key(self, clip: Clip) -> str:
        return f"room:{clip.meta['room_id']}|source:{clip.meta['source_id']}"

    def baseline_features(self, clip: Clip) -> np.ndarray:
        """RMS level of the clip as heard, and the DRR measured from the true room impulse
        response (an oracle: a real system would have to estimate it blindly)."""
        return np.array([rms_dbfs(clip.audio), clip.meta["drr_db"]])

    def baseline_description(self) -> str:
        return "RMS + oracle DRR"


def dry_source(spec: dict, sr: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """The source sound before it enters the room, RMS-normalised to data.source_dbfs."""
    kind, source = spec["source_type"], spec["source"]
    if kind == "speech":
        path = source["files"][int(rng.integers(len(source["files"])))]
        audio, file_sr = sf.read(path, dtype="float32")
        assert file_sr == sr, "LibriSpeech is 16 kHz"
        audio = pick_segment(audio, n, rng)
    elif kind == "say":
        text = SAY_SENTENCES[int(rng.integers(len(SAY_SENTENCES)))]
        audio = pick_segment(say_audio(source["voice"], text, sr), n, rng)
    elif kind == "noise":
        audio = noise_bursts(np.random.default_rng([spec["seed"][0], *source["pattern_seed"]]), n, sr)
    elif kind == "tone":
        pat = np.random.default_rng([spec["seed"][0], *source["pattern_seed"]])
        f0 = float(np.exp(pat.uniform(np.log(100), np.log(1000))))
        weights, phases = random_timbre(pat, f0, 7500.0, [0.0, 2.0])
        audio = harmonic_tone(f0, weights, phases, sr, n)
    else:
        raise ValueError(f"unknown source type {kind!r}")
    return set_rms_dbfs(fade(audio, sr), spec["cfg"]["source_dbfs"])


def render_clip(spec: dict) -> Clip:
    """Dry source -> room impulse response -> convolution -> absolute gain -> mic noise floor."""
    d = spec["cfg"]
    sr, n = d["sr"], int(d["sr"] * d["duration_s"])
    rng = np.random.default_rng(spec["seed"])
    dry = dry_source(spec, sr, n, rng)
    rir = simulate_rir(spec["dims"], spec["absorption"], spec["mic"], spec["src"], sr, d["max_order"])
    # The RIR already falls off as 1/d with gain ~1 at 1 m; this fixed gain puts the direct sound
    # at level_at_1m_dbfs for a source 1 m away, the same for every clip (so loudness carries distance).
    gain = db_to_amp(d["level_at_1m_dbfs"] - d["source_dbfs"])
    wet = gain * fftconvolve(dry, rir)[:n]
    wet = wet + db_to_amp(d["noise_floor_dbfs"]) * rng.standard_normal(n)
    peak = float(np.max(np.abs(wet)))
    audio = np.clip(wet, -1, 1).astype(np.float32)
    return Clip(spec["id"], audio, sr, spec["distance"], {
        "room_id": spec["room_id"], "source_id": spec["source"]["source_id"], "source_type": spec["source_type"],
        "distance_m": spec["distance"], "room_dims_m": spec["dims"], "absorption": spec["absorption"],
        "rt60_sabine_s": sabine_rt60(spec["dims"], spec["absorption"]), "mic_pos": spec["mic"],
        "src_pos": spec["src"], "drr_db": direct_to_reverb_db(rir, spec["distance"], sr),
        "peak": peak, "clipped": peak > 1.0,
    })
