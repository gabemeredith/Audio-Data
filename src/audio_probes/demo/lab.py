"""The demo's engine: build one clip from slider values, then ask every layer's trained probe
what it thinks. Whisper and the probes are loaded once and kept warm."""

from __future__ import annotations

import base64
import io
import pickle
import threading

import numpy as np
import soundfile as sf

from audio_probes.audio import (WHISPER_SR, band_level_dbfs, band_limited_noise, db_to_amp, fade,
                                resample_like_whisper, set_rms_dbfs)
from audio_probes.config import load_config, probe_dir
from audio_probes.data import apply_condition
from audio_probes.probes import inverse_label
from audio_probes.properties.distance import librispeech_index, place_pair, render_clip, sabine_rt60, say_voices
from audio_probes.properties.pitch import autocorr_f0, harmonic_tone, random_timbre


class Lab:
    def __init__(self):
        from audio_probes.encoders.whisper import build_encoder

        self.cfgs = {name: load_config(name) for name in ("pitch", "highband", "distance")}
        self.encoder = build_encoder(self.cfgs["pitch"])  # all properties share the same encoder settings
        self.lock = threading.Lock()  # one Whisper call at a time
        self.probes = {}
        for name, cfg in self.cfgs.items():
            for condition in cfg["conditions"]:
                path = probe_dir(cfg, condition) / "models.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        self.probes[(name, condition)] = pickle.load(f)
        if not self.probes:
            raise FileNotFoundError("No trained probes found; run `make all` first")

    # ---- info for the UI ---------------------------------------------------------------
    def info(self) -> dict:
        d = self.cfgs["distance"]["data"]
        try:
            speakers = list(librispeech_index(d["librispeech_dir"]))
        except FileNotFoundError:
            speakers = []
        return {
            "probes": [f"{p}/{c}" for p, c in self.probes],
            "bins": {name: cfg["probe"]["bin_edges"] for name, cfg in self.cfgs.items()},
            "speakers": speakers, "voices": say_voices()[:30],
            "device": self.encoder.device,
        }

    # ---- clip builders ------------------------------------------------------------------
    def pitch_clip(self, p: dict) -> tuple[np.ndarray, float, dict]:
        rng = np.random.default_rng(int(p.get("seed", 0)))
        f0, n = float(p["f0"]), int(WHISPER_SR * 5)
        weights, phases = random_timbre(rng, f0, 7500.0, [float(p.get("decay", 1.0))] * 2)
        audio = fade(set_rms_dbfs(harmonic_tone(f0, weights, phases, WHISPER_SR, n), -25.0), WHISPER_SR)
        return audio, f0, {"harmonics": len(weights), "autocorrelation_f0_hz": round(autocorr_f0(audio, WHISPER_SR), 1)}

    def highband_clip(self, p: dict) -> tuple[np.ndarray, float, dict]:
        rng = np.random.default_rng(int(p.get("seed", 0)))
        sr48, n = 48000, 48000 * 5
        f0 = float(np.exp(rng.uniform(np.log(100), np.log(1000))))
        weights, phases = random_timbre(rng, f0, 7000.0, [0.0, 2.0])
        base = set_rms_dbfs(harmonic_tone(f0, weights, phases, sr48, n), 0) + \
            set_rms_dbfs(band_limited_noise(rng, n, sr48, 50, 7000), -10)
        base = set_rms_dbfs(base, float(p["base_level"]))
        band_level = float(p["band_level"])
        mix = fade(base + set_rms_dbfs(band_limited_noise(rng, n, sr48, 9000, 15000), band_level), sr48)
        audio = resample_like_whisper(mix, sr48)[: WHISPER_SR * 5]
        audio = audio + db_to_amp(-70) * rng.standard_normal(len(audio)).astype(np.float32)
        return audio, band_level, {"leak_7k5_8k_dbfs_after_resampling": round(band_level_dbfs(audio, WHISPER_SR, 7500, 8000), 1),
                                   "note": "the 9-15 kHz band is removed by resampling to 16 kHz; you cannot hear it here either"}

    def distance_clip(self, p: dict) -> tuple[np.ndarray, float, dict]:
        d = self.cfgs["distance"]["data"]
        seed = int(p.get("seed", 0))
        dims = [float(p["room_l"]), float(p["room_w"]), float(p["room_h"])]
        distance = float(p["distance"])
        pair = place_pair(np.random.default_rng(seed), dims, distance, d["wall_margin_m"])
        if pair is None:
            raise ValueError(f"A {distance:.1f} m gap does not fit in a {dims[0]:.0f} x {dims[1]:.0f} x "
                             f"{dims[2]:.1f} m room (with 0.3 m from every wall). Make the room bigger.")
        source_type, source = self._source(p.get("source", "speech"), seed)
        spec = {"id": "demo", "seed": [seed, 0], "distance": distance, "room_id": "demo", "dims": dims,
                "absorption": float(p["absorption"]), "mic": pair[0].tolist(), "src": pair[1].tolist(),
                "source_type": source_type, "source": source, "cfg": d}
        clip = render_clip(spec)
        return clip.audio, distance, {
            "rt60_s": round(sabine_rt60(dims, float(p["absorption"])), 2), "drr_db": round(clip.meta["drr_db"], 1),
            "source": source["source_id"], "mic_pos": [round(v, 2) for v in spec["mic"]],
            "src_pos": [round(v, 2) for v in spec["src"]]}

    def _source(self, choice: str, seed: int) -> tuple[str, dict]:
        if choice.startswith("say:"):
            voice = choice[4:]
            return "say", {"source_id": f"say_{voice}", "voice": voice}
        if choice.startswith("speech"):
            index = librispeech_index(self.cfgs["distance"]["data"]["librispeech_dir"])
            spk = choice.split(":", 1)[1] if ":" in choice else list(index)[seed % len(index)]
            return "speech", {"source_id": f"spk{spk}", "files": index[spk]}
        if choice in ("noise", "tone"):
            return choice, {"source_id": f"{choice}_demo{seed}", "pattern_seed": [seed, 0]}
        raise ValueError(f"unknown source {choice!r}")

    # ---- probing ------------------------------------------------------------------------
    def predict(self, audio: np.ndarray, properties: list[tuple[str, str]]) -> dict:
        with self.lock:
            pooled = self.encoder.layers_batch([audio], WHISPER_SR)
        out = {}
        for name, condition in properties:
            models = self.probes.get((name, condition))
            if models is None:
                continue
            cfg = self.cfgs[name]
            rows = []
            for layer_id, vec in zip(sorted(models), pooled):
                reg, clf = models[layer_id]
                value = float(inverse_label(reg.predict(vec), cfg["probe"]["label_transform"])[0])
                probs = clf.predict_proba(vec)[0] if hasattr(clf, "predict_proba") else None
                rows.append({"layer": int(layer_id), "value": value, "bin": int(clf.predict(vec)[0]),
                             "bin_probs": probs.tolist() if probs is not None else None})
            out[f"{name}/{condition}"] = rows
        return out

    def run(self, kind: str, params: dict) -> dict:
        """Build a clip from slider values and probe it."""
        builders = {"pitch": self.pitch_clip, "highband": self.highband_clip, "distance": self.distance_clip}
        audio, label, extra = builders[kind](params)
        condition = params.get("condition", "raw")
        heard = apply_condition(audio, condition, self.cfgs[kind])
        return {"label": label, "extra": extra, **media(heard),
                "predictions": self.predict(heard, [(kind, condition)])}

    def run_file(self, audio: np.ndarray, condition: str = "raw") -> dict:
        audio = audio[: WHISPER_SR * 5]
        wanted = [("pitch", "raw"), ("highband", "raw"), ("distance", condition)]
        heard = apply_condition(audio, condition, self.cfgs["distance"]) if condition != "raw" else audio
        return {**media(heard), "predictions": self.predict(heard, wanted)}

    def sweep(self, params: dict, distances: list[float]) -> list[dict]:
        """Same room and source, source moved to each distance; one prediction set per step."""
        results = []
        for dist in distances:
            try:
                r = self.run("distance", {**params, "distance": dist})
                results.append({"distance": dist, "predictions": r["predictions"], "rms_dbfs": r["rms_dbfs"]})
            except ValueError as err:
                results.append({"distance": dist, "error": str(err)})
        return results


def media(audio: np.ndarray) -> dict:
    """Base64 WAV (16-bit, for the browser) and spectrogram PNG, plus the level."""
    from audio_probes.inspect_view import _spectrogram_png64

    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), WHISPER_SR, format="WAV", subtype="PCM_16")
    level = float(20 * np.log10(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + 1e-12))
    return {"wav_b64": base64.b64encode(buf.getvalue()).decode(), "spec_b64": _spectrogram_png64(audio, WHISPER_SR),
            "rms_dbfs": round(level, 1), "peak": round(float(np.max(np.abs(audio))), 3)}

