"""Original deterministic demo soundtracks and measurements of their actual PCM.

All oscillators, envelopes and percussion are synthesized here. No samples,
external recordings, generative service, model, or copyrighted source is used.
"""
from __future__ import annotations

from array import array
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import threading
import wave

RATE = 48_000
DURATION = 4.0
_LOCK = threading.RLock()
_SPECS = {
    "pulse": {"id": "pulse", "label": "Amber Pulse", "description": "Warm minor chords, rounded bass and a steady electronic groove; a stronger percussion accent arrives near 2.4 seconds.", "bpm": 100, "accent_authored_s": 2.4},
    "syncopated": {"id": "syncopated", "label": "Velvet Offbeat", "description": "A light syncopated groove with plucked harmonies, bass and brushed noise percussion; its main accent arrives near 2.73 seconds.", "bpm": 110, "accent_authored_s": 30 / 11},
    "late_rise": {"id": "late_rise", "label": "Late Rise", "description": "Original warm chords, bass and steady percussion with an authored late accent near 3.0 seconds. Added after the initial app job as another soundtrack option; physical synchronization performance remains untested.", "bpm": 120, "accent_authored_s": 3.0},
}


@lru_cache(maxsize=3)
def _synthesize(identity):
    spec = _SPECS[identity]
    count = int(RATE * DURATION)
    sound = array("d", [0.0]) * count
    rng = random.Random(20261027 + {"pulse": 0, "syncopated": 1, "late_rise": 2}[identity])
    beat = 60 / spec["bpm"]
    onsets = [i * beat for i in range(int(DURATION / beat) + 1)]

    def tone(at, duration, frequency, amplitude, decay=5, overtone=.2):
        start = round(at * RATE)
        for k in range(min(round(duration * RATE), count - start)):
            t = k / RATE
            envelope = min(t / .008, 1) * math.exp(-decay * t) * min((duration - t) / .025, 1)
            sound[start + k] += amplitude * envelope * (math.sin(math.tau * frequency * t) + overtone * math.sin(math.tau * frequency * 2 * t))

    # Original A-minor and F-major voicings with a gentle tremolo bed.
    for at, chord in [(0., (220., 261.6256, 329.6276)), (2., (174.6141, 220., 261.6256))]:
        for frequency in chord:
            tone(at, 2., frequency, .033, decay=.9, overtone=.1)
    for index, at in enumerate(onsets):
        if at >= DURATION:
            continue
        tone(at, min(.48, DURATION - at), 55. if at < 2 else 43.6535, .12, decay=6)
        if identity == "syncopated" and at + beat * .75 < DURATION:
            tone(at + beat * .75, .18, (440., 523.2511, 659.2551)[index % 3], .047, decay=15)
        accented = abs(at - spec["accent_authored_s"]) < .001
        start = round(at * RATE)
        # A pitched kick plus noise snare creates real musical transients.
        for k in range(min(round(.21 * RATE), count - start)):
            t = k / RATE
            kick = math.sin(math.tau * (50 * t + 1.8 * (1 - math.exp(-38 * t)))) * math.exp(-23 * t)
            noise = rng.uniform(-1, 1) * math.exp(-48 * t)
            gain = .68 if accented else (.20 if index % 2 == 0 else .14)
            sound[start + k] += gain * (.72 * kick + .55 * noise)
    # Hi-hats on offbeats keep the loop musical without dominating the accent.
    for index in range(int(DURATION / (beat / 2))):
        at = (index + .5) * beat / 2
        start = round(at * RATE)
        for k in range(min(round(.045 * RATE), count - start)):
            sound[start + k] += .032 * rng.uniform(-1, 1) * math.exp(-95 * k / RATE)
    pcm = array("h")
    for index, value in enumerate(sound):
        fade = min(index / (RATE * .003), (count - 1 - index) / (RATE * .018), 1.)
        pcm.append(round(max(-.96, min(.96, value * fade)) * 32767))
    if sys.byteorder != "little":
        pcm.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, RATE, count, "NONE", "not compressed"))
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _analyze(path):
    """Decode stored WAV; report 10ms RMS bins and detected positive transients."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, RATE):
            raise ValueError("Unexpected soundtrack PCM format")
        samples = array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    hop = RATE // 100
    energy = [math.sqrt(sum((v / 32768.) ** 2 for v in samples[start:start + hop]) / len(samples[start:start + hop])) for start in range(0, len(samples), hop)]
    peak_index = max(range(len(energy)), key=energy.__getitem__)
    rises = [energy[i] - (energy[i - 1] if i else 0) for i in range(len(energy))]
    threshold = max(rises) * .18
    candidates = [i for i, rise in enumerate(rises) if rise >= threshold and rise >= (rises[i - 1] if i else 0) and rise >= (rises[i + 1] if i + 1 < len(rises) else 0)]
    selected = []
    for index in sorted(candidates, key=lambda i: rises[i], reverse=True):
        if all(abs(index - prior) >= 18 for prior in selected):
            selected.append(index)
    waveform = []
    for index in range(128):
        segment = samples[index * len(samples) // 128:(index + 1) * len(samples) // 128]
        waveform.append(round(max(abs(v) for v in segment) / 32768., 6))
    return {"duration_s": len(samples) / RATE, "accent_time_s": round((peak_index + .5) / 100, 3),
            "accent_measurement": "center of highest 10ms RMS window in decoded PCM; no target-window filtering",
            "onset_times_s": [round(i / 100, 3) for i in sorted(selected)],
            "energy_profile": [{"time_s": round((i + .5) / 100, 3), "rms": round(value, 6)} for i, value in enumerate(energy)],
            "waveform128": waveform, "rms": round(math.sqrt(sum((v / 32768.) ** 2 for v in samples) / len(samples)), 6),
            "peak_amplitude": round(max(abs(v) for v in samples) / 32768., 6), "sample_rate": RATE, "channels": 1, "pcm_bits": 16}


class AppMusic:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[4]).resolve()
        self.directory = self.root / "artifacts/media/agentic/app_music_v1"

    def catalog(self):
        return [{**spec, "duration_s": DURATION, "url": f"/media/music/{identity}.wav", "bpm_source": "authored; not an inferred tempo"} for identity, spec in _SPECS.items()]

    def ensure(self, identity=None):
        return self.get(identity) if identity is not None else [self.get(key) for key in _SPECS]

    def get(self, identity):
        if identity not in _SPECS:
            raise KeyError("Unknown original soundtrack")
        with _LOCK:
            expected = _synthesize(identity)
            digest = hashlib.sha256(expected).hexdigest()
            path = self.directory / f"{identity}.wav"
            self.directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("Existing soundtrack differs from deterministic source; preserved without overwrite")
            else:
                with path.open("xb") as target:
                    target.write(expected)
            spec = _SPECS[identity]
            result = {**spec, **_analyze(path), "path": str(path), "sha256": digest,
                      "url": f"/media/music/{identity}.wav", "bpm_source": "authored; not an inferred tempo",
                      "beat_times_s": [round(index * 60 / spec["bpm"], 6) for index in range(math.ceil(DURATION * spec["bpm"] / 60))],
                      "beat_times_source": "authored score grid; measured transients are onset_times_s",
                      "rights": "Original procedural composition and synthesis for PulseMorph; no external samples or recordings.",
                      "provenance": {"source": "src/pulsemorph/agentic/app/music.py", "generator": "original_oscillators_and_seeded_noise_v1", "external_sources": []}}
            if identity == "late_rise":
                result["provenance"]["authorship_context"] = "Added after initial app job as an explicit additional timing option; existing audio, failed-job evidence and fixed targets were not changed. No physical success is implied."
            manifest = path.with_suffix(".json")
            stable_manifest = dict(result, path=path.relative_to(self.root).as_posix())
            encoded = json.dumps(stable_manifest, indent=2, ensure_ascii=False)
            if manifest.exists():
                prior = json.loads(manifest.read_text(encoding="utf-8"))
                if prior != stable_manifest:
                    raise ValueError("Existing soundtrack manifest differs; preserved without overwrite")
            else:
                with manifest.open("x", encoding="utf-8") as target:
                    target.write(encoded)
            return result
