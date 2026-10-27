"""Create a reproducible non-user-private WAV for the real browser import smoke."""
from pathlib import Path
import hashlib
import json
import wave

import numpy as np
from scipy.signal import resample_poly


def main():
    root = Path(__file__).resolve().parents[3]
    output = root / "artifacts/runs/agentic/app_user_music_v1"
    output.mkdir(parents=True, exist_ok=True)
    source = root / "artifacts/media/agentic/app_music_v1/late_rise.wav"
    with wave.open(str(source), "rb") as stream:
        assert stream.getframerate() == 48000 and stream.getnchannels() == 1
        mono = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float) / 32768
    converted = resample_poly(mono, 147, 160)
    timeline = np.zeros(44100*8)
    timeline[44100*2:44100*6] = converted
    # The two channels intentionally differ; import must actually downmix.
    stereo = np.stack([timeline * .9, timeline * .7], axis=1)
    destination = output / "Stereo 44k import smoke.wav"
    if destination.exists():
        raise FileExistsError(destination)
    with wave.open(str(destination), "wb") as stream:
        stream.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(np.rint(stereo*32767).astype("<i2").tobytes())
    receipt = {"source": str(source.relative_to(root)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "destination": str(destination.relative_to(root)), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "channels": 2, "sample_rate": 44100, "duration_s": 8, "clip_start_s": 2, "clip_duration_s": 4,
               "evidence_class": "original-procedural-audio-derived-browser-input; not private user media",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (output / "source_input.json").open("x", encoding="utf-8") as target:
        json.dump(receipt, target, indent=2)
    print(str(destination))


if __name__ == "__main__":
    main()
