"""Prepare an owned eight-second two-section soundtrack, without model/physics.

The second half is louder by construction. Measured PCM features, not authored
intent, are supplied to the experimental phrase planner. No tempo inference.
"""
from pathlib import Path
import hashlib
import io
import json
import sys
import wave

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import numpy as np
    from pulsemorph.agentic.app.music import _synthesize, _analyze

    out = ROOT / 'artifacts/runs/agentic/phrase_program_music_v1'
    out.mkdir(parents=True, exist_ok=False)
    source = _synthesize('pulse')
    with wave.open(io.BytesIO(source), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 48000, 192000):
            raise ValueError('original four-second PCM16 source changed')
        decoded = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    samples = np.r_[np.rint(decoded.astype(float) * .45),
                    np.rint(decoded.astype(float) * .90)].astype('<i2')
    path = out / 'two_section_original.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, 48000, 384000, 'NONE', 'not compressed'))
        wav.writeframes(samples.tobytes())
    # Decode the file again, so all following measurements bind stored samples.
    with wave.open(str(path), 'rb') as wav:
        observed = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    if observed.tobytes() != samples.tobytes():
        raise ValueError('stored PCM differs')
    measured = _analyze(path)
    sections = []
    for start, end in ((0., 4.), (4., 8.)):
        values = observed[round(start * 48000):round(end * 48000)].astype(float) / 32768.
        bins = values.reshape(-1, 480)
        rms = np.sqrt(np.mean(bins ** 2, axis=1))
        sections.append({'start_s': start, 'end_s': end,
                         'rms': float(np.sqrt(np.mean(values ** 2))),
                         'peak_amplitude': float(np.max(np.abs(values))),
                         'peak_10ms_rms': float(rms.max()),
                         'peak_10ms_center_s': start + (int(rms.argmax()) + .5) / 100})
    summary = {'schema': 'pulsemorph.phrase_music_summary.v1',
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'duration_s': measured['duration_s'], 'sample_rate': 48000,
               'channels': 1, 'pcm_bits': 16, 'sample_frames': len(observed),
               'rms': measured['rms'], 'peak_amplitude': measured['peak_amplitude'],
               'accent_time_s': measured['accent_time_s'],
               'accent_measurement': measured['accent_measurement'],
               'onset_times_s': measured['onset_times_s'], 'sections': sections,
               'bpm': None, 'bpm_source': 'not inferred',
               'rights': 'Original procedural composition/synthesis; no external samples.',
               'limitations': ['PCM energy/onsets do not measure human musical preference.',
                               'Fixed transition time is not synchronized by this program tool.']}
    receipt = {'summary': summary, 'wav_path': str(path.relative_to(ROOT)),
               'source_pcm_sha256': hashlib.sha256(source).hexdigest(),
               'generator_sha256': hashlib.sha256((ROOT / 'src/pulsemorph/agentic/app/music.py').read_bytes()).hexdigest(),
               'preparation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'construction': 'Two original pulse takes, first PCM gain .45 and second .90, no reordering, no external audio.',
               'new_llm_calls': 0, 'onnx_calls': 0, 'physics_steps': 0}
    for name, value in (('summary.json', summary), ('receipt.json', receipt), ('analysis.json', measured)):
        with (out / name).open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'output': str(out), 'sha256': summary['sha256'],
                      'duration_s': summary['duration_s'], 'sections': sections}))


if __name__ == '__main__':
    main()
