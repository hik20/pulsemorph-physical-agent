"""Actual generated PCM fixtures for local WAV import; no network/model/physics."""
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from pulsemorph.agentic.app.uploads import StudioMusic, MAX_BYTES, _canonical


def fixture(rate=44100, channels=2, duration=7., accent=3.6, silent=False):
    times = np.arange(round(rate * duration)) / rate
    signal = np.zeros_like(times) if silent else .08 * np.sin(2 * np.pi * 220 * times) + .65 * np.exp(-((times - accent) / .025) ** 2) * np.sin(2 * np.pi * 1100 * times)
    values = np.column_stack([signal, signal * .8]) if channels == 2 else signal
    pcm = np.rint(values * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((channels, 2, rate, len(times), "NONE", "not compressed"))
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.music = StudioMusic(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_stereo_44100_crop_actual_pcm_and_provenance(self):
        data = fixture()
        record = self.music.import_wav(data, "My recording", 1.2)
        with wave.open(record["path"], "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()), (1, 2, 48000, 192000))
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(float)
        energies = np.mean(pcm.reshape(-1, 480) ** 2, axis=1)
        measured = (np.argmax(energies) + .5) / 100
        self.assertAlmostEqual(record["accent_time_s"], measured)
        self.assertLess(abs(measured - 2.4), .025)
        self.assertIsNone(record["bpm"])
        self.assertEqual(record["beat_times_s"], [])
        self.assertEqual(len(record["waveform128"]), 128)
        self.assertEqual(len(record["energy_profile"]), 400)
        self.assertEqual(record["provenance"]["source_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(record["provenance"]["original_frame_count"], 7 * 44100)
        self.assertEqual(record["provenance"]["read_frame_count"], 7 * 44100)
        self.assertEqual(record["provenance"]["start_frame"], 52920)
        self.assertFalse(record["provenance"]["source_bytes_retained"])
        self.assertEqual(len(list(self.music.directory.iterdir())), 2)
        self.assertIn("unknown", record["rights"])

    def test_early_accent_is_measured_without_hidden_window_filter(self):
        record = self.music.import_wav(fixture(rate=8000, channels=1, duration=4, accent=.35), "Early", 0)
        self.assertLess(record["accent_time_s"], .4)

    def test_idempotency_reopen_and_catalog(self):
        data = fixture(rate=96000, channels=1, duration=5)
        first = self.music.import_wav(data, "First label", .5)
        paths = list(self.music.directory.iterdir())
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        self.assertEqual(first, self.music.import_wav(data, "Other label", .5))
        reopened = StudioMusic(self.root)
        self.assertEqual(first, reopened.get(first["id"]))
        self.assertEqual([item["id"] for item in reopened.catalog()][:3], ["pulse", "syncopated", "late_rise"])
        self.assertIn(first["id"], [item["id"] for item in reopened.catalog()])
        self.assertTrue(all((p.read_bytes(), p.stat().st_mtime_ns) == before[p.name] for p in paths))

    def test_truncated_and_misaligned_samples_rejected(self):
        data = fixture(rate=8000, channels=1, duration=4)
        for invalid in [data[:-2], data + b"x", b"not a wave" * 10]:
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                self.music.import_wav(invalid, "Bad", 0)
        invalid = bytearray(data[:-1])
        struct.pack_into("<I", invalid, 4, len(invalid) - 8)
        struct.pack_into("<I", invalid, 40, len(invalid) - 44)
        with self.assertRaises(ValueError): self.music.import_wav(bytes(invalid), "Misaligned", 0)
        self.assertFalse(self.music.directory.exists())

    def test_invalid_formats_duration_size_and_start_rejected(self):
        data = fixture(rate=8000, channels=1, duration=4)
        for start in [float("nan"), float("inf"), -.1, True, "0", .001]:
            with self.subTest(start=start), self.assertRaises(ValueError): self.music.import_wav(data, "Bad", start)
        for invalid in [fixture(rate=8000, channels=1, duration=3.9), fixture(rate=8000, channels=1, duration=120.1), b"x" * (MAX_BYTES + 1)]:
            with self.assertRaises(ValueError): self.music.import_wav(invalid, "Bad", 0)
        for offset, value in [(20, 3), (22, 3), (34, 24)]:
            malformed = bytearray(data); struct.pack_into("<H", malformed, offset, value)
            with self.assertRaises(ValueError): self.music.import_wav(bytes(malformed), "Bad", 0)

    def test_silent_clip_rejected_without_writing(self):
        with self.assertRaisesRegex(ValueError, "sessiz"):
            self.music.import_wav(fixture(rate=8000, channels=1, duration=4, silent=True), "Silent", 0)
        self.assertFalse(self.music.directory.exists())

    def test_tamper_is_preserved_and_never_overwritten(self):
        data = fixture(rate=8000, channels=1, duration=4)
        record = self.music.import_wav(data, "Tamper", 0)
        wav = Path(record["path"]); wav.write_bytes(b"damaged fixture")
        for action in [lambda: self.music.get(record["id"]), lambda: self.music.import_wav(data, "Retry", 0)]:
            with self.assertRaises(ValueError): action()
        self.assertNotIn(record["id"], [item["id"] for item in self.music.catalog()])
        self.assertTrue(self.music.status()["warnings"])
        self.assertEqual(wav.read_bytes(), b"damaged fixture")

    def test_manifest_path_and_directory_boundary(self):
        data = fixture(rate=8000, channels=1, duration=4)
        record = self.music.import_wav(data, "Boundary", 0)
        manifest = Path(record["path"]).with_suffix(".json")
        altered = json.loads(manifest.read_text()); altered.pop("descriptor_sha256"); altered["path"] = str(self.root / "outside.wav")
        altered["descriptor_sha256"] = hashlib.sha256(_canonical(altered)).hexdigest()
        manifest.write_text(json.dumps(altered))
        with self.assertRaises(ValueError): self.music.get(record["id"])
        for identity in ["upload_../outside", "../outside", "upload_" + "a" * 24 + "/other"]:
            with self.assertRaises(KeyError): self.music.get(identity)
        original_resolve = Path.resolve
        with patch.object(Path, "resolve", autospec=True, side_effect=lambda p, *a, **k: self.root / "outside" if p == self.music.directory else original_resolve(p, *a, **k)):
            with self.assertRaises(ValueError): self.music.import_wav(data, "Boundary", 0)

    def test_labels_are_plain_text_not_paths(self):
        data = fixture(rate=8000, channels=1, duration=4)
        for label in ["../escape", "a\\b", "line\nfeed", "bidi\u202econtrol", "x" * 81, 12]:
            with self.subTest(label=label), self.assertRaises(ValueError): self.music.import_wav(data, label, 0)
        self.assertFalse(self.music.directory.exists())

    def test_failed_manifest_publish_preserves_bytes_without_blocking_valid_catalog(self):
        """Regression: an upload IO failure must not hide unrelated valid tracks."""
        data = fixture(rate=8000, channels=1, duration=5)
        valid = self.music.import_wav(data, "Valid before failure", 0)
        def interrupted_manifest(record, stream, **kwargs):
            stream.write('{"schema":')
            stream.flush()
            raise OSError("fixture simulated disk write interruption")
        with patch("pulsemorph.agentic.app.uploads.json.dump", side_effect=interrupted_manifest):
            with self.assertRaises(OSError):
                self.music.import_wav(data, "Interrupted fixture", .5)
        failed = {p: p.read_bytes() for p in self.music.directory.iterdir() if p.stem != valid["id"]}
        self.assertTrue(failed)
        self.assertTrue(all(p.name.startswith(".stage_") for p in failed))
        self.assertTrue(any(p.suffix == ".wav" for p in failed))
        self.assertTrue(any(p.suffix == ".json" and content == b'{"schema":' for p, content in failed.items()))
        # Intended behavior: valid catalog survives; broken upload stays excluded.
        catalog = self.music.catalog()
        self.assertIn(valid["id"], [item["id"] for item in catalog])
        self.assertEqual(len([item for item in catalog if item["id"].startswith("upload_")]), 1)
        self.assertTrue(all(p.read_bytes() == content for p, content in failed.items()))
        self.assertTrue(self.music.status()["warnings"])

    def test_canonical_wav_orphan_can_retry_only_verified_identical_bytes(self):
        data = fixture(rate=8000, channels=1, duration=4)
        original_link = os.link
        def interrupt_commit(source, destination):
            if Path(destination).suffix == ".json":
                raise OSError("fixture crash before manifest commit")
            return original_link(source, destination)
        with patch("pulsemorph.agentic.app.uploads.os.link", side_effect=interrupt_commit):
            with self.assertRaises(OSError): self.music.import_wav(data, "Orphan", 0)
        canonical = next(p for p in self.music.directory.glob("upload_*.wav"))
        self.assertFalse(canonical.with_suffix(".json").exists())
        original_bytes, timestamp = canonical.read_bytes(), canonical.stat().st_mtime_ns
        self.assertEqual(len(self.music.catalog()), 3)
        self.assertTrue(self.music.status()["warnings"])
        # Retry regenerates and verifies bytes, then commits metadata last.
        record = self.music.import_wav(data, "Verified recovery", 0)
        self.assertEqual(record["id"], canonical.stem)
        self.assertEqual(canonical.read_bytes(), original_bytes)
        self.assertEqual(canonical.stat().st_mtime_ns, timestamp)
        self.assertEqual(StudioMusic(self.root).get(record["id"]), record)

    def test_bad_partial_manifest_is_preserved_and_cannot_be_repaired_silently(self):
        data = fixture(rate=8000, channels=1, duration=4)
        valid = self.music.import_wav(data, "Corrupt metadata", 0)
        manifest = Path(valid["path"]).with_suffix(".json")
        manifest.write_bytes(b'{"schema":')
        with self.assertRaises(ValueError): self.music.import_wav(data, "Retry", 0)
        self.assertEqual(manifest.read_bytes(), b'{"schema":')
        self.assertEqual(len(self.music.catalog()), 3)
        self.assertTrue(any(item["id"] == valid["id"] for item in self.music.status()["warnings"]))


if __name__ == "__main__": unittest.main()
