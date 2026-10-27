"""Real PCM tests for original demo music; no external audio or model is used."""
from array import array
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
import wave

from pulsemorph.agentic.app.music import AppMusic, RATE


class MusicTests(unittest.TestCase):
    def test_catalog_does_not_create_assets_and_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            music = AppMusic(temporary)
            self.assertEqual([item["id"] for item in music.catalog()], ["pulse", "syncopated", "late_rise"])
            self.assertFalse(music.directory.exists())
            with self.assertRaises(KeyError):
                music.get("../outside")

    def test_real_pcm_peak_measurements_and_musical_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            music = AppMusic(temporary)
            for identity, target in [("pulse", 2.4), ("syncopated", 30 / 11), ("late_rise", 3.)]:
                record = music.get(identity)
                with wave.open(record["path"], "rb") as wav:
                    self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()), (48000, 1, 2, 192000))
                    pcm = array("h", wav.readframes(wav.getnframes()))
                energies = [sum(v * v for v in pcm[start:start + 480]) / 480 for start in range(0, len(pcm), 480)]
                peak = (max(range(len(energies)), key=energies.__getitem__) + .5) / 100
                self.assertAlmostEqual(record["accent_time_s"], peak)
                self.assertLess(abs(peak - target), .025)
                self.assertTrue(1.6 <= peak <= 3.2)
                self.assertEqual(len(record["waveform128"]), 128)
                self.assertEqual(len(record["energy_profile"]), 400)
                self.assertGreater(record["rms"], .03)
                self.assertLess(record["peak_amplitude"], .96)
                self.assertTrue(all(any(abs(beat - onset) < .03 for onset in record["onset_times_s"]) for beat in record["beat_times_s"]))
                # Energy between percussion events confirms pitched/sustained content.
                between = pcm[round(.25 * RATE):round(.4 * RATE)]
                self.assertGreater(math.sqrt(sum((v / 32768) ** 2 for v in between) / len(between)), .008)
                self.assertIn("authored", record["bpm_source"])

    def test_initial_tracks_keep_their_original_pcm_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            music = AppMusic(temporary)
            expected = {"pulse": "82ae86f4c58c2f35050d6e8721f36f5b4306d3fb7c135ce41554218c7a29859f", "syncopated": "9f00e003d7053edea0319bcc7163b9841bfae19c02727afc5671a121b7498253"}
            for identity, digest in expected.items():
                self.assertEqual(music.get(identity)["sha256"], digest)
            self.assertIn("after initial app job", music.get("late_rise")["provenance"]["authorship_context"])

    def test_deterministic_reopen_and_tamper_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            music = AppMusic(temporary)
            first = music.get("pulse")
            path = Path(first["path"])
            timestamp = path.stat().st_mtime_ns
            self.assertEqual(first, AppMusic(temporary).get("pulse"))
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), first["sha256"])
            path.write_bytes(b"tampered fixture")
            with self.assertRaises(ValueError):
                music.get("pulse")
            self.assertEqual(path.read_bytes(), b"tampered fixture")


if __name__ == "__main__":
    unittest.main()
