"""Tamper/boundary fixtures for the read-only uploaded-job evidence gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
from pulsemorph.agentic.app.uploads import StudioMusic

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/agentic/app/audit_uploaded_job.py"
SPEC = importlib.util.spec_from_file_location("uploaded_job_audit_fixture", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class UploadedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        smoke = self.root / "artifacts/runs/agentic/app_user_music_v1"
        smoke.mkdir(parents=True)
        self.source = smoke / "procedural fixture.wav"
        times = np.arange(44100 * 8) / 44100
        signal = .08 * np.sin(2 * np.pi * 220 * times) + .6 * np.exp(-((times - 5.) / .025) ** 2) * np.sin(2 * np.pi * 1100 * times)
        with wave.open(str(self.source), "wb") as stream:
            stream.setparams((2, 2, 44100, 352800, "NONE", "not compressed"))
            stream.writeframes(np.rint(np.column_stack([signal, signal * .8]) * 32767).astype("<i2").tobytes())
        self.descriptor = StudioMusic(self.root).import_wav(self.source.read_bytes(), "Audit fixture", 2.)
        self.witness = smoke / "source_input.json"
        self.record = {"destination": self.source.relative_to(self.root).as_posix(), "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(), "channels": 2, "sample_rate": 44100, "duration_s": 8, "clip_start_s": 2, "clip_duration_s": 4}
        self.witness.write_text(json.dumps(self.record))

    def tearDown(self): self.temp.cleanup()

    def test_actual_pcm_descriptor_and_witness_crop_match_without_writes(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        public = {key: value for key, value in self.descriptor.items() if key != "path"}
        result, descriptor = AUDIT.audit_music({"music": public}, self.root)
        self.assertEqual(result["state"], "verified")
        witness = AUDIT.audit_source_witness(descriptor, self.root)
        self.assertEqual(witness["reconstructed_derived_sha256"], self.descriptor["sha256"])
        self.assertEqual(witness["source_crop_s"], [2, 6])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        public["accent_time_s"] += .1
        with self.assertRaisesRegex(ValueError, "descriptor differs"):
            AUDIT.audit_music({"music": public}, self.root)

    def test_changed_full_source_hash_rejected(self):
        self.source.write_bytes(self.source.read_bytes() + b"tampered fixture")
        with self.assertRaisesRegex(ValueError, "source witness hash differs"):
            AUDIT.audit_source_witness(self.descriptor, self.root)

    def test_witness_cannot_read_arbitrary_outside_path(self):
        self.record["destination"] = "outside.wav"
        self.witness.write_text(json.dumps(self.record))
        with patch.object(AUDIT, "sha", side_effect=AssertionError("must not read outside file")) as digest:
            with self.assertRaisesRegex(ValueError, "escapes smoke directory"):
                AUDIT.audit_source_witness(self.descriptor, self.root)
            digest.assert_not_called()


if __name__ == "__main__": unittest.main()
