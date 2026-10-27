"""Validation and integrity tests; live physical smoke evidence is recorded separately."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from pulsemorph.agentic.app.runtime import AppRuntime,evaluate_trial

ROOT=Path(__file__).resolve().parents[1]

class RuntimeTests(unittest.TestCase):
    def test_rejects_boolean_parameter(self):
        with self.assertRaisesRegex(ValueError,'outside app domain'):
            AppRuntime(ROOT).run_trial('wave',dict(left_gain_a=True,right_gain_a=1,left_gain_b=1,right_gain_b=1,timing_shift_s=0),'global_clock_v1',ROOT/'artifacts/runs/agentic/not_created','validation',0,'test')

    def test_rejects_unknown_tool(self):
        with self.assertRaisesRegex(ValueError,'unsupported'):
            AppRuntime(ROOT).run_trial('wave',{},'shell',ROOT/'artifacts/runs/agentic/not_created','validation',0,'test')

    def test_source_hash_allowlist_before_deserialization(self):
        spec=importlib.util.spec_from_file_location('app_execute_test',ROOT/'experiments/agentic/app/execute_app.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with self.assertRaisesRegex(ValueError,'allowlist'):
            module.resolve_source_motion({'source_motion':{'path':'research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_wave.motion','sha256':'0'*64}})

    def test_tampered_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);(directory/'qpos.npy').write_bytes(b'tampered')
            (directory/'manifest.json').write_text(json.dumps({'artifact_hashes':{'qpos.npy':'0'*64}}))
            result=evaluate_trial(directory,directory,[1,1,1,1,2],[.1]*5)
            self.assertFalse(result['integrity_passed'])
            self.assertFalse(result['physical_gate_passed'])
            self.assertTrue(any('hash mismatch' in e for e in result['integrity_errors']))
