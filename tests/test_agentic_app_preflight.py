"""Trusted pickle probing uses isolated synthetic fixtures; no vendor assets required."""
import importlib.util
from pathlib import Path
import tempfile
import hashlib
from enum import Enum
import unittest
from unittest.mock import Mock,patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('app_preflight_fixture',ROOT/'experiments/agentic/app/preflight.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.vendor=self.root/'research/vendor/ProtoMotions';self.motion=self.vendor/'data/g1-kimodo-generated/proto/output_wave.motion';self.motion.parent.mkdir(parents=True);self.motion.write_bytes(b'synthetic trusted fixture, not pickle/physics');self.pins={'wave':hashlib.sha256(self.motion.read_bytes()).hexdigest()}
        conversion=Enum('StateConversion',{'COMMON':0},module='protomotions.simulator.base_simulator.simulator_state').COMMON
        self.raw={'state_conversion':conversion,'fps':30}
        for k,tail in {'dof_pos':(29,),'dof_vel':(29,),'rigid_body_pos':(33,3),'rigid_body_rot':(33,4),'rigid_body_vel':(33,3),'rigid_body_ang_vel':(33,3),'rigid_body_contacts':(33,)}.items():self.raw[k]=np.zeros((150,)+tail)
    def test_hash_before_deserialization(self):
        loader=Mock(return_value=self.raw);self.motion.write_bytes(b'not allowlisted')
        with self.assertRaisesRegex(ValueError,'before deserialization'):p.validate_trusted_motion(self.root,'wave',pins=self.pins,source_pins={},loader=loader)
        loader.assert_not_called()
    def test_missing_serialization_source_actionable_before_load(self):
        loader=Mock(return_value=self.raw)
        with self.assertRaisesRegex(ModuleNotFoundError,'simulator_state.py'):p.validate_trusted_motion(self.root,'wave',pins=self.pins,source_pins={'protomotions/simulator/base_simulator/simulator_state.py':'0'*64},loader=loader)
        loader.assert_not_called()
    def test_good_payload_coverage_and_no_external_origins(self):
        with patch.dict(p.sys.modules,{'protomotions.fixture':type('Module',(),{'__file__':str(self.vendor/'fixture.py')})()}):
            result=p.validate_trusted_motion(self.root,'wave',pins=self.pins,source_pins={},loader=lambda _:self.raw)
        self.assertEqual(result['frames'],150);self.assertGreater(result['coverage_s'],4.14);self.assertEqual(result['inference_calls'],0);self.assertIn('protomotions.fixture',result['vendor_import_origins'])
    def test_import_origin_outside_selected_root_rejected(self):
        with patch.dict(p.sys.modules,{'protomotions.fixture':type('Module',(),{'__file__':'D:/elsewhere/fallback.py'})()}),self.assertRaisesRegex(ValueError,'outside selected root'):p.validate_trusted_motion(self.root,'wave',pins=self.pins,source_pins={},loader=lambda _:self.raw)
    def test_nonfinite_or_short_payload_rejected(self):
        self.raw['dof_pos'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'invalid trusted motion field'):p.validate_trusted_motion(self.root,'wave',pins=self.pins,source_pins={},loader=lambda _:self.raw)
    def test_source_LF_CRLF_equivalent_semantic_change_rejected_before_load(self):
        name='protomotions/fixture.py';path=self.vendor/name;path.parent.mkdir(parents=True,exist_ok=True)
        lf=b'# fixture source\nvalue = 1\n';expected=hashlib.sha256(lf).hexdigest()
        for data in (lf,lf.replace(b'\n',b'\r\n')):
            path.write_bytes(data)
            detail=p.validate_trusted_motion(self.root,'wave',pins=self.pins,source_pins={name:expected},loader=lambda _:self.raw)
            self.assertEqual(detail['source_code_hashes'][name]['observed_raw_sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(detail['source_code_hashes'][name]['canonical_LF_sha256'],expected)
        path.write_bytes(b'# fixture source\nvalue = 2\n');loader=Mock(return_value=self.raw)
        with self.assertRaisesRegex(ValueError,'source hash mismatch'):p.validate_trusted_motion(self.root,'wave',pins=self.pins,source_pins={name:expected},loader=loader)
        loader.assert_not_called()

    def profile_fixture(self):
        xml=b'<mujoco/>\r\n';yaml=b'control_dt: 0.02\r\n'
        source=self.root/'src/pulsemorph/agentic/app/exports.py';source.parent.mkdir(parents=True);source.write_text('BODY_HASH = '+repr(hashlib.sha256(xml).hexdigest())+'\nCONTROLLER_HASH = '+repr(hashlib.sha256(yaml).hexdigest()))
        a=self.vendor/'protomotions/data/assets/mjcf/g1_holo_compat.xml';a.parent.mkdir(parents=True);a.write_bytes(xml)
        b=self.vendor/'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml';b.parent.mkdir(parents=True);b.write_bytes(yaml)
        return a,b
    def test_raw_profile_assets_admitted_from_export_single_source(self):
        self.profile_fixture();rows=p.raw_profile_asset_checks(self.root)
        self.assertTrue(all(r['ok'] for r in rows));self.assertTrue(all(r['detail']['observed_raw_sha256']==r['detail']['expected_raw_sha256'] for r in rows))
    def test_raw_profile_LF_equivalent_rejected_without_normalization(self):
        a,b=self.profile_fixture();a.write_bytes(a.read_bytes().replace(b'\r\n',b'\n'));rows=p.raw_profile_asset_checks(self.root)
        self.assertFalse(rows[0]['ok']);self.assertTrue(rows[1]['ok']);self.assertIn('core.autocrlf=true',rows[0]['detail']['action']);self.assertIn('NONE',rows[0]['detail']['normalization'])
    def test_raw_profile_content_mutation_and_missing_rejected(self):
        a,b=self.profile_fixture();a.write_bytes(b'<changed/>\r\n');b.unlink();rows=p.raw_profile_asset_checks(self.root)
        self.assertFalse(rows[0]['ok']);self.assertFalse(rows[1]['ok']);self.assertIsNone(rows[1]['detail']['observed_raw_sha256'])

    def test_fresh_probe_isolated_child_missing_class_failure(self):
        failure=type('Result',(),{'returncode':1,'stdout':'{"ok":false,"error_type":"ModuleNotFoundError","error":"simulator_state.py"}'})()
        with patch.object(p.subprocess,'run',return_value=failure) as run:
            ok,detail=p.trusted_motion_probe(self.root,'wave')
        self.assertFalse(ok);self.assertIn('simulator_state',detail['error']);self.assertEqual(run.call_args.args[0][1:3],['-I','-B']);self.assertNotIn('PYTHONPATH',run.call_args.kwargs['env'])

if __name__=='__main__':unittest.main()
