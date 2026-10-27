"""Preparation/authority/parity gates only, with explicit synthetic arrays."""
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase import motor_response_v1 as m
class MotorResponseTests(unittest.TestCase):
    def test_single_factor_order_and_bounds(self):
        rows=m.programs();m.validate_design(rows)
        self.assertEqual([r['label'] for r in rows],['unit','low','high']);self.assertEqual([r['trial_index'] for r in rows],[-1,0,1]);self.assertEqual([r['program']['first_left_gain'] for r in rows],[1.,.75,1.25])
        for row in rows:self.assertTrue(all(row['program'][k]==1 for k in m.GAIN_FIELDS if k!='first_left_gain'))
        rows[2]['program']['first_right_gain']=.9
        with self.assertRaises(ValueError):m.validate_design(rows)
    def make_bridge(self,root):
        for folder in (root/'fresh',root/'old'):
            (folder/'common_reference').mkdir(parents=True)
            for name in m.PAYLOADS:(folder/name).write_bytes(b'exact saved fixture payload')
            for name in m.NUMERIC_ARCHIVES:np.savez(folder/name,feed=np.arange(12,dtype=float).reshape(3,4),warmup=np.array([1.]))
            (folder/'policy_calls.json').write_text(json.dumps([{'role':'warmup'},{'role':'control'}]))
            for name in ('request.json','result.json'):(folder/name).write_text(json.dumps({'environment':{'fixture':True},'counts':{'fixture':True}}))
    def test_complete_byte_bridge_and_changed_output_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_bridge(root);self.assertTrue(m.bridge(root/'fresh',root/'old')['passed'])
            np.savez(root/'fresh/policy_outputs.npz',feed=np.arange(12,dtype=float).reshape(3,4)+1,warmup=np.array([1.]))
            with self.assertRaisesRegex(RuntimeError,'bridge mismatch'):m.bridge(root/'fresh',root/'old')
    def test_changed_payload_or_environment_reject(self):
        for mode in ('payload','environment'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.make_bridge(root)
                if mode=='payload':(root/'fresh/qpos.npy').write_bytes(b'changed')
                else:(root/'fresh/request.json').write_text(json.dumps({'environment':{'changed':True},'counts':{'fixture':True}}))
                with self.assertRaises(RuntimeError):m.bridge(root/'fresh',root/'old')
    def test_authority_missing_never_constructs_runtime(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'BASE',Path(temp)),patch.object(m,'PhraseRuntime') as runtime:
            with self.assertRaises(ValueError):m.execute(ROOT,Path(temp)/'run',{},None,None)
            runtime.assert_not_called()
    def test_output_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'BASE',Path(temp)/'allowed'):
            with self.assertRaises(ValueError):m.safe_output(Path(temp)/'elsewhere')
    def test_exclusive_attempt_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'attempt.json';m.write(path,{'attempt':1})
            with self.assertRaises(FileExistsError):m.write(path,{'attempt':2})
            self.assertEqual(json.loads(path.read_text()),{'attempt':1})
if __name__=='__main__':unittest.main()
