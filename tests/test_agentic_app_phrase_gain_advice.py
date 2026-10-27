"""Conditional cubic mathematics and bounded orchestration; no physical calls."""
from pathlib import Path
import copy,sys,threading,unittest,tempfile,hashlib
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase import gain_advice_v1 as g
PROGRAM={'schema':'pulsemorph.phrase_program.v1','order':['wave','high5'],'heading_policy':'align_join_heading','first_left_gain':.75,'first_right_gain':1.25,'second_left_gain':1.25,'second_right_gain':.75,'rationale':'fixture'}
class AdviceTests(unittest.TestCase):
    def test_constant_affine_active_upper(self):
        a=np.zeros((4,1,1));b=np.zeros_like(a);b[3]=2
        result=g._conditional_bounds(a,b,[0,1],['joint'],{'joint':(-1,.2)})
        self.assertAlmostEqual(result['upper']['gain'],1.1);self.assertEqual(result['lower']['gain'],.75)
    def test_cubic_interior_not_only_knots(self):
        a=np.zeros((4,1,1));b=np.zeros_like(a);b[1,0,0]=-4;b[2,0,0]=4
        result=g._conditional_bounds(a,b,[0,1],['joint'],{'joint':(-1,.1)})
        self.assertAlmostEqual(result['upper']['gain'],1.1);self.assertAlmostEqual(result['upper']['time_s'],.5)
    def test_parallel_slope_and_empty_fixed_bound(self):
        a=np.zeros((4,1,1));b=np.zeros_like(a)
        self.assertEqual(g._conditional_bounds(a,b,[0,1],['j'],{'j':(-1,1)})['status'],'available')
        a[3]=2;self.assertEqual(g._conditional_bounds(a,b,[0,1],['j'],{'j':(-1,1)})['status'],'empty')
    def test_conflicting_gain_bounds_are_empty(self):
        a=np.zeros((4,1,2));b=np.zeros_like(a);b[3]=1
        result=g._conditional_bounds(a,b,[0,1],['left','right'],{'left':(.2,1),'right':(-1,-.2)})
        self.assertEqual(result['status'],'empty');self.assertGreater(result['lower']['gain'],result['upper']['gain'])
    def test_nonfinite_and_clock_rejected(self):
        a=np.zeros((4,1,1));b=np.zeros_like(a)
        for bad in (np.full_like(a,np.nan),np.full_like(a,np.inf)):
            with self.assertRaises(ValueError):g._conditional_bounds(bad,b,[0,1],['j'],{'j':(-1,1)})
        with self.assertRaises(ValueError):g._conditional_bounds(a,b,[1,0],['j'],{'j':(-1,1)})
    def test_cancel_and_invalid_field_before_assets(self):
        event=threading.Event();event.set()
        with self.assertRaises(g.AdviceCancelled):g.advise_gain_interval(None,PROGRAM,'second_left_gain',cancel_event=event)
        with self.assertRaises(ValueError):g.advise_gain_interval(None,PROGRAM,'bad_field')
    def test_integrity_failure_never_becomes_unavailable(self):
        class Assets:
            def verify(self):raise RuntimeError('fixture integrity')
        with self.assertRaisesRegex(RuntimeError,'fixture integrity'):g.advise_gain_interval(Assets(),PROGRAM,'second_left_gain')
    def test_four_basis_cap_unavailable_and_final_integrity_check(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);body=directory/'body';body.write_bytes(b'fixture body');config=directory/'config';config.write_bytes(b'fixture config');(directory/'manifest.json').write_bytes(b'{}')
            bodysha=hashlib.sha256(body.read_bytes()).hexdigest();configsha=hashlib.sha256(config.read_bytes()).hexdigest()
            class Assets:
                manifest={'live_numeric_pins':{'body':bodysha,'config':configsha}}
                calls=0
                def verify(self):self.calls+=1
                def references(self):return {}
                def source_records(self):return []
            a=Assets();a.directory=directory
            profile={'body_path':body,'yaml_path':config,'body_sha256':bodysha,'yaml_sha256':configsha}
            rejection=g.ProgramCompileError('fixture',{'status':'rejected_joint_ranges','violations':['left_elbow_joint']})
            with patch.object(g,'profile_from_yaml',return_value=profile),patch.object(g,'compile_program',side_effect=rejection) as compile:
                result=g.advise_gain_interval(a,PROGRAM,'second_left_gain')
            self.assertEqual(compile.call_count,4);self.assertEqual(a.calls,2);self.assertEqual(result['status'],'unavailable');self.assertEqual(result['reason'],'fewer_than_two_admitted_bases');self.assertEqual([r['gain'] for r in result['basis_attempts']],[1.,1.2,.8,.75])

if __name__=='__main__':unittest.main()
