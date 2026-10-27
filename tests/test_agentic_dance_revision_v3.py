"""Clock-language contract and failed-render result honesty; fixture only."""
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.dance import planner_v2 as p,manager_v2 as manager,server_v3,server_v1
import test_agentic_dance_manager as fixtures

class Tests(unittest.TestCase):
    def test_clock_sign_windows_and_turkish_in_actual_prepared_prompt(self):
        context={'music_summary':{'duration_s':4,'sample_rate':48000,'sample_frames':192000,'sha256':'a'*64},'source':{'id':'output_dance','baseline_parameters':p.baseline_parameters(),'baseline_qualification':p.BASELINE_QUALIFICATION},'constraints':{}}
        ready=p.prepare_dance_request({'instruction':'Hareketi biraz öne al'},context)
        self.assertEqual(ready['schema'],'pulsemorph.dance_planner_request.v2');self.assertEqual(ready['fixed_scope']['arm_gain_windows_s'],[[.4,2.],[2.,3.6]])
        self.assertIn('tau(t)=t+timing_shift_s*sin(pi*t/4)^2',ready['prompt']);self.assertIn('ADVANCES',ready['prompt']);self.assertIn('NOT a delay',ready['prompt']);self.assertIn('MUST be natural Turkish',ready['prompt']);self.assertIn('NOT uniform50ms',ready['prompt'])
    def test_renderer_failure_has_no_provisional_completed_result_or_external_message(self):
        old=fixtures.m;fixtures.m=manager;f=fixtures.Tests('test_first_physical_pass_stops_without_intent_score');f.setUp()
        try:
            m=f.manager([True,True])
            def fail(*args,**kwargs):raise RuntimeError('SECRET_SHOULD_NOT_PERSIST')
            m.renderer=fail;job=f.runjob(m)
            self.assertEqual(job['status'],'failed');self.assertIsNone(job['result']);self.assertTrue(job['finalized']);self.assertEqual(job['new_model_calls'],1);self.assertEqual(job['new_physical_trials'],2)
            raw=(m.jobs_dir/job['id']/'failure.json').read_text();self.assertNotIn('SECRET_SHOULD_NOT_PERSIST',raw);self.assertIn('RuntimeError',raw);self.assertNotIn('traceback',json.dumps(m.get(job['id'])))
        finally:f.tearDown();fixtures.m=old
    def test_http_reuses_frozen_handlers_and_new_manager_only(self):
        self.assertIs(server_v3.Facade,server_v1.Facade);self.assertIs(server_v3.Handler,server_v1.Handler);self.assertIs(server_v3.DanceManager,manager.DanceManager)

if __name__=='__main__':unittest.main()
