"""Body app worker fixtures; no actual provider, renderer or physics backend."""
from pathlib import Path
import sys,json,tempfile,time,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.app.assets_v1 import prepare_request,tasks,RIGHT_TARGET,LEFT_TARGET
from pulsemorph.agentic.embodied.app.worker_v1 import safe_text,FileCancel,Progress,run_job
class WorkerTests(unittest.TestCase):
 def test_request_and_goals_fixed_owned(self):
  p=prepare_request({'scenario':'both_reach','instruction':'  küçük hareket  '});self.assertEqual(p['instruction'],'küçük hareket');g=tasks(p,'fixture');self.assertEqual(g['final_task']['goals'][0]['target_world_m'],RIGHT_TARGET);self.assertEqual(g['final_task']['goals'][1]['target_world_m'],LEFT_TARGET);self.assertEqual(g['final_task']['goals'][0]['tolerance_m'],.012);p['fixed_targets']['right_rubber_hand'][0]=99;self.assertEqual(prepare_request({'scenario':'right_reach'})['fixed_targets']['right_rubber_hand'],RIGHT_TARGET)
  for bad in ({'scenario':'arbitrary'},{'scenario':'right_reach','goals':[]},{'scenario':'right_reach','instruction':True},{'scenario':'right_reach','instruction':'x'*1501}):
   with self.assertRaises(ValueError):prepare_request(bad)
 def test_public_text_paths_redacted_ratios_urls_preserved(self):
  original='sol/sağ 110/90; https://example.org/ref';self.assertEqual(safe_text(original),original)
  text=safe_text(r'file C:\Users\private\auth.json and C:/private/auth.json and /home/private/file and \\server\private\file')
  self.assertNotIn('auth.json',text);self.assertNotIn('server',text);self.assertNotIn('/home',text)
 def test_file_cancel_no_execution(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'cancel.request';c=FileCancel(p,time.monotonic()+60);self.assertFalse(c.is_set());p.write_text('{}');self.assertTrue(c.is_set())
 def test_progress_public_whitelist_without_paths_or_pins(self):
  with tempfile.TemporaryDirectory() as temp:
   progress=Progress(Path(temp),prepare_request({'scenario':'right_reach'}));progress.model_decision={'decision':'preview','reason':safe_text(r'private C:\Users\x\secret'),'summary':'öneri'};progress.publish();v=json.loads((Path(temp)/'progress.json').read_bytes());self.assertEqual(v,json.loads((Path(temp)/'public.json').read_bytes()));self.assertNotIn('source_pins',v);self.assertNotIn('Users',json.dumps(v));self.assertEqual(v['counts']['actual_steps'],0)
 def test_cancel_before_profile_or_backend(self):
  with tempfile.TemporaryDirectory() as temp:
   jobs=Path(temp);job=jobs/'body_fixture';job.mkdir();request=job/'request.json';request.write_text(json.dumps({'scenario':'right_reach','instruction':''}));(job/'cancel.request').write_text('{}')
   with patch('pulsemorph.agentic.embodied.app.worker_v1.JOBS_ROOT',jobs),patch('pulsemorph.agentic.embodied.app.worker_v1.build_pins') as pins:
    result=run_job(ROOT,request,job/'execution',backend_factory=lambda *a,**k:self.fail('No backend allowed'),enable_final_video=False,enable_live_images=False)
   self.assertEqual(result['status'],'cancelled');self.assertFalse(pins.called);self.assertEqual(result['counts']['actual_steps'],0)
 def test_initialization_failure_retains_frozen_goals_and_unknown_flag(self):
  with tempfile.TemporaryDirectory() as temp:
   jobs=Path(temp);job=jobs/'body_fixture';job.mkdir();request=job/'request.json';request.write_text(json.dumps({'scenario':'both_reach','instruction':''}))
   def fail(root,authority):
    self.assertTrue((job/'execution/frozen_goals.json').is_file());self.assertEqual(authority['max_control_steps'],500);raise RuntimeError('synthetic initialization failure')
   with patch('pulsemorph.agentic.embodied.app.worker_v1.JOBS_ROOT',jobs),patch('pulsemorph.agentic.embodied.app.worker_v1.build_pins',return_value=({},{})):
    result=run_job(ROOT,request,job/'execution',backend_factory=fail,enable_final_video=False,enable_live_images=False)
   self.assertEqual(result['status'],'failed');self.assertTrue(result['counts']['physical_completion_unconfirmed']);self.assertFalse(result['task_verified_actual']);self.assertTrue((job/'execution/private_failure.json').is_file())
if __name__=='__main__':unittest.main()
