"""Dance backend boundaries and numeric-closure review; no inference/physics."""
import ast,copy,io,json,sys,tempfile,unittest,wave
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.dance import assets_v1 as a
from pulsemorph.agentic.dance.runtime_v1 import music_observation,verify_trial,DanceRuntime,verify_history

class DanceTests(unittest.TestCase):
 def test_baseline_is_explicit_edited_owned_copy(self):
  value=a.baseline_parameters();self.assertEqual(value['left_gain_a'],.9);self.assertEqual(value['left_gain_b'],1.)
  self.assertIn('notunit',a.BASELINE_QUALIFICATION);value['left_gain_a']=1.;self.assertEqual(a.baseline_parameters()['left_gain_a'],.9)
 def test_bounded_parameters_exact(self):
  unit={k:0. if k=='timing_shift_s' else 1. for k in a.FIELDS}
  self.assertEqual(a.parameters(unit),unit)
  for k,v in [('left_gain_a',True),('left_gain_a',10**1000),('left_gain_b',float('nan')),('right_gain_a',.79),('timing_shift_s',.151)]:
   with self.assertRaises(ValueError):a.parameters({**unit,k:v})
  with self.assertRaises(ValueError):a.parameters({**unit,'extra':1})
 def test_actual_pcm_bound_hash_and_no_fictitious_audio(self):
  self.assertEqual(music_observation(None),{'available':False,'duration_s':0.,'observed_through_sample':None})
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'music.wav'
   with wave.open(str(p),'wb') as w:w.setparams((1,2,48000,192000,'NONE','not compressed'));w.writeframes(np.zeros(192000,dtype='<i2').tobytes())
   m={'available':True,'path':str(p),'sha256':a.sha(p),'sample_rate':48000,'duration_s':4,'observed_through_sample':192000}
   self.assertEqual(music_observation(m)['sha256'],m['sha256'])
   with self.assertRaises(ValueError):music_observation({**m,'observed_through_sample':384000})
   p.write_bytes(b'tampered')
   with self.assertRaises(ValueError):music_observation(m)
 def test_hash_guard_before_source_use(self):
  with tempfile.TemporaryDirectory() as d:
   for name in (a.SOURCE,a.MODEL,a.BODY,a.CONTROLLER):p=Path(d)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'bad')
   with self.assertRaises(RuntimeError):a.DanceAssets(d).verify()
 def test_output_and_cancel_before_environment(self):
  r=DanceRuntime.__new__(DanceRuntime);r.artifact_root=a.ROOT_CACHE.resolve()
  unit={k:0. if k=='timing_shift_s' else 1. for k in a.FIELDS}
  with patch('pulsemorph.agentic.dance.runtime_v1.environment') as probe:
   with self.assertRaises(ValueError):r.run_trial(unit,ROOT/'not_allowed','fixture',0)
   self.assertFalse(probe.called)
 def test_wrapper_identity_and_path_guards(self):
  for value in ({},{'id':'fixture','index':True},{'id':'fixture','index':0,'paths':{'directory':'C:/outside'}}):
   with self.assertRaises(ValueError):verify_trial(value,ROOT)
 def test_history_is_previous_targets_not_actions(self):
  targets=np.ones((1,29),np.float32)*.4;outputs={'call_0001__joint_pos_targets':targets,'call_0001__actions':np.ones((1,29),np.float32)*.7}
  verify_history(np.zeros((1,1,29),np.float32),outputs,1)
  verify_history(targets[:,None,:],outputs,2)
  for bad in (outputs['call_0001__actions'][:,None,:],targets,targets[:,None,:].astype('float64')):
   with self.assertRaises(RuntimeError):verify_history(bad,outputs,2)
 def test_saved_wrapper_and_raw_tamper(self):
  # Explicit synthetic proof-shaped fixture only, not physics evidence.
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);d=root/'trial';d.mkdir();(d/'common_reference').mkdir()
   source=root/'source';source.write_bytes(b'synthetic dependency')
   artifact=d/'array.fixture';artifact.write_bytes(b'synthetic numeric bytes')
   metrics={'physical_gate_passed':True,'min_root_height_m':.8,'mean_joint_rmse_rad':.1}
   manifest={'candidate_id':'fixture__trial_0','trial_index':0,'implementation_pins':{str(source):a.sha(source)}}
   a.write_new(d/'manifest.json',manifest)
   audit={'integrity_passed':True,'metrics':metrics,'inventory':{n:a.sha(d/n) for n in ('manifest.json','array.fixture')}};a.write_new(d/'audit.json',audit)
   t={'schema':'pulsemorph.dance_trial.v1','id':'fixture__trial_0','index':0,'status':'completed','integrity_passed':True,'parameters':{},'paths':{'directory':str(d),'manifest':str(d/'manifest.json'),'audit':str(d/'audit.json'),'common_reference':str(d/'common_reference')},'metrics':metrics}
   a.write_new(d/'result.json',t);t['proof_sha256']={n:a.sha(d/(n+'.json')) for n in ('manifest','audit','result')}
   with patch('pulsemorph.agentic.dance.runtime_v1.ROOT_CACHE',root):
    self.assertTrue(verify_trial(t,ROOT)['integrity_passed'])
    for changed in ({**t,'id':'spoof'},{**t,'index':True},{**t,'paths':{**t['paths'],'manifest':str(source)}}):
     with self.assertRaises((ValueError,RuntimeError)):verify_trial(changed,ROOT)
    artifact.write_bytes(b'tampered')
    with self.assertRaises(RuntimeError):verify_trial(t,ROOT)
 def test_snapshot_source_copies_live_bytes(self):
  from pulsemorph.agentic.dance.worker_v1 import snapshot_source
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);source=d/'live';source.write_bytes(b'fixture source bytes')
   destination=d/'trial/source_snapshot/copy';objects=d/'objects';digest=a.sha(source)
   with patch('pulsemorph.agentic.dance.assets_v1.ROOT_CACHE',d):snapshot_source(source,destination,objects,digest)
   self.assertEqual(destination.read_bytes(),source.read_bytes())
   source.write_bytes(b'changed live input')
   self.assertEqual(destination.read_bytes(),b'fixture source bytes')
 def test_worker_numeric_reset_ast_preserved(self):
  old=ast.parse((ROOT/'experiments/agentic/app/execute_app.py').read_text(encoding='utf8'));new=ast.parse((ROOT/'src/pulsemorph/agentic/dance/worker_v1.py').read_text(encoding='utf8'))
  # Controller initialization/perturbation mechanics are copied literally.
  for name in ('apply_initial_perturbation','validate_initial_perturbation'):
   x=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);y=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(x,include_attributes=False),ast.dump(y,include_attributes=False))
  for path in (ROOT/'src/pulsemorph/agentic/dance').glob('*.py'):compile(path.read_bytes(),str(path),'exec')

if __name__=='__main__':unittest.main()
