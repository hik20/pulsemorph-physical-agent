"""Exact existing dance source and fixed tracker profile; no generative model."""
from pathlib import Path
import hashlib,json,math,os,ast
ROOT_CACHE=Path.home()/'.cache/pulsemorph-agentic/dance-studio-v1'
SOURCE='research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_dance.motion'
SOURCE_SHA='934309f9357dbb833b7d68411c6c41fd6e31e14d792d5eca43952a9c5f28f784'
MODEL='research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'
MODEL_SHA='a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c'
BODY='research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
BODY_SHA='e3926a6578cafaee38d0a7a3b7e7ebb0031f623c69e5cd34aa97d61e943b7e8e'
CONTROLLER=MODEL[:-5]+'.yaml'
CONTROLLER_SHA='57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139'
PUBLIC_PYTHON=Path('C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe')
FIELDS=('left_gain_a','right_gain_a','left_gain_b','right_gain_b','timing_shift_s')
BASELINE_QUALIFICATION='edited source start profile; smoothfirst-left-arm10%attenuation, notunit'

def baseline_parameters():
 return {'left_gain_a':.90,'right_gain_a':1.,'left_gain_b':1.,'right_gain_b':1.,'timing_shift_s':0.}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def ordinary(path):
 p=Path(path).absolute()
 for item in (p,*p.parents):
  if item.is_symlink() or (item.exists() and getattr(item.lstat(),'st_file_attributes',0)&1024):raise ValueError('Linked dance path forbidden')
 return p

def parameters(value):
 if type(value) is not dict or set(value)!=set(FIELDS):raise ValueError('Exact five dance parameters required')
 for k,v in value.items():
  lo,hi=(-.15,.15) if k=='timing_shift_s' else (.8,1.2)
  if type(v) not in (int,float) or not lo<=v<=hi or not math.isfinite(v):raise ValueError('Dance parameter outside bounded domain')
 return dict(value)

def write_new(path,value):
 with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)

class DanceAssets:
 def __init__(self,root):self.root=ordinary(root).resolve()
 def verify(self):
  pins={SOURCE:SOURCE_SHA,MODEL:MODEL_SHA,BODY:BODY_SHA,CONTROLLER:CONTROLLER_SHA}
  for name,digest in pins.items():
   p=ordinary(self.root/name)
   if not p.is_file() or sha(p)!=digest:raise RuntimeError('Pinned dance source/profile missing or changed: '+name)
  return pins
 def implementation_pins(self):
  self.verify();paths=list((self.root/'src/pulsemorph/agentic/dance').glob('*.py'))
  paths+=list((self.root/'src/pulsemorph/motion').glob('*.py'))
  paths+=[self.root/'src/pulsemorph/agentic'/n for n in ('reference.py','multifeature_reference.py','local_multifeature_reference.py','common_player.py','phrase_program.py')]
  paths+=[self.root/'experiments/agentic'/n for n in ('execute_trial.py','evaluate_trial.py','evaluate_multifeature.py')]
  paths+=[self.root/'experiments/causal_music/reference.py']
  vendor=self.root/'research/vendor/ProtoMotions'
  paths+=[vendor/'deployment'/n for n in ('test_tracker_mujoco.py','motion_utils.py','state_utils.py','tracker_inputs.py')]
  paths+=list((vendor/'protomotions/data/assets/mesh/G1').glob('*.stl'))
  paths+=[self.root/'src/pulsemorph/agentic/app/phrase/worker.py',self.root/'src/pulsemorph/agentic/app/phrase/assets.py']
  # Reuse reviewed exact-commit LF code pins; recorded copy hashes remain RAW.
  preflight=self.root/'experiments/agentic/app/preflight.py'
  tree=ast.parse(preflight.read_text(encoding='utf8'));pins=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MOTION_CLASS_SOURCE_PINS' for t in n.targets))
  for name,digest in pins.items():
   item=ordinary(vendor/name)
   if hashlib.sha256(item.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:raise RuntimeError('Trusted motion serialization code changed')
   paths.append(item)
  paths.append(preflight)
  return {**{str((self.root/n).resolve()):h for n,h in self.verify().items()},**{str(ordinary(p).resolve()):sha(p) for p in paths}}
 def prepare(self,output=None):
  pins=self.implementation_pins();result={'schema':'pulsemorph.dance_asset_preparation.v1','source_sha256':SOURCE_SHA,'model_sha256':MODEL_SHA,'body_sha256':BODY_SHA,'controller_sha256':CONTROLLER_SHA,'pins':pins,'reference_compiled':False,'new_model_or_onnx_or_physics_calls':0,'scope':'Existing fullbody dance choreography, bounded arm/clock edits; no new skill, style or learned weights.'}
  if output is not None:
   p=ordinary(output).resolve();allowed=ROOT_CACHE.resolve()
   if p==allowed or not p.is_relative_to(allowed):raise ValueError('Dedicated C dance preparation required')
   p.mkdir(parents=True,exist_ok=False);write_new(p/'assets.json',result)
  return result


def check_reference(root,values):
 """Pure trusted-source compile, no ONNX session/FK/physics. Never clamps."""
 import sys,io,yaml,numpy as np
 from dataclasses import replace
 assets=DanceAssets(root);pins=assets.implementation_pins();p=parameters(values)
 sys.path.insert(0,str(assets.root/'research/vendor/ProtoMotions'))
 import torch
 from ..multifeature_reference import MultiFeatureReferencePlayer
 from ..common_player import materialize_reference
 from ..phrase_program import _continuous_ranges,ProgramCompileError
 from pulsemorph.motion.g1 import profile_from_yaml,assemble_reference_inputs
 source=ordinary(assets.root/SOURCE);data=source.read_bytes()
 if hashlib.sha256(data).hexdigest()!=SOURCE_SHA:raise RuntimeError('Trusted dance samebuffer hash differs')
 raw=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
 config=yaml.safe_load((assets.root/CONTROLLER).read_text());profile=profile_from_yaml(assets.root/CONTROLLER)
 generator=MultiFeatureReferencePlayer(raw,joint_names=config['robot']['joint_names'],body_names=config['robot']['body_names'],source_start_s=0.,control_steps=200,section_windows_s=[[.4,2.],[2.,3.6]],source_provenance={'path':SOURCE,'sha256':SOURCE_SHA,'kind':'existing upstream generated dance'},**p)
 ref=materialize_reference(generator,profile=profile,candidate_id='dance_pure_preflight',observation_id='no_audio',task_id='dance_pure_preflight',runtime_session_id='dance_pure_preflight',source_path=SOURCE,source_sha256=SOURCE_SHA,audio_observed_through_sample=None,audio_duration_s=0.,producer_kind='agentic')
 ref=replace(ref,metadata={**ref.metadata,'producer':{'kind':'agentic','id':'pulsemorph-dance-source-tool','version':'1','control_tool':'global_clock_v1'}})
 rows=_continuous_ranges(ref,profile)
 for i in range(200):assemble_reference_inputs(ref,i*.02)
 if any(sha(path)!=digest for path,digest in pins.items()):raise RuntimeError('Dance inputs changed during pure compile')
 report={'schema':'pulsemorph.dance_compile_report.v1','admitted':all(r['within_unchanged_range'] for r in rows),'parameters':p,'source_sha256':SOURCE_SHA,'joint_ranges':rows,'strict_common_rows':200,'execution_s':4,'horizon_s':4.14,'new_onnx_calls':0,'new_FK_calls':0,'physics_steps':0}
 if not report['admitted']:raise ProgramCompileError('Dance common reference exceeds unchanged ranges',report)
 return ref,report
