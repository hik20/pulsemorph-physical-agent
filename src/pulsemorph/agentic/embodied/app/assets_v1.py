"""Direct body-agent profile, known scenario DTOs, no historical job dependency."""
from pathlib import Path
import copy,hashlib,importlib.util,json,math
from pulsemorph.agentic.app.phrase.assets import PhraseAssets,PUBLIC_PYTHON,MODEL,MODEL_SHA,BODY_SHA,CONTROLLER_SHA
JOBS_ROOT=Path.home()/'.cache/pulsemorph-agentic/body-agent-v1/jobs'
WORKER=Path(__file__).with_name('worker_v1.py')
RIGHT_TARGET=[.16240383871601374,-.14286599443025425,.6080935474884148]
LEFT_TARGET=[RIGHT_TARGET[0],-RIGHT_TARGET[1],RIGHT_TARGET[2]]
LIMITS={'model_calls':16,'previews':8,'actual_control_steps':500,'sandbox_control_steps':400,'actual_physics_steps':10000,'sandbox_physics_steps':8000,'warmup_ONNX':1,'source_initializations':1,'wall_seconds':1800,'prelude_control_steps':100,'max_live_images':20,'final_video_render_FK':250}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def ordinary(path):
 path=Path(path).absolute()
 for p in (path,*path.parents):
  if p.is_symlink() or p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked body-agent paths forbidden')
 return path

def prepare_request(value):
 if type(value) is not dict or set(value)-{'scenario','instruction'} or 'scenario' not in value or type(value['scenario']) is not str or value['scenario'] not in ('right_reach','both_reach'):raise ValueError('Known fixed body scenario required')
 instruction=value.get('instruction','')
 if type(instruction) is not str or len(instruction)>1500:raise ValueError('Instruction must be bounded text')
 instruction.encode('utf8')
 return {'schema':'pulsemorph.body_agent_request.v1','scenario':value['scenario'],'instruction':instruction.strip(),'fixed_targets':{'right_rubber_hand':RIGHT_TARGET.copy(),**({'left_rubber_hand':LEFT_TARGET.copy()} if value['scenario']=='both_reach' else {})},'tolerance_m':.012,'hold_s':0.,'limits':copy.deepcopy(LIMITS),'qualification':'Fixed scenario; optional text is an execution hint, not authority to redefine goal/tool/body/controller or a semantic-intent guarantee.'}

def tasks(prepared,session_id):
 def goal(body,target,suffix):return {'schema':'pulsemorph.embodied_wrist_task.v1','task_id':session_id+suffix,'body':body,'frame':'world','target_world_m':list(target),'tolerance_m':.012,'hold_s':0.,'max_sample_gap_s':.05,'min_root_height_m':.6}
 right=goal('right_rubber_hand',RIGHT_TARGET,'_right');left=goal('left_rubber_hand',LEFT_TARGET,'_left');both={'schema':'pulsemorph.embodied_dual_wrist_task.v1','task_id':session_id+'_both','goals':[right,left]}
 return {'right_phase':right,**({'both_phase':both} if prepared['scenario']=='both_reach' else {}),'final_task':both if prepared['scenario']=='both_reach' else right}

def build_pins(root,*,include_environment=False):
 root=ordinary(root).resolve();assets=PhraseAssets(root);manifest=assets.verify();pins={str(ordinary(root/p).resolve()):h for p,h in manifest['live_numeric_pins'].items()}
 # Only the admitted safe initialization packages are consumed, not old trial logs.
 pins[str(assets.directory/'manifest.json')]=sha(assets.directory/'manifest.json')
 for row in assets.source_records():pins[str(ordinary(row['path']).resolve())]=row['sha256']
 paths=list((root/'src/pulsemorph/agentic/embodied').rglob('*.py'))+list((root/'src/pulsemorph/motion').glob('*.py'))
 paths+=[root/'src/pulsemorph/agentic'/name for name in ('phrase_program.py','phrase_reference.py','phrase_reference_v2.py','phrase_expression_v1.py','phrase_planner.py')]
 paths+=[root/'src/pulsemorph/agentic/expression'/name for name in ('__init__.py','planner_v2.py','goals_v1.py')]
 paths+=[root/'src/pulsemorph/agentic/app'/name for name in ('providers.py',)]
 paths+=[root/'src/pulsemorph/agentic/app/phrase'/name for name in ('assets.py','worker.py')]
 paths+=list((root/'research/vendor/ProtoMotions/protomotions/data/assets/mesh/G1').glob('*.stl'))
 paths+=[root/'research/vendor/ProtoMotions/deployment'/name for name in ('test_tracker_mujoco.py','tracker_inputs.py','state_utils.py','motion_utils.py')]
 for path in paths:
  path=ordinary(path).resolve()
  if not path.is_file():raise ValueError('Body-agent source dependency missing')
  pins[str(path)]=sha(path)
 pins[str(PUBLIC_PYTHON.resolve())]=sha(PUBLIC_PYTHON)
 required={str((root/MODEL).resolve()):MODEL_SHA,str((root/(MODEL[:-5]+'.yaml')).resolve()):CONTROLLER_SHA,str((root/'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml').resolve()):BODY_SHA}
 for path,digest in required.items():
  if sha(ordinary(path))!=digest:raise ValueError('Pinned body/model/controller changed')
  pins[path]=digest
 environment=None
 if include_environment:
  from pulsemorph.agentic.app.phrase.worker import environment as numerical_environment
  environment=numerical_environment();pins.update(environment['module_pins'])
  for name in ('scipy.spatial.transform._rotation','scipy.interpolate._ppoly'):
   spec=importlib.util.find_spec(name)
   if spec is None or not spec.origin:raise RuntimeError('Numeric dependency origin unavailable')
   pins[str(Path(spec.origin).resolve())]=sha(spec.origin)
 return pins,environment

def status(root):
 try:
  pins,_=build_pins(root)
  return {'available':True,'state':'pinned_local_profile_ready','source_pin_count':len(pins),'controller_mode':'classical_arm_PD_velocity_and_sampled_bias_plus_pretrained_legs_trunk_v1','scenarios':['right_reach','both_reach'],'numerical_environment_checked_in_worker':False,'no_model_NN_FK_physics_calls':True}
 except (OSError,ValueError,RuntimeError):return {'available':False,'state':'local_profile_unavailable_or_changed','no_model_NN_FK_physics_calls':True}
