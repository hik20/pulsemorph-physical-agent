"""Prepare-first finite portable CLI. Startup never implies execution authority."""
import hashlib,json,re,time
from pathlib import Path
from .config_v5 import load_config,ordinary,digest,probe_environment
from .profile_v5 import verify_vendor,verify_profile

BUDGETS={'max_model_calls':4,'max_previews':2,'prelude_controls':100,'max_actual_controls':200,'max_sandbox_controls':100,'max_total_online_ONNX':300,'warmup_ONNX':1,'max_total_physics_steps':6000,'runtime_FK_max':301,'wrist_Jacobian_max':602,'terminal_failure_extra_FK_max':1,'terminal_failure_extra_Jacobian_max':2,'wall_seconds':600,'render_FK':0}
def budgets_for(request):
    return dict(BUDGETS)if request['mode']=='task'else {**BUDGETS,'max_model_calls':0,'max_previews':0,'max_actual_controls':120,'max_sandbox_controls':0,'max_total_online_ONNX':120,'max_total_physics_steps':2400,'runtime_FK_max':121,'wrist_Jacobian_max':242,'wall_seconds':180}
def write_new(path,value):
    with Path(path).open('x',encoding='utf8')as f:json.dump(value,f,indent=2,allow_nan=False)

def close_evidence(episode,backend,fork,output,result,source_pins):
    """Preserve parent and every sandbox, even when another export fails."""
    errors=[]
    if backend is not None:
        try:
            if episode is not None:episode.export(output/'actual_trace')
            else:
                (output/'actual_trace').mkdir(exist_ok=False)
                backend.export(output/'actual_trace')
            result['actual_counts']={'controls':backend.control_index,'steps':backend.step_count,'runtime_FK':backend.forward_count,'wrist_Jacobian':backend.jacobian_calls}
        except Exception as error:errors.append({'stage':'actual_trace','error_type':type(error).__name__})
    if fork is not None:
        for identity in sorted(fork._previews):
            try:fork.export_preview(identity,output/('sandbox_'+identity))
            except Exception as error:errors.append({'stage':'sandbox_'+identity,'error_type':type(error).__name__})
    try:
        result['source_pins_unchanged']=all(digest(ordinary(p))==h for p,h in source_pins.items())
    except (OSError,ValueError):result['source_pins_unchanged']=False
    if errors:result.update(status='failed',artifact_export_incomplete=True,artifact_errors=errors)
    if not result['source_pins_unchanged']:result.update(status='failed',source_integrity_failed=True)
    return result
def parse_request(v):
    if type(v)is not dict or set(v)!={'schema','mode','instruction','task','provider'} or v['schema']!='pulsemorph.portable_firstuse_request.v5' or v['mode']not in ('hold','task')or type(v['instruction'])is not str or not 3<=len(v['instruction'])<=1500:raise ValueError('Exact finite request required')
    if v['mode']=='hold':
        if v['task']is not None or v['provider']is not None:raise ValueError('HOLD uses no model/task rewrite')
    else:
        if v['provider']not in ('openrouter_free','nvidia_nim'):raise ValueError('Explicit hosted provider required')
        from ..tasks_v2 import validate_task
        validate_task(v['task'])
    return json.loads(json.dumps(v,allow_nan=False))
def prepare(root,config_path,request,output):
    root=Path(root).resolve();cfg=load_config(config_path);request=parse_request(request);verify_vendor(cfg);verify_profile(cfg)
    output=ordinary(output).resolve()
    if output.exists() or not output.is_relative_to(cfg['artifact_root'])or output==cfg['artifact_root'] or re.fullmatch('[A-Za-z0-9_-]{1,80}',output.name)is None:raise ValueError('Exclusive configured output required')
    env=probe_environment(cfg['runtime_python']);pins={}
    def add(p):p=ordinary(p).resolve();pins[str(p)]=digest(p)
    for row in json.loads(cfg['asset_identity'].read_bytes())['files']:add(cfg['vendor_root']/row['path'])
    for p in [Path(config_path),cfg['asset_identity'],cfg['profile_dir']/'manifest.json',cfg['profile_dir']/'standing.npz',cfg['runtime_python'],root/'experiments/agentic/embodied/run_body_agent_v5.py']:add(p)
    for row in env['module_pins'].values():
        for p,h in row['files'].items():
            if digest(p)!=h:raise ValueError('Environment changed during preparation')
            add(p)
    for p in Path(env['module_pins']['mujoco']['origin']).parent.glob('mujoco*.dll'):add(p)
    for folder in ('src/pulsemorph/agentic','src/pulsemorph/motion'):
        for p in (root/folder).rglob('*.py'):add(p)
    return {'schema':'pulsemorph.portable_firstuse_plan.v5','config_path':str(Path(config_path).resolve()),'root':str(root),'output':str(output),'request':request,'budgets':budgets_for(request),'environment':env,'source_pins':pins,'qualification':'Unexecuted finite command; prior portable HOLD does not prove general reach/live provider/task success.'}

def execute(plan_path,*,approved_plan_sha256):
    plan_path=ordinary(plan_path)
    if not re.fullmatch('[0-9a-f]{64}',approved_plan_sha256)or digest(plan_path)!=approved_plan_sha256:raise ValueError('Explicit exact prepared-plan digest required')
    plan=json.loads(plan_path.read_bytes())
    request=parse_request(plan['request']);limits=budgets_for(request)
    if plan['schema']!='pulsemorph.portable_firstuse_plan.v5' or plan['budgets']!=limits:raise ValueError('Fixed finite budget differs')
    for p,h in plan['source_pins'].items():
        if digest(ordinary(p))!=h:raise ValueError('Prepared input changed')
    cfg=load_config(plan['config_path']);request=parse_request(plan['request']);output=ordinary(plan['output']).resolve()
    if output.exists()or output==cfg['artifact_root']or not output.is_relative_to(cfg['artifact_root']):raise ValueError('Output already exists or outside configured root')
    import sys
    if Path(sys.executable).resolve()!=cfg['runtime_python']:raise ValueError('Execute under configured interpreter')
    if request['mode']=='task':
        from .provider_portable_v5 import HostedProvider
        provider=HostedProvider(request['provider'])
        if provider.status()['enabled']is not True:raise ValueError('Provider key absent; zero native work')
    output.mkdir(parents=True,exist_ok=False);write_new(output/'plan.json',plan);write_new(output/'execution_attempt.json',{'maximum_attempts':1,'automatic_retry':False,'completion_unconfirmed':True})
    from ..backend_portable_hybrid_v5 import PortableHybridArmBackend
    from ..runtime_portable_v5 import PersistentEpisode,ActionRejected,fingerprint
    from ..fork_v2 import PreviewSession,PreviewPhysicalFailure
    from ..tasks_v2 import evaluate_task
    started=time.monotonic();used={'online_ONNX':0,'warmup_ONNX':0}
    class SharedBudget:
        def __deepcopy__(self,memo):return self
        def reserve(self,role):
            if time.monotonic()-started>limits['wall_seconds']:raise RuntimeError('Global wall budget exhausted')
            key='warmup_ONNX'if role=='warmup'else'online_ONNX';limit=1 if role=='warmup'else limits['max_total_online_ONNX']
            if used[key]>=limit:raise RuntimeError('Aggregate ONNX/control budget exhausted')
            used[key]+=1
            p=output/'budget_latest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(used));tmp.replace(p)
    budget=SharedBudget()
    class Backend(PortableHybridArmBackend):
        def __init__(self,*a,**k):self.shared_budget=budget;super().__init__(*a,**k)
        def _infer(self,feed,role):self.shared_budget.reserve(role);return super()._infer(feed,role)
    authority={'schema':'pulsemorph.embodied_execution_authority.v1','session_id':output.name,'max_control_steps':limits['max_actual_controls'],'source_pins':plan['source_pins']}
    backend=episode=fork=None;result={'schema':'pulsemorph.portable_firstuse_result.v5','status':'failed','task_verified_actual':False,'physical_completion_unconfirmed':True}
    stage='construct_backend'
    try:
        backend=Backend.start(plan['root'],authority=authority,config=cfg)
        stage='create_episode';episode=PersistentEpisode(backend,output.name,max_control_steps=limits['max_actual_controls'])
        def hold(identity,duration):
            o=episode.observe();return episode.execute({'action_id':identity,'expected_revision':o['revision'],'observation_sha256':fingerprint(o),'tool':'move_joints','arguments':{'duration_s':duration,'joint_targets_rad':{backend.joint_names[0]:o['previous_reference']['q'][0]}}})
        for i in range(2):
            stage='prelude_'+str(i);write_new(output/('prelude_'+str(i)+'.json'),hold('prelude_'+str(i),1.))
        if request['mode']=='hold':
            stage='sampled_hold'
            o=episode.observe();task={'schema':'pulsemorph.embodied_wrist_task.v1','task_id':'owned_stand_hold','body':'right_rubber_hand','frame':'world','target_world_m':o['body_poses']['right_rubber_hand']['position_world_m'],'tolerance_m':.012,'hold_s':.2,'max_sample_gap_s':.05,'min_root_height_m':.6};write_new(output/'frozen_task.json',task)
            assessment=evaluate_task(task,o);transition=hold('hold_measurement',.4);write_new(output/'hold_transition.json',transition)
            samples=episode.sampled_observations('hold_measurement');write_new(output/'hold_samples.json',samples)
            for sample in samples:assessment=evaluate_task(task,sample,assessment['state'])
            result.update(status='completed'if assessment['done']else'stopped',task_verified_actual=assessment['done'],assessment=assessment,new_live_model_calls=0)
        else:
            stage='agent_loop'
            from ..preview_orchestrator_portable_v5 import PreviewOrchestrator
            fork=PreviewSession(episode,max_preview_controls=50,max_total_preview_controls=100)
            loop=PreviewOrchestrator(provider,output_root=cfg['artifact_root'],task_verifier=evaluate_task,max_model_calls=4,max_previews=2,max_sandbox_steps=2000,max_actual_steps=2000,wall_seconds=max(1,600-(time.monotonic()-started)),action_rejection_type=ActionRejected,preview_physical_failure_type=PreviewPhysicalFailure)
            result.update(loop.run(request['instruction'],request['task'],fork,output/'agent_loop'))
        result['physical_completion_unconfirmed']=False
    except Exception as error:result.update(error_type=type(error).__name__,error_stage=stage,retry_allowed=False)
    finally:
        result.update(aggregate_counts=used,elapsed_seconds=time.monotonic()-started,qualification='Portable owned initialization; numerical physical task only, not aesthetics/dance/hardware/general-control proof.')
        write_new(output/'numeric_checkpoint.json',result)
        close_evidence(episode,backend,fork,output,result,plan['source_pins'])
        write_new(output/'result.json',result)
    return result
