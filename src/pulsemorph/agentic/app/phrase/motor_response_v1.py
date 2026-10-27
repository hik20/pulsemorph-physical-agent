"""One-factor motor response harness; preparation never runs a worker/compiler."""
import copy,hashlib,json,re,shutil,threading,time
from pathlib import Path
import numpy as np
from ...phrase_program import parse_program,GAIN_FIELDS,compile_program,ProgramCompileError
from .assets import PhraseAssets,PUBLIC_PYTHON,sha
from .runtime import PhraseRuntime
from .music import PhraseMusic

BASE=Path.home()/'.cache/pulsemorph-agentic/analysis/motor-response-v1'
HISTORICAL=Path.home()/'.cache/pulsemorph-agentic/app_phrase_v1/jobs/20261027T112858_b653183b/trials/baseline'
MUSIC_SHA='0e0d161cbf1f109e65af17d5483c6f5172715494391859d24108438e781a6893'
PAYLOADS=('qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy','common_reference/arrays.npz')
NUMERIC_ARCHIVES=('policy_inputs.npz','policy_outputs.npz','substep_telemetry.npz')
def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2).encode('utf8')
def digest(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    with Path(path).open('xb') as f:f.write(encoded(value))
def linked(path):return path.is_symlink() or bool(getattr(path.lstat(),'st_file_attributes',0)&1024)
def safe_output(path):
    path=Path(path).absolute()
    if not path.resolve().is_relative_to(BASE.resolve()) or path.resolve()==BASE.resolve():raise ValueError('Dedicated private motor-response C output required')
    for part in (path,*path.parents):
        if part.exists() and linked(part):raise ValueError('Linked output forbidden')
    return path
def programs():
    result=[]
    for label,index,gain in (('unit',-1,1.),('low',0,.75),('high',1,1.25)):
        p={'schema':'pulsemorph.phrase_program.v1','order':['wave','high5'],'heading_policy':'align_join_heading',**{k:1. for k in GAIN_FIELDS},'rationale':'Preregistered one-factor motor response, no model/goal optimization.'};p['first_left_gain']=gain
        result.append({'label':label,'trial_index':index,'program':parse_program(p)})
    return result
def validate_design(rows):
    if rows!=programs():raise ValueError('Exact unit/low/high one-factor design required')
def _own_pins(root):
    files=[root/'src/pulsemorph/agentic/app/phrase/motor_response_v1.py',root/'experiments/agentic/app/probe_motor_response_v1.py']
    return {str(p):sha(p) for p in files}
def prepare(root,output):
    root=Path(root).resolve();output=safe_output(output);assets=PhraseAssets(root)
    request=json.loads((HISTORICAL/'request.json').read_text(encoding='utf8'));result=json.loads((HISTORICAL/'result.json').read_text(encoding='utf8'))
    # Explicitly use saved environment metadata, not a fresh worker --environment
    # probe. Hash every recorded binary/source; live admission is deferred.
    runtime=PhraseRuntime.__new__(PhraseRuntime);runtime.root=root;runtime.python=PUBLIC_PYTHON.resolve();runtime.assets=assets;runtime.environment=request['environment'];runtime.artifact_root=HISTORICAL.parents[3]
    runtime._implementation=runtime.implementation_pins()
    baseline={'id':result['id'],'index':-1,'program':result['program'],'paths':{'directory':str(HISTORICAL)},'metrics':result['metrics']}
    runtime._verify_trial(baseline)
    if any(baseline['program'][k]!=1 for k in GAIN_FIELDS) or baseline['program']['heading_policy']!='align_join_heading':raise ValueError('Historical aligned unit baseline required')
    music=PhraseMusic.__new__(PhraseMusic);music.root=root;music.directory=Path.home()/'.cache/pulsemorph-agentic/app_phrase_v1/music';music._warnings=[];m=music.get('phrase_original_pulse_v1')
    if m['sha256']!=MUSIC_SHA:raise ValueError('Fixed owned music changed')
    observation=PhraseRuntime._music({'available':True,'path':m['path'],'sha256':m['sha256'],'duration_s':8,'sample_rate':48000,'observed_through_sample':384000})
    records=programs();validate_design(records)
    pins={**runtime._implementation,**_own_pins(root),m['path']:m['sha256']}
    inventory={p.relative_to(HISTORICAL).as_posix():sha(p) for p in HISTORICAL.rglob('*') if p.is_file()}
    if any(linked(p) for p in HISTORICAL.rglob('*')):raise ValueError('Linked historical proof forbidden')
    parent=output
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<1024**3:raise ValueError('Motor response requires1GiB free')
    return {'schema':'pulsemorph.motor_response_prepared.v1','output':str(output),'job_id':'motor_response_v1','design':records,'source_pins':pins,'environment':copy.deepcopy(request['environment']),'environment_provenance':{'historical_request_sha256':sha(HISTORICAL/'request.json'),'fresh_worker_probe_run':False},'music_observation':observation,'music_role':'Identical required runtime provenance in all trials; not a manipulated factor and no model sees music.','historical_unit_directory':str(HISTORICAL),'historical_inventory':inventory,'bridge':{'raw_byte_equal_files':list(PAYLOADS),'all_key_byte_equal_archives':list(NUMERIC_ARCHIVES),'policy_call_json_equal':True,'gate_before_endpoint_trials':True},'budget':{'trials':3,'online_ONNX':1200,'warmup_ONNX':3,'total_ONNX':1203,'physics_steps':24000,'model_calls':0,'retries':0,'reference_compiles':6,'runtime_saved_state_FK':1200,'max_controller_wall_s':900,'max_harness_wall_s':1200},'preflight':{'saved_historical_proof_verified':True,'reference_compilation':'NOT_RUN','fresh_worker_environment':'NOT_RUN','physical_execution':'NOT_RUN'},'initialization':'Same pinned initial_carrier/golden_prefix, fresh worker reset for every trial; no state restoration or branch replay.','qualification':'One-factor total downstream effect within this pinned simulated profile; state/history are mediators, not direct same-state policy sensitivity. One sample per gain, no uncertainty/global feasibility/skill learning/rights claim.'}

def bridge(unit,historical):
    unit=Path(unit);historical=Path(historical);checks={}
    for name in PAYLOADS:checks[name]=sha(unit/name)==sha(historical/name)
    for name in NUMERIC_ARCHIVES:
        with np.load(unit/name,allow_pickle=False) as fresh,np.load(historical/name,allow_pickle=False) as old:
            if set(fresh.files)!=set(old.files):raise RuntimeError('Unit archive key parity failed')
            checks[name]=all(fresh[k].dtype==old[k].dtype and fresh[k].shape==old[k].shape and fresh[k].tobytes()==old[k].tobytes() for k in fresh.files)
    checks['policy_calls.json']=json.loads((unit/'policy_calls.json').read_text())==json.loads((historical/'policy_calls.json').read_text())
    for name in ('request.json','result.json'):
        fresh=json.loads((unit/name).read_text());old=json.loads((historical/name).read_text())
        checks[name+':environment']=fresh['environment']==old['environment']
        checks[name+':counts']=fresh.get('counts')==old.get('counts')
    if not all(checks.values()):raise RuntimeError('Fresh unit bridge mismatch; endpoints forbidden')
    return {'schema':'pulsemorph.motor_response_bridge.v1','passed':True,'checks':checks,'qualification':'Archived bridge reproducibility only; no tolerance or numeric exemptions.'}

def execute(root,output,prepared,authority_path,authority_sha256):
    output=safe_output(output)
    if not re.fullmatch('[a-f0-9]{64}',authority_sha256 or '') or sha(authority_path)!=authority_sha256:raise ValueError('Root authority hash required')
    authority=json.loads(Path(authority_path).read_text(encoding='utf8'))
    expected={'schema':'pulsemorph.motor_response_authority.v1','approved':True,'prepared_sha256':digest(encoded(prepared)),'max_trials':3,'max_ONNX_calls':1203,'max_steps':24000,'model_calls':0,'max_reference_compiles':6,'max_saved_state_FK':1200,'max_controller_wall_s':900,'max_harness_wall_s':1200}
    if any(type(authority.get(k)) is not type(v) or authority.get(k)!=v for k,v in expected.items()):raise ValueError('Root authority does not match exact prepared budget')
    if encoded(json.loads((output/'prepared.json').read_text()))!=encoded(prepared) or prepare(root,output)!=prepared:raise ValueError('Prepared bindings changed')
    write(output/'execution_attempt.json',{'attempted_once':True,'authority_sha256':authority_sha256,'prepared_sha256':digest(encoded(prepared)),'retry_allowed':False})
    trials=[];deadline=time.monotonic()+1200
    def check_deadline():
        if time.monotonic()>deadline:raise RuntimeError('Harness wall budget exhausted')
    try:
        from .entrypoint_v2 import prepare_parent_imports
        prepare_parent_imports(root)
        assets=PhraseAssets(root)
        for row in prepared['design']:
            check_deadline()
            try:reference,report=compile_program(assets.references(),row['program'],reference_id='motor_response_preflight_'+row['label'],sources=[*assets.source_records(),{'path':prepared['music_observation']['path'],'sha256':prepared['music_observation']['sha256']}])
            except ProgramCompileError as error:
                write(output/('compile_preflight_'+row['label']+'_rejected.json'),{'report':error.report,'new_ONNX':0,'new_steps':0});raise
            write(output/('compile_preflight_'+row['label']+'.json'),{'report':report,'numeric_array_sha256':{key:digest(getattr(reference,key).tobytes()) for key in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw')},'new_ONNX':0,'new_steps':0})
            check_deadline()
        runtime=PhraseRuntime(root,PUBLIC_PYTHON,output)
        if runtime.environment!=prepared['environment']:raise RuntimeError('Fresh worker environment differs from frozen preparation')
        if runtime.implementation_pins()!={k:v for k,v in prepared['source_pins'].items() if k not in _own_pins(Path(root)) and k!=prepared['music_observation']['path']}:raise RuntimeError('Runtime pins differ')
        for row in prepared['design']:
            check_deadline()
            for path,h in prepared['source_pins'].items():
                if sha(path)!=h:raise RuntimeError('Frozen source changed')
            for relative,h in prepared['historical_inventory'].items():
                if sha(HISTORICAL/relative)!=h:raise RuntimeError('Historical bridge proof changed')
            label=row['label'];write(output/(label+'_attempt.json'),{'attempted_once':True,'trial_index':row['trial_index'],'program':row['program']})
            cancelled=threading.Event();timer=threading.Timer(max(0.,deadline-time.monotonic()),cancelled.set);timer.daemon=True;timer.start()
            try:trial=runtime.run_trial(row['program'],output/'trials'/label,prepared['job_id'],row['trial_index'],prepared['music_observation'],cancelled)
            finally:timer.cancel()
            check_deadline();trials.append({'label':label,**trial});write(output/(label+'_completed.json'),trial)
            expected=json.loads((output/('compile_preflight_'+label+'.json')).read_text())['numeric_array_sha256']
            with np.load(Path(trial['paths']['directory'])/'common_reference/arrays.npz',allow_pickle=False) as arrays:
                if {key:digest(arrays[key].tobytes()) for key in arrays.files}!=expected:raise RuntimeError('Actual reference differs from exact compile preflight')
            if label=='unit':write(output/'unit_bridge.json',bridge(trial['paths']['directory'],HISTORICAL))
        for path,h in prepared['source_pins'].items():
            if sha(path)!=h:raise RuntimeError('Frozen source changed before completion')
        for relative,h in prepared['historical_inventory'].items():
            if sha(HISTORICAL/relative)!=h:raise RuntimeError('Historical bridge proof changed before completion')
        write(output/'completed.json',{'schema':'pulsemorph.motor_response_completed.v1','trials':trials,'qualification':prepared['qualification'],'new_model_calls':0})
        return trials
    except Exception:
        write(output/'failed.json',{'status':'stopped','completed_labels':[t['label'] for t in trials],'retry_allowed':False,'partial_attempt_counts':'Read preserved worker attempts/results; no inferred zero counts.','new_model_calls':0});raise
