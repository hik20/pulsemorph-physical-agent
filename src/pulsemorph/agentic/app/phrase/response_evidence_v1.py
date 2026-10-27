"""Read-only, exact-condition observed motor points; no response interpolation."""
import copy,hashlib,json,math,stat,threading
from io import BytesIO
from pathlib import Path
import numpy as np
from .assets import BODY_SHA,CONTROLLER_SHA,MODEL_SHA,MANIFEST_SHA256,REFERENCE_HASHES
from .exports_v1 import _Reader,relative,linked
from ...phrase_program import GAIN_FIELDS,_arm_indices

ROOT=Path.home()/'.cache/pulsemorph-agentic/analysis/motor-response-v1'
PILOT=ROOT/'prepared_v1';AUDIT=ROOT/'exclusive_v1'
MUSIC_ROOT=Path.home()/'.cache/pulsemorph-agentic/app_phrase_v1/music'
INDEX={'prepared.json':'888bea169d350381c22bbb4b9c53e24c681237d9bd0f8216f6f9360d2ffb4997','completed.json':'dbbce9eacaa63fd0dda6892b2a9427a00f21d0d065ee579d21ed959de9a7d76a','unit_bridge.json':'d4a9aea9a1e166e8a8a8893f0feb2fb9d6bbc8f90cb03c96e341ca445a6d0db9'}
AUDIT_INDEX={'receipt.json':'39993fd2872fe16a0e5288eaf54e8d0fdbb6a5d64961ceb22ebebed54a113cff','final_receipt.json':'1060a20d0277789f0bc255984e06a42a3cb6dd88a10c30565bfec1398293fed5'}
CONDITION={'order':['wave','high5'],'heading_policy':'align_join_heading','other_gains':{'first_right_gain':1.,'second_left_gain':1.,'second_right_gain':1.}}
SCHEMA='pulsemorph.measured_motor_response_evidence.v1'
MAX_VERIFY_READ_ESTIMATE=768*1024**2
class ResponseEvidenceIntegrityError(RuntimeError):pass
class ResponseEvidenceBusy(RuntimeError):pass
_VERIFY_LOCK=threading.Lock()
def _require(value,message):
    if not value:raise ResponseEvidenceIntegrityError(message)
def _canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode('utf8')
def _hash(v):return hashlib.sha256(v).hexdigest()
def _guard(runtime):
    try:
        total=0
        for value in runtime._implementation:
            path=Path(value)
            for part in (path,*path.parents):
                if part.exists() and linked(part):raise ValueError('Linked dependency')
            size=path.stat().st_size
            if size>128*1024**2:raise ValueError('Oversized dependency')
            total+=size
        if total>256*1024**2:raise ValueError('Oversized profile')
        runtime._unchanged()
    except Exception as error:raise ResponseEvidenceIntegrityError('Current response evidence profile integrity failed') from error
def _base(status):return {'schema':SCHEMA,'status':status,'experiment_id':'motor_response_v1','field':'first_left_gain','conditioning':copy.deepcopy(CONDITION),'points':[],'qualification':'Only three observed points in this exact pinned simulation condition, one realization each. No interpolation, calibration, reachability, suggested control, uncertainty, direct same-state policy derivative or transfer to different gains/conditions.'}

def _ordinary_store_paths():
    # lstat sees dangling links/junctions that exists() would misclassify as
    # optional absence. Every known store path and ancestor is a directory.
    paths=set()
    for path in (ROOT,PILOT,AUDIT):paths.update((path,*path.parents))
    for path in paths:
        try:info=path.lstat()
        except FileNotFoundError:continue
        _require(not stat.S_ISLNK(info.st_mode) and not (getattr(info,'st_file_attributes',0)&1024),'Linked response store forbidden')
        _require(stat.S_ISDIR(info.st_mode),'Response store ancestor is not an ordinary directory')

def _raw_sets_unchanged(inventories):
    for label in ('unit','low','high'):
        directory=PILOT/'trials'/label
        for path in (directory,*directory.parents):_require(not linked(path),'Linked response directory changed during verification')
        actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
        _require(actual==set(inventories[label]),'Raw response inventory changed during verification')
        for path in directory.rglob('*'):_require(not linked(path),'Linked response proof changed during verification')

def load_response_evidence(runtime,*,wait_s=0):
    if type(wait_s) not in (int,float) or not 0<=wait_s<=30 or not math.isfinite(wait_s):raise ValueError('Trusted evidence wait must be finite in [0,30]')
    acquired=_VERIFY_LOCK.acquire(timeout=float(wait_s)) if wait_s else _VERIFY_LOCK.acquire(blocking=False)
    if not acquired:raise ResponseEvidenceBusy('Motor response evidence verification already in progress')
    try:
        try:return _load(runtime)
        except ResponseEvidenceIntegrityError:raise
        except (OSError,ValueError,KeyError,TypeError,AttributeError,OverflowError) as error:raise ResponseEvidenceIntegrityError('Saved motor response evidence is incomplete or changed') from error
    finally:_VERIFY_LOCK.release()
def _load(runtime):
    _guard(runtime)
    _ordinary_store_paths()
    if not PILOT.exists() and not AUDIT.exists():
        _guard(runtime);result=_base('unavailable');result['reason']='optional_evidence_store_missing';return result
    _require(PILOT.is_dir() and AUDIT.is_dir(),'Partial motor response evidence store')
    reader=_Reader([Path(runtime.root),ROOT,MUSIC_ROOT,runtime.assets.directory,runtime.python.parent.parent])
    module=Path(__file__);reader.hash(module)
    prepared=reader.json(PILOT/'prepared.json',INDEX['prepared.json'],512*1024);completed=reader.json(PILOT/'completed.json',INDEX['completed.json'],512*1024);bridge=reader.json(PILOT/'unit_bridge.json',INDEX['unit_bridge.json'],512*1024)
    audit=reader.json(AUDIT/'receipt.json',AUDIT_INDEX['receipt.json'],512*1024);final=reader.json(AUDIT/'final_receipt.json',AUDIT_INDEX['final_receipt.json'],512*1024)
    _require(prepared.get('schema')=='pulsemorph.motor_response_prepared.v1' and completed.get('schema')=='pulsemorph.motor_response_completed.v1' and bridge.get('passed') is True and audit.get('historical_unit_bridge_byte_exact') is True and final.get('status')=='passed' and final.get('physical_trials_observed')==3 and final.get('ONNX_calls_observed')==1203 and final.get('physics_steps_observed')==24000 and final.get('runtime_audit_FK_observed')==1200,'Independent response scope/index mismatch')
    _require(prepared['environment']==runtime.environment,'Current response environment differs')
    runtime_pins=runtime.implementation_pins()
    _require(all(prepared['source_pins'].get(path)==digest for path,digest in runtime_pins.items()),'Current response numeric profile/source differs')
    for source,files in REFERENCE_HASHES.items():
        for name,digest in files.items():_require(runtime.assets.manifest['files'][source+'/'+name]==digest,'Response source identity mismatch')
    verifier=copy.copy(runtime);verifier.artifact_root=PILOT
    trials=completed.get('trials');_require(isinstance(trials,list) and [t.get('label') for t in trials]==['unit','low','high'],'Observed point trial assignment mismatch')
    rawfiles=[];estimate=0;results={}
    # Bound all inherited verifier reads before calling it. The estimate counts
    # both current guards, three saved verifiers and own raw rehashes; no hidden
    # model/audit/worker execution is permitted.
    live=list(runtime_pins)
    for path in live:
        p=reader.safe(path);_require(p.stat().st_size<=128*1024**2,'Evidence dependency too large');estimate+=4*p.stat().st_size
    for trial,label,gain,index in zip(trials,('unit','low','high'),(1.,.75,1.25),(-1,0,1)):
        directory=PILOT/'trials'/label
        _require(trial['index']==index and Path(trial['paths']['directory']).absolute()==directory.absolute(),'Point directory/index mismatch')
        program=trial['program'];_require(program['order']==CONDITION['order'] and program['heading_policy']==CONDITION['heading_policy'] and program['first_left_gain']==gain and all(program[k]==1 for k in GAIN_FIELDS if k!='first_left_gain'),'Point conditioning mismatch')
        inventory=audit['trial_raw_inventories'][label];_require(isinstance(inventory,dict) and 1<=len(inventory)<=500,'Bounded immutable raw inventory missing')
        actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()};_require(actual==set(inventory),'Raw response file inventory changed')
        for p in directory.rglob('*'):_require(not linked(p),'Linked response proof forbidden')
        for name,digest in inventory.items():
            path=relative(directory,name);reader.hash(path,digest);rawfiles.append(path);estimate+=3*path.stat().st_size
        saved=reader.json(directory/'result.json',inventory['result.json'],128*1024);request=reader.json(directory/'request.json',inventory['request.json'],128*1024);saved_audit=reader.json(directory/'audit.json',inventory['audit.json'],128*1024)
        _require(saved['program']==program and saved['request_sha256']==_hash((directory/'request.json').read_bytes()) and saved['audit_sha256']==inventory['audit.json'] and saved['environment']==request['environment']==runtime.environment,'Point request/audit/environment binding mismatch')
        expected={**prepared['source_pins']}
        # Harness source pins are preparation evidence, not worker inputs.
        expected={p:h for p,h in expected.items() if Path(p).name not in ('motor_response_v1.py','probe_motor_response_v1.py')}
        for name in ('compile_report.json','common_reference/manifest.json','common_reference/arrays.npz'):expected[str(directory/name)]=inventory[name]
        _require(saved['source_pins']==request['source_pins']==expected,'Point exact source assignment mismatch')
        for path,digest in saved['source_pins'].items():
            p=reader.safe(path);_require(p.stat().st_size<=128*1024**2,'Response source exceeds bound');estimate+=p.stat().st_size
        results[label]=(trial,saved)
    _require(estimate<=MAX_VERIFY_READ_ESTIMATE,'Response proof read estimate exceeds768MiB')
    for trial in trials:verifier._verify_trial(trial)
    points=[];names=tuple(runtime.assets.references()['wave'].metadata['body']['joint_names']);indices=_arm_indices(names)[0];q0=runtime.assets.references()['wave'].joint_position_rad[0,indices]
    table=final.get('point_table');_require(isinstance(table,list) and len(table)==12,'Independent observed table shape mismatch')
    for label,gain in (('low',.75),('unit',1.),('high',1.25)):
        trial,saved=results[label];directory=PILOT/'trials'/label
        qpos=np.load(BytesIO(reader.data(directory/'qpos.npy',limit=2*1024**2)),allow_pickle=False);times=np.load(BytesIO(reader.data(directory/'state_times.npy',limit=128*1024)),allow_pickle=False)
        _require(qpos.shape==(401,36) and times.shape==(401,) and qpos.dtype.kind=='f' and np.isfinite(qpos).all() and np.isfinite(times).all() and np.allclose(times,np.arange(401)*.02,rtol=0,atol=1e-10),'Saved observed array layout invalid')
        keep=(times>=1.2-1e-10)&(times<=3.2+1e-10);values=qpos[keep][:,7+indices];_require(len(values)==101,'Observed feature window invalid')
        measured=float(np.sqrt(np.mean((values-q0)**2)));centered=float(np.sqrt(np.mean((values-values.mean(axis=0))**2)))
        matches=[r for r in table if r.get('trial')==label and r.get('phase')==0 and r.get('side')=='left'];_require(len(matches)==1,'Observed row not unique');row=matches[0]
        _require(row['gain']==gain and abs(row['q0_displacement_rms_rad']-measured)<1e-12 and abs(row['centered_temporal_rms_rad']-centered)<1e-12 and abs(saved['metrics']['arm_rms_rad'][0]-measured)<1e-12,'Independent observed metric binding mismatch')
        point={'gain':gain,'trial_label':label,'own_q0_ratio_to_unit':row['q0_ratio_to_unit'],'temporal_ratio_to_unit':row['temporal_ratio_to_unit'],'q0_displacement_rms_rad':row['q0_displacement_rms_rad'],'centered_temporal_rms_rad':row['centered_temporal_rms_rad'],'physical_gate_passed':row['physical_gate_passed'],'min_root_height_m':row['min_root_height_m'],'mean_joint_rmse_rad':row['mean_joint_rmse_rad'],'trial_result_sha256':audit['trial_raw_inventories'][label]['result.json']}
        for key,value in point.items():
            if key not in ('trial_label','physical_gate_passed','trial_result_sha256'):_require(type(value) in (int,float) and math.isfinite(value) and 0<=value<100,'Nonfinite/out-of-bounds observed value')
        _require(type(point['physical_gate_passed']) is bool and point['physical_gate_passed']==saved['metrics']['physical_gate_passed'],'Observed physical gate mismatch');points.append(point)
    unit=next(p for p in points if p['gain']==1)
    _require(unit['q0_displacement_rms_rad']>1e-8 and unit['centered_temporal_rms_rad']>1e-8,'Observed unit ratio denominator undefined')
    for point in points:_require(abs(point['own_q0_ratio_to_unit']-point['q0_displacement_rms_rad']/unit['q0_displacement_rms_rad'])<1e-12 and abs(point['temporal_ratio_to_unit']-point['centered_temporal_rms_rad']/unit['centered_temporal_rms_rad'])<1e-12,'Observed ratio denominator mismatch')
    reader.finish([module,*rawfiles]);_guard(runtime);_raw_sets_unchanged(audit['trial_raw_inventories'])
    result=_base('available');result.update(points=points,source_profile={'body_profile_sha256':BODY_SHA,'controller_profile_sha256':CONTROLLER_SHA,'policy_ONNX_sha256':MODEL_SHA,'asset_manifest_sha256':MANIFEST_SHA256,'reference_hashes':copy.deepcopy(REFERENCE_HASHES),'initial_carrier_sha256':runtime.assets.manifest['files']['initial_carrier.npz'],'golden_prefix_sha256':runtime.assets.manifest['files']['golden_prefix.npz'],'environment_sha256':_hash(_canonical(runtime.environment))},clock={'execution_s':8,'reference_horizon_s':8.14,'control_dt_s':.02,'physics_dt_s':.001},measurement_windows={'physical_global_s':[1.2,3.2],'source_q0':'Wave source at0s','sample_count':101,'joints_per_arm':7},definitions={'own_q0_ratio_to_unit':'Measured joint-displacement RMS from Wave source q0 divided by same unit trial measure.','temporal_ratio_to_unit':'Window-centered joint temporal RMS divided by unit temporal RMS; distinct from own-q0 goals.'},proof={'prepared_sha256':INDEX['prepared.json'],'independent_receipt_sha256':AUDIT_INDEX['receipt.json'],'independent_final_sha256':AUDIT_INDEX['final_receipt.json'],'immutable_raw_inventories_verified':True,'inherited_runtime_saved_proof_verified':True,'new_FK':0,'new_model':0,'new_ONNX':0,'new_steps':0,'read_estimate_upper_bytes':estimate})
    _require(len(_canonical(result))<=16*1024,'Response DTO exceeds compact bound');return json.loads(_canonical(result))

def _query(query):
    if type(query) is not dict or set(query)!={'schema','field','conditioning','gain'} or query['schema']!='pulsemorph.response_evidence_query.v1' or type(query['field']) is not str or query['field'] not in GAIN_FIELDS:raise ValueError('Exact response evidence query required')
    c=query['conditioning']
    if type(c) is not dict or set(c)!={'order','heading_policy','other_gains'} or type(c['order']) is not list or c['order'] not in (['wave','high5'],['high5','wave']) or c['heading_policy'] not in ('align_join_heading','recorded_world') or type(c['other_gains']) is not dict or set(c['other_gains'])!=set(GAIN_FIELDS)-{query['field']}:raise ValueError('Malformed response conditioning')
    if any(type(v) not in (int,float) or not .75<=v<=1.25 or not math.isfinite(v) for v in c['other_gains'].values()):raise ValueError('Finite bounded other gains required')
    gain=query['gain']
    if gain is not None and (type(gain) not in (int,float) or (type(gain) is float and not math.isfinite(gain))):raise ValueError('Finite observed gain or null required')
    return copy.deepcopy(query)
def query_response_evidence(runtime,query):
    query=_query(query);data=load_response_evidence(runtime)
    if data['status']!='available':return data
    if query['field']!='first_left_gain' or query['conditioning']!=CONDITION:reason='conditioning_mismatch'
    elif query['gain'] is not None and query['gain'] not in (.75,1.,1.25):reason='unobserved_gain'
    else:
        if query['gain'] is not None:data['points']=[p for p in data['points'] if p['gain']==query['gain']]
        data['query']=query;data['applicability']='exact_condition_observed_points_only';return data
    return {'schema':SCHEMA,'status':'not_applicable','reason':reason,'points':[],'qualification':_base('available')['qualification']}
