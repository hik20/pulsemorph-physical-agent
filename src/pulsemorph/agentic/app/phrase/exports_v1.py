"""Portable eight-second selected-result ZIP from saved proof, never execution.

Hashes bind recorded bytes, not signatures/rights/training admission. No runtime
source-list refresh, model, ONNX, physics, FK or saved-state auditor is invoked.
"""
from __future__ import annotations
import copy,hashlib,json,math,re,stat,zipfile
from io import BytesIO
from pathlib import Path,PurePosixPath,PureWindowsPath
import numpy as np
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError,SchemaError
from pulsemorph.motion.contracts import MotionReference
from ...phrase_program import parse_program,GAIN_FIELDS
from ..exports import BODY_HASH,CONTROLLER_HASH,JOINT_NAMES
from .assets import ASSET_ROOT,PUBLIC_PYTHON,MANIFEST_SHA256,MODEL,MODEL_SHA,REFERENCE_HASHES

MAX_BUNDLE_BYTES=16*1024*1024
MAX_RESULT_BYTES=128*1024
MAX_AUDIT_BYTES=128*1024*1024
SOURCE_ID='phrase_wave_high5_v1'
NATIVE_FILES=('mujoco/__init__.py','mujoco/_functions.cp311-win_amd64.pyd','mujoco/_structs.cp311-win_amd64.pyd','mujoco/mujoco.dll','numpy/__init__.py','numpy/core/_multiarray_umath.cp311-win_amd64.pyd','onnxruntime/__init__.py','onnxruntime/capi/onnxruntime_pybind11_state.pyd')
ARRAY_SHAPES={'qpos.npy':(401,36),'qvel.npy':(401,35),'state_times.npy':(401,),'applied_joint_targets.npy':(400,29)}
REF_SHAPES={'time_offsets_s':(408,),'joint_position_rad':(408,29),'joint_velocity_rad_s':(408,29),'anchor_orientation_xyzw':(408,4)}
class PhraseResultExportError(ValueError):pass

def require(condition,message):
    if not condition:raise PhraseResultExportError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2).encode('utf8')
def is_hash(value):return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None
def linked(path):return path.is_symlink() or bool(getattr(path.lstat(),'st_file_attributes',0)&1024)
def relative(root,value):
    require(isinstance(value,str) and value and '\0' not in value,'Invalid recorded artifact path');text=value.replace('\\','/');p=PurePosixPath(text)
    require(not p.is_absolute() and not PureWindowsPath(value).drive and '..' not in p.parts and ':' not in text,'Recorded artifact escapes result')
    return root.joinpath(*p.parts)
def fingerprint(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

class _Reader:
    def __init__(self,roots):self.roots=tuple(Path(p).resolve() for p in roots);self.cache={};self.bytes_read=0
    def safe(self,path):
        path=Path(path).absolute();require(path.is_absolute() and '..' not in path.parts,'Absolute ordinary recorded file required')
        require(any(path.resolve().is_relative_to(root) for root in self.roots),'Recorded source path outside admitted local roots')
        for p in (path,*path.parents):require(not linked(p),'Linked recorded path is forbidden')
        require(path.is_file(),'Required saved proof is missing');return path
    def _charge(self,size):
        self.bytes_read+=size;require(self.bytes_read<=MAX_AUDIT_BYTES,'Saved proof exceeds128MiB read budget')
    def hash(self,path,expected=None):
        path=self.safe(path);before=fingerprint(path)
        if expected is not None:require(is_hash(expected),'Invalid recorded SHA256')
        if path not in self.cache:
            require(before[2]<=MAX_AUDIT_BYTES-self.bytes_read,'Saved proof exceeds128MiB read budget');d=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):self._charge(len(chunk));d.update(chunk)
            require(fingerprint(path)==before,'Recorded file changed during hashing');self.cache[path]={'fingerprint':before,'sha256':d.hexdigest()}
        entry=self.cache[path];require(entry['fingerprint']==before,'Recorded file changed during export')
        if expected is not None:require(entry['sha256']==expected,'Saved proof hash mismatch')
        return entry['sha256']
    def data(self,path,expected=None,limit=2*1024*1024):
        path=self.safe(path);require(path.stat().st_size<=limit,'Saved proof metadata/payload exceeds bound')
        if path in self.cache and 'bytes' in self.cache[path]:
            self.hash(path,expected);return self.cache[path]['bytes']
        before=fingerprint(path);require(before[2]<=MAX_AUDIT_BYTES-self.bytes_read,'Saved proof exceeds128MiB read budget')
        with path.open('rb') as f:data=f.read(limit+1)
        self._charge(len(data));require(len(data)<=limit and fingerprint(path)==before,'Saved file changed or exceeds bound');digest=sha(data)
        if path in self.cache:require(self.cache[path]['fingerprint']==before and self.cache[path]['sha256']==digest,'Saved file changed between hash and read')
        if expected is not None:require(is_hash(expected) and digest==expected,'Saved proof hash mismatch')
        self.cache[path]={'fingerprint':before,'sha256':digest,'bytes':data};return data
    def json(self,path,expected=None,limit=2*1024*1024):return json.loads(self.data(path,expected,limit).decode('utf8'))
    def finish(self,payload_paths):
        # Buffers to be exported are checked against one final bounded byte read;
        # shared large binary proof is deduplicated and final-stat checked.
        for path in payload_paths:
            path=self.safe(path);entry=self.cache[path];limit=MAX_BUNDLE_BYTES
            require(path.stat().st_size<=limit,'Payload exceeds bundle bound')
            with path.open('rb') as f:data=f.read(limit+1)
            self._charge(len(data));require(len(data)<=limit and sha(data)==entry['sha256'],'Payload changed during export')
        for path,entry in self.cache.items():require(fingerprint(self.safe(path))==entry['fingerprint'],'Recorded proof changed during export')

def _npy(data,shape):
    buffer=BytesIO(data);version=np.lib.format.read_magic(buffer);require(version in ((1,0),(2,0)),'Unsupported numeric file version')
    header=np.lib.format.read_array_header_1_0(buffer) if version==(1,0) else np.lib.format.read_array_header_2_0(buffer)
    actual,fortran,dtype=header;require(actual==shape and dtype.kind=='f' and dtype.itemsize==8,'Recorded numeric shape/dtype mismatch');require(len(data)-buffer.tell()==int(np.prod(shape))*8,'Recorded numeric byte length mismatch')
    value=np.load(BytesIO(data),allow_pickle=False);require(np.isfinite(value).all(),'Nonfinite recorded numeric data');return value

def _reference(manifest_data,arrays_data):
    m=json.loads(manifest_data.decode('utf8'));require(set(m.get('arrays',{}))=={'path','sha256'} and m['arrays']['path']=='arrays.npz' and m['arrays']['sha256']==sha(arrays_data),'Reference archive binding mismatch')
    with zipfile.ZipFile(BytesIO(arrays_data)) as archive:
        names=archive.namelist();require(len(names)==4 and set(names)=={k+'.npy' for k in REF_SHAPES},'Reference array keys mismatch');require(sum(i.file_size for i in archive.infolist())<=MAX_BUNDLE_BYTES,'Reference expansion exceeds bound');arrays={k:_npy(archive.read(k+'.npy'),shape) for k,shape in REF_SHAPES.items()}
    reference=MotionReference(metadata=m['metadata'],**arrays);metadata=m['metadata'];require(tuple(metadata['body']['joint_names'])==JOINT_NAMES and metadata['body']['profile_id']=='g1' and metadata['body']['anchor_body_name']=='torso_link' and metadata['body']['profile_sha256']==BODY_HASH and metadata['controller']['profile_id']=='g1-bones-deploy' and metadata['controller']['profile_sha256']==CONTROLLER_HASH and metadata['controller']['control_dt_s']==.02,'Reference body/controller profile mismatch');require(reference.time_offsets_s[-1]==8.14 and np.allclose(reference.time_offsets_s,np.arange(408)*.02,rtol=0,atol=1e-12),'Reference8.14second clock mismatch');return m

def _profile(reader,root):
    manifest=reader.json(ASSET_ROOT/'manifest.json',MANIFEST_SHA256,128*1024)
    expected_files={f'{name}/{file}' for name in ('wave','high5') for file in ('manifest.json','arrays.npz')}|{'initial_carrier.npz','golden_prefix.npz'}
    require(manifest.get('schema')=='pulsemorph.private_phrase_assets.v1' and set(manifest.get('files',{}))==expected_files,'Golden phrase profile inventory mismatch')
    expected={}
    for name,digest in manifest['live_numeric_pins'].items():expected[relative(root,name)]=digest
    require(manifest['live_numeric_pins'].get(MODEL)==MODEL_SHA,'Policy profile pin missing')
    require(manifest['live_numeric_pins'].get('research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml')==BODY_HASH and manifest['live_numeric_pins'].get('research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml')==CONTROLLER_HASH,'Raw profile pins missing')
    for name,digest in manifest['files'].items():expected[relative(ASSET_ROOT,name)]=digest
    expected[ASSET_ROOT/'manifest.json']=MANIFEST_SHA256
    for name,files in REFERENCE_HASHES.items():
        for file,digest in files.items():require(manifest['files'][f'{name}/{file}']==digest,'Two-source lineage is not admitted')
    for path,digest in expected.items():reader.hash(path,digest)
    return expected,manifest

def _inventory(reader,directory,audit):
    inventory=audit.get('inventory');require(isinstance(inventory,dict) and 1<=len(inventory)<=500,'Saved audit inventory missing or too large');normalized={}
    for name,digest in inventory.items():
        text=name.replace('\\','/');require(text not in normalized,'Duplicate inventory artifact');path=relative(directory,name);reader.hash(path,digest);normalized[text]=digest
    required={'request.json','worker_started.json','compile_report.json','qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy','policy_inputs.npz','policy_outputs.npz','policy_calls.json','substep_telemetry.npz','common_reference/manifest.json','common_reference/arrays.npz'}|{f'attempts/call_{i:04d}.json' for i in range(401)}
    require(required<=set(normalized),'Saved control proof inventory incomplete')
    expected=set(normalized)|{'result.json','audit.json','worker.log'}
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual==expected,'Recorded trial contains missing/unrecorded files')
    for p in directory.rglob('*'):require(not linked(p),'Linked saved artifact forbidden')
    return normalized,actual

def _trial(reader,root,jobdir,job_id,index,state,profile,music_path,music):
    directory=jobdir/'trials'/('baseline' if index==-1 else f'trial_{index:02d}')
    require(isinstance(state,dict) and state.get('index')==index and isinstance(state.get('paths'),dict),'Selected/baseline state is invalid')
    require(Path(state['paths'].get('directory','')).absolute()==directory.absolute(),'Selected trial directory assignment mismatch')
    result_bytes=reader.data(directory/'result.json');result=json.loads(result_bytes);request_bytes=reader.data(directory/'request.json');request=json.loads(request_bytes);audit_bytes=reader.data(directory/'audit.json');audit=json.loads(audit_bytes)
    expected_id=job_id+('__baseline' if index==-1 else '__trial_'+str(index))
    require(result.get('schema')=='pulsemorph.app_phrase_physical_trial.v1' and result.get('status')=='completed' and result.get('integrity_passed') is True and result.get('id')==request.get('id')==state.get('id')==expected_id and result.get('job_id')==request.get('job_id')==job_id and result.get('index')==request.get('index')==index,'Trial completion/job identity mismatch')
    require(request.get('schema')=='pulsemorph.app_phrase_request.v1' and result.get('request_sha256')==sha(request_bytes) and result.get('audit_sha256')==sha(audit_bytes),'Trial request/audit hash binding mismatch')
    require(audit.get('schema')=='pulsemorph.app_phrase_audit.v1' and audit.get('integrity_passed') is True and audit.get('fresh_mj_kinematics_calls')==400 and audit.get('new_physics_steps')==0 and audit.get('new_onnx_calls')==0,'Saved audit scope missing')
    counts={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1}
    require(result.get('counts')==counts and result.get('prefix_gate')=={'passed':True,'inputs_checked':19,'outputs_checked':19,'states_checked':19,'targets_checked':18} and result.get('warnings_before')==result.get('warnings_after'),'Saved counts/prefix/warnings mismatch')
    program=parse_program(result.get('program'));require(program['order']==['wave','high5'] and request.get('program')==result['program']==state.get('program'),'Trial program assignment mismatch')
    physical=state.get('physical_metrics',state.get('metrics'));require(isinstance(physical,dict) and physical==result.get('metrics')==audit.get('metrics'),'Physical metric proof binding mismatch')
    rms=physical.get('arm_rms_rad');require(isinstance(rms,list) and len(rms)==4 and all(type(v) in (int,float) and math.isfinite(v) and v>1e-8 for v in rms),'Physical RMS values unavailable')
    require(type(physical.get('physical_gate_passed')) is bool and physical.get('actual_physics_steps')==8000 and physical.get('online_onnx_call_count')==400,'Physical measurement count mismatch')
    for name in ('min_root_height_m','mean_joint_rmse_rad'):require(type(physical.get(name)) in (int,float) and math.isfinite(physical[name]),'Physical measurement nonfinite')
    require(physical['physical_gate_passed']==(physical['min_root_height_m']>=.6 and physical['mean_joint_rmse_rad']<=.35),'Physical screen flag mismatch')
    env=request.get('environment');require(isinstance(env,dict) and env==result.get('environment') and env.get('isolated') is True and env.get('user_site') is False,'Saved isolated environment proof missing')
    require(env.get('versions')=={'numpy':'1.26.4','scipy':'1.16.3','mujoco':'3.3.7','onnxruntime':'1.31.0','torch':'2.7.1'},'Saved controller environment versions differ')
    native={PUBLIC_PYTHON.parent.parent/'Lib/site-packages'/name for name in NATIVE_FILES};native_spec=env.get('module_pins');require(isinstance(native_spec,dict) and {Path(p) for p in native_spec}==native,'Saved native environment inventory mismatch')
    expected=dict(profile)
    for name in ('runtime.py','assets.py','worker.py'):
        path=root/'src/pulsemorph/agentic/app/phrase'/name;expected[path]=None
    expected[PUBLIC_PYTHON]=env.get('python_sha256')
    for path,digest in native_spec.items():expected[Path(path)]=digest
    expected[music_path]=music['sha256']
    for name in ('common_reference/manifest.json','common_reference/arrays.npz','compile_report.json'):expected[directory/name]=None
    pins=request.get('source_pins');require(isinstance(pins,dict) and pins==result.get('source_pins') and {Path(p) for p in pins}==set(expected),'Trial full source/profile assignment mismatch')
    for path,digest in pins.items():
        known=expected[Path(path)];require(known is None or digest==known,'Trial source/profile SHA assignment mismatch');reader.hash(path,digest)
    observation={'available':True,'path':str(music_path),'sha256':music['sha256'],'duration_s':8.,'sample_rate':48000,'observed_through_sample':384000}
    require(request.get('music_observation')==result.get('music_observation')==observation,'Saved planning-audio binding mismatch')
    normalized,raw_files=_inventory(reader,directory,audit)
    started=reader.json(directory/'worker_started.json');require(started.get('status')=='started_once' and started.get('request_sha256')==sha(request_bytes),'Worker start binding missing')
    calls=reader.json(directory/'policy_calls.json');require(isinstance(calls,list) and len(calls)==401,'Saved policy call count mismatch')
    for i,row in enumerate(calls):
        require(row.get('call_index')==i and row.get('control_index')==(None if i==0 else i-1) and row.get('role')==('warmup' if i==0 else 'control') and not row.get('error') and row.get('output_names')==['actions','joint_pos_targets','stiffness_targets','damping_targets'],'Saved policy call layout invalid')
        require(reader.json(directory/f'attempts/call_{i:04d}.json')=={'index':i,'status':'attempted_before_session_run'},'Durable policy attempt missing')
    manifest_path=directory/'common_reference/manifest.json';arrays_path=directory/'common_reference/arrays.npz';manifest_bytes=reader.data(manifest_path,normalized['common_reference/manifest.json']);arrays_bytes=reader.data(arrays_path,normalized['common_reference/arrays.npz']);common=_reference(manifest_bytes,arrays_bytes);metadata=common['metadata']
    require(metadata['reference_id']==expected_id and metadata['context']['task_id']==job_id and metadata['context']['runtime_session_id']==expected_id and metadata['producer']['id']=='phrase-program-reference' and metadata['producer']['version']=='1','Reference execution context mismatch')
    canonical_program=json.dumps(program,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
    require(metadata['provenance'].get('phrase_program')==program and metadata['provenance'].get('phrase_program_sha256')==sha(canonical_program),'Reference program hash mismatch')
    require(metadata['observation'].get('mode')=='offline_file' and metadata['observation'].get('observed_through_time_s')==8 and metadata['observation'].get('audio_observed_through_sample')==384000,'Reference observed audio scope mismatch')
    data={};arrays={}
    for name,shape in ARRAY_SHAPES.items():data[name]=reader.data(directory/name,normalized[name]);arrays[name]=_npy(data[name],shape)
    require(np.allclose(arrays['state_times.npy'],np.arange(401)*.02,rtol=0,atol=1e-10),'Realized8second clock mismatch')
    return {'directory':directory,'result':result,'request':request,'physical':physical,'program':program,'common':common,'manifest_bytes':manifest_bytes,'arrays_bytes':arrays_bytes,'data':data,'raw_files':raw_files,'payload_paths':[manifest_path,arrays_path,*[directory/n for n in ARRAY_SHAPES]],'proof':{'result_sha256':sha(result_bytes),'request_sha256':sha(request_bytes),'audit_sha256':sha(audit_bytes),'source_pin_count':len(pins),'inventory_file_count':len(normalized)}}

def _model_plan(reader,jobdir,job):
    calls=job.get('model_calls');require(isinstance(calls,list) and 1<=len(calls)<=5,'Completed real model plan proof required');initial=None
    for index,entry in enumerate(calls):
        require(isinstance(entry,dict) and entry.get('index')==index and entry.get('completion_unconfirmed') is False and entry.get('live') is True,'Unconfirmed/nonlive model evidence cannot export')
        matches=list((jobdir/'model_calls').glob(f'{index:02d}_*'));require(len(matches)==1 and matches[0].is_dir() and not linked(matches[0]),'Model-call directory missing or ambiguous');directory=matches[0]
        request_data=reader.data(directory/'request.json',limit=128*1024);response_data=reader.data(directory/'response.json',limit=128*1024);request=json.loads(request_data);response=json.loads(response_data);receipt=reader.json(directory/'receipt.json');attempt=reader.json(directory/'attempt.json')
        require(receipt==entry and receipt.get('request_sha256')==sha(request_data) and receipt.get('response_sha256')==sha(response_data) and receipt.get('model_requested')=='gpt-6.1-sol' and receipt.get('tool_events')==0 and receipt.get('response_rejected') is False and attempt.get('attempted') is True,'Model request/response/receipt binding mismatch')
        Draft202012Validator(request['schema']).validate(response)
        if index==0:
            require(receipt.get('purpose')=='plan' and response.get('supported') is True and response.get('unsupported_requests')==[],'Unsupported intent cannot export');initial=copy.deepcopy(response);parse_program(initial['program'])
    require(initial['program']['order']==['wave','high5'],'Unadmitted source order')
    return initial

def _prose(value,known_paths):
    if not isinstance(value,str):return value
    value.encode('utf8')
    for path in sorted(known_paths,key=len,reverse=True):
        if path:value=value.replace(path,'[local-path omitted]')
    value=re.sub(r'https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?[^\s\"<>]*','[local-URL omitted]',value,flags=re.I)
    pattern=r'(?<![\w:/\\.\-])(?:[A-Za-z]:[\\/]|\\\\|/(?!/))[^\s\"<>]*'
    def redact(match):
        token=match.group()
        # Slash-prefixed ratio prose is not treated as an inferred filesystem
        # path. Any actual observed absolute source path was removed above.
        if re.fullmatch(r'/[^/\s]*\d+/\d+[.,;:]?',token):return token
        return '[local-path omitted]'
    return re.sub(pattern,redact,value)

def _portable_manifest(common,known_paths):
    result=copy.deepcopy(common);changes=[]
    sources=result['metadata']['provenance']['sources']
    for index,source in enumerate(sources):
        path=source.get('path');require(isinstance(path,str) and is_hash(source.get('sha256')),'Malformed reference source lineage')
        absolute=PureWindowsPath(path).is_absolute() or PurePosixPath(path).is_absolute()
        if absolute:
            suffix=PureWindowsPath(path).suffix.lower();suffix=suffix if re.fullmatch(r'\.[a-z0-9]{1,10}',suffix) else '.bin'
            source['path']='external_inputs/'+source['sha256']+'/input'+suffix;changes.append({'field':f'/metadata/provenance/sources/{index}/path','original_path_sha256':sha(path.encode('utf8'))})
        else:require(not PureWindowsPath(path).drive and '..' not in PurePosixPath(path.replace('\\','/')).parts and not re.match(r'[a-z]+://',path,re.I),'Unsafe relative reference provenance')
    def check(value):
        if isinstance(value,dict):
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
        elif isinstance(value,str):require(_prose(value,known_paths)==value,'Absolute/private path outside permitted reference provenance transformation')
    check(result)
    return result,changes

def _metrics(state,physical,baseline,targets):
    m=state.get('metrics');require(isinstance(m,dict) and m.get('integrity_passed') is True,'Selected goal measurement missing integrity')
    values=(np.asarray(physical['arm_rms_rad'],dtype=float)/np.asarray(baseline['arm_rms_rad'],dtype=float)).tolist();errors=[abs(a-b)/.08 for a,b in zip(values,targets)];passes=[e<=1 for e in errors];gate=physical['physical_gate_passed'] and baseline['physical_gate_passed'];worst=max(errors);l2=math.sqrt(sum(e*e for e in errors))
    require(m.get('features')==values and m.get('targets')==targets and m.get('tolerances')==[.08]*4 and m.get('normalized_errors')==errors and m.get('per_dimension_pass')==passes and m.get('goals_passed')==sum(passes) and m.get('physical_gate_passed')==gate and m.get('intent_passed')==(all(passes) and gate) and m.get('worst_error')==worst and m.get('utility_l2')==l2,'Fixed goal/physical measurement binding mismatch')
    for key,value in physical.items():require(m.get(key)==value,'Raw physical fields changed in goal wrapper')
    return {'features':values,'targets':targets,'tolerances':[.08]*4,'normalized_errors':errors,'per_dimension_pass':passes,'goals_passed':sum(passes),'total':4,'physical_gate_passed':gate,'intent_passed':all(passes) and gate,'worst_error':worst,'utility_l2':l2}

SELECTION_POLICY='physical gate, worst normalized error, utility_l2, earlier index'

def _music_provenance(music):
    """Export a finite data transfer object, never arbitrary saved metadata."""
    p=music.get('provenance');require(isinstance(p,dict) and p.get('source_bytes_retained') is False,'Music provenance missing')
    if music['id']=='phrase_original_pulse_v1':
        require(p.get('kind')=='procedural_owned' and is_hash(p.get('generator_sha256')),'Owned music provenance missing')
        return {k:p[k] for k in ('kind','generator_sha256','source_bytes_retained')}
    fields=('kind','source_sha256','source_bytes','source_bytes_retained','original_frame_count','original_sample_rate','original_channels','selected_frame_count','start_frame','start_s','end_s','transform','derived_wav_sha256','derived_frame_count')
    require(all(k in p for k in fields),'Upload provenance incomplete')
    rate=p['original_sample_rate'];frames=p['original_frame_count'];begin=p['start_frame']
    require(type(rate) is int and 8000<=rate<=96000 and type(frames) is int and 8*rate<=frames<=120*rate and type(begin) is int and 0<=begin<=frames-8*rate,'Upload source frame lineage invalid')
    require(p['kind']=='local_uploaded_PCM16' and is_hash(p['source_sha256']) and type(p['source_bytes']) is int and 44<=p['source_bytes']<=32*1024*1024 and type(p['original_channels']) is int and p['original_channels'] in (1,2),'Upload source lineage invalid')
    require(p['transform']=='pcm16_downmix_resample_poly_48k_8s_v1' and p['selected_frame_count']==8*rate and p['start_s']==begin/rate and p['end_s']==(begin+8*rate)/rate and p['derived_frame_count']==384000 and p['derived_wav_sha256']==music['sha256'],'Upload crop lineage invalid')
    identity={'source_sha256':p['source_sha256'],'start_frame':begin,'duration_s':8,'transform':p['transform']}
    require(music['id']=='phrase_upload_'+sha(json.dumps(identity,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8'))[:24],'Upload identity mismatch')
    return {k:p[k] for k in fields}

def _result_paths_safe(value,known_paths,key=None):
    if isinstance(value,dict):
        for k,v in value.items():
            require(isinstance(k,str) and _prose(k,known_paths)==k,'Private path in portable result key')
            _result_paths_safe(v,known_paths,k)
    elif isinstance(value,list):
        for v in value:_result_paths_safe(v,known_paths,key)
    elif isinstance(value,str):
        if key=='field' and re.fullmatch(r'/metadata/provenance/sources/\d+/path',value):return
        require(_prose(value,known_paths)==value,'Private path in portable result metadata')

README='''# Private eight-second measured phrase result

reference/ is the selected target MotionReference, not realized motion. Its
numeric arrays.npz is byte-identical; manifest.json is a sanitized portable copy.
ONLY absolute provenance source-path values become deterministic
external_inputs/<sha256>/input.ext identities. These external files are NOT
bundled. Original/exported manifest hashes and JSON-pointer/path-hash changes
are recorded; no original absolute strings/original manifest are bundled.
The target covers0..8.14s for controller lookahead; realized execution is0..8s.

realized/ is exact recorded physical NumPy data, not a kinematic target replay:
qpos401x36: world XYZ metres, pelvis/root quaternion wxyz, then29joint radians.
qvel401x35: world root linear XYZ m/s, body-local root angular XYZ rad/s, then
29joint velocities rad/s. state_times401seconds includes initial state.
applied_joint_targets400x29radians is the actual PD target at each20msboundary.
Joint order is in result.json. Reference orientation is torso xyzw in its own
recorded-reference frame; do not interpret root wxyz as that torso target.

Saved request/result/audit/inventory/source/environment hashes, count/layout,
contract and ratio-consistency checks ran during download. No model/ONNX,
physics, FK, fresh saved-state auditor or metric recomputation from raw dynamics
ran. Existing scalar-substep flags are not full-state dynamics replay.
The completed same-heading unit-gain baseline defines the fixed four RMS goals;
program reference gains and realized RMS ratios remain separate. Selection is
FINAL job selection, independent of any inspected UI row.

Only these eight files are bundled. No source assets, audio, video, policy
weights, logs, authentication, raw prompts or raw actions are included. Shared
playback soundtrack/media have separate links. This is a private developmental
simulated artifact, not a shared TrialRecord/training or hardware admission,
new skill/style learning, aesthetics, beat alignment, recovery, broad safety or
redistribution license grant. Hashes are integrity bindings, not signatures or
authenticated rights. Code/model/motion/robot/mesh/media rights remain separate.
'''

def build_phrase_result_bundle(manager,job_id):
    try:return _build(manager,job_id)
    except PhraseResultExportError:raise
    except (OSError,ValueError,KeyError,TypeError,OverflowError,zipfile.BadZipFile,AttributeError,ValidationError,SchemaError) as error:raise PhraseResultExportError('Saved phrase result is incomplete or invalid') from error

def _build(manager,job_id):
    require(isinstance(job_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',job_id),'Invalid phrase job identifier');job=manager.get(job_id)
    require(isinstance(job,dict) and job.get('id')==job_id and job.get('source_id')==SOURCE_ID and job.get('status')=='complete' and job.get('outcome') in ('target_met','partial','physical_gate_failed'),'Only completed measured phrase results can export')
    require(isinstance(job.get('selected'),dict) and isinstance(job.get('baseline'),dict),'Measured final selection and fresh baseline required')
    index=job['selected'].get('index');require(type(index) is int and -1<=index<=2,'Invalid final selected index')
    root=Path(manager.root).resolve();jobs=Path(manager.jobs_dir).resolve();jobdir=jobs/job_id;reader=_Reader([root,jobs.parent,ASSET_ROOT,PUBLIC_PYTHON.parent.parent]);disk_job=reader.json(jobdir/'state.json');require(disk_job==job,'Memory/disk selected job state mismatch')
    if index==-1:require(job['selected']==job['baseline'],'Selected baseline identity mismatch')
    else:require(any(t==job['selected'] and t.get('index')==index and t.get('status')=='measured' for t in job.get('trials',[])),'Selected candidate is not the recorded measured trial')
    implementation=job.get('implementation_hashes');require(isinstance(implementation,dict) and 1<=len(implementation)<=256,'Job implementation evidence missing')
    for path,digest in implementation.items():require(Path(path).resolve().is_relative_to(root) and Path(path).suffix in ('.py','.js','.ts','.css','.html','.ps1'),'Job implementation path is not own source');reader.hash(path,digest)
    initial=_model_plan(reader,jobdir,job);targets=[initial['program'][k] for k in GAIN_FIELDS];require(job.get('targets')==targets and job.get('tolerances')==[.08]*4 and job.get('plan',{}).get('fixed_targets')==targets and job['plan'].get('program')==initial['program'] and job['plan'].get('supported') is True and job['plan'].get('unsupported_requests')==[],'Fixed original intent-plan targets missing')
    music=job.get('music');require(isinstance(music,dict) and isinstance(music.get('id'),str) and re.fullmatch(r'phrase_(?:original_pulse_v1|upload_[a-f0-9]{24})',music['id']),'Recorded music identity missing');music_path=jobs.parent/'music'/(music['id']+'.wav');music_bytes=reader.data(music_path,music.get('sha256'),800*1024)
    import wave
    with wave.open(BytesIO(music_bytes),'rb') as audio:require((audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),audio.getnframes())==(1,2,48000,384000) and len(audio.readframes(384000))==768000,'Recorded planning music is not complete8second PCM')
    descriptor=reader.json(jobs.parent/'music'/(music['id']+'.json'),limit=128*1024);require(descriptor==music,'Recorded job/music descriptor mismatch')
    profile,profile_manifest=_profile(reader,root);baseline=_trial(reader,root,jobdir,job_id,-1,job['baseline'],profile,music_path,music);selected=baseline if index==-1 else _trial(reader,root,jobdir,job_id,index,job['selected'],profile,music_path,music)
    require(all(baseline['program'][k]==1. for k in GAIN_FIELDS) and baseline['program']['heading_policy']==selected['program']['heading_policy']==initial['program']['heading_policy'],'Same-heading unit-gain baseline missing')
    goals=_metrics(job['selected'],selected['physical'],baseline['physical'],targets);baseline_goals=_metrics(job['baseline'],baseline['physical'],baseline['physical'],targets)
    known_paths={str(p) for p in reader.cache};known_paths.update(str(p).replace('\\','/') for p in reader.cache)
    for source in selected['common']['metadata']['provenance']['sources']:
        if PureWindowsPath(source['path']).is_absolute() or PurePosixPath(source['path']).is_absolute():known_paths.add(source['path'])
    portable,changes=_portable_manifest(selected['common'],known_paths);new_manifest=encoded(portable);_reference(new_manifest,selected['arrays_bytes'])
    payload={'README.md':README.encode('utf8'),'reference/manifest.json':new_manifest,'reference/arrays.npz':selected['arrays_bytes'],**{'realized/'+name:data for name,data in selected['data'].items()}}
    sources=[]
    for name in ('wave','high5'):
        data=reader.json(ASSET_ROOT/name/'manifest.json',REFERENCE_HASHES[name]['manifest.json']);sources.append({'source_id':name,'reference_id':data['metadata']['reference_id'],'manifest_sha256':REFERENCE_HASHES[name]['manifest.json'],'arrays_sha256':REFERENCE_HASHES[name]['arrays.npz'],'recorded_provenance_sha256':[s['sha256'] for s in data['metadata']['provenance']['sources']]})
    music_fields=('id','label','sha256','duration_s','sample_rate','sample_frames','channels','pcm_bits','accent_time_s','provenance','rights');music_summary={k:copy.deepcopy(music[k]) for k in music_fields if k in music}
    result={'schema':'pulsemorph.private_phrase_result_bundle.v1','job_id':job_id,'source_id':SOURCE_ID,'instruction':_prose(job['request']['instruction'],known_paths),'intent_summary':_prose(initial['intent_summary'],known_paths),'selected_index':index,'final_selection_policy':job.get('selection_policy'),'selected_program':{**selected['program'],'rationale':_prose(selected['program']['rationale'],known_paths)},'fixed_targets':targets,'tolerances':[.08]*4,'goal_measurement':goals,'selected_physical_metrics':selected['physical'],'baseline_physical_metrics':baseline['physical'],'baseline_goal_measurement':baseline_goals,'physical_feature_windows_s':[[1.2,3.2],[5.2,7.2]],'measurement_basis':'arm joint-displacement RMS ratios vs fresh same-heading unit-gain baseline; own Wave0/High50 centers; recorded audited values only','body_profile_sha256':BODY_HASH,'controller_profile_sha256':CONTROLLER_HASH,'policy_ONNX_sha256':MODEL_SHA,'golden_profile_manifest_sha256':MANIFEST_SHA256,'joint_names':JOINT_NAMES,'source_lineages':sources,'planning_music':music_summary,'clock':{'execution_s':8,'reference_end_s':8.14,'control_dt_s':.02,'physics_dt_s':.001,'state_count':401,'PD_rows':400},'saved_proof':{'baseline':baseline['proof'],'selected':selected['proof']},'reference_manifest_transformation':{'sanitized_portable_copy':True,'original_sha256':sha(selected['manifest_bytes']),'exported_sha256':sha(new_manifest),'changed_fields':changes,'absolute_path_redaction_count':len(changes),'numeric_arrays_byte_identical':True,'external_inputs_bundled':False},'qualification':'Private developmental simulated result; saved-proof consistency only, no new inference/physics/FK/auditor, no shared TrialRecord/training/rights admission. Model interpretation is not independent semantic validation.','files':{name:{'sha256':sha(data),'bytes':len(data)} for name,data in payload.items()}}
    # These free-form strings may contain user prose; preserve ratios and public
    # URLs, omit only local observed paths/absolute tokens and loopback URLs.
    require(result['final_selection_policy']==SELECTION_POLICY,'Unknown final selection policy')
    result['planning_music']['provenance']=_music_provenance(music)
    result['array_layout']={'qpos':{'shape':[401,36],'units':'world XYZ metres; root quaternion wxyz; joint radians'},'qvel':{'shape':[401,35],'units':'world linear m/s; root body-local angular rad/s; joint rad/s'},'state_times':{'shape':[401],'units':'seconds'},'applied_joint_targets':{'shape':[400,29],'units':'radians'},'reference_orientation':'torso quaternion xyzw; recorded-reference frame'}
    for key in ('label','rights'):
        if key in result['planning_music']:result['planning_music'][key]=_prose(result['planning_music'][key],known_paths)
    _result_paths_safe(result,known_paths)
    payload['result.json']=encoded(result);require(len(payload['result.json'])<=MAX_RESULT_BYTES and sum(len(v) for v in payload.values())<=MAX_BUNDLE_BYTES,'Portable phrase result exceeds bundle budget')
    reader.finish(selected['payload_paths']);require(manager.get(job_id)==job and reader.json(jobdir/'state.json')==job,'Job selection changed during export')
    for trial in (baseline,selected):require({p.relative_to(trial['directory']).as_posix() for p in trial['directory'].rglob('*') if p.is_file()}==trial['raw_files'],'Saved trial changed during export')
    stream=BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=(stat.S_IFREG|0o600)<<16;archive.writestr(info,data)
    require(stream.tell()<=MAX_BUNDLE_BYTES,'Compressed result exceeds bundle budget');return stream.getvalue()
