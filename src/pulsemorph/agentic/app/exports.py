"""Portable, hash-checked local results; no inference, simulation or evidence writes.

build_result_bundle(manager, job_id) returns a deterministic ZIP as bytes.
ResultExportError is a safe, fail-closed ValueError for unavailable/invalid results.
Reference arrays are original bytes; the manifest is a labelled portable copy with
absolute provenance paths replaced by non-bundled, content-addressed identities.
"""
from __future__ import annotations
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import zipfile
import numpy as np
from pulsemorph.motion.contracts import MotionReference
from .runtime import PINS, PARAMETERS, TOOLS

MAX_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_RESULT_BYTES = 128 * 1024
MAX_AUDIT_BYTES = 128 * 1024 * 1024
BODY_HASH = 'e3926a6578cafaee38d0a7a3b7e7ebb0031f623c69e5cd34aa97d61e943b7e8e'
CONTROLLER_HASH = '57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139'
JOINT_NAMES = tuple(side+'_'+name+'_joint' for side in ('left','right') for name in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')) + ('waist_yaw_joint','waist_roll_joint','waist_pitch_joint') + tuple(side+'_'+name+'_joint' for side in ('left','right') for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))

class ResultExportError(ValueError):
    """The recorded result cannot safely be exported."""

def _require(condition, message):
    if not condition:
        raise ResultExportError(message)

def _sha(data):
    return hashlib.sha256(data).hexdigest()

def _json(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode('utf-8')

def _linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)) if path.exists() or path.is_symlink() else False

def _path(base, relative):
    _require(isinstance(relative,str) and relative and '\x00' not in relative, 'Invalid artifact path')
    portable=relative.replace('\\','/')
    p=PurePosixPath(portable)
    _require(not p.is_absolute() and not PureWindowsPath(relative).drive and all(x not in ('..','.') for x in p.parts), 'Artifact path escapes result')
    target=base.joinpath(*p.parts)
    for item in [base,*[base.joinpath(*p.parts[:i]) for i in range(1,len(p.parts)+1)]]:
        _require(not _linked(item), 'Linked artifact path is forbidden')
    _require(target.resolve().is_relative_to(base.resolve()), 'Artifact path escapes result')
    _require(target.is_file(), 'Required recorded artifact is missing')
    return target

def _read(path, limit=MAX_AUDIT_BYTES):
    _require(path.stat().st_size<=limit, 'Recorded artifact exceeds export budget')
    data=path.read_bytes()
    _require(len(data)<=limit, 'Recorded artifact exceeds export budget')
    return data

def _safe_text(value):
    # User prose is allowed, but local absolute filenames are not portable metadata.
    if isinstance(value,str):
        return re.sub(r'(?<![\w:/\\.\-])(?:[A-Za-z]:[\\/]|\\\\|/(?!/))[^\s\"<>]*', '[local-path omitted]', value)
    if isinstance(value,list):return [_safe_text(v) for v in value]
    if isinstance(value,dict):return {k:_safe_text(v) for k,v in value.items() if k not in ('path','paths','directory','manifest','log','url')}
    return value

def _has_absolute_text(value):
    if isinstance(value,str):
        return bool(re.search(r'(?:[A-Za-z]:[\\/]|\\\\)',value)) or value.startswith('/')
    if isinstance(value,list):return any(_has_absolute_text(v) for v in value)
    if isinstance(value,dict):return any(_has_absolute_text(v) for v in value.values())
    return False

def _portable_manifest(common):
    result=copy.deepcopy(common);changes=[]
    for i,source in enumerate(result['metadata'].get('provenance',{}).get('sources',[])):
        value=source.get('path')
        if not isinstance(value,str):continue
        if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
            digest=source.get('sha256');_require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Invalid provenance hash')
            suffix=PureWindowsPath(value).suffix.lower()
            suffix=suffix if re.fullmatch(r'\.[a-z0-9]{1,10}',suffix) else '.bin'
            source['path']='external_inputs/'+digest+'/input'+suffix
            changes.append({'field':f'/metadata/provenance/sources/{i}/path','original_path_sha256':_sha(value.encode('utf-8'))})
        else:
            _require('..' not in PurePosixPath(value.replace('\\','/')).parts and not PureWindowsPath(value).drive,'Unsafe provenance path')
    _require(not _has_absolute_text(result), 'Portable reference contains an absolute path outside supported provenance fields')
    return result,changes

README = """# PulseMorph local measured result

reference/ is the selected target MotionReference, not physically realized motion.
Its manifest is a sanitized portable copy: absolute input provenance paths are
replaced by deterministic external_inputs/<sha256>/input.<extension> identities.
Those external files are NOT bundled. Original/exported manifest hashes and the
field transformation locations are recorded in result.json; numeric arrays.npz
is byte-identical. Hashes are integrity bindings, not producer signatures.

realized/ contains exact recorded NumPy arrays from a four-second G1 MuJoCo run.
qpos:201x36, root XYZ metres, root quaternion wxyz, then29 joint angles radians.
qvel:201x35, root linear XYZ m/s (world), root angular XYZ rad/s (body-local),
then29 joint velocities rad/s. state_times:201 seconds including initial state.
applied_joint_targets:200x29 radians, PD targets at each20ms control boundary.
Joint order is in result.json. No raw policy actions are exported.
The target reference extends to4.14s solely for controller future lookahead;
realized execution ends at4s. Reference frame/quaternion conventions are specified
in its own manifest and differ from MuJoCo qpos quaternion order.

Only recorded hashes, contract validation and array layout are checked during
export. No new physics, model call, independent FK or complete safety audit runs.
Recorded metrics are simulation joint proxies, not aesthetics or hardware safety.
No new skill learning, general choreography, recovery or license grant is implied.
No vendor assets, audio, video, raw model prompts, logs or authentication included.
Media has a separate local download/view link. Code/model/motion/mesh/media rights
remain separate and unresolved clip-level rights are not approved by export.
"""

def build_result_bundle(manager, job_id):
    """Return a <=16MiB deterministic portable ZIP, or raise ResultExportError."""
    try:
        return _build(manager, job_id)
    except ResultExportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, OverflowError, zipfile.BadZipFile) as exc:
        raise ResultExportError('Recorded result is incomplete or invalid') from exc

def _build(manager,job_id):
    _require(isinstance(job_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',job_id),'Invalid job identifier')
    job=manager.get(job_id)
    _require(job.get('status')=='complete' and job.get('outcome') in ('target_met','partial','physical_gate_failed'),'Only completed measured results can be exported')
    selected=job.get('selected');_require(isinstance(selected,dict),'No selected measured result')
    metrics=selected.get('metrics',{})
    _require(metrics.get('integrity_passed') is True and not metrics.get('integrity_errors') and len(metrics.get('features',[]))==5,'Selected measurement failed integrity')
    index=selected.get('index');_require(type(index) is int and index>=-1,'Invalid selected trial')
    jobs=Path(manager.jobs_dir).absolute();directory=jobs/job_id/'trials'/('baseline' if index==-1 else f'trial_{index:02d}')
    for parent in [jobs,*directory.parents]:
        _require(not _linked(parent),'Linked result directory is forbidden')
    declared=selected.get('paths',{}).get('directory');_require(declared is not None and Path(declared).resolve()==directory.resolve(),'Selected trial directory mismatch')
    manifest_bytes=_read(_path(directory,'manifest.json'),2*1024*1024);manifest=json.loads(manifest_bytes)
    _require(manifest.get('status')=='success' and manifest.get('error') is None and manifest.get('job_id')==job_id and manifest.get('trial_index')==index,'Trial completion identity mismatch')
    _require(manifest.get('common_contract_validated') is True and manifest.get('strict_g1_admission_passed') is True and manifest.get('preexecution_hashes_unchanged') is True,'Recorded contract admission failed')
    _require(manifest.get('actual_control_steps')==200 and manifest.get('actual_physics_steps')==4000 and manifest.get('online_onnx_call_count')==200 and manifest.get('control_dt')==.02 and manifest.get('physics_dt')==.001,'Unsupported physical trace layout')
    hashes=manifest.get('artifact_hashes',{});_require(isinstance(hashes,dict) and 1<=len(hashes)<=256,'Missing artifact inventory')
    normalized={};records={};audit_bytes=0
    for name,digest in hashes.items():
        normalized_name=name.replace('\\','/');_require(normalized_name not in normalized,'Duplicate artifact inventory path')
        _require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Invalid artifact hash')
        path=_path(directory,name);data=_read(path);audit_bytes+=len(data)
        _require(audit_bytes<=MAX_AUDIT_BYTES,'Artifact audit exceeds export budget')
        _require(_sha(data)==digest,'Recorded artifact hash mismatch')
        normalized[normalized_name]=(path,digest);records[normalized_name]=data if normalized_name in ('proposal.json','common_reference/manifest.json','common_reference/arrays.npz','qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy') else None
    names=['proposal.json','common_reference/manifest.json','common_reference/arrays.npz','qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy']
    _require(all(name in records for name in names),'Required export files are unrecorded')
    proposal=json.loads(records['proposal.json']);source=job.get('request',{}).get('source_id')
    _require(source in PINS and manifest.get('source_id')==source and proposal.get('source_id')==source and proposal.get('job_id')==job_id and proposal.get('trial_index')==index,'Proposal/source identity mismatch')
    _require(manifest.get('proposal_sha256')==_sha(records['proposal.json']) and proposal.get('source_motion',{}).get('sha256')==PINS[source] and manifest.get('motion_sha256')==PINS[source] and manifest.get('source_motion',{}).get('sha256')==PINS[source],'Proposal/source hash mismatch')
    for key,value in proposal.get('parameters',{}).items():
        low,high=(-.3,.3) if key=='timing_shift_s' else (.35,1.6)
        _require(type(value) in (int,float) and np.isfinite(value) and low<=value<=high,'Proposal parameter outside domain')
    _require(proposal.get('parameters')==selected.get('parameters') and set(proposal.get('parameters',{}))==set(PARAMETERS) and proposal.get('control_tool')==selected.get('tool_id') and selected.get('tool_id') in TOOLS,'Selected proposal binding mismatch')
    common=json.loads(records['common_reference/manifest.json']);_require(common.get('arrays',{}).get('path')=='arrays.npz' and common['arrays'].get('sha256')==_sha(records['common_reference/arrays.npz']),'Common archive binding mismatch')
    with zipfile.ZipFile(BytesIO(records['common_reference/arrays.npz'])) as z:
        _require(sum(x.file_size for x in z.infolist())<=MAX_BUNDLE_BYTES,'Common array expansion exceeds budget')
    with np.load(BytesIO(records['common_reference/arrays.npz']),allow_pickle=False) as archive:
        _require(len(archive.files)==4 and set(archive.files)=={'time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw'},'Common array keys do not match contract')
        reference=MotionReference(metadata=common['metadata'],**{key:archive[key] for key in archive.files})
    m=common['metadata'];_require(m['body']['profile_id']=='g1' and m['controller']['profile_id']=='g1-bones-deploy' and m['body']['anchor_body_name']=='torso_link' and m['controller']['control_dt_s']==.02,'Unsupported reference profile')
    _require(tuple(m['body']['joint_names'])==JOINT_NAMES and m['body']['profile_sha256']==manifest.get('body_profile_sha256')==BODY_HASH and m['controller']['profile_sha256']==manifest.get('controller_profile_sha256')==CONTROLLER_HASH,'Body/controller identity mismatch')
    _require(m['context']['task_id']==job_id and m['reference_id']==manifest['candidate_id']+':common' and any(s.get('sha256')==PINS[source] for s in m['provenance']['sources']),'Common proposal identity mismatch')
    _require(np.isclose(reference.time_offsets_s[0],0) and np.isclose(reference.time_offsets_s[-1],4.14),'Unexpected reference time domain')
    shapes={'qpos.npy':(201,36),'qvel.npy':(201,35),'state_times.npy':(201,),'applied_joint_targets.npy':(200,29)}
    arrays={}
    for name,shape in shapes.items():
        header=BytesIO(records[name]);version=np.lib.format.read_magic(header)
        _require(version in ((1,0),(2,0)), 'Unsupported NumPy array format')
        header_shape,fortran,dtype=(np.lib.format.read_array_header_1_0(header) if version==(1,0) else np.lib.format.read_array_header_2_0(header))
        _require(header_shape==shape and dtype.kind=='f' and dtype.itemsize in (4,8),'Invalid realized array layout')
        _require(len(records[name])-header.tell()==int(np.prod(shape))*dtype.itemsize,'Invalid realized array payload size')
        value=np.load(BytesIO(records[name]),allow_pickle=False);_require(value.shape==shape and value.dtype.kind=='f' and np.isfinite(value).all(),'Invalid realized array layout');arrays[name]=value
    _require(np.allclose(arrays['state_times.npy'],np.arange(201)*.02,rtol=0,atol=1e-9),'Invalid realized timestamps')
    portable,changes=_portable_manifest(common)
    # Apply the same existing contract validation to the portable metadata.
    MotionReference(metadata=portable['metadata'],**{key:getattr(reference,key) for key in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw')})
    payload={'reference/manifest.json':_json(portable),'reference/arrays.npz':records['common_reference/arrays.npz'],'README.md':README.encode()}
    payload.update({'realized/'+name:records[name] for name in shapes})
    music=job.get('music',{});music_fields=('schema','id','sha256','sample_rate','duration_s','accent_time_s','provenance')
    metric_fields=('schema','integrity_passed','features','feature_errors','per_dimension_pass','physical_gate_passed','intent_passed','score','utility','min_root_height_m','mean_joint_rmse_rad','audit_scope')
    result={'schema':'pulsemorph.local_result_export.v1','job_id':job_id,'instruction':job.get('request',{}).get('instruction'),'plan':job.get('plan'),'targets':job.get('targets'),'tolerances':job.get('tolerances'),'selected_index':index,'tool_id':selected['tool_id'],'parameters':selected['parameters'],'metrics':{k:metrics[k] for k in metric_fields if k in metrics},'source_id':source,'source_sha256':PINS[source],'body_profile_sha256':BODY_HASH,'controller_profile_sha256':CONTROLLER_HASH,'onnx_sha256':manifest.get('onnx_sha256'),'music':{k:music[k] for k in music_fields if k in music},'joint_names':JOINT_NAMES,'qualification':'Recorded successful trial; inventory/contract/layout checked at download; recorded metrics not recomputed; no fresh FK/physics/model calls. Optional raw actions omitted.','reference_manifest_transformation':{'sanitized_portable_copy':True,'original_sha256':_sha(records['common_reference/manifest.json']),'exported_sha256':_sha(payload['reference/manifest.json']),'absolute_path_redactions':changes,'numeric_arrays_byte_identical':True,'external_inputs_bundled':False},'files':{name:{'sha256':_sha(data),'bytes':len(data)} for name,data in payload.items()}}
    result['instruction']=_safe_text(result['instruction']); result['plan']=_safe_text(result['plan']); result['music']=_safe_text(result['music']); payload['result.json']=_json(result);_require(len(payload['result.json'])<=MAX_RESULT_BYTES,'Result metadata exceeds export budget')
    _require(sum(map(len,payload.values()))<=MAX_BUNDLE_BYTES,'Result exceeds export budget')
    # Re-read inventory and manifest after all validation to close ordinary mutation races.
    _require(_read(_path(directory,'manifest.json'))==manifest_bytes,'Trial manifest changed during export')
    for name,(path,digest) in normalized.items():
        _require(_sha(_read(_path(directory,name)))==digest,'Recorded artifact changed during export')
    _require(manager.get(job_id)==job,'Job state changed during export')
    stream=BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o600)<<16;archive.writestr(info,data)
    _require(stream.tell()<=MAX_BUNDLE_BYTES,'ZIP exceeds export budget')
    return stream.getvalue()
