"""Versioned goal-aware saved-proof export; old jobs retain frozen exporter."""
from . import exports_v1 as legacy
from .goals_v1 import validate_goals
from .manager_goals_v4 import INITIAL_SCHEMA_V4
from .exports_v1 import (require,Path,re,SOURCE_ID,_Reader,ASSET_ROOT,PUBLIC_PYTHON,copy,_model_plan,GAIN_FIELDS,_profile,_trial,_metrics,PureWindowsPath,PurePosixPath,_portable_manifest,encoded,_reference,README,REFERENCE_HASHES,_prose,BODY_HASH,CONTROLLER_HASH,MODEL_SHA,MANIFEST_SHA256,JOINT_NAMES,sha,SELECTION_POLICY,_music_provenance,_result_paths_safe,MAX_RESULT_BYTES,MAX_BUNDLE_BYTES,BytesIO,zipfile,stat,PhraseResultExportError,ValidationError,SchemaError)

def validate_goal_binding(job,initial,prepared,request):
    goal_dto=validate_goals(initial.get('goals'));origin=job.get('goal_origin')
    require(job.get('schema')=='pulsemorph.phrase_app_job.v2' and job.get('plan',{}).get('goals')==goal_dto,'Separate goal DTO missing')
    require(origin in ('explicit_user','model_interpretation') and job['plan'].get('goal_origin')==origin,'Goal origin missing')
    require((origin=='explicit_user' and validate_goals(job['request'].get('goals'))==goal_dto) or (origin=='model_interpretation' and 'goals' not in job['request']),'Authoritative explicit goals/model interpretation binding mismatch')
    require(prepared.get('schema')=='pulsemorph.phrase_app_prepared_plan.v4' and prepared.get('explicit_goals')==job['request'].get('goals') and prepared.get('instruction')==job['request']['instruction'],'Prepared user goals/instruction binding mismatch')
    require(prepared.get('response_schema')==request.get('schema')==INITIAL_SCHEMA_V4 and prepared.get('prompt')==request.get('prompt'),'Prepared model request binding mismatch')
    require(prepared.get('computed_music_summary',{}).get('sha256')==job.get('music',{}).get('sha256'),'Prepared measured music binding mismatch')
    return goal_dto

def build_phrase_result_bundle(manager,job_id):
    job=manager.get(job_id)
    require(isinstance(job,dict),'Unknown phrase job schema')
    if job.get('schema')=='pulsemorph.phrase_app_job.v1':return legacy.build_phrase_result_bundle(manager,job_id)
    require(job.get('schema')=='pulsemorph.phrase_app_job.v2','Unknown phrase job schema')
    try:return _build(manager,job_id)
    except PhraseResultExportError:raise
    except (OSError,ValueError,KeyError,TypeError,OverflowError,zipfile.BadZipFile,AttributeError,ValidationError,SchemaError) as error:raise PhraseResultExportError('Saved phrase result is incomplete or invalid') from error

def attach_goal_metadata(result,goal_dto,origin):
    result['goals']=validate_goals(goal_dto)
    result['goal_origin']=origin
    result['goal_qualification']='Explicit API goals are authoritative; model-interpreted goals have no independent semantic validation. Reference gains remain separate controls.'
    return result

README_V4=README.replace('The completed same-heading unit-gain baseline defines the fixed four RMS goals;','The four desired RMS goals are frozen BEFORE baseline measurement: explicit API\ngoals are authoritative, or model interpretation is visibly labelled. The\ncompleted same-heading unit-gain baseline supplies ONLY the ratio denominator;')

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
    initial=_model_plan(reader,jobdir,job);prepared=reader.json(jobdir/'prepared_plan_v4.json');plan_request=reader.json(next((jobdir/'model_calls').glob('00_*/request.json')));goal_dto=validate_goal_binding(job,initial,prepared,plan_request);targets=goal_dto['ratios'];require(job.get('schema')=='pulsemorph.phrase_app_job.v2' and job.get('plan',{}).get('goals')==goal_dto,'Separate goal DTO missing');origin=job.get('goal_origin');require(origin in ('explicit_user','model_interpretation') and job['plan'].get('goal_origin')==origin,'Goal origin missing');require((origin=='explicit_user' and validate_goals(job['request'].get('goals'))==goal_dto) or (origin=='model_interpretation' and 'goals' not in job['request']),'Authoritative explicit goals/model interpretation binding mismatch');require(job.get('targets')==targets and job.get('tolerances')==[.08]*4 and job.get('plan',{}).get('fixed_targets')==targets and job['plan'].get('program')==initial['program'] and job['plan'].get('supported') is True and job['plan'].get('unsupported_requests')==[],'Fixed original intent-plan targets missing')
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
    payload={'README.md':README_V4.encode('utf8'),'reference/manifest.json':new_manifest,'reference/arrays.npz':selected['arrays_bytes'],**{'realized/'+name:data for name,data in selected['data'].items()}}
    sources=[]
    for name in ('wave','high5'):
        data=reader.json(ASSET_ROOT/name/'manifest.json',REFERENCE_HASHES[name]['manifest.json']);sources.append({'source_id':name,'reference_id':data['metadata']['reference_id'],'manifest_sha256':REFERENCE_HASHES[name]['manifest.json'],'arrays_sha256':REFERENCE_HASHES[name]['arrays.npz'],'recorded_provenance_sha256':[s['sha256'] for s in data['metadata']['provenance']['sources']]})
    music_fields=('id','label','sha256','duration_s','sample_rate','sample_frames','channels','pcm_bits','accent_time_s','provenance','rights');music_summary={k:copy.deepcopy(music[k]) for k in music_fields if k in music}
    result={'schema':'pulsemorph.private_phrase_result_bundle.v2','job_id':job_id,'source_id':SOURCE_ID,'instruction':_prose(job['request']['instruction'],known_paths),'intent_summary':_prose(initial['intent_summary'],known_paths),'selected_index':index,'final_selection_policy':job.get('selection_policy'),'selected_program':{**selected['program'],'rationale':_prose(selected['program']['rationale'],known_paths)},'fixed_targets':targets,'tolerances':[.08]*4,'goal_measurement':goals,'selected_physical_metrics':selected['physical'],'baseline_physical_metrics':baseline['physical'],'baseline_goal_measurement':baseline_goals,'physical_feature_windows_s':[[1.2,3.2],[5.2,7.2]],'measurement_basis':'arm joint-displacement RMS ratios vs fresh same-heading unit-gain baseline; own Wave0/High50 centers; recorded audited values only','body_profile_sha256':BODY_HASH,'controller_profile_sha256':CONTROLLER_HASH,'policy_ONNX_sha256':MODEL_SHA,'golden_profile_manifest_sha256':MANIFEST_SHA256,'joint_names':JOINT_NAMES,'source_lineages':sources,'planning_music':music_summary,'clock':{'execution_s':8,'reference_end_s':8.14,'control_dt_s':.02,'physics_dt_s':.001,'state_count':401,'PD_rows':400},'saved_proof':{'baseline':baseline['proof'],'selected':selected['proof']},'reference_manifest_transformation':{'sanitized_portable_copy':True,'original_sha256':sha(selected['manifest_bytes']),'exported_sha256':sha(new_manifest),'changed_fields':changes,'absolute_path_redaction_count':len(changes),'numeric_arrays_byte_identical':True,'external_inputs_bundled':False},'qualification':'Private developmental simulated result; saved-proof consistency only, no new inference/physics/FK/auditor, no shared TrialRecord/training/rights admission. Model interpretation is not independent semantic validation.','files':{name:{'sha256':sha(data),'bytes':len(data)} for name,data in payload.items()}}
    # These free-form strings may contain user prose; preserve ratios and public
    # URLs, omit only local observed paths/absolute tokens and loopback URLs.
    require(result['final_selection_policy']==SELECTION_POLICY,'Unknown final selection policy')
    result['planning_music']['provenance']=_music_provenance(music)
    result['array_layout']={'qpos':{'shape':[401,36],'units':'world XYZ metres; root quaternion wxyz; joint radians'},'qvel':{'shape':[401,35],'units':'world linear m/s; root body-local angular rad/s; joint rad/s'},'state_times':{'shape':[401],'units':'seconds'},'applied_joint_targets':{'shape':[400,29],'units':'radians'},'reference_orientation':'torso quaternion xyzw; recorded-reference frame'}
    for key in ('label','rights'):
        if key in result['planning_music']:result['planning_music'][key]=_prose(result['planning_music'][key],known_paths)
    attach_goal_metadata(result,goal_dto,origin)
    _result_paths_safe(result,known_paths)
    payload['result.json']=encoded(result);require(len(payload['result.json'])<=MAX_RESULT_BYTES and sum(len(v) for v in payload.values())<=MAX_BUNDLE_BYTES,'Portable phrase result exceeds bundle budget')
    reader.finish(selected['payload_paths']);require(manager.get(job_id)==job and reader.json(jobdir/'state.json')==job,'Job selection changed during export')
    for trial in (baseline,selected):require({p.relative_to(trial['directory']).as_posix() for p in trial['directory'].rglob('*') if p.is_file()}==trial['raw_files'],'Saved trial changed during export')
    stream=BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=(stat.S_IFREG|0o600)<<16;archive.writestr(info,data)
    require(stream.tell()<=MAX_BUNDLE_BYTES,'Compressed result exceeds bundle budget');return stream.getvalue()
