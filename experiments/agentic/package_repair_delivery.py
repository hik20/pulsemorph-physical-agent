"""Read-only closing verification; creates only one exclusive delivery manifest.

Does not execute physics, propose candidates, generate reports, or edit evidence.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import report_repair

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/runs/agentic'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class DigestCache:
    """Hash hardlinked immutable objects once; invalidate on either stat change."""
    def __init__(self):
        self.objects={}
        self.paths={}
        self.content_reads=0

    @staticmethod
    def signature(stat):
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    def sha(self,path):
        path=Path(path).resolve();before=self.signature(path.stat())
        key=before if before[1] else (str(path),)+before
        if key not in self.objects:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                require(self.signature(path.stat())==before,'file changed before hashing: '+str(path))
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            require(self.signature(path.stat())==before,'file changed while hashing: '+str(path))
            self.objects[key]=digest.hexdigest();self.content_reads+=1
        require(self.signature(path.stat())==before,'file changed during cached check: '+str(path))
        self.paths[path]=before
        return self.objects[key]

    def require_paths_unchanged(self):
        for path,signature in self.paths.items():
            require(self.signature(path.stat())==signature,'verified file changed during packaging: '+str(path))


def hardlink_cache_selfcheck():
    """Tiny isolated cache regression; no project evidence or packaging writes."""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        first=Path(temporary)/'first';second=Path(temporary)/'second'
        first.write_bytes(b'old');os.link(first,second)
        cache=DigestCache()
        require(cache.sha(first)==cache.sha(second) and cache.content_reads==1,'hardlink hash reuse failed')
        second.write_bytes(b'changed-length')
        expected=hashlib.sha256(b'changed-length').hexdigest()
        require(cache.sha(first)==cache.sha(second)==expected and cache.content_reads==2,'hardlink mutation invalidation failed')
    return dict(passed=True,hardlink_content_reads=2)


def package(output=BASE/'repair_delivery_v1.json'):
    output=Path(output).resolve()
    require(output==BASE/'repair_delivery_v1.json' and not output.exists(), 'exclusive canonical new delivery path required')
    panel=BASE/'repair_development_v1'
    audit_path=BASE/'repair_development_v1_audit_v2/audit.json'
    failed_audit_path=BASE/'repair_development_v1_audit_v1/audit.json'
    report_path=BASE/'repair_development_v1_report_v1/report.json'
    cache=DigestCache();sha=cache.sha
    def resolve(path):
        path=Path(path)
        return path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    def pinned(record):
        require(isinstance(record,dict) and set(record)=={'path','sha256'},'exact path/hash link required')
        path=resolve(record['path'])
        require(path.is_file() and sha(path)==record['sha256'],'linked evidence changed: '+str(path))
        return path
    def inventory(path):
        path=resolve(path)
        return {item.relative_to(path).as_posix():sha(item) for item in sorted(path.rglob('*')) if item.is_file()}
    def normalize(values):
        result={}
        for key,value in values.items():
            normalized=key.replace('\\','/')
            require(normalized not in result,'inventory normalized path collision')
            result[normalized]=value
        return result
    protocol,audit,report=read(panel/'protocol.json'),read(audit_path),read(report_path)
    require(audit['audit_passed'] is True and audit['errors']==[], 'passed independent audit required')
    require(audit['schema']=='pulsemorph.agentic_repair_audit.v2','accepted versioned full audit required')
    require(resolve(audit['input_panel'])==panel and audit['input_protocol_sha256']==sha(panel/'protocol.json'),'audit panel/protocol identity')
    for key in ('panel_evidence_unchanged','protected_evidence_unchanged','prior_provenance_verified',
                'numeric_decisions_replayed','observation_isolation_verified'):
        require(audit.get(key) is True,'audit acceptance flag: '+key)
    require(audit['new_physical_trials']==audit['new_llm_calls']==0,'audit must be read only')
    for key in ('protected_inventories_before_after_equal','audit_sources_unchanged','bound_prior_reports_unchanged'):
        require(audit[key] is True,'v2 audit stability: '+key)
    require(audit['prior_partial_pass_flags_reused'] is False,'v1 partial acceptance must not be reused')
    require(audit['audit_source_sha256_before']==audit['audit_source_sha256_after'],'v2 auditor before/after source identity')
    for name,digest in audit['audit_source_sha256_before'].items():
        source=resolve(name)
        require(sha(source)==digest and sha(audit_path.parent/'source_snapshot'/source.relative_to(ROOT))==digest,'v2 auditor source/snapshot changed: '+name)
    failed=read(pinned(audit['bound_prior_reports']['preserved_v1_failed_audit']))
    require(pinned(audit['bound_prior_reports']['preserved_v1_failed_audit'])==failed_audit_path
            and failed['audit_passed'] is False and failed['panel_evidence_unchanged'] is True,'preserved failed v1 audit binding')
    raw_cache=audit['raw_audit_cache']
    require(raw_cache['misses']==raw_cache['unique_artifact_paths']==276
            and raw_cache['cached_values_and_returns_are_deep_copies'] is True,'fresh250 plus26 baseline raw audit cache coverage')
    for name,digest in raw_cache['raw_source_sha256'].items():require(sha(resolve(name))==digest,'raw auditor cache source identity')
    require(not (panel/'abort.json').exists(),'aborted panel cannot be packaged')
    for name,digest in protocol['execution_code_sha256'].items():
        require(sha(ROOT/name)==sha(panel/'source_snapshot'/name)==digest,'frozen source/snapshot changed: '+name)
    for name,digest in protocol['runtime_input_sha256'].items():
        require(sha(ROOT/name)==digest,'motor/body/runtime changed: '+name)
    protected_roots=[]
    def protect(items):
        for item in items:
            path=resolve(item['path'])
            require(inventory(path)==normalize(item['files']),'protected inventory changed: '+str(path))
            protected_roots.append(path)
    require(len(protocol['protected_inventories'])==4,'four direct protected roots required')
    protect(protocol['protected_inventories'])
    robust=read(pinned(protocol['robustness_protocol']))
    protect(robust['protected_inventories'])
    for key in ('quarantine','quarantine_inventory'):pinned(robust[key])
    for key in ('original_protocol','accepted_prior_audit'):pinned(protocol[key])
    for record in protocol['shared_priors'].values():pinned(record)
    current=inventory(panel)
    manifests=sorted((panel/'trials').glob('*/manifest.json'))
    require(len(manifests)==250,'exact250 actual trial manifests required')
    identities={path.parent.name for path in manifests}
    starts={};completes={}
    for path in (panel/'events').glob('*.json'):
        event=read(path)
        require(path.name==event['event']+'__'+event['identity']+'.json','event path identity')
        if event['event'] in ('trial_start','trial_complete'):
            target=starts if event['event']=='trial_start' else completes
            require(event['identity'] not in target,'duplicate trial event')
            target[event['identity']]=(event,path)
    require(set(starts)==set(completes)==identities,'actual trial/start/complete inventory')
    controls=physics=0;phases={}
    for path in manifests:
        identity=path.parent.name;manifest=read(path)
        require(manifest['status']=='success' and manifest['candidate_id']==identity,'successful trial identity')
        require((manifest['actual_control_steps'],manifest['actual_physics_steps'])==(200,4000),'actual trial counts')
        controls+=manifest['actual_control_steps'];physics+=manifest['actual_physics_steps']
        for name,digest in manifest['artifact_hashes'].items():
            key='trials/'+identity+'/'+name.replace('\\','/')
            require(current.get(key)==digest,'trial artifact changed: '+key)
        trial_files={key.removeprefix('trials/'+identity+'/') for key in current if key.startswith('trials/'+identity+'/')}
        require(trial_files==set(normalize(manifest['artifact_hashes']))|{'manifest.json'},'complete trial artifact inventory')
        start,start_path=starts[identity];complete,complete_path=completes[identity]
        require(pinned(complete['manifest'])==path,'completion manifest link')
        proposal_path=pinned(start['proposal'])
        require(proposal_path==panel/'trial_proposals'/(identity+'.json'),'start proposal link')
        require(sha(proposal_path)==sha(path.parent/'proposal.json'),'executed proposal bytes')
        proposal=read(proposal_path)
        require(sha(pinned(proposal['parameter_source']))==sha(path.parent/'source_proposal.json'),'seed snapshot identity')
        require(start_path.stat().st_mtime_ns<complete_path.stat().st_mtime_ns,'trial event order')
        require(start['phase']==complete['phase'] and complete['status']=='success','event phase/status')
        phases[start['phase']]=phases.get(start['phase'],0)+1
    require(phases=={'gate':4,'candidate':192,'confirmation':54},'exact fresh phase budget')
    require((controls,physics)==(50000,1000000),'total actual steps')
    receipts=sorted((panel/'model_calls').glob('*.json'))
    expected={f'{task}__{method}__{index}' for task in ('wave_a','high5_b')
              for method in ('llm_feedback','llm_feedback_off') for index in range(8)}
    require({path.stem for path in receipts}==expected,'exact32 model receipt IDs')
    require({path.stem for path in (panel/'model_requests').glob('*.json')}==expected,'exact32 request IDs')
    for path in receipts:
        receipt=read(path);identity=path.stem
        require(receipt['candidate_id']==identity and receipt['model']=='gpt-6.1-sol' and receipt['fork_turns']=='none','receipt identity/model')
        request_path=pinned(receipt['request']);request=read(request_path)
        require(request_path==panel/'model_requests'/(identity+'.json'),'request canonical path')
        for key,directory in (('observation','observations'),('response','responses')):
            require(pinned(receipt[key])==panel/directory/(identity+'.json'),'receipt '+key+' path')
        require(request['observation']==receipt['observation'] and request['candidate_id']==identity,'request/receipt observation identity')
        require(resolve(request['expected_response_path'])==pinned(receipt['response']),'request response destination')
        committed=read(panel/'events'/('proposal_committed__'+identity+'.json'))
        proposal=read(pinned(committed['proposal']))
        require(proposal['observation']==receipt['observation'] and proposal['response']==receipt['response'],'committed model artifacts')
    summary=read(panel/'summary.json')
    require(summary['completed'] is True and summary['errors']==[] and summary['protocol_sha256']==sha(panel/'protocol.json'),'completed bound summary')
    require((summary['physical_trials'],summary['control_steps'],summary['physics_steps'],summary['model_proposal_turns'])==(250,50000,1000000,32),'summary actual budget')
    require(len(summary['winners'])==6 and {(r['task_id'],r['method']) for r in summary['winners']}==
            {(t,m) for t in ('wave_a','high5_b') for m in ('surrogate_search','llm_feedback','llm_feedback_off')},'six distinct winners')
    require(report==report_repair.collect(panel,audit_path),'report differs from pure collector')
    disclosure_dir=BASE/'repair_legacy_fields_review_v1'
    disclosure_path=disclosure_dir/'report.json'
    disclosure=read(disclosure_path);annotations=read(disclosure_dir/'annotations.json')
    require(pinned(audit['bound_prior_reports']['legacy_metadata_disclosure'])==disclosure_path,'v2 audit metadata exception binding')
    require(disclosure['schema']=='pulsemorph.repair_legacy_fields_disclosure.v1'
            and disclosure['status']=='provenance_incomplete_disclosure','legacy metadata disclosure required')
    require(disclosure['bounded_impact_checks_passed'] is True
            and disclosure['raw_metadata_semantically_correct'] is False
            and disclosure['errors']==[] and disclosure['inputs_unchanged_during_review'] is True,'bounded metadata exception checks')
    require(pinned(disclosure['protocol'])==panel/'protocol.json'
            and pinned(disclosure['selection'])==panel/'selection.json','metadata exception protocol/selection binding')
    require(len(annotations)==250 and {row['trial_id'] for row in annotations}==identities,'250 exact disclosure annotations')
    require(disclosure['counts']['gate_trials']==4 and disclosure['counts']['scored_trials']==246,'disclosure gate/scored split')
    for annotation in annotations:
        require(all(value is True for value in annotation['checks'].values()),'disclosure annotation failed check')
        for key in ('raw_proposal','copied_proposal','seed_source','preserved_seed_snapshot'):
            pinned(annotation[key])
        if annotation['current_high_level_proposal'] is not None:pinned(annotation['current_high_level_proposal'])
    for key in ('runtime','driver','common_player'):pinned(disclosure['source_evidence'][key])
    for key,value in dict(raw_execution_metadata_complete=False,legacy_metadata_disclosure_required=True,
                          acceptance_scope='numerical_active_lineage_with_legacy_metadata_exception').items():
        require(report.get(key)==value,'report must disclose legacy metadata exception: '+key)
    require(pinned(report['reporter_source'])==Path(report_repair.__file__).resolve(),'reporter source link')
    require(sha(report_path.parent/'report_repair.py')==sha(report_repair.__file__),'reporter source copy')
    for name in ('report.md','development_best_so_far.png','full13_heatmaps.png','visual_qa.json'):
        require((report_path.parent/name).is_file(),'report/visual QA file missing: '+name)
    qa=read(report_path.parent/'visual_qa.json')
    images=('development_best_so_far.png','full13_heatmaps.png')
    require(qa['inspected_all_pngs'] is True and qa['report_sha256']==sha(report_path),'visual QA report binding')
    require(set(qa['files'])==set(images) and set(qa['image_dimensions_px'])==set(images),'visual QA exact image inventory')
    import struct
    for name in images:
        image=report_path.parent/name
        require(qa['files'][name]==sha(image),'visual QA PNG changed: '+name)
        with image.open('rb') as stream:header=stream.read(24)
        require(header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR','valid PNG header required')
        require(list(struct.unpack('>II',header[16:24]))==qa['image_dimensions_px'][name],'visual QA image dimensions')
    media_root=ROOT/'artifacts/media/agentic/repair_development_v1'
    preview_source=ROOT/'experiments/agentic/repair_preview.py'
    media_directories=[]
    frames=('middle.png','near_accent.png','last.png')
    for task,condition in (('wave_a','combined_minus'),('high5_b','joint_position_minus')):
        directory=media_root/task;media_directories.append(directory)
        manifest_path=directory/'manifest.json';preview=read(manifest_path)
        require(preview['schema']=='pulsemorph.agentic_repair_preview.v1'
                and preview['task_id']==task and preview['condition_id']==condition,'preview task/condition identity')
        require(preview['execution_kind']=='recorded_physics_state_render_mj_forward'
                and preview['development_only'] is True,'saved-state preview scope')
        require(preview.get('raw_execution_metadata_complete') is False
                and preview.get('acceptance_scope')=='numerical_active_lineage_with_legacy_metadata_exception','preview legacy metadata exception scope')
        exception=preview['legacy_metadata_exception']
        require(resolve(exception['report_path'])==disclosure_path
                and exception['report_sha256']==sha(disclosure_path)
                and resolve(exception['annotations_path'])==disclosure_dir/'annotations.json'
                and exception['annotations_sha256']==sha(disclosure_dir/'annotations.json')
                and exception['bounded_impact_checks_passed'] is True
                and exception['raw_metadata_semantically_correct'] is False,'preview legacy metadata addendum binding')
        inputs={resolve(name):digest for name,digest in preview['inputs_sha256'].items()}
        require(len(inputs)==len(preview['inputs_sha256']),'preview input path collision')
        for path,digest in inputs.items():require(sha(path)==digest,'preview source/input changed: '+str(path))
        for path in (audit_path,panel/'protocol.json',panel/'summary.json',panel/'selection.json',preview_source,
                     ROOT/'experiments/agentic/generalization_preview.py',disclosure_path,disclosure_dir/'annotations.json'):
            require(inputs.get(path.resolve())==sha(path),'preview required binding: '+str(path))
        require(sha(directory/'preview_source.py')==sha(preview_source),'preview source copy')
        require(len(preview['panels'])==4 and [row['method'] for row in preview['panels']]==
                ['incumbent','surrogate_search','llm_feedback','llm_feedback_off'],'preview four-panel method order')
        for row in preview['panels']:
            evaluation_path=resolve(row['source_evaluation']);trial=resolve(row['source_trial'])
            require(row['task_id']==task and row['condition_id']==condition,'preview panel task/condition')
            require(sha(evaluation_path)==row['source_evaluation_sha256'] and sha(trial/'manifest.json')==row['source_manifest_sha256'],'preview panel source pins')
            evaluation=read(evaluation_path)
            require(row['primary']==evaluation['primary'] and row['physical_gate_passed']==evaluation['physical_gate_passed'],'preview measured annotations')
            require(inputs.get(trial/'qpos.npy')==sha(trial/'qpos.npy')
                    and inputs.get(trial/'state_times.npy')==sha(trial/'state_times.npy'),'preview recorded-state input bindings')
        media_inventory=inventory(directory)
        outputs=normalize(preview['outputs_sha256'])
        require(set(media_inventory)==set(outputs)|{'manifest.json','visual_qa.json'},'complete preview output inventory')
        for name,digest in outputs.items():require(media_inventory[name]==digest,'preview rendered output changed: '+name)
        probe=read(directory/'ffprobecheck.json')
        require(probe==preview['ffprobe'] and probe['verified'] is True,'preview ffprobe receipt binding')
        require((probe['width'],probe['height'],probe['frames'],probe['fps'],probe['duration_s'])==(960,720,100,25,4)
                and abs(probe['audio_duration_s']-4.)<=.04,'preview audiovisual dimensions/timing')
        visual=read(directory/'visual_qa.json')
        require(visual['schema']=='pulsemorph.agentic_repair_preview_visual_qa.v1' and visual['task_id']==task,'preview visual QA identity')
        require(visual['layout_passed'] is True and visual['body_visibility_passed'] is True,'preview layout/body visibility QA')
        require(set(visual['checked_frames'])==set(frames) and set(visual['frame_sha256'])==set(frames),'three reviewed preview frames required')
        require(visual['comparison_sha256']==sha(directory/'comparison.mp4')
                and visual['manifest_sha256']==sha(manifest_path),'preview visual QA video/manifest binding')
        for name in frames:
            require(visual['frame_sha256'][name]==sha(directory/name),'preview QA frame changed: '+name)
            with (directory/name).open('rb') as stream:header=stream.read(24)
            require(header[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',header[16:24])==(960,720),'preview QA frame dimensions')
    verification=BASE/'repair_verification_v1';test=read(verification/'test_run.json')
    require((test['passed'],test['subtests_passed'],test['exit_code'])==(30,4,0),'accepted30+4 verification required')
    require(test['physics_steps']==test['model_proposal_turns']==0,'verification must not add trials')
    pinned(test['log'])
    for name,digest in test['test_sources'].items():require(sha(ROOT/name)==digest,'verified test source changed: '+name)
    precheck=BASE/'repair_numeric_replay_precheck_v1';check=read(precheck/'report.json')
    require(check['precheck_passed'] is True and resolve(check['input_panel'])==panel,'accepted numeric replay precheck required')
    require(check['new_physical_trials']==check['new_model_calls']==0,'precheck must be read only')
    documents=[ROOT/'experiments/agentic/REPAIR_RESULTS.md',ROOT/'design/agentic/REPAIR_EXPERIMENT.md',
               ROOT/'design/agentic/REPAIR_PROTOCOL_REVIEW.md',ROOT/'design/coordination/AGENTIC_STATUS.md',
               ROOT/'design/agentic/REPAIR_RESULTS_REVIEW.md',ROOT/'design/agentic/REPAIR_NEXT.md']
    checked_links=[]
    for target in re.findall(r'\]\(([^)]+)\)',documents[0].read_text(encoding='utf8')):
        if target.startswith(('https://','http://','#')):continue
        destination=(documents[0].parent/target.split('#')[0].strip('<>')).resolve()
        require(destination.exists(),'result document broken link: '+target)
        checked_links.append(str(destination))
    directories=[panel,audit_path.parent,failed_audit_path.parent,report_path.parent,verification,precheck,disclosure_dir]+media_directories
    files={path.relative_to(ROOT).as_posix():sha(path) for path in documents+[Path(__file__).resolve(),Path(report_repair.__file__).resolve(),preview_source,
                                                                           ROOT/'experiments/agentic/generalization_preview.py',
                                                                           ROOT/'experiments/agentic/audit_repair_legacy_fields.py',
                                                                           ROOT/'experiments/agentic/audit_repair_v2.py']}
    for path in sorted((ROOT/'tests').glob('test_agentic_repair_audit_v2*.py')):files[path.relative_to(ROOT).as_posix()]=sha(path)
    inventories={}
    for directory in directories+list(dict.fromkeys(protected_roots)):
        values=inventory(directory);relative=directory.relative_to(ROOT).as_posix()
        inventories[relative]=values
        files.update({relative+'/'+key:digest for key,digest in values.items()})
    require(inventory(panel)==current,'panel changed during packaging')
    protect(protocol['protected_inventories'])
    protect(robust['protected_inventories'])
    for key in ('quarantine','quarantine_inventory'):pinned(robust[key])
    for path in BASE.glob('repair_development_v1*.log'):files[path.relative_to(ROOT).as_posix()]=sha(path)
    result=dict(schema='pulsemorph.agentic_repair_delivery.v1',verified=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),files=files,file_count=len(files),
        inventories=inventories,checked_result_links=checked_links,
        completed_panel=dict(physical_trials=250,control_steps=50000,physics_steps=1000000,new_model_proposal_turns=32),
        recorded_state_previews=dict(tasks=['wave_a','high5_b'],videos=2,duration_s_each=4,
                                     physics_rollouts=0,visual_scope='layout and body visibility only'),
        new_physical_trials_during_packaging=0,new_model_calls_during_packaging=0,
        verification=dict(tests_passed=30,subtests_passed=4),training_admission=False,
        raw_execution_metadata_complete=False,legacy_metadata_disclosure_required=True,
        acceptance_scope='numerical_active_lineage_with_legacy_metadata_exception',
        legacy_metadata_addendum=dict(path=str(disclosure_path),sha256=sha(disclosure_path)),
        accepted_audit=dict(path=str(audit_path),sha256=sha(audit_path)),
        preserved_failed_audit=dict(path=str(failed_audit_path),sha256=sha(failed_audit_path)),
        scope='Known warm-start development; immutable closing inventory, no held-out, recovery, motor learning or aesthetic claim.')
    cache.require_paths_unchanged()
    with output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    result=package()
    print(json.dumps(dict(verified=result['verified'],files=result['file_count'],links=len(result['checked_result_links']))))
