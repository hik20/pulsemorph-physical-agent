"""Read-only terminal app job delivery audit; never calls a model or steps physics."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
TERMINAL={'complete','failed','cancelled'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(condition,message):
    if not condition: raise ValueError(message)
def path_in(base,name):
    path=(base/name).resolve()
    require(base.resolve() in path.parents,'artifact path escapes its directory')
    return path

def inventory(directory,manifest):
    hashes=manifest['artifact_hashes']
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
    require(actual==set(hashes),'trial artifact inventory mismatch')
    for name,digest in hashes.items():require(sha(path_in(directory,name))==digest,'trial artifact hash mismatch: '+name)


def audit_calls(directory,job):
    completed=[];unconfirmed=[]
    for folder in sorted((directory/'model_calls').glob('*')):
        if not folder.is_dir():continue
        request=folder/'request.json';response=folder/'response.json';receipt=folder/'receipt.json'
        require(request.is_file(),'model attempt has no request')
        if not receipt.exists():
            failure=read(folder/'failure.json') if (folder/'failure.json').exists() else {}
            unconfirmed.append({'attempt':folder.name,'status':'completion_unconfirmed',
                                'failure_recorded':bool(failure),'response_present':response.exists()})
            continue
        row=read(receipt)
        require(response.is_file(),'receipt has no response')
        require(row['request_sha256']==sha(request),'model request hash mismatch')
        require(row['response_sha256']==sha(response),'model response hash mismatch')
        require(type(row['index']) is int and row['index']==len(completed),'model receipt index mismatch')
        require(int(folder.name.split('_',1)[0])==row['index'],'model folder index mismatch')
        require(row['purpose']==read(request)['purpose'],'model purpose mismatch')
        import jsonschema
        jsonschema.validate(read(response),read(request)['schema'])
        if row['purpose']=='interpret':require(read(response)==job.get('plan'),'interpreted plan differs from model response')
        completed.append(row)
    require(completed==job.get('model_calls',[]),'state model receipts differ from disk')
    require(job.get('live',False)==(bool(completed) and all(r.get('live') is True for r in completed)),
            'state live flag differs from completed receipt evidence')
    return {'completed':len(completed),'live_completed':sum(r.get('live') is True for r in completed),
            'completion_unconfirmed':unconfirmed}


def audit_targets(directory,job):
    if job.get('targets') is None:
        require(not job.get('baseline') and not job.get('selected'),'measured job has no frozen targets')
        return {'state':'not_established'}
    record=read(directory/'targets.json');plan=job['plan'];targets=job['targets'];tolerances=job['tolerances']
    require(record['targets']==targets and record['tolerances']==tolerances and record['plan']==plan,
            'saved target contract differs from state')
    require(len(targets)==len(tolerances)==5 and targets[:4]==plan['arm_targets'],'target arms differ from interpreted plan')
    require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in targets+tolerances)
            and all(v>0 for v in tolerances),'invalid target/tolerance numbers')
    mode=plan['timing_mode']
    accent=job['music']['accent_time_s'] if mode=='music' else plan['explicit_accent_s'] if mode=='explicit' else job['baseline']['metrics']['features'][4]
    require(targets[4]==accent,'timing target differs from original timing source')
    require(record['timing_source']==mode,'saved timing source mismatch')
    for row in [job.get('baseline')]+job.get('trials',[]):
        if not row:continue
        metrics=row['metrics']
        require(metrics.get('targets',targets)==targets and metrics.get('tolerances',tolerances)==tolerances,
                'trial changed frozen targets/tolerances')
    return {'state':'verified','targets':targets,'tolerances':tolerances,'timing_source':mode}


def audit_selection(job):
    rows=[r for r in [job.get('baseline')]+job.get('trials',[]) if r and r.get('execution_state')!='failed']
    selected=job.get('selected')
    policy=job.get('selection_policy')
    require(policy is None or policy=='physical gate, worst normalized error, summed squared normalized error, earlier index','unknown selection policy')
    for candidate in rows:
        require(candidate.get('score')==candidate['metrics'].get('score'),'top-level score differs from metrics')
        require(candidate.get('passed')==bool(candidate['metrics']['physical_gate_passed'] and candidate['metrics']['intent_passed']),'candidate pass label differs from measurements')
    for row in job.get('trials',[]):
        if row.get('execution_state')=='failed':
            require(row.get('score') is None and row['metrics'].get('score') is None and
                    row['metrics'].get('features')==[None]*5 and not row.get('passed'),
                    'failed trial has invented successful metrics')
    if not selected:return {'state':'not_selected','policy':policy or 'legacy physical/max/index'}
    require(bool(rows),'selected job has no measured candidates')
    def key(row):
        m=row['metrics'];score=row.get('score')
        prefix=(not m['physical_gate_passed'],score if score is not None else float('inf'))
        return prefix+((m.get('utility_l2',float('inf')),) if policy else ())+ (row['index'],)
    expected=min(rows,key=key)
    require(selected==expected,'selected result differs from documented selection policy')
    outcome='target_met' if selected['passed'] else 'partial' if selected['metrics']['physical_gate_passed'] else 'physical_gate_failed'
    require(job.get('outcome')==outcome,'job outcome differs from selected measurements')
    return {'state':'verified','selected_index':selected['index'],'policy':policy or 'legacy physical/max/index'}


def audit_implementation(directory,job):
    pinfile=directory/'implementation.json'
    if not pinfile.exists():
        require(not job.get('selection_policy'),'new-policy job lacks implementation snapshots')
        return {'state':'legacy_absence_disclosed'}
    pins=read(pinfile)
    require(bool(pins),'empty implementation pins')
    for name,digest in pins.items():require(sha(path_in(directory/'implementation',name))==digest,'implementation snapshot hash mismatch: '+name)
    return {'state':'verified','files':len(pins),'manifest_sha256':sha(pinfile)}


def audit_media(directory,job):
    if not job.get('media'):return {'state':'not_created'}
    media=directory/'media';manifest=read(media/'manifest.json')
    for name,digest in manifest['outputs_sha256'].items():
        require(sha(path_in(media,name))==digest,'render output hash mismatch: '+name)
    expected=job['music']['sha256']
    require(manifest['audio_sha256']==expected,'render soundtrack differs from job observed music')
    if manifest.get('audio_observation'):
        require(manifest['audio_observation'].get('sha256')==expected,'render audio binding mismatch')
        if manifest['audio_observation'].get('path'):
            audio=Path(manifest['audio_observation']['path']).resolve()
            require((ROOT/'artifacts/media/agentic').resolve() in audio.parents,'render audio path outside app media')
            require(sha(audio)==expected,'actual soundtrack file differs from observed hash')
    baseline=directory/'trials/baseline'
    selected=baseline if job['selected']['index']==-1 else directory/'trials'/f"trial_{job['selected']['index']:02d}"
    require(manifest['input_manifest_sha256']==[sha(p/'manifest.json') for p in (baseline,selected)],'render physical input binding mismatch')
    for p in (baseline,selected):
        require(read(p/'manifest.json')['audio_observation']['sha256']==expected,'trial audio link differs from job music')
    require(manifest.get('physics_steps')==0 and manifest.get('onnx_calls')==0,'render claims new physical execution')
    return {'state':'verified','audio_sha256':expected,'outputs':sorted(manifest['outputs_sha256']),
            'audio_binding_mode':'explicit_observation' if manifest.get('audio_observation') else 'legacy_digest_only_disclosed'}


def close_enough(a,b):
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(close_enough(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
    return a==b


class PhysicsAuditor:
    def __init__(self,root):
        from pulsemorph.agentic.app import runtime
        self.runtime=runtime;self.root=root;self.cache={};self.module=runtime._module(root,'evaluate_multifeature')
        self.original=self.module._audit
    def get(self,directory):
        key=str(Path(directory).resolve())
        if key not in self.cache:
            self.cache[key]=self.runtime._audit_app(self.module,self.original,Path(directory))
        return self.cache[key]
    def evaluate(self,directory,baseline,targets,tolerances):
        # Isolated function globals reuse audited arrays; no app module is modified.
        module=self.runtime._module(self.root,'evaluate_multifeature')
        module._audit=lambda path:self.get(path)
        namespace={**self.runtime.evaluate_trial.__globals__,'_module':lambda root,name:module,
                   '_audit_app':lambda module,original,path:self.get(path)}
        evaluate=types.FunctionType(self.runtime.evaluate_trial.__code__,namespace)
        return evaluate(directory,baseline,targets,tolerances)


def audit_physics(directory,job,auditor):
    rows=[];baseline=directory/'trials/baseline'
    recorded={-1:job.get('baseline'),**{r['index']:r for r in job.get('trials',[])}}
    for trial in sorted((directory/'trials').glob('*')):
        if not trial.is_dir():continue
        path=trial/'manifest.json'
        if not path.exists():
            rows.append({'trial':trial.name,'state':'partial_no_manifest','counted_as_success':False});continue
        manifest=read(path)
        if manifest['status']!='success':
            inventory(trial,manifest)
            rows.append({'trial':trial.name,'state':'failed_physical_manifest','counted_as_success':False,
                         'actual_physics_steps':manifest.get('actual_physics_steps')});continue
        audited=auditor.get(trial)
        index=-1 if trial.name=='baseline' else int(trial.name.split('_')[-1])
        row=recorded.get(index)
        proposal=read(trial/'proposal.json')
        require(proposal['source_id']==job['request']['source_id'],'trial source differs from job request')
        if row:
            require(proposal['parameters']==row['parameters'] and proposal['control_tool']==row['tool_id'],'trial parameters/tool differ from recorded proposal')
        result={'trial':trial.name,'state':'independently_verified','counted_as_success':True,
                'physics_steps':audited['manifest']['actual_physics_steps'],
                'online_onnx_calls':audited['manifest']['online_onnx_call_count']}
        runtime_pin=next((v for k,v in manifest['preexecution_input_hashes'].items()
                          if k.replace('\\','/').endswith('src/pulsemorph/agentic/app/runtime.py')),None)
        if runtime_pin!=sha(auditor.root/'src/pulsemorph/agentic/app/runtime.py'):
            result['evaluation']='not_recomputed_runtime_revision_differs_or_missing'
        elif row and job.get('targets'):
            recalculated=auditor.evaluate(trial,baseline,job['targets'],job['tolerances'])
            require(recalculated['integrity_passed'],'recomputed evaluation integrity failed')
            for key in ('features','physical_gate_passed','intent_passed','per_dimension_pass','score',
                        'min_root_height_m','mean_joint_rmse_rad'):
                require(close_enough(recalculated.get(key),row['metrics'].get(key)),'recorded evaluation differs: '+trial.name+'/'+key)
            result['evaluation']='matches_current_evaluator'
        else:result['evaluation']='not_recorded_or_targets_not_established'
        if index>=0 and row:
            require(read(directory/'evaluations'/f'trial_{index:02d}.json')==row,'saved evaluation differs from state')
        rows.append(result)
    return rows


def audit_jobs(jobs_root,root=ROOT,*,check_memory=True,physics_auditor=None):
    jobs_root=Path(jobs_root).resolve();root=Path(root).resolve()
    result={'schema':'pulsemorph.agentic_app_delivery_audit.v1','created_at':datetime.now(timezone.utc).isoformat(),
            'read_only':True,'model_calls':0,'physics_steps':0,'jobs':[],'skipped_nonterminal':[],
            'zetaris':{'new_query_executed':False,'live_integration_claimed':False}}
    auditor=physics_auditor
    for state_path in sorted(jobs_root.glob('*/state.json')):
        try:job=read(state_path)
        except Exception as exc:
            result['jobs'].append({'id':state_path.parent.name,'passed':False,'errors':['state unreadable: '+type(exc).__name__]});continue
        if job.get('status') not in TERMINAL:
            result['skipped_nonterminal'].append({'id':state_path.parent.name,'status':job.get('status')});continue
        row={'id':job['id'],'status':job['status'],'errors':[]}
        if job['id']!=state_path.parent.name:row['errors'].append('job identity mismatch')
        for name,operation in [('model',lambda:audit_calls(state_path.parent,job)),
            ('targets',lambda:audit_targets(state_path.parent,job)),('selection',lambda:audit_selection(job)),
            ('implementation',lambda:audit_implementation(state_path.parent,job)),('media',lambda:audit_media(state_path.parent,job))]:
            try:row[name]=operation()
            except Exception as exc:row['errors'].append(name+': '+str(exc))
        try:
            if auditor is None:auditor=PhysicsAuditor(root)
            row['physics']=audit_physics(state_path.parent,job,auditor)
        except Exception as exc:row['errors'].append('physics: '+str(exc))
        row['passed']=not row['errors'];row['state_sha256']=sha(state_path)
        row['zetaris_observation']=job.get('integration_context',{'state':'not_recorded','executed':False})
        result['jobs'].append(row)
    if check_memory:
        try:
            from pulsemorph.agentic.app.integrations import AppIntegrations
            integration=AppIntegrations(root,jobs_root.parent)
            records=integration.list_memory(500)
            result['memory']={'read_only_list':True,'records':len(records),
                              'jobs_present':[r['id'] for r in records if r.get('type')=='app_job'],
                              'status':integration.status()['meterless']}
        except Exception as exc:result['memory']={'read_only_list':True,'state':'unavailable','error_type':type(exc).__name__}
    else:result['memory']={'state':'not_requested'}
    result['coverage_complete']=not result['skipped_nonterminal']
    result['passed']=result['coverage_complete'] and all(row['passed'] for row in result['jobs'])
    result['successfully_audited_physical_trials']=sum(sum(p['counted_as_success'] for p in row.get('physics',[])) for row in result['jobs'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--jobs-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('audit output already exists')
    result=audit_jobs(args.jobs_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'passed':result['passed'],'terminal_jobs':len(result['jobs']),'output':str(args.output)}))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
