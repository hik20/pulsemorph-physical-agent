"""Read-only capability completion audit. Recomputes FK, never steps physics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))
from audit_multifeature import read,sha,require,inventory,common_parity
from audit_generalization import METRICS
from evaluate_generalization import evaluate_generalization
from capability_driver import linked,load_prior,valid,parity,NAMES
from capability_search import next_candidate


def best(rows):
    eligible=[r.get('evaluation',r) for r in rows if r.get('evaluation',r).get('integrity_passed') is True and r.get('evaluation',r).get('physical_gate_passed') is True and r.get('evaluation',r).get('utility') is not None]
    return min(eligible,key=lambda r:(r['utility'],r['utility_l2'],r['candidate_id'])) if eligible else None


def check_stopping(summary, rows, confirmations, budget):
    require(0<=len(rows)<=budget<=64,'search budget exceeded')
    passing=[i for i,r in enumerate(rows) if valid(r)]
    require(not passing or passing==[len(rows)-1],'search continued after first witness')
    status=summary['status']
    if status=='confirmed_witness':
        require(bool(passing) and len(confirmations)==1 and valid(confirmations[0]),'confirmed witness requires one passing confirmation')
    elif status=='no_witness_within_budget':
        require(not passing and len(rows)==budget==64 and not confirmations,'no-witness requires all64 candidates and no confirmation')
    elif status=='execution_error':
        require(bool(summary.get('error')) and len(confirmations)<=1,'technical failure must include reason')
    else: raise ValueError('unknown completion status')


def check_metadata(proposal, manifest, decision):
    require(proposal['method']=='surrogate_search','capability proposal mislabeled as another method')
    require(proposal['search_algorithm']==decision,'search metadata linkage mismatch')
    require(decision['algorithm_version']=='capability_witness_v1' and decision['prediction_is_physical_evidence'] is False,'prediction evidence/method label mismatch')
    require(manifest['schema']=='pulsemorph.agentic_generalization_trial.v1','runtime schema changed')
    require(manifest['producer']=={'kind':'baseline','id':'generalization-surrogate_search','version':'2'},'runtime producer linkage mismatch')


def check_index(proposal, index):
    require(proposal['candidate_id']==f'high5_b__capability__{index:02d}' and proposal['search_algorithm']['new_candidate_index']==index,'canonical search index mismatch')
    require(proposal.get('trial_index')==0,'unexpected legacy template trial_index')


def raw_inventory(trial):
    manifest=read(trial/'manifest.json')
    expected=manifest['artifact_hashes']
    actual={str(p.relative_to(trial)) for p in trial.rglob('*') if p.is_file() and p!=trial/'manifest.json'}
    require(actual==set(expected),'raw artifact inventory mismatch')
    for name,digest in expected.items():
        p=(trial/name).resolve()
        require(trial in p.parents and sha(p)==digest,'raw artifact hash mismatch: '+name)
    return manifest


def audit(pilot,output):
    pilot=Path(pilot).resolve();output=Path(output).resolve()
    require(output!=pilot and pilot not in output.parents,'output must be outside pilot')
    output.mkdir(parents=True,exist_ok=False)
    before=inventory(pilot)
    report=dict(schema='pulsemorph.agentic_capability_audit.v1',audit_passed=False,errors=[],trials={},new_llm_proposal_calls=0,
                scope='Post-comparison exploratory measured feasibility; no new matched planner comparison or generalization score.',
                metadata_status='legacy_trial_index_stale',metadata_warning='legacy_template_trial_index=0; canonical search indices are candidate_id suffix, search_algorithm.new_candidate_index and measured observation history length. Legacy trial_index is not a truthful candidate index.',training_ready=False,
                quality_scope='Joint proxies and physical gates are separate from perceived expression, human preference, robustness and training readiness.')
    try:
        p=read(pilot/'protocol.json');summary=read(pilot/'summary.json');digest=sha(pilot/'protocol.json')
        report['reported_status']=summary.get('status');report['technical_failure']=summary.get('error')
        report['reported_actual_counts']={key:summary.get(key) for key in ('calibrations','search_candidates','confirmations','total_new_trials','actual_control_steps','actual_physics_steps','online_onnx_call_count','warmup_onnx_call_count')}
        require(p['new_candidate_budget']==64 and p['max_new_physical_trials']==66,'frozen trial budget changed')
        for name,value in p['execution_code_sha256'].items():
            require(sha(ROOT/name)==value and sha(pilot/'protocol_source_snapshot'/name)==value,'code/snapshot pin mismatch '+name)
        oldfiles={k:linked(p[k],ROOT) for k in ('old_protocol','old_comparison','old_audit')}
        old=read(oldfiles['old_protocol']);old_audit=read(oldfiles['old_audit'])
        require(old_audit['audit_passed'] and old_audit['original_evidence_unchanged'],'old audit failed')
        task=next(t for t in old['tasks'] if t['task_id']=='high5_b')
        require(p['task']==task and p['goal']==task['goal'] and p['source_motion']==task['source_motion'],'old source/goal changed')
        for key in ('parameter_names','bounds','feature_names','feature_windows_s','feature_tolerances','physical_gate','anchor'):
            require(p[key]==old[key],'old frozen field changed '+key)
        prior=load_prior(p,pilot,task)
        require(len(prior)==24 and len(set(tuple(r['parameters']) for r in prior))==20,'old24/20 warmstarts mismatch')
        preserved={str(path):sha(path) for path in oldfiles.values()}
        for row in prior:
            ep=linked(row['provenance']['evaluation'],ROOT)
            for ref in row['provenance'].values():
                if isinstance(ref,dict) and 'path' in ref:
                    path=linked(ref,ROOT);preserved[str(path)]=sha(path)
            m=raw_inventory(linked(row['provenance']['execution_manifest'],ROOT).parent)
            for key in METRICS:
                require(row['evaluation'].get(key)==old_audit['trials'][row['candidate_id']].get(key),'prior differs from old audit '+key)
        cp=pilot/p['calibration_proposal'];cal=pilot/'trials'/read(cp)['candidate_id']
        oldcal=linked(p['old_calibration']['execution_manifest'],ROOT).parent
        for ref in p['old_calibration'].values(): linked(ref,ROOT)
        parity(oldcal,cal)
        for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):
            require((oldcal/name).read_bytes()==(cal/name).read_bytes(),'full old/fresh calibration byte parity')
        baseline=read(cal/'manifest.json');oldbase=read(oldcal/'manifest.json')
        for key in ('preexecution_input_hashes','versions','onnx_sha256','motion_sha256','physics_contract'):
            require(baseline[key]==oldbase[key],'old baseline pin mismatch '+key)
        proposals=list((pilot/'proposals').glob('*.json'))
        searches=sorted([x for x in proposals if read(x)['candidate_id'].startswith('high5_b__capability__')])
        confirms=[x for x in proposals if read(x)['candidate_id'].startswith('high5_b__confirm__')]
        require(len(confirms)<=1,'extra confirmation proposals')
        history=[];confirmed=[];counts={k:0 for k in ('actual_control_steps','actual_physics_steps','online_onnx_call_count','warmup_onnx_call_count')}
        executed=[]
        for path in [cp]+searches+confirms:
            proposal=read(path);identity=proposal['candidate_id'];trial=pilot/'trials'/identity
            require(proposal['protocol_sha256']==digest and proposal['task_id']==task['task_id'] and proposal['source_motion']==task['source_motion'],'new identity/source/protocol mismatch')
            manifest=raw_inventory(trial);executed.append(identity)
            require(manifest['candidate_id']==identity,'manifest identity mismatch')
            for key in ('preexecution_input_hashes','versions','onnx_sha256','motion_sha256','physics_contract'):
                require(manifest[key]==baseline[key],'new execution source/runtime pin mismatch '+key)
            require(manifest['proposal_sha256']==sha(path) and sha(trial/'proposal.json')==sha(path),'proposal execution linkage')
            for key in counts:counts[key]+=manifest[key]
            if manifest['status']!='success':
                require(summary['status']=='execution_error','failed trial mislabeled');continue
            result=evaluate_generalization(trial,cal,p,p['goal'])
            require(result['integrity_passed'] and result.get('fresh_anchor_audit') is not None,'new integrity/FK failed')
            recorded=read(pilot/'evaluations'/f'{identity}.json')
            for key in METRICS: require(recorded.get(key)==result.get(key),'stored metric mismatch '+identity+':'+key)
            require(recorded['execution_manifest_sha256']==sha(trial/'manifest.json') and recorded['protocol_sha256']==digest,'evaluation hashes mismatch')
            record=read(pilot/'trial_records'/f'{identity}.json')
            for key,target in [('proposal',path),('evaluation',pilot/'evaluations'/f'{identity}.json'),('manifest',trial/'manifest.json')]:
                require(linked(record[key],pilot).resolve()==target.resolve(),'record hash/link mismatch')
            result.update(candidate_id=identity,parameters=[proposal['parameters'][n] for n in NAMES],provenance=record)
            result['common_parity']=common_parity(trial,proposal,output/'roundtrip_packages'/identity)
            report['trials'][identity]=result
            if path in searches:
                index=len(history);check_index(proposal,index)
                decision=next_candidate(prior,history,p,p['goal'])
                require(result['parameters']==decision['parameters'],'deterministic proposal replay mismatch')
                check_metadata(proposal,manifest,decision['decision'])
                obs=read(linked(proposal['observation_artifact'],pilot))
                require(linked(obs['protocol'],pilot)==pilot/'protocol.json','observation protocol link')
                require(obs['prior_history']==[r['provenance'] for r in prior] and obs['new_history']==[r['provenance'] for r in history] and obs['search_metadata']==decision['decision'],'measured history linkage')
                history.append(result)
            elif path in confirms:
                require(history and valid(history[-1]),'confirmation before witness')
                source=linked(proposal['confirmation_of'],pilot)
                require(source==searches[-1] and proposal['parameters']==read(source)['parameters'],'confirmation identity/parameters')
                check_metadata(proposal,manifest,read(source)['search_algorithm'])
                parity(pilot/'trials'/history[-1]['candidate_id'],trial)
                for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):
                    require((trial/name).read_bytes()==(pilot/'trials'/history[-1]['candidate_id']/name).read_bytes(),'full confirmation byte parity')
                confirmed.append(result)
        require({x.name for x in (pilot/'trials').iterdir() if x.is_dir()}==set(executed),'unaccounted trial directory')
        check_stopping(summary,history,confirmed,p['new_candidate_budget'])
        if confirmed:
            require(summary['witness']==history[-1]['provenance'] and summary['confirmation']==confirmed[0]['provenance'],'summary witness linkage mismatch')
        require(summary['search_candidates']==len(searches) and summary['calibrations']==1 and summary['confirmations']==len(confirms) and summary['total_new_trials']==len(executed),'new trial counts mismatch')
        for key,value in counts.items():require(summary[key]==value,'actual count mismatch '+key)
        for label,rows in [('best_prior',prior),('best_new',history)]:
            selected=best(rows);require(summary[label]==(selected['candidate_id'] if selected else None),'selector mismatch '+label)
            report[label]=selected
        protected=read(pilot/'protected_hashes_before.json')
        require(protected and all(sha(ROOT/name)==value for name,value in protected.items()),'protected learned/rhythm evidence changed')
        require(all(sha(Path(name))==value for name,value in preserved.items()),'original generalization/prior evidence changed')
        report.update(audit_passed=True,status=summary['status'],actual_counts=dict(counts,total_new_trials=len(executed),calibrations=1,search_candidates=len(searches),confirmations=len(confirms),reused_prior_rows=24,reused_unique_vectors=20),witness=history[-1] if confirmed else None,original_scores_unchanged=True,original_link_hashes=preserved)
    except Exception as error:report['errors'].append(type(error).__name__+': '+str(error))
    report['original_evidence_unchanged']=inventory(pilot)==before
    if not report['original_evidence_unchanged']:report['audit_passed']=False;report['errors'].append('pilot changed during read-only audit')
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.pilot,args.output)
    print(json.dumps({k:result[k] for k in ('audit_passed','errors')},indent=2));sys.exit(0 if result['audit_passed'] else 1)
