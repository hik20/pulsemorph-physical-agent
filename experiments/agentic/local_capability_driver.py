"""Section-local sequential witness search with immutable evidence and exact fresh-reset confirmation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['left_gain_a','right_gain_a','left_gain_b','right_gain_b','timing_shift_s']
ALGORITHM = 'local_capability_search_v1'
CONTROL_TOOL = 'local_clock_1p4_2p8_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def linked(ref, base):
    path = Path(ref['path'])
    if not path.is_absolute(): path = base / path
    if sha(path) != ref['sha256']: raise ValueError('artifact hash mismatch: '+str(path))
    return path


def verify_code(protocol):
    pins = protocol['execution_code_sha256']
    if not isinstance(pins, dict) or not pins: raise ValueError('execution code pins required')
    for path, digest in pins.items():
        if sha(ROOT/path) != digest: raise ValueError('execution code changed: '+path)


def valid(result):
    return (result.get('integrity_passed') is True and result.get('physical_gate_passed') is True
            and result.get('intent_passed') is True and result.get('per_dimension_pass') == [True]*5)


def next_local_candidate(calibration, history, protocol, goal, suggest=None):
    """Pure replayable seed/probe schedule followed by the frozen search algorithm."""
    if len(calibration)!=1 or not calibration[0].get('provenance'):
        raise ValueError('one fresh calibration with provenance required')
    if len(history)>=protocol['new_candidate_budget']: raise ValueError('budget exhausted')
    base=calibration[0].get('evaluation',calibration[0])
    target=np.asarray(goal['feature_targets'],dtype=float)
    features=np.asarray(base['features'],dtype=float)
    bounds=np.asarray(protocol['bounds'],dtype=float)
    if target.shape!=(5,) or features.shape!=(5,) or not np.isfinite(features).all() or np.any(features[:4]<=0):
        raise ValueError('invalid measured calibration features')
    phase=np.sin(np.pi*(target[4]-1.4)/1.4)**2
    if not 1.4<target[4]<2.8 or phase<=1e-12: raise ValueError('goal outside usable local timing envelope')
    if base.get('accent',{}).get('available') is False: raise ValueError('baseline accent unavailable')
    seed=np.r_[target[:4]/features[:4],np.clip((features[4]-target[4])/phase,-.3,.3)]
    seed=np.clip(seed,bounds[:,0],bounds[:,1]); n=len(history)
    if n<6:
        values=seed.copy()
        if n:
            axis=n-1;step=.1 if axis==4 else .2
            values[axis]=np.clip(seed[axis]+step,*bounds[axis])
            if values[axis]==seed[axis]: values[axis]=np.clip(seed[axis]-step,*bounds[axis])
            if values[axis]==seed[axis]: raise ValueError('coordinate probe cannot move')
        return {'parameters':values.tolist(),'decision':{'algorithm_version':ALGORITHM,
                'strategy':'target_informed_seed' if n==0 else 'fixed_seed_coordinate_probe',
                'new_candidate_index':n,'seed':seed.tolist(),'prediction_is_physical_evidence':False}}
    if suggest is None:
        from capability_search import next_candidate
        suggest=next_candidate
    return suggest(calibration,history,protocol,goal)


def parity(first, second):
    for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):
        a,b=np.load(first/name,allow_pickle=False),np.load(second/name,allow_pickle=False)
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
            raise ValueError('fresh reset exact parity failed: '+name)


def run(pilot, execute=None, evaluate=None, suggest=None):
    out=Path(pilot).resolve(); protocol=read(out/'protocol.json')
    if (out/'summary.json').exists(): raise FileExistsError(out/'summary.json')
    if execute is None:
        from execute_local_capability import execute_trial
        execute=execute_trial
    if evaluate is None:
        from evaluate_generalization import evaluate_generalization
        evaluate=evaluate_generalization
    task=protocol['task']
    budget=protocol['new_candidate_budget']
    if protocol['calibration_trials'] != 1 or protocol['confirmation_trials'] != 1: raise ValueError('one calibration and confirmation required')
    if type(budget) is not int or not 0 <= budget <= 64: raise ValueError('budget must be 0..64')
    summary={'status':'execution_error','search_candidates':0,'calibrations':0,'confirmations':0,
             'actual_control_steps':0,'actual_physics_steps':0,'online_onnx_call_count':0,'warmup_onnx_call_count':0,
             'search_algorithm':ALGORITHM,'robustness_claim':False,'prior_candidates':0,
             'imported_prior_candidates':0,'fresh_calibration_as_search_prior':0,'error':None}
    history=[]; prior=[]
    try:
        protocol_digest=sha(out/'protocol.json')
        verify_code(protocol)
        if protocol['prior_trial_records']: raise ValueError('local tool cannot import old feature labels')
        if protocol['fixed_control_tool']!=CONTROL_TOOL: raise ValueError('unsupported local control tool')
        for key in ('old_protocol','old_audit'): linked(protocol[key],ROOT)
        for ref in protocol['old_calibration'].values(): linked(ref,ROOT)
        calibration_ref=protocol['calibration_proposal']
        cp=linked(calibration_ref,out) if isinstance(calibration_ref,dict) else out/calibration_ref
        template=read(cp)
        if template['source_motion']!=task['source_motion']: raise ValueError('calibration source mismatch')
        if (template.get('method')!='calibration' or template.get('task_id')!=task['task_id']
                or template.get('protocol_sha256')!=protocol_digest
                or [template['parameters'][n] for n in NAMES]!=protocol['anchor']
                or template.get('control_tool')!=protocol['fixed_control_tool']
                or template.get('duration_s')!=4 or template.get('source_start_s')!=0):
            raise ValueError('invalid or stale calibration proposal')
        calibration=out/'trials'/template['candidate_id']

        def trial(path, kind):
            verify_code(protocol)
            if sha(out/'protocol.json')!=protocol_digest: raise ValueError('protocol changed during execution')
            proposal=read(path); directory=out/'trials'/proposal['candidate_id']
            if (proposal['protocol_sha256']!=protocol_digest or proposal['source_motion']!=task['source_motion']
                    or proposal['task_id']!=task['task_id'] or proposal['control_tool']!=protocol['fixed_control_tool']): raise ValueError('trial protocol/source/task mismatch')
            if directory.exists(): raise FileExistsError(directory)
            summary[kind]+=1
            manifest=execute(path,directory,video=False)
            for key in ('actual_control_steps','actual_physics_steps','online_onnx_call_count','warmup_onnx_call_count'):
                summary[key]+=int(manifest.get(key,0))
            verify_code(protocol)
            if manifest.get('status')!='success': raise ValueError('execution failed: '+str(manifest.get('error')))
            result=evaluate(directory,directory if kind=='calibrations' else calibration,protocol,protocol['goal'])
            result.update(candidate_id=proposal['candidate_id'],parameters=[proposal['parameters'][n] for n in NAMES],
                          execution_manifest_sha256=sha(directory/'manifest.json'),protocol_sha256=protocol_digest)
            ep=out/'evaluations'/(proposal['candidate_id']+'.json'); write(ep,result)
            record={'candidate_id':proposal['candidate_id'],'proposal':artifact(path),'evaluation':artifact(ep),
                    'manifest':artifact(directory/'manifest.json')}
            write(out/'trial_records'/(proposal['candidate_id']+'.json'),record)
            result['provenance']=record
            if result.get('integrity_passed') is not True: raise ValueError('evaluation integrity failed')
            return result,directory

        calibration_result,_=trial(cp,'calibrations')
        prior=[calibration_result]; summary['prior_candidates']=1; summary['fresh_calibration_as_search_prior']=1
        old_cal=protocol['old_calibration']['execution_manifest']
        if old_cal:
            old=linked(old_cal,ROOT).parent
            for key in ('onnx_sha256','motion_sha256','physics_contract','versions'):
                if read(old/'manifest.json')[key]!=read(calibration/'manifest.json')[key]:
                    raise ValueError('old/fresh calibration mismatch: '+key)
            old_pins={k.replace('\\','/'):v for k,v in read(old/'manifest.json')['preexecution_input_hashes'].items()}
            new_pins={k.replace('\\','/'):v for k,v in read(calibration/'manifest.json')['preexecution_input_hashes'].items()}
            if any(new_pins.get(k)!=v for k,v in old_pins.items()): raise ValueError('old input pin subset mismatch')
            extras=set(new_pins)-set(old_pins)
            if extras!={k.replace('\\','/') for k in protocol['declared_input_hash_extras']}: raise ValueError('undeclared local runtime input pins')
            parity(old,calibration)
        for index in range(budget):
            verify_code(protocol)
            for row in prior+history:
                for ref in row['provenance'].values():
                    if isinstance(ref,dict): linked(ref,ROOT)
            decision=next_local_candidate(prior,history,protocol,protocol['goal'],suggest); values=decision['parameters']; metadata=decision['decision']
            values=np.asarray(values,dtype=float); bounds=np.asarray(protocol['bounds'],dtype=float)
            if values.shape!=(5,) or not np.isfinite(values).all() or np.any(values<bounds[:,0]) or np.any(values>bounds[:,1]):
                raise ValueError('invalid candidate parameters')
            identity=f'high5_b__local_capability__{index:02d}'
            obs=out/'observations'/(identity+'.json')
            write(obs,{'candidate_id':identity,'protocol':artifact(out/'protocol.json'),
                       'prior_history':[r['provenance'] for r in prior], 'new_history':[r['provenance'] for r in history],
                       'search_metadata':metadata})
            proposal=dict(template,candidate_id=identity,observation_id=identity,method='surrogate_search',
                          trial_index=index,stage='search',search_algorithm=metadata,parameters=dict(zip(NAMES,values.tolist())),
                          observation_artifact=artifact(obs),protocol_sha256=sha(out/'protocol.json'))
            path=out/'proposals'/(identity+'.json'); write(path,proposal)
            result,directory=trial(path,'search_candidates'); history.append(result)
            if valid(result):
                confirm=dict(proposal,candidate_id=f'high5_b__local_confirm__{index:02d}',confirmation_of=artifact(path),stage='confirmation')
                confirm['observation_id']=confirm['candidate_id']
                path2=out/'proposals'/(confirm['candidate_id']+'.json'); write(path2,confirm)
                second,directory2=trial(path2,'confirmations')
                if not valid(second): raise ValueError('fresh confirmation failed intent/physical gate')
                parity(directory,directory2)
                summary.update(status='confirmed_witness',witness=result['provenance'],confirmation=second['provenance'])
                break
        else: summary['status']='no_witness_within_budget'
    except Exception as error:
        summary['error']=type(error).__name__+': '+str(error)
    def best(rows):
        rows=[dict(r.get('evaluation',r),candidate_id=r['candidate_id']) for r in rows]
        eligible=[r for r in rows if r.get('integrity_passed') is True and r.get('physical_gate_passed') is True and r.get('utility') is not None]
        return min(eligible,key=lambda r:(r['utility'],r.get('utility_l2',float('inf')),r['candidate_id']))['candidate_id'] if eligible else None
    summary.update(best_prior=best(prior),best_new=best(history),total_new_trials=summary['calibrations']+summary['search_candidates']+summary['confirmations'])
    write(out/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=Path,required=True)
    args=parser.parse_args(); result=run(args.pilot); print(json.dumps(result,indent=2))
    return 1 if result['status']=='execution_error' else 0


if __name__=='__main__': raise SystemExit(main())
