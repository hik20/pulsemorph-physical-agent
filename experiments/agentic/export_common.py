"""Export recorded selected trial targets to the newly available shared API.

This verifies saved/loaded target-input parity, without rerunning physics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def dense_grid(cache, anchor):
    """Only the union of recorded samples; duplicates must match exactly."""
    rows = {}
    duplicates = 0
    def add(index, q, dq, rot):
        nonlocal duplicates
        values = (q, dq, rot)
        if index in rows:
            if not all(np.array_equal(a,b) for a,b in zip(rows[index], values)):
                raise ValueError(f'inconsistent recorded cache at grid index {index}')
            duplicates += 1
        else:
            rows[index] = values
    for i in range(201):
        add(i,cache['committed__dof_pos'][i],cache['committed__dof_vel'][i],cache['committed__body_rot'][i,anchor])
    for i in range(200):
        for j, offset in enumerate((1,2,4,8)):
            add(i+offset,cache['forecast__dof_pos'][i,j],cache['forecast__dof_vel'][i,j],cache['forecast__body_rot'][i,j,anchor])
    if sorted(rows) != list(range(208)):
        raise ValueError('recorded grid is not complete 0..4.14s')
    return (np.arange(208)*.02, *[np.stack([rows[i][j] for i in range(208)]) for j in range(3)]), duplicates


def export_pilot(pilot, output):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    allowed = ROOT/'artifacts/runs/agentic'
    if allowed.resolve() not in output.parents:
        raise ValueError('output must be strictly under agentic artifact scope')
    output.mkdir(parents=True, exist_ok=False)
    tracked = [Path(__file__), *sorted((ROOT/'src/pulsemorph/motion').glob('*.py')),
               ROOT/'research/vendor/ProtoMotions/deployment/state_utils.py']
    before = {str(p.relative_to(ROOT)):sha(p) for p in tracked}
    for path in tracked:
        target = output/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path,target)
    write(output/'source_hashes_before.json',before)
    result = dict(schema='pulsemorph.agentic_common_export.v1',status='failed',trials=[],
                  limitation='saved/loaded reference input parity only; no new physics replay; off-grid derivative equivalence is not established')
    try:
        sys.path.insert(0,str(ROOT/'src'))
        from pulsemorph.motion.contracts import MotionReference, SCHEMA_VERSION, save_reference, load_reference
        from pulsemorph.motion.g1 import profile_from_yaml, assemble_reference_inputs
        spec=importlib.util.spec_from_file_location('_export_heading',tracked[-1])
        state=importlib.util.module_from_spec(spec); spec.loader.exec_module(state)
        comparison=read(pilot/'comparison.json'); protocol=read(pilot/'protocol.json')
        if not comparison['comparison_valid'] or sha(pilot/'protocol.json') != comparison['protocol_sha256']:
            raise ValueError('comparison/protocol invalid')
        selections=[('baseline',comparison['calibration']['candidate_id'])]+[(m['method'],m['selected']['candidate_id']) for m in comparison['methods']]
        for method, candidate in selections:
            trial=pilot/'trials'/candidate
            manifest=read(trial/'manifest.json'); proposal=read(trial/'proposal.json')
            for name,digest in manifest['artifact_hashes'].items():
                if sha(trial/name)!=digest: raise ValueError('trial artifact changed: '+name)
            pins=manifest['preexecution_input_hashes']
            yaml_name=next(name for name in pins if name.endswith('unified_pipeline.yaml'))
            yaml_path=trial/'source_snapshot'/yaml_name
            profile=profile_from_yaml(yaml_path)
            if profile['yaml_sha256']!=pins[yaml_name]: raise ValueError('YAML pin mismatch')
            cache=dict(np.load(trial/'reference_cache.npz',allow_pickle=False))
            arrays,duplicates=dense_grid(cache,profile['anchor_body_index'])
            qpos=np.load(trial/'qpos.npy',allow_pickle=False); qvel=np.load(trial/'qvel.npy',allow_pickle=False)
            start_hash=hashlib.sha256(qpos[0].tobytes()+qvel[0].tobytes()).hexdigest()
            sources=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in
                     [trial/'source.motion',trial/'proposal.json',trial/'reference_cache.npz',trial/'manifest.json',pilot/'protocol.json',pilot/'comparison.json',yaml_path,Path(profile['body_path'])]]
            audio=protocol['audio']
            if sha(audio['audio_path'])!=audio['audio_sha256']: raise ValueError('audio provenance mismatch')
            sources.append({'path':audio['audio_path'],'sha256':audio['audio_sha256']})
            kind='agentic' if method in ('llm_feedback','llm_feedback_off') else 'baseline'
            metadata=dict(schema_version=SCHEMA_VERSION,reference_id='compact_arms_v1:'+candidate,
                producer=dict(kind=kind,id='recorded-agentic-pilot:'+method,version='1'),
                body=dict(profile_id='g1',profile_sha256=profile['body_sha256'],joint_names=list(profile['joint_names']),anchor_body_name='torso_link'),
                controller=dict(profile_id='g1-bones-deploy',profile_sha256=profile['yaml_sha256'],control_dt_s=.02),
                frame=dict(id='recorded_reference_world',up_axis='z',handedness='right',quaternion_order='xyzw',alignment='recorded_reference_world'),
                context=dict(decision_id=candidate+':offline-take',observation_id=proposal.get('observation_id') or 'offline-protocol',task_id=protocol['task']['intent_id'],runtime_session_id=candidate,clock_epoch='fresh-simulation-zero',decision_time_s=0.),
                validity=dict(start_state_id='sha256:'+start_hash,start_time_s=0.,end_time_s=4.14,start_mode='fresh_source_reset',end_behavior='trial_end_no_extrapolation'),
                observation=dict(mode='offline_file',observed_through_time_s=4.,audio_observed_through_sample=88200,video_observed_through_time_s=None),
                provenance=dict(sources=sources,notes=['Actual recorded cache union; all duplicate samples checked exactly.','Executed interval 0..4s; targets extend to 4.14s for last decision future padding.','Recorded source-world torso xyzw; one heading alignment in shared input adapter.','Initial state fingerprint hashes qpos then qvel raw float64 bytes.','Offline full four-second authored audio available through 88200 samples; not live causal.','No new learning or physical replay; only target-input parity checked.','Shared Hermite/SLERP reconstruction parity measured at recorded policy query grid only.']))
            reference=MotionReference(metadata,*arrays)
            save_reference(reference,output/candidate)
            loaded=load_reference(output/candidate)
            feeds=dict(np.load(trial/'policy_inputs.npz',allow_pickle=False))
            heading=state.compute_yaw_offset_np(feeds['call_0001__current_anchor_rot'][0],cache['committed__body_rot'][0,profile['anchor_body_index']])
            maxima={key:0. for key in ('mimic_future_dof_pos','mimic_future_dof_vel','mimic_future_anchor_rot')}
            for i in range(200):
                assembled=assemble_reference_inputs(loaded,i*.02,heading_offset_xyzw=heading)
                for key,value in assembled.items():
                    recorded=feeds[f'call_{i+1:04d}__'+key]
                    error=float(np.max(np.abs(value.astype(float)-recorded.astype(float))))
                    maxima[key]=max(maxima[key],error)
                    if value.shape!=recorded.shape or error>2e-6: raise ValueError(f'input parity failed {candidate} decision {i} {key}: {error}')
            result['trials'].append(dict(candidate_id=candidate,method=method,producer_kind=kind,package=candidate,
                saved_loaded_valid=True,decision_count=200,numeric_grid_samples=208,duplicate_samples_checked=duplicates,
                max_abs_float32_error=maxima,tolerance_abs=2e-6,heading_offset_xyzw=heading.tolist(),initial_state_sha256=start_hash))
        after={str(p.relative_to(ROOT)):sha(p) for p in tracked}
        if before!=after: raise ValueError('exporter/common source changed during export')
        result.update(status='success',source_hashes_unchanged=True,source_hashes=before)
    except Exception as error:
        result['error']=repr(error)
    result['artifact_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=export_pilot(args.pilot,args.output)
    print(json.dumps(result,indent=2)); return 0 if result['status']=='success' else 1


if __name__=='__main__':
    raise SystemExit(main())

