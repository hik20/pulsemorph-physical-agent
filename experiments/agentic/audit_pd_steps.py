"""Read-only saved-command audit; never runs physics, models, or training imports."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def require(ok, message):
    if not ok: raise ValueError(message)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='artifacts/runs/agentic/multifeature_music_v1')
    parser.add_argument('--output', default='artifacts/runs/agentic/multifeature_pd_steps_v1')
    args = parser.parse_args()
    source, out = (ROOT/args.input).resolve(), (ROOT/args.output).resolve()
    require(not out.exists(), 'output already exists; refusing evidence overwrite')
    inputs = {}
    def load(p):
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return read(p)
    comparison = load(source/'comparison.json')
    rows = {}
    selected = []
    for task in comparison['tasks']:
        for method in task['methods']:
            selected.append(method['selected']['candidate_id'])
            for row in method['trials']:
                cid = row['candidate_id']
                if cid == 'calibration_0': continue
                require(cid not in rows, 'duplicate candidate')
                rows[cid] = row
    require(len(rows) == 48 and len(selected) == 6, 'expected 48 candidates and six selections')
    results = []
    def metric(q, names):
        delta = np.diff(q.astype('float64'), axis=0)
        require(delta.shape == (199,29), 'expected 199 command transitions')
        maxima = np.max(np.abs(delta), axis=1)
        i,j = np.unravel_index(np.argmax(np.abs(delta)),delta.shape)
        windows=[]
        for center in [.4,2.,3.6]:
            transition_times=np.arange(1,200)*.02
            mask=np.abs(transition_times-center)<=.16+1e-12
            d=delta[mask]; maxima_window=np.max(abs(d),axis=1)
            local_i,local_j=np.unravel_index(np.argmax(abs(d)),d.shape)
            global_i=int(np.flatnonzero(mask)[local_i])
            windows.append(dict(center_s=center,start_s=center-.16,end_s=center+.16,
                interval_count=int(mask.sum()),max_abs_component_rad=float(abs(d[local_i,local_j])),
                p95_per_step_max_rad=float(np.percentile(maxima_window,95)),p99_per_step_max_rad=float(np.percentile(maxima_window,99)),
                rms_delta_rad=float(np.sqrt(np.mean(d**2))),maximum_time_s=float((global_i+1)*.02),
                maximum_to_control_index=global_i+1,maximum_joint_name=names[local_j]))
        return dict(interval_count=199, posthoc_matched_boundary_windows=windows, max_abs_component_rad=float(abs(delta[i,j])),
            p95_per_step_max_rad=float(np.percentile(maxima,95)),
            p99_per_step_max_rad=float(np.percentile(maxima,99)),
            rms_delta_rad=float(np.sqrt(np.mean(delta**2))),
            maximum=dict(diff_index=int(i), from_control_index=int(i), to_control_index=int(i+1),
                         time_s=float((i+1)*.02), joint_index=int(j), joint_name=names[j], signed_delta_rad=float(delta[i,j])),
            per_joint_delta_rad=delta.tolist(), per_step_max_abs_rad=maxima.tolist())
    for cid in ['calibration_0',*rows]:
        directory = source/'trials'/cid
        manifest = load(directory/'manifest.json')
        row = rows.get(cid)
        label_row = row if row is not None else comparison['calibration']
        require(sha(directory/'manifest.json') == label_row['execution_manifest_sha256'], 'comparison manifest mismatch')
        require(manifest['status']=='success' and manifest['actual_control_steps']==200 and
                manifest['control_dt']==.02 and manifest['physics_dt']==.001 and manifest['decimation']==20 and
                manifest['online_onnx_call_count']==200 and manifest['warmup_onnx_call_count']==1,
                'execution contract mismatch')
        # Verify every preserved artifact against its execution manifest, including source snapshots.
        for name,digest in manifest['artifact_hashes'].items():
            p = (directory/name).resolve()
            require(directory in p.parents and p.is_file() and sha(p)==digest, 'artifact hash mismatch: '+str(p))
            inputs[str(p.relative_to(ROOT))] = digest
        def array(name):
            p=directory/name
            require(name in manifest['artifact_hashes'], 'unhashed numerical input')
            return np.load(p,allow_pickle=False)
        targets=array('applied_joint_targets.npy'); qpos=array('qpos.npy'); times=array('state_times.npy')
        require(targets.shape==(200,29) and qpos.shape==(201,36) and times.shape==(201,), 'state shape mismatch')
        require(all(np.isfinite(a).all() for a in [targets,qpos,times]), 'nonfinite state')
        require(np.allclose(times,np.arange(201)*.02,rtol=0,atol=1e-10), 'state clock mismatch')
        names=load(directory/'motion_reference_draft.json')['joint_names']
        require(len(names)==29 and len(set(names))==29, 'joint names invalid')
        calls=load(directory/'policy_calls.json')
        require(len(calls)==201, 'policy count mismatch')
        with array('policy_inputs.npz') as feeds, array('policy_outputs.npz') as outputs:
            for k,call in enumerate(calls):
                require(call['call_index']==k and call['control_index']==(None if k==0 else k-1) and
                        call['role']==('warmup' if k==0 else 'control') and call['all_finite'], 'policy ordering mismatch')
                if k:
                    prefix=f'call_{k:04d}__'
                    require(np.array_equal(outputs[prefix+'joint_pos_targets'][0],targets[k-1]), 'policy/target mismatch')
                    require(np.array_equal(feeds[prefix+'current_dof_pos'][0],qpos[k-1,7:].astype('float32')), 'policy/state indexing mismatch')
        with array('reference_cache.npz') as cache:
            reference=cache['committed__dof_pos'].copy()
        require(reference.shape==(201,29) and np.isfinite(reference).all(), 'reference shape/finite mismatch')
        initial=targets[0].astype('float64')-qpos[0,7:]
        j=int(np.argmax(abs(initial)))
        result=dict(candidate_id=cid, role='shared_calibration' if row is None else 'candidate',
                    selected=cid in selected, motion_group_count=1, joint_names=names,
                    labels={k:label_row[k] for k in ['task_id','method','integrity_passed','physical_gate_passed','intent_passed']},
                    pd_command_steps=metric(targets,names), reference_committed_steps_0_through_199=metric(reference[:200],names),
                    initial_target_minus_initial_qpos=dict(max_abs_component_rad=float(abs(initial[j])), rms_rad=float(np.sqrt(np.mean(initial**2))),joint_index=j,joint_name=names[j],signed_delta_rad=float(initial[j]), per_joint_delta_rad=initial.tolist()),
                    integrity=dict(all_manifest_artifact_hashes_verified=True, policy_state_target_indexing_verified=True,
                                   state_clock_verified=True, manifest_sha256=sha(directory/'manifest.json')))
        results.append(result)
    # Recheck all read evidence before publishing any output.
    require(all(sha(ROOT/p)==digest for p,digest in inputs.items()), 'input changed during audit')
    def summary(items):
        values=np.array([r['pd_command_steps']['max_abs_component_rad'] for r in items])
        return dict(count=len(items), min_trial_max_rad=float(values.min()),median_trial_max_rad=float(np.median(values)),max_trial_max_rad=float(values.max()),intent_pass_count=sum(r['labels'].get('intent_passed') is True for r in items))
    report=dict(schema='pulsemorph.agentic_pd_steps_audit.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv], environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__),
        source_snapshot_sha256=sha(__file__), input_hashes=inputs, integrity_passed=True,
        clock_semantics='targets[i] applies over [i*0.02,(i+1)*0.02); diff[i] is target[i+1]-target[i] at (i+1)*0.02. Reference committed indices 0..199 use the same times. State row 200 is terminal and excluded from command-matched reference steps.',
        metric_semantics='RMS over all 199*29 signed deltas; percentiles use NumPy linear interpolation on 199 per-step absolute component maxima. Initial target minus initial physical joint qpos is separate, not an inter-command step.',
        thresholds_established=0, gates_changed=0, motion_group_count=1,
        limitations=['PD target changes are not actual torque, jerk, physical abruptness or human-perceived abruptness.', 'No simulation, model call, training import, gate change or reranking. Existing pass/fail labels preserved.', 'Source/reference smoothing does not establish motor-command smoothing. No safety or quality threshold is inferred from these values or other transition pilots.', 'Boundary windows are post-hoc descriptive windows with identical +/-0.16s duration and inclusive transition timestamps at 0.4,2.0,3.6s. Timing warp affects the whole take: these windows do not isolate causal effects; full-run and window maxima answer different questions.'],
        summaries=dict(shared_calibration=summary(results[:1]),all_48_candidates=summary(results[1:]),selected_6=summary([r for r in results if r['selected']])),trials=results)
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/'audit_pd_steps.py')
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    (out/'README.md').write_text('# Saved PD target step audit\n\nRead-only supplemental analysis of 49 existing trials (one shared calibration, 48 candidates; six selected). `report.json` contains all 199 by 29 signed command deltas per trial, matched committed-reference deltas, separate initialization offsets, preserved intent labels, input SHA-256 inventory, source snapshot, command and environment. All execution-manifest artifact hashes and policy/state/target indexing were verified.\n\nTargets apply for 20 ms; transition i→i+1 occurs at (i+1)*0.02 s. Reference uses committed rows 0..199; terminal row 200 is excluded. No threshold established, gate altered, simulation run, model called or training imported. PD target steps do not measure torque, jerk or perceived abruptness. One motion group only.\n',encoding='utf8')
    print(json.dumps(report['summaries'],indent=2))
if __name__=='__main__': main()
