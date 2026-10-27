"""Freeze the separate high5_b feasibility witness search before fresh trials."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
from musical_tasks import ROOT, read, sha256, write_new_json


def initialize(output):
    out=Path(output).resolve()
    if (ROOT/'artifacts/runs/agentic').resolve() not in out.parents:
        raise ValueError('new agentic run directory required')
    old=ROOT/'artifacts/runs/agentic/generalization_music_v1'
    original=read(old/'protocol.json')
    audited=ROOT/'artifacts/runs/agentic/generalization_music_v1_audit_v1/audit.json'
    audit=read(audited)
    if not audit['audit_passed'] or not audit['original_evidence_unchanged']:
        raise ValueError('successful previous audit required')
    task=next(t for t in original['tasks'] if t['task_id']=='high5_b')
    def ref(path):
        return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path))
    prior=[]
    for method in original['methods']:
        for i in range(8):
            identity=f'high5_b__{method}__{i}'
            ep=old/'evaluations'/f'{identity}.json'
            e=read(ep)
            if not e['integrity_passed'] or e['features']!=audit['trials'][identity]['features']:
                raise ValueError('prior outcome differs from audited measurement')
            prior.append(dict(candidate_id=identity,proposal=ref(old/'proposals'/f'{identity}.json'),
                evaluation=ref(ep),execution_manifest=ref(old/'trials'/identity/'manifest.json')))
    old_calibration=dict(proposal=ref(old/'proposals/calibration_high5.json'),
        evaluation=ref(old/'evaluations/calibration_high5.json'),
        execution_manifest=ref(old/'trials/calibration_high5/manifest.json'))
    pins=dict(original['execution_code_sha256'])
    for name,digest in pins.items():
        if sha256(ROOT/name)!=digest: raise ValueError('prior implementation changed: '+name)
    for name in ['capability_plan.py','capability_driver.py','capability_search.py']:
        path=Path('experiments/agentic')/name;pins[path.as_posix()]=sha256(ROOT/path)
    fields=['parameter_names','bounds','anchor','feature_names','feature_windows_s','feature_tolerances',
        'physical_gate','accent_window_s','accent_minimum_prominence','accent_smoothing_samples',
        'section_windows_s','duration_s','source_start_s']
    protocol={key:original[key] for key in fields}
    protocol.update(schema='pulsemorph.agentic_capability_protocol.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        task=task,goal=task['goal'],source_motion=task['source_motion'],
        new_candidate_budget=64,calibration_trials=1,confirmation_trials=1,max_new_physical_trials=66,
        calibration_proposal='proposals/calibration.json',prior_trial_records=prior,
        old_protocol=ref(old/'protocol.json'),old_comparison=ref(old/'comparison.json'),old_audit=ref(audited),
        old_calibration=old_calibration,execution_code_sha256=pins,
        original_execution_code_sha256=original['execution_code_sha256'],
        selection='Physical/integrity gate, minimum worst tolerance-normalized error; squared normalized error then candidate ID as ties.',
        search=dict(algorithm='capability_witness_v1',
            fixed_schedule=['local_minmax','coordinate_probe','local_minmax','global_rbf','local_minmax','coordinate_probe','global_rbf','halton_exploration'],
            trust_radii_normalized_by_16_candidate_block=[.20,.12,.07,.04],
            candidate_pool='2048 deterministic Halton global points and512 local points',
            local_model='weighted affine residual regression with ridge1e-5, nearest24 unique measured vectors; SLSQP minmax',
            global_model='linear RBF, degree0, smoothing1e-5',
            prediction_is_evidence=False),
        denominator_policy='Fresh unchanged calibration; full qpos/qvel/PD byte parity and identical execution pins with original calibration REQUIRED before old measurements may seed proposals. Abort on mismatch; no renormalization.',
        stopping_policy='Stop at first candidate passing all5 original targets and original physical/integrity gates; run exactly one identical-parameter fresh-reset confirmation. Require full qpos/qvel/PD byte parity and all gates. Confirmation failure aborts with execution_error, no retry or new candidate. Otherwise stop after64 candidates.',
        accounting='Old24 observations (20 unique vectors including duplicate anchors) are reused prior data, not new physics. New maximum1 calibration+64 candidates+1 confirmation=66. Zero new LLM proposal calls; algorithm engineering is separate.',
        scope='Post-comparison exploratory witness search, informed by old outcomes; not a new matched planner comparison or held-out generalization score.',
        limitations=['Finite failure is not an impossibility certificate.',
            'A witness proves only these targets within this tool/body/controller/reset/fixture; not perturbation robustness, perception, root/contact fidelity or training readiness.',
            'Original3/4,1/4,0/4 scores, thresholds, parameter domain and evidence remain unchanged.',
            'Local-time or larger-domain tools are not part of this frozen phase. Any later tool variant needs a separately frozen protocol.',
            'No new weights, downloads, corpus admission, deployment or external publication.'])
    out.mkdir(parents=True,exist_ok=False)
    write_new_json(out/'protocol.json',protocol)
    for relative in pins:
        target=out/'protocol_source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    cp=dict(candidate_id='calibration',task_id='high5_b',source_id='high5',source_motion=task['source_motion'],
        method='calibration',trial_index=0,observation_id='capability_fresh_calibration',
        parameters=dict(zip(protocol['parameter_names'],protocol['anchor'])),duration_s=4.,source_start_s=0.,
        section_windows_s=protocol['section_windows_s'],audio={'duration_s':4.,'observed_through_sample':88200},
        protocol_sha256=sha256(out/'protocol.json'),created_at_utc=datetime.now(timezone.utc).isoformat(),
        experiment_kind='independent_postcomparison_capability_witness_search')
    write_new_json(out/'proposals/calibration.json',cp)
    protected=read(old/'protected_hashes_before.json')
    if any(sha256(ROOT/p)!=digest for p,digest in protected.items()): raise ValueError('protected evidence changed')
    write_new_json(out/'protected_hashes_before.json',protected)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(initialize(p.parse_args().output))
