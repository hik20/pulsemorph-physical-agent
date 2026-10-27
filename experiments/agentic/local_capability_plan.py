"""Freeze a separate local-clock existence search after the global budget ends."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
from musical_tasks import ROOT, read, sha256, write_new_json


def initialize(output):
    out=Path(output).resolve()
    if (ROOT/'artifacts/runs/agentic').resolve() not in out.parents:
        raise ValueError('new agentic output required')
    old=ROOT/'artifacts/runs/agentic/generalization_music_v1'
    global_run=ROOT/'artifacts/runs/agentic/high5_capability_v1'
    global_summary=read(global_run/'summary.json')
    if (global_summary['status']!='no_witness_within_budget' or global_summary['search_candidates']!=64
            or global_summary['total_new_trials']!=65 or global_summary['error'] is not None):
        raise ValueError('completed unsuccessful fixed global search required')
    original=read(old/'protocol.json')
    audited=ROOT/'artifacts/runs/agentic/generalization_music_v1_audit_v1/audit.json'
    if not read(audited)['audit_passed']: raise ValueError('original audit required')
    def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path))
    task=next(t for t in original['tasks'] if t['task_id']=='high5_b')
    pins=dict(original['execution_code_sha256'])
    for name,digest in pins.items():
        if sha256(ROOT/name)!=digest: raise ValueError('original implementation changed: '+name)
    for relative in ['experiments/agentic/local_capability_plan.py','experiments/agentic/local_capability_driver.py',
        'experiments/agentic/capability_search.py','experiments/agentic/execute_local_capability.py',
        'src/pulsemorph/agentic/local_multifeature_reference.py']:
        pins[relative]=sha256(ROOT/relative)
    fields=['parameter_names','bounds','anchor','feature_names','feature_windows_s','feature_tolerances',
        'physical_gate','accent_window_s','accent_minimum_prominence','accent_smoothing_samples',
        'section_windows_s','duration_s','source_start_s']
    protocol={key:original[key] for key in fields}
    protocol.update(schema='pulsemorph.agentic_local_capability_protocol.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        task=task,goal=task['goal'],source_motion=task['source_motion'],new_candidate_budget=64,
        calibration_trials=1,confirmation_trials=1,max_new_physical_trials=66,
        calibration_proposal='proposals/calibration.json',prior_trial_records=[],
        old_protocol=ref(old/'protocol.json'),old_comparison=ref(old/'comparison.json'),old_audit=ref(audited),
        old_calibration=dict(proposal=ref(old/'proposals/calibration_high5.json'),
            evaluation=ref(old/'evaluations/calibration_high5.json'),execution_manifest=ref(old/'trials/calibration_high5/manifest.json')),
        global_phase=dict(protocol=ref(global_run/'protocol.json'),summary=ref(global_run/'summary.json'),
            interpretation='65 new global-tool trials including calibration, no witness in64 candidates; finite search only.'),
        execution_code_sha256=pins,fixed_control_tool='local_clock_1p4_2p8_v1',clock_window_s=[1.4,2.8],
        declared_input_hash_extras=['experiments/agentic/execute_local_capability.py','src/pulsemorph/agentic/local_multifeature_reference.py'],
        clock='source_time=t+shift*sin(pi*(t-1.4)/1.4)^2 only for1.4<t<2.8, identity outside; analytic derivative; C1 only',
        minimum_possible_source_rate=1-.3*3.141592653589793/1.4,
        search={'initial_stage':'candidate0 target-informed gains and clipped measured-calibration timing equation; candidates1..5 independent positive gain.20 or timing.10 probes about that fixed seed, negative if upper-bound blocked',
            'continuation':'candidate6 onward frozen capability_search.next_candidate with fresh calibration only as prior and all actual local results',
            'old_feature_labels_imported':False,'surrogate_predictions_are_evidence':False},
        selection='Physical/integrity gate then worst normalized target error, squared normalized error then candidate ID.',
        denominator_policy='Fresh local unchanged calibration; requires identical body/controller/source/timing/version pins, old input pins preserved with exactly declared new code dependencies, and full byte parity qpos/qvel/PD with original unchanged calibration.',
        stopping_policy='First full5-target physical/integrity success gets exactly one same-parameter fresh-reset confirmation with full byte parity; stop on confirmation pass or error. Otherwise64 new candidates maximum. No retries or threshold changes.',
        accounting='Zero imported old feature labels. One fresh calibration is the only prior measurement in local search and is already counted. Max1 calibration+64 candidates+1 confirmation=66. Zero LLM proposal calls. Global phase used24 old rows whereas this phase has a new seed/probe schedule: not a matched causal tool comparison.',
        scope='Exploratory local-tool solution witness for the same previously failed goal; tool designed using earlier failures, not held-out testing.',
        limitations=['Finding a local solution does not prove no global-tool solution exists or isolate clock support from search strategy.',
            'Reference time/gain support does not physically lock motion outside the edit; tracker lookahead and dynamic state propagate effects.',
            'Source clock is monotone and C1; no C2, torque, perceived smoothness, contact or hardware safety acceptance.',
            'Original3/4,1/4,0/4 comparison and previous65 global trials remain unchanged.',
            'No new learned weights, model downloads, training admission, independent music perception or product release.'])
    out.mkdir(parents=True,exist_ok=False)
    write_new_json(out/'protocol.json',protocol)
    for relative in pins:
        target=out/'protocol_source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    proposal=dict(candidate_id='calibration',task_id='high5_b',source_id='high5',source_motion=task['source_motion'],
        method='calibration',stage='calibration',trial_index=0,observation_id='local_capability_fresh_calibration',
        parameters=dict(zip(protocol['parameter_names'],protocol['anchor'])),duration_s=4.,source_start_s=0.,
        section_windows_s=protocol['section_windows_s'],audio={'duration_s':4.,'observed_through_sample':88200},
        control_tool=protocol['fixed_control_tool'],protocol_sha256=sha256(out/'protocol.json'),
        created_at_utc=datetime.now(timezone.utc).isoformat(),experiment_kind='separate_local_clock_witness_search')
    write_new_json(out/'proposals/calibration.json',proposal)
    protected=read(old/'protected_hashes_before.json')
    if any(sha256(ROOT/p)!=digest for p,digest in protected.items()):raise ValueError('protected evidence changed')
    write_new_json(out/'protected_hashes_before.json',protected)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(initialize(p.parse_args().output))
