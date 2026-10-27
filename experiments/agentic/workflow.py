"""File-backed agentic trial workflow. Model proposals never serve as acceptance.

The pilot deliberately separates an external proposer from local execution.
All immutable records use exclusive creation; a run is a new offline take.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pulsemorph.evidence import sha256, write_new_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def create_pilot(output):
    out = Path(output).resolve()
    allowed = (ROOT / 'artifacts/runs/agentic').resolve()
    if allowed not in out.parents:
        raise ValueError('pilot output must be inside agentic artifacts')
    out.mkdir(parents=True, exist_ok=False)
    bundle_path = ROOT / 'artifacts/runs/music_control/audio/authored_window_v1/conditions.json'
    audio = read(bundle_path)
    window = audio['window']
    if sha256(Path(window['audio_path'])) != window['audio_sha256']:
        raise ValueError('audio fixture bytes changed')
    beats = [b['time_s'] - window['original_start_s'] for b in audio['truth_provenance']['data']['beats']
             if window['original_start_s'] <= b['time_s'] <= window['original_start_s'] + 4]
    protocol = dict(schema='pulsemorph.agentic_pilot_protocol.v1', created_at_utc=stamp(),
        evidence_kind='single_task_exploratory_mechanism_pilot',
        task=dict(intent_id='compact_arms_070',
            text='Keep the four-second take and source timing. Make the arm excursion in the middle about 70% of the unchanged take; return smoothly. Preserve the lower-body reference.',
            target_ratio=.70, tolerance=.05, measurement_window_s=[1.5, 2.5],
            metric='RMS arm-joint displacement from shared initial joint posture, measured on actual physics states; not perceptual width'),
        gates=dict(minimum_root_height_m=.60, maximum_joint_rmse_rad=.35,
            scope='local offline trial screening, not safety, aesthetic quality or runtime promotion'),
        tool=dict(name='edit_arm_amplitude', arm_amplitude_bounds=[0., 1.],
            source_rate=1., source_start_s=0., duration_s=4., edit_window_s=[1., 3.],
            envelope='1+(amplitude-1)*sin(pi*(t-1)/2)^2 within (1,3); source outside'),
        methods=['simple_search', 'llm_feedback', 'llm_feedback_off'], trial_budget_per_method=3,
        shared_calibration_trials=1, feedback_off_definition='All three proposals committed before any candidate outcome; same independent final selector for all methods',
        selection='Among physically eligible trials minimize abs(intent_ratio-.70), deterministic candidate_id tie-break; no eligible candidate => no selection',
        paired_first_amplitude=.5, evaluation_beats_s=beats,
        audio=dict(**window, mode='file_lookahead', observed_through_sample=window['sample_count'],
            lookahead_s=4., role='fixed authored audio + beat map available; does not change rate in this pilot'),
        source_bundle=dict(path=str(bundle_path.relative_to(ROOT)), sha256=sha256(bundle_path)),
        initialization='fresh source reset for each offline take, not mid-rollout snapshot restore',
        model=dict(requested_model='gpt-6.1-sol', access='Codex collaboration subagent',
            provider_api_receipt=None, token_counts=None, billed_cost=None,
            note='Actual model tool experiment; no external-provider API benchmark or cost claim'),
        limitations=['One source, one structured objective, no held-out semantic benchmark',
            'No human preference or VLM/video evidence', 'No new learned weights',
            'Full-take forecast may affect actual body before edit starts; no live past rewriting'])
    write_new_json(out / 'protocol.json', protocol)
    freeze = read(ROOT / 'artifacts/models/learned_rhythm/v2/freeze.json')['sha256']
    write_new_json(out / 'protected_hashes_before.json', {p:sha256(ROOT/p) for p in freeze})
    proposal = dict(candidate_id='calibration_0', method='calibration', trial_index=0,
        arm_amplitude=1., **{k:protocol['tool'][k] for k in ['source_rate','source_start_s','duration_s','edit_window_s']},
        protocol_sha256=sha256(out/'protocol.json'), producer_kind='baseline', observation_id=None,
        brief_reason='Shared unchanged reference calibration; counted separately from the equal trial budgets.')
    write_new_json(out/'proposals/calibration_0.json', proposal)
    for method in ['llm_feedback','llm_feedback_off']:
        observation = dict(schema='pulsemorph.agentic_observation.v1', observation_id=method+'_initial',
            method=method, protocol_sha256=sha256(out/'protocol.json'), task=protocol['task'],
            gates=protocol['gates'], tool=protocol['tool'], budget=3,
            paired_first_amplitude=.5, prior_trial_results=[], baseline_intent_ratio=1.,
            model_access_limits='Only this supplied observation; no other run files or sibling outcomes.',
            response_format={'arm_amplitudes':[.5, 'second','third'] if method.endswith('_off') else [.5],
                             'brief_reason':'short decision summary, no private reasoning'},
            instructions='Return three amplitudes now, without physics feedback, and commit them all.' if method.endswith('_off') else 'Return first amplitude .5 now. Future observations will contain only your earlier trial outcomes.',
            audio_context=protocol['audio'], evaluation_beats_s=beats)
        write_new_json(out/f'observations/{method}_initial.json', observation)
    return out


def submit(output, method, index, amplitude, observation, reason, response_path=None):
    out=Path(output).resolve(); protocol=read(out/'protocol.json')
    if method not in protocol['methods'] or not isinstance(index,int) or not 0<=index<3:
        raise ValueError('method/index outside fixed budget')
    if isinstance(amplitude,bool) or not np.isfinite(amplitude) or not 0<=amplitude<=1:
        raise ValueError('amplitude outside authorized tool domain')
    if index == 0 and amplitude != .5:
        raise ValueError('first candidate must be paired at .5')
    if index and not (out/f'proposals/{method}_{index-1}.json').exists():
        raise ValueError('missing prior committed proposal')
    if method.startswith('llm') and not response_path:
        raise ValueError('LLM proposal requires recorded external response')
    if method=='llm_feedback_off' and any(p.is_dir() and p.name!='calibration_0' for p in (out/'trials').glob('*')):
        raise ValueError('feedback-off proposals must all precede candidate trial outcomes')
    obs=Path(observation).resolve() if observation else None
    if obs and (out/'observations').resolve() not in obs.parents:
        raise ValueError('observation outside this pilot')
    if obs and read(obs).get('method') != method:
        raise ValueError('observation method mismatch')
    if method.startswith('llm'):
        response=read(response_path)
        values=response['arm_amplitudes']
        if not obs or Path(response['only_observation_path']).resolve()!=obs:
            raise ValueError('response observation mismatch')
        if method=='llm_feedback_off':
            if len(values)!=3 or values[index]!=amplitude:
                raise ValueError('amplitude differs from precommitted response')
        elif len(values)!=1 or values[0]!=amplitude or len(read(obs)['prior_trial_results'])!=index:
            raise ValueError('amplitude or feedback revision mismatch')
    candidate_id=f'{method}_{index}'
    proposal=dict(candidate_id=candidate_id, method=method, trial_index=index, arm_amplitude=float(amplitude),
        **{k:protocol['tool'][k] for k in ['source_rate','source_start_s','duration_s','edit_window_s']},
        producer_kind='baseline' if method=='simple_search' else 'agentic',
        protocol_sha256=sha256(out/'protocol.json'), created_at_utc=stamp(),
        observation_id=read(obs)['observation_id'] if obs else None,
        observation_artifact={'path':str(obs.relative_to(out)), 'sha256':sha256(obs)} if obs else None,
        response_artifact={'path':str(Path(response_path).resolve().relative_to(out)), 'sha256':sha256(Path(response_path))} if response_path else None,
        brief_reason=reason, intended_application='new offline take only')
    write_new_json(out/f'proposals/{candidate_id}.json',proposal)
    return proposal


def simple_amplitude(observed, target=.7):
    """Bounded secant with unchanged calibration, no model calls."""
    points=[(1.,1.)]+[(float(r['arm_amplitude']),float(r['intent_ratio'])) for r in observed
                     if r.get('integrity_passed') and r.get('intent_ratio') is not None]
    if not observed:return .5
    points.sort(key=lambda p:abs(p[1]-target))
    a,b=points[:2] if len(points)>1 else (points[0],(0.,0.))
    if abs(a[1]-b[1])<1e-8:return float((a[0]+b[0])/2)
    return float(np.clip(a[0]+(target-a[1])*(a[0]-b[0])/(a[1]-b[1]),0,1))


def observe(output, method, results):
    out=Path(output); initial=read(out/f'observations/{method}_initial.json')
    if method!='llm_feedback' or len(results) not in (1,2):
        raise ValueError('feedback permitted only after first/second feedback trial')
    initial.update(observation_id=f'{method}_after_{len(results)}', prior_trial_results=results,
        budget_remaining=3-len(results), instructions='Propose exactly one next amplitude within [0,1] based on only these observed physics results. Short reason only.',
        response_format={'arm_amplitudes':['next'], 'brief_reason':'short summary'})
    path=out/f'observations/{initial["observation_id"]}.json';write_new_json(path,initial)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--initialize',type=Path,required=True)
    args=parser.parse_args();print(create_pilot(args.initialize))
