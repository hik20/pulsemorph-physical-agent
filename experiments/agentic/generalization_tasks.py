"""Freeze a small source-file transfer experiment before new physics outcomes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import wave
import numpy as np
from musical_tasks import ROOT, RATE, NAMES, read, sha256, write_new_json, synthesize, analyze_waveform

METHODS = ['surrogate_search', 'llm_feedback', 'llm_feedback_off']


def music(cue, seed):
    # Old synthesizer with no strong grid beat, plus one explicitly off-grid kick.
    pcm, truth = synthesize(-1., seed)
    samples = pcm.astype(float) / 32767
    u = np.arange(round(.35 * RATE)) / RATE
    phase = 2*np.pi*(48*u + 45*.025*(1-np.exp(-u/.025)))
    kick = .62*np.sin(phase)*np.exp(-u/.075)*(1-np.exp(-u/.003))
    start = round(cue*RATE)
    count = min(len(kick), len(samples)-start)
    samples[start:start+count] += kick[:count]
    if np.max(abs(samples)) >= .98:
        raise ValueError('fixture clipping')
    truth.update(authored_primary_cue_s=cue, strong_cue_on_beat_grid=False,
                 source='Original procedural music plus off-grid prominent kick; CC0-1.0')
    return np.rint(samples*32767).astype('<i2'), truth


def initialize(output):
    out = Path(output).resolve()
    if (ROOT/'artifacts/runs/agentic').resolve() not in out.parents:
        raise ValueError('agentic output required')
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ['wave', 'high5']:
        path = Path('research/vendor/ProtoMotions/data/g1-kimodo-generated/proto')/f'output_{name}.motion'
        sources[name] = dict(path=path.as_posix(), sha256=sha256(ROOT/path))
    # Cues selected from source-only feasibility, before any new physical outcome.
    # Reference strongest events: wave3.14, high5 2.04. Reachability is not assumed.
    tasks = []
    specs = [('wave','a',2.98,3101), ('wave','b',3.10,3102),
             ('high5','a',2.16,3103), ('high5','b',2.28,3104)]
    for source_id, variant, cue, seed in specs:
        task_id = source_id+'_'+variant
        folder = out/'music'/task_id
        folder.mkdir(parents=True)
        pcm, truth = music(cue, seed)
        path = folder/'music.wav'
        with wave.open(str(path),'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(RATE); f.writeframes(pcm.tobytes())
        observation = analyze_waveform(path)
        event = next(e for e in observation['measured_bass_events'] if e['event_id']==observation['strongest_bass_event_id'])
        if abs(event['time_s']-cue) > .08:
            raise ValueError('cue waveform observation check failed')
        write_new_json(folder/'authored_truth.json', truth)
        write_new_json(folder/'waveform_observation.json', observation)
        ratios = [.80,1.20,1.20,.80] if variant=='a' else [1.10,.75,.75,1.10]
        instruction = ('İlk bölümde sol kolun hareketini normalin yüzde 80\'ine indir, sağ kolu yüzde 120\'ye çıkar; ikinci bölümde bu oranları değiştir.' if variant=='a' else
            'İlk bölümde sol kol yüzde 110, sağ kol yüzde 75 düzeyinde olsun; ikinci bölümde sol yüzde 75, sağ yüzde 110 olsun.')
        instruction += ' Ölçüm penceresindeki baskın hareket vurgusunu müziğin en güçlü bas olayına hizala. Fiziksel geçerlilik öncelikli.'
        tasks.append(dict(task_id=task_id, source_id=source_id, source_motion=sources[source_id],
            instruction=instruction, music_observation=observation,
            authored_cue_observation_error_s=event['time_s']-cue,
            goal=dict(task_id=task_id, feature_targets=ratios+[event['time_s']], cue_event_id=event['event_id'],
                timing_target_source='Measured waveform event, not authored onset')))
    code_names = ['generalization_tasks.py','generalization_driver.py','generalization_search.py',
        'execute_generalization.py','evaluate_generalization.py','evaluate_trial.py','musical_tasks.py','multifeature_driver.py']
    paths = [Path('experiments/agentic')/name for name in code_names]
    paths += list(Path('src/pulsemorph/agentic').glob('*.py'))
    paths += list(Path('src/pulsemorph/motion').glob('*.py'))
    code_pins = {p.as_posix():sha256(ROOT/p) for p in paths}
    for relative in code_pins:
        target = out/'protocol_source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    protocol = dict(schema='pulsemorph.agentic_generalization_protocol.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),tasks=tasks,sources=sources,methods=METHODS,
        execution_code_sha256=code_pins,parameter_names=NAMES,bounds=[[.35,1.6]]*4+[[-.30,.30]],
        anchor=[1.,1.,1.,1.,0.],trial_budget_per_method_per_task=8,shared_calibration_trials=2,
        duration_s=4.,source_start_s=0.,section_windows_s=[[.4,2.],[2.,3.6]],
        feature_windows_s=[[.8,1.6],[2.4,3.2]],feature_tolerances=[.08]*4+[.10],
        feature_names=['left_rms_A_ratio','right_rms_A_ratio','left_rms_B_ratio','right_rms_B_ratio','dominant_accent_time_s'],
        physical_gate={'min_root_height_m':.60,'max_mean_joint_rmse_rad':.35},
        accent_window_s=[1.8,3.2],accent_smoothing_samples=5,accent_minimum_prominence=.15,
        selection='Physical gate first, then worst absolute error divided by tolerance, squared normalized sum, candidate ID. Quality diagnostics do not rerank.',
        optimizer={'method':'deterministic linear RBF feature surrogate + Halton/global-local pools',
            'schedule':'anchor0, task-informed1, fitted search; maximin exploration at index4 within eight proposals',
            'evidence':'No unexecuted surrogate prediction counts as a measurement.'},
        feedback_off_definition='Eight individual model responses per task, sequentially conditioned on common calibration and own previous proposals only. All32 OFF proposals committed before any candidate physics.',
        model={'requested_model':'gpt-6.1-sol','access':'Codex collaboration','calls_per_method_task':8,
            'provider_tokens':None,'provider_billed_cost':None,'reasoning_setting':'same inherited setting for both LLM conditions'},
        semantic_stage='None. Canonical goals supplied equally; Turkish instructions provide context. No new semantic parsing success claim.',
        evidence_scope='Four authored tasks over two motion files unused in prior agentic optimization; same source family/body/pretrained motor. Source-only feasibility informed cue placement before physics.',
        quality_diagnostics=['PD target increments','20ms sampled finite-difference actual joint acceleration',
            'unwarped source-world versus observed root displacement, without heading correction'],
        planned_budget={'physical_trials':98,'control_steps':19600,'physics_steps':392000,'model_proposal_responses':64},
        limitations=['Four tasks are two correlated source groups, not independent human/body/motor generalization.',
            'Call counts matched; tokens, runtime compute, context size and billed costs are not matched or known.',
            'All audio observed offline through4s; body feedback from prior completed fresh-reset takes only.',
            'Target reachability is not established. All failures remain in the report; no target tuning after physics.',
            'Quality diagnostics are descriptive, not torque/foot-slip/perception or safety acceptance.',
            'No pretrained motor or generative weights are changed; no training admission, human-preference or product claim.'])
    write_new_json(out/'protocol.json',protocol)
    frozen = read(ROOT/'artifacts/models/learned_rhythm/v2/freeze.json')['sha256']
    write_new_json(out/'protected_hashes_before.json',{p:sha256(ROOT/p) for p in frozen})
    for source_id, source_motion in sources.items():
        candidate = 'calibration_'+source_id
        task = next(t for t in tasks if t['source_id']==source_id)
        proposal = dict(candidate_id=candidate,method='calibration',task_id=task['task_id'],source_id=source_id,
            source_motion=source_motion,trial_index=0,observation_id=candidate,
            parameters=dict(zip(NAMES,protocol['anchor'])),duration_s=4.,source_start_s=0.,
            section_windows_s=protocol['section_windows_s'],audio={'duration_s':4.,'observed_through_sample':88200},
            protocol_sha256=sha256(out/'protocol.json'),created_at_utc=datetime.now(timezone.utc).isoformat(),
            brief_reason='Shared unchanged calibration for this source; counted separately.')
        write_new_json(out/f'proposals/{candidate}.json',proposal)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(initialize(parser.parse_args().output))
