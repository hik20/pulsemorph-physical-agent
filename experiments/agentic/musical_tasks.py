"""Original short music fixtures, waveform observations and frozen experiment goals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import wave

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.evidence import sha256, write_new_json

NAMES=['left_gain_a','right_gain_a','left_gain_b','right_gain_b','timing_shift_s']
METHODS=['jacobian_search','llm_feedback','llm_feedback_off']
RATE=22050


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def synthesize(cue_s,seed):
    rng=np.random.default_rng(seed)
    audio=np.zeros(4*RATE,dtype=float)
    def add(t,signal):
        i=round(t*RATE);count=min(len(signal),len(audio)-i)
        if count>0:audio[i:i+count]+=signal[:count]
    beats=np.arange(0,4,.5)
    for index,t in enumerate(beats):
        u=np.arange(round(.35*RATE))/RATE
        primary=abs(t-cue_s)<1e-9
        phase=2*np.pi*(48*u+45*.025*(1-np.exp(-u/.025)))
        kick=(.68 if primary else .18)*np.sin(phase)*np.exp(-u/.075)*(1-np.exp(-u/.003))
        add(t,kick)
        if index%2:
            add(t,.065*rng.standard_normal(len(u))*np.exp(-u/.025))
        base=130.813 if t<2 else 146.832
        chord=sum(np.sin(2*np.pi*base*r*u) for r in [1,1.25,1.5])
        add(t,.024*chord*np.exp(-u/.16)*(1-np.exp(-u/.006)))
        h=np.arange(round(.06*RATE))/RATE
        noise=rng.standard_normal(len(h));high=noise-np.r_[0,noise[:-1]]
        add(t+.25,.018*high*np.exp(-h/.012))
    if np.max(np.abs(audio))>=.98:raise ValueError('unexpected audio clipping')
    return np.rint(audio*32767).astype('<i2'),dict(license='CC0-1.0',seed=seed,authored_primary_cue_s=cue_s,
        beat_times_s=beats.tolist(),section_boundary_s=2.,source='Original procedural kick/snare/chord/hi-hat; no external recordings')


def analyze_waveform(path):
    with wave.open(str(path),'rb') as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,RATE):raise ValueError('fixture format mismatch')
        pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(float)/32767
    low=sosfiltfilt(butter(4,140,fs=RATE,output='sos'),pcm)
    width=round(.020*RATE);hop=round(.005*RATE)
    starts=np.arange(0,len(pcm)-width+1,hop)
    energy=np.array([np.sqrt(np.mean(low[i:i+width]**2)) for i in starts])
    t=(starts+width/2)/RATE
    indices,properties=find_peaks(energy,distance=round(.28*RATE/hop),prominence=.008)
    events=[dict(event_id=f'bass_peak_{j:02d}',time_s=float(t[i]),low_band_rms=float(energy[i]),
                 prominence=float(properties['prominences'][j])) for j,i in enumerate(indices)]
    if not events:raise ValueError('no measured bass accents')
    chosen=max(events,key=lambda e:e['low_band_rms'])
    return dict(schema='pulsemorph.agentic_music_observation.v1',mode='offline_file',
        waveform_path=str(Path(path).resolve()),waveform_sha256=sha256(Path(path)),
        sample_rate_hz=RATE,observed_through_sample=len(pcm),duration_s=len(pcm)/RATE,
        measured_bass_events=events,strongest_bass_event_id=chosen['event_id'],
        analysis=dict(lowpass_hz=140,order=4,zero_phase=True,rms_window_s=.020,hop_samples=hop,
                      minimum_prominence=.008,minimum_peak_separation_s=.28),
        limitations=['Offline zero-phase waveform features, not live causal analysis.',
            'Low-band peaks are measured; bass/drum semantics and musical beat ground truth are not validated.',
            'Two sections at 0..2/2..4s come from the authored task timeline, not learned phrase detection.'])


def initialize(output):
    out=Path(output).resolve()
    if (ROOT/'artifacts/runs/agentic').resolve() not in out.parents:raise ValueError('agentic output scope required')
    out.mkdir(parents=True,exist_ok=False)
    tasks=[];semantic=[]
    variants=[('swap_left_to_right',2.5,202610261,[.70,1.15,1.15,.70],
        'İlk bölümde sol kolu dar, sağ kolu geniş tut. İkinci bölümde rolleri değiştir: sol kol genişlesin, sağ kol daralsın. İkinci yarıdaki baskın hareket vurgusu müzikteki en güçlü bas vuruşuna gelsin. Başlangıç ve bitişe yumuşak dön; fiziksel geçerlilik öncelikli.'),
        ('swap_right_to_left',3.0,202610262,[1.15,.70,.70,1.15],
        'İlk bölümde sağ kolu dar, sol kolu geniş tut. İkinci bölümde rolleri değiştir: sağ kol genişlesin, sol kol daralsın. İkinci yarıdaki baskın hareket vurgusu müzikteki en güçlü bas vuruşuna gelsin. Başlangıç ve bitişe yumuşak dön; fiziksel geçerlilik öncelikli.')]
    for task_id,cue,seed,ratios,instruction in variants:
        directory=out/'music'/task_id;directory.mkdir(parents=True)
        pcm,truth=synthesize(cue,seed)
        path=directory/'music.wav'
        with wave.open(str(path),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(RATE);f.writeframes(pcm.tobytes())
        write_new_json(directory/'authored_truth.json',truth)
        observation=analyze_waveform(path)
        write_new_json(directory/'waveform_observation.json',observation)
        event=next(e for e in observation['measured_bass_events'] if e['event_id']==observation['strongest_bass_event_id'])
        if abs(event['time_s']-cue)>.08:raise ValueError('waveform cue detector missed authored event')
        goal=dict(task_id=task_id,feature_targets=ratios+[event['time_s']],cue_event_id=event['event_id'],
            timing_target_source='measured low-band waveform peak; authored cue retained only for observation audit')
        tasks.append(dict(task_id=task_id,instruction=instruction,goal=goal,music_observation=observation,
            authored_cue_observation_error_s=event['time_s']-cue))
        semantic.append(dict(task_id=task_id,instruction=instruction,music_observation=observation,
            glossary={'dar':.70,'geniş':1.15,'normal':1.0},
            section_names=['A: ilk bölüm 0..2s','B: ikinci bölüm 2..4s'],
            goal_semantics='Ratios of actual joint-space RMS excursion versus unchanged baseline, not literal tool gains.',
            response_schema={'task_id':task_id,'feature_targets':['left_A','right_A','left_B','right_B','cue_time_seconds'],
                             'cue_event_id':'measured event identifier','brief_reason':'short decision summary only'}))
    protocol=dict(schema='pulsemorph.agentic_multifeature_protocol.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,methods=METHODS,parameter_names=NAMES,bounds=[[.35,1.6]]*4+[[-.30,.30]],
        anchor=[1.,1.,1.,1.,0.],trial_budget_per_method_per_task=8,shared_calibration_trials=1,
        duration_s=4.,source_start_s=0.,section_windows_s=[[.4,2.],[2.,3.6]],
        feature_windows_s=[[.8,1.6],[2.4,3.2]],feature_tolerances=[.08,.08,.08,.08,.10],
        feature_names=['left_rms_A_ratio','right_rms_A_ratio','left_rms_B_ratio','right_rms_B_ratio','dominant_accent_time_s'],
        physical_gate={'min_root_height_m':.60,'max_mean_joint_rmse_rad':.35},
        accent_window_s=[1.8,3.2],accent_smoothing_samples=5,accent_minimum_prominence=.15,
        selection='Physical gate first; then minimize maximum absolute feature error divided by predeclared tolerance; sum squared normalized errors then candidate_id as ties. Individual dimensions always reported.',
        optimizer={'method':'finite-difference Jacobian + bounded damped least squares + Broyden repair',
                   'finite_difference_steps':[.15,.15,.15,.15,.10],
                   'budget_schedule':'anchor, five independent parameter probes, two observed-result repairs'},
        feedback_off_definition='All eight proposals committed before any candidate result is exposed; common independent final selector still sees results.',
        semantic_stage='Raw Turkish instruction + explicit glossary + actual waveform observation. Interpretation scored separately; all numerical conditions then receive the same authored canonical goal.',
        model={'requested_model':'gpt-6.1-sol','access':'Codex collaboration','provider_token_count':None,'billed_cost':None},
        evidence_scope='Two authored development tasks on one existing motion and fixed pretrained tracker; not a held-out generalization or perceptual benchmark',
        limitations=['All execution is a new offline take with full file lookahead.',
            'No new weights, video reconstruction, human preference, universal dance quality or hardware safety.',
            'Source motion retiming also affects the lower body; motion features and physics coupling are recorded separately.',
            'The shared MotionReference is actually sampled in the control path; full-body carriers remain source interpolation, not edited-limb FK.'])
    write_new_json(out/'protocol.json',protocol)
    write_new_json(out/'observations/semantic_tasks.json',{'schema':'pulsemorph.agentic_semantic_tasks.v1','tasks':semantic,
        'instructions':'For each Turkish instruction infer the five requested target features and the referenced measured bass event. Do not optimize tool parameters. Return short structured results, no private reasoning.'})
    frozen=read(ROOT/'artifacts/models/learned_rhythm/v2/freeze.json')['sha256']
    write_new_json(out/'protected_hashes_before.json',{p:sha256(ROOT/p) for p in frozen})
    proposal=dict(candidate_id='calibration_0',method='calibration',task_id='shared_baseline',trial_index=0,
        observation_id='offline_source_reset',parameters=dict(zip(NAMES,protocol['anchor'])),
        duration_s=4.,source_start_s=0.,section_windows_s=protocol['section_windows_s'],
        audio={'duration_s':4.,'observed_through_sample':4*RATE},protocol_sha256=sha256(out/'protocol.json'),
        source_kind='baseline',brief_reason='One shared unchanged physical calibration; costs counted separately.')
    write_new_json(out/'proposals/calibration_0.json',proposal)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(initialize(a.output))
