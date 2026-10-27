"""Private experimental eight-second phrase trials with immutable local assets."""
from pathlib import Path
import json
import math
import os
import re
import shutil
import subprocess
import time
import wave
import numpy as np
from pulsemorph.agentic.phrase_program import parse_program,compile_program,GAIN_FIELDS,ProgramCompileError
from pulsemorph.agentic.phrase_reference import plain
from pulsemorph.motion.contracts import MotionReference,save_reference
from .assets import PhraseAssets,ASSET_ROOT,PUBLIC_PYTHON,sha,write_new


class PhraseRuntime:
    def __init__(self,root,python_executable,artifact_root):
        self.root=Path(root).resolve();self.python=Path(python_executable).resolve();self.artifact_root=Path(artifact_root).resolve()
        if self.python!=PUBLIC_PYTHON.resolve():raise ValueError('phrase runtime requires reviewed public isolated interpreter')
        self.assets=PhraseAssets(self.root)
        probe=subprocess.run([str(self.python),'-I',str(Path(__file__).with_name('worker.py')),'--environment'],cwd=self.root,capture_output=True,text=True,check=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.environment=json.loads(probe.stdout)
        self._implementation=self.implementation_pins()

    def implementation_pins(self):
        pins={str(self.root/p):h for p,h in self.assets.manifest['live_numeric_pins'].items()}
        pins.update({str(self.assets.directory/p):h for p,h in self.assets.manifest['files'].items()})
        pins[str(self.assets.directory/'manifest.json')]=sha(self.assets.directory/'manifest.json')
        for name in ('runtime.py','assets.py','worker.py'):
            path=Path(__file__).with_name(name);pins[str(path)]=sha(path)
        pins[str(self.python)]=sha(self.python)
        pins.update({p:sha(p) for p in self.environment['module_pins']})
        return pins

    def _unchanged(self):
        try:self.assets.verify()
        except (OSError,ValueError,KeyError) as error:raise RuntimeError('phrase assets/integrity changed') from error
        if self.implementation_pins()!=self._implementation:raise RuntimeError('phrase implementation changed; restart required')

    def _directory(self,output):
        path=Path(output).resolve()
        if path==self.artifact_root or not path.is_relative_to(self.artifact_root):raise ValueError('phrase trial path outside configured artifact root')
        return path

    def status(self):
        """Read-only admission; controller execution is verified per trial only."""
        required=256*1024**2
        result={'available':False,'state':'integrity_changed','execution_verified':False,'scope':'prepared_private_profile','free_bytes':None,'required_bytes':required}
        try:self._unchanged()
        except Exception:return result
        parent=self.artifact_root
        while not parent.exists() and parent.parent!=parent:parent=parent.parent
        try:result['free_bytes']=int(shutil.disk_usage(parent).free)
        except OSError:result['state']='insufficient_disk';return result
        result.update(available=result['free_bytes']>=required,state='profile_verified_no_controller_execution' if result['free_bytes']>=required else 'insufficient_disk')
        return result

    @staticmethod
    def _music(observation):
        if type(observation) is not dict or observation.get('available') is not True:raise ValueError('actual eight-second music observation required')
        path=Path(observation['path']).resolve();count=observation.get('observed_through_sample',observation.get('sample_frames'))
        if observation.get('duration_s')!=8 or observation.get('sample_rate')!=48000 or count!=384000 or path.stat().st_size>32*1024**2 or sha(path)!=observation.get('sha256'):raise ValueError('music observation binding differs')
        with wave.open(str(path),'rb') as pcm:
            if (pcm.getframerate(),pcm.getnframes(),pcm.getsampwidth(),pcm.getcomptype())!=(48000,384000,2,'NONE') or pcm.getnchannels() not in (1,2) or len(pcm.readframes(384000))!=384000*2*pcm.getnchannels():raise ValueError('music must be complete eight-second PCM16 WAV')
        return {'available':True,'path':str(path),'sha256':sha(path),'duration_s':8.,'sample_rate':48000,'observed_through_sample':384000}

    def run_trial(self,program,output,job_id,trial_index,music_observation,cancel_event=None):
        p=parse_program(program)
        if p['order']!=['wave','high5']:raise ValueError('reverse phrase remains reference-only; app admission is Wave→High5 only')
        if type(job_id) is not str or re.fullmatch(r'[A-Za-z0-9_-]{1,100}',job_id) is None or type(trial_index) is not int or trial_index< -1:raise ValueError('invalid phrase job/index')
        if trial_index==-1 and any(p[k]!=1 for k in GAIN_FIELDS):raise ValueError('baseline must have unit gains')
        self._unchanged();output=self._directory(output)
        if cancel_event and cancel_event.is_set():raise RuntimeError('phrase trial cancelled before execution')
        if output.exists():raise FileExistsError('phrase trial output already consumed')
        output.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(output.parent).free<256*1024**2:raise RuntimeError('phrase trial requires256MiB free')
        music=self._music(music_observation)
        output.mkdir()
        identifier=job_id+('__baseline' if trial_index==-1 else '__trial_'+str(trial_index))
        try:
            refs=self.assets.references()
            reference,report=compile_program(refs,p,reference_id=identifier,sources=[*self.assets.source_records(),{'path':music['path'],'sha256':music['sha256']}])
            metadata=plain(reference.metadata);metadata['context'].update(task_id=job_id,runtime_session_id=identifier,observation_id='phrase:offline-music:'+music['sha256'])
            metadata['observation'].update(mode='offline_file',observed_through_time_s=8.,audio_observed_through_sample=384000)
            metadata['provenance']['notes'].append('Actual8s PCM binding; caller supplies intent/program. Fixed clock is not beat synchronization; historical origin records do not imply runtime file availability.')
            reference=MotionReference(metadata,reference.time_offsets_s,reference.joint_position_rad,reference.joint_velocity_rad_s,reference.anchor_orientation_xyzw)
            save_reference(reference,output/'common_reference');write_new(output/'compile_report.json',report)
        except ProgramCompileError as e:
            write_new(output/'rejection.json',{'status':'rejected_compiler','report':e.report,'new_physics_steps':0,'new_onnx_calls':0});raise
        self._unchanged()
        pins=dict(self._implementation);pins[music['path']]=music['sha256']
        for name in ('common_reference/manifest.json','common_reference/arrays.npz','compile_report.json'):pins[str(output/name)]=sha(output/name)
        request={'schema':'pulsemorph.app_phrase_request.v1','id':identifier,'index':trial_index,'job_id':job_id,'program':p,'artifact_root':str(self.artifact_root),'music_observation':music,'environment':self.environment,'source_pins':pins}
        write_new(output/'request.json',request);request_hash=sha(output/'request.json')
        if cancel_event and cancel_event.is_set():write_new(output/'interrupted.json',{'reason':'cancelled_before_worker'});raise RuntimeError('phrase trial cancelled before worker')
        command=[str(self.python),'-I',str(Path(__file__).with_name('worker.py')),'--request',str(output/'request.json')]
        with (output/'worker.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=self.root,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));started=time.monotonic()
            while process.poll() is None:
                if (cancel_event and cancel_event.is_set()) or time.monotonic()-started>300:
                    process.kill();process.wait();write_new(output/'interrupted.json',{'reason':'cancelled' if cancel_event and cancel_event.is_set() else 'timeout','durable_attempt_records':len(list((output/'attempts').glob('call_*.json'))),'completed_calls':'unconfirmed; partial evidence retained'});raise RuntimeError('phrase trial cancelled or timed out')
                time.sleep(.05)
        self._unchanged()
        if process.returncode or not (output/'result.json').is_file():raise RuntimeError('phrase worker failed; preserved '+str(output))
        result=json.loads((output/'result.json').read_text(encoding='utf8'))
        if result['request_sha256']!=request_hash or result['program']!=p or result['status']!='completed' or result.get('integrity_passed') is not True:raise RuntimeError('phrase completion binding/audit failed')
        trial={'id':identifier,'index':trial_index,'program':p,'paths':{'directory':str(output),'result':str(output/'result.json'),'audit':str(output/'audit.json'),'common_reference':str(output/'common_reference')},'metrics':result['metrics'],'integrity_passed':True}
        self._verify_trial(trial)
        return trial

    def _verify_trial(self,trial):
        try:return self._verify_trial_files(trial)
        except (OSError,ValueError,KeyError,TypeError) as error:raise RuntimeError('phrase saved integrity invalid') from error

    def _verify_trial_files(self,trial):
        directory=self._directory(trial['paths']['directory']);result=json.loads((directory/'result.json').read_text(encoding='utf8'));audit=json.loads((directory/'audit.json').read_text(encoding='utf8'))
        physical=trial.get('physical_metrics',trial['metrics'])
        if result.get('integrity_passed') is not True or result['status']!='completed' or audit.get('integrity_passed') is not True or sha(directory/'audit.json')!=result['audit_sha256'] or trial['program']!=result['program'] or physical!=result['metrics'] or audit['metrics']!=result['metrics']:raise RuntimeError('phrase saved audit/result binding failed')
        if result['schema']!='pulsemorph.app_phrase_physical_trial.v1' or audit['schema']!='pulsemorph.app_phrase_audit.v1' or audit['fresh_mj_kinematics_calls']!=400 or result['counts']!={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1}:raise RuntimeError('phrase proof scope/counters differ')
        if any(result['source_pins'].get(p)!=h for p,h in self._implementation.items()) or sha(directory/'request.json')!=result['request_sha256']:raise RuntimeError('phrase implementation/request proof missing')
        required={'qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy','policy_inputs.npz','policy_outputs.npz','policy_calls.json','substep_telemetry.npz','request.json','common_reference/manifest.json','common_reference/arrays.npz'}
        if not required.issubset(audit['inventory']):raise RuntimeError('phrase proof inventory incomplete')
        for relative,digest in audit['inventory'].items():
            path=(directory/relative).resolve()
            if not path.is_relative_to(directory) or sha(path)!=digest:raise RuntimeError('phrase saved trial was altered')
        for path,digest in result['source_pins'].items():
            if sha(path)!=digest:raise RuntimeError('phrase trial external input changed')
        return result

    def evaluate_trial(self,trial,baseline,targets):
        self._unchanged();self._verify_trial(trial);self._verify_trial(baseline)
        p,b=trial['program'],baseline['program']
        if p['order']!=b['order'] or p['heading_policy']!=b['heading_policy'] or any(b[k]!=1 for k in GAIN_FIELDS):raise ValueError('same-heading/order unit-gain baseline required')
        if type(targets) is not list or len(targets)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in targets):raise ValueError('four finite positive phrase targets required')
        physical_trial=trial.get('physical_metrics',trial['metrics']);physical_baseline=baseline.get('physical_metrics',baseline['metrics'])
        raw=np.asarray(physical_trial['arm_rms_rad'],float);denom=np.asarray(physical_baseline['arm_rms_rad'],float)
        if raw.shape!=(4,) or denom.shape!=(4,) or not np.isfinite(raw).all() or not np.isfinite(denom).all() or np.any(denom<=1e-8):raise RuntimeError('undefined phrase baseline RMS denominator')
        features=(raw/denom).tolist();errors=[abs(f-t)/.08 for f,t in zip(features,targets)];physical=bool(physical_trial['physical_gate_passed'] and physical_baseline['physical_gate_passed'])
        return {'integrity_passed':True,'features':features,'targets':list(targets),'tolerances':[.08]*4,'normalized_errors':errors,'goals_passed':sum(e<=1 for e in errors),'goal_total':4,'physical_gate_passed':physical,'worst_error':max(errors),'utility_l2':math.sqrt(sum(e*e for e in errors)),'qualification':'Actual arm-RMS ratios vs same-heading unit-gain physical baseline, windows1.2..3.2/5.2..7.2,own Wave0/High50 centers; no beat/aesthetic/learning guarantee.'}
