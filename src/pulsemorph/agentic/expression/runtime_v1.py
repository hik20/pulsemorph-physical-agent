"""Separate expression runtime; explicit new schemas with fixed numeric controller."""
from pathlib import Path
import json,math,os,re,shutil,subprocess,time
import numpy as np
from pulsemorph.agentic.phrase_expression_v1 import parse_expression,compile_expression,GAIN_FIELDS
from pulsemorph.agentic.phrase_program import ProgramCompileError
from pulsemorph.agentic.phrase_reference import plain
from pulsemorph.motion.contracts import MotionReference,save_reference,load_reference
from pulsemorph.agentic.app.phrase.assets import PhraseAssets,ASSET_ROOT,PUBLIC_PYTHON,sha,write_new
from pulsemorph.agentic.app.phrase.runtime import PhraseRuntime as FrozenRuntime
from .player_v1 import validate_expression_reference

def expected_implementation_pins(root,assets,python,environment):
    root=Path(root);pins={str(root/p):h for p,h in assets.manifest['live_numeric_pins'].items()}
    pins.update({str(assets.directory/p):h for p,h in assets.manifest['files'].items()});pins[str(assets.directory/'manifest.json')]=sha(assets.directory/'manifest.json')
    names=('src/pulsemorph/agentic/app/phrase/runtime.py','src/pulsemorph/agentic/app/phrase/assets.py','src/pulsemorph/agentic/app/phrase/worker.py','src/pulsemorph/agentic/phrase_expression_v1.py','src/pulsemorph/agentic/expression/__init__.py','src/pulsemorph/agentic/expression/player_v1.py','src/pulsemorph/agentic/expression/runtime_v1.py','src/pulsemorph/agentic/expression/worker_v1.py')
    for relative in names:
        path=root/relative;pins[str(path)]=sha(path)
    pins[str(Path(python))]=sha(python);pins.update({p:sha(p) for p in environment['module_pins']});return pins

def validate_trial_identity(trial,request,result):
    if type(trial.get('id')) is not str or type(trial.get('index')) is not int:raise RuntimeError('Expression wrapper ID/index types invalid')
    for value in (request,result):
        if type(value.get('id')) is not str or type(value.get('index')) is not int or type(value.get('job_id')) is not str:raise RuntimeError('Expression saved ID/index/job types invalid')
        if value['id']!=trial['id'] or value['index']!=trial['index']:raise RuntimeError('Expression wrapper ID/index assignment mismatch')
    if request['job_id']!=result['job_id']:raise RuntimeError('Expression saved job assignment mismatch')

def validate_trial_paths(trial,directory):
    paths=trial.get('paths');directory=Path(directory).resolve()
    if type(paths) is not dict or set(paths)!={'directory','result','audit','common_reference'}:raise RuntimeError('Exact expression wrapper paths required')
    expected={'directory':directory,'result':directory/'result.json','audit':directory/'audit.json','common_reference':directory/'common_reference'}
    for name,path in paths.items():
        if type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts or Path(path).absolute()!=expected[name]:raise RuntimeError('Expression wrapper path assignment mismatch')
        for part in (Path(path),*Path(path).parents):
            if part.is_symlink() or (part.exists() and bool(getattr(part.lstat(),'st_file_attributes',0)&1024)):raise RuntimeError('Linked expression wrapper path forbidden')

class ExpressionRuntime:
    def __init__(self,root,python_executable,artifact_root):
        self.root=Path(root).resolve();self.python=Path(python_executable).resolve();self.artifact_root=Path(artifact_root).resolve()
        if self.python!=PUBLIC_PYTHON.resolve():raise ValueError('phrase runtime requires reviewed public isolated interpreter')
        self.assets=PhraseAssets(self.root)
        probe=subprocess.run([str(self.python),'-I',str(Path(__file__).with_name('worker_v1.py')),'--environment'],cwd=self.root,capture_output=True,text=True,check=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.environment=json.loads(probe.stdout)
        self._implementation=self.implementation_pins()

    def implementation_pins(self):
        return expected_implementation_pins(self.root,self.assets,self.python,self.environment)

    def _unchanged(self):
        try:self.assets.verify()
        except (OSError,ValueError,KeyError) as error:raise RuntimeError('phrase assets/integrity changed') from error
        if self.implementation_pins()!=self._implementation:raise RuntimeError('phrase implementation changed; restart required')

    def _directory(self,output):
        raw=Path(output).absolute()
        for part in (raw,*raw.parents):
            if part.is_symlink() or (part.exists() and bool(getattr(part.lstat(),'st_file_attributes',0)&1024)):raise ValueError('Linked expression artifact path forbidden')
        path=raw.resolve()
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

    _music=staticmethod(FrozenRuntime._music)

    def run_trial(self,expression,output,job_id,trial_index,music_observation,cancel_event=None):
        p=parse_expression(expression)
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
            reference,report=compile_expression(refs,p,reference_id=identifier,sources=[*self.assets.source_records(),{'path':music['path'],'sha256':music['sha256']}])
            metadata=plain(reference.metadata);metadata['context'].update(task_id=job_id,runtime_session_id=identifier,observation_id='phrase:offline-music:'+music['sha256'])
            metadata['observation'].update(mode='offline_file',observed_through_time_s=8.,audio_observed_through_sample=384000)
            metadata['provenance']['notes'].append('Actual8s PCM binding; caller supplies original expression. Fixed clock is not beat synchronization; historical origin records do not imply runtime file availability.')
            reference=MotionReference(metadata,reference.time_offsets_s,reference.joint_position_rad,reference.joint_velocity_rad_s,reference.anchor_orientation_xyzw)
            save_reference(reference,output/'common_reference');write_new(output/'compile_report.json',report)
        except ProgramCompileError as e:
            write_new(output/'rejection.json',{'status':'rejected_compiler','report':e.report,'new_physics_steps':0,'new_onnx_calls':0});raise
        self._unchanged()
        pins=dict(self._implementation);pins[music['path']]=music['sha256']
        for name in ('common_reference/manifest.json','common_reference/arrays.npz','compile_report.json'):pins[str(output/name)]=sha(output/name)
        request={'schema':'pulsemorph.expression_physical_request.v1','id':identifier,'index':trial_index,'job_id':job_id,'expression':p,'artifact_root':str(self.artifact_root),'music_observation':music,'environment':self.environment,'source_pins':pins}
        write_new(output/'request.json',request);request_hash=sha(output/'request.json')
        if cancel_event and cancel_event.is_set():write_new(output/'interrupted.json',{'reason':'cancelled_before_worker'});raise RuntimeError('phrase trial cancelled before worker')
        command=[str(self.python),'-I',str(Path(__file__).with_name('worker_v1.py')),'--request',str(output/'request.json')]
        with (output/'worker.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=self.root,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));started=time.monotonic()
            while process.poll() is None:
                if (cancel_event and cancel_event.is_set()) or time.monotonic()-started>300:
                    process.kill();process.wait();write_new(output/'interrupted.json',{'reason':'cancelled' if cancel_event and cancel_event.is_set() else 'timeout','durable_attempt_records':len(list((output/'attempts').glob('call_*.json'))),'completed_calls':'unconfirmed; partial evidence retained'});raise RuntimeError('phrase trial cancelled or timed out')
                time.sleep(.05)
        self._unchanged()
        if process.returncode or not (output/'result.json').is_file():raise RuntimeError('phrase worker failed; preserved '+str(output))
        result=json.loads((output/'result.json').read_text(encoding='utf8'))
        if result['request_sha256']!=request_hash or result['expression']!=p or result['status']!='completed' or result.get('integrity_passed') is not True:raise RuntimeError('expression completion binding/audit failed')
        trial={'id':identifier,'index':trial_index,'expression':p,'paths':{'directory':str(output),'result':str(output/'result.json'),'audit':str(output/'audit.json'),'common_reference':str(output/'common_reference')},'metrics':result['metrics'],'diagnostics':result['expression_diagnostics'],'integrity_passed':True}
        self._verify_trial(trial)
        return trial

    def _verify_trial(self,trial):
        try:return self._verify_trial_files(trial)
        except (OSError,ValueError,KeyError,TypeError) as error:raise RuntimeError('phrase saved integrity invalid') from error

    def _verify_trial_files(self,trial):
        directory=self._directory(trial['paths']['directory'])
        validate_trial_paths(trial,directory)
        for name in ('result.json','audit.json','request.json'):
            if (directory/name).stat().st_size>128*1024:raise RuntimeError('Expression proof metadata exceeds bound')
        for path in directory.rglob('*'):
            if path.is_symlink() or bool(getattr(path.lstat(),'st_file_attributes',0)&1024):raise RuntimeError('Linked expression saved proof forbidden')
        result=json.loads((directory/'result.json').read_text(encoding='utf8'));audit=json.loads((directory/'audit.json').read_text(encoding='utf8'))
        if (directory/'request.json').stat().st_size>128*1024:raise RuntimeError('Expression saved request exceeds bound')
        request=json.loads((directory/'request.json').read_text(encoding='utf8'))
        validate_trial_identity(trial,request,result)
        if request.get('schema')!='pulsemorph.expression_physical_request.v1' or parse_expression(request['expression'])!=parse_expression(trial['expression']) or request['expression']!=result['expression'] or request['id']!=result['id'] or request['index']!=result['index'] or request['job_id']!=result['job_id']:raise RuntimeError('Expression saved request/result identity mismatch')
        physical=trial.get('physical_metrics',trial['metrics'])
        if result.get('integrity_passed') is not True or result['status']!='completed' or audit.get('integrity_passed') is not True or sha(directory/'audit.json')!=result['audit_sha256'] or trial['expression']!=result['expression'] or physical!=result['metrics'] or audit['metrics']!=result['metrics']:raise RuntimeError('phrase saved audit/result binding failed')
        if result['schema']!='pulsemorph.expression_physical_trial.v1' or audit['schema']!='pulsemorph.expression_saved_audit.v1' or audit['fresh_mj_kinematics_calls']!=400 or result['counts']!={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1}:raise RuntimeError('phrase proof scope/counters differ')
        if any(result['source_pins'].get(p)!=h for p,h in self._implementation.items()) or sha(directory/'request.json')!=result['request_sha256']:raise RuntimeError('phrase implementation/request proof missing')
        expected=dict(self._implementation);music=self._music(request['music_observation']);expected[music['path']]=music['sha256']
        for name in ('compile_report.json','common_reference/manifest.json','common_reference/arrays.npz'):expected[str(directory/name)]=sha(directory/name)
        if request['source_pins']!=result['source_pins'] or result['source_pins']!=expected or request['environment']!=result['environment'] or request['environment']!=self.environment or request['music_observation']!=result['music_observation']:raise RuntimeError('Expression exact saved source/environment/music assignment mismatch')
        required={'qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy','policy_inputs.npz','policy_outputs.npz','policy_calls.json','substep_telemetry.npz','request.json','common_reference/manifest.json','common_reference/arrays.npz'}
        if not required.issubset(audit['inventory']):raise RuntimeError('phrase proof inventory incomplete')
        if len(audit['inventory'])>500 or {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}!=set(audit['inventory'])|{'result.json','audit.json','worker.log'}:raise RuntimeError('Expression raw inventory assignment mismatch')
        for relative,digest in audit['inventory'].items():
            path=(directory/relative).resolve()
            if not path.is_relative_to(directory) or sha(path)!=digest:raise RuntimeError('phrase saved trial was altered')
        for path,digest in result['source_pins'].items():
            if sha(path)!=digest:raise RuntimeError('phrase trial external input changed')
        ref=load_reference(directory/'common_reference');validate_expression_reference(ref,request['expression'])
        if ref.metadata['reference_id']!=request['id'] or ref.metadata['context']['task_id']!=request['job_id'] or ref.metadata['context']['runtime_session_id']!=request['id']:raise RuntimeError('Expression saved reference context mismatch')
        numerical_sha=sha(self.root/'src/pulsemorph/agentic/app/phrase/worker.py')
        if audit['numerical_audit_source_sha256']!=numerical_sha or result['audit_numerical_source_sha256']!=numerical_sha or trial['diagnostics']!=result['expression_diagnostics'] or result['expression_diagnostics']!=audit['expression_diagnostics']:raise RuntimeError('Expression audit/diagnostic source binding mismatch')
        return result

