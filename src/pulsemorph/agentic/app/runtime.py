"""Bounded app trials using a pinned pretrained controller and real MuJoCo steps."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import time
import numpy as np

PINS = {'wave':'cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93',
        'high5':'ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44'}
PARAMETERS = ['left_gain_a','right_gain_a','left_gain_b','right_gain_b','timing_shift_s']
TOOLS = ('global_clock_v1','local_clock_1p4_2p8_v1')

def _module(root, name):
    spec=importlib.util.spec_from_file_location('_app_'+name,root/'experiments/agentic'/f'{name}.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def _audit_app(module,original,directory):
    directory=Path(directory).resolve()
    result=original(directory)
    result['fresh_anchor_audit']=module._audit_fresh_anchor(directory,result)
    from pulsemorph.motion.contracts import load_reference
    from pulsemorph.motion.g1 import assemble_reference_inputs
    common=load_reference(directory/'common_reference')
    feeds=dict(np.load(directory/'policy_inputs.npz',allow_pickle=False))
    for index in range(200):
        expected=assemble_reference_inputs(common,index*.02)
        for key in ('mimic_future_dof_pos','mimic_future_dof_vel'):
            if not np.allclose(expected[key],feeds[f'call_{index+1:04d}__'+key],rtol=0,atol=2e-6):
                raise ValueError('strict common/actual feed mismatch: '+key)
    result['strict_common_control_rows']=200
    return result


def evaluate_trial(directory, baseline_dir, targets, tolerances):
    root=Path(__file__).resolve().parents[4]
    module=_module(root,'evaluate_multifeature')
    protocol={'parameter_names':PARAMETERS,'bounds':[[.35,1.6]]*4+[[-.3,.3]],
              'feature_tolerances':tolerances,'feature_windows_s':[[.8,1.6],[2.4,3.2]],
              'accent_window_s':[1.8,3.2],'accent_minimum_prominence':.15,
              'physical_gate':{'min_root_height_m':.6,'max_mean_joint_rmse_rad':.35}}
    original_audit=module._audit
    module._audit=lambda directory: _audit_app(module,original_audit,directory)
    result=module.evaluate_multifeature(directory,baseline_dir,protocol,{'feature_targets':targets})
    if isinstance(result.get('accent'),dict):
        for entry in (result['accent'],result['accent'].get('reference',{})):
            entry.pop('rms_speed_rad_s',None);entry.pop('smoothed_rms_speed_rad_s',None)
    result['audit_scope']='Full artifact/controller/physics audit, independent same-time torso FK, strict common joint future feeds at all 200 control rows; generic audit also checks heading-composed torso feeds.'
    result.update(targets=targets,tolerances=tolerances,
                  score=result.get('utility'),schema='pulsemorph.agentic_app_evaluation.v1')
    return result

class AppRuntime:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.python=self.root/'.venv-motor/Scripts/python.exe'

    def sources(self):
        return [{'id':key,'label':{'wave':'Wave','high5':'High five'}[key],
                 'description':'Four second pinned G1 source; pretrained tracker in MuJoCo',
                 'sha256':digest,'duration_s':4} for key,digest in PINS.items()]

    evaluate_trial=staticmethod(evaluate_trial)

    def run_trial(self,source_id,parameters,tool_id,output,job_id,trial_index,reason,cancel_event=None,music_observation=None):
        if source_id not in PINS or tool_id not in TOOLS:
            raise ValueError('unsupported app source/tool')
        if set(parameters)!=set(PARAMETERS):
            raise ValueError('exactly five parameters required')
        for key,value in parameters.items():
            low,high=(-.3,.3) if key=='timing_shift_s' else (.35,1.6)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high:
                raise ValueError('parameter outside app domain: '+key)
        output=Path(output).resolve(); allowed=(self.root/'artifacts/runs/agentic').resolve()
        if allowed not in output.parents or output.exists():
            raise ValueError('new output strictly under agentic runs required')
        if cancel_event and cancel_event.is_set(): raise RuntimeError('trial cancelled')
        if shutil.disk_usage(self.root).free < 256*1024*1024:
            raise RuntimeError('at least 256MB free disk required for physical trial')
        output.parent.mkdir(parents=True,exist_ok=True)
        audio={'available':False,'duration_s':0.,'observed_through_sample':None}
        if music_observation is not None:
            import wave
            path=Path(music_observation['path']).resolve()
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with wave.open(str(path),'rb') as wav:
                rate,count=wav.getframerate(),wav.getnframes()
            if digest!=music_observation['sha256'] or count/rate!=4. or rate!=music_observation['sample_rate']:
                raise ValueError('music observation does not match actual four-second PCM')
            audio={'available':True,'path':str(path),'sha256':digest,'sample_rate':rate,
                   'duration_s':count/rate,'observed_through_sample':count}
        candidate=f'{job_id}__trial_{trial_index}'
        proposal={'schema':'pulsemorph.agentic_app_proposal.v1','candidate_id':candidate,
            'job_id':str(job_id),'trial_index':trial_index,'reason':str(reason),'source_id':source_id,
            'control_tool':tool_id,'parameters':dict(parameters),'duration_s':4,'source_start_s':0,
            'observation_id':str(job_id)+(':offline-audio' if audio['available'] else ':no_audio'),'task_id':str(job_id),
            'runtime_session_id':candidate+':physical-runtime',
            'audio':audio,
            'source_motion':{'path':f'research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_{source_id}.motion','sha256':PINS[source_id]}}
        proposal_path=output.parent/(output.name+'.proposal.json')
        with proposal_path.open('x',encoding='utf-8') as f: json.dump(proposal,f,indent=2)
        command=[str(self.python),str(self.root/'experiments/agentic/app/execute_app.py'),
                 '--proposal',str(proposal_path),'--output',str(output)]
        log_path=output.parent/(output.name+'.runtime.log')
        with log_path.open('xb') as log:
            proc=subprocess.Popen(command,cwd=self.root,stdout=log,stderr=log,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            started=time.monotonic()
            while proc.poll() is None:
                if (cancel_event and cancel_event.is_set()) or time.monotonic()-started>180:
                    if os.name=='nt': subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    else: proc.kill()
                    proc.wait(); raise RuntimeError('trial cancelled or timed out; partial evidence retained')
                time.sleep(.1)
        if proc.returncode: raise RuntimeError('physical trial failed; inspect '+str(output))
        manifest=json.loads((output/'manifest.json').read_text())
        audit_module=_module(self.root,'evaluate_multifeature')
        audit=_audit_app(audit_module,audit_module._audit,output)
        groups=[[i for i,n in enumerate(audit['names']) if n.startswith(side+'_') and any(x in n for x in ('shoulder_','elbow_','wrist_'))] for side in ('left','right')]
        q=audit['arrays']['qpos'];times=audit['arrays']['state_times'];rms=[]
        for start,end in ((.8,1.6),(2.4,3.2)):
            for group in groups:
                mask=(times>=start-1e-10)&(times<=end+1e-10)
                rms.append(float(np.sqrt(np.mean((q[mask,7:][:,group]-q[0,7:][group])**2))))
        accent=_module(self.root,'evaluate_multifeature').accent_metrics(audit['arrays']['qvel'][:,6:],times,[1.8,3.2])
        metrics={'arm_rms_rad':rms,'accent':{k:v for k,v in accent.items() if k not in ('rms_speed_rad_s','smoothed_rms_speed_rad_s')},'min_root_height_m':manifest['all_substeps_min_root_height_m'],
                 'mean_joint_rmse_rad':audit['mean_joint_rmse_rad'],'actual_physics_steps':manifest['actual_physics_steps'],
                 'online_onnx_call_count':manifest['online_onnx_call_count']}
        return {'id':candidate,'index':trial_index,'tool_id':tool_id,'parameters':dict(parameters),
                'paths':{'directory':str(output),'manifest':str(output/'manifest.json')},'metrics':metrics}
