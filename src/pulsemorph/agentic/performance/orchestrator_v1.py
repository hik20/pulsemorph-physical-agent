"""Music→LLM joint programs→copied physics feedback→LLM revision/commit."""
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .choreography_v1 import compile_program,ChoreographyError
from .provider_v1 import ChoreographyProvider,map_program
from .evaluation_v1 import evaluate_performance,GROUP_THRESHOLDS,HAND_THRESHOLD,PELVIS_THRESHOLD,MIN_ROOT_HEIGHT,MIN_ROOT_UP_Z
from ..embodied.runtime_v1 import canonical,fingerprint,ActionRejected
from ..embodied.fork_v1 import PreviewPhysicalFailure

def write(path,value):
    with Path(path).open('xb') as stream:stream.write(canonical(value))

def numerical_motion_hash(curve):
    """Motion identity excludes narrative, program IDs and validation reports."""
    h=hashlib.sha256(canonical({'duration_s':curve.duration_s}))
    for value in (curve.times,curve.q,curve.dq):
        a=np.asarray(value,dtype=np.float64);h.update(repr(a.shape).encode()+a.tobytes())
    return h.hexdigest()

def physical_flags(episode):
    b=episode.backend;obs=episode.observe()['physical_flags']
    force=all(r.get('joint_caps_observed_compliant') is True for r in b.hybrid_force_rows)
    return {'finite':bool(obs['finite']),'warnings_unchanged':bool(obs['warnings_unchanged']),'stopped':bool(episode.stopped or b.stopped),'force_caps_passed':force,'physical_gate_passed':bool(not episode.stopped and not b.stopped and obs['finite'] and obs['warnings_unchanged'] and force)}

def evaluate_episode(episode,music,start_s):
    b=episode.backend
    refs={'times':np.asarray([r['sim_time_s'] for r in episode.reference_samples]),'q':np.asarray([r['q'] for r in episode.reference_samples])}
    return evaluate_performance(np.asarray(b.states),np.asarray(b.velocities),np.asarray(b.times),episode.kinematics,b.joint_names,refs,music,start_s=start_s,physics_flags=physical_flags(episode))

class PerformanceOrchestrator:
    def __init__(self,provider,*,max_model_calls=8,min_previews=2,wall_seconds=1800):
        if type(max_model_calls) is not int or not 1<=max_model_calls<=10 or min_previews!=2 or not 0<wall_seconds<=2400:raise ValueError('Explicit finite reasoning budget required')
        self.provider=ChoreographyProvider(provider);self.max_model_calls=max_model_calls;self.min_previews=min_previews;self.wall_seconds=wall_seconds
    def run(self,pool,music,instruction,output,*,cancel_event=None,on_event=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=False);parent=pool.parent;start=time.monotonic();music_start=parent.backend.sim_time_s
        fixed={'schema':'pulsemorph.music_body_task.v1','instruction':instruction,'music':copy.deepcopy(music),'music_start_sim_time_s':music_start,'initial_observation_sha256':fingerprint(parent.observe()),'capabilities':parent.capabilities(),
            'screen':{'actual_group_min_span_rad':GROUP_THRESHOLDS,'each_hand_workspace_m':HAND_THRESHOLD,'pelvis_vertical_and_lateral_m':PELVIS_THRESHOLD,'min_root_height_m':MIN_ROOT_HEIGHT,'min_root_up_z':MIN_ROOT_UP_Z,'continuous_hold_or_aesthetic_goal':False},
            'budget':{'model_calls':self.max_model_calls,'min_previews_before_commit':self.min_previews,'max_physics_previews':pool.max_previews,'max_sandbox_controls':pool.max_total_preview_controls}}
        write(output/'fixed.json',fixed);fixed_sha=hashlib.sha256((output/'fixed.json').read_bytes()).hexdigest();candidates=[];history=[];model_attempts=0;model_receipts=[];commit=None;reason='reasoning_budget_exhausted';status='stopped'
        def emit(kind,message,**values):
            if on_event:on_event(kind,message,values)
        try:
            for index in range(self.max_model_calls):
                if cancel_event and cancel_event.is_set():status='cancelled';reason='cancelled';break
                if time.monotonic()-start>self.wall_seconds:reason='wall_budget_exhausted';break
                eligible=[c['preview_id'] for c in candidates if c.get('assessment',{}).get('performance_screen_passed') is True] if len([c for c in candidates if c.get('status')=='ready'])>=self.min_previews else []
                context={'instruction':instruction,'joint_order':parent.backend.joint_names,'initial_reference_q':parent.reference_q.tolist(),'actual_body':parent.observe(),
                    'music':{k:v for k,v in music.items() if k not in ('path','waveform128')},'fixed_requirements':fixed['screen'],'capabilities':fixed['capabilities'],
                    'eligible_preview_ids':eligible,'physics_previews_used':len(pool.entries),'reasoning_calls_remaining':self.max_model_calls-index,'prior_decisions':history,
                    'candidate_feedback':[{'preview_id':c.get('preview_id'),'status':c['status'],'program_sha256':c.get('program_sha256'),'feedback':c.get('assessment',{}).get('feedback'),'music_alignment':c.get('assessment',{}).get('music_alignment'),'physical_failure':c.get('physical_failure'),
                         'compile_rejection':None if c['status']!='compile_rejected' else {'message':c['message'],'violations':[r for r in c['report'].get('joint_ranges',[]) if not r['admitted']],'activation':c['report'].get('activation')}} for c in candidates]}
                call=output/('model_%02d'%index);call.mkdir();write(call/'context.json',context);write(call/'attempt.json',{'model_requested':self.provider.provider.model,'completion_unconfirmed':True});model_attempts+=1;emit('model_thinking','LLM müzik ve beden sonucuna göre hareket programını yazıyor.')
                result,pins=self.provider.propose(context,timeout=min(300,max(1,self.wall_seconds-(time.monotonic()-start))),cancel_event=cancel_event)
                if result.receipt.get('live') is not True or type(result.receipt.get('tool_events')) is not int or result.receipt['tool_events']!=0 or result.receipt.get('model_requested')!='gpt-6.1-sol' or result.receipt.get('model_returned') not in (None,'gpt-6.1-sol'):raise RuntimeError('Real tool-free requested model receipt required')
                data=result.data;write(call/'raw_response.json',data);receipt={**result.receipt,**pins,'completion_unconfirmed':False,'response_sha256':fingerprint(data)};write(call/'receipt.json',receipt);model_receipts.append(receipt)
                history.append({'decision':data['decision'],'summary':data['summary'],'reason':data['reason']});emit('model_decision',data['summary'],decision=data['decision'],reason=data['reason'])
                if data['decision']=='stop':
                    if data['program'] is not None or data['preview_id']:raise ValueError('Exact stop transport required')
                    reason=data['reason'];break
                if data['decision']=='commit':
                    if data['program'] is not None or data['preview_id'] not in eligible:raise ValueError('Only explicitly eligible preview may commit')
                    request={'preview_id':data['preview_id'],'expected_revision':parent.revision,'observation_sha256':fingerprint(parent.observe()),'commit_id':'llm_commit_%02d'%index};write(call/'commit_pending.json',request);emit('committing','LLM seçtiği hareketi aynı gerçek simülasyon bedenine uyguluyor.')
                    commit=pool.commit(request);write(call/'commit_returned.json',commit);assessment=evaluate_episode(parent,music,music_start);write(output/'actual_assessment.json',assessment);status='completed' if assessment['performance_screen_passed'] is True else 'stopped';reason='actual_screen_passed' if status=='completed' else 'actual_screen_failed';emit('actual_completed','Gerçek performans ölçüldü; önizleme ile tam durum eşliği doğrulandı.',assessment=assessment);break
                if data['decision']!='preview' or data['preview_id'] or data['program'] is None:raise ValueError('Exact preview transport required')
                mapped=map_program(data['program'],parent.backend.joint_names);write(call/'mapped_program.json',mapped)
                if mapped['music_sha256']!=music['sha256']:raise ValueError('Music identity changed')
                try:
                    curve=compile_program(mapped,parent.backend.joint_names,parent.backend.joint_ranges,parent.reference_q,parent.reference_dq)
                    motion_sha=numerical_motion_hash(curve)
                    if any(c.get('motion_sha256')==motion_sha for c in candidates):raise ChoreographyError('A revision must change numerical motion, not only rationale',{'category':'identical_numeric_revision'})
                    write(call/'compile_report.json',curve.report)
                except ChoreographyError as error:
                    rejection={'status':'compile_rejected','program_sha256':fingerprint(mapped),'message':str(error),'report':error.report,'physics_steps':0};write(call/'compile_rejected.json',rejection);candidates.append(rejection);emit('compile_rejected','Hareket sınırları nedeniyle program reddedildi; LLM düzeltecek.',compile_report=error.report);continue
                request={'action_id':'llm_program_%02d'%index,'expected_revision':parent.revision,'observation_sha256':fingerprint(parent.observe()),'program_sha256':curve.program_sha256};write(call/'preview_pending.json',request);emit('preview_started','Tüm beden hareketi aynı durumun sanal kopyasında deneniyor.')
                try:preview=pool.preview(request,curve)
                except PreviewPhysicalFailure as error:
                    row={'status':'physical_failed','preview_id':error.report['preview_id'],'program_sha256':curve.program_sha256,'motion_sha256':motion_sha,'physical_failure':error.report};candidates.append(row);write(call/'preview_failure.json',error.report);emit('preview_failed','Sanal performans fizik sınırında durdu; gerçek beden ilerlemedi.',preview=row);continue
                assessment=evaluate_episode(pool.entries[preview['preview_id']]['branch'],music,music_start);write(call/'preview_returned.json',preview);write(call/'preview_assessment.json',assessment)
                row={'status':'ready','preview_id':preview['preview_id'],'program_sha256':curve.program_sha256,'motion_sha256':motion_sha,'assessment':assessment};candidates.append(row);emit('preview_completed','Sanal performans ölçüldü; LLM fizik geri bildirimini inceleyecek.',preview=row)
            if hashlib.sha256((output/'fixed.json').read_bytes()).hexdigest()!=fixed_sha:raise RuntimeError('Fixed task changed')
        except Exception as error:
            status='cancelled' if cancel_event and cancel_event.is_set() else 'failed';reason=type(error).__name__;write(output/'failure.json',{'category':reason,'message':str(error),'model_attempt_completion_unconfirmed':model_attempts-len(model_receipts),'retry_allowed':False})
        assessment=evaluate_episode(parent,music,music_start)
        summary={'schema':'pulsemorph.music_body_run.v1','status':status,'reason':reason,'performance_screen_passed':assessment['performance_screen_passed'] is True,'live_model_calls':len(model_receipts),'model_attempts':model_attempts,'model_completion_unconfirmed':model_attempts-len(model_receipts),'physical_preview_attempts':len(pool.entries),'actual_commits':len(pool.commits),'actual_assessment':assessment,'commit':commit,'wall_seconds':time.monotonic()-start,'fixed_sha256':fixed_sha,'training_admission':'NOT_ADMITTED'}
        write(output/'result.json',summary);return summary
