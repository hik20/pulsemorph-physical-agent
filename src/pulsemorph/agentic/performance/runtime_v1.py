"""Persistent music-time programs, full-state previews and explicit commits.

The token model authors the complete joint program. This module interpolates,
executes and records it; it neither invents choreography nor selects candidates.
"""
import copy
import hashlib
import re
import threading
from dataclasses import asdict
from pathlib import Path
import numpy as np
from pulsemorph.agentic.embodied.runtime_v1 import PersistentEpisode,ActionRejected,ObservationMismatch,canonical,fingerprint
from pulsemorph.agentic.embodied.fork_v1 import fork_episode,snapshot_episode,_origin,_suffix,PreviewIntegrityError,PreviewPhysicalFailure

BODY_NAMES=('pelvis','torso_link','left_rubber_hand','right_rubber_hand','left_ankle_roll_link','right_ankle_roll_link')

def curve_hash(curve):
    h=hashlib.sha256(canonical({'duration_s':curve.duration_s,'program_sha256':curve.program_sha256,'report':curve.report}))
    for value in (curve.times,curve.q,curve.dq):
        a=np.asarray(value);h.update(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes())
    return h.hexdigest()

class HoldCurve:
    def __init__(self,q,duration_s):
        self.duration_s=float(duration_s);self.times=np.arange(round(duration_s/.02)+9)*.02
        self.q=np.tile(np.asarray(q,dtype=float),(len(self.times),1));self.dq=np.zeros_like(self.q)
        self.report={'admitted':True,'numeric_validity_passed':True,'authorship':'root preparation hold; not choreography'}
        self.program_sha256=fingerprint({'hold_q':self.q[0].tolist(),'duration_s':self.duration_s})
    def sample(self,t):return self.q[0].copy(),np.zeros(29)

def make_reference(episode,curve,action_id,start_time,observation_sha):
    from pulsemorph.motion.contracts import MotionReference,SCHEMA_VERSION
    b=episode.backend;p=b.profile
    m={'schema_version':SCHEMA_VERSION,'reference_id':episode.session_id+':'+action_id,
       'producer':{'kind':'agentic','id':'llm-music-joint-program','version':'1'},
       'body':{'profile_id':'g1','profile_sha256':p['body_sha256'],'joint_names':list(b.joint_names),'anchor_body_name':'torso_link'},
       'controller':{'profile_id':'g1-bones-deploy','profile_sha256':p['yaml_sha256'],'control_dt_s':.02},
       'frame':{'id':'episode_world','up_axis':'z','handedness':'right','quaternion_order':'xyzw','alignment':'recorded_reference_world'},
       'context':{'decision_id':action_id,'observation_id':observation_sha,'task_id':episode.session_id,'runtime_session_id':episode.session_id,'clock_epoch':episode.session_id,'decision_time_s':start_time},
       'validity':{'start_state_id':'full-executor-closure:'+b.closure_sha256(),'start_time_s':start_time,'end_time_s':start_time+float(curve.times[-1]),'start_mode':'continue_same_MjData_policy_history_reference_clock','end_behavior':'terminal_reference_hold'},
       'observation':{'mode':'live_causal','observed_through_time_s':start_time,'audio_observed_through_sample':episode.music.get('sample_frames') if episode.music else None,'video_observed_through_time_s':None},
       'provenance':{'sources':[{'path':path,'sha256':sha} for path,sha in b.source_pins.items()],
          'notes':['Complete music file is available for offline choreography planning; body observation stops at decision time.',
                   'Program SHA256 '+curve.program_sha256+'; immutable compiled curve SHA256 '+curve_hash(curve),
                   'All29 q/dq references authored by the program; deterministic interpolation is not a learned dance generator.',
                   'Fixed initial torso orientation is a declared tracker carrier. No planned body FK or root/contact path is fabricated.',
                   'Physical realized states are separate from references and from hypothetical copied branches.']}}
    return MotionReference(m,curve.times,curve.q,curve.dq,np.tile(b.anchor_target,(len(curve.times),1)))

class PerformanceEpisode(PersistentEpisode):
    def __init__(self,backend,session_id,*,music=None,cancel_event=None):
        # Reuse the frozen observation schema, not its short-action budget/parser.
        cap=backend.authority['max_control_steps']
        if backend.authority['schema']!='pulsemorph.performance_execution_authority.v1' or not 1<=cap<=1200 or backend.authority['session_id']!=session_id:raise ValueError('Matching explicit performance authority required')
        if re.fullmatch('[A-Za-z0-9_-]{1,100}',session_id) is None:raise ValueError('Bounded session id required')
        self.backend=backend;self.session_id=session_id;self.max_control_steps=cap;self.music=copy.deepcopy(music or {});self.cancel_event=cancel_event
        self.revision=0;self.control_index=0;self.stopped=False;self._lock=threading.RLock();self._receipts={};self.actions=[];self.references=[];self.common_references=[]
        self.reference_q=np.asarray(backend.observe()['qpos'][7:],dtype=float);self.reference_dq=np.zeros(29)
        self._observation=self._make_observation(backend.observe());self.kinematics=[];self.reference_samples=[{'sim_time_s':backend.sim_time_s,'q':self.reference_q.tolist(),'dq':self.reference_dq.tolist()}]
        self._body_ids={n:backend.mj.mj_name2id(backend.model,backend.mj.mjtObj.mjOBJ_BODY,n) for n in BODY_NAMES}
        if min(self._body_ids.values())<0:raise RuntimeError('Required actual body names unavailable')
        self._capture_kinematics()
    def capabilities(self):
        return {'schema':'pulsemorph.performance_capabilities.v1','joint_names':list(self.backend.joint_names),'joint_ranges_rad':[list(x) for x in self.backend.joint_ranges],
                'max_control_steps':self.max_control_steps,'control_step_s':.02,'physics_step_s':.001,'future_offsets_s':[.02,.04,.08,.16],
                'controller_mode':self.backend.performance_contract,'source_choreography_playback':False,'simulation_pauses_between_decisions':True}
    def _capture_kinematics(self):
        b=self.backend;d=b.data;m=b.model
        q=np.asarray(d.qpos[3:7]);w,x,y,z=q;up=1-2*(x*x+y*y)
        contacts={'left':False,'right':False}
        for i in range(d.ncon):
            c=d.contact[i];one,two=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            for side in contacts:
                foot=self._body_ids[side+'_ankle_roll_link']
                if (one==foot and two==0) or (two==foot and one==0):contacts[side]=True
        self.kinematics.append({'sim_time_s':b.sim_time_s,'root_height_m':float(d.qpos[2]),'root_up_z':float(up),'com_world_m':d.subtree_com[0].tolist(),
            'positions_world_m':{n:d.xpos[index].tolist() for n,index in self._body_ids.items()},'foot_contacts':contacts,'ncon':int(d.ncon),
            'source':'read already-fresh derived state; zero additional FK'})
    def _admit(self,request,curve):
        if type(request) is not dict or set(request)!={'action_id','expected_revision','observation_sha256','program_sha256'}:raise ActionRejected('Exact performance envelope required')
        if type(request['action_id']) is not str or re.fullmatch('[A-Za-z0-9_-]{1,100}',request['action_id']) is None:raise ActionRejected('Bounded action identity required')
        if type(request['expected_revision']) is not int or request['expected_revision']!=self.revision or request['observation_sha256']!=fingerprint(self.observe()):raise ObservationMismatch('Stale actual body binding')
        if request['program_sha256']!=curve.program_sha256 or curve.report.get('admitted') is not True:raise ActionRejected('Admitted exact authored program required')
        duration=float(curve.duration_s);count=round(duration/.02)
        if not 0<duration<=20 or abs(count*.02-duration)>1e-9 or self.control_index+count>self.max_control_steps:raise ActionRejected('Physical program budget exceeded')
        q,dq=curve.sample(0.)
        if not np.array_equal(q,self.reference_q) or not np.array_equal(dq,self.reference_dq):raise ActionRejected('Reference continuity changed; no overwritten first pose')
        return count
    def execute_curve(self,request,curve):
        with self._lock:
            identifier=request.get('action_id') if isinstance(request,dict) else None
            if identifier in self._receipts:
                old=self._receipts[identifier]
                if old['request_sha256']!=fingerprint(request) or old['curve_sha256']!=curve_hash(curve):raise ObservationMismatch('Changed duplicate program')
                return {**copy.deepcopy(old),'replay_no_new_work':True}
            if self.stopped:raise RuntimeError('Stopped body; no automatic reset/recovery')
            count=self._admit(request,curve);b=self.backend;b._verify();frozen_curve=curve_hash(curve);start=b.sim_time_s;before=fingerprint(self.observe());revision=self.revision
            ref=make_reference(self,curve,request['action_id'],start,before);self.common_references.append(ref)
            action={'request':copy.deepcopy(request),'curve_sha256':frozen_curve,'start_sim_time_s':start,'planned_control_steps':count,'status':'attempted'};self.actions.append(action)
            from pulsemorph.motion.g1 import sample_reference
            try:
                for i in range(count):
                    if self.cancel_event is not None and self.cancel_event.is_set():raise RuntimeError('PERFORMANCE_CANCELLED')
                    sample=sample_reference(ref,np.asarray([(i+o)*.02 for o in (1,2,4,8)]));b.advance(sample['dof_pos'],sample['dof_vel']);self.control_index+=1
                    self.reference_q,self.reference_dq=curve.sample((i+1)*.02)
                    self.references.append({'control_index':self.control_index-1,'time_s':start+i*.02,'future_q':sample['dof_pos'].tolist(),'future_dq':sample['dof_vel'].tolist()})
                    self.reference_samples.append({'sim_time_s':b.sim_time_s,'q':self.reference_q.tolist(),'dq':self.reference_dq.tolist()});self._capture_kinematics()
                if curve_hash(curve)!=frozen_curve:raise RuntimeError('Compiled program changed during execution')
                b._verify();self.revision+=1;self._observation=self._make_observation(b.observe());action['status']='completed'
                receipt={'schema':'pulsemorph.performance_transition.v1','action_id':request['action_id'],'request_sha256':fingerprint(request),'curve_sha256':frozen_curve,
                    'before_revision':revision,'after_revision':self.revision,'before_observation_sha256':before,'control_steps':count,'steps':count*20,'observation':self.observe(),'replay_no_new_work':False}
                self._receipts[identifier]=copy.deepcopy(receipt);return receipt
            except Exception:
                self.stopped=True;action['status']='failed_partial';self._observation=self._make_observation(b.observe());raise
    def export(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        self.backend.export(directory)
        (directory/'episode.json').write_bytes(canonical({'schema':'pulsemorph.performance_episode.v1','session_id':self.session_id,'revision':self.revision,'control_steps':self.control_index,'stopped':self.stopped,'actions':self.actions,'final_observation':self.observe(),'capabilities':self.capabilities()}))
        (directory/'kinematics.json').write_bytes(canonical(self.kinematics));(directory/'references.json').write_bytes(canonical(self.references))
        np.savez_compressed(directory/'reference_samples.npz',times=np.asarray([r['sim_time_s'] for r in self.reference_samples]),q=np.asarray([r['q'] for r in self.reference_samples]),dq=np.asarray([r['dq'] for r in self.reference_samples]))
        from pulsemorph.motion.contracts import save_reference
        for i,ref in enumerate(self.common_references):save_reference(ref,directory/('reference_%02d'%i))

def full_snapshot(parent):
    from pulsemorph.motion.contracts import _plain
    common=[]
    for ref in parent.common_references:
        h=hashlib.sha256(canonical(_plain(ref.metadata)))
        for value in (ref.time_offsets_s,ref.joint_position_rad,ref.joint_velocity_rad_s,ref.anchor_orientation_xyzw):
            a=np.asarray(value);h.update(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes())
        common.append(h.hexdigest())
    return {'native_and_executor':asdict(snapshot_episode(parent)),
        'auxiliary_sha256':fingerprint({'actions':parent.actions,'references':parent.references,'common_references':common,'kinematics':parent.kinematics,'reference_samples':parent.reference_samples,'receipts':parent._receipts,'music':parent.music})}

class PerformancePreviews:
    def __init__(self,parent,*,max_previews=3,max_total_preview_controls=2520):
        if type(max_previews) is not int or not 1<=max_previews<=4 or type(max_total_preview_controls) is not int or not 1<=max_total_preview_controls<=4000:raise ValueError('Finite independent preview budgets required')
        self.parent=parent;self.max_previews=max_previews;self.max_total_preview_controls=max_total_preview_controls;self.reserved_controls=0;self.entries={};self.commits={};self.stopped=False
    def preview(self,request,curve):
        if self.stopped:raise RuntimeError('Preview pool stopped')
        with self.parent._lock:
            count=self.parent._admit(request,curve)
            if len(self.entries)>=self.max_previews or self.reserved_controls+count>self.max_total_preview_controls:raise ActionRejected('Sandbox budget exhausted')
            self.reserved_controls+=count;identifier='preview_%03d'%len(self.entries);before=full_snapshot(self.parent);origin=_origin(self.parent.backend);branch=fork_episode(self.parent,identifier);frozen=copy.deepcopy(curve)
            entry={'branch':branch,'curve':frozen,'curve_sha256':curve_hash(frozen),'request':copy.deepcopy(request),'before':before,'origin':origin,'status':'attempted'};self.entries[identifier]=entry
            try:transition=branch.execute_curve(request,frozen)
            except Exception as error:
                if full_snapshot(self.parent)!=before:self.stopped=True;raise PreviewIntegrityError('Failed preview mutated actual body')
                suffix=_suffix(branch.backend,origin);entry.update(status='failed',report={'preview_id':identifier,'hypothetical_only':True,'parent_unchanged':True,'suffix':suffix,'error_type':type(error).__name__,'physical_flags':branch.observe()['physical_flags']})
                if isinstance(error,RuntimeError) and str(error)=='Actual physical finite/rootheight/warning stop':raise PreviewPhysicalFailure(entry['report']) from None
                self.stopped=True;raise
            if full_snapshot(self.parent)!=before:self.stopped=True;raise PreviewIntegrityError('Preview mutated actual body')
            suffix=_suffix(branch.backend,origin)
            if suffix['new_initializations'] or suffix['new_warmup_calls']:self.stopped=True;raise PreviewIntegrityError('Preview reinitialized body/actor')
            receipt={'schema':'pulsemorph.performance_preview.v1','preview_id':identifier,'hypothetical_only':True,'parent_unchanged':True,'program_sha256':curve.program_sha256,'curve_sha256':entry['curve_sha256'],'suffix':suffix,'hypothetical_observation':transition['observation']}
            entry.update(status='ready',receipt=receipt,after=full_snapshot(branch));return copy.deepcopy(receipt)
    def commit(self,value):
        required={'preview_id','expected_revision','observation_sha256','commit_id'}
        if self.stopped or type(value) is not dict or set(value)!=required:raise PreviewIntegrityError('Exact explicit commit required')
        if type(value['expected_revision']) is not int or type(value['commit_id']) is not str or re.fullmatch('[A-Za-z0-9_-]{1,100}',value['commit_id']) is None:raise PreviewIntegrityError('Typed revision and bounded commit identity required')
        if value['commit_id'] in self.commits:
            old=self.commits[value['commit_id']]
            if old['request_sha256']!=fingerprint(value):raise PreviewIntegrityError('Changed duplicate commit')
            return {**copy.deepcopy(old),'replay_no_new_work':True}
        with self.parent._lock:
            entry=self.entries.get(value['preview_id'])
            if entry is None or entry['status']!='ready' or full_snapshot(self.parent)!=entry['before']:raise PreviewIntegrityError('Unknown/stale actual preview binding')
            if value['expected_revision']!=self.parent.revision or value['observation_sha256']!=fingerprint(self.parent.observe()) or curve_hash(entry['curve'])!=entry['curve_sha256']:raise PreviewIntegrityError('Commit observation/program binding changed')
            if full_snapshot(entry['branch'])!=entry['after'] or _suffix(entry['branch'].backend,entry['origin'])!=entry['receipt']['suffix']:raise PreviewIntegrityError('Stored hypothetical state/trace changed before commit')
            try:
                result=self.parent.execute_curve(entry['request'],entry['curve']);actual=_suffix(self.parent.backend,entry['origin'])
                if actual!=entry['receipt']['suffix'] or full_snapshot(self.parent)!=entry['after']:raise PreviewIntegrityError('Full actual/preview numeric parity failed')
            except Exception:self.stopped=True;self.parent.stopped=True;raise
            entry['status']='committed';receipt={**result,'schema':'pulsemorph.performance_commit.v1','preview_id':value['preview_id'],'commit_id':value['commit_id'],'request_sha256':fingerprint(value),'hypothetical_only':False,'preview_actual_byte_equal':True,'actual_suffix':actual};self.commits[value['commit_id']]=copy.deepcopy(receipt);return receipt
    def export(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        for identifier,entry in self.entries.items():
            p=directory/identifier;p.mkdir();(p/'preview.json').write_bytes(canonical(entry.get('receipt',entry.get('report',{}))));entry['branch'].export(p/'hypothetical_trace')
        (directory/'commits.json').write_bytes(canonical(self.commits))
