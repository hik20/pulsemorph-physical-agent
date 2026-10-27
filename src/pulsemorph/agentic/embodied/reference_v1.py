"""Live-causal chunk targets, distinct from realized state and future observations."""
import hashlib
import numpy as np
from pulsemorph.motion.contracts import MotionReference,SCHEMA_VERSION
from pulsemorph.motion.g1 import assemble_reference_inputs

def make_reference(backend,segment,session_id,action_id,revision,start_time,observation_sha256,q,dq):
 grid=np.arange(round(segment.duration/.02)+9)*.02
 values=[segment.sample(t) for t in grid];profile=backend.profile
 closure=hashlib.sha256((backend.closure_sha256()+observation_sha256).encode()+np.asarray(q).tobytes()+np.asarray(dq).tobytes()).hexdigest()
 metadata={'schema_version':SCHEMA_VERSION,'reference_id':session_id+':'+action_id,'producer':{'kind':'agentic','id':'agentic-embodied-joint-tool','version':'1'},'body':{'profile_id':'g1','profile_sha256':profile['body_sha256'],'joint_names':list(backend.joint_names),'anchor_body_name':'torso_link'},'controller':{'profile_id':'g1-bones-deploy','profile_sha256':profile['yaml_sha256'],'control_dt_s':.02},'frame':{'id':'episode_world','up_axis':'z','handedness':'right','quaternion_order':'xyzw','alignment':'recorded_reference_world'},'context':{'decision_id':action_id,'observation_id':observation_sha256,'task_id':session_id,'runtime_session_id':session_id,'clock_epoch':session_id,'decision_time_s':start_time},'validity':{'start_state_id':'full-executor-closure:'+closure,'start_time_s':start_time,'end_time_s':start_time+float(grid[-1]),'start_mode':'continue_same_MjData_policy_history_reference_clock','end_behavior':'terminal_reference_hold_prediction_may_be_revised_by_next_action'},'observation':{'mode':'live_causal','observed_through_time_s':start_time,'audio_observed_through_sample':None,'video_observed_through_time_s':None},'provenance':{'sources':[{'path':p,'sha256':h} for p,h in backend.source_pins.items()],'notes':['LLM authored named absolute joint target; no source choreography after initialization.','Reference q/dq targets are not realized body states or a balance guarantee.','Episode reference world is the recorded initial simulation world; initial torso orientation is held as the tracker anchor request; no target body FK is fabricated.','Full executor closure hash binds actual MuJoCo integration state plus policy history/alignment and reference state; export is not an implemented resume API.','160ms future reference is a prediction, not future observation or irrevocable next-action commitment.']}}
 ref=MotionReference(metadata,grid,np.asarray([v[0] for v in values]),np.asarray([v[1] for v in values]),np.tile(backend.anchor_target,(len(grid),1)))
 for i in range(round(segment.duration/.02)):assemble_reference_inputs(ref,i*.02)
 return ref
