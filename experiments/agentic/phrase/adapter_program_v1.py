"""Pinned phrase adapter and exact eighteen-control row gates; no physical/model operations."""
import numpy as np
from pulsemorph.motion.g1 import sample_reference

INPUTS=('current_anchor_rot','current_dof_pos','current_dof_vel','current_root_local_ang_vel','historical_processed_actions','mimic_future_anchor_rot','mimic_future_dof_pos','mimic_future_dof_vel')
OUTPUTS=('actions','joint_pos_targets','stiffness_targets','damping_targets')


class PhrasePlayer:
    """Only source frame0 is a full carrier; future body_rot is auxiliary layout.

    Non-torso future rotations repeat authentic first-source frame0 solely to satisfy
    the unmodified legacy 33-body layout. They are derived static auxiliaries,
    not composed body rotations/FK/contact/root targets, and are never exported
    as motion evidence. Exact-eight-feed guard prevents their consumption.
    """
    total_frames=400
    control_dt=.02

    def __init__(self,reference,initial_carrier,anchor_index):
        reference.validate()
        if reference.time_offsets_s[-1]!=8.14 or reference.metadata['controller']['control_dt_s']!=.02 or reference.metadata['producer']['id']!='phrase-program-reference' or reference.metadata['producer']['version']!='1':
            raise ValueError('exact8.14s G1 phrase required')
        self.reference=reference
        self.initial={k:np.asarray(v).copy() for k,v in initial_carrier.items()}
        self.anchor=anchor_index
        if self.initial['body_pos'].shape!=(33,3) or self.initial['body_rot'].shape!=(33,4) or not all(np.isfinite(v).all() for v in self.initial.values()):
            raise ValueError('actual Wave initial body carrier required')
        first=sample_reference(reference,np.array([0.]))
        if not np.array_equal(first['dof_pos'][0].astype('float32'),self.initial['dof_pos']) or not np.array_equal(first['anchor_rot'][0].astype('float32'),self.initial['body_rot'][anchor_index]):
            raise ValueError('phrase initial q/torso differs from authentic first-source carrier')

    def get_state_at_frame(self,index):
        if type(index) is not int or not 0<=index<=400:raise ValueError('frame outside committed0..400')
        if index==0:return {k:v.copy() for k,v in self.initial.items()}
        q=sample_reference(self.reference,np.array([index*.02]))
        return {'dof_pos':q['dof_pos'][0].astype('float32'),'dof_vel':q['dof_vel'][0].astype('float32')}

    def get_future_references(self,index,steps):
        if type(index) is not int or not 0<=index<400 or list(steps)!=[1,2,4,8]:
            raise ValueError('only400control exact1/2/4/8 horizon supported')
        # Preserve legacy generator operation order: (index+offset)*control_dt.
        s=sample_reference(self.reference,np.array([(index+i)*.02 for i in steps]))
        rotations=np.broadcast_to(self.initial['body_rot'],(4,33,4)).copy()
        rotations[:,self.anchor]=s['anchor_rot'].astype('float32')
        return {'dof_pos':s['dof_pos'].astype('float32'),'dof_vel':s['dof_vel'].astype('float32'),'body_rot':rotations}


def same(actual,expected,label):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.tobytes()!=expected.tobytes():
        raise ValueError('exact prefix bridge mismatch: '+label)


class PrefixGate:
    """Checks every warmup/online prefix row before permitting next work."""
    def __init__(self,inputs,outputs,qpos,qvel,times,targets):
        self.inputs,self.outputs=inputs,outputs
        self.qpos,self.qvel,self.times,self.targets=qpos,qvel,times,targets
        self.inputs_checked=self.outputs_checked=self.states_checked=self.targets_checked=0
        self.passed=False

    def feed(self,call,feed):
        if call>18:
            if not self.passed:raise ValueError('prefix boundary not passed before decision18')
            return
        if tuple(feed)!=INPUTS:raise ValueError('exact eight-feed interface required')
        prefix=f'call_{call:04d}__'
        for k,v in feed.items():same(v,self.inputs[prefix+k],'input '+str(call)+' '+k)
        self.inputs_checked+=1

    def output(self,call,names,values):
        if tuple(names)!=OUTPUTS or len(values)!=4:raise ValueError('exact four-output interface required')
        if call<=18:
            for n,v in zip(names,values):same(v,self.outputs[f'call_{call:04d}__'+n],'output '+str(call)+' '+n)
            self.outputs_checked+=1

    def state(self,index,qpos,qvel,time):
        if index<=18:
            same(qpos,self.qpos[index],'qpos '+str(index));same(qvel,self.qvel[index],'qvel '+str(index))
            if float(time)!=float(self.times[index]):raise ValueError('prefix time differs')
            self.states_checked+=1
        if index==18:
            if (self.inputs_checked,self.outputs_checked,self.states_checked,self.targets_checked)!=(19,19,19,18):
                raise ValueError('incomplete mandatory prefix boundary')
            self.passed=True

    def target(self,index,target):
        if index<=17:
            same(target,self.targets[index],'appliedPD '+str(index));self.targets_checked+=1


def interface_guard(session):
    ins=[(n.name,n.shape,n.type) for n in session.get_inputs()]
    shapes=[['batch_size',4],['batch_size',29],['batch_size',29],['batch_size',3],['batch_size',1,29],['batch_size',4,4],['batch_size',4,29],['batch_size',4,29]]
    if ins!=[(n,s,'tensor(float)') for n,s in zip(INPUTS,shapes)] or [(n.name,n.shape,n.type) for n in session.get_outputs()]!=[(n,['batch_size',29],'tensor(float)') for n in OUTPUTS]:
        raise ValueError('graph consumes unsupported feed/carrier or output layout')


def call_budget(index,prefix_passed):
    if type(index) is not int or not 0<=index<401:raise RuntimeError('401 total calls exhausted')
    if index>=19 and not prefix_passed:raise RuntimeError('mandatory prefix gate absent')


def step_budget(completed):
    if type(completed) is not int or not 0<=completed<8000:raise RuntimeError('8000step budget exhausted')
