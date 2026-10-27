"""Reference-only Wave→High5 phrase composition; no controller or root motion.

The curve consumes already validated typed references. Joint Hermite endpoint
derivatives give C1 joins. Torso SLERP gives orientation continuity only.
"""
from copy import deepcopy
from collections.abc import Mapping
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp
from pulsemorph.motion.contracts import MotionReference
from pulsemorph.motion.g1 import sample_reference


def plain(value):
    if isinstance(value, Mapping): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    return value


def compose_phrase(wave, high5, *, reference_id, sources):
    """Fixed 0..3.5 Wave; 3.5..4.5 Hermite; High5(t−4) through8.14.

    No source extrapolation, clamping, full-body carrier or root synthesis.
    Sources must bind same body/controller/frame and provide full4.14s horizon.
    """
    for ref in (wave,high5):
        ref.validate()
        if ref.time_offsets_s[0] != 0 or ref.time_offsets_s[-1] < 4.14:
            raise ValueError('both source packages must cover 0..4.14 without padding')
    for group in ('body','controller','frame'):
        if plain(wave.metadata[group]) != plain(high5.metadata[group]):
            raise ValueError('source '+group+' mismatch')
    if wave.metadata['body']['anchor_body_name'] != 'torso_link' or len(wave.metadata['body']['joint_names']) != 29 or wave.metadata['controller']['control_dt_s'] != .02:
        raise ValueError('only the admitted 29-joint G1 torso profile is supported')
    left=sample_reference(wave,np.array([3.5]));right=sample_reference(high5,np.array([.5]))
    bridge=CubicHermiteSpline([3.5,4.5],np.vstack([left['dof_pos'],right['dof_pos']]),np.vstack([left['dof_vel'],right['dof_vel']]),extrapolate=False)
    turn=Slerp([3.5,4.5],Rotation.from_quat(np.vstack([left['anchor_rot'],right['anchor_rot']])))
    # Preserve source knots/bytes in the unchanged prefix; translated High5 knots
    # retain its piecewise curve. Inserting transition knots subdivides a cubic.
    wt=wave.time_offsets_s[wave.time_offsets_s <= 3.5]
    if wt[-1] != 3.5: wt=np.r_[wt,3.5]
    middle=np.arange(176,225,dtype=float)*.02
    ht=high5.time_offsets_s[(high5.time_offsets_s > .5)&(high5.time_offsets_s <= 4.14)]+4
    times=np.r_[wt,middle,4.5,ht]
    if np.any(np.diff(times)<=0) or times[-1]!=8.14:
        raise ValueError('unexpected source knot layout/end; no repair or padding')
    q=[];dq=[];rot=[]
    for t in times:
        if t<=3.5: s=sample_reference(wave,np.array([t]));a,b,c=s['dof_pos'][0],s['dof_vel'][0],s['anchor_rot'][0]
        elif t<4.5: a,b,c=bridge(t),bridge(t,1),turn([t]).as_quat()[0]
        else: s=sample_reference(high5,np.array([t-4]));a,b,c=s['dof_pos'][0],s['dof_vel'][0],s['anchor_rot'][0]
        q.append(a);dq.append(b);rot.append(c)
    metadata=plain(wave.metadata)
    metadata['reference_id']=reference_id
    metadata['producer']={'kind':'agentic','id':'fixed-wave-high5-phrase-reference','version':'1'}
    metadata['context'].update(decision_id=reference_id+':offline',observation_id='phrase:no_audio',task_id='phrase_reference_plan_v1',runtime_session_id='reference_only',protocol_sha256=None)
    metadata['validity'].update(end_time_s=8.14)
    metadata['observation'].update(observed_through_time_s=0.,audio_observed_through_sample=None,video_observed_through_time_s=None)
    metadata['provenance']={'sources':sources,'notes':[
        'Wave0..3.5; joint-Hermite and shortest torso SLERP3.5..4.5; High5(t-4) thereafter through8.14.',
        'Joint position/first derivative matched at joins; second derivatives not matched. Torso orientation C0 only, angular-velocity continuity not claimed.',
        'Only29q/dq and torso orientation; no full-body carrier/FK measurement or root trajectory composed.',
        'Recorded reference worlds share named frame semantics; no spatial/root registration or physical transition guarantee.',
        'Reference-only tool feasibility; no policy inference, physics, audio, LLM or learning.']}
    return MotionReference(metadata,times,np.asarray(q),np.asarray(dq),np.asarray(rot))
