"""Prospective in-place stage screen; immutable V1 measurements are retained."""
import numpy as np
from .evaluation_v1 import evaluate_performance as evaluate_v1,diameter,HAND_THRESHOLD

MAX_STAGE_RADIUS_M=.75
MAX_END_OFFSET_M=.35

def evaluate_performance(qpos,qvel,times,kinematics,joint_names,reference_q,music,*,start_s=0,physics_flags=None):
    report=evaluate_v1(qpos,qvel,times,kinematics,joint_names,reference_q,music,start_s=start_s,physics_flags=physics_flags)
    t=np.asarray(times,dtype=float);end=start_s+music['duration_s'];indices=np.flatnonzero((t>=start_s-1e-10)&(t<=end+1e-10))
    pelvis=np.asarray([kinematics[int(i)]['positions_world_m']['pelvis'] for i in indices],float)
    # Never redefine the stage origin using a later convenient sample.
    anchored=bool(abs(float(t[indices[0]])-start_s)<=1e-9)
    radius=end_offset=None
    if anchored:
        displacement=pelvis[:,:2]-pelvis[0,:2];radius=float(np.max(np.linalg.norm(displacement,axis=1)));end_offset=float(np.linalg.norm(displacement[-1]))
    stage_pass=anchored and radius<=MAX_STAGE_RADIUS_M and end_offset<=MAX_END_OFFSET_M
    relative={}
    for side in ('left','right'):
        hand=np.asarray([kinematics[int(i)]['positions_world_m'][side+'_rubber_hand'] for i in indices],float);points=hand-pelvis
        span=diameter(points)
        relative[side]={'temporal_diameter_m':span,'axis_span_pelvis_relative_world_axes_m':np.ptp(points,axis=0).tolist(),'minimum_m':HAND_THRESHOLD,'passed':span>=HAND_THRESHOLD,'qualification':'Translation-relative hand−pelvis positions in fixed WORLD axes; not a rotating pelvis-local frame or reachability certificate.'}
    relative_pass=all(v['passed'] for v in relative.values());legacy=report['performance_screen_passed']
    combined=report['full_window_complete'] and legacy is True and stage_pass and relative_pass
    stage={'origin_time_s':float(start_s),'origin_available':anchored,'origin_pelvis_world_xy_m':pelvis[0,:2].tolist() if anchored else None,'max_planar_radius_from_start_m':radius,'end_planar_offset_from_start_m':end_offset,'limits':{'max_planar_radius_m':MAX_STAGE_RADIUS_M,'max_end_planar_offset_m':MAX_END_OFFSET_M},'passed_so_far':bool(stage_pass),'qualification':'Fixed prospective in-place demo boundary measured from music-start pelvis; no root constraint, recentering, external force, clipping or goal rewrite.'}
    report.update(schema='pulsemorph.whole_body_performance_evaluation.v2',legacy_schema='pulsemorph.whole_body_performance_evaluation.v1',legacy_performance_screen_passed=legacy,stage_bound=stage,hand_pelvis_relative_motion=relative,relative_hand_minimums_met_so_far=relative_pass,performance_screen_passed=bool(combined))
    report['feedback'].update(performance_screen_passed=bool(combined),stage_bound_passed_so_far=bool(stage_pass),pelvis_planar_radius_max_m=radius,pelvis_end_planar_offset_m=end_offset,stage_limits_m={'radius':MAX_STAGE_RADIUS_M,'end_offset':MAX_END_OFFSET_M},relative_hand_workspace_m={k:v['temporal_diameter_m'] for k,v in relative.items()},relative_hand_below_minimum=[k for k,v in relative.items() if not v['passed']],contact_adjacent_foot_motion={k:{f:v[f] for f in ('eligible_adjacent_pairs','horizontal_path_m','mean_speed_m_s','max_speed_m_s')} for k,v in report['contact_adjacent_foot_motion'].items()})
    report['qualification']+=' V2 adds preregistered stage radius/end-offset and BOTH translation-relative hand-workspace gates. V1 absolute metrics remain descriptive and unchanged; prior V1 traces are not relabelled.'
    return report
