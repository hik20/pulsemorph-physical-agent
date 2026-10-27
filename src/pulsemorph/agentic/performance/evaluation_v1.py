"""Saved-array movement/music diagnostics; no simulator, FK, NN or planning.

Demo minimums are transparent finite-example requirements, not aesthetic,
general dance, balance or hardware-safety certification.
"""
import math
import numpy as np

GROUP_THRESHOLDS={'arms':.6,'trunk':.08,'hips':.10,'knees':.16,'ankles':.04}
HAND_THRESHOLD=.20
PELVIS_THRESHOLD=.025
MIN_ROOT_HEIGHT=.66
MIN_ROOT_UP_Z=.90
BODY_NAMES=('pelvis','torso_link','left_rubber_hand','right_rubber_hand','left_ankle_roll_link','right_ankle_roll_link')

def number(value):return type(value) in (int,float) and math.isfinite(value)
def array(value,ndim,name):
    a=np.asarray(value)
    if a.ndim!=ndim or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():raise ValueError('Finite numeric '+name+' required')
    return a.astype(float,copy=False)
def clock(value,name):
    t=array(value,1,name)
    if len(t)<2 or t[0]<0 or np.any(np.diff(t)<=0):raise ValueError('Strict nonnegative clock required: '+name)
    return t
def groups(names):
    if type(names) not in (list,tuple) or len(names)!=29 or any(type(n) is not str for n in names) or len(set(names))!=29:raise ValueError('Exact named29-joint G1 layout required')
    result={'arms':[],'trunk':[],'hips':[],'knees':[],'ankles':[]}
    for i,n in enumerate(names):
        if any(x in n for x in ('shoulder_','elbow_','wrist_')):key='arms'
        elif n.startswith('waist_'):key='trunk'
        elif '_hip_' in n:key='hips'
        elif '_knee_' in n:key='knees'
        elif '_ankle_' in n:key='ankles'
        else:raise ValueError('Unrecognized G1 joint: '+n)
        result[key].append(i)
    if {k:len(v) for k,v in result.items()}!={'arms':14,'trunk':3,'hips':6,'knees':2,'ankles':4}:raise ValueError('G1 group counts differ')
    return result
def spans(q,names,indexes):
    values=np.ptp(q,axis=0)
    return {k:{'span_rad':float(max(values[v])),'side_max_span_rad':{side:float(max(values[j] for j in v if names[j].startswith(side+'_'))) for side in ('left','right') if any(names[j].startswith(side+'_') for j in v)},'per_joint_peak_to_peak_rad':{names[j]:float(values[j]) for j in v},'definition':'maximum individual joint temporal peak-to-peak; not every joint or side active'} for k,v in indexes.items()}
def diameter(points):
    maximum=0.
    # Bounded blocks avoid a quadratic full distance matrix allocation.
    for begin in range(0,len(points),128):
        d=points[begin:begin+128,None,:]-points[None,:,:]
        maximum=max(maximum,float(np.sqrt(np.max(np.sum(d*d,axis=2)))))
    return maximum
def trajectory(points):
    return {'temporal_diameter_m':diameter(points),'axis_span_world_m':np.ptp(points,axis=0).tolist(),'vertical_span_m':float(np.ptp(points[:,2])),'path_length_m':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())}
def peak_alignment(times,energy,beat_times,start):
    spread=float(np.ptp(energy));candidates=[]
    if spread>1e-12:
        threshold=float(np.min(energy)+.25*spread)
        candidates=[i for i in range(1,len(energy)-1) if energy[i]>energy[i-1] and energy[i]>=energy[i+1] and energy[i]>=threshold]
    chosen=[]
    for i in sorted(candidates,key=lambda i:(-energy[i],i)):
        if all(abs(times[i]-times[j])>=.15-1e-12 for j in chosen):chosen.append(i)
    peaks=[float(times[i]-start) for i in sorted(chosen)];rows=[]
    for peak in peaks:
        if not beat_times:rows.append({'motion_peak_s':peak,'nearest_beat_delta_s':None,'beat_phase_cycles':None});continue
        nearest=min(beat_times,key=lambda b:(abs(peak-b),b));idx=int(np.searchsorted(beat_times,peak,side='right'))-1;phase=None
        if 0<=idx<len(beat_times)-1:phase=(peak-beat_times[idx])/(beat_times[idx+1]-beat_times[idx])
        rows.append({'motion_peak_s':peak,'nearest_beat_delta_s':float(peak-nearest),'beat_phase_cycles':None if phase is None else float(phase)})
    distances=[abs(r['nearest_beat_delta_s']) for r in rows if r['nearest_beat_delta_s'] is not None]
    reverse=[min(abs(b-p) for p in peaks) for b in beat_times] if peaks else []
    return {'motion_energy_peaks':rows,'mean_peak_to_nearest_beat_s':float(np.mean(distances)) if distances else None,'p90_peak_to_nearest_beat_s':float(np.percentile(distances,90)) if distances else None,'mean_beat_to_nearest_peak_s':float(np.mean(reverse)) if reverse else None,'beat_fraction_with_peak_within_100ms':float(np.mean(np.asarray(reverse)<=.1)) if reverse else None,'algorithm':{'relative_energy_threshold':.25,'peak_min_separation_s':.15,'beat_independent_peak_detection':True},'qualification':'Joint-speed-squared activity peaks are descriptive, not physical energy, dance quality, beat inference or aesthetic validation.'}

def evaluate_performance(qpos,qvel,times,kinematics,joint_names,reference_q,music,*,start_s=0,physics_flags=None):
    """Analyze actual data in the music window; partial windows remain partial.

    reference_q may be None, an Nx29 array aligned to actual times, or an
    explicit {times,q} mapping. No reference padding/retiming is performed.
    """
    if not number(start_s) or start_s<0:raise ValueError('Finite nonnegative music start required')
    t=clock(times,'actual times');q=array(qpos,2,'qpos');v=array(qvel,2,'qvel');index=groups(joint_names)
    if q.shape!=(len(t),36) or v.shape!=(len(t),35) or len(t)>12001:raise ValueError('Aligned bounded G1 qpos36/qvel35 required')
    if type(music) is not dict or not number(music.get('duration_s')) or not 0<music['duration_s']<=120:raise ValueError('Actual music duration required')
    if 'beats_s' in music and 'beat_times_s' in music and music['beats_s']!=music['beat_times_s']:raise ValueError('Conflicting beat grids forbidden')
    beats=music.get('beats_s',music.get('beat_times_s',[]))
    music_sha=music.get('sha256')
    if music_sha is not None and (type(music_sha) is not str or len(music_sha)!=64 or any(c not in '0123456789abcdef' for c in music_sha)):raise ValueError('Music digest must be SHA256')
    if type(beats) is not list or len(beats)>1000 or any(not number(b) or not 0<=b<=music['duration_s'] for b in beats) or any(a>=b for a,b in zip(beats,beats[1:])):raise ValueError('Explicit strictly ordered relative beat grid required')
    end=start_s+music['duration_s'];mask=(t>=start_s-1e-10)&(t<=end+1e-10);tw=t[mask]
    if len(tw)<1:raise ValueError('At least one actual sample in music window required')
    complete=bool(tw[0]<=start_s+1e-9 and tw[-1]>=end-1e-9);aw=q[mask,7:];vw=v[mask,6:];actual_groups=spans(aw,joint_names,index)
    if type(kinematics) is not list or len(kinematics)!=len(t):raise ValueError('Row-aligned already-measured kinematics required')
    rows=[]
    for i in np.flatnonzero(mask):
        row=kinematics[int(i)]
        if type(row) is not dict or not number(row.get('sim_time_s')) or abs(row['sim_time_s']-t[i])>1e-9:raise ValueError('Kinematic timestamp differs; no alignment guessing')
        if any(not number(row.get(k)) for k in ('root_height_m','root_up_z')) or type(row.get('ncon')) is not int or row['ncon']<0:raise ValueError('Actual root/contact metadata required')
        if not -1.-1e-9<=row['root_up_z']<=1.+1e-9:raise ValueError('Root upright cosine outside physical range')
        com=row.get('com_world_m')
        if type(com) is not list or len(com)!=3 or any(not number(x) for x in com):raise ValueError('Actual measured COM required')
        for name in BODY_NAMES:
            p=row.get('positions_world_m',{}).get(name)
            if type(p) is not list or len(p)!=3 or any(not number(x) for x in p):raise ValueError('Actual measured body position required: '+name)
        if type(row.get('foot_contacts')) is not dict or any(type(row['foot_contacts'].get(side)) is not bool for side in ('left','right')):raise ValueError('Actual Boolean foot contacts required')
        rows.append(row)
    bodies={name:np.asarray([row['positions_world_m'][name] for row in rows],float) for name in BODY_NAMES}
    hands={side:trajectory(bodies[side+'_rubber_hand']) for side in ('left','right')};pelvis=trajectory(bodies['pelvis']);pelvis['lateral_span_m']=float(np.ptp(bodies['pelvis'][:,1]));com_motion=trajectory(np.asarray([r['com_world_m'] for r in rows]));feet={}
    dt=np.diff(tw)
    for side in ('left','right'):
        contact=np.asarray([r['foot_contacts'][side] for r in rows]);eligible=contact[:-1]&contact[1:]&(dt<=.05+1e-12);motion=np.linalg.norm(np.diff(bodies[side+'_ankle_roll_link'],axis=0)[:,:2],axis=1);duration=float(dt[eligible].sum());speeds=motion[eligible]/dt[eligible]
        feet[side]={'eligible_adjacent_pairs':int(eligible.sum()),'measured_contact_pair_duration_s':duration,'horizontal_path_m':float(motion[eligible].sum()),'mean_speed_m_s':float(motion[eligible].sum()/duration) if duration else None,'max_speed_m_s':float(max(speeds)) if len(speeds) else None,'qualification':'Contact-adjacent ankle-link horizontal motion proxy, max gap50ms; not sole contact-patch slip or continuous/hardware proof.'}
    reference=None
    if reference_q is not None:
        if type(reference_q) is dict:
            if set(reference_q)!={'times','q'}:raise ValueError('Exact reference times/q mapping required')
            rt=clock(reference_q['times'],'reference times');rq=array(reference_q['q'],2,'reference q')
        else:rt=t;rq=array(reference_q,2,'reference q')
        if rq.shape!=(len(rt),29):raise ValueError('Explicit aligned29 reference joints required')
        rm=(rt>=start_s-1e-10)&(rt<=end+1e-10)
        if int(rm.sum())<1:raise ValueError('Reference window lacks samples')
        reference={'groups':spans(rq[rm],joint_names,index),'sample_count':int(rm.sum()),'coverage_s':[float(rt[rm][0]),float(rt[rm][-1])],'qualification':'Committed reference motion, separate from actual realization; no inferred tracking alignment.'}
    activities={k:np.mean(vw[:,ids]**2,axis=1) for k,ids in index.items()};energy=np.mean(vw**2,axis=1);observed_beats=[b for b in beats if tw[0]-start_s-1e-10<=b<=tw[-1]-start_s+1e-10]
    alignment=peak_alignment(tw,energy,observed_beats,start_s)
    alignment.update(beat_count_in_observed_window=len(observed_beats),beat_grid_source=music.get('bpm_source','Caller-declared grid; no tempo inference performed'),music_sha256=music_sha)
    evidence={k:{'observed':actual_groups[k]['span_rad'],'minimum':minimum,'passed':actual_groups[k]['span_rad']>=minimum} for k,minimum in GROUP_THRESHOLDS.items()}
    for side in ('left','right'):evidence[side+'_hand_workspace']={'observed':hands[side]['temporal_diameter_m'],'minimum':HAND_THRESHOLD,'passed':hands[side]['temporal_diameter_m']>=HAND_THRESHOLD}
    for key,observed in [('pelvis_vertical',pelvis['vertical_span_m']),('pelvis_lateral',pelvis['lateral_span_m'])]:evidence[key]={'observed':observed,'minimum':PELVIS_THRESHOLD,'passed':observed>=PELVIS_THRESHOLD}
    physical={'passed':None,'qualification':'Backend physical gate not supplied; movement spans do not establish balance/safety.'}
    if physics_flags is not None:
        if type(physics_flags) is not dict or any(type(physics_flags.get(k)) is not bool for k in ('finite','warnings_unchanged','stopped','physical_gate_passed','force_caps_passed')):raise ValueError('Explicit backend Boolean physical gate/flags required')
        physical={'passed':physics_flags['physical_gate_passed'] and physics_flags['finite'] and physics_flags['warnings_unchanged'] and physics_flags['force_caps_passed'] and not physics_flags['stopped'] and min(r['root_height_m'] for r in rows)>=MIN_ROOT_HEIGHT and min(r['root_up_z'] for r in rows)>=MIN_ROOT_UP_Z,'flags':{k:physics_flags[k] for k in ('finite','warnings_unchanged','stopped','physical_gate_passed','force_caps_passed')},'demo_thresholds':{'min_root_height_m':MIN_ROOT_HEIGHT,'min_root_up_z':MIN_ROOT_UP_Z},'qualification':'Caller-supplied backend flags plus sampled .66m/.90 demo screen; original backend hardstops unchanged. Not reconstructed substep balance or hardware safety certification.'}
    below=[key for key,e in evidence.items() if not e['passed']];movement=not below;passed=complete and movement
    return {'schema':'pulsemorph.whole_body_performance_evaluation.v1','status':'complete_window' if complete else 'partial_window','music_window_s':[float(start_s),float(end)],'actual_coverage_s':[float(tw[0]),float(tw[-1])],'full_window_complete':complete,'sample_count':len(tw),'actual_groups':actual_groups,'reference':reference,'hand_world_motion':hands,'pelvis_world_motion':pelvis,'com_world_motion':com_motion,'root_observed':{'min_height_m':min(r['root_height_m'] for r in rows),'min_up_z':min(r['root_up_z'] for r in rows)},'contact_adjacent_foot_motion':feet,'mean_joint_speed_squared_rad2_s2':{k:float(np.mean(e)) for k,e in activities.items()},'music_alignment':alignment,'demo_minimum_evidence':evidence,'demo_minimums_met_so_far':movement,'demo_minimum_passed':passed,'performance_screen_passed':None if physical['passed'] is None else passed and physical['passed'],'physical_gate':physical,'feedback':{'full_window_complete':complete,'below_minimums':below,'physical_gate_passed':physical['passed'],'demo_minimum_passed':passed,'actual_group_span_rad':{k:v['span_rad'] for k,v in actual_groups.items()},'hand_workspace_m':{k:v['temporal_diameter_m'] for k,v in hands.items()},'pelvis_vertical_m':pelvis['vertical_span_m'],'pelvis_lateral_m':pelvis['lateral_span_m'],'mean_motion_peak_to_beat_s':alignment['mean_peak_to_nearest_beat_s']},'qualification':'Fixed demo movement minimums; both hand temporal workspaces and pelvis vertical+lateral spans required. Partial trace deficits are not global infeasibility. Tracking, music timing, expression, instruction compliance and aesthetics are separate; no plan selection or goal relaxation performed.'}
