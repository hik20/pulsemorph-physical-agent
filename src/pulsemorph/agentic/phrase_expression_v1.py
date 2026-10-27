"""Pose-offset and temporal source expression; pure references, no execution grant."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from pulsemorph.motion.contracts import MotionReference
from pulsemorph.motion.g1 import profile_from_yaml,assemble_reference_inputs
from .phrase_reference import plain
from .phrase_reference_v2 import compose_ordered_pair
from .phrase_program import _gain_source,_arm_indices,_continuous_ranges,ProgramCompileError

SCHEMA='pulsemorph.phrase_expression.v1'
GAIN_FIELDS=tuple(f'{phase}_{side}_{kind}_gain' for phase in ('first','second') for side in ('left','right') for kind in ('pose','motion'))
FIELDS=frozenset(('schema','order','heading_policy','rationale',*GAIN_FIELDS))
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def parse_expression(value):
    if type(value) is not dict or set(value)!=FIELDS or value.get('schema')!=SCHEMA:raise ValueError('Exact phrase expression schema fields required')
    if type(value['order']) is not list or value['order'] not in (['wave','high5'],['high5','wave']):raise ValueError('Distinct Wave/High5 ordered pair required')
    if type(value['heading_policy']) is not str or value['heading_policy'] not in ('recorded_world','align_join_heading'):raise ValueError('Explicit supported heading policy required')
    if type(value['rationale']) is not str or len(value['rationale'])>1200:raise ValueError('Rationale at most1200 characters required')
    value['rationale'].encode('utf8')
    gains={}
    for field in GAIN_FIELDS:
        number=value[field]
        if type(number) not in (int,float) or not .75<=number<=1.25 or not math.isfinite(number):raise ValueError('Finite expression gain in [.75,1.25], not bool, required')
        gains[field]=float(number)
    return {'schema':SCHEMA,'order':list(value['order']),'heading_policy':value['heading_policy'],**gains,'rationale':value['rationale']}

def inspect_source_expression(reference):
    """Fixed component means at native knots, never motion/physical calibration."""
    reference.validate();t=reference.time_offsets_s;q=reference.joint_position_rad
    if t.shape!=(208,) or q.shape!=(208,29) or not np.allclose(t,np.arange(208)*.02,rtol=0,atol=1e-12) or not np.any(t==.5) or not np.any(t==1.):raise ValueError('Unchanged native 0..4.14s20ms source clock required')
    keep=(t>=1.2-1e-10)&(t<=3.2+1e-10)
    if int(keep.sum())!=101:raise ValueError('Exact101 source mean knots required')
    names=tuple(reference.metadata['body']['joint_names']);groups=_arm_indices(names);arms={}
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for side,indices in zip(('left','right'),groups):
                values=q[keep][:,indices].copy();q0=q[0,indices].copy();mean=values.mean(axis=0);pose=mean-q0;motion=values-mean
                pp=float(np.mean(pose**2));mm=float(np.mean(motion**2));pm=float(np.mean(motion*pose[None,:]));norms=[math.sqrt(pp),math.sqrt(mm)]
                condition=max(norms)/min(norms) if min(norms)>0 else None
                arms[side]={'joint_names':[names[i] for i in indices],'q0_rad':q0.tolist(),'mean_rad':mean.tolist(),'pose_basis_rms_rad':norms[0],'motion_basis_rms_rad':norms[1],'basis_cross_mean_rad2':pm,'basis_gram_mean_rad2':[[pp,pm],[pm,mm]],'basis_norm_condition_number':condition,'pose_basis_exact_zero':bool(np.all(pose==0)),'motion_basis_exact_zero':bool(np.all(motion==0))}
        result={'source_reference_id':reference.metadata['reference_id'],'mean_source_window_s':[1.2,3.2],'sample_count':101,'componentwise_mean':True,'native_dt_s':.02,'arms':arms,'qualification':'Source-joint numerical bases only; zero/conditioning diagnostics are not physical controllability or inactive-goal admission.'}
        _canonical(result);return result
    except (FloatingPointError,OverflowError) as error:raise ValueError('Nonfinite expression basis arithmetic') from error

def describe_bases(references):
    if type(references) is not dict or set(references)!={'wave','high5'}:raise ValueError('Exact source pair required')
    return {'schema':'pulsemorph.phrase_expression_bases.v1','sources':{name:inspect_source_expression(reference) for name,reference in references.items()},'execution_authority':False}

def edit_source_expression(reference,left_pose_gain,left_motion_gain,right_pose_gain,right_motion_gain):
    """Exact ramped product-rule knots; resulting common Hermite owns interiors."""
    gains=(left_pose_gain,left_motion_gain,right_pose_gain,right_motion_gain)
    if any(type(g) not in (int,float) or not .75<=g<=1.25 or not math.isfinite(g) for g in gains):raise ValueError('Finite bounded source expression gains required')
    inspection=inspect_source_expression(reference)
    if left_pose_gain==left_motion_gain and right_pose_gain==right_motion_gain:return _gain_source(reference,left_pose_gain,right_pose_gain),inspection
    t=reference.time_offsets_s;active=t>.5;u=(t-.5)/.5;w=np.zeros_like(t);wd=np.zeros_like(t);inside=(t>.5)&(t<1.);w[inside]=3*u[inside]**2-2*u[inside]**3;wd[inside]=(6*u[inside]-6*u[inside]**2)/.5;w[t>=1.]=1.
    q=reference.joint_position_rad.copy();dq=reference.joint_velocity_rad_s.copy();names=tuple(reference.metadata['body']['joint_names']);groups=_arm_indices(names)
    for side,indices,p,m in zip(('left','right'),groups,(left_pose_gain,right_pose_gain),(left_motion_gain,right_motion_gain)):
        if p==m:
            legacy=_gain_source(reference,p if side=='left' else 1.,p if side=='right' else 1.)
            q[:,indices]=legacy.joint_position_rad[:,indices];dq[:,indices]=legacy.joint_velocity_rad_s[:,indices];continue
        q0=np.asarray(inspection['arms'][side]['q0_rad']);mean=np.asarray(inspection['arms'][side]['mean_rad'])
        delta=(p-1)*(mean-q0)[None,:]+(m-1)*(reference.joint_position_rad[:,indices]-mean[None,:])
        edited_q=reference.joint_position_rad[:,indices]+w[:,None]*delta
        edited_dq=reference.joint_velocity_rad_s[:,indices]+w[:,None]*(m-1)*reference.joint_velocity_rad_s[:,indices]+wd[:,None]*delta
        q[np.ix_(active,indices)]=edited_q[active];dq[np.ix_(active,indices)]=edited_dq[active]
    metadata=plain(reference.metadata);metadata['provenance']['notes'].append('Expressionv1 fixed native101-knot component mean; independent pose-offset/temporal gains, smoothstep.5..1s and exact product-rule dq at source knots. No timing/root/contact/physical claim.')
    return MotionReference(metadata,t,q,dq,reference.anchor_orientation_xyzw),inspection

def compile_expression(references,value,*,reference_id,sources):
    """Caller pins exact source packages; profile/range checks do not grant trust."""
    p=parse_expression(value);digest=hashlib.sha256(_canonical(p)).hexdigest();profile=profile_from_yaml()
    if type(references) is not dict or set(references)!={'wave','high5'}:raise ValueError('Exact admitted source pair required')
    if type(reference_id) is not str or not reference_id.strip():raise ValueError('Reference ID required')
    for ref in references.values():
        ref.validate()
        if tuple(ref.metadata['body']['joint_names'])!=profile['joint_names'] or ref.metadata['body']['profile_sha256']!=profile['body_sha256'] or ref.metadata['controller']['profile_sha256']!=profile['yaml_sha256']:raise ValueError('Expression source profile/order mismatch')
    if type(sources) is not list:raise ValueError('Explicit source records required')
    lineage=[];seen=set()
    for record in [*sources,*[plain(s) for ref in references.values() for s in ref.metadata['provenance']['sources']]]:
        if type(record) is not dict or set(record)!={'path','sha256'} or type(record['path']) is not str or not record['path'].strip() or type(record['sha256']) is not str or len(record['sha256'])!=64 or any(c not in '0123456789abcdef' for c in record['sha256']):raise ValueError('Finite explicit source hash lineage required')
        key=(record['path'],record['sha256'])
        if key not in seen:seen.add(key);lineage.append(dict(record))
    adjusted={};inspections={}
    for index,name in enumerate(p['order']):
        prefix='first' if index==0 else 'second'
        adjusted[name],inspections[name]=edit_source_expression(references[name],*[p[f'{prefix}_{side}_{kind}_gain'] for side in ('left','right') for kind in ('pose','motion')])
    composed,transform=compose_ordered_pair(adjusted,order=p['order'],heading_policy=p['heading_policy'],reference_id=reference_id,sources=lineage)
    metadata=plain(composed.metadata);metadata['producer']={'kind':'agentic','id':'phrase-expression-reference','version':'1'}
    metadata['context'].update(decision_id=reference_id+':expression',observation_id='phrase-expression:'+digest,task_id='phrase_expression_v1',runtime_session_id=reference_id+':reference-only',expression_sha256=digest)
    metadata['provenance'].update(phrase_expression=p,phrase_expression_sha256=digest,expression_source_bases=inspections,source_reference_ids={n:references[n].metadata['reference_id'] for n in references})
    metadata['provenance']['notes'] += ['Independent source-joint mean-offset and centered-temporal gains are numerical controls, not perceptual style or new skills.',
        'q_new=q+w*((pose-1)*(mean-q0)+(motion-1)*(q-mean));dq_new=dq+w*(motion-1)*dq+w_dot*((pose-1)*(mean-q0)+(motion-1)*(q-mean)). Mean fixed per joint from101native1.2..3.2s knots.',
        'At/before.5 q/dq source bytes unchanged; equal pose/motion routes frozen gain helper. Product-rule dq exact at source knots; final common cubic Hermite/SLERP authoritative between knots. JointC1/torsoC0 only, no angular velocity/root/contact/SE3/physical guarantee.']
    reference=MotionReference(metadata,composed.time_offsets_s,composed.joint_position_rad,composed.joint_velocity_rad_s,composed.anchor_orientation_xyzw);ranges=_continuous_ranges(reference,profile)
    report={'schema':'pulsemorph.phrase_expression_compile.v1','expression':p,'expression_sha256':digest,'reference_id':reference_id,'execution_authority':False,'duration_s':8.,'horizon_s':8.14,'transition_s':[3.5,4.5],'timing_changed':False,'source_bases':inspections,'heading_transform':transform,'sources':lineage,'source_reference_ids':{n:references[n].metadata['reference_id'] for n in references},'body_sha256':profile['body_sha256'],'controller_sha256':profile['yaml_sha256'],'continuous_joint_ranges':ranges,'range_numerical_tolerance_rad':1e-8,'strict_future_queries':0,'equal_gain_legacy_route':{prefix:all(p[f'{prefix}_{side}_pose_gain']==p[f'{prefix}_{side}_motion_gain'] for side in ('left','right')) for prefix in ('first','second')},'unit_all_numeric_bypass':all(p[k]==1 for k in GAIN_FIELDS),'new_physics_steps':0,'new_onnx_calls':0,'new_FK_calls':0,'new_llm_calls':0,'qualification':'Expression reference-only proposal, not physical admission, goal attainment, style/skill learning or rights grant.'}
    violations=[r['joint'] for r in ranges if not r['within_unchanged_range']]
    if violations:report.update(status='rejected_joint_ranges',violations=violations);raise ProgramCompileError('Expression reference exceeds unchanged joint ranges',report)
    equal=True
    for i in range(400):
        inputs=assemble_reference_inputs(reference,i*.02)
        if i<=17:
            original=assemble_reference_inputs(references[p['order'][0]],i*.02);equal=equal and all(inputs[k].tobytes()==original[k].tobytes() for k in inputs)
    if not equal:report.update(status='rejected_source_prefix',first18_reference_future_feed_bytes_equal=False);raise ProgramCompileError('Expression first18 reference feed prefix changed',report)
    for key,path in (('body_sha256',profile['body_path']),('yaml_sha256',profile['yaml_path'])):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=profile[key]:raise ValueError('Expression body/controller bytes changed during compilation')
    report.update(status='reference_validated_no_execution_authority',strict_future_queries=400,first18_reference_future_feed_bytes_equal=True,prefix_qualification='Three reference future tensors only; not actual current/history policy inputs or physical prefix proof.')
    _canonical(report);return reference,report
