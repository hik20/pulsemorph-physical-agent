"""Strict finite phrase programs compile references; they grant no execution authority.

Only two ordered source references, fixed timing and four bounded arm gains are
expressible. No policy/model/physics invocation, music analysis or learning.
"""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from pulsemorph.motion.contracts import MotionReference
from pulsemorph.motion.g1 import profile_from_yaml,assemble_reference_inputs
from .phrase_reference import plain
from .phrase_reference_v2 import compose_ordered_pair

SCHEMA='pulsemorph.phrase_program.v1'
GAIN_FIELDS=('first_left_gain','first_right_gain','second_left_gain','second_right_gain')
FIELDS=frozenset(('schema','order','heading_policy','rationale',*GAIN_FIELDS))
ARM_PARTS=('shoulder_pitch_joint','shoulder_roll_joint','shoulder_yaw_joint','elbow_joint','wrist_roll_joint','wrist_pitch_joint','wrist_yaw_joint')


class ProgramCompileError(ValueError):
    """Rejected reference; callers may preserve report, never execute the proposal."""
    def __init__(self,message,report):
        super().__init__(message)
        self.report=report


def parse_program(value):
    """Validate a JSON-object vocabulary exactly and return an owned canonical dict."""
    if type(value) is not dict or set(value)!=FIELDS:raise ValueError('phrase program requires exact schema fields; unknown/missing keys rejected')
    if value['schema']!=SCHEMA or type(value['schema']) is not str:raise ValueError('unsupported phrase program schema')
    if type(value['order']) is not list or value['order'] not in (['wave','high5'],['high5','wave']):raise ValueError('order must be one of the two distinct admitted source pairs')
    if type(value['heading_policy']) is not str or value['heading_policy'] not in ('recorded_world','align_join_heading'):raise ValueError('explicit supported heading_policy required')
    if type(value['rationale']) is not str or len(value['rationale'])>1200:raise ValueError('rationale must be a string of at most1200 characters')
    gains={}
    for field in GAIN_FIELDS:
        number=value[field]
        if type(number) not in (int,float) or not .75<=number<=1.25 or not math.isfinite(number):raise ValueError(field+' must be a finite JSON number in[.75,1.25], not bool')
        gains[field]=float(number)
    return {'schema':SCHEMA,'order':list(value['order']),'heading_policy':value['heading_policy'],**gains,'rationale':value['rationale']}


def _program_hash(program):
    return hashlib.sha256(json.dumps(program,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()


def _arm_indices(names):
    groups=[]
    for side in ('left','right'):
        wanted=tuple(side+'_'+part for part in ARM_PARTS)
        found=tuple(n for n in names if n.startswith(side+'_') and any(part in n for part in ('shoulder_','elbow_','wrist_')))
        if set(found)!=set(wanted) or len(found)!=7:raise ValueError('exact existing seven-joint '+side+' arm group required')
        groups.append(np.array([names.index(n) for n in wanted],dtype=int))
    return groups


def _gain_source(reference,left,right):
    """Exact product values/derivatives at original knots; Hermite owns interiors."""
    names=tuple(reference.metadata['body']['joint_names']);groups=_arm_indices(names)
    if left==right==1.:return reference
    t=reference.time_offsets_s
    if not np.any(t==.5) or not np.any(t==1.):raise ValueError('source knots must explicitly contain gain-ramp boundaries.5/1.0')
    u=(t-.5)/.5;w=np.zeros_like(t);wd=np.zeros_like(t)
    inside=(t>.5)&(t<1.);after=t>=1.
    w[inside]=3*u[inside]**2-2*u[inside]**3;w[after]=1.
    wd[inside]=(6*u[inside]-6*u[inside]**2)/.5
    q=reference.joint_position_rad.copy();dq=reference.joint_velocity_rad_s.copy();q0=reference.joint_position_rad[0]
    for indices,gain in zip(groups,(left,right)):
        if gain==1.:continue
        active=t>.5
        displacement=reference.joint_position_rad[:,indices]-q0[indices]
        multiplier=1+(gain-1)*w;derivative=(gain-1)*wd
        # Assignment only after.5 preserves source initial/q/dq bytes exactly.
        q[np.ix_(active,indices)]=(q0[indices]+multiplier[:,None]*displacement)[active]
        dq[np.ix_(active,indices)]=(multiplier[:,None]*reference.joint_velocity_rad_s[:,indices]+derivative[:,None]*displacement)[active]
    metadata=plain(reference.metadata)
    metadata['provenance']['notes'].append('Phrase-program source-clock arm gains around ownq0; smoothstep ramp.5..1s/product-rule dq at knots; resulting common Hermite authoritative betweenknots.')
    return MotionReference(metadata,t,q,dq,reference.anchor_orientation_xyzw)


def _joint_limits(profile):
    doc=ET.parse(profile['body_path']).getroot()
    if doc.find('compiler').get('angle')!='radian':raise ValueError('unchanged radian G1 XML required')
    limits={j.attrib['name']:tuple(float(v) for v in j.attrib['range'].split()) for j in doc.findall('.//joint') if 'name' in j.attrib and 'range' in j.attrib}
    if any(n not in limits or len(limits[n])!=2 or not np.isfinite(limits[n]).all() or limits[n][0]>=limits[n][1] for n in profile['joint_names']):raise ValueError('explicit finite G1 joint limits required')
    return limits


def _continuous_ranges(reference,profile):
    curve=CubicHermiteSpline(reference.time_offsets_s,reference.joint_position_rad,reference.joint_velocity_rad_s,extrapolate=False)
    roots=curve.derivative().roots(extrapolate=False);limits=_joint_limits(profile);rows=[]
    for j,name in enumerate(profile['joint_names']):
        critical=roots[j][np.isfinite(roots[j])&(roots[j]>=0)&(roots[j]<=8.14)]
        times=np.unique(np.r_[reference.time_offsets_s,critical]);values=curve(times)[:,j];lo,hi=limits[name]
        rows.append({'joint':name,'minimum_rad':float(values.min()),'maximum_rad':float(values.max()),'limit_min_rad':lo,'limit_max_rad':hi,'within_unchanged_range':bool(values.min()>=lo-1e-8 and values.max()<=hi+1e-8)})
    return rows


def compile_program(references,program,*,reference_id,sources):
    """Return a typed reference and report, or reject; never clamp or execute.

    Caller must bind the two input packages to admitted source hashes. The
    report/provenance retain both original source lineages plus supplied records.
    """
    p=parse_program(program);digest=_program_hash(p);profile=profile_from_yaml()
    if type(references) is not dict or set(references)!={'wave','high5'}:raise ValueError('exact admitted source-reference mapping required')
    for ref in references.values():
        ref.validate()
        if tuple(ref.metadata['body']['joint_names'])!=profile['joint_names'] or ref.metadata['body']['profile_sha256']!=profile['body_sha256'] or ref.metadata['controller']['profile_sha256']!=profile['yaml_sha256']:
            raise ValueError('source body/controller/order does not match current raw-pinned G1 profile')
        _arm_indices(tuple(ref.metadata['body']['joint_names']))
    if type(sources) is not list:raise ValueError('sources must be explicit observed-source records')
    lineage=[];seen=set()
    for record in [*sources,*[plain(s) for ref in references.values() for s in ref.metadata['provenance']['sources']]]:
        if type(record) is not dict or set(record)!={'path','sha256'} or type(record['path']) is not str or not record['path'].strip() or type(record['sha256']) is not str or len(record['sha256'])!=64 or any(c not in '0123456789abcdef' for c in record['sha256']):raise ValueError('source records require a nonempty path and lowercase sha256')
        key=(record['path'],record['sha256'])
        if key not in seen:lineage.append(dict(record));seen.add(key)
    adjusted=dict(references)
    for index,name in enumerate(p['order']):
        prefix='first' if index==0 else 'second'
        adjusted[name]=_gain_source(references[name],p[prefix+'_left_gain'],p[prefix+'_right_gain'])
    composed,transform=compose_ordered_pair(adjusted,order=p['order'],heading_policy=p['heading_policy'],reference_id=reference_id,sources=lineage)
    metadata=plain(composed.metadata)
    metadata['producer']={'kind':'agentic','id':'phrase-program-reference','version':'1'}
    metadata['context'].update(decision_id=reference_id+':program',observation_id='phrase-program:'+digest,task_id='phrase_program_v1',runtime_session_id=reference_id+':reference-only',program_sha256=digest)
    metadata['provenance'].update(phrase_program=p,phrase_program_sha256=digest,source_reference_ids={n:references[n].metadata['reference_id'] for n in references})
    metadata['provenance']['notes'] += ['Program is untrusted intent data, not execution authority. No model/physics/LLM/music operation performed.',
        'Arm q=q0+m(t)*(q-q0),dq=m*dq+m_dot*(q-q0);m=1 until.5,smoothstep to gain at1s,constant thereafter. Local source clock; no timing change.',
        'Original20ms source knots evaluate product-rule values/derivatives. Smoothstep times source Hermite can be degree6; materialized common cubic is authoritative betweenknots, not claimed exactly analytic there.',
        'Initial source0..0.5 unmodified. With160ms lookahead, original policy future-feed prefix only extends through decision17/time.34 when gains are active. No physical prefix proof implied.']
    reference=MotionReference(metadata,composed.time_offsets_s,composed.joint_position_rad,composed.joint_velocity_rad_s,composed.anchor_orientation_xyzw)
    ranges=_continuous_ranges(reference,profile)
    report={'schema':'pulsemorph.phrase_program_compile.v1','program':p,'program_sha256':digest,'reference_id':reference_id,'execution_authority':False,'duration_s':8.,'horizon_s':8.14,'transition_s':[3.5,4.5],'timing_changed':False,'heading_transform':transform,'sources':lineage,'source_reference_ids':metadata['provenance']['source_reference_ids'],
        'body_sha256':profile['body_sha256'],'controller_sha256':profile['yaml_sha256'],'continuous_joint_ranges':ranges,'range_numerical_tolerance_rad':1e-8,'strict_future_queries':0,'source_gain_prefix_unchanged_through_s':.5,'original_future_prefix_last_decision':17,'unit_gains_numeric_bypass':all(p[k]==1 for k in GAIN_FIELDS),'new_physics_steps':0,'new_onnx_calls':0,'new_llm_calls':0,
        'curve_semantics':'Product rule exact at unchanged source knots; authoritative resulting common Hermite/SLERP; jointC1,torsoC0 only; no root/contact/SE3/physical guarantee.'}
    violations=[r['joint'] for r in ranges if not r['within_unchanged_range']]
    if violations:
        report.update(status='rejected_joint_ranges',violations=violations)
        raise ProgramCompileError('program reference exceeds unchanged G1 joint ranges: '+','.join(violations),report)
    prefix_equal=True
    for i in range(400):
        inputs=assemble_reference_inputs(reference,i*.02)
        if i<=17:
            original=assemble_reference_inputs(references[p['order'][0]],i*.02)
            prefix_equal=prefix_equal and all(inputs[k].tobytes()==original[k].tobytes() for k in inputs)
    if not prefix_equal:
        report.update(status='rejected_source_prefix',first18_reference_future_feed_bytes_equal=False)
        raise ProgramCompileError('first18 reference future-feed prefix changed',report)
    for key,path in (('body_sha256',profile['body_path']),('yaml_sha256',profile['yaml_path'])):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=profile[key]:raise ValueError('body/controller bytes changed during compilation')
    report.update(status='reference_validated_no_execution_authority',strict_future_queries=400,first18_reference_future_feed_bytes_equal=True,prefix_qualification='Three reference future-feed tensors only; actual eight-feed/current-state/history/physical bridge belongs to caller executor.')
    return reference,report
