"""One High5→Wave aligned reference; no controller/model/physics operations."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
OUT=ROOT/'artifacts/runs/agentic/phrase_reference_plan_v2'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    from scipy.spatial.transform import Rotation
    from pulsemorph.agentic.phrase_reference_v2 import compose_ordered_pair
    from pulsemorph.motion.contracts import load_reference,save_reference
    from pulsemorph.motion.g1 import profile_from_yaml,assemble_reference_inputs,sample_reference
    from deployment.tracker_inputs import init_tracker_alignment,align_future_motion_refs,build_tracker_onnx_inputs
    OUT.mkdir(parents=True,exist_ok=False)
    report={'status':'failed_reference_only','physics_steps':0,'onnx_calls':0,'llm_calls':0,'pickle_loads':0}
    try:
        v1plan=ROOT/'artifacts/runs/agentic/phrase_reference_plan_v1'
        old=json.loads((v1plan/'report.json').read_text());pins={}
        for p,h in old['source_pins'].items():
            path=ROOT/p
            if sha(path)!=h:raise ValueError('frozenv1 source changed: '+p)
            pins[str(path.relative_to(ROOT))]=h
        audit=ROOT/'artifacts/runs/agentic/phrase_execution_audit_v1/receipt.json'
        verdict=json.loads(audit.read_text())
        if not verdict['verdict'].startswith('PASS') or verdict['all307_pins_match'] is not True:raise ValueError('v1 physical audit missing')
        refs={};sources=[]
        for name,job in {'wave':'20261027T023422_8efe4bc4','high5':'20261027T022322_76db8965'}.items():
            directory=ROOT/'artifacts/runs/agentic/app/jobs'/job/'trials/baseline'
            refs[name]=load_reference(directory/'common_reference')
            for n in ('manifest.json','arrays.npz'):
                p=directory/'common_reference'/n;sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
        regression,_=compose_ordered_pair(refs,order=('wave','high5'),heading_policy='recorded_world',reference_id='regression_not_materialized',sources=sources)
        original=load_reference(v1plan/'common_reference')
        equality={n:getattr(original,n).tobytes()==getattr(regression,n).tobytes() for n in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw')}
        if not all(equality.values()):raise ValueError('v1 recorded-world numeric regression failed')
        phrase,transform=compose_ordered_pair(refs,order=('high5','wave'),heading_policy='align_join_heading',reference_id='high5_wave_aligned_phrase_v2',sources=sources)
        save_reference(phrase,OUT/'common_reference');phrase=load_reference(OUT/'common_reference')
        profile=profile_from_yaml();strict=[]
        for i in range(400):strict.append(assemble_reference_inputs(phrase,i*.02))
        curve=CubicHermiteSpline(phrase.time_offsets_s,phrase.joint_position_rad,phrase.joint_velocity_rad_s,extrapolate=False)
        limits={j.attrib['name']:[float(v) for v in j.attrib['range'].split()] for j in ET.parse(profile['body_path']).getroot().findall('.//joint') if 'name' in j.attrib and 'range' in j.attrib}
        roots=curve.derivative().roots(extrapolate=False);ranges=[]
        for j,name in enumerate(profile['joint_names']):
            points=np.unique(np.r_[phrase.time_offsets_s,roots[j][np.isfinite(roots[j])&(roots[j]>=0)&(roots[j]<=8.14)]])
            q=curve(points)[:,j];lo,hi=limits[name]
            ranges.append({'joint':name,'minimum_rad':float(q.min()),'maximum_rad':float(q.max()),'limit_min_rad':lo,'limit_max_rad':hi,'within_unchanged_range':bool(q.min()>=lo-1e-8 and q.max()<=hi+1e-8)})
        base=ROOT/'artifacts/runs/agentic/app/jobs/20261027T022322_76db8965/trials/baseline'
        with np.load(base/'reference_cache.npz',allow_pickle=False) as cache:
            initial={k:cache['committed__'+k][0].copy() for k in ('dof_pos','body_rot','body_pos')}
        import yaml
        name_map=yaml.safe_load(Path(profile['yaml_path']).read_text())['_runtime']['onnx_name_to_in_key']
        names=tuple(name_map);rows=[]
        with np.load(base/'policy_inputs.npz',allow_pickle=False) as inputs:
            alignment=init_tracker_alignment(robot_anchor_pos=initial['body_pos'][profile['anchor_body_index']],robot_anchor_rot=inputs['call_0001__current_anchor_rot'][0],motion_anchor_pos=initial['body_pos'][profile['anchor_body_index']],motion_anchor_rot=initial['body_rot'][profile['anchor_body_index']])
            for i in range(168):
                prefix=f'call_{i+1:04d}__';states=sample_reference(phrase,np.array([(i+j)*.02 for j in (1,2,4,8)]))
                auxiliary=np.broadcast_to(initial['body_rot'],(4,33,4)).copy();auxiliary[:,profile['anchor_body_index']]=states['anchor_rot'].astype('float32')
                future={'dof_pos':states['dof_pos'].astype('float32'),'dof_vel':states['dof_vel'].astype('float32'),'body_rot':auxiliary}
                feed=build_tracker_onnx_inputs(onnx_input_names=list(names),onnx_name_to_key=name_map,dof_pos=inputs[prefix+'current_dof_pos'][0],dof_vel=inputs[prefix+'current_dof_vel'][0],anchor_rot=inputs[prefix+'current_anchor_rot'][0],root_local_ang_vel=inputs[prefix+'current_root_local_ang_vel'][0],future_refs=align_future_motion_refs(future,alignment),anchor_body_index=profile['anchor_body_index'],num_dofs=29,prev_actions=inputs[prefix+'historical_processed_actions'][0,0])
                rows.append({'decision':i,'time_s':i*.02,'inputs':{n:{'byte_equal':feed[n].tobytes()==inputs[prefix+n].tobytes(),'max_abs_error':float(np.max(np.abs(feed[n]-inputs[prefix+n])))} for n in names}})
        with (OUT/'prefix_actual_assembly_parity.json').open('x') as f:json.dump(rows,f,indent=2)
        joins=[]
        for t in (3.5,4.5):
            k=int(np.flatnonzero(phrase.time_offsets_s==t)[0]);h=phrase.time_offsets_s[k]-phrase.time_offsets_s[k-1];c=curve.c[:,k-1]
            lv=((c[0]*h+c[1])*h+c[2])*h+c[3];ld=(3*c[0]*h+2*c[1])*h+c[2]
            eps=1e-6;rot=Rotation.from_quat(sample_reference(phrase,np.array([t-eps,t,t+eps]))['anchor_rot'])
            wl=(rot[1]*rot[0].inv()).as_rotvec()/eps;wr=(rot[2]*rot[1].inv()).as_rotvec()/eps
            joins.append({'time_s':t,'joint_position_residual_max':float(np.max(np.abs(lv-curve.c[3,k]))),'joint_velocity_residual_max':float(np.max(np.abs(ld-curve.c[2,k]))),'torso_angular_velocity_jump_rad_s':float(np.linalg.norm(wr-wl)),'left_angular_velocity_rad_s':wl.tolist(),'right_angular_velocity_rad_s':wr.tolist(),'torso_guarantee':'C0 only; no angular derivative matching'})
        target=sample_reference(refs['high5'],np.array([3.5]))['anchor_rot'][0]
        raw=sample_reference(refs['wave'],np.array([.5]))['anchor_rot'][0]
        aligned=sample_reference(phrase,np.array([4.5]))['anchor_rot'][0]
        angles={'untransformed_endpoint_SO3_difference_rad':float((Rotation.from_quat(target)*Rotation.from_quat(raw).inv()).magnitude()),'aligned_endpoint_SO3_difference_rad':float((Rotation.from_quat(target)*Rotation.from_quat(aligned).inv()).magnitude()),'qualification':'descriptive endpoint orientation differences, not physical preference/tracking evidence'}
        extras=[Path(__file__),ROOT/'src/pulsemorph/agentic/phrase_reference_v2.py',ROOT/'research/vendor/ProtoMotions/deployment/state_utils.py',ROOT/'research/vendor/ProtoMotions/deployment/tracker_inputs.py',v1plan/'report.json',v1plan/'common_reference/manifest.json',v1plan/'common_reference/arrays.npz',audit,ROOT/'artifacts/runs/agentic/phrase_execution_audit_v1/REVIEW.md']
        for path in extras:pins[str(path.relative_to(ROOT))]=sha(path)
        passed=all(x['within_unchanged_range'] for x in ranges) and all(x['byte_equal'] for row in rows for x in row['inputs'].values()) and all(j['joint_position_residual_max']<1e-10 and j['joint_velocity_residual_max']<1e-10 for j in joins)
        report.update(status='reference_only_passed_no_execution_authority' if passed else 'failed_reference_checks_no_controller_calls',reference_checks_passed=passed,source_pins=pins,ordered_pair=['high5','wave'],heading_transform=transform,v1_numeric_regression_byte_equal=equality,materialized_cases=1,strict_future_queries=400,actual_assembled_prefix_rows=168,actual_assembled_prefix_all_bytes_equal=all(x['byte_equal'] for row in rows for x in row['inputs'].values()),knots=len(phrase.time_offsets_s),horizon_s=8.14,continuous_joint_ranges=ranges,joins=joins,endpoint_orientation_readout=angles,phrase_manifest_sha256=sha(OUT/'common_reference/manifest.json'),phrase_arrays_sha256=sha(OUT/'common_reference/arrays.npz'),limitations=['Torso C0, not C1; no root/contact/fullbody/SE3 registration','No v2 executor or physical/model calls, app/catalog admission or learned general composition','Unassigned corpus data remains NOT_ADMITTED'])
    except Exception as e:report['error']=str(e)
    with (OUT/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','error','reference_checks_passed','actual_assembled_prefix_all_bytes_equal','endpoint_orientation_readout')}))


if __name__=='__main__':main()
