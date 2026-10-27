"""Materialize/check one fixed phrase; no pickle, ONNX or physics operations."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'artifacts/runs/agentic/phrase_reference_plan_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    from scipy.spatial.transform import Rotation
    from pulsemorph.agentic.phrase_reference import compose_phrase
    from pulsemorph.motion.contracts import load_reference,save_reference
    from pulsemorph.motion.g1 import profile_from_yaml,assemble_reference_inputs,sample_reference
    OUT.mkdir(parents=True,exist_ok=False)
    report={'status':'failed_reference_only','physics_steps':0,'onnx_calls':0,'llm_calls':0,'pickle_loads':0}
    try:
        profile=profile_from_yaml()
        jobs={'wave':'20261027T023422_8efe4bc4','high5':'20261027T022322_76db8965'}
        audit_path=ROOT/'artifacts/runs/agentic/app_validation_v1/jobs_audit_v1.json'
        audit=json.loads(audit_path.read_text())
        pins={str(audit_path.relative_to(ROOT)):sha(audit_path)};refs={};sources=[]
        for source,job in jobs.items():
            trial=ROOT/'artifacts/runs/agentic/app/jobs'/job/'trials/baseline'
            prior=next(x for x in audit['jobs'] if x['id']==job)
            if not prior['passed'] or sha(trial.parents[1]/'state.json')!=prior['state_sha256'] or not any(p['trial']=='baseline' and p['state']=='independently_verified' for p in prior['physics']):
                raise ValueError('archived identity lacks unchanged independent audit')
            manifest=json.loads((trial/'manifest.json').read_text())
            if manifest['status']!='success':raise ValueError('source baseline failed')
            for relative,digest in manifest['artifact_hashes'].items():
                p=trial/relative
                if sha(p)!=digest:raise ValueError('archived source artifact changed: '+relative)
                pins[str(p.relative_to(ROOT))]=digest
            for p in (trial/'manifest.json',trial/'proposal.json',trial.parents[1]/'state.json'):
                pins[str(p.relative_to(ROOT))]=sha(p)
            proposal=json.loads((trial/'proposal.json').read_text())
            if proposal['control_tool']!='global_clock_v1' or any(v!=(0 if k=='timing_shift_s' else 1) for k,v in proposal['parameters'].items()):
                raise ValueError('requires identity common source reference')
            refs[source]=load_reference(trial/'common_reference')
            for filename in ('manifest.json','arrays.npz'):
                p=trial/'common_reference'/filename;sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
        phrase=compose_phrase(refs['wave'],refs['high5'],reference_id='wave_high5_phrase_reference_v1',sources=sources)
        save_reference(phrase,OUT/'common_reference')
        phrase=load_reference(OUT/'common_reference')
        strict=[]
        for i in range(400):
            inputs=assemble_reference_inputs(phrase,i*.02)
            if not all(np.isfinite(x).all() for x in inputs.values()):raise ValueError('nonfinite strict input')
            strict.append(inputs)
        # Both equality and measured differences are retained; failure is not repaired.
        parity=[]
        for i in range(168):
            original=assemble_reference_inputs(refs['wave'],i*.02)
            parity.append({'decision':i,'time_s':i*.02,'inputs':{k:{'byte_equal':original[k].tobytes()==strict[i][k].tobytes(),'max_abs_error':float(np.max(np.abs(original[k]-strict[i][k])))} for k in original}})
        with (OUT/'prefix_input_parity.json').open('x') as f:json.dump(parity,f,indent=2)
        curve=CubicHermiteSpline(phrase.time_offsets_s,phrase.joint_position_rad,phrase.joint_velocity_rad_s,extrapolate=False)
        limits={j.attrib['name']:[float(v) for v in j.attrib['range'].split()] for j in ET.parse(profile['body_path']).getroot().findall('.//joint') if 'name' in j.attrib and 'range' in j.attrib}
        roots=curve.derivative().roots(extrapolate=False);ranges=[]
        for j,name in enumerate(profile['joint_names']):
            points=np.unique(np.r_[phrase.time_offsets_s,roots[j][np.isfinite(roots[j])&(roots[j]>=0)&(roots[j]<=8.14)]])
            q=curve(points)[:,j];lo,hi=limits[name]
            ranges.append({'joint':name,'minimum_rad':float(q.min()),'maximum_rad':float(q.max()),'limit_min_rad':lo,'limit_max_rad':hi,'within_unchanged_range':bool(q.min()>=lo-1e-8 and q.max()<=hi+1e-8)})
        joins=[]
        for t in (3.5,4.5):
            k=int(np.flatnonzero(phrase.time_offsets_s==t)[0]);h=phrase.time_offsets_s[k]-phrase.time_offsets_s[k-1]
            c=curve.c[:,k-1];lv=((c[0]*h+c[1])*h+c[2])*h+c[3];ld=(3*c[0]*h+2*c[1])*h+c[2]
            eps=1e-6;rot=sample_reference(phrase,np.array([t-eps,t,t+eps]))['anchor_rot'];rr=Rotation.from_quat(rot)
            wleft=(rr[1]*rr[0].inv()).as_rotvec()/eps;wright=(rr[2]*rr[1].inv()).as_rotvec()/eps
            joins.append({'time_s':t,'joint_position_join_max_abs_error':float(np.max(np.abs(lv-curve.c[3,k]))),'joint_derivative_join_max_abs_error':float(np.max(np.abs(ld-curve.c[2,k]))),'torso_left_angular_velocity_rad_s':wleft.tolist(),'torso_right_angular_velocity_rad_s':wright.tolist(),'torso_angular_velocity_jump_norm_rad_s':float(np.linalg.norm(wright-wleft)),'orientation_guarantee':'SO3 C0 shortest SLERP; no angular derivative matching/C1 claim'})
        for p in (Path(__file__),ROOT/'src/pulsemorph/agentic/phrase_reference.py',ROOT/'src/pulsemorph/motion/g1.py',ROOT/'src/pulsemorph/motion/contracts.py',Path(profile['yaml_path']),Path(profile['body_path'])):
            pins[str(p.relative_to(ROOT))]=sha(p)
        good_parity=all(v['byte_equal'] for x in parity for v in x['inputs'].values())
        report.update(source_pins=pins,source_packages=sources,strict_future_queries=400,last_query_time_s=7.98+.16,knots=len(phrase.time_offsets_s),horizon_s=float(phrase.time_offsets_s[-1]),continuous_joint_ranges=ranges,joins=joins,prefix_decisions=168,prefix_byte_parity=good_parity,
            phrase_manifest_sha256=sha(OUT/'common_reference/manifest.json'),phrase_arrays_sha256=sha(OUT/'common_reference/arrays.npz'),
            schedule={'wave_source_s':[0,3.5],'bridge_execution_s':[3.5,4.5],'high5_clock':'t-4 for4.5..8.14','no_extrapolation_or_padding':True},
            prospective_execution={'authority':'NONE; reference preparation only','maximum_trials':1,'maximum_control_decisions':400,'maximum_physics_steps':8000,'audio':False,'llm':False,'required_new_adapter':'29q/dq/torso-only future sampler; reset from pinned archived Wave initial qpos/qvel. Do not invent full-body carriers/root motion. Frozen 4s app executor cannot be silently extended.',
                'mandatory_in_loop_prefix_gate':'After control167 physics and BEFORE decision168, require unchanged Wave baseline initial state + states0..168, per-key inputs/outputs controls0..167 and appliedPD0..167 exact; fail stops remaining trial immediately. Reference feed parity alone does not satisfy this.',
                'gate_feasibility':'A new bounded executor loop must intercept the boundary; posthoc comparison alone cannot enforce stop-before-continuation. New executor/control-reset/numeric feed proof and root review required before any call.',
                'screen':'Current unchanged finite-state/height/tracking/integrity screens, separately from transition appearance; not a full-goal or safety claim.'})
        good_joins=all(j['joint_position_join_max_abs_error']<1e-10 and j['joint_derivative_join_max_abs_error']<1e-10 for j in joins)
        report['reference_checks_passed']=good_parity and good_joins and all(x['within_unchanged_range'] for x in ranges)
        report['status']='reference_only_passed_awaiting_design_review' if report['reference_checks_passed'] else 'reference_checks_failed_no_controller_calls'
    except Exception as e:report['error']=str(e)
    with (OUT/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','error','reference_checks_passed','prefix_byte_parity','knots')}))


if __name__=='__main__':main()
