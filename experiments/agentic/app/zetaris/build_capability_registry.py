"""Build two audited shared capability records; no user audio/prompts or remote writes."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
COLUMNS=['source_id','title','body_id','capability','evidence_class','license_scope']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def evidence(path,purpose):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'purpose':purpose}

def build(output):
    if output.exists():raise FileExistsError('Registry output version already exists')
    sys.path.insert(0,str(ROOT/'src'))
    from pulsemorph.agentic.app.integrations import CAPABILITY_LIMITS
    from pulsemorph.agentic.app.runtime import PINS,PARAMETERS,TOOLS
    import numpy as np
    spec=importlib.util.spec_from_file_location('registry_features',ROOT/'experiments/agentic/evaluate_multifeature.py')
    features=importlib.util.module_from_spec(spec);spec.loader.exec_module(features)
    public=ROOT/'artifacts/runs/agentic/app_public_environment_v1'
    wave_audit=read(public/'physics_independent_audit.json');assert wave_audit['passed']
    high5_global_path=ROOT/'artifacts/runs/agentic/high5_capability_v1_audit_v1/audit.json'
    high5_local_path=ROOT/'artifacts/runs/agentic/high5_local_capability_v1_audit_v2/audit.json'
    global_audit,local_audit=read(high5_global_path),read(high5_local_path)
    assert global_audit['audit_passed'] and local_audit['audit_passed']
    assert local_audit['status']=='confirmed_witness' and local_audit['witness']['integrity_passed']
    model=ROOT/'research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'
    assert sha(model)==wave_audit['controller_onnx_sha256']
    shared_evidence=[evidence(ROOT/'src/pulsemorph/agentic/app/runtime.py','current immutable source/tool allowlist and evaluator configuration'),
                     evidence(ROOT/'experiments/agentic/app/execute_app.py','finite parameter validation and numeric executor'),
                     evidence(ROOT/'src/pulsemorph/agentic/multifeature_reference.py','global clock and section arm envelope'),
                     evidence(ROOT/'src/pulsemorph/agentic/local_multifeature_reference.py','fixed local clock window'),
                     evidence(ROOT/'experiments/agentic/evaluate_multifeature.py','observed joint/accent feature implementation')]
    velocities=np.load(public/'physics_baseline/qvel.npy',allow_pickle=False)
    times=np.load(public/'physics_baseline/state_times.npy',allow_pickle=False)
    accent=features.accent_metrics(velocities[:,6:],times,[1.8,3.2])
    rows=[]
    for source in ['wave','high5']:
        motion=ROOT/f'research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_{source}.motion'
        assert sha(motion)==PINS[source]
        capability={'schema':'pulsemorph.shared_source_capability.v1','registry_version':'1',
            'scope':'Shared cold-start capability guide; not user memory, planner authority, bounds/admission authority or rights approval.',
            'source':{'id':source,'repository_relative_path':motion.relative_to(ROOT).as_posix(),'sha256':PINS[source]},
            'body_profile_id':'g1','duration_s':4,'controller':{'onnx_sha256':sha(model),'fixed_pretrained':True,'control_hz':50,'physics_hz':1000,'pd_mode':'BUILT_IN_PD'},
            'parameter_order':PARAMETERS,'parameter_bounds':{name:[-.3,.3] if name=='timing_shift_s' else [.35,1.6] for name in PARAMETERS},
            'tools':list(TOOLS),'arm_envelope_windows_s':[[.4,2.],[2.,3.6]],'observed_arm_windows_s':[[.8,1.6],[2.4,3.2]],
            'observed_accent_window_s':[1.8,3.2],
            'timing_semantics':{'positive_shift':'Advances sampled source time; usually earlier events, not an additive delay.',
                'global_clock_v1':'t + shift*sin(pi*t/4)^2',
                'local_clock_1p4_2p8_v1':'Same bounded warp inside [1.4,2.8] only; source clock unchanged outside.',
                'caution':'Dominant observed whole-body joint-speed peak can switch events; target gains are not observed arm ratios.'},
            'limitations':['Only two pinned four-second existing G1 references; no new learned weights or skills.',
                'Joint proxies and local physical gates are not aesthetic/human-preference or hardware/recovery certification.',
                'Each new request needs its own baseline, fixed goals and fresh measured physical acceptance.'],
            'evidence':list(shared_evidence)}
        if source=='wave':
            capability['known_local_measurement']={'class':'no_audio_identity_baseline',
                'observed_accent_s':accent['peak_time_s'],'min_root_height_m':wave_audit['min_root_height_m'],
                'mean_joint_rmse_rad':wave_audit['mean_joint_rmse_rad'],'control_rows':200,'physics_steps':4000}
            capability['evidence'] += [evidence(public/'physics_independent_audit.json','audited no-audio wave baseline'),
                evidence(public/'physics_baseline/qvel.npy','recorded actual joint velocity for accent'),
                evidence(public/'physics_baseline/state_times.npy','recorded 50Hz actual state clock')]
        else:
            witness=local_audit['witness'];actual=witness['features'];errors=witness['feature_errors']
            capability['known_local_measurement']={'class':'separate_fixed_protocol_exploratory_feasibility_not_current_user_goal',
                'calibration_accent_s':local_audit['trials']['calibration']['features'][4],
                'global_clock_search_status':global_audit['status'],'global_best_observed_features':global_audit['best_new']['features'],
                'local_clock_search_status':local_audit['status'],'local_witness_candidate':witness['candidate_id'],
                'local_witness_parameters':dict(zip(PARAMETERS,witness['parameters'])),
                'local_witness_features':actual,'specific_protocol_targets':[a-e for a,e in zip(actual,errors)],
                'specific_protocol_tolerances':witness['feature_tolerances'],'min_root_height_m':witness['min_root_height_m'],
                'mean_joint_rmse_rad':witness['mean_joint_rmse_rad'],
                'limitation':'One bounded exploratory witness; not a new matched planner comparison, generalization guarantee or reusable target for another soundtrack.'}
            capability['evidence'] += [evidence(high5_global_path,'audited global-clock finite-budget no-witness result'),
                                      evidence(high5_local_path,'audited local-clock specific-goal witness')]
        text=json.dumps(capability,separators=(',',':'),ensure_ascii=False,allow_nan=False)
        assert len(text)<=CAPABILITY_LIMITS['capability']
        rows.append({'source_id':source,'title':f'{source} / G1 bounded reference tools','body_id':'g1','capability':text,
            'evidence_class':'local_audited_capability_metadata',
            'license_scope':'Authored lab metadata only. Source code, pretrained model, motion clips and robot meshes retain separate terms; no redistribution or rights approval. See design/agentic/APP_COMPONENT_PROVENANCE.md.'})
    output.mkdir(parents=True)
    with (output/'capabilities.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'PulseMorph shared capability registry row v1','type':'object',
            'properties':{key:{'type':'string','minLength':1,'maxLength':CAPABILITY_LIMITS[key]} for key in COLUMNS},
            'required':COLUMNS,'additionalProperties':False}
    (output/'schema.json').write_text(json.dumps(schema,indent=2))
    (output/'records.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    csv_rows=list(csv.DictReader((output/'capabilities.csv').open(encoding='utf-8',newline='')));assert csv_rows==rows
    import jsonschema
    for row in rows:jsonschema.validate(row,schema);json.loads(row['capability'])
    inputs={e['path']:e['sha256'] for row in rows for e in json.loads(row['capability'])['evidence']}
    manifest={'schema':'pulsemorph.shared_capability_registry_build.v1','rows':len(rows),'columns':COLUMNS,
        'contains_private_audio_or_raw_prompts':False,'real_remote_query_executed':False,'remote_table_created':False,
        'csv_roundtrip_verified':True,'row_schema_verified':True,'capability_lengths':[len(row['capability']) for row in rows],
        'input_evidence_sha256':inputs,'builder_sha256':sha(Path(__file__)),
        'outputs_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'output':str(output),'rows':len(rows),'capability_lengths':manifest['capability_lengths']}))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path(__file__).parent/'registry_v1');args=parser.parse_args();build(args.output.resolve())
if __name__=='__main__':main()
