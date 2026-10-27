"""One private ordinary-copy phrase profile, never upstream acquisition proof."""
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    from pulsemorph.agentic.app.phrase.assets import ASSET_ROOT,REFERENCE_HASHES,MODEL,MODEL_SHA,sha,write_new
    ASSET_ROOT.mkdir(parents=True,exist_ok=False)
    provenance=[];files={}
    for name,job in {'wave':'20261027T023422_8efe4bc4','high5':'20261027T022322_76db8965'}.items():
        source=ROOT/'artifacts/runs/agentic/app/jobs'/job/'trials/baseline/common_reference'
        (ASSET_ROOT/name).mkdir()
        for filename,digest in REFERENCE_HASHES[name].items():
            if sha(source/filename)!=digest:raise ValueError('source common reference changed')
            shutil.copyfile(source/filename,ASSET_ROOT/name/filename)
            files[f'{name}/{filename}']=sha(ASSET_ROOT/name/filename)
            provenance.append({'path':str(source/filename),'sha256':digest,'scope':'historical origin only, not runtime dependency'})
    base=ROOT/'artifacts/runs/agentic/app/jobs/20261027T023422_8efe4bc4/trials/baseline'
    with np.load(base/'reference_cache.npz',allow_pickle=False) as c:
        np.savez_compressed(ASSET_ROOT/'initial_carrier.npz',**{k:c['committed__'+k][0] for k in ('dof_pos','dof_vel','body_pos','body_rot','body_vel','body_ang_vel')})
    prefix={}
    for group in ('inputs','outputs'):
        with np.load(base/f'policy_{group}.npz',allow_pickle=False) as z:
            prefix.update({group+'__'+k:z[k] for k in z.files if int(k.split('__')[0].split('_')[1])<=18})
    for key,filename,count in (('qpos','qpos.npy',19),('qvel','qvel.npy',19),('time','state_times.npy',19),('PD','applied_joint_targets.npy',18)):
        prefix[key]=np.load(base/filename,allow_pickle=False)[:count]
    np.savez_compressed(ASSET_ROOT/'golden_prefix.npz',**prefix)
    for name in ('initial_carrier.npz','golden_prefix.npz'):files[name]=sha(ASSET_ROOT/name)
    for filename in ('reference_cache.npz','policy_inputs.npz','policy_outputs.npz','qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy','manifest.json'):
        provenance.append({'path':str(base/filename),'sha256':sha(base/filename),'scope':'historical source for extracted bounded safe arrays, no runtime read'})
    relative=['src/pulsemorph/agentic/phrase_program.py','src/pulsemorph/agentic/phrase_reference.py','src/pulsemorph/agentic/phrase_reference_v2.py','src/pulsemorph/motion/contracts.py','src/pulsemorph/motion/g1.py','experiments/agentic/app/execute_app.py',MODEL,MODEL.replace('.onnx','.yaml'),'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml']
    relative += ['research/vendor/ProtoMotions/deployment/'+n for n in ('test_tracker_mujoco.py','tracker_inputs.py','state_utils.py','motion_utils.py')]
    relative += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'research/vendor/ProtoMotions/protomotions/data/assets/mesh/G1').glob('*.stl'))]
    pins={p:sha(ROOT/p) for p in relative}
    if pins[MODEL]!=MODEL_SHA:raise ValueError('model pin differs')
    manifest={'schema':'pulsemorph.private_phrase_assets.v1','files':files,'historical_origins':provenance,'live_numeric_pins':pins,'upstream_commit':'f69fdcaabff31846b4d2247207417b57dc788828','source_motion_sha256':{'wave':'cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93','high5':'ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44'},'scope':'Private local ordinary prepared profile, not clean upstream acquisition. No pickle/full history/checkpoints/audio copied. Historical paths are provenance only.','prepare_source_sha256':sha(__file__),'new_physics_steps':0,'new_onnx_calls':0}
    write_new(ASSET_ROOT/'manifest.json',manifest)
    print(json.dumps({'manifest_sha256':sha(ASSET_ROOT/'manifest.json'),'files':len(files),'bytes':sum((ASSET_ROOT/f).stat().st_size for f in files),'path':str(ASSET_ROOT),'physics':0,'onnx':0}))


if __name__=='__main__':main()
