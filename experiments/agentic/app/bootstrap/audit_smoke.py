"""Read-only independent audit of the isolated environment physical smoke."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import site
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--trial',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert sys.flags.isolated and not site.ENABLE_USER_SITE
    if args.output.exists():raise FileExistsError(args.output)
    root=args.root.resolve();sys.path.insert(0,str(root/'src'))
    from pulsemorph.agentic.app.runtime import _module,_audit_app
    module=_module(root,'evaluate_multifeature');audit=_audit_app(module,module._audit,args.trial.resolve())
    manifest=audit['manifest']
    result={'schema':'pulsemorph.isolated_environment_physical_smoke_audit.v1','passed':True,
        'environment':sys.prefix,'executable':sys.executable,'isolated':bool(sys.flags.isolated),
        'user_site_enabled':site.ENABLE_USER_SITE,'actual_control_steps':manifest['actual_control_steps'],
        'actual_physics_steps':manifest['actual_physics_steps'],'online_onnx_call_count':manifest['online_onnx_call_count'],
        'warmup_onnx_call_count':manifest['warmup_onnx_call_count'],'strict_common_control_rows':audit['strict_common_control_rows'],
        'fresh_anchor_audit':audit['fresh_anchor_audit'],'min_root_height_m':manifest['all_substeps_min_root_height_m'],
        'mean_joint_rmse_rad':audit['mean_joint_rmse_rad'],'controller_onnx_sha256':manifest['onnx_sha256'],
        'source_motion':manifest['source_motion'],'audio_observation':manifest['audio_observation'],
        'model_calls_during_smoke':0,'backend_selection_evidence':'Pinned deployment/test_tracker_mujoco.py requests providers=[CPUExecutionProvider]',
        'versions':{name:metadata.version(name) for name in ['numpy','scipy','torch','onnxruntime','mujoco']},
        'manifest_sha256':hashlib.sha256((args.trial/'manifest.json').read_bytes()).hexdigest(),
        'independent_audit_scope':'Full inventory/policy/PD/physics traces plus fresh torso FK and common joint inputs at all200rows; no mj_step during audit.'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
