"""Static, hash-pinned inventory only; no policy, physics, network or training."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/agentic'))
from execute_generalization import resolve_source_motion

PINNED = {
    'dance': '934309f9357dbb833b7d68411c6c41fd6e31e14d792d5eca43952a9c5f28f784',
    'high5': 'ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44',
    'jumpjack': '5349a1acc3b2f8cf06d89ae6080dc957591a512223cbc43132d5ab7a4d1f89ef',
    'walk': '09061b8fa0df9e99daaf857293b32cfa8d2faa68631ff20ed0fdde0683e80884',
    'wave': 'cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93',
}

def main():
    import numpy as np
    import yaml
    import torch
    sys.path.insert(0, str(ROOT / 'research/vendor/ProtoMotions'))
    sys.path.insert(0, str(ROOT / 'src'))
    from pulsemorph.agentic.reference import _causal
    metadata_path = ROOT / 'research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml'
    robot = yaml.safe_load(metadata_path.read_text())['robot']
    rows = []
    for name, digest in PINNED.items():
        relative = f'research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_{name}.motion'
        source, _ = resolve_source_motion({'source_motion': {'path': relative, 'sha256': digest}})
        # Fixed five names AND expected hashes, checked by the existing resolver.
        raw = torch.load(source, map_location='cpu', weights_only=False)
        q = np.asarray(raw['dof_pos'], dtype=float)
        fps = float(raw['fps'])
        t = np.arange(len(q)) / fps
        interpolator = _causal.SourceInterpolator(raw)
        try:
            interpolator.state(4.14, 1.)
            supported, rejection = True, None
        except ValueError as exc:
            supported, rejection = False, str(exc)
        rms = {}
        for window, (a, b) in {'a': (.8, 1.6), 'b': (2.4, 3.2)}.items():
            mask = (t >= a-1e-12) & (t <= b+1e-12)
            for side, indices in [('left', slice(15,22)), ('right', slice(22,29))]:
                values = q[mask, indices]
                rms[f'{side}_{window}'] = {
                    'displacement_rms_rad': float(np.sqrt(np.mean((values-q[0,indices])**2))),
                    'velocity_rms_rad_s': float(np.sqrt(np.mean(np.diff(values,axis=0)**2)) * fps),
                    'sample_count': len(values),
                }
        rows.append({'name': name, 'path': relative, 'sha256': digest,
            'raw_keys': sorted(raw), 'frames': len(q), 'fps': fps, 'timestep_s': 1/fps,
            'duration_s': float(t[-1]), 'dof_pos_shape': list(q.shape),
            'rigid_body_pos_shape': list(np.asarray(raw['rigid_body_pos']).shape),
            'rigid_body_rot_shape': list(np.asarray(raw['rigid_body_rot']).shape),
            'finite_joint_positions': bool(np.isfinite(q).all()),
            'profile_dimensions_match': bool(q.shape[1] == len(robot['joint_names']) and np.asarray(raw['rigid_body_pos']).shape[1] == len(robot['body_names'])),
            'supports_4p14_s': supported, 'unsupported_time_error': rejection,
            'agentic_development_used': name in ('dance','high5','wave'),
            'arm_windows': rms})
    output = ROOT / 'artifacts/runs/agentic/tool_source_inventory_v1'
    output.mkdir(parents=True, exist_ok=False)
    inventory = {'scope': 'static source inspection only; no physics/model invocation',
        'required_last_source_time_s': 4.14, 'clamp_or_repeat': False,
        'profile_path': str(metadata_path.relative_to(ROOT)),
        'profile_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'joint_names': robot['joint_names'], 'body_names': robot['body_names'],
        'profile_name_assertion': 'Names supplied by pinned G1 deployment metadata; dimensions are checked, raw file name semantics are not independently proven.',
        'sources': rows, 'new_eligible_source_count': sum(r['supports_4p14_s'] and not r['agentic_development_used'] for r in rows),
        'required_new_sources': 4, 'main_panel_status': 'blocked_insufficient_new_sources'}
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'output': str(output), 'new_eligible_source_count': inventory['new_eligible_source_count'], 'main_panel_status': inventory['main_panel_status']}))

if __name__ == '__main__':
    main()
