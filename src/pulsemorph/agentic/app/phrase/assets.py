"""Immutable private local phrase profile; safe JSON/NPZ, no historical loaders."""
from pathlib import Path
import hashlib
import json
import numpy as np
from pulsemorph.motion.contracts import load_reference

ASSET_ROOT=Path('C:/Users/halke/.cache/pulsemorph-agentic/phrase-assets-v1')
MANIFEST_SHA256='3657eafcb7846fab0bccd16ba59c3e18aa1475fc394c4296e4ae8dab4117b622'
REFERENCE_HASHES={
 'wave':{'manifest.json':'1a45ddb3e3d39a759d830d7cd3224ce026ec65457436f73ea8db76979f547acc','arrays.npz':'bcbced6d1eca5d46b51db04cce5bb1c4bedb54b89aea74c614d62e9c043f5672'},
 'high5':{'manifest.json':'a56ff6c90ccfc0035318be597d11b8f18d58e8991af2c6f1e8ebfab8415bf36a','arrays.npz':'890baf6195cf3155c35ada53a8f788b8a438c1438681b0f3e2080e2834eaa30b'}}
MODEL='research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'
MODEL_SHA='a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c'
BODY_SHA='e3926a6578cafaee38d0a7a3b7e7ebb0031f623c69e5cd34aa97d61e943b7e8e'
CONTROLLER_SHA='57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139'
PUBLIC_PYTHON=Path('C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe')
INPUTS=('current_anchor_rot','current_dof_pos','current_dof_vel','current_root_local_ang_vel','historical_processed_actions','mimic_future_anchor_rot','mimic_future_dof_pos','mimic_future_dof_vel')
OUTPUTS=('actions','joint_pos_targets','stiffness_targets','damping_targets')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_new(path,value):
    import os
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


class PhraseAssets:
    def __init__(self,root,directory=ASSET_ROOT):
        self.root=Path(root).resolve();self.directory=Path(directory).resolve()
        self.verify()

    def verify(self):
        path=self.directory/'manifest.json'
        if not MANIFEST_SHA256 or path.is_symlink() or path.stat().st_size>128*1024 or sha(path)!=MANIFEST_SHA256:raise ValueError('phrase asset manifest missing or not admitted')
        manifest=json.loads(path.read_text(encoding='utf8'))
        expected={f'{n}/{f}' for n in ('wave','high5') for f in ('manifest.json','arrays.npz')}|{'initial_carrier.npz','golden_prefix.npz'}
        if manifest['schema']!='pulsemorph.private_phrase_assets.v1' or set(manifest['files'])!=expected:raise ValueError('unexpected phrase asset inventory')
        for relative,digest in manifest['files'].items():
            p=self.directory/relative
            if p.is_symlink() or not p.resolve().is_relative_to(self.directory) or p.stat().st_size>2*1024**2 or sha(p)!=digest:raise ValueError('phrase asset changed: '+relative)
        for name,files in REFERENCE_HASHES.items():
            for filename,digest in files.items():
                if manifest['files'][f'{name}/{filename}']!=digest:raise ValueError('unadmitted source package')
        for relative,digest in manifest['live_numeric_pins'].items():
            if not (self.root/relative).resolve().is_relative_to(self.root) or sha(self.root/relative)!=digest:raise ValueError('live numeric dependency changed: '+relative)
        if manifest['live_numeric_pins'].get(MODEL)!=MODEL_SHA:raise ValueError('unadmitted model')
        self.manifest=manifest
        return manifest

    def references(self):return {n:load_reference(self.directory/n) for n in ('wave','high5')}

    def arrays(self,name):
        if name not in ('initial_carrier','golden_prefix'):raise ValueError('unknown phrase array bundle')
        self.verify()
        with np.load(self.directory/(name+'.npz'),allow_pickle=False) as z:
            values={k:z[k].copy() for k in z.files}
        if not all(v.dtype.kind in 'fiu' and np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite/non-numeric safe asset arrays')
        return values

    def source_records(self):
        # Runtime reads cache only. Original paths in manifest are provenance,
        # deliberately not dereferenced or represented as currently available.
        return [{'path':str(self.directory/p),'sha256':h} for p,h in self.manifest['files'].items()]
