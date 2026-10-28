"""Public body-only standing recipe; never loads source motion or historical jobs."""
import json,hashlib,os,zipfile
from pathlib import Path
import numpy as np
from .config_v5 import ordinary,digest
from . import config_v5

def read_identity(config):
    path=ordinary(config['asset_identity'])
    if path.stat().st_size>128*1024 or digest(path)!=config_v5.IDENTITY_SHA256:raise ValueError('Admitted public asset identity changed')
    return json.loads(path.read_bytes())

def owned_q(names):
    if type(names)is not list or len(names)!=29 or any(type(n)is not str for n in names) or len(set(names))!=29:raise ValueError('Exact29 named joints')
    q=np.zeros(29,dtype=np.float64)
    for i,n in enumerate(names):
        if 'hip_pitch' in n:q[i]=-.15
        elif 'knee' in n:q[i]=.30
        elif 'ankle_pitch' in n:q[i]=-.15
        elif 'elbow' in n:q[i]=.30
    return q

def verify_vendor(config):
    identity=read_identity(config)
    if identity['schema']!='pulsemorph.body_asset_identity.v1':raise ValueError('Asset identity schema')
    for row in identity['files']:
        rel=Path(row['path']);p=ordinary(config['vendor_root']/rel)
        if rel.is_absolute() or '..' in rel.parts or not p.resolve().is_relative_to(config['vendor_root']):raise ValueError('Asset escape')
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('Asset changed')
    return identity

def materialize_declared_newlines(config):
    """Explicit preparation only; accepts exact tracked bytes, preserves originals."""
    identity=read_identity(config);changes=[]
    for row in identity['files']:
        p=ordinary(config['vendor_root']/row['path']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()==row['sha256']:continue
        if row['materialization']!='lf_to_crlf' or hashlib.sha256(raw).hexdigest()!=row['tracked_blob_sha256']:raise ValueError('Unknown source bytes; no normalization')
        target=raw.replace(b'\n',b'\r\n')
        if hashlib.sha256(target).hexdigest()!=row['sha256']:raise ValueError('Materialization mismatch')
        # Never modify vendor. Caller must use a separately acquired preparation copy.
        changes.append({'path':row['path'],'required':'LF→CRLF in explicitly owned materialization copy','sha256':row['sha256']})
    return changes

def build_profile(config):
    verify_vendor(config)
    import yaml,xml.etree.ElementTree as ET
    model='data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml'
    cfg=yaml.safe_load((config['vendor_root']/model).read_text());names=cfg['joint_names']
    if len(names)!=29 or len(set(names))!=29:raise ValueError('Exact29 named joints')
    # Owned symmetric standing recipe; no clip, copied array or private cache.
    q=owned_q(names)
    xml=ET.parse(config['vendor_root']/'protomotions/data/assets/mjcf/g1_holo_compat.xml')
    joints={x.get('name'):x for x in xml.iter('joint') if x.get('name')}
    for n,v in zip(names,q):
        bounds=[float(x) for x in joints[n].get('range','').split()]
        if len(bounds)!=2 or not bounds[0]<=v<=bounds[1]:raise ValueError('Owned pose outside XML range')
    directory=ordinary(config['profile_dir']);directory.mkdir(parents=True,exist_ok=False)
    arrays={'qpos':np.r_[0.,0.,.79,1.,0.,0.,0.,q],'qvel':np.zeros(35),'reference_q':q,'reference_dq':np.zeros(29)}
    np.savez_compressed(directory/'standing.npz',**arrays)
    manifest={'schema':'pulsemorph.public_body_profile.v5','recipe':'owned_symmetric_stand_v1','joint_names':names,'quaternion':'qpos WXYZ','root_height_m':.79,'array_sha256':digest(directory/'standing.npz'),'asset_identity_sha256':digest(config['asset_identity']),'builder_sha256':digest(__file__),'physical_admission':False,'qualification':'Reference-only owned initialization; no source choreography, FK, inference or physics; finite native admission required.'}
    with (directory/'manifest.json').open('x')as f:json.dump(manifest,f,indent=2)
    return manifest

def verify_profile(config):
    read_identity(config)
    d=ordinary(config['profile_dir']);p=ordinary(d/'manifest.json')
    if p.stat().st_size>16384:raise ValueError('Profile manifest bound')
    v=json.loads(p.read_bytes())
    expected={'schema','recipe','joint_names','quaternion','root_height_m','array_sha256','asset_identity_sha256','builder_sha256','physical_admission','qualification'}
    if type(v)is not dict or set(v)!=expected or v['schema']!='pulsemorph.public_body_profile.v5' or v['recipe']!='owned_symmetric_stand_v1' or v['physical_admission']is not False or v['quaternion']!='qpos WXYZ' or type(v['root_height_m'])is not float or v['root_height_m']!=.79 or v['asset_identity_sha256']!=digest(config['asset_identity']) or v['builder_sha256']!=digest(__file__) or v['array_sha256']!=digest(ordinary(d/'standing.npz')):raise ValueError('Profile identity changed')
    archive=ordinary(d/'standing.npz')
    if archive.stat().st_size>32768:raise ValueError('Profile NPZ exceeds32KiB')
    with zipfile.ZipFile(archive)as z:
        members=z.infolist()
        if len(members)!=4 or {m.filename for m in members}!={'qpos.npy','qvel.npy','reference_q.npy','reference_dq.npy'} or any(m.file_size>4096 or m.flag_bits&1 for m in members):raise ValueError('Profile decompressed bounds/inventory')
    import yaml
    names=yaml.safe_load((config['vendor_root']/'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml').read_text())['joint_names']
    if v['joint_names']!=names:raise ValueError('Profile joint layout differs')
    expected_q=owned_q(names)
    with np.load(d/'standing.npz',allow_pickle=False)as z:
        if set(z.files)!={'qpos','qvel','reference_q','reference_dq'}:raise ValueError('Exact profile arrays')
        for k,n in [('qpos',36),('qvel',35),('reference_q',29),('reference_dq',29)]:
            if z[k].shape!=(n,) or z[k].dtype!=np.float64 or not np.isfinite(z[k]).all():raise ValueError('Profile numeric layout')
        if not np.array_equal(z['qpos'][7:],expected_q) or not np.array_equal(z['reference_q'],expected_q) or not np.array_equal(z['qvel'],np.zeros(35)) or not np.array_equal(z['reference_dq'],np.zeros(29)) or not np.array_equal(z['qpos'][:7],np.array([0.,0.,.79,1.,0.,0.,0.])):raise ValueError('Owned initialization assignment changed')
    return v
