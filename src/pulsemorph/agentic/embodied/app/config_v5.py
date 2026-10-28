"""Portable, explicit paths and per-install admission. No native construction."""
import hashlib,json,os,re,subprocess
from pathlib import Path

SCHEMA='pulsemorph.body_config.v5'
IDENTITY_SHA256='19c0a28c9455ba096f8bf2c90a65cce55a9ac2c3a037fb336d3406ac36f2cfce'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ordinary(path):
    p=Path(path).absolute()
    for x in (p,*p.parents):
        if x.is_symlink() or x.exists() and getattr(x.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked path forbidden')
    return p
def load_config(path):
    path=ordinary(path)
    if path.stat().st_size>16384:raise ValueError('Config exceeds16KiB')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('Duplicate config key')
            d[k]=v
        return d
    v=json.loads(path.read_bytes(),object_pairs_hook=pairs)
    required={'schema','runtime_python','vendor_root','profile_dir','artifact_root','asset_identity','port'}
    if type(v)is not dict or set(v)!=required or v['schema']!=SCHEMA:raise ValueError('ExactV5config required')
    if type(v['port'])is not int or not 1024<=v['port']<=65535:raise ValueError('Loopback port1024..65535')
    for k in required-{'schema','port'}:
        if type(v[k])is not str or not v[k] or len(v[k])>4096:raise ValueError('Path string required')
        p=Path(v[k]);v[k]=ordinary(p if p.is_absolute() else path.parent/p).resolve()
    if digest(v['asset_identity'])!=IDENTITY_SHA256:raise ValueError('Public asset identity changed')
    if v['artifact_root']==v['vendor_root'] or v['artifact_root'].is_relative_to(v['vendor_root']):raise ValueError('Artifacts cannot overwrite vendor')
    return v

VERSIONS={'numpy':'1.26.4','scipy':'1.16.3','mujoco':'3.3.7','onnxruntime':'1.31.0','PyYAML':'6.0.3','Pillow':'13.0.0','imageio':'2.38.1','imageio-ffmpeg':'0.6.0','jsonschema':'4.25.1','torch':'2.7.1'}
def probe_environment(python):
    script="""import sys,json,site,importlib.metadata as m,importlib.util as u,hashlib,pathlib
names=['numpy','scipy','mujoco','onnxruntime','yaml','PIL','torch'];pins={}
for name in names:
 s=u.find_spec(name)
 if s is None or not s.origin:raise RuntimeError('Modulemissing')
 p=pathlib.Path(s.origin).resolve();files=[p]+sorted(p.parent.rglob('*.pyd'))+sorted(p.parent.rglob('*.so'))
 pins[name]={'origin':str(p),'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
print(json.dumps({'python':list(sys.version_info[:3]),'executable':sys.executable,'isolated':sys.flags.isolated,'user_site':site.ENABLE_USER_SITE,'versions':{k:m.version(k) for k in %r},'module_pins':pins}))
""" % list(VERSIONS)
    p=subprocess.run([str(ordinary(python)),'-I','-B','-c',script],capture_output=True,timeout=30)
    if p.returncode or len(p.stdout)>65536:raise ValueError('Configured environment probe failed')
    v=json.loads(p.stdout)
    if v['python'][:2]!=[3,11] or v['isolated']!=1 or v['user_site']is not False:raise ValueError('Isolated CPython3.11 required')
    if v['versions']!=VERSIONS:raise ValueError('Dependency versions differ')
    v['interpreter_sha256']=digest(python)
    return v

def metadata_status(config):
    from .profile_v5 import verify_profile,verify_vendor
    try:
        verify_vendor(config);profile=verify_profile(config)
        return {'available':False,'state':'physical_admission_required','profile_verified':True,'physical_admission':False,'scope':'portable_owned_stand_pose_unexecuted'}
    except (ValueError,OSError,KeyError):
        return {'available':False,'state':'public_assets_or_profile_missing_or_changed','profile_verified':False,'physical_admission':False,'scope':'portable_owned_stand_pose_unexecuted'}
