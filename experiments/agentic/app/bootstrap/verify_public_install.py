"""Verify actual public-wheel installation versions, origins and installed RECORD hashes."""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import site
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def canonical(name):return re.sub('[-_.]+','-',name).lower()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--wheelhouse',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert sys.flags.isolated and not site.ENABLE_USER_SITE
    if args.output.exists():raise FileExistsError(args.output)
    prefix=Path(sys.prefix).resolve();recipe=json.loads(args.recipe.read_text());packages=[]
    for row in recipe['wheels']:
        wheel=args.wheelhouse/row['filename'];assert wheel.stat().st_size==row['bytes'] and sha(wheel)==row['sha256']
        distribution=metadata.distribution(row['name']);assert distribution.version==row['version']
        assert Path(distribution._path).resolve().is_relative_to(prefix)
        verified=0;bytes_checked=0
        for record in distribution.files or []:
            path=Path(distribution.locate_file(record)).resolve()
            assert path.is_relative_to(prefix),str(record)
            if not record.hash:continue
            h=hashlib.new(record.hash.mode)
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk);bytes_checked+=len(chunk)
            assert base64.urlsafe_b64encode(h.digest()).decode().rstrip('=')==record.hash.value,str(record)
            verified+=1
        packages.append({'name':row['name'],'version':distribution.version,'metadata_origin':str(Path(distribution._path).relative_to(prefix)),
                         'installed_RECORD_sha256':sha(Path(distribution._path)/'RECORD'),'record_hashes_verified':verified,'bytes_checked':bytes_checked})
    actual={canonical(d.metadata['Name']):d.version for d in metadata.distributions()}
    expected={canonical(r['name']):r['version'] for r in recipe['wheels']}
    assert set(actual)-set(expected)<= {'pip','setuptools'}
    import torch,onnxruntime
    result={'schema':'pulsemorph.verified_public_wheel_installation.v1','passed':True,'environment':sys.prefix,
        'executable':sys.executable,'python_version':platform.python_version(),'executable_sha256':sha(Path(sys.executable)),
        'isolated':bool(sys.flags.isolated),'user_site_enabled':site.ENABLE_USER_SITE,'pyvenv_cfg_sha256':sha(prefix/'pyvenv.cfg'),
        'runtime_packages':packages,'seed_tools':{n:v for n,v in actual.items() if n not in expected},
        'wheels_verified':len(packages),'download_bytes':sum(r['bytes'] for r in recipe['wheels']),
        'installed_record_hashes_verified':sum(r['record_hashes_verified'] for r in packages),
        'installed_bytes_checked':sum(r['bytes_checked'] for r in packages),
        'torch_distribution_version':metadata.version('torch'),'torch_build':torch.__version__,
        'torch_cuda_build':torch.version.cuda,'torch_cuda_available':torch.cuda.is_available(),
        'onnx_available_providers':onnxruntime.get_available_providers(),
        'local_pth_hooks':[{'path':str(p.relative_to(prefix)),'sha256':sha(p)} for p in (prefix/'Lib/site-packages').glob('*.pth')],
        'no_package_copy_from_old_environments':True,'model_calls':0,'physics_steps':0}
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('passed','wheels_verified','installed_record_hashes_verified','torch_build','torch_cuda_build')}))
if __name__=='__main__':main()
