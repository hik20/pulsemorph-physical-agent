"""Record the actual isolated interpreter, distribution closure and local hooks."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import site
import sys

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert sys.flags.isolated and not site.ENABLE_USER_SITE
    if args.output.exists():raise FileExistsError(args.output)
    root=args.root.resolve();prefix=Path(sys.prefix).resolve();sites=[Path(p) for p in site.getsitepackages()]
    packages=[{'name':d.metadata['Name'],'version':d.version,'metadata':str(Path(d._path).relative_to(prefix))}
              for d in sorted(metadata.distributions(),key=lambda d:d.metadata['Name'].lower())]
    hooks=[{'path':str(p.relative_to(prefix)),'sha256':sha(p),'content':p.read_text()} for base in sites for p in base.glob('*.pth')]
    assert all('AppData/Roaming' not in h['content'].replace('\\','/') for h in hooks)
    copied=json.loads((root/'artifacts/runs/agentic/app_environment_v1/clone_manifest.json').read_text())
    samples=[]
    for name in ['torch','numpy','scipy','onnxruntime','mujoco']:
        row=max((r for r in copied['files'] if r['distribution'].lower()==name),key=lambda r:r['bytes'])
        target=prefix/row['destination'];source=Path(row['source'])
        assert not os.path.samefile(source,target) and sha(target)==row['sha256']
        samples.append({'distribution':name,'destination':row['destination'],'same_file_as_source':False,
                        'destination_link_count':target.stat().st_nlink,'sha256_verified':True})
    result={'schema':'pulsemorph.actual_cloned_cpu_environment_lock.v1','mechanism':'selected_installed_distribution_ordinary_copy_plus_four_public_wheels',
            'python':platform.python_version(),'architecture':platform.machine(),'platform':platform.platform(),
            'executable':sys.executable,'executable_sha256':sha(Path(sys.executable)),
            'base_interpreter':sys._base_executable,'base_interpreter_sha256':sha(Path(sys._base_executable)),
            'pyvenv_cfg_sha256':sha(prefix/'pyvenv.cfg'),'isolated':True,'user_site_enabled':False,
            'system_site_packages':False,'packages':packages,'local_pth_hooks':hooks,'ordinary_copy_samples':samples,
            'copied_bytes':copied['copied_bytes'],'record_verified_files':copied['record_hash_verified_count'],
            'public_wheel_download_bytes':sum(r['bytes'] for r in copied['network_wheels']),
            'fresh_full_wheel_install_tested':False,'global_environment_modified':False}
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'packages':len(packages),'local_pth_hooks':len(hooks),'ordinary_copy_samples':len(samples),'output':str(args.output)}))
if __name__=='__main__':main()
