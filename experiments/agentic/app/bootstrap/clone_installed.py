"""Guarded selected-distribution clone into one NEW isolated Windows venv.

This is not a wheel installation or a fresh-machine reproducibility claim.
Source distributions are read only; ordinary copies never share inodes.
"""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

def copy_verified(source,destination,record_hash):
    destination.parent.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256();expected=hashlib.new(record_hash.mode) if record_hash else None
    with source.open('rb') as origin,destination.open('xb') as target:
        for chunk in iter(lambda:origin.read(1024*1024),b''):
            target.write(chunk);sha.update(chunk)
            if expected:expected.update(chunk)
    if expected and base64.urlsafe_b64encode(expected.digest()).decode().rstrip('=')!=record_hash.value:
        raise ValueError('source RECORD hash mismatch: '+str(source))
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=sha.hexdigest():
        raise ValueError('copy verification mismatch: '+str(destination))
    return sha.hexdigest()


def build(plan_path,wheels_path,target,receipt_dir):
    plan=json.loads(plan_path.read_text());wheels=json.loads(wheels_path.read_text())
    allowed=(Path(os.environ['USERPROFILE'])/'.cache/pulsemorph-agentic/envs').resolve()
    target=target.resolve()
    if target.parent!=allowed or target.name!='app-cpu-py311-v1':raise ValueError('unexpected dedicated environment target')
    if target.exists():raise FileExistsError('target already exists; never modify an existing environment')
    if sys.version_info[:2]!=(3,11):raise ValueError('Python 3.11 required')
    if shutil.disk_usage('C:/').free<plan['estimated_copy_bytes']+512*1024*1024:raise RuntimeError('insufficient C drive space')
    target.parent.mkdir(parents=True,exist_ok=True);receipt_dir.mkdir(parents=True,exist_ok=True)
    receipt={'schema':'pulsemorph.isolated_selected_distribution_clone.v1','kind':'ordinary_copied_installed_closure_plus_four_public_wheels',
             'environment':str(target),'source_plan':str(plan_path),'system_site_packages':False,
             'files':[],'excluded':[],'network_wheels':[],'status':'failed'}
    try:
        base_python=Path(plan['base_prefix'])/'python.exe'
        subprocess.run([str(base_python),'-I','-m','venv','--copies',str(target)],check=True)
        site=target/'Lib/site-packages'
        for row in plan['selected']:
            distribution=metadata.Distribution.at(row['metadata_directory']);source_site=Path(row['site']).resolve()
            for record in distribution.files or []:
                source=Path(distribution.locate_file(record)).resolve()
                relative=Path(str(record))
                if (not source.is_relative_to(source_site) or not source.is_file() or '__pycache__' in relative.parts
                        or source.suffix in ('.pyc','.pth')):
                    receipt['excluded'].append({'distribution':row['name'],'record':str(record),'reason':'outside-site script, cache, pth, or absent'})
                    continue
                destination=(site/relative).resolve()
                if not destination.is_relative_to(site):raise ValueError('record path escaped dedicated site')
                if destination.exists():
                    if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(destination.read_bytes()).digest():raise ValueError('distribution file collision')
                    continue
                digest=copy_verified(source,destination,record.hash)
                receipt['files'].append({'distribution':row['name'],'version':row['version'],'source':str(source),
                    'destination':str(destination.relative_to(target)),'sha256':digest,'record_hash_verified':bool(record.hash),'bytes':destination.stat().st_size})
        wheelhouse=target/'bootstrap_wheelhouse';wheelhouse.mkdir()
        for row in wheels['wheels']:
            if not row['url'].startswith('https://files.pythonhosted.org/'):raise ValueError('non-public wheel source rejected')
            with urllib.request.urlopen(row['url'],timeout=30) as response:content=response.read(row['bytes']+1)
            if len(content)!=row['bytes'] or hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('public wheel pin mismatch')
            path=wheelhouse/row['filename']
            with path.open('xb') as stream:stream.write(content)
            receipt['network_wheels'].append(row)
        lock=wheelhouse/'missing.lock'
        lock.write_text(''.join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in wheels['wheels']))
        python=target/'Scripts/python.exe'
        subprocess.run([str(python),'-I','-m','pip','--isolated','install','--no-index','--find-links',str(wheelhouse),
                        '--no-deps','--no-compile','--require-hashes','-r',str(lock)],check=True)
        check=subprocess.run([str(python),'-I','-m','pip','--isolated','check'],capture_output=True,text=True)
        receipt['pip_check']={'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr}
        if check.returncode:raise RuntimeError('isolated metadata closure failed pip check')
        receipt['status']='success';receipt['copied_bytes']=sum(r['bytes'] for r in receipt['files'])
        receipt['record_hash_verified_count']=sum(r['record_hash_verified'] for r in receipt['files'])
    except Exception as error:
        receipt['error_type']=type(error).__name__;receipt['error']=str(error)
        receipt['traceback']=traceback.format_exc()
        raise
    finally:write_new(receipt_dir/'clone_manifest.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','environment','copied_bytes','record_hash_verified_count')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--wheel-pins',type=Path,required=True);parser.add_argument('--env',type=Path,required=True)
    parser.add_argument('--receipts',type=Path,required=True);args=parser.parse_args()
    build(args.plan,args.wheel_pins,args.env,args.receipts)
if __name__=='__main__':main()
