"""Execute the exact prior 30-wheel public recipe in a NEW isolated environment.

Only pinned files.pythonhosted.org wheel URLs are fetched. No index discovery,
source build, model download, credential read or existing environment mutation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import urllib.request


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def build(recipe_path,env,wheelhouse,output):
    root=Path(__file__).resolve().parents[4]
    expected_root=Path(os.environ['USERPROFILE'])/'.cache/pulsemorph-agentic'
    env,wheelhouse,output=env.resolve(),wheelhouse.resolve(),output.resolve()
    if env!=(expected_root/'envs/app-public-win-cp311-v1').resolve():raise ValueError('unexpected dedicated environment target')
    if wheelhouse!=(expected_root/'wheelhouse/app-win-cp311-v1').resolve():raise ValueError('unexpected dedicated wheelhouse target')
    if output!=(root/'artifacts/runs/agentic/app_public_environment_v1').resolve():raise ValueError('unexpected receipt directory')
    if any(path.exists() for path in (env,wheelhouse,output)):raise FileExistsError('never reuse an existing environment, wheelhouse or receipt directory')
    if sys.version_info[:2]!=(3,11) or platform.system()!='Windows':raise ValueError('Windows Python3.11 required')
    recipe=json.loads(recipe_path.read_text());wheels=recipe['wheels']
    if len(wheels)!=30 or recipe['unresolved'] or sum(r['bytes'] for r in wheels)!=344153316:raise ValueError('prior exact recipe inventory mismatch')
    if shutil.disk_usage('C:/').free<sum(r['bytes'] for r in wheels)+3*1024**3:raise RuntimeError('insufficient C space')
    output.mkdir(parents=True);wheelhouse.mkdir(parents=True)
    lock=threading.RLock();started=time.monotonic()
    progress={'schema':'pulsemorph.public_wheel_environment_execution.v1','pid':os.getpid(),'started_at':now(),
              'environment':str(env),'wheelhouse':str(wheelhouse),'recipe_sha256':sha(recipe_path),
              'script_sha256':sha(Path(__file__)),'status':'running','stage':'downloading','wheels':{},
              'download_expected_bytes':sum(r['bytes'] for r in wheels),'model_calls':0,'physics_steps':0}
    def save():
        with lock:
            progress['updated_at']=now();progress['elapsed_seconds']=round(time.monotonic()-started,3)
            temp=output/'progress.tmp';temp.write_text(json.dumps(progress,indent=2));temp.replace(output/'progress.json')
    save();shutil.copyfile(recipe_path,output/'exact_public_recipe.json');shutil.copyfile(__file__,output/'install_script_snapshot.py')
    def download(row):
        parsed=urlsplit(row['url'])
        if parsed.scheme!='https' or parsed.hostname!='files.pythonhosted.org' or parsed.username or parsed.password or parsed.query:
            raise ValueError('wheel URL is not an exact public artifact URL')
        if Path(row['filename']).name!=row['filename'] or not row['filename'].endswith('.whl'):raise ValueError('invalid wheel filename')
        partial=wheelhouse/(row['filename']+'.part');target=wheelhouse/row['filename'];tick=time.monotonic();digest=hashlib.sha256();count=0;reported=0
        with lock:progress['wheels'][row['filename']]={'status':'downloading','bytes':0,'expected_bytes':row['bytes']}
        save()
        try:
            with urllib.request.urlopen(row['url'],timeout=30) as response,partial.open('xb') as stream:
                if urlsplit(response.url).hostname!='files.pythonhosted.org':raise ValueError('wheel request redirected outside public artifact host')
                for chunk in iter(lambda:response.read(1024*1024),b''):
                    count+=len(chunk)
                    if count>row['bytes'] or time.monotonic()-tick>300:raise ValueError('wheel exceeded pinned size or timeout')
                    stream.write(chunk);digest.update(chunk)
                    if count-reported>=10*1024*1024:
                        with lock:progress['wheels'][row['filename']]['bytes']=count
                        save();reported=count
            if count!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('wheel SHA256/size verification failed')
            partial.rename(target)
            with lock:progress['wheels'][row['filename']]={'status':'verified','bytes':count,'sha256':digest.hexdigest(),'seconds':round(time.monotonic()-tick,3)}
            save()
            return {'name':row['name'],'version':row['version'],'filename':row['filename'],'url':row['url'],'sha256':row['sha256'],'bytes':count}
        except Exception as error:
            with lock:progress['wheels'][row['filename']]={'status':'failed_partial_preserved','bytes':count,'error_type':type(error).__name__}
            save();raise
    try:
        fetched=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(download,row) for row in wheels]
            for future in as_completed(futures):fetched.append(future.result())
        progress['stage']='creating_new_venv';save()
        base=Path(sys.base_prefix)/'python.exe'
        with (output/'venv_create.log').open('xb') as log:
            subprocess.run([str(base),'-I','-m','venv','--copies',str(env)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        local_lock=output/'requirements-local-verified.lock'
        local_lock.write_text(''.join(f"{(wheelhouse/row['filename']).as_posix()} --hash=sha256:{row['sha256']}\n" for row in wheels))
        python=env/'Scripts/python.exe';progress['stage']='installing_verified_local_wheels';save()
        command=[str(python),'-I','-m','pip','--isolated','--disable-pip-version-check','install','--no-index',
                 '--no-deps','--no-compile','--only-binary=:all:','--require-hashes','-r',str(local_lock)]
        with (output/'pip_install.log').open('xb') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        checked=subprocess.run([str(python),'-I','-m','pip','--isolated','--disable-pip-version-check','check'],capture_output=True,text=True,timeout=30)
        progress['pip_check']={'returncode':checked.returncode,'stdout':checked.stdout,'stderr':checked.stderr}
        if checked.returncode:raise RuntimeError('public recipe dependency closure failed pip check')
        progress['stage']='installed_awaiting_import_tests_and_physics';progress['status']='installed_verified_wheels'
        progress['actual_download_bytes']=sum(r['bytes'] for r in fetched);progress['downloaded_wheels']=len(fetched)
        progress['python_executable']=str(python);save()
        with (output/'wheel_manifest.json').open('x',encoding='utf-8') as f:json.dump({'wheels':sorted(fetched,key=lambda r:r['filename']),'actual_bytes':sum(r['bytes'] for r in fetched)},f,indent=2)
        print(json.dumps({key:progress[key] for key in ('status','downloaded_wheels','actual_download_bytes','elapsed_seconds','python_executable')}))
    except Exception as error:
        progress['status']='failed_preserved';progress['error_type']=type(error).__name__;save();raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--env',type=Path,required=True);parser.add_argument('--wheelhouse',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    build(args.recipe,args.env,args.wheelhouse,args.output)
if __name__=='__main__':main()
