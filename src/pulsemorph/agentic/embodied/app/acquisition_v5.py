"""Offline deterministic materialization into a new owned destination only."""
import hashlib,json,subprocess
from pathlib import Path,PurePosixPath
from . import config_v5
from .config_v5 import ordinary,digest

def canonical_asset(raw,row):
    h=hashlib.sha256(raw).hexdigest()
    if h==row['sha256'] and len(raw)==row['bytes']:return raw
    if h!=row['tracked_blob_sha256'] or row['materialization']!='lf_to_crlf':raise ValueError('Source bytes neither tracked nor canonical')
    value=raw.replace(b'\n',b'\r\n')
    if len(value)!=row['bytes'] or hashlib.sha256(value).hexdigest()!=row['sha256']:raise ValueError('Declared materialization does not match')
    return value

def materialize(source,destination,identity_path):
    source=ordinary(source).resolve();destination=ordinary(destination).resolve();identity_path=ordinary(identity_path)
    if digest(identity_path)!=config_v5.IDENTITY_SHA256:raise ValueError('Unadmitted public identity')
    identity=json.loads(identity_path.read_bytes())
    if destination.exists() or destination==source or destination.is_relative_to(source) or source.is_relative_to(destination):raise ValueError('New disjoint owned destination required')
    def git(*args):
        result=subprocess.run(['git','-C',str(source),*args],capture_output=True,timeout=30)
        if result.returncode:raise ValueError('Pinned Git source unavailable')
        return result.stdout
    if git('rev-parse','HEAD').decode().strip()!=identity['upstream']['commit']:raise ValueError('Upstream commit differs')
    rows=identity['files'];names=[r['path']for r in rows]
    if len(names)!=70 or len(set(names))!=70:raise ValueError('Exact70 public assets required')
    for row in rows:
        p=PurePosixPath(row['path'])
        if p.is_absolute() or '..'in p.parts or '\\'in row['path'] or ':'in row['path']:raise ValueError('Unsafe asset path')
        blob=git('show','HEAD:'+row['path'])
        if hashlib.sha256(blob).hexdigest()!=row['tracked_blob_sha256']:raise ValueError('Tracked upstream blob differs')
        file=ordinary(source/row['path'])
        # Reading declared materialized LFS payloads is allowed; network fetching is not.
        if file.stat().st_size>max(row['bytes'],1024):raise ValueError('Source file bound')
        canonical_asset(file.read_bytes(),row)
    destination.mkdir(parents=True,exist_ok=False)
    # Preserve any partial failed destination; never delete, overwrite, or resume.
    with (destination/'materialization_attempt.json').open('x')as f:json.dump({'schema':'pulsemorph.body_materialization_attempt.v5','asset_identity_sha256':digest(identity_path),'upstream_commit':identity['upstream']['commit'],'files':70},f)
    inventory=[]
    for row in rows:
        file=ordinary(source/row['path']);raw=canonical_asset(file.read_bytes(),row)
        out=destination/row['path'];out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb')as f:f.write(raw)
        if digest(out)!=row['sha256']:raise ValueError('Published asset changed')
        inventory.append({'path':row['path'],'bytes':len(raw),'sha256':row['sha256']})
    receipt={'schema':'pulsemorph.body_materialization.v5','asset_identity_sha256':digest(identity_path),'upstream_commit':identity['upstream']['commit'],'files':inventory,'new_downloads':0,'new_NN':0,'new_FK':0,'new_steps':0,'scope':'Offline byte materialization only; no environment or physical admission.'}
    with (destination/'materialization.json').open('x')as f:json.dump(receipt,f,indent=2)
    return receipt
