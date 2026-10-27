"""Bounded ordinary-copy private relocation from already materialized pinned inputs.
Plan before materialize; never fetch, install, link, copy credentials or run physics.
"""
import argparse,hashlib,json,pathlib,shutil,stat,subprocess,zipfile

CAP=256*1024*1024
MIN_FREE=2*1024*1024*1024
TARGET=pathlib.Path('C:/Users/halke/.cache/pulsemorph-agentic/deployments/sourcekit-v6-pilot-v1')
ZIP_SHA='bb9ab870961f0c763d299c51d6aad15f07398b78941a3ba85d89b05140343ddf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def linked(p):return p.is_symlink() or bool(getattr(p.lstat(),'st_file_attributes',0)&1024)
def plan(root):
    include=set()
    def tree(relative):
        p=root/relative
        include.update(f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and '.bin' not in f.parts)
    def file(relative):include.add(root/relative)
    for vendor in ['ProtoMotions','Meterless']:
        base='research/vendor/'+vendor+'/'
        for name in ['HEAD','index','packed-refs','shallow']:file(base+'.git/'+name)
        tree(base+'.git/refs')
        # Public Git objects only; excludes promisor descriptors, alternates,
        # original config/remotes, logs, hooks, LFS object caches and credentials.
        objects=root/base/'.git/objects'
        include.update(p for p in objects.rglob('*') if p.is_file() and p.suffix in ('.pack','.idx','.rev'))
    tree('research/vendor/ProtoMotions/deployment')
    tree('research/vendor/ProtoMotions/protomotions/data/assets/mesh/G1')
    for name in ['LICENSE.md','legal/MODEL-LICENSING.md','legal/NVIDIA-Open-Model-License.pdf','protomotions/data/assets/mjcf/g1_holo_compat.xml','data/g1-kimodo-generated/proto/output_wave.motion','data/g1-kimodo-generated/proto/output_high5.motion','data/g1-kimodo-generated/proto/output_walk.motion']:
        file('research/vendor/ProtoMotions/'+name)
    model='research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/'
    for name in ['MODEL_CARD.md','LICENSE.pdf','NOTICE.txt','last.ckpt','compiled_models/unified_pipeline.onnx','compiled_models/unified_pipeline.yaml']:file(model+name)
    tree('research/vendor/Meterless/engines/world-model/reference/src')
    tree('research/vendor/Meterless/engines/world-model/reference/node_modules/tsx')
    for name in ['LICENSE','engines/world-model/reference/package.json']:file('research/vendor/Meterless/'+name)
    records=[]
    for p in sorted(include):
        assert p.is_file() and not linked(p),str(p)
        for ancestor in p.parents:
            if ancestor==root:break
            assert not linked(ancestor),str(ancestor)
        records.append({'relative':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    archive=root/'artifacts/runs/agentic/app_demo_package_v1/source_package_v6/pulsemorph-sourcekit-draft.zip';assert sha(archive)==ZIP_SHA
    with zipfile.ZipFile(archive) as z:source_bytes=sum(i.file_size for i in z.infolist())
    total=sum(r['bytes'] for r in records)+source_bytes
    assert total<CAP;assert shutil.disk_usage('C:/').free>=MIN_FREE
    return {'schema':'pulsemorph.private_relocation_plan.v1','target':str(TARGET),'archive_sha256':ZIP_SHA,'vendor_files':records,'vendor_bytes':sum(r['bytes'] for r in records),'source_uncompressed_bytes':source_bytes,'total_planned_bytes':total,'cap_bytes':CAP,'min_free_bytes':MIN_FREE,'free_bytes_before':shutil.disk_usage('C:/').free,'git_metadata_scope':'authentic HEAD/index/refs/shallow/public pack-index-rev; generated minimal local core config only; no original config/hooks/logs/LFS cache/promisor/alternates','external_host_prerequisites':['existing public Python venv','Node','FFmpeg','Git','Codex binary presence only; no auth/config copied'],'no_fresh_machine_or_acquisition_claim':True}
def materialize(root,receipt):
    assert not TARGET.exists(),'refuse existing target'
    TARGET.mkdir(parents=True);kit=TARGET/'pulsemorph-sourcekit'
    archive=root/'artifacts/runs/agentic/app_demo_package_v1/source_package_v6/pulsemorph-sourcekit-draft.zip'
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            p=pathlib.PurePosixPath(item.filename)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='pulsemorph-sourcekit' and ':' not in item.filename
            assert not stat.S_ISLNK(item.external_attr>>16)
            target=TARGET.joinpath(*p.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(z.read(item))
    copied=[]
    for row in receipt['vendor_files']:
        src=root/row['relative'];dst=kit/row['relative'];assert sha(src)==row['sha256'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(dst)==row['sha256'];copied.append(row)
    for vendor in ['ProtoMotions','Meterless']:
        config=kit/'research/vendor'/vendor/'.git/config'
        with config.open('x') as f:f.write('[core]\n repositoryformatversion = 0\n bare = false\n filemode = false\n')
    assert sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())<CAP
    return {'target':str(TARGET),'sourcekit':str(kit),'ordinary_copied_files':copied,'no_links':all(not linked(p) for p in TARGET.rglob('*')),'total_staged_bytes':sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file()),'physics_model_or_memory_operations':0}
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=pathlib.Path,required=True);a.add_argument('--receipt',type=pathlib.Path,required=True);a.add_argument('--materialize',action='store_true');args=a.parse_args();assert not args.receipt.exists();r=plan(args.root.resolve());print(json.dumps({k:v for k,v in r.items() if k!='vendor_files'},indent=2))
    if args.materialize:r['materialized']=materialize(args.root.resolve(),r)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as f:json.dump(r,f,indent=2)
if __name__=='__main__':main()
