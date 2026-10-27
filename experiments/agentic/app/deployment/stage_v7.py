"""Stage canonical v7 plus exact existing pinned upstream payloads; preflight only."""
import ast,hashlib,json,pathlib,shutil,zipfile,stat,subprocess,os
ROOT=pathlib.Path('D:/projects/dance');OUT=ROOT/'artifacts/runs/agentic/app_relocation_v7_setup_v1';OLD=pathlib.Path('C:/Users/halke/.cache/pulsemorph-agentic/deployments/sourcekit-v6-pilot-v1/pulsemorph-sourcekit');TARGET=pathlib.Path('C:/Users/halke/.cache/pulsemorph-agentic/deployments/sourcekit-v7-pilot-v1');KIT=TARGET/'pulsemorph-sourcekit';PYTHON='C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe';ZIP_SHA='830fa051b472b928cefd0eab500afce637fdf43433e7c2eb3216a81e77e24e53'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():return [{'path':p.relative_to(KIT).as_posix(),'bytes':p.stat().st_size,'sha256':h(p)} for p in sorted(KIT.rglob('*')) if p.is_file()]
def main():
 assert not OUT.exists() and not TARGET.exists();assert shutil.disk_usage('C:/').free>=2*1024**3;OUT.mkdir()
 archive=ROOT/'artifacts/runs/agentic/app_demo_package_v1/source_package_v7/pulsemorph-sourcekit-draft.zip';assert h(archive)==ZIP_SHA
 base=json.loads((ROOT/'artifacts/runs/agentic/app_relocation_pilot_plan_v1/staged_final_inventory.json').read_text());copies=[]
 for row in base:
  assert h(OLD/row['path'])==row['sha256']
  if row['path'].startswith('research/vendor/'):
   copies.append({'path':row['path'],'from':str(OLD/row['path']),'raw_sha256':row['sha256'],'bytes':row['bytes']})
 tree=ast.parse((ROOT/'experiments/agentic/app/preflight.py').read_text());pins=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MOTION_CLASS_SOURCE_PINS' for t in n.targets))
 for name,canonical in pins.items():
  src=ROOT/'research/vendor/ProtoMotions'/name;raw=src.read_bytes();assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==canonical
  copies.append({'path':'research/vendor/ProtoMotions/'+name,'from':str(src),'raw_sha256':h(src),'canonical_LF_sha256':canonical,'bytes':len(raw)})
 with zipfile.ZipFile(archive) as z:total=sum(i.file_size for i in z.infolist())+sum(r['bytes'] for r in copies)
 assert total<256*1024**2
 (OUT/'staging_plan.json').write_text(json.dumps({'archive_sha256':ZIP_SHA,'copy_files':copies,'total_bytes':total,'cap_bytes':256*1024**2,'free_before':shutil.disk_usage('C:/').free,'no_config_auth_copy':'Only already-generated core-only pilot Git config validated against fixed contents; no original/user/remotes/auth config','raw_vs_canonical':'Raw copies verify raw SHA; only Python source probes normalize CRLF to pinned LF SHA'},indent=2))
 TARGET.mkdir(parents=True)
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   p=pathlib.PurePosixPath(item.filename);assert p.parts[0]=='pulsemorph-sourcekit' and not p.is_absolute() and '..' not in p.parts and ':' not in item.filename and not stat.S_ISLNK(item.external_attr>>16)
   dst=TARGET.joinpath(*p.parts);dst.parent.mkdir(parents=True,exist_ok=True)
   with dst.open('xb') as f:f.write(z.read(item))
 for row in copies:
  src=pathlib.Path(row['from']);assert not src.is_symlink() and h(src)==row['raw_sha256']
  if row['path'].endswith('/.git/config'):assert src.read_text()=='[core]\n repositoryformatversion = 0\n bare = false\n filemode = false\n autocrlf = true\n'
  dst=KIT/row['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert h(dst)==row['raw_sha256']
 assert all(not p.is_symlink() and not bool(getattr(p.lstat(),'st_file_attributes',0)&1024) for p in TARGET.rglob('*'))
 before=inventory();(OUT/'inventory_before_preflight.json').write_text(json.dumps(before,indent=2));assert sum(x['bytes'] for x in before)<256*1024**2
 env=dict(os.environ)
 for k in list(env):
  if k in ('PYTHONHOME','PYTHONPATH') or k.startswith(('ZETARIS_','OPENAI_')):env.pop(k)
 env.update(PYTHONDONTWRITEBYTECODE='1',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
 cmd=[PYTHON,'-I','-B',str(KIT/'experiments/agentic/app/preflight.py'),'--runtime-python',PYTHON,'--json'];r=subprocess.run(cmd,cwd=KIT,env=env,capture_output=True,text=True,timeout=90);(OUT/'preflight.json').write_text(r.stdout);(OUT/'preflight.stderr.txt').write_text(r.stderr);report=json.loads(r.stdout)
 after=inventory();(OUT/'inventory_after_preflight.json').write_text(json.dumps(after,indent=2));a={x['path']:x['sha256'] for x in before};b={x['path']:x['sha256'] for x in after};changed=[p for p in sorted(set(a)|set(b)) if a.get(p)!=b.get(p)];assert set(changed)<= {'research/vendor/ProtoMotions/.git/index','research/vendor/Meterless/.git/index'}
 manifest=json.loads((KIT/'SOURCE_MANIFEST.json').read_text());assert all(h(KIT/row['path'])==row['sha256'] for row in manifest['sources']);assert all(h(OLD/row['path'])==row['sha256'] for row in base)
 receipt={'root':str(KIT),'archive_sha256':ZIP_SHA,'canonical_manifest_source_hashes_verified':len(manifest['sources']),'files':len(after),'bytes':sum(x['bytes'] for x in after),'preflight_exit':r.returncode,'preflight_ok':report['ok'],'critical_missing':[x['name'] for x in report['checks'] if x['critical'] and not x['ok']],'all_file_hashes_unchanged':not changed,'changed_only_git_index_stat_caches':changed,'immutable_source_scientific_model_controller_motion_bytes_unchanged':True,'old_v1_all272unchanged':True,'inference_calls':0,'physics_steps':0,'render_memory_operations':0,'script_sha256':h(pathlib.Path(__file__))};(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
