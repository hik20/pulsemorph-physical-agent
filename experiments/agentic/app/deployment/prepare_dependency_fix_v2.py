"""Prepare setup v2 only; no model/inference/physics/retry execution."""
import ast,hashlib,json,pathlib,shutil,os,subprocess
ROOT=pathlib.Path('D:/projects/dance');OUT=ROOT/'artifacts/runs/agentic/app_relocation_dependency_fix_v2';OLD=pathlib.Path('C:/Users/halke/.cache/pulsemorph-agentic/deployments/sourcekit-v6-pilot-v1/pulsemorph-sourcekit');NEW=pathlib.Path('C:/Users/halke/.cache/pulsemorph-agentic/deployments/sourcekit-v6-pilot-v2/pulsemorph-sourcekit');PYTHON='C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not OUT.exists() and not NEW.parent.exists();assert shutil.disk_usage('C:/').free>=2*1024**3
 OUT.mkdir();frozen=json.loads((ROOT/'artifacts/runs/agentic/app_relocation_pilot_plan_v1/staged_final_inventory.json').read_text());assert len(frozen)==272
 for row in frozen:assert h(OLD/row['path'])==row['sha256']
 source=ROOT/'experiments/agentic/app/preflight.py';tree=ast.parse(source.read_text());pins=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MOTION_CLASS_SOURCE_PINS' for t in n.targets))
 additions=[{'relative':'research/vendor/ProtoMotions/'+n,'sha256':digest,'bytes':(ROOT/'research/vendor/ProtoMotions'/n).stat().st_size} for n,digest in pins.items()]
 total=sum(r['bytes'] for r in frozen)+sum(r['bytes'] for r in additions)+source.stat().st_size;assert total<256*1024**2
 plan={'base_frozen_files':272,'base_pilot_preserved':str(OLD),'new_root':str(NEW),'new_vendor_closure':additions,'new_closure_bytes':sum(r['bytes'] for r in additions),'planned_upper_bound_bytes':total,'cap_bytes':256*1024**2,'source_delta':{'path':'experiments/agentic/app/preflight.py','old_sha256':h(OLD/'experiments/agentic/app/preflight.py'),'new_sha256':h(source)},'source_manifest_qualification':'Based on frozen canonical v6, but updated preflight is explicit setup-v2 overlay; NOT canonical v6/current-sourcekit manifest parity. Root may produce v7 later.','inference_physics_operations':0}
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
 # Read-only new validator against old closure; no source is written to old root.
 env=dict(os.environ)
 for k in list(env):
  if k in ('PYTHONHOME','PYTHONPATH') or k.startswith(('ZETARIS_','OPENAI_')):env.pop(k)
 env.update(PYTHONDONTWRITEBYTECODE='1',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0')
 code="import importlib.util,pathlib,json; s=importlib.util.spec_from_file_location('new_preflight',%r);m=importlib.util.module_from_spec(s);s.loader.exec_module(m); m.validate_trusted_motion(pathlib.Path(%r),'wave')"%(str(source),str(OLD))
 r=subprocess.run([PYTHON,'-I','-B','-c',code],env=env,cwd=OLD,capture_output=True,text=True,timeout=30);assert r.returncode!=0 and 'Missing pinned serialization/interpolation source' in r.stderr
 (OUT/'old_closure_new_probe_failure.json').write_text(json.dumps({'exit_code':r.returncode,'stderr':r.stderr,'oldroot_unmodified':True,'new_validator_sha256':h(source)},indent=2))
 NEW.mkdir(parents=True)
 for row in frozen:
  dst=NEW/row['path'];dst.parent.mkdir(parents=True,exist_ok=True)
  if row['path'].endswith('/.git/config'):
   assert (OLD/row['path']).read_text()=='[core]\n repositoryformatversion = 0\n bare = false\n filemode = false\n autocrlf = true\n'
   shutil.copyfile(OLD/row['path'],dst)
  else:shutil.copyfile(OLD/row['path'],dst)
  assert h(dst)==row['sha256']
 shutil.copyfile(source,NEW/'experiments/agentic/app/preflight.py')
 for row in additions:
  src=ROOT/row['relative'];assert h(src)==row['sha256'];dst=NEW/row['relative'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert h(dst)==row['sha256']
 records=[{'path':p.relative_to(NEW).as_posix(),'sha256':h(p),'bytes':p.stat().st_size} for p in sorted(NEW.rglob('*')) if p.is_file()];(OUT/'v2_staged_inventory.json').write_text(json.dumps(records,indent=2));assert sum(r['bytes'] for r in records)<256*1024**2
 cmd=[PYTHON,'-I','-B',str(NEW/'experiments/agentic/app/preflight.py'),'--runtime-python',PYTHON,'--json'];r=subprocess.run(cmd,cwd=NEW,env=env,capture_output=True,text=True,timeout=90);(OUT/'v2_preflight.json').write_text(r.stdout);(OUT/'v2_preflight.stderr.txt').write_text(r.stderr)
 assert all(h(OLD/row['path'])==row['sha256'] for row in frozen)
 report=json.loads(r.stdout);receipt={'preflight_exit':r.returncode,'preflight_ok':report['ok'],'critical_missing':[x['name'] for x in report['checks'] if x['critical'] and not x['ok']],'files':len(records),'bytes':sum(r['bytes'] for r in records),'old272hashes_unchanged':True,'v2_payload_unchanged_during_probe':all(h(NEW/r['path'])==r['sha256'] for r in records),'new_probe_checks':[c for c in report['checks'] if c['name'].startswith('trusted_motion_payload:')],'inference_calls':0,'physics_steps':0,'script_sha256':h(pathlib.Path(__file__))};(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='new_probe_checks'},indent=2))
if __name__=='__main__':main()
