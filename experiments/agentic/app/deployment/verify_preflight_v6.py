"""Read-only relocated preflight; no physics, model, network or memory operation."""
import hashlib,json,os,pathlib,subprocess
PYTHON=pathlib.Path('C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe')
KIT=pathlib.Path('C:/Users/halke/.cache/pulsemorph-agentic/deployments/sourcekit-v6-pilot-v1/pulsemorph-sourcekit')
OUT=pathlib.Path('D:/projects/dance/artifacts/runs/agentic/app_relocation_pilot_plan_v1')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
env=dict(os.environ)
for key in list(env):
 if key in ('PYTHONPATH','PYTHONHOME') or key.startswith(('ZETARIS_','OPENAI_')):env.pop(key)
env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0',GIT_OPTIONAL_LOCKS='0',PYTHONDONTWRITEBYTECODE='1')
command=[str(PYTHON),'-I','-B',str(KIT/'experiments/agentic/app/preflight.py'),'--runtime-python',str(PYTHON),'--json']
before={str(p.relative_to(KIT)):h(p) for p in KIT.rglob('*') if p.is_file()}
r=subprocess.run(command,cwd=KIT,env=env,capture_output=True,text=True,timeout=90)
(OUT/'preflight.stdout.json').write_text(r.stdout);(OUT/'preflight.stderr.txt').write_text(r.stderr)
after={str(p.relative_to(KIT)):h(p) for p in KIT.rglob('*') if p.is_file()}
receipt={'command':command,'cwd':str(KIT),'exit_code':r.returncode,'no_original_project_PYTHONPATH':True,'git_system_global_config_disabled':True,'git_lazy_fetch_disabled':True,'files_before':len(before),'files_after':len(after),'source_and_vendor_bytes_unchanged':before==after,'model_calls':0,'ONNX_inference':0,'physics_steps':0,'memory_operations':0,'script_sha256':h(pathlib.Path(__file__))}
if r.stdout:
 try:
  report=json.loads(r.stdout);receipt['preflight_ok']=report['ok'];receipt['critical_missing']=[x['name'] for x in report['checks'] if x['critical'] and not x['ok']]
 except Exception:receipt['unparsed_stdout']=True
with (OUT/'preflight_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
