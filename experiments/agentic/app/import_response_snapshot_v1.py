"""Prepare by default; one root-authorized full verification imports a snapshot."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase.assets import PhraseAssets,PUBLIC_PYTHON
from pulsemorph.agentic.app.phrase.runtime import PhraseRuntime
from pulsemorph.agentic.app.phrase.response_evidence_v1 import PILOT,INDEX
from pulsemorph.agentic.app.phrase.exports_v1 import _Reader,linked
from pulsemorph.agentic.app.phrase.response_snapshot_v1 import BASE,prepare_snapshot_import,import_verified_snapshot,_bytes,_hash,_write
def runtime_without_worker():
    reader=_Reader([PILOT]);saved=reader.json(PILOT/'prepared.json',INDEX['prepared.json'],128*1024)
    runtime=PhraseRuntime.__new__(PhraseRuntime);runtime.root=ROOT;runtime.python=PUBLIC_PYTHON;runtime.assets=PhraseAssets(ROOT);runtime.environment=saved['environment'];runtime.artifact_root=Path.home()/'.cache/pulsemorph-agentic/app_phrase_v1';runtime._implementation=runtime.implementation_pins();return runtime
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-import',action='store_true');parser.add_argument('--authority',type=Path);parser.add_argument('--authority-sha256');args=parser.parse_args()
    output=args.output.absolute()
    if not output.resolve().is_relative_to((BASE/'preparations').resolve()):raise ValueError('Dedicated snapshot preparation directory required')
    for path in (output,*output.parents):
        if path.exists() and linked(path):raise ValueError('Linked preparation path forbidden')
    runtime=runtime_without_worker();plan=prepare_snapshot_import(runtime)
    if not output.exists():output.mkdir(parents=True,exist_ok=False);_write(output/'import_plan.json',plan);_write(output/'prepared_status.json',{'status':'prepared_only','plan_sha256':_hash(_bytes(plan)),'new_full_raw_verifications':0,'new_model':0,'new_compile':0,'new_ONNX':0,'new_FK':0,'new_steps':0})
    elif not args.execute_import:raise FileExistsError('New exclusive preparation required')
    elif _bytes(json.loads((output/'import_plan.json').read_text()))!=_bytes(plan):raise ValueError('Prepared import plan changed')
    if not args.execute_import:print(json.dumps({'status':'prepared_only','plan_sha256':_hash(_bytes(plan)),'new_full_raw_verifications':0}));return
    if not args.authority or not args.authority_sha256:raise ValueError('Explicit root import authority path/hash required')
    result=import_verified_snapshot(runtime,plan,args.authority,args.authority_sha256);_write(output/'import_result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
