"""Prepare-only three-gain motor contrast; execution requires root's exact hash authority."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase.motor_response_v1 import prepare,execute,safe_output,write,encoded,digest
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--authority',type=Path);p.add_argument('--authority-sha256');args=p.parse_args()
    output=safe_output(args.output);prepared=prepare(ROOT,output)
    if not output.exists():output.mkdir(parents=True,exist_ok=False);write(output/'prepared.json',prepared)
    elif not args.execute:raise FileExistsError('New exclusive preparation required')
    elif encoded(json.loads((output/'prepared.json').read_text()))!=encoded(prepared):raise ValueError('Existing prepared plan differs')
    if not args.execute:print(json.dumps({'status':'prepared_only','prepared_sha256':digest(encoded(prepared)),'new_compile':0,'new_ONNX':0,'new_FK':0,'new_steps':0,'new_model':0}));return
    if not args.authority or not args.authority_sha256:raise ValueError('Explicit root authority path/hash required')
    execute(ROOT,output,prepared,args.authority,args.authority_sha256)
if __name__=='__main__':main()
