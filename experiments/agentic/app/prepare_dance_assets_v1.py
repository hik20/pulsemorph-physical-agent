"""Hash-only preparation by default; no model/session/physics execution."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.dance.assets_v1 import DanceAssets,check_reference,FIELDS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--check-reference',action='store_true');args=p.parse_args()
 result=DanceAssets(ROOT).prepare(args.output)
 if args.check_reference:
  try:
   _,report=check_reference(ROOT,{k:0. if k=='timing_shift_s' else 1. for k in FIELDS})
  except Exception as error:
   if hasattr(error,'report'):
    if args.output:
     from pulsemorph.agentic.dance.assets_v1 import write_new
     write_new(args.output/'compile_report.json',error.report)
    print(json.dumps(error.report,allow_nan=False))
   raise
  print(json.dumps(report,allow_nan=False))
  return
 print(json.dumps({'schema':result['schema'],'source_sha256':result['source_sha256'],'pin_count':len(result['pins']),'reference_compiled':False,'new_model_onnx_physics_calls':0}))
if __name__=='__main__':main()
