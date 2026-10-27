"""Private phrase planning; preparation default, one explicit model attempt maximum."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.phrase_planner import prepare_phrase_plan,verify_music_binding,generate_phrase_plan,canonical,safe_provider_error_category

def write_new(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--music',type=Path,required=True);p.add_argument('--instruction',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute-model',action='store_true');p.add_argument('--timeout',type=float,default=120);a=p.parse_args()
    summary=verify_music_binding(json.loads(a.summary.read_text(encoding='utf-8')),a.music);prepared=prepare_phrase_plan(summary,a.instruction)
    if not a.output.exists():
        a.output.mkdir(parents=True);write_new(a.output/'request.json',prepared);write_new(a.output/'schema.json',prepared['provider_schema']);write_new(a.output/'local_schema.json',prepared['program_schema']);write_new(a.output/'prepared_status.json',{'status':'prepared_only','model_calls':0,'request_sha256':prepared['request_sha256'],'music_sha256':summary['sha256'],'physical_steps':0,'instruction_preserved_exact':True})
    else:
        if not a.execute_model:raise FileExistsError('new preparation directory required')
        if canonical(json.loads((a.output/'request.json').read_text(encoding='utf-8')))!=canonical(prepared):raise ValueError('existing prepared request differs')
    if not a.execute_model:
        print(json.dumps({'status':'prepared_only','request_sha256':prepared['request_sha256'],'model_calls':0,'physical_steps':0}));return
    from pulsemorph.agentic.app.providers import CodexLocalProvider
    # Exclusive durable sentinel forbids retries even after failure/cancellation.
    write_new(a.output/'model_attempt.json',{'status':'attempted','request_sha256':prepared['request_sha256'],'model_requested':'gpt-6.1-sol','at_UTC':datetime.now(timezone.utc).isoformat(),'max_attempts':1})
    try:
        result=generate_phrase_plan(CodexLocalProvider(trusted_local=True),prepared,timeout=a.timeout)
        write_new(a.output/'result.json',result);write_new(a.output/'program.json',result['program']);write_new(a.output/'completed_status.json',{'status':'model_proposal_completed','completed_model_calls':1,'physical_steps':0,'request_sha256':prepared['request_sha256']});print(json.dumps({'status':'model_proposal_completed','completed_model_calls':1,'physical_steps':0}))
    except Exception as exc:
        write_new(a.output/'failed_status.json',{'status':'proposal_failed','error_type':type(exc).__name__,'error_category':safe_provider_error_category(exc),'attempts':1,'completion_unconfirmed':True,'retry_allowed':False,'physical_steps':0});print(json.dumps({'status':'proposal_failed','error_type':type(exc).__name__,'error_category':safe_provider_error_category(exc),'retry_allowed':False}));raise SystemExit(1)
if __name__=='__main__':main()
