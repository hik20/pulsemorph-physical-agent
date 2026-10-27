"""Prepare by default; one explicitly requested private descriptor-aware plan only."""
import argparse,copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from jsonschema import Draft202012Validator
from pulsemorph.agentic.app.phrase.assets import PhraseAssets
from pulsemorph.agentic.app.phrase.source_descriptor_v1 import describe_sources
from pulsemorph.agentic.app.phrase.music import PhraseMusic
from pulsemorph.agentic.app.phrase.manager_goals_v4 import prepare_initial,INITIAL_SCHEMA_V4
from pulsemorph.agentic.app.phrase.goals_v1 import validate_goals
from pulsemorph.agentic.phrase_planner import validate_program,safe_provider_error_category

BASE=Path.home()/'.cache/pulsemorph-agentic/analysis/descriptor-plan-probe-v1'
GOALS={'schema':'pulsemorph.phrase_goals.v1','ratios':[.75,1.25,1.25,.75]}
INSTRUCTION='Önce Wave, sonra High5 yap; birleşimde yönü hizala. Gerçekleşmiş kol-RMS oranı hedeflerini ilk bölüm sol0.75/sağ1.25, ikinci bölüm sol1.25/sağ0.75 olarak sabitle. Uygulamanın ölçüm ve sınırlı geri bildirim akışını kullan; bu tek çağrıda yalnız parametre JSON döndür. Kaynak betimleyicisindeki başlangıç pozundan eklem uzaklığı ile pencere içindeki zamansal değişimin farklı olduğunu gerekçende belirt; hedef metriğini değiştirme. Fiziksel başarı, ritim eşleşmesi veya yeni beceri garantisi istemiyorum.'
def encoded(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2).encode('utf8')
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    with path.open('xb') as f:f.write(encoded(value))
def linked(path):return path.is_symlink() or bool(getattr(path.lstat(),'st_file_attributes',0)&1024)
def pins():
    relatives=['experiments/agentic/app/probe_descriptor_plan_v1.py','src/pulsemorph/agentic/app/phrase/source_descriptor_v1.py','src/pulsemorph/agentic/app/phrase/assets.py','src/pulsemorph/agentic/app/phrase/music.py','src/pulsemorph/agentic/app/phrase/manager_goals_v4.py','src/pulsemorph/agentic/app/phrase/goals_v1.py','src/pulsemorph/agentic/app/phrase/catalog.py','src/pulsemorph/agentic/app/providers.py','src/pulsemorph/agentic/phrase_planner.py','src/pulsemorph/agentic/phrase_program.py']
    return {p:sha((ROOT/p).read_bytes()) for p in relatives}
def prepare(instruction):
    before=pins();assets=PhraseAssets(ROOT);descriptor=describe_sources(assets)
    # Avoid constructor/materialization: read only the already-existing actual
    # music store through the frozen getter and its PCM/metadata validation.
    music=PhraseMusic.__new__(PhraseMusic);music.root=ROOT;music.directory=Path.home()/'.cache/pulsemorph-agentic/app_phrase_v1/music';music._warnings=[]
    measured=music.get('phrase_original_pulse_v1')
    if measured['sha256']!='0e0d161cbf1f109e65af17d5483c6f5172715494391859d24108438e781a6893':raise ValueError('Expected owned original8s music changed')
    summary={k:v for k,v in measured.items() if k in ('schema','sha256','duration_s','sample_rate','sample_frames','channels','pcm_bits','rms','peak_amplitude','accent_time_s','accent_measurement','onset_times_s','sections','bpm','bpm_source','rights','limitations')}
    context={'source_reference_descriptor':descriptor,'qualification':'Descriptive source knots only; not physical response or goal feasibility.'}
    prepared=prepare_initial(summary,instruction,GOALS,context,candidate_budget=3)
    if before!=pins():raise ValueError('Probe implementation changed during preparation')
    return {'schema':'pulsemorph.descriptor_plan_probe.v1','prepared_plan':prepared,'source_pins':before,'music_sha256':measured['sha256'],'descriptor_sha256':sha(encoded(descriptor)),'max_model_attempts':1,'physical_authority':False,'qualification':'Plan-only integration probe, not V6 physical/E2E success, semantic causality, skill learning or goal attainability.'}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--instruction',default=INSTRUCTION);parser.add_argument('--execute-model',action='store_true');parser.add_argument('--timeout',type=int,default=120);args=parser.parse_args()
    if not 1<=args.timeout<=120:raise ValueError('Timeout1..120 required')
    output=args.output.absolute()
    if not output.resolve().is_relative_to(BASE.resolve()):raise ValueError('Dedicated private C cache output required')
    for path in (output,*output.parents):
        if path.exists() and linked(path):raise ValueError('Linked output path forbidden')
    prepared=prepare(args.instruction)
    if not output.exists():
        output.mkdir(parents=True,exist_ok=False);write(output/'prepared_request.json',prepared);write(output/'prepared_status.json',{'status':'prepared_only','model_attempts':0,'new_ONNX':0,'new_FK':0,'new_steps':0,'request_sha256':sha(encoded(prepared))})
    elif not args.execute_model:raise FileExistsError('New exclusive preparation directory required')
    elif encoded(json.loads((output/'prepared_request.json').read_text(encoding='utf8')))!=encoded(prepared):raise ValueError('Prepared request/source binding changed')
    if not args.execute_model:print(json.dumps({'status':'prepared_only','output':str(output),'request_sha256':sha(encoded(prepared))}));return
    from pulsemorph.agentic.app.providers import CodexLocalProvider
    write(output/'model_attempt.json',{'attempted':True,'completion_unconfirmed':True,'at':datetime.now(timezone.utc).isoformat(),'max_attempts':1,'model_requested':'gpt-6.1-sol','request_sha256':sha(encoded(prepared))})
    completed=False
    try:
        provider=CodexLocalProvider(trusted_local=True)
        result=provider.generate_json(prepared['prepared_plan']['prompt'],copy.deepcopy(INITIAL_SCHEMA_V4),timeout=args.timeout)
        completed=True
        clean={k:copy.deepcopy(result.receipt[k]) for k in ('provider','model_requested','model_returned','usage','live','tool_events','elapsed_seconds','trust_boundary') if k in result.receipt}
        clean.update(completion_unconfirmed=False,request_sha256=sha(encoded(prepared)),response_sha256=sha(encoded(result.data)))
        write(output/'response.json',result.data);write(output/'receipt.json',clean)
        if clean.get('model_requested')!='gpt-6.1-sol' or clean.get('tool_events')!=0:raise ValueError('Provider boundary mismatch')
        Draft202012Validator(INITIAL_SCHEMA_V4).validate(result.data);validate_program(result.data['program'])
        if validate_goals(result.data['goals'])!=GOALS:raise ValueError('Explicit goals echo mismatch')
        if pins()!=prepared['source_pins']:raise ValueError('Probe source changed during model call')
        write(output/'completed_status.json',{'status':'plan_only_response_validated','completed_model_calls':1,'source_pins_unchanged':True,'supported':result.data['supported'],'unsupported_requests':result.data['unsupported_requests'],'new_ONNX':0,'new_FK':0,'new_steps':0,'receipt_sha256':sha(encoded(clean)),'qualification':prepared['qualification']})
        print(json.dumps({'status':'plan_only_response_validated','completed_model_calls':1,'new_steps':0}))
    except Exception as error:
        write(output/'failed_status.json',{'status':'probe_failed','attempts':1,'completed_provider_return':completed,'completion_unconfirmed':not completed,'error_category':safe_provider_error_category(error),'retry_allowed':False,'new_ONNX':0,'new_FK':0,'new_steps':0});print(json.dumps({'status':'probe_failed','retry_allowed':False}));raise SystemExit(1)
    finally:
        after=pins();write(output/'source_after.json',{'source_pins':after,'unchanged':after==prepared['source_pins'],'new_ONNX':0,'new_FK':0,'new_steps':0})
if __name__=='__main__':main()
