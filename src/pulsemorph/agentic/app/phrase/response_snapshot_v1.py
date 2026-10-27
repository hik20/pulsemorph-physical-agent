"""Root-admitted historical snapshots; fast reads never claim fresh raw checks."""
import copy,hashlib,json,stat,threading
from datetime import datetime,timezone
from pathlib import Path
from . import response_evidence_v1 as full
from .exports_v1 import _Reader

BASE=Path.home()/'.cache/pulsemorph-agentic/analysis/response-snapshot-v1'
STORE=BASE/'store_v1';ADMISSION_PATH=BASE/'admission.json'
CATALOG_SHA256='bea6b0e44b6f08c01542f7391ecd1936ef3d85532b09757ab547e66b1d827005'
SNAPSHOT_ID='motor_response_v1_snapshot_v1'
SCHEMA='pulsemorph.imported_motor_response_evidence.v1'
_IMPORT_LOCK=threading.Lock()
def _now():return datetime.now(timezone.utc).isoformat()
def _bytes(v):return json.dumps(v,sort_keys=True,ensure_ascii=True,allow_nan=False,indent=2).encode('utf8')
def _hash(b):return hashlib.sha256(b).hexdigest()
def _require(ok,message):
    if not ok:raise full.ResponseEvidenceIntegrityError(message)
def _write(path,value):
    with Path(path).open('xb') as stream:stream.write(_bytes(value))
def _profile(runtime):
    # Callers freshly guard entry/exit: _unchanged has already compared actual
    # binary/source hashes to this declared map. Avoid a third native read pass.
    return _hash(full._canonical(copy.deepcopy(runtime._implementation)))
def _admitted_sha():
    from .response_snapshot_admission_v1 import ADMISSION_SHA256
    _require(isinstance(ADMISSION_SHA256,str) and (not ADMISSION_SHA256 or len(ADMISSION_SHA256)==64 and all(c in '0123456789abcdef' for c in ADMISSION_SHA256)),'Invalid source admission anchor')
    return ADMISSION_SHA256
def _paths():
    paths=set()
    for p in (BASE,STORE,ADMISSION_PATH):paths.update((p,*p.parents))
    for p in paths:
        try:info=p.lstat()
        except FileNotFoundError:continue
        _require(not stat.S_ISLNK(info.st_mode) and not(getattr(info,'st_file_attributes',0)&1024),'Linked snapshot path forbidden')
        if p==ADMISSION_PATH:_require(stat.S_ISREG(info.st_mode),'Admission is not an ordinary file')
        else:_require(stat.S_ISDIR(info.st_mode),'Snapshot ancestor is not an ordinary directory')
def _source_pins():
    root=Path(__file__).resolve().parents[5]
    names=('src/pulsemorph/agentic/app/phrase/response_snapshot_v1.py','src/pulsemorph/agentic/app/phrase/manager_response_v9.py','src/pulsemorph/agentic/app/phrase/response_snapshot_admission_v1.py','experiments/agentic/app/import_response_snapshot_v1.py','src/pulsemorph/agentic/app/phrase/response_evidence_v1.py','src/pulsemorph/agentic/app/phrase/runtime.py','src/pulsemorph/agentic/app/phrase/assets.py')
    return {name:_hash((root/name).read_bytes()) for name in names}
def prepare_snapshot_import(runtime):
    """Pins import code/profile only; does not invoke the full evidence loader."""
    _paths();full._guard(runtime);pins=_source_pins()
    plan={'schema':'pulsemorph.response_snapshot_import_plan.v1','snapshot_id':SNAPSHOT_ID,'catalog_admission_sha256':CATALOG_SHA256,'runtime_profile_sha256':_profile(runtime),'environment_sha256':_hash(full._canonical(runtime.environment)),'source_pins':pins,'budget':{'full_verifications':1,'model_calls':0,'compile':0,'ONNX':0,'FK':0,'steps':0},'qualification':'Explicit import preparation only; second source-pinned root admission is required before fast use.'}
    full._guard(runtime);_require(pins==_source_pins(),'Import source changed during preparation');return plan
def import_verified_snapshot(runtime,plan,authority_path,authority_sha256):
    """Exactly one full verification under explicit root hash authority."""
    _paths();_require(not STORE.exists(),'Snapshot import target already consumed')
    _require(isinstance(authority_sha256,str) and len(authority_sha256)==64 and all(c in '0123456789abcdef' for c in authority_sha256),'Explicit root import authority SHA256 required')
    authority_file=Path(authority_path).absolute();authority_reader=_Reader([authority_file.parent]);raw=authority_reader.data(authority_file,authority_sha256,128*1024);authority=json.loads(raw)
    expected={'schema':'pulsemorph.response_snapshot_import_authority.v1','approved':True,'plan_sha256':_hash(_bytes(plan)),'max_full_verifications':1,'max_model_calls':0,'max_compiles':0,'max_ONNX_calls':0,'max_FK_calls':0,'max_steps':0}
    _require(isinstance(authority,dict) and all(type(authority.get(k)) is type(v) and authority.get(k)==v for k,v in expected.items()),'Root import authority does not match')
    _require(plan==prepare_snapshot_import(runtime),'Import plan source/profile changed')
    if not _IMPORT_LOCK.acquire(blocking=False):raise full.ResponseEvidenceBusy('Snapshot import already in progress')
    try:
        STORE.mkdir(parents=True,exist_ok=False);_write(STORE/'import_attempt.json',{'attempted_once':True,'authority_sha256':authority_sha256,'plan_sha256':_hash(_bytes(plan)),'retry_allowed':False})
        try:
            catalog=full.load_response_evidence(runtime,wait_s=30);catalog_raw=_bytes(catalog)
            _require(catalog.get('status')=='available' and _hash(catalog_raw)==CATALOG_SHA256,'Full catalog is not the root-admitted catalog')
            _require(plan==prepare_snapshot_import(runtime),'Import code/profile changed during full verification')
            verified_at=_now()
            with (STORE/'catalog.json').open('xb') as f:f.write(catalog_raw)
            receipt={'schema':'pulsemorph.response_snapshot_import_receipt.v1','snapshot_id':SNAPSHOT_ID,'catalog_sha256':CATALOG_SHA256,'full_verified_at_utc':verified_at,'imported_at_utc':_now(),'runtime_profile_sha256':plan['runtime_profile_sha256'],'environment_sha256':plan['environment_sha256'],'import_source_pins':plan['source_pins'],'authority_sha256':authority_sha256,'plan_sha256':_hash(_bytes(plan)),'full_verification_mode':'frozen_v8_full_raw_loader_once','new_model':0,'new_compile':0,'new_ONNX':0,'new_FK':0,'new_steps':0}
            _write(STORE/'import_receipt.json',receipt)
            envelope={'schema':'pulsemorph.response_snapshot_envelope.v1','snapshot_id':SNAPSHOT_ID,'catalog_sha256':CATALOG_SHA256,'import_receipt_sha256':_hash(_bytes(receipt)),'full_verified_at_utc':verified_at,'requires_second_root_admission':True}
            _write(STORE/'envelope.json',envelope);return {'status':'imported_pending_root_admission','catalog_sha256':CATALOG_SHA256,'import_receipt_sha256':_hash(_bytes(receipt)),'envelope_sha256':_hash(_bytes(envelope))}
        except Exception:
            _write(STORE/'import_failed.json',{'status':'failed','retry_allowed':False,'qualification':'Partial import preserved, not admitted.'});raise
    finally:_IMPORT_LOCK.release()

def load_response_snapshot(runtime):
    try:return _load(runtime)
    except (full.ResponseEvidenceIntegrityError,full.ResponseEvidenceBusy):raise
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:raise full.ResponseEvidenceIntegrityError('Imported response snapshot is incomplete or changed') from error
def _load(runtime):
    _paths();full._guard(runtime);anchor=_admitted_sha()
    if not anchor:
        full._guard(runtime);return {'schema':SCHEMA,'status':'unavailable','reason':'snapshot_not_root_admitted','points':[],'verification':{'mode':'not_admitted','raw_store_checked_this_read':False,'current_profile_checked':True}}
    reader=_Reader([runtime.root,BASE]);module=Path(__file__);reader.hash(module)
    admission=reader.json(ADMISSION_PATH,anchor,16*1024)
    _require(type(admission) is dict and set(admission)=={'schema','snapshot_id','admitted','catalog_sha256','envelope_sha256','import_receipt_sha256'} and admission['schema']=='pulsemorph.response_snapshot_admission.v1' and admission['snapshot_id']==SNAPSHOT_ID and admission['admitted'] is True and admission['catalog_sha256']==CATALOG_SHA256,'Snapshot admission identity mismatch')
    _require(STORE.is_dir() and {p.name for p in STORE.iterdir()}=={'catalog.json','envelope.json','import_receipt.json','import_attempt.json'},'Snapshot file whitelist mismatch')
    for path in STORE.iterdir():reader.safe(path)
    catalog_raw=reader.data(STORE/'catalog.json',CATALOG_SHA256,16*1024);catalog=json.loads(catalog_raw);envelope=reader.json(STORE/'envelope.json',admission['envelope_sha256'],16*1024);receipt=reader.json(STORE/'import_receipt.json',admission['import_receipt_sha256'],32*1024)
    _require(envelope['schema']=='pulsemorph.response_snapshot_envelope.v1' and receipt['schema']=='pulsemorph.response_snapshot_import_receipt.v1' and envelope['snapshot_id']==receipt['snapshot_id']==SNAPSHOT_ID and envelope['catalog_sha256']==receipt['catalog_sha256']==CATALOG_SHA256 and envelope['import_receipt_sha256']==admission['import_receipt_sha256'] and envelope['full_verified_at_utc']==receipt['full_verified_at_utc'],'Admitted snapshot envelope binding mismatch')
    _require(receipt['runtime_profile_sha256']==_profile(runtime) and receipt['environment_sha256']==_hash(full._canonical(runtime.environment)),'Current numeric profile differs from imported observation')
    _require(catalog['schema']==full.SCHEMA and catalog['status']=='available' and catalog['conditioning']==full.CONDITION and [p['gain'] for p in catalog['points']]==[.75,1.,1.25],'Imported catalog schema/conditioning mismatch')
    reader.finish([module,ADMISSION_PATH,STORE/'catalog.json',STORE/'envelope.json',STORE/'import_receipt.json']);full._guard(runtime);_paths()
    _require({p.name for p in STORE.iterdir()}=={'catalog.json','envelope.json','import_receipt.json','import_attempt.json'},'Snapshot files changed during read')
    for path in STORE.iterdir():reader.safe(path)
    result=copy.deepcopy(catalog);result['schema']=SCHEMA;result['imported_verification_proof']=result.pop('proof')
    result['verification']={'mode':'imported_verified_snapshot','raw_store_checked_this_read':False,'current_profile_checked':True,'full_verified_at_utc':receipt['full_verified_at_utc'],'imported_at_utc':receipt['imported_at_utc'],'catalog_sha256':CATALOG_SHA256,'import_receipt_sha256':admission['import_receipt_sha256'],'admission_sha256':anchor,'scope':'Historical root-admitted import. Raw-store corruption/removal after import is not checked by this read. Current profile/native/source/init pins are freshly checked.'}
    return result
def query_response_snapshot(runtime,query):
    query=full._query(query);data=load_response_snapshot(runtime)
    if data['status']!='available':return data
    if query['field']!='first_left_gain' or query['conditioning']!=full.CONDITION:reason='conditioning_mismatch'
    elif query['gain'] is not None and query['gain'] not in (.75,1.,1.25):reason='unobserved_gain'
    else:
        if query['gain'] is not None:data['points']=[p for p in data['points'] if p['gain']==query['gain']]
        data.update(query=query,applicability='exact_condition_observed_points_only');return data
    return {'schema':SCHEMA,'status':'not_applicable','reason':reason,'points':[],'verification':data['verification'],'qualification':data['qualification']}
def recheck_snapshot(runtime):
    """Explicit raw recheck, separate result; never updates/adopts a snapshot."""
    snapshot=load_response_snapshot(runtime)
    if snapshot['status']!='available':return {'schema':'pulsemorph.response_snapshot_recheck.v1','status':'failed','reason':'snapshot_not_admitted','checked_at_utc':_now(),'raw_store_checked_this_read':False,'snapshot_modified':False}
    try:
        catalog=full.load_response_evidence(runtime)
        status=('matched' if _hash(_bytes(catalog))==CATALOG_SHA256 else 'failed') if catalog.get('status')=='available' else 'raw_unavailable'
    except full.ResponseEvidenceBusy:raise
    except full.ResponseEvidenceIntegrityError:status='failed'
    full._guard(runtime)
    return {'schema':'pulsemorph.response_snapshot_recheck.v1','status':status,'checked_at_utc':_now(),'raw_store_checked_this_read':True,'admission_sha256':snapshot['verification']['admission_sha256'],'snapshot_modified':False,'qualification':'Separate current raw check; historical snapshot/import timestamp/admission are unchanged. No model/compile/ONNX/FK/physics.'}
