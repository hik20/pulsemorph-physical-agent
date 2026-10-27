"""Admitted historical snapshot fixtures; no full real scan or controller calls."""
import copy,json,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase import response_snapshot_v1 as s
from pulsemorph.agentic.app.phrase.manager_response_v9 import ManagerResponseV9,PhraseDescriptorManager
def query():return {'schema':'pulsemorph.response_evidence_query.v1','field':'first_left_gain','conditioning':copy.deepcopy(s.full.CONDITION),'gain':None}
def catalog():return {'schema':s.full.SCHEMA,'status':'available','conditioning':copy.deepcopy(s.full.CONDITION),'points':[{'gain':g,'own_q0_ratio_to_unit':r,'temporal_ratio_to_unit':t} for g,r,t in ((.75,.78,.94),(1.,1.,1.),(1.25,1.22,1.09))],'proof':{'immutable_raw_inventories_verified':True},'qualification':'Synthetic test fixture, not physical evidence.'}
class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name);self.store=self.base/'store';self.admission=self.base/'admission.json';self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(s,BASE=self.base,STORE=self.store,ADMISSION_PATH=self.admission))
        self.guard=self.stack.enter_context(patch.object(s.full,'_guard'))
        self.runtime=SimpleNamespace(root=ROOT,environment={},_implementation={},implementation_pins=lambda:{})
    def admitted(self):
        self.store.mkdir();data=catalog();digest=s._hash(s._bytes(data));self.stack.enter_context(patch.object(s,'CATALOG_SHA256',digest));profile=s._profile(self.runtime);time='2026-10-27T13:30:00+00:00'
        receipt={'schema':'pulsemorph.response_snapshot_import_receipt.v1','snapshot_id':s.SNAPSHOT_ID,'catalog_sha256':digest,'full_verified_at_utc':time,'imported_at_utc':time,'runtime_profile_sha256':profile,'environment_sha256':s._hash(s.full._canonical({}))}
        envelope={'schema':'pulsemorph.response_snapshot_envelope.v1','snapshot_id':s.SNAPSHOT_ID,'catalog_sha256':digest,'import_receipt_sha256':s._hash(s._bytes(receipt)),'full_verified_at_utc':time}
        for name,value in (('catalog.json',data),('import_receipt.json',receipt),('envelope.json',envelope),('import_attempt.json',{})):s._write(self.store/name,value)
        admission={'schema':'pulsemorph.response_snapshot_admission.v1','snapshot_id':s.SNAPSHOT_ID,'admitted':True,'catalog_sha256':digest,'envelope_sha256':s._hash(s._bytes(envelope)),'import_receipt_sha256':s._hash(s._bytes(receipt))};s._write(self.admission,admission)
        self.stack.enter_context(patch.object(s,'_admitted_sha',return_value=s._hash(s._bytes(admission))))
    def test_fast_read_no_full_loader_honest_scope_owned(self):
        self.admitted()
        with patch.object(s.full,'load_response_evidence',side_effect=AssertionError('full raw scan forbidden')):
            data=s.load_response_snapshot(self.runtime)
        self.assertEqual(data['verification']['mode'],'imported_verified_snapshot');self.assertFalse(data['verification']['raw_store_checked_this_read']);self.assertTrue(data['verification']['current_profile_checked']);self.assertNotIn('proof',data);self.assertIn('imported_verification_proof',data);self.assertEqual(self.guard.call_count,2)
        data['points'].clear();self.assertEqual(len(s.load_response_snapshot(self.runtime)['points']),3)
    def test_self_rehashed_metadata_cannot_admit_changed_data(self):
        self.admitted();data=catalog();data['points'][0]['own_q0_ratio_to_unit']=99
        (self.store/'catalog.json').write_bytes(s._bytes(data))
        admission=json.loads(self.admission.read_text());admission['catalog_sha256']=s._hash(s._bytes(data));self.admission.write_bytes(s._bytes(admission))
        with self.assertRaises(s.full.ResponseEvidenceIntegrityError):s.load_response_snapshot(self.runtime)
    def test_blank_root_anchor_returns_no_points(self):
        with patch.object(s,'_admitted_sha',return_value=''):
            result=s.load_response_snapshot(self.runtime)
        self.assertEqual(result['status'],'unavailable');self.assertEqual(result['points'],[]);self.assertFalse(result['verification']['raw_store_checked_this_read'])
    def test_no_extra_native_pass_outside_fresh_guards(self):
        self.admitted();self.runtime.implementation_pins=lambda:(_ for _ in ()).throw(AssertionError('extra native pass'))
        self.assertEqual(s.load_response_snapshot(self.runtime)['status'],'available');self.assertEqual(self.guard.call_count,2)
    def test_profile_drift_fails_before_data_use(self):
        self.admitted();self.runtime._implementation={'changed':'profile'}
        with self.assertRaises(s.full.ResponseEvidenceIntegrityError):s.load_response_snapshot(self.runtime)
    def test_unobserved_and_different_condition_no_numbers(self):
        self.admitted();q=query();q['gain']=.85;result=s.query_response_snapshot(self.runtime,q);self.assertEqual(result['reason'],'unobserved_gain');self.assertEqual(result['points'],[])
        q=query();q['conditioning']['other_gains']['second_left_gain']=1.2;result=s.query_response_snapshot(self.runtime,q);self.assertEqual(result['reason'],'conditioning_mismatch');self.assertEqual(result['points'],[])
        q=query();q['gain']=True
        with self.assertRaises(ValueError):s.query_response_snapshot(self.runtime,q)
    def test_raw_recheck_failure_does_not_rewrite_historical_snapshot(self):
        self.admitted();before={p.name:p.read_bytes() for p in self.store.iterdir()}
        with patch.object(s.full,'load_response_evidence',side_effect=s.full.ResponseEvidenceIntegrityError('raw changed')):
            result=s.recheck_snapshot(self.runtime)
        self.assertEqual(result['status'],'failed');self.assertTrue(result['raw_store_checked_this_read']);self.assertFalse(result['snapshot_modified']);self.assertEqual(before,{p.name:p.read_bytes() for p in self.store.iterdir()});self.assertEqual(s.load_response_snapshot(self.runtime)['status'],'available')
    def test_recheck_without_admission_does_not_claim_raw_read(self):
        with patch.object(s,'_admitted_sha',return_value=''),patch.object(s.full,'load_response_evidence') as load:result=s.recheck_snapshot(self.runtime)
        load.assert_not_called();self.assertEqual(result['status'],'failed');self.assertFalse(result['raw_store_checked_this_read'])
    def test_exclusive_import_one_full_verification_no_auto_admission(self):
        data=catalog();self.stack.enter_context(patch.object(s,'CATALOG_SHA256',s._hash(s._bytes(data))))
        plan={'fixture':'explicit import plan'};authority={'schema':'pulsemorph.response_snapshot_import_authority.v1','approved':True,'plan_sha256':s._hash(s._bytes(plan)),'max_full_verifications':1,'max_model_calls':0,'max_compiles':0,'max_ONNX_calls':0,'max_FK_calls':0,'max_steps':0};path=self.base/'authority.json';s._write(path,authority)
        plan.update(runtime_profile_sha256=s._profile(self.runtime),environment_sha256=s._hash(s.full._canonical({})),source_pins={});authority['plan_sha256']=s._hash(s._bytes(plan));path.write_bytes(s._bytes(authority))
        with patch.object(s,'prepare_snapshot_import',return_value=plan),patch.object(s.full,'load_response_evidence',return_value=data) as load:
            result=s.import_verified_snapshot(self.runtime,plan,path,s._hash(path.read_bytes()));self.assertEqual(result['status'],'imported_pending_root_admission');load.assert_called_once_with(self.runtime,wait_s=30)
            with self.assertRaises(s.full.ResponseEvidenceIntegrityError):s.import_verified_snapshot(self.runtime,plan,path,s._hash(path.read_bytes()))
            self.assertEqual(load.call_count,1);self.assertFalse(self.admission.exists())
    def test_prepare_and_missing_authority_never_full_scan(self):
        with patch.object(s.full,'load_response_evidence',side_effect=AssertionError('full scan forbidden')):
            plan=s.prepare_snapshot_import(self.runtime);self.assertEqual(plan['budget']['full_verifications'],1)
            with self.assertRaises(s.full.ResponseEvidenceIntegrityError):s.import_verified_snapshot(self.runtime,plan,None,None)
    def test_manager_persists_mode_before_model(self):
        self.admitted();m=ManagerResponseV9.__new__(ManagerResponseV9);m.jobs_dir=self.base/'jobs';(m.jobs_dir/'job').mkdir(parents=True);m.runtime=self.runtime;m._verify_startup=lambda:None;m._check=lambda _:None;m.updates=[];m._update=lambda identity,**kw:m.updates.append(kw)
        with patch.object(PhraseDescriptorManager,'_context',return_value={'fixture_prior':[1]}):result=m._context('job')
        self.assertEqual(result['measured_motor_response_evidence']['verification']['mode'],'imported_verified_snapshot');self.assertFalse(m.updates[0]['measured_motor_response_receipt']['verification']['raw_store_checked_this_read'])

if __name__=='__main__':unittest.main()
