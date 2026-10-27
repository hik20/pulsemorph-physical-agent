"""Exact applicability, integrity and pre-planning persistence; no physical calls."""
import copy,json,stat,sys,tempfile,unittest,threading,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase import response_evidence_v1 as e
from pulsemorph.agentic.app.phrase.manager_response_v8 import ManagerResponseV8,PhraseDescriptorManager
def query():return {'schema':'pulsemorph.response_evidence_query.v1','field':'first_left_gain','conditioning':copy.deepcopy(e.CONDITION),'gain':None}
def evidence():return {**e._base('available'),'points':[{'gain':g,'own_q0_ratio_to_unit':r,'temporal_ratio_to_unit':t} for g,r,t in ((.75,.78,.94),(1.,1.,1.),(1.25,1.22,1.09))]}
class ResponseEvidenceTests(unittest.TestCase):
    def test_exact_three_points_no_interpolation_and_owned_query(self):
        original=evidence()
        with patch.object(e,'load_response_evidence',side_effect=lambda _:copy.deepcopy(original)):
            allpoints=e.query_response_evidence(None,query());self.assertEqual(len(allpoints['points']),3)
            one=query();one['gain']=.75;result=e.query_response_evidence(None,one);self.assertEqual(len(result['points']),1);self.assertEqual(result['points'][0]['gain'],.75)
            result['query']['conditioning']['order'].reverse();self.assertEqual(one['conditioning']['order'],['wave','high5'])
            missing=query();missing['gain']=.85;result=e.query_response_evidence(None,missing);self.assertEqual(result['status'],'not_applicable');self.assertEqual(result['reason'],'unobserved_gain');self.assertEqual(result['points'],[]);self.assertNotIn('ratios',result)
    def test_different_valid_condition_or_supported_field_has_no_numbers(self):
        variants=[]
        for key,value in (('heading_policy','recorded_world'),('order',['high5','wave'])):
            q=query();q['conditioning'][key]=value;variants.append(q)
        q=query();q['conditioning']['other_gains']['second_left_gain']=1.2;variants.append(q)
        q=query();q['field']='second_left_gain';q['conditioning']['other_gains']={'first_left_gain':1.,'first_right_gain':1.,'second_right_gain':1.};variants.append(q)
        with patch.object(e,'load_response_evidence',return_value=evidence()):
            for q in variants:
                result=e.query_response_evidence(None,q);self.assertEqual(result['reason'],'conditioning_mismatch');self.assertEqual(result['points'],[])
    def test_malformed_boolean_nan_unknown_field_and_shape_reject(self):
        for value in (True,float('nan'),float('inf'),'1'):
            q=query();q['gain']=value
            with self.subTest(value=value),self.assertRaises(ValueError):e._query(q)
        for q in ([],None,{**query(),'extra':1},{**query(),'field':'arbitrary'}, {**query(),'conditioning':{}}):
            with self.assertRaises(ValueError):e._query(q)
        q=query();q['gain']=10**1000
        with patch.object(e,'load_response_evidence',return_value=evidence()):self.assertEqual(e.query_response_evidence(None,q)['status'],'not_applicable')
    def test_integrity_failure_never_becomes_unknown(self):
        with patch.object(e,'load_response_evidence',side_effect=e.ResponseEvidenceIntegrityError('fixture changed')):
            with self.assertRaises(e.ResponseEvidenceIntegrityError):e.query_response_evidence(None,query())
    def test_missing_store_vs_partial_or_tampered_index(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);pilot=base/'pilot';audit=base/'audit';runtime=SimpleNamespace(root=ROOT,python=base/'env/Scripts/python.exe',assets=SimpleNamespace(directory=base/'assets'),_implementation={},_unchanged=lambda:None)
            with patch.multiple(e,ROOT=base,PILOT=pilot,AUDIT=audit):
                self.assertEqual(e.load_response_evidence(runtime)['status'],'unavailable')
                pilot.mkdir()
                with self.assertRaises(e.ResponseEvidenceIntegrityError):e.load_response_evidence(runtime)
                audit.mkdir();(pilot/'prepared.json').write_text('{}')
                with self.assertRaises(e.ResponseEvidenceIntegrityError):e.load_response_evidence(runtime)
    def test_dangling_reparse_cannot_masquerade_as_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);pilot=base/'absent-pilot';audit=base/'absent-audit';original=Path.lstat
            def fake(path,*a,**kw):
                if path==pilot:return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=1024)
                return original(path,*a,**kw)
            with patch.multiple(e,ROOT=base,PILOT=pilot,AUDIT=audit),patch.object(Path,'lstat',fake):
                with self.assertRaises(e.ResponseEvidenceIntegrityError):e._ordinary_store_paths()
    def manager(self,directory):
        m=ManagerResponseV8.__new__(ManagerResponseV8);m.jobs_dir=directory;m.runtime=object();m._verify_startup=lambda:None;m._check=lambda _:None;m.updates=[];m._update=lambda identity,**kw:m.updates.append((identity,kw));return m
    def test_context_evidence_persisted_exclusive_before_model(self):
        with tempfile.TemporaryDirectory() as temp:
            m=self.manager(Path(temp));(m.jobs_dir/'job').mkdir();source={'prior':{'values':[1]}}
            with patch.object(PhraseDescriptorManager,'_context',return_value=source),patch('pulsemorph.agentic.app.phrase.manager_response_v8.load_response_evidence',return_value=evidence()) as load:
                result=m._context('job');path=m.jobs_dir/'job/measured_motor_response_evidence.json';self.assertTrue(path.is_file());self.assertEqual(json.loads(path.read_text())['status'],'available');self.assertEqual(m.updates[0][1]['measured_motor_response_receipt']['file'],path.name)
                load.assert_called_once_with(m.runtime,wait_s=30)
                result['prior']['values'].append(2);self.assertEqual(source['prior']['values'],[1])
                with self.assertRaises(FileExistsError):m._context('job')
    def test_startup_failure_is_safe_integrity_error(self):
        m=self.manager(Path('.'));m._verify_startup=lambda:(_ for _ in ()).throw(ValueError('private fixture path'))
        with self.assertRaises(e.ResponseEvidenceIntegrityError) as caught:m.describe_response_evidence()
        self.assertNotIn('private fixture path',str(caught.exception))
    def test_nonblocking_verification_and_bounded_internal_wait(self):
        lock=threading.Lock();lock.acquire()
        with patch.object(e,'_VERIFY_LOCK',lock),patch.object(e,'_load',return_value=evidence()) as load:
            with self.assertRaises(e.ResponseEvidenceBusy):e.load_response_evidence(None)
            load.assert_not_called()
            releaser=threading.Timer(.02,lock.release);releaser.start()
            try:self.assertEqual(e.load_response_evidence(None,wait_s=.2)['status'],'available')
            finally:releaser.cancel()
            self.assertFalse(lock.locked())
            for bad in (True,-1,31,float('nan'),10**1000):
                with self.subTest(bad=bad),self.assertRaises(ValueError):e.load_response_evidence(None,wait_s=bad)
    def test_late_added_unrecorded_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot=Path(temp);inventories={}
            for label in ('unit','low','high'):
                directory=pilot/'trials'/label;directory.mkdir(parents=True);(directory/'saved').write_bytes(b'fixture');inventories[label]={'saved':'unused fixture digest'}
            with patch.object(e,'PILOT',pilot):
                e._raw_sets_unchanged(inventories)
                (pilot/'trials/low/new_unrecorded').write_bytes(b'changed during scan')
                with self.assertRaises(e.ResponseEvidenceIntegrityError):e._raw_sets_unchanged(inventories)

if __name__=='__main__':unittest.main()
