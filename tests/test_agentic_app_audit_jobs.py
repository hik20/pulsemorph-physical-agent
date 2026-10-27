"""Delivery-auditor tamper fixtures; no model, physics or remote integration calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('app_delivery_audit',ROOT/'experiments/agentic/app/audit_jobs.py')
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)

class AuditorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def write(self,name,data):
        path=self.directory/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data));return path
    def call(self):
        folder='model_calls/00_interpret/'
        plan={'supported':False}
        request=self.write(folder+'request.json',{'purpose':'interpret','schema':{'type':'object'}})
        response=self.write(folder+'response.json',plan)
        receipt={'index':0,'purpose':'interpret','live':True,'request_sha256':AUDIT.sha(request),'response_sha256':AUDIT.sha(response)}
        self.write(folder+'receipt.json',receipt)
        return {'model_calls':[receipt],'live':True,'plan':plan}
    def test_receipts_verify_and_response_tamper_rejected(self):
        job=self.call();self.assertEqual(AUDIT.audit_calls(self.directory,job)['live_completed'],1)
        self.write('model_calls/00_interpret/response.json',{'supported':True})
        with self.assertRaisesRegex(ValueError,'response hash'):AUDIT.audit_calls(self.directory,job)
    def test_unconfirmed_attempt_not_live_completion(self):
        self.write('model_calls/00_interpret/request.json',{'purpose':'interpret'})
        self.write('model_calls/00_interpret/failure.json',{'status':'completion_unconfirmed'})
        result=AUDIT.audit_calls(self.directory,{'model_calls':[],'live':False})
        self.assertEqual(result['completed'],0);self.assertEqual(len(result['completion_unconfirmed']),1)
    def candidate(self,index,score,l2):
        return {'index':index,'score':score,'passed':False,'metrics':{'score':score,'utility_l2':l2,'physical_gate_passed':True,'intent_passed':False}}
    def test_selection_l2_and_legacy_policy_distinct(self):
        baseline=self.candidate(-1,2,5);trial=self.candidate(0,2,4)
        job={'baseline':baseline,'trials':[trial],'selected':baseline,'outcome':'partial'}
        self.assertEqual(AUDIT.audit_selection(job)['selected_index'],-1)
        job['selection_policy']='physical gate, worst normalized error, summed squared normalized error, earlier index'
        with self.assertRaisesRegex(ValueError,'selection policy'):AUDIT.audit_selection(job)
        job['selected']=trial;self.assertEqual(AUDIT.audit_selection(job)['selected_index'],0)
    def test_frozen_targets_differ_from_plan_rejected(self):
        plan={'arm_targets':[1,1,1,1],'timing_mode':'music'}
        job={'plan':plan,'targets':[1,1,1,1,2.4],'tolerances':[.1]*5,'music':{'accent_time_s':2.4},'trials':[]}
        self.write('targets.json',{'targets':job['targets'],'tolerances':job['tolerances'],'plan':plan,'timing_source':'music'})
        self.assertEqual(AUDIT.audit_targets(self.directory,job)['state'],'verified')
        job['music']['accent_time_s']=2.7
        with self.assertRaisesRegex(ValueError,'timing target'):AUDIT.audit_targets(self.directory,job)
    def test_render_audio_link_tamper_rejected(self):
        audio_hash='a'*64;job={'media':{'comparison_url':'fixture'},'music':{'sha256':audio_hash},'selected':{'index':-1}}
        trial=self.write('trials/baseline/manifest.json',{'audio_observation':{'sha256':audio_hash}})
        output=self.directory/'media/comparison.mp4';output.parent.mkdir();output.write_bytes(b'fixture media')
        manifest={'outputs_sha256':{'comparison.mp4':AUDIT.sha(output)},'audio_sha256':audio_hash,
                  'input_manifest_sha256':[AUDIT.sha(trial)]*2,'physics_steps':0,'onnx_calls':0}
        self.write('media/manifest.json',manifest)
        self.assertEqual(AUDIT.audit_media(self.directory,job)['state'],'verified')
        manifest['audio_sha256']='b'*64;self.write('media/manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'soundtrack'):AUDIT.audit_media(self.directory,job)
    def test_nonterminal_job_skipped_without_audit(self):
        self.write('live/state.json',{'id':'live','status':'running'})
        result=AUDIT.audit_jobs(self.directory,check_memory=False)
        self.assertEqual(result['jobs'],[]);self.assertEqual(result['skipped_nonterminal'][0]['id'],'live')
        self.assertFalse(result['coverage_complete']);self.assertFalse(result['passed'])
    def test_partial_and_failed_traces_excluded_from_success(self):
        (self.directory/'trials/partial').mkdir(parents=True)
        self.write('trials/failed/manifest.json',{'status':'failed','artifact_hashes':{},'actual_physics_steps':17})
        result=AUDIT.audit_physics(self.directory,{'trials':[]},None)
        self.assertEqual(len(result),2)
        self.assertFalse(any(row['counted_as_success'] for row in result))
    def test_unique_directory_independent_audit_cached(self):
        from pulsemorph.agentic.app import runtime
        module=SimpleNamespace(_audit=lambda directory:{})
        with patch.object(runtime,'_module',return_value=module),patch.object(runtime,'_audit_app',return_value={'verified':True}) as independent:
            auditor=AUDIT.PhysicsAuditor(ROOT)
            self.assertEqual(auditor.get(self.directory),auditor.get(self.directory))
            self.assertEqual(independent.call_count,1)
    def test_cli_never_overwrites_existing_output(self):
        output=self.write('existing.json',{'preserve':True})
        with patch.object(sys,'argv',['audit_jobs','--jobs-root',str(self.directory),'--output',str(output)]):
            with self.assertRaises(FileExistsError):AUDIT.main()
        self.assertEqual(json.loads(output.read_text()),{'preserve':True})
