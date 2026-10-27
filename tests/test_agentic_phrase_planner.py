"""Synthetic planner fixtures; never invoke actual model, ONNX or physics."""
import copy,hashlib,tempfile,unittest,wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from threading import Event
from pulsemorph.agentic.phrase_planner import prepare_phrase_plan,generate_phrase_plan,validate_program,validate_music_summary,verify_music_binding,PROGRAM_SCHEMA,PROVIDER_SCHEMA,audit_provider_schema,safe_provider_error_category

class PhrasePlannerTests(unittest.TestCase):
    def setUp(self):
        self.summary={'schema':'fixture','sha256':'a'*64,'duration_s':8,'sample_rate':48000,'sample_frames':384000,'channels':1,'pcm_bits':16,'rms':.1,'sections':[{'start_s':0,'end_s':4,'rms':.1},{'start_s':4,'end_s':8,'rms':.2}],'bpm':None,'bpm_source':'not inferred'}
        self.program={'schema':'pulsemorph.phrase_program.v1','order':['wave','high5'],'heading_policy':'align_join_heading','first_left_gain':.8,'first_right_gain':1.,'second_left_gain':1.15,'second_right_gain':1.1,'rationale':'Fixture bounded proposal; no physical guarantee.'}
    def provider(self,program=None):
        return SimpleNamespace(model='gpt-6.1-sol',generate_json=Mock(return_value=SimpleNamespace(data=copy.deepcopy(self.program if program is None else program),receipt={'provider':'fixture','model_requested':'gpt-6.1-sol','model_returned':None,'tool_events':0,'live':False,'usage':{'input_tokens':0},'raw_secret':'must not copy'})))
    def test_prepare_preserves_instruction_and_measured_sections(self):
        instruction='Önce Wave; sol/sağ 80/100, sonra iki kolu belirgin kullan.'
        request=prepare_phrase_plan(self.summary,instruction);self.assertEqual(request['instruction'],instruction);self.assertEqual(request['music_summary'],self.summary);self.assertIn('not authored beat',request['prompt']);self.assertEqual(request['fixed_clock']['join_s'],[3.5,4.5]);self.assertEqual(request['program_schema'],PROGRAM_SCHEMA)
    def test_one_mock_call_exact_model_schema_null_returned(self):
        provider=self.provider();request=prepare_phrase_plan(self.summary,'fixture instruction');r=generate_phrase_plan(provider,request);provider.generate_json.assert_called_once();self.assertIsNone(r['receipt']['model_returned']);self.assertNotIn('raw_secret',r['receipt']);self.assertFalse(r['physical_validation'])
    def test_provider_failure_never_retried(self):
        provider=self.provider();provider.generate_json.side_effect=RuntimeError('fixture failure')
        with self.assertRaises(RuntimeError):generate_phrase_plan(provider,prepare_phrase_plan(self.summary,'fixture instruction'))
        provider.generate_json.assert_called_once()
    def test_cancel_before_call(self):
        provider=self.provider();event=Event();event.set()
        with self.assertRaises(ValueError):generate_phrase_plan(provider,prepare_phrase_plan(self.summary,'fixture instruction'),cancel_event=event)
        provider.generate_json.assert_not_called()
    def test_malformed_unknown_bool_nonfinite_gain_rejected(self):
        for value in [None,[],{'schema':'wrong'}]:
            with self.assertRaises(Exception):validate_program(value)
        for gain in [True,float('nan'),float('inf'),.749,1.251]:
            program=copy.deepcopy(self.program);program['first_left_gain']=gain
            with self.assertRaises(Exception):validate_program(program)
        program=copy.deepcopy(self.program);program['timing_shift_s']=.1
        with self.assertRaises(Exception):validate_program(program)
    def test_catalog_order_heading_and_rationale(self):
        for key,value in [('order',['wave','wave']),('order',['walk','high5']),('order',['high5']),('heading_policy','new_root_path'),('rationale','x'*1201)]:
            program=copy.deepcopy(self.program);program[key]=value
            with self.assertRaises(Exception):validate_program(program)
        program=copy.deepcopy(self.program);program['order']=['high5','wave'];self.assertEqual(validate_program(program)['order'],['high5','wave'])
    def test_modified_prepared_request_rejected_before_call(self):
        request=prepare_phrase_plan(self.summary,'fixture instruction');request['prompt']+=' shell command';provider=self.provider()
        with self.assertRaises(ValueError):generate_phrase_plan(provider,request)
        provider.generate_json.assert_not_called()
    def test_unrequested_model_and_tool_receipt_rejected(self):
        request=prepare_phrase_plan(self.summary,'fixture instruction');provider=self.provider();provider.model='other'
        with self.assertRaises(ValueError):generate_phrase_plan(provider,request)
        provider.generate_json.assert_not_called();provider=self.provider();provider.generate_json.return_value.receipt['tool_events']=1
        with self.assertRaises(ValueError):generate_phrase_plan(provider,request)
    def test_actual_PCM_binding_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'owned.wav'
            with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);w.writeframes(b'\0\0'*64000)
            summary=dict(self.summary,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sample_rate=8000,sample_frames=64000)
            self.assertEqual(verify_music_binding(summary,path),summary);path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaises(ValueError):verify_music_binding(summary,path)
    def test_provider_schema_separate_local_unique_guard(self):
        self.assertTrue(PROGRAM_SCHEMA['properties']['order']['uniqueItems'])
        self.assertNotIn('uniqueItems',PROVIDER_SCHEMA['properties']['order'])
        self.assertTrue(audit_provider_schema(PROVIDER_SCHEMA)['offline_keyword_audit_passed'])
        provider=self.provider();generate_phrase_plan(provider,prepare_phrase_plan(self.summary,'fixture instruction'))
        self.assertEqual(provider.generate_json.call_args.args[1],PROVIDER_SCHEMA)
        duplicate=dict(self.program,order=['wave','wave']);provider=self.provider(duplicate)
        with self.assertRaises(Exception):generate_phrase_plan(provider,prepare_phrase_plan(self.summary,'fixture instruction'))
        provider.generate_json.assert_called_once()
    def test_unsupported_transport_keyword_fails_offline(self):
        schema=copy.deepcopy(PROVIDER_SCHEMA);schema['properties']['order']['uniqueItems']=True
        with self.assertRaisesRegex(ValueError,'unsupported transport'):audit_provider_schema(schema)
        schema=copy.deepcopy(PROVIDER_SCHEMA);schema['properties']['order']['allOf']=[]
        with self.assertRaises(ValueError):audit_provider_schema(schema)
    def test_safe_static_error_category_no_raw_text(self):
        from pulsemorph.agentic.app.providers import ProviderError
        self.assertEqual(safe_provider_error_category(ProviderError('Codex failed; check local authentication/model access')),'CLI_NONZERO_EXIT_CAUSE_UNCONFIRMED')
        secret='secret diagnostic content'
        self.assertEqual(safe_provider_error_category(ProviderError(secret)),'PROVIDER_ERROR_UNCLASSIFIED')
        self.assertNotIn(secret,safe_provider_error_category(RuntimeError(secret)))

    def test_summary_must_be8s_finite_measured_counts(self):
        for key,value in [('duration_s',4),('sample_rate',True),('sample_frames',3),('rms',float('nan')),('channels',True)]:
            summary=dict(self.summary);summary[key]=value
            with self.assertRaises(ValueError):validate_music_summary(summary)
if __name__=='__main__':unittest.main()
