import io,json,os,unittest
from unittest.mock import patch
from pulsemorph.agentic.performance.nemotron_openrouter_v1 import NemotronOpenRouterProvider,MODEL,FREE_MODEL,ProviderError

class OpenRouterTests(unittest.TestCase):
    @patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture-private'})
    def test_free_schema_transport_no_paid_fallback(self):
        calls=[]
        def open_(request,timeout):
            calls.append(json.loads(request.data))
            return io.BytesIO(json.dumps({'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}]}).encode())
        result=NemotronOpenRouterProvider(opener=open_).generate_json('critic',{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False})
        self.assertEqual(calls[0]['model'],FREE_MODEL)
        self.assertEqual(calls[0]['response_format']['type'],'json_schema')
        self.assertFalse(result.receipt['live']);self.assertNotIn('fixture-private',json.dumps(result.receipt))
        self.assertEqual(result.receipt['model_requested'],FREE_MODEL)
        self.assertEqual(len(calls),1)

    @patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture-private'})
    def test_failure_single_attempt_no_fallback(self):
        calls=[]
        def open_(*a,**k):calls.append(1);raise TimeoutError('fixture-private')
        with self.assertRaises(ProviderError):NemotronOpenRouterProvider(opener=open_).generate_json('x',{})
        self.assertEqual(len(calls),1)

    def test_explicit_paid_and_missing_key(self):
        self.assertEqual(NemotronOpenRouterProvider(free=False).model,MODEL)
        with self.assertRaises(ValueError):NemotronOpenRouterProvider(free='false')
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ProviderError):
            NemotronOpenRouterProvider().generate_json('x',{})
