import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pulsemorph.agentic.performance.nemotron_v1 import NemotronProvider, MODEL, ProviderError

SCHEMA={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}

class NemotronTests(unittest.TestCase):
    def response(self, content='{"ok":true}', model=MODEL):
        return io.BytesIO(json.dumps({'model':model,'choices':[{'finish_reason':'stop','message':{'content':content}}], 'usage':{'prompt_tokens':3}}).encode())

    @patch.dict(os.environ, {'NVIDIA_API_KEY':'fixture-secret'})
    def test_transport_and_private_receipt(self):
        seen=[]
        def open_(request,timeout):
            seen.append((request,timeout));return self.response()
        result=NemotronProvider(opener=open_).generate_json('critic',SCHEMA)
        body=json.loads(seen[0][0].data)
        self.assertEqual(body['model'],MODEL);self.assertNotIn('tools',body)
        self.assertFalse(result.receipt['live']);self.assertEqual(result.data,{'ok':True})
        self.assertNotIn('fixture-secret',json.dumps(result.receipt));self.assertEqual(len(seen),1)

    @patch.dict(os.environ, {'NVIDIA_API_KEY':'fixture-secret'})
    def test_timeout_one_attempt_safe_error(self):
        calls=[]
        def open_(*a,**k):calls.append(1);raise TimeoutError('fixture-secret')
        with self.assertRaisesRegex(ProviderError,'no automatic retry') as caught:
            NemotronProvider(opener=open_).generate_json('x',SCHEMA)
        self.assertNotIn('fixture-secret',str(caught.exception));self.assertEqual(len(calls),1)

    @patch.dict(os.environ, {'NVIDIA_API_KEY':'fixture-secret'})
    def test_images_only_in_request_memory(self):
        captured=[]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'own.png';path.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            def open_(request,timeout):captured.append(json.loads(request.data));return self.response()
            result=NemotronProvider(opener=open_).generate_json('critic',SCHEMA,image_paths=[path])
        self.assertTrue(captured[0]['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
        receipt=json.dumps(result.receipt)
        self.assertNotIn('base64',receipt);self.assertNotIn('own.png',receipt)
        self.assertEqual(result.receipt['images'][0]['bytes'],15)

    @patch.dict(os.environ, {'NVIDIA_API_KEY':'fixture-secret'})
    def test_malformed_or_wrong_model_never_success(self):
        for content,model in [('{"ok":1}',MODEL),('{"ok":true,"ok":false}',MODEL),('{"ok":NaN}',MODEL),('{"ok":true}','other')]:
            with self.subTest(content=content,model=model),self.assertRaises(ProviderError):
                NemotronProvider(opener=lambda *a,**k:self.response(content,model)).generate_json('x',SCHEMA)

    def test_bounds_and_missing_key_before_network(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ProviderError):NemotronProvider().generate_json('x',SCHEMA)
        for timeout in (True,float('nan'),121):
            with self.assertRaises(ValueError):NemotronProvider().generate_json('x',SCHEMA,timeout=timeout)
