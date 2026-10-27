"""Bounded OpenRouter Nemotron visual critic; no tools, retries or execution authority."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request
import jsonschema
from pulsemorph.agentic.app.providers import ProviderError, ProviderResult

MODEL = 'nvidia/nemotron-3.5-super-vl'
FREE_MODEL = MODEL + ':free'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'

def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def _reject_constant(value):
    raise ValueError('Nonfinite JSON')

def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key] = value
    return result

def _load(value):
    return json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_pairs)

class NemotronOpenRouterProvider:
    name = 'openrouter'
    model = MODEL

    def __init__(self, *, opener=None, max_tokens=4096, free=True):
        if type(free) is not bool: raise ValueError('Explicit free bool required')
        self.model = FREE_MODEL if free else MODEL
        if type(max_tokens) is not int or not 1 <= max_tokens <= 16384:
            raise ValueError('max_tokens must be1..16384')
        self.max_tokens = max_tokens
        self._live_transport = opener is None
        self._open = opener or urllib.request.urlopen

    def status(self):
        return {'provider': self.name, 'model_requested': self.model,
                'configured': bool(os.environ.get('OPENROUTER_API_KEY')),
                'authentication_verified': False, 'live_use_verified': False}

    def generate_json(self, prompt, schema, *, timeout=120, cancel_event=None, image_paths=None):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError('timeout must be finite and <=120 seconds')
        if type(prompt) is not str or not 1 <= len(prompt.encode()) <= 256*1024:
            raise ValueError('Bounded text prompt required')
        jsonschema.Draft202012Validator.check_schema(schema)
        if cancel_event and cancel_event.is_set(): raise ProviderError('Generation cancelled')
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key: raise ProviderError('OPENROUTER_API_KEY is not configured')
        paths = [] if image_paths is None else image_paths
        if type(paths) not in (list, tuple) or len(paths) > 4: raise ValueError('At most4 image files')
        content = [{'type': 'text', 'text': prompt + '\nReturn ONLY JSON matching this schema:\n' + _json(schema).decode()}]
        images = []; total = 0
        for item in paths:
            path = Path(item)
            if path.is_symlink() or not path.is_file(): raise ValueError('Existing regular image required')
            if path.stat().st_size > 4*1024*1024: raise ValueError('Image exceeds4MiB')
            raw = path.read_bytes(); total += len(raw)
            if total > 12*1024*1024: raise ValueError('Images exceed12MiB')
            mime = 'image/png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if raw.startswith(b'\xff\xd8\xff') else None
            if mime is None: raise ValueError('PNG or JPEG required')
            images.append({'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'mime': mime})
            content.append({'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(raw).decode('ascii')}})
        body = _json({'model': self.model, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'body_critic', 'strict': True, 'schema': schema}}, 'provider': {'require_parameters': True}, 'messages': [{'role': 'user', 'content': content}],
                      'max_tokens': self.max_tokens, 'temperature': 0.2, 'stream': False})
        request = urllib.request.Request(ENDPOINT, data=body,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        started = time.monotonic()
        try:
            with self._open(request, timeout=timeout) as response:
                raw_response = response.read(2*1024*1024+1)
            if len(raw_response) > 2*1024*1024: raise ValueError('Response exceeds2MiB')
            value = _load(raw_response)
            if value.get('model') not in ((MODEL, FREE_MODEL) if self.model == FREE_MODEL else (MODEL,)): raise ValueError('Returned model identity mismatch')
            choices = value['choices']
            if len(choices) != 1 or choices[0].get('finish_reason') != 'stop': raise ValueError('Incomplete response')
            message = choices[0]['message']
            if message.get('tool_calls') or message.get('function_call'): raise ValueError('Tool response forbidden')
            data = _load(message['content'])
            if type(data) is not dict: raise ValueError('JSON object required')
            jsonschema.validate(data, schema)
            if cancel_event and cancel_event.is_set(): raise ProviderError('Generation cancelled after request')
        except Exception:
            # HTTP diagnostics may contain credentials or request content. Never persist them.
            raise ProviderError('OpenRouter generation failed; no automatic retry') from None
        usage = value.get('usage', {})
        safe_usage = {k: v for k, v in usage.items() if k in ('prompt_tokens', 'completion_tokens', 'total_tokens') and type(v) is int and v >= 0}
        receipt = {'schema': 'pulsemorph.nemotron_openrouter_receipt.v1', 'provider': self.name,
            'model_requested': self.model, 'model_reported': value['model'], 'live': self._live_transport,
            'elapsed_s': time.monotonic()-started, 'tool_events': 0, 'attempts': 1,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'schema_sha256': hashlib.sha256(_json(schema)).hexdigest(),
            'request_sha256': hashlib.sha256(body).hexdigest(),
            'response_sha256': hashlib.sha256(raw_response).hexdigest(), 'images': images, 'usage': safe_usage}
        return ProviderResult(data=data, receipt=receipt)
