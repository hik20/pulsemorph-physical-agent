"""Environment-only hosted model selection, honest model identity, no Codex dependency."""
from pulsemorph.agentic.app.providers import ProviderResult
from pulsemorph.agentic.performance.nemotron_v1 import NemotronProvider
from pulsemorph.agentic.performance.nemotron_openrouter_v1 import NemotronOpenRouterProvider

class HostedProvider:
    def __init__(self,kind):
        if kind=='openrouter_free':self.transport=NemotronOpenRouterProvider(free=True,max_tokens=4096)
        elif kind=='nvidia_nim':self.transport=NemotronProvider(max_tokens=4096)
        else:raise ValueError('Explicit supported hosted provider required')
        self.model=self.transport.model
    def status(self):
        value=self.transport.status();return {**value,'enabled':value['configured']is True}
    def generate_json(self,prompt,schema,**kwargs):
        result=self.transport.generate_json(prompt,schema,**kwargs);r=result.receipt
        receipt={**r,'model_returned':r['model_reported'],'elapsed_seconds':r['elapsed_s'],'trust_boundary':'Hosted typed proposal only; no execution authority; no tools or retries.'}
        return ProviderResult(data=result.data,receipt=receipt)
