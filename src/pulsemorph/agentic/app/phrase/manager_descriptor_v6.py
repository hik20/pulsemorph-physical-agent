"""Persist descriptive source measurements before planning; keep goals intact."""
import copy
import hashlib
from .manager_advice_v5 import PhraseAdviceManager
from .manager import write
from .source_descriptor_v1 import describe_sources


class PhraseDescriptorManager(PhraseAdviceManager):
    def describe_sources(self):
        self._verify_startup()
        return describe_sources(self.runtime.assets)

    def _context(self, identity):
        context = copy.deepcopy(super()._context(identity))
        self._check(identity)
        descriptor = self.describe_sources()
        path = self.jobs_dir / identity / 'source_descriptor.json'
        write(path, descriptor, exclusive=True)
        self._check(identity)
        self._update(identity, source_descriptor_receipt={
            'schema': descriptor['schema'], 'file': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'kind': 'source_reference_measurements_only',
            'physical_response_or_feasibility': False})
        context['source_reference_descriptor'] = descriptor
        return context
