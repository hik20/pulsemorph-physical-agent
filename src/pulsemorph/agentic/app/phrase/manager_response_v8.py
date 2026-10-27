"""Persist exact-condition measured evidence before planning; retain fixed goals."""
import copy,hashlib
from .manager_descriptor_v6 import PhraseDescriptorManager
from .manager import write
from .response_evidence_v1 import load_response_evidence,query_response_evidence,ResponseEvidenceIntegrityError

class ManagerResponseV8(PhraseDescriptorManager):
    def _response_guard(self):
        try:self._verify_startup()
        except Exception as error:raise ResponseEvidenceIntegrityError('Restart required for response evidence integrity') from error
    def describe_response_evidence(self):
        self._response_guard();result=load_response_evidence(self.runtime);self._response_guard();return copy.deepcopy(result)
    def query_response_evidence(self,query):
        self._response_guard();result=query_response_evidence(self.runtime,query);self._response_guard();return copy.deepcopy(result)
    def _context(self,identity):
        context=copy.deepcopy(super()._context(identity));self._check(identity)
        # Wait only for read verification, never retry a provider/model call.
        # Worst case is30s queue wait plus a full verification (~15s observed).
        self._response_guard();evidence=load_response_evidence(self.runtime,wait_s=30);self._response_guard();self._check(identity)
        path=self.jobs_dir/identity/'measured_motor_response_evidence.json'
        write(path,evidence,exclusive=True);self._check(identity)
        self._update(identity,measured_motor_response_receipt={'schema':evidence['schema'],'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':evidence['status'],'kind':'conditional_observed_points_only','goal_or_control_authority':False})
        context['measured_motor_response_evidence']=copy.deepcopy(evidence)
        return context

PhraseResponseManager=ManagerResponseV8
