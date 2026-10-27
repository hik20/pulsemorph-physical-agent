"""Fast admitted historical evidence; goals and initial prompt binding inherited."""
import copy,hashlib
from .manager_descriptor_v6 import PhraseDescriptorManager
from .manager import write
from .response_evidence_v1 import ResponseEvidenceIntegrityError
from .response_snapshot_v1 import load_response_snapshot,query_response_snapshot,recheck_snapshot
class ManagerResponseV9(PhraseDescriptorManager):
    def _response_guard(self):
        try:self._verify_startup()
        except Exception as error:raise ResponseEvidenceIntegrityError('Restart required for response snapshot integrity') from error
    def describe_response_evidence(self):
        self._response_guard();result=load_response_snapshot(self.runtime);self._response_guard();return copy.deepcopy(result)
    def query_response_evidence(self,query):
        self._response_guard();result=query_response_snapshot(self.runtime,query);self._response_guard();return copy.deepcopy(result)
    def recheck_response_evidence(self):
        self._response_guard();result=recheck_snapshot(self.runtime);self._response_guard();return copy.deepcopy(result)
    def _context(self,identity):
        context=copy.deepcopy(super()._context(identity));self._check(identity);evidence=self.describe_response_evidence();self._check(identity)
        path=self.jobs_dir/identity/'measured_motor_response_evidence.json';write(path,evidence,exclusive=True);self._check(identity)
        self._update(identity,measured_motor_response_receipt={'schema':evidence['schema'],'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':evidence['status'],'kind':'root_admitted_imported_snapshot','verification':copy.deepcopy(evidence.get('verification',{})),'goal_or_control_authority':False})
        context['measured_motor_response_evidence']=copy.deepcopy(evidence);return context

PhraseSnapshotManager=ManagerResponseV9
