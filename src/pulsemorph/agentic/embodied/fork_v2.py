"""Versioned preview proof and complete actual-sample journal parity; no execution on import."""
import copy,json
from .runtime_v2 import PersistentEpisode, ActionRejected, fingerprint, canonical
from .fork_v1 import (PreviewSession as _V1Session, PreviewIntegrityError,
                      PreviewPhysicalFailure, snapshot_episode, fork_episode, _require)

class PreviewSession(_V1Session):
    def __init__(self,parent,**kwargs):
        if not isinstance(parent,PersistentEpisode):
            raise ValueError('Sample-journaling v2 episode required')
        super().__init__(parent,**kwargs)
    def preview(self,action):
        with self._lock,self.parent._lock:
            before=snapshot_episode(self.parent)
            try:self.parent._parse(action)
            except ActionRejected as error:
                _require(snapshot_episode(self.parent)==before,'Compile rejection changed actual parent')
                error.report.update(physics_steps=0,revision_unchanged=before.revision,
                    observation_sha256=before.observation_sha256,
                    parent_snapshot_fingerprint=before.fingerprint)
                for row in error.report.get('violations',[]):
                    row['joint_name']=self.parent.backend.joint_names[row['joint_index']]
                # Compiler extrema contain NumPy scalar floats. Transport must be
                # strict JSON primitives before entering the model context auditor.
                error.report=json.loads(canonical(error.report))
                raise
            receipt=super().preview(action)
            entry=self._previews[receipt['preview_id']]
            samples=entry['branch'].sampled_observations(action['action_id'])
            receipt.update(sampled_observations_sha256=fingerprint(samples),
                           sampled_observations_count=len(samples))
            entry['receipt']=copy.deepcopy(receipt)
            return receipt
    def preview_samples(self,preview_id):
        with self._lock:
            entry=self._previews[preview_id]
            _require(entry['status']=='ready','Only a ready preview has samples')
            samples=entry['branch'].sampled_observations(entry['action']['action_id'])
            _require(fingerprint(samples)==entry['receipt']['sampled_observations_sha256'],
                     'Preview sample journal changed')
            return samples
    def commit(self,value):
        with self._lock,self.parent._lock:
            if type(value) is dict and value.get('commit_id') not in self._commits and value.get('preview_id') in self._previews:
                self.preview_samples(value['preview_id']) # tampering is rejected before actual work
            result=super().commit(value)
            entry=self._previews[value['preview_id']]
            samples=self.parent.sampled_observations(entry['action']['action_id'])
            try:
                _require(fingerprint(samples)==entry['receipt']['sampled_observations_sha256']
                    and samples==entry['branch'].sampled_observations(entry['action']['action_id']),
                    'Actual/preview sampled observation journal parity failed')
            except Exception:
                self.stopped=True;self.parent.stopped=True;raise
            result.update(sampled_observations_sha256=fingerprint(samples),
                          sampled_observations_count=len(samples))
            self._commits[value['commit_id']]=copy.deepcopy(result)
            return result
    def committed_samples(self,commit_id):
        with self._lock:
            receipt=self._commits[commit_id]
            samples=self.parent.sampled_observations(receipt['action_id'])
            _require(fingerprint(samples)==receipt['sampled_observations_sha256'],
                     'Committed observation journal changed')
            return samples
