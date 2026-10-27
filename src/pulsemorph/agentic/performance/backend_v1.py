"""Whole-body reference tracker adapter. Construction executes the motor.

All 29 authored joint references reach the fixed actor. Fourteen arms retain the
previously witnessed classical servo; legs/trunk retain the pretrained actor.
Fixed initial torso orientation is a declared carrier, not inferred reference FK.
No stepping, balance, choreography or musical quality is established by this API.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import HybridArmBackend
from pulsemorph.agentic.embodied.runtime_v1 import canonical

SCHEMA = 'pulsemorph.performance_execution_authority.v1'
MAX_CONTROL_STEPS = 1200


def validate_authority(value):
    if type(value) is not dict or set(value) != {'schema', 'session_id', 'max_control_steps', 'source_pins'}:
        raise ValueError('Exact performance authority required')
    if value['schema'] != SCHEMA or type(value['max_control_steps']) is not int or not 1 <= value['max_control_steps'] <= MAX_CONTROL_STEPS:
        raise ValueError('Performance control cap must be integer 1..1200')
    if type(value['session_id']) is not str or not 1 <= len(value['session_id']) <= 160:
        raise ValueError('Bounded session identity required')
    value['session_id'].encode('utf8')
    if type(value['source_pins']) is not dict or not value['source_pins']:
        raise ValueError('Pinned implementation and assets required')
    for path, digest in value['source_pins'].items():
        if type(path) is not str or type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Exact SHA256 source pins required')
    return copy.deepcopy(value)


def validate_future(q, dq, ranges):
    q = np.asarray(q, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64)
    limits = np.asarray(ranges, dtype=np.float64)
    if q.shape != (4, 29) or dq.shape != (4, 29) or limits.shape != (29, 2):
        raise ValueError('Four future samples of all29 joints required')
    if not all(np.isfinite(a).all() for a in (q, dq, limits)) or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError('Finite ordered joint bounds required')
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]) or np.any(np.abs(dq) > 2.0):
        raise ValueError('Future joint position/rate bound violated; no clamp')
    return q.copy(), dq.copy()


class PerformanceBackend(HybridArmBackend):
    def __init__(self, root, *, authority):
        approved = validate_authority(authority)
        own = str(Path(__file__).resolve())
        if approved['source_pins'].get(own) != hashlib.sha256(Path(own).read_bytes()).hexdigest():
            raise ValueError('Authority must pin performance backend')
        # The frozen constructor permits <=500 and performs initialization plus
        # exactly one warmup, with no online advance. Its rule remains unchanged.
        bootstrap = copy.deepcopy(approved)
        bootstrap.update(schema='pulsemorph.embodied_execution_authority.v1',
                         max_control_steps=min(500, approved['max_control_steps']))
        super().__init__(root, authority=bootstrap)
        if self.control_index != 0 or self.step_count != 0 or len(self.calls) != 1:
            raise RuntimeError('Bootstrap unexpectedly performed online work')
        self.authority = approved
        self._performance_authority_sha = hashlib.sha256(canonical(approved)).hexdigest()
        self.performance_contract = {
            'schema': 'pulsemorph.performance_backend.v1',
            'reference_joints': list(self.joint_names),
            'future_offsets_s': [.02, .04, .08, .16],
            'anchor': 'fixed_initial_torso_world_orientation; no planned torso FK',
            'actuation': '14 classical arm servos; 15 unchanged pretrained leg/trunk outputs',
            'qualifications': ['All29 reference q/dq reach actor inputs.',
                               'Whole-body tracking and balance require physical evidence.',
                               'No root trajectory, footstep planner or contact-plan control.',
                               'Existing joint aggregate force caps and limited physical stops unchanged.']}

    def _check_authority(self):
        if hashlib.sha256(canonical(self.authority)).hexdigest() != self._performance_authority_sha:
            raise RuntimeError('Performance authority changed')

    def advance(self, future_q, future_dq):
        self._check_authority()
        q, dq = validate_future(future_q, future_dq, self.joint_ranges)
        return super().advance(q, dq)

    def closure_sha256(self):
        parent = super().closure_sha256()
        return hashlib.sha256(parent.encode() + canonical({
            'performance_contract': getattr(self, 'performance_contract', None),
            'authority_sha256': getattr(self, '_performance_authority_sha', None)})).hexdigest()

    def export(self, directory):
        self._check_authority()
        super().export(directory)
        path = Path(directory) / 'backend.json'
        data = json.loads(path.read_bytes())
        data.update(schema='pulsemorph.performance_backend.v1',
                    performance_contract=self.performance_contract,
                    performance_authority=self.authority,
                    performance_authority_sha256=self._performance_authority_sha)
        path.write_bytes(canonical(data))
