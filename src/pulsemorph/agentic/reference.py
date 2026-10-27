"""Draft G1 reference edits; this is not the shared MotionReference API.

The source-world carrier is left unaligned. The existing tracker performs its
single heading alignment. Edited limb body arrays remain source measurements,
not FK of edited joints, and must not be used as edited limb tracking targets.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np

_SOURCE_FILE = Path(__file__).resolve().parents[3] / 'experiments/causal_music/reference.py'
_spec = importlib.util.spec_from_file_location('_agentic_causal_reference', _SOURCE_FILE)
_causal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_causal)


class EditedReferencePlayer:
    """Finite constant retiming plus an optional smooth arm displacement edit.

    ``arm_amplitude`` scales displacement from the initial source arm posture,
    with a sin² envelope in execution time. Bounds are search bounds only, not
    physical acceptance criteria. Source provenance is copied into every export.
    """
    control_dt = .02

    def __init__(self, raw, *, joint_names, body_names, source_start_s,
                 control_steps, source_rate=1., arm_amplitude=1.,
                 edit_window_s=None, source_provenance):
        self.joint_names, self.body_names = tuple(joint_names), tuple(body_names)
        if (len(self.joint_names) != 29 or len(set(self.joint_names)) != 29
                or len(set(self.body_names)) != len(self.body_names)
                or self.body_names.count('torso_link') != 1):
            raise ValueError('explicit unique G1 joint/body names and torso_link required')
        if not isinstance(control_steps, int) or isinstance(control_steps, bool) or control_steps <= 0:
            raise ValueError('positive integer control_steps required')
        values = [source_start_s, source_rate, arm_amplitude]
        if not np.isfinite(values).all() or not .5 <= source_rate <= 1.5 or not 0 <= arm_amplitude <= 2:
            raise ValueError('finite bounded rate/amplitude required')
        if not isinstance(source_provenance, dict) or not source_provenance:
            raise ValueError('source provenance required')
        self.source_provenance = copy.deepcopy(source_provenance)
        self.total_frames = control_steps
        self.num_dofs, self.num_bodies = 29, len(self.body_names)
        self.anchor_body_index = self.body_names.index('torso_link')
        self.source_start_s, self.source_rate = float(source_start_s), float(source_rate)
        self.arm_amplitude = float(arm_amplitude)
        for key, shape in [('dof_pos', (29,)), ('rigid_body_pos', (self.num_bodies, 3)),
                           ('rigid_body_rot', (self.num_bodies, 4))]:
            array = np.asarray(raw[key])
            if array.ndim != len(shape)+1 or array.shape[1:] != shape or len(array) < 3 or not np.isfinite(array).all():
                raise ValueError('invalid source array: ' + key)
        if not np.isfinite(raw['fps']) or raw['fps'] <= 0:
            raise ValueError('positive source fps required')
        lengths = {len(raw[key]) for key in ('dof_pos', 'rigid_body_pos', 'rigid_body_rot')}
        if len(lengths) != 1 or not np.allclose(np.linalg.norm(raw['rigid_body_rot'], axis=-1), 1, atol=1e-4):
            raise ValueError('source length/quaternion mismatch')
        self.source = _causal.SourceInterpolator(raw)
        self.source.state(self.source_start_s, self.source_rate)
        # Last decision still needs +160ms; never clamp or repeat the ending.
        self._end_s = (control_steps - 1 + 8) * self.control_dt
        self.source.state(self.source_start_s + self._end_s*self.source_rate, self.source_rate)
        self.arm_indices = np.array([i for i, name in enumerate(self.joint_names)
                                     if any(x in name for x in ('shoulder_', 'elbow_', 'wrist_'))], dtype=int)
        if arm_amplitude != 1 and len(self.arm_indices) != 14:
            raise ValueError('arm edit requires fourteen named G1 arm joints')
        self.window = tuple(edit_window_s or (0., control_steps*self.control_dt))
        if (len(self.window) != 2 or not np.isfinite(self.window).all()
                or not 0 <= self.window[0] < self.window[1] <= control_steps*self.control_dt):
            raise ValueError('edit window outside execution interval')
        self.center = self.source.state(self.source_start_s, self.source_rate)['dof_pos'].astype(float)

    def state_at_time(self, execution_time_s):
        t = float(execution_time_s)
        if not np.isfinite(t) or not 0 <= t <= self._end_s:
            raise ValueError('execution time outside finite reference')
        state = self.source.state(self.source_start_s + t*self.source_rate, self.source_rate)
        a, b = self.window
        if self.arm_amplitude != 1 and a < t < b:
            u = (t-a)/(b-a)
            scale = 1 + (self.arm_amplitude-1)*np.sin(np.pi*u)**2
            derivative = (self.arm_amplitude-1)*np.pi*np.sin(2*np.pi*u)/(b-a)
            idx = self.arm_indices
            displacement = state['dof_pos'][idx].astype(float)-self.center[idx]
            state['dof_vel'][idx] = scale*state['dof_vel'][idx] + derivative*displacement
            state['dof_pos'][idx] = self.center[idx] + scale*displacement
        return state

    def get_state_at_frame(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index <= self.total_frames:
            raise IndexError('committed reference exhausted')
        return self.state_at_time(index*self.control_dt)

    def get_future_references(self, index, step_indices):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.total_frames:
            raise IndexError('decision exhausted')
        steps = list(step_indices)
        if not steps or any(not isinstance(s, (int, np.integer)) or not 0 <= s <= 8 for s in steps):
            raise IndexError('only finite integer forecast offsets 0..8 supported')
        states = [self.state_at_time((index+s)*self.control_dt) for s in steps]
        return {key: np.stack([state[key] for state in states]) for key in _causal.KEYS}

    def draft_motion_reference(self, index):
        refs = self.get_future_references(index, [1, 2, 4, 8])
        arrays = dict(time_offsets_s=np.array([.02, .04, .08, .16]),
                      joint_position_rad=refs['dof_pos'], joint_velocity_rad_s=refs['dof_vel'],
                      anchor_orientation_xyzw=refs['body_rot'][:, self.anchor_body_index])
        metadata = dict(schema_version='pulsemorph.agentic.motion_reference_draft.v1',
                        shared_contract_implemented=False, producer_kind='agentic',
                        joint_names=list(self.joint_names), anchor_body_name='torso_link',
                        decision_index=int(index), decision_time_s=index*self.control_dt,
                        frame_id='recorded_reference_world', axes='right_handed_z_up',
                        quaternion_order='xyzw', heading_alignment='deferred_to_tracker_once',
                        source_start_s=self.source_start_s, source_rate=self.source_rate,
                        arm_amplitude=self.arm_amplitude, edit_window_s=list(self.window),
                        source_provenance=copy.deepcopy(self.source_provenance),
                        carrier_body_arrays='source interpolation; not edited limb FK',
                        physical_acceptance='not_evaluated')
        return arrays, metadata

    def archive(self):
        current = [self.get_state_at_frame(i) for i in range(self.total_frames+1)]
        future = [self.get_future_references(i, [1, 2, 4, 8]) for i in range(self.total_frames)]
        return {**{'committed__'+k: np.stack([s[k] for s in current]) for k in _causal.KEYS},
                **{'forecast__'+k: np.stack([s[k] for s in future]) for k in _causal.KEYS}}
