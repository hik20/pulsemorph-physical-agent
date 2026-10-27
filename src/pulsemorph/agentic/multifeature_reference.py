"""Independent two-section arm edits and a finite smooth source-clock warp.

Body carriers are warped source measurements, never FK of edited arm joints.
The parameter bounds describe the edit search space, not physical acceptance.
"""
from __future__ import annotations

import copy
import numpy as np

from .reference import EditedReferencePlayer


class MultiFeatureReferencePlayer(EditedReferencePlayer):
    """Four independent displacement gains and one temporal accent shift.

    The clock is ``start+t+shift*sin(pi*t/T)**2`` inside the take,
    and ``start+t`` outside. Positive shift advances source events (their
    execution occurs earlier); this is not an additive event-time delay.
    Position and first derivative agree at both take boundaries. Section
    gains use separate sin² envelopes and agree with the source at boundaries.
    """

    def __init__(self, raw, *, joint_names, body_names, source_start_s,
                 control_steps, source_provenance, left_gain_a=1.,
                 right_gain_a=1., left_gain_b=1., right_gain_b=1.,
                 timing_shift_s=0., section_windows_s=None):
        supplied = [left_gain_a, right_gain_a, left_gain_b, right_gain_b]
        gains = np.asarray(supplied, dtype=float)
        if (gains.shape != (4,) or any(isinstance(v, (bool, np.bool_)) for v in supplied)
                or not np.isfinite(gains).all() or np.any((gains < 0) | (gains > 2))):
            raise ValueError('four finite arm gains in 0..2 required')
        if not isinstance(control_steps, int) or isinstance(control_steps, bool) or control_steps <= 0:
            raise ValueError('positive integer control_steps required')
        self.duration_s = control_steps*self.control_dt
        shift = float(timing_shift_s)
        if isinstance(timing_shift_s, (bool, np.bool_)) or not np.isfinite(shift) or abs(shift) > min(.35, self.duration_s/(4*np.pi)):
            raise ValueError('timing shift outside monotone search bound')
        windows = np.asarray(section_windows_s if section_windows_s is not None else
                             [(.1*self.duration_s, self.duration_s/2),
                              (self.duration_s/2, .9*self.duration_s)], dtype=float)
        if (windows.shape != (2, 2) or not np.isfinite(windows).all()
                or not 0 <= windows[0, 0] < windows[0, 1] <= windows[1, 0]
                or not windows[1, 0] < windows[1, 1] <= self.duration_s):
            raise ValueError('two ordered nonoverlapping section windows required')
        self.gains, self.timing_shift_s, self.section_windows_s = gains, shift, windows
        super().__init__(raw, joint_names=joint_names, body_names=body_names,
                         source_start_s=source_start_s, control_steps=control_steps,
                         source_provenance=source_provenance)
        self.side_indices = tuple(np.array([i for i, name in enumerate(self.joint_names)
                                          if name.startswith(side+'_') and
                                          any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))], dtype=int)
                                  for side in ('left', 'right'))
        if any(len(indices) != 7 for indices in self.side_indices):
            raise ValueError('seven explicitly named arm joints per side required')
        for key, expected in [('joint_names', self.joint_names), ('body_names', self.body_names)]:
            if key in raw and tuple(raw[key]) != expected:
                raise ValueError('source '+key+' ordering mismatch')

    def source_clock(self, execution_time_s):
        t = float(execution_time_s)
        if not np.isfinite(t) or not 0 <= t <= self._end_s:
            raise ValueError('execution time outside finite reference')
        if self.timing_shift_s == 0 or not 0 < t < self.duration_s:
            return self.source_start_s+t, 1.
        u = np.pi*t/self.duration_s
        return (self.source_start_s+t+self.timing_shift_s*np.sin(u)**2,
                1+self.timing_shift_s*np.pi/self.duration_s*np.sin(2*u))

    def state_at_time(self, execution_time_s):
        t = float(execution_time_s)
        tau, rate = self.source_clock(t)
        state = self.source.state(tau, rate)
        for section, (a, b) in enumerate(self.section_windows_s):
            if not a < t < b:
                continue
            u = (t-a)/(b-a)
            envelope, envelope_dot = np.sin(np.pi*u)**2, np.pi*np.sin(2*np.pi*u)/(b-a)
            for side, indices in enumerate(self.side_indices):
                gain = self.gains[2*section+side]
                if gain == 1:
                    continue
                displacement = state['dof_pos'][indices].astype(float)-self.center[indices]
                scale, scale_dot = 1+(gain-1)*envelope, (gain-1)*envelope_dot
                state['dof_vel'][indices] = scale*state['dof_vel'][indices]+scale_dot*displacement
                state['dof_pos'][indices] = self.center[indices]+scale*displacement
        return state

    def get_state_at_frame(self, index):
        if isinstance(index, (bool, np.bool_)):
            raise IndexError('integer frame index required')
        return super().get_state_at_frame(index)

    def get_future_references(self, index, step_indices):
        steps = list(step_indices)
        if isinstance(index, (bool, np.bool_)) or any(isinstance(s, (bool, np.bool_)) for s in steps):
            raise IndexError('integer decision and forecast offsets required')
        return super().get_future_references(index, steps)

    def draft_motion_reference(self, index):
        arrays, metadata = super().draft_motion_reference(index)
        for key in ('arm_amplitude', 'edit_window_s', 'source_rate'):
            metadata.pop(key)
        metadata.update(edit_kind='two_section_independent_arm_gains_and_smooth_clock',
                        identity_class=('exact_source_identity' if self.timing_shift_s == 0 and
                                        np.all(self.gains == 1) else 'edited_source_reference'),
                        left_gain_a=float(self.gains[0]), right_gain_a=float(self.gains[1]),
                        left_gain_b=float(self.gains[2]), right_gain_b=float(self.gains[3]),
                        timing_shift_s=self.timing_shift_s,
                        base_source_rate=1.,
                        source_rate_semantics='analytic derivative of source_clock; time varying when shift is nonzero',
                        section_windows_s=self.section_windows_s.tolist(),
                        source_clock='start+t+shift*sin(pi*t/duration)^2 inside take; start+t outside',
                        source_provenance=copy.deepcopy(self.source_provenance))
        return arrays, metadata
