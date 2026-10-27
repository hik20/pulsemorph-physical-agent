"""Conditional local clock reference tool; no physical acceptance claim."""
from __future__ import annotations

import numpy as np

from .multifeature_reference import MultiFeatureReferencePlayer


CONTROL_TOOL = 'local_clock_1p4_2p8_v1'


class LocalMultiFeatureReferencePlayer(MultiFeatureReferencePlayer):
    """Unchanged section gains with a fixed C1 time map on [1.4, 2.8].

    C1 describes source time only; neither C2 nor PD smoothness is asserted.
    Positive shift advances source events within the fixed support.
    """

    def __init__(self, raw, *, clock_window_s=(1.4, 2.8), **kwargs):
        window = np.asarray(clock_window_s, dtype=float)
        if window.shape != (2,) or not np.array_equal(window, [1.4, 2.8]):
            raise ValueError('fixed clock window [1.4, 2.8] required')
        if kwargs.get('control_steps') != 200 or isinstance(kwargs.get('control_steps'), bool):
            raise ValueError('local clock requires exactly 200 controls (4 seconds)')
        for name in ('left_gain_a', 'right_gain_a', 'left_gain_b', 'right_gain_b'):
            gain = kwargs.get(name, 1.)
            if isinstance(gain, (bool, np.bool_)) or not np.isfinite(gain) or not .35 <= gain <= 1.6:
                raise ValueError('local clock gains must be finite in .35..1.6')
        shift = kwargs.get('timing_shift_s', 0.)
        if isinstance(shift, (bool, np.bool_)) or not np.isfinite(shift) or abs(shift) > .3:
            raise ValueError('local clock shift must be finite in -.3...3 seconds')
        self.clock_window_s = (1.4, 2.8)
        self.minimum_source_rate = 1-abs(float(shift))*np.pi/(2.8-1.4)
        if self.minimum_source_rate <= .25:
            raise ValueError('local clock minimum source rate must exceed .25')
        super().__init__(raw, **kwargs)

    def source_clock(self, execution_time_s):
        t = float(execution_time_s)
        if not np.isfinite(t) or not 0 <= t <= self._end_s:
            raise ValueError('execution time outside finite reference')
        a, b = self.clock_window_s
        if self.timing_shift_s == 0 or not a < t < b:
            return self.source_start_s+t, 1.
        u = (t-a)/(b-a)
        return (self.source_start_s+t+self.timing_shift_s*np.sin(np.pi*u)**2,
                1+self.timing_shift_s*np.pi/(b-a)*np.sin(2*np.pi*u))

    def draft_motion_reference(self, index):
        arrays, metadata = super().draft_motion_reference(index)
        metadata.update(
            control_tool=CONTROL_TOOL,
            edit_kind='two_section_independent_arm_gains_and_fixed_local_clock',
            clock_window_s=list(self.clock_window_s),
            source_clock='start+t+shift*sin(pi*(t-a)/(b-a))^2 for 1.4<t<2.8; start+t outside',
            minimum_source_rate=self.minimum_source_rate,
            clock_regularity='C1 source time only; no C2 or PD smoothness claim',
            physical_acceptance='not_evaluated')
        return arrays, metadata
