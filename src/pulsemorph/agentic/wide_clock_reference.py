"""Prospective four-second global clock tool; research-only, no app admission."""
from __future__ import annotations
import copy
import numpy as np

from .reference import EditedReferencePlayer
from .multifeature_reference import MultiFeatureReferencePlayer


class WideClockReferencePlayer(EditedReferencePlayer):
    """Construct a validated ±0.6s clock directly, without bypassing an old guard.

    For T=4, tau=t+s*sin(pi*t/T)^2 and tau'=1+s*pi/T*sin(2*pi*t/T).
    Thus tau' is in [0.5287611,1.4712389], tau is monotone on [0,4],
    and both tau/tau' join the identity clock at the boundaries. The finite
    future interval after4seconds remains unwarped, through4.14seconds.
    """
    tool_id = "wide_global_clock_0p6_v1"
    maximum_shift_s = .6

    def __init__(self, raw, *, joint_names, body_names, source_start_s,
                 control_steps, source_provenance, left_gain_a=1., right_gain_a=1.,
                 left_gain_b=1., right_gain_b=1., timing_shift_s=0., section_windows_s=None):
        supplied = [left_gain_a, right_gain_a, left_gain_b, right_gain_b]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in supplied):
            raise ValueError("Four numeric arm gains required")
        gains = np.asarray(supplied, dtype=float)
        if not np.isfinite(gains).all() or np.any((gains < .35) | (gains > 1.6)):
            raise ValueError("Four finite arm gains in[.35,1.6] required")
        if type(control_steps) is not int or control_steps != 200:
            raise ValueError("Wide clock v1 requires exactly200 controls/four seconds")
        if isinstance(timing_shift_s, (bool, np.bool_)) or not isinstance(timing_shift_s, (int, float, np.integer, np.floating)):
            raise ValueError("Numeric wide clock shift required")
        shift = float(timing_shift_s)
        if not np.isfinite(shift) or abs(shift) > self.maximum_shift_s:
            raise ValueError("Wide clock shift outside[-.6,.6]")
        windows = np.asarray([[.4, 2.], [2., 3.6]] if section_windows_s is None else section_windows_s, dtype=float)
        if windows.shape != (2, 2) or not np.array_equal(windows, [[.4, 2.], [2., 3.6]]):
            raise ValueError("Wide clock v1 retains fixed original arm envelopes")
        self.duration_s = 4.
        self.gains, self.timing_shift_s, self.section_windows_s = gains, shift, windows
        self.minimum_source_rate = 1 - abs(shift) * np.pi / self.duration_s
        self.maximum_source_rate = 1 + abs(shift) * np.pi / self.duration_s
        if not self.minimum_source_rate > .5 or not self.maximum_source_rate < 1.5:
            raise ValueError("Wide clock rate proof failed")
        # Call the unchanged finite source player directly. Never instantiate the
        # old narrow clock and then mutate its validated shift after construction.
        super().__init__(raw, joint_names=joint_names, body_names=body_names,
                         source_start_s=source_start_s, control_steps=control_steps,
                         source_provenance=source_provenance)
        self.side_indices = tuple(np.array([i for i, name in enumerate(self.joint_names)
            if name.startswith(side + "_") and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))], dtype=int)
            for side in ("left", "right"))
        if any(len(indices) != 7 for indices in self.side_indices): raise ValueError("Seven named arm joints per side required")
        for key, expected in [("joint_names", self.joint_names), ("body_names", self.body_names)]:
            if key in raw and tuple(raw[key]) != expected: raise ValueError("Source name ordering mismatch: " + key)

    def source_clock(self, execution_time_s):
        if isinstance(execution_time_s, (bool, np.bool_)) or not isinstance(execution_time_s, (int, float, np.integer, np.floating)):
            raise ValueError("Numeric execution time required")
        t = float(execution_time_s)
        if not np.isfinite(t) or not 0 <= t <= self._end_s: raise ValueError("Execution time outside finite reference")
        if self.timing_shift_s == 0 or not 0 < t < self.duration_s: return self.source_start_s + t, 1.
        u = np.pi * t / self.duration_s
        return (self.source_start_s + t + self.timing_shift_s * np.sin(u) ** 2,
                1 + self.timing_shift_s * np.pi / self.duration_s * np.sin(2 * u))

    # The unchanged arm envelope/derivative implementation dispatches through
    # this class's source_clock. It contains no constructor or super() bypass.
    def state_at_time(self, execution_time_s):
        if isinstance(execution_time_s, (bool, np.bool_)) or not isinstance(execution_time_s, (int, float, np.integer, np.floating)):
            raise ValueError("Numeric execution time required")
        return MultiFeatureReferencePlayer.state_at_time(self, execution_time_s)

    def get_state_at_frame(self, index):
        if isinstance(index, (bool, np.bool_)): raise IndexError("Integer frame index required")
        return super().get_state_at_frame(index)

    def get_future_references(self, index, step_indices):
        steps = list(step_indices)
        if isinstance(index, (bool, np.bool_)) or any(isinstance(v, (bool, np.bool_)) for v in steps):
            raise IndexError("Integer decision/forecast indices required")
        return super().get_future_references(index, steps)

    def draft_motion_reference(self, index):
        arrays, metadata = super().draft_motion_reference(index)
        for key in ("arm_amplitude", "edit_window_s", "source_rate"): metadata.pop(key)
        metadata.update(edit_kind="research_wide_global_clock_with_original_arm_envelopes", control_tool=self.tool_id,
                        identity_class="exact_source_identity" if self.timing_shift_s == 0 and np.all(self.gains == 1) else "edited_source_reference",
                        left_gain_a=float(self.gains[0]), right_gain_a=float(self.gains[1]), left_gain_b=float(self.gains[2]), right_gain_b=float(self.gains[3]),
                        timing_shift_s=self.timing_shift_s, maximum_shift_s=self.maximum_shift_s,
                        minimum_source_rate=self.minimum_source_rate, maximum_source_rate=self.maximum_source_rate,
                        section_windows_s=self.section_windows_s.tolist(), base_source_rate=1.,
                        source_rate_semantics="analytic derivative of monotone four-second source clock",
                        source_clock="start+t+shift*sin(pi*t/4)^2 for0<t<4; identity outside through4.14",
                        source_provenance=copy.deepcopy(self.source_provenance), public_app_admitted=False)
        return arrays, metadata
