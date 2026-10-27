"""Experimental root-XY outer loop, expressed only as actor torso requests.

No root force/constraint/teleport. The nominal common reference anchor stays
fixed; actual actor anchor input is an explicitly logged local stabilization
request and MUST NOT be audited as identical to that nominal carrier.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .backend_v1 import PerformanceBackend as NominalBackend, validate_future
from pulsemorph.agentic.embodied.runtime_v1 import canonical

CONFIG = {'kp_rad_per_m': .8, 'kd_rad_per_m_s': .25,
          'tilt_cap_rad': .12, 'filter_alpha': .2,
          'rate_rad_s': .2, 'dt_s': .02}


def tilt_request(position_error_xy, velocity_xy, previous, config=CONFIG):
    values = [np.asarray(x, dtype=float) for x in (position_error_xy, velocity_xy, previous)]
    if any(x.shape != (2,) or not np.isfinite(x).all() for x in values):
        raise ValueError('Finite XY error/velocity and previous roll,pitch required')
    error, velocity, previous = values
    # Values here are roll,pitch in WORLD axes; these are explicit controller
    # saturation/rate limits, not hidden changes to authored joint targets.
    xy = config['kp_rad_per_m'] * error + config['kd_rad_per_m_s'] * velocity
    raw = np.array([xy[1], -xy[0]])
    capped = np.clip(raw, -config['tilt_cap_rad'], config['tilt_cap_rad'])
    filtered = previous + config['filter_alpha'] * (capped - previous)
    maximum = config['rate_rad_s'] * config['dt_s']
    return previous + np.clip(filtered - previous, -maximum, maximum), raw


def multiply_xyzw(a, b):
    av, bv = np.asarray(a)[:3], np.asarray(b)[:3]
    aw, bw = a[3], b[3]
    return np.r_[aw*bv + bw*av + np.cross(av, bv), aw*bw - np.dot(av, bv)]


def applied_anchor(nominal_xyzw, tilt_roll_pitch):
    nominal = np.asarray(nominal_xyzw, dtype=float)
    tilt = np.asarray(tilt_roll_pitch, dtype=float)
    if nominal.shape != (4,) or tilt.shape != (2,) or not np.isfinite(nominal).all() or not np.isfinite(tilt).all() or abs(np.linalg.norm(nominal)-1) > 1e-5:
        raise ValueError('Finite unit nominal quaternion and tilt required')
    roll, pitch = tilt
    rx = np.array([np.sin(roll/2), 0., 0., np.cos(roll/2)])
    ry = np.array([0., np.sin(pitch/2), 0., np.cos(pitch/2)])
    return multiply_xyzw(multiply_xyzw(ry, rx), nominal)


class PerformanceBackend(NominalBackend):
    def __init__(self, root, *, authority):
        own = str(Path(__file__).resolve())
        if type(authority) is not dict or authority.get('source_pins', {}).get(own) != hashlib.sha256(Path(own).read_bytes()).hexdigest():
            raise ValueError('Authority must pin root-XY outer-loop implementation')
        super().__init__(root, authority=authority)
        self.root_xy_start = self.data.qpos[:2].copy()
        self.outer_tilt = np.zeros(2)
        self.outer_config = dict(CONFIG)
        self.outer_rows = []
        self.performance_contract.update(schema='pulsemorph.performance_backend.v3',
            anchor='Fixed nominal reference; actor carrier projected by logged rootXY feedback',
            outer_loop='WORLD Ry(pitch)*Rx(roll)*nominal quaternion; classical local request, not planned FK',
            qualifications=self.performance_contract['qualifications'] + [
                'RootXY stabilization efficacy and feedback sign require physical evidence.',
                'Actual actor anchor differs from nominal common-reference carrier.'])

    def advance(self, future_q, future_dq):
        self._check_authority()
        q, dq = validate_future(future_q, future_dq, self.joint_ranges)
        if self.stopped or self.control_index >= self.authority['max_control_steps']:
            raise RuntimeError('Performance episode stopped or controlbudget exhausted')
        if self.outer_config != CONFIG:
            raise RuntimeError('Outer-loop configuration changed')
        nominal = self.anchor_target.copy()
        error = self.data.qpos[:2] - self.root_xy_start
        velocity = self.data.qvel[:2].copy()
        next_tilt, raw = tilt_request(error, velocity, self.outer_tilt, self.outer_config)
        carrier = applied_anchor(nominal, next_tilt)
        self.outer_rows.append({'control_index': self.control_index,
            'sim_time_s': self.sim_time_s, 'root_error_world_xy_m': error.tolist(),
            'root_velocity_world_xy_m_s': velocity.tolist(),
            'previous_roll_pitch_rad': self.outer_tilt.tolist(),
            'raw_roll_pitch_rad': raw.tolist(), 'applied_roll_pitch_rad': next_tilt.tolist(),
            'nominal_anchor_xyzw': nominal.tolist(), 'applied_actor_anchor_xyzw': carrier.tolist()})
        self.outer_tilt = next_tilt
        self.anchor_target = carrier
        try:
            return super().advance(q, dq)
        finally:
            self.anchor_target = nominal

    def closure_sha256(self):
        parent = super().closure_sha256()
        extra = {name: (getattr(self, name).tolist() if isinstance(getattr(self, name), np.ndarray) else getattr(self, name))
                 for name in ('root_xy_start', 'outer_tilt', 'outer_config', 'outer_rows') if hasattr(self, name)}
        return hashlib.sha256(parent.encode() + canonical(extra)).hexdigest()

    def export(self, directory):
        super().export(directory)
        directory = Path(directory)
        proof = {'schema': 'pulsemorph.performance_root_xy_outer_loop.v1',
                 'config': self.outer_config, 'root_xy_start_m': self.root_xy_start.tolist(),
                 'rows': self.outer_rows, 'authored_joint_q_dq_changed': False,
                 'nominal_anchor_equals_actual_actor_carrier': False}
        path = directory / 'root_xy_outer_loop.json'
        path.write_bytes(canonical(proof))
        meta_path = directory / 'backend.json'
        meta = json.loads(meta_path.read_bytes())
        meta.update(schema='pulsemorph.performance_backend.v3',
                    root_xy_outer_loop_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        meta_path.write_bytes(canonical(meta))
