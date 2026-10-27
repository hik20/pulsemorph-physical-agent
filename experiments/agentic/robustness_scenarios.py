"""Predeclared finite initial-true-state panel; no simulation or random sampling."""
from __future__ import annotations

from hashlib import sha256
import math

SCHEMA = 'agentic-initial-state-panel-v1'
DEFAULT_SEED = 'pulsemorph-initial-state-v1'
LEVELS = {'joint_position_rad': .01, 'joint_velocity_rad_s': .05,
          'root_planar_velocity_m_s': .03, 'root_yaw_rad': math.pi / 180}


def build_scenarios(joint_names, seed=DEFAULT_SEED):
    """Return fresh JSON-safe dictionaries in fixed nominal, pair order.

    Names identify coordinates; reordering a profile only reorders its vectors.
    Each joint direction is a SHA256-derived sign, never a hidden PRNG draw.
    Root velocities use world x/y; yaw is a world-z left rotation of wxyz.
    """
    names = list(joint_names)
    if (not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('joint names must be nonempty unique strings')
    if not isinstance(seed, str) or not seed:
        raise ValueError('seed must be a nonempty string')

    def direction(channel, name):
        # Length-prefixed fields avoid delimiter ambiguity in arbitrary names.
        fields = (SCHEMA, seed, channel, name)
        value = ''.join(str(len(f)) + ':' + f for f in fields).encode('utf8')
        return 1 if sha256(value).digest()[0] & 1 else -1

    q = [LEVELS['joint_position_rad'] * direction('q', n) for n in names]
    dq = [LEVELS['joint_velocity_rad_s'] * direction('dq', n) for n in names]

    def scenario(kind, sign):
        combined = kind == 'combined'
        return {'schema_version': SCHEMA,
                'scenario_id': 'nominal' if sign == 0 else kind + ('_plus' if sign > 0 else '_minus'),
                'kind': kind, 'sign': sign, 'seed': seed, 'joint_names': names.copy(),
                'initial_perturbation': {
                'joint_position_offset_rad': [sign*v if combined or kind == 'joint_position' else 0. for v in q],
                'joint_velocity_offset_rad_s': [sign*v if combined or kind == 'joint_velocity' else 0. for v in dq],
                'root_linear_velocity_offset_world_m_s': [
                    sign*LEVELS['root_planar_velocity_m_s'] if combined or kind == 'root_vx' else 0.,
                    sign*LEVELS['root_planar_velocity_m_s'] if combined or kind == 'root_vy' else 0., 0.],
                'root_yaw_offset_rad': sign*LEVELS['root_yaw_rad'] if combined or kind == 'root_yaw' else 0.}}
    result = [scenario('nominal', 0)]
    for kind in ('joint_position', 'joint_velocity', 'root_vx', 'root_vy', 'root_yaw', 'combined'):
        result.extend(scenario(kind, sign) for sign in (1, -1))
    return result


def yaw_left_multiply_wxyz(quaternion, yaw_rad):
    """Unit world-z quaternion * unit MuJoCo wxyz quaternion; no clipping."""
    w, x, y, z = map(float, quaternion)
    yaw_rad = float(yaw_rad)
    if not all(math.isfinite(v) for v in (w, x, y, z, yaw_rad)):
        raise ValueError('nonfinite rotation')
    if abs(w*w+x*x+y*y+z*z-1) > 1e-5:
        raise ValueError('initial quaternion is not unit')
    if yaw_rad == 0:
        return [w, x, y, z]
    c, s = math.cos(yaw_rad/2), math.sin(yaw_rad/2)
    return [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
