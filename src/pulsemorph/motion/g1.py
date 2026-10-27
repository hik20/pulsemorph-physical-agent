"""Finite G1 reference sampling and the deployed tracker's three target inputs."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import numpy as np
import yaml
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp
from .contracts import MotionReference, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = ROOT / 'research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml'
FUTURE_OFFSETS_S = np.array([.02, .04, .08, .16])


def profile_from_yaml(path=DEFAULT_PROFILE):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding='utf8'))
    names = tuple(data['joint_names'])
    bodies = tuple(data['body_names'])
    anchor = data['robot']['anchor_body_name']
    offsets = np.asarray(data['motion']['future_dt_seconds'], dtype=float)
    if (len(names) != 29 or len(set(names)) != 29 or len(set(bodies)) != len(bodies)
            or anchor != 'torso_link' or anchor not in bodies
            or tuple(data['robot']['joint_names']) != names
            or tuple(data['robot']['body_names']) != bodies
            or data['robot']['anchor_body_index'] != bodies.index(anchor)
            or float(data['timing']['control_dt']) != .02
            or not np.array_equal(offsets, FUTURE_OFFSETS_S)
            or data['motion']['future_step_indices'] != [1, 2, 4, 8]):
        raise ValueError('unsupported G1 controller profile')
    expected = {'mimic_future_dof_pos': [1,4,29], 'mimic_future_dof_vel': [1,4,29], 'mimic_future_anchor_rot': [1,4,4]}
    refs = {r['name']: r for r in data['policy_inputs'] if r['name'].startswith('mimic_')}
    if set(refs) != set(expected) or any(refs[k]['shape'] != v for k,v in expected.items()):
        raise ValueError('unsupported reference inputs (including root targets)')
    if any(tuple(refs[k]['element_names'][0]) != names for k in ('mimic_future_dof_pos','mimic_future_dof_vel')) or refs['mimic_future_anchor_rot']['element_names'] != [[anchor], ['x','y','z','w']]:
        raise ValueError('reference input ordering mismatch')
    mjcf = Path(data['robot']['mjcf_path'])
    candidates = [mjcf, path.parent / mjcf] + [p / 'protomotions/data/assets' / mjcf for p in path.parents]
    body_path = next((p.resolve() for p in candidates if p.is_file()), None)
    if body_path is None:
        raise ValueError('profile MJCF unavailable; cannot bind body identity')
    return {'yaml_path': str(path), 'yaml_sha256': sha256(path.read_bytes()).hexdigest(),
            'body_path': str(body_path), 'body_sha256': sha256(body_path.read_bytes()).hexdigest(),
            'joint_names': names, 'body_names': bodies, 'anchor_body_name': anchor,
            'anchor_body_index': bodies.index(anchor), 'control_dt_s': .02,
            'future_offsets_s': tuple(offsets)}


def from_source_motion(raw, metadata_yaml_path, *, reference_id, source_path, duration_s=None, sample_dt_s=.02):
    """Export an offline baseline; caller owns trusted source deserialization.

    Source knots plus the 20ms grid preserve the source interpolation curve.
    Export velocities are analytic derivatives of that same curve.
    """
    p = profile_from_yaml(metadata_yaml_path)
    q = np.asarray(raw['dof_pos'], dtype=float)
    rot = np.asarray(raw['rigid_body_rot'], dtype=float)
    fps = float(raw['fps'])
    if (q.ndim != 2 or q.shape[1] != 29 or len(q) < 3 or not np.isfinite(q).all()
            or rot.shape != (len(q), len(p['body_names']), 4) or not np.isfinite(rot).all()
            or not np.allclose(np.linalg.norm(rot, axis=-1), 1, atol=1e-5, rtol=0)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError('invalid named-profile source arrays/fps')
    for key, expected in [('joint_names',p['joint_names']), ('body_names',p['body_names'])]:
        if key in raw and tuple(raw[key]) != expected:
            raise ValueError('source ' + key + ' ordering mismatch')
    t = np.arange(len(q))/fps
    end = t[-1] if duration_s is None else float(duration_s)
    if not np.isfinite([end,sample_dt_s]).all() or not 0 < end <= t[-1] or sample_dt_s <= 0:
        raise ValueError('invalid export duration/sample interval')
    grid = np.arange(int(np.floor(end/sample_dt_s))+1)*sample_dt_s
    if end-grid[-1] > 1e-10:
        grid = np.append(grid,end)
    else:
        grid[-1] = end
    # Retain source breakpoints as well as the policy grid: subdividing a
    # Hermite segment preserves its curve, including non-grid tail queries.
    grid = np.unique(np.r_[grid, t[t <= end]])
    spline = CubicHermiteSpline(t,q,np.gradient(q,t,axis=0,edge_order=2),extrapolate=False)
    orientation = Slerp(t,Rotation.from_quat(rot[:,p['anchor_body_index']]))(grid).as_quat()
    source_path = Path(source_path)
    metadata = {'schema_version':SCHEMA_VERSION, 'reference_id':reference_id,
        'producer':{'kind':'baseline','id':'g1-source-hermite-slerp','version':'1'},
        'body':{'profile_id':'g1','profile_sha256':p['body_sha256'],'joint_names':list(p['joint_names']),'anchor_body_name':p['anchor_body_name']},
        'controller':{'profile_id':'g1-bones-deploy','profile_sha256':p['yaml_sha256'],'control_dt_s':.02},
        'frame':{'id':'reference_world','up_axis':'z','handedness':'right','quaternion_order':'xyzw','alignment':'recorded_reference_world'},
        'context':{'decision_id':reference_id+':offline','observation_id':'offline-source','task_id':'baseline-export','runtime_session_id':'offline-export','clock_epoch':'source-zero','decision_time_s':0.},
        'validity':{'start_state_id':'source-frame-zero','start_time_s':0.,'end_time_s':end,'start_mode':'reset_from_source','end_behavior':'offline_trial_end'},
        'observation':{'mode':'offline_file','observed_through_time_s':end,'audio_observed_through_sample':None,'video_observed_through_time_s':None},
        'provenance':{'sources':[{'path':str(source_path),'sha256':sha256(source_path.read_bytes()).hexdigest()}], 'notes':['Source body arrays initialize simulation separately; only torso orientation is exported.','Source array order is bound to the supplied pinned YAML; absent source names are not independently verified.','Hermite analytic derivatives; SLERP xyzw; no heading alignment applied.']}}
    return MotionReference(metadata,grid,spline(grid),spline(grid,1),orientation)


def sample_reference(ref, times_s):
    ref.validate()
    t = np.asarray(times_s,dtype=float)
    if not np.isfinite(t).all() or np.any(t < 0) or np.any(t > ref.time_offsets_s[-1]+1e-12):
        raise ValueError('reference exhausted; extrapolation is prohibited')
    t = np.minimum(t,ref.time_offsets_s[-1])
    spline = CubicHermiteSpline(ref.time_offsets_s,ref.joint_position_rad,ref.joint_velocity_rad_s,extrapolate=False)
    flat = t.reshape(-1)
    r = Slerp(ref.time_offsets_s,Rotation.from_quat(ref.anchor_orientation_xyzw))(flat).as_quat().reshape(t.shape+(4,))
    return {'dof_pos':spline(t),'dof_vel':spline(t,1),'anchor_rot':r}


def assemble_reference_inputs(ref, decision_offset_s, *, heading_offset_xyzw=None):
    p = profile_from_yaml()
    m = ref.metadata
    if (tuple(m['body']['joint_names']) != p['joint_names'] or m['body']['anchor_body_name'] != p['anchor_body_name']
            or m['body']['profile_sha256'] != p['body_sha256'] or m['controller']['profile_sha256'] != p['yaml_sha256']
            or m['controller']['control_dt_s'] != p['control_dt_s']):
        raise ValueError('reference body/controller identity or joint ordering mismatch')
    allowed = {'schema_version','reference_id','producer','body','controller','frame','context','validity','observation','provenance'}
    if set(m)-allowed:
        raise ValueError('unsupported target metadata fields')
    group_fields = {
        'body': {'profile_id','profile_sha256','joint_names','anchor_body_name'},
        'controller': {'profile_id','profile_sha256','control_dt_s'},
        'frame': {'id','up_axis','handedness','quaternion_order','alignment'},
    }
    if any(set(m[group])-fields for group,fields in group_fields.items()):
        raise ValueError('unsupported body/controller/frame semantics')
    offset = float(decision_offset_s)
    if not np.isfinite(offset) or offset < 0:
        raise ValueError('invalid decision offset')
    state = sample_reference(ref,offset+FUTURE_OFFSETS_S)
    # SourceInterpolator.state quantizes BEFORE heading composition. Keep that
    # boundary and the deployed Hamilton product's float32 operation order.
    q = state['anchor_rot'].astype(np.float32)
    if heading_offset_xyzw is not None:
        heading = np.asarray(heading_offset_xyzw,dtype=np.float32)
        if heading.shape != (4,) or not np.isfinite(heading).all() or not np.isclose(np.linalg.norm(heading),1,atol=1e-5,rtol=0):
            raise ValueError('unit xyzw heading required')
        h = Rotation.from_quat(heading)
        if np.linalg.norm(h.apply([0,0,1])-[0,0,1]) > 1e-6:
            raise ValueError('heading must be a yaw rotation')
        q = _heading_product_float32(heading, q)
    return {'mimic_future_dof_pos':state['dof_pos'][None].astype('float32'),
            'mimic_future_dof_vel':state['dof_vel'][None].astype('float32'),
            'mimic_future_anchor_rot':q[None].astype('float32')}



def _heading_product_float32(heading, q):
    """Deployed Hamilton operation order; deliberately no renormalization.

    Adapted from ProtoMotions deployment/state_utils.py::_quat_mul_np,
    copyright NVIDIA CORPORATION & AFFILIATES, Apache-2.0. The pinned source
    and its license remain in research/vendor/ProtoMotions.
    """
    a = np.broadcast_to(np.asarray(heading, dtype=np.float32), q.shape)
    b = np.asarray(q, dtype=np.float32)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1).astype(np.float32)
