"""Explicit body profile boundary; shared numerical assembly is retained.

Does not mutate the shared motion module or read its default vendor path.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pulsemorph.motion.g1 import FUTURE_OFFSETS_S,sample_reference,_heading_product_float32

def assemble_reference_inputs(ref, decision_offset_s, *, profile, heading_offset_xyzw=None):
    p = profile
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
