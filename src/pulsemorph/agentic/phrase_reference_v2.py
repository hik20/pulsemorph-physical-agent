"""Reference-only ordered-pair/heading experiment, separate from frozen v1.

No controller, root registration, FK, contact model or learning is introduced.
"""
import numpy as np
from pulsemorph.motion.contracts import MotionReference
from pulsemorph.motion.g1 import sample_reference
from .phrase_reference import compose_phrase,plain


def compose_ordered_pair(references, *, order, heading_policy, reference_id, sources):
    """Compose the same two admitted packages in either order.

    align_join_heading uses pinned deployed G1 float32 yaw extraction/Hamilton
    multiplication on the second source torso only. Local q/dq are unchanged.
    The caller pins imported deployment/state_utils.py before using this tool.
    """
    if tuple(order) not in (('wave','high5'),('high5','wave')) or set(references)!={'wave','high5'}:
        raise ValueError('only ordered permutations of admitted Wave/High5 are supported')
    if heading_policy not in ('recorded_world','align_join_heading'):
        raise ValueError('explicit supported heading policy required')
    first,second=(references[n] for n in order)
    for ref in (first,second):ref.validate()
    for group in ('body','controller','frame'):
        if plain(first.metadata[group])!=plain(second.metadata[group]):raise ValueError('source '+group+' mismatch')
    transform={'policy':heading_policy,'second_source':order[1],'anchor':'torso_link','offset_xyzw':[0.,0.,0.,1.],
        'scope':'orientation yaw only; no root position, joint, contact, fullbody or SE3 transform'}
    adjusted=second
    if heading_policy=='align_join_heading':
        from deployment.state_utils import compute_yaw_offset_np,apply_heading_offset_np
        target=sample_reference(first,np.array([3.5]))['anchor_rot'][0].astype('float32')
        entry=sample_reference(second,np.array([.5]))['anchor_rot'][0].astype('float32')
        offset=compute_yaw_offset_np(target,entry)
        rotated=apply_heading_offset_np(offset,second.anchor_orientation_xyzw.astype('float32'))
        metadata=plain(second.metadata)
        metadata['provenance']['notes'].append('Experimental second-source torso yaw offset only; local joints unchanged; no root/contact/body transform.')
        adjusted=MotionReference(metadata,second.time_offsets_s,second.joint_position_rad,second.joint_velocity_rad_s,rotated)
        transform.update(offset_xyzw=offset.astype(float).tolist(),first_join_orientation_xyzw=target.astype(float).tolist(),untransformed_second_entry_xyzw=entry.astype(float).tolist(),numerical_semantics='pinned deployment.state_utils float32 yaw extraction and Hamilton product at source knots; common SLERP normalizes and is authoritative')
    composed=compose_phrase(first,adjusted,reference_id=reference_id,sources=sources)
    metadata=plain(composed.metadata)
    metadata['producer']={'kind':'agentic','id':'ordered-pair-phrase-reference','version':'2'}
    metadata['provenance']['notes']=[
        f'Ordered pair {order[0]}→{order[1]}; heading policy {heading_policy}.',
        'First source0..3.5; endpoint-position/derivative Hermite and shortest torso SLERP3.5..4.5; second source(t-4) thereafter through8.14.',
        'Joint joins C1; torso SO3 C0 only, angular derivative not matched.',
        'Only29localq/dq and torso orientation; no reference root trajectory, fullbody/FK/contact or SE3 registration.',
        'Heading transform: '+str(transform),
        'Reference-only ordered tool composition, not general phrase execution/admission, learning, style or physical superiority.']
    return MotionReference(metadata,composed.time_offsets_s,composed.joint_position_rad,composed.joint_velocity_rad_s,composed.anchor_orientation_xyzw),transform
