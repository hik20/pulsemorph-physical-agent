"""Materialize agentic targets and consume their validated shared package.

The 20 ms knots are exact generator samples. Between knots, the common
Hermite/SLERP curve is authoritative; it is not claimed identical to the
nonpolynomial generator there. Body carriers retain source interpolation.
"""
from __future__ import annotations

from hashlib import sha256
import numpy as np

from .reference import EditedReferencePlayer
from pulsemorph.motion.contracts import MotionReference, SCHEMA_VERSION
from pulsemorph.motion.g1 import sample_reference


def materialize_reference(generator, *, profile, candidate_id, observation_id,
                          task_id, runtime_session_id, source_path, source_sha256,
                          audio_observed_through_sample, audio_duration_s,
                          protocol_sha256=None, producer_kind='agentic'):
    """Build shared metadata with explicit offline audio access and source reset."""
    grid = np.arange(generator.total_frames+8)*generator.control_dt
    states = [generator.state_at_time(t) for t in grid]
    posture = states[0]['dof_pos']
    fingerprint = sha256(posture.tobytes()).hexdigest()
    metadata = dict(schema_version=SCHEMA_VERSION, reference_id=candidate_id+':common',
        producer=dict(kind=producer_kind, id='multifeature-reference-tool', version='1'),
        body=dict(profile_id='g1', profile_sha256=profile['body_sha256'],
                  joint_names=list(generator.joint_names), anchor_body_name='torso_link'),
        controller=dict(profile_id='g1-bones-deploy', profile_sha256=profile['yaml_sha256'], control_dt_s=.02),
        frame=dict(id='reference_world', up_axis='z', handedness='right', quaternion_order='xyzw', alignment='recorded_reference_world'),
        context=dict(decision_id=candidate_id+':offline', observation_id=observation_id,
                     task_id=task_id, runtime_session_id=runtime_session_id, clock_epoch='source-zero',
                     decision_time_s=0., protocol_sha256=protocol_sha256),
        validity=dict(start_state_id='reference-joint-posture:'+fingerprint, start_time_s=0.,
                      end_time_s=float(grid[-1]), start_mode='reset_from_source', end_behavior='offline_trial_end'),
        observation=dict(mode='offline_file', observed_through_time_s=float(audio_duration_s),
                         audio_observed_through_sample=audio_observed_through_sample,
                         video_observed_through_time_s=None),
        provenance=dict(sources=[dict(path=str(source_path), sha256=source_sha256)], notes=[
            'Start-state fingerprint identifies reference joint posture only; actual physics initial qpos/qvel are stored separately.',
            'Source body carriers are not edited-joint FK. Only shared q/dq and torso orientation condition the tracker.',
            'Dense 20 ms generator knots; common Hermite/SLERP defines between-knot sampling.',
            'Offline audio observed through take duration; 160 ms reference lookahead does not imply further audio access.']))
    return MotionReference(metadata, grid, np.stack([s['dof_pos'] for s in states]),
                           np.stack([s['dof_vel'] for s in states]),
                           np.stack([s['body_rot'][generator.anchor_body_index] for s in states]))


class CommonReferencePlayer(EditedReferencePlayer):
    """Use loaded common targets and explicitly retain generator body carriers."""
    def __init__(self, generator, reference):
        reference.validate()
        if (tuple(reference.metadata['body']['joint_names']) != generator.joint_names or
                reference.metadata['body']['anchor_body_name'] != 'torso_link' or
                reference.time_offsets_s[-1] != generator._end_s):
            raise ValueError('common reference/generator identity or horizon mismatch')
        self.__dict__.update(generator.__dict__)
        self.generator, self.reference = generator, reference

    def state_at_time(self, execution_time_s):
        t = float(execution_time_s)
        state = self.generator.state_at_time(t)
        targets = sample_reference(self.reference, np.array([t]))
        state['dof_pos'] = targets['dof_pos'][0].astype('float32')
        state['dof_vel'] = targets['dof_vel'][0].astype('float32')
        state['body_rot'][self.anchor_body_index] = targets['anchor_rot'][0].astype('float32')
        return state

    def draft_motion_reference(self, index):
        arrays, metadata = self.generator.draft_motion_reference(index)
        refs = self.get_future_references(index, [1, 2, 4, 8])
        arrays.update(joint_position_rad=refs['dof_pos'], joint_velocity_rad_s=refs['dof_vel'],
                      anchor_orientation_xyzw=refs['body_rot'][:, self.anchor_body_index])
        metadata.update(shared_contract_implemented=True, shared_reference_id=self.reference.metadata['reference_id'],
                        authoritative_target_curve='loaded common MotionReference Hermite/SLERP')
        return arrays, metadata
