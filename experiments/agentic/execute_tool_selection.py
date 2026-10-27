"""Explicit two-tool clock-selection trials: edited reference -> pretrained ONNX -> PD -> MuJoCo.

Only this wrapper is instrumented; the pinned upstream controller is unmodified.
Targets are saved, loaded, and consumed through the validated common contract.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'research/vendor/ProtoMotions'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def sha256_array(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def snapshot_source(source, destination, object_store, expected_sha256):
    """Publish copied bytes, then link the snapshot; never link a live input."""
    source, destination, object_store = map(Path, (source, destination, object_store))
    allowed = (ROOT / 'artifacts/runs/agentic').resolve()
    for path in (destination.resolve(), object_store.resolve()):
        if allowed not in path.parents:
            raise ValueError('snapshot destination/store must be strictly under agentic runs')
    if (len(expected_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('invalid snapshot SHA256')
    object_store.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    object_path = object_store / expected_sha256
    if object_path.exists():
        if object_path.is_symlink() or sha256(object_path) != expected_sha256:
            raise ValueError('snapshot object hash collision/corruption')
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix='.snapshot-', dir=object_store)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'wb') as target, source.open('rb') as origin:
                shutil.copyfileobj(origin, target)
            if sha256(temporary) != expected_sha256:
                raise ValueError('source changed before snapshot publication')
            try:
                os.link(temporary, object_path)
            except FileExistsError:
                if object_path.is_symlink() or sha256(object_path) != expected_sha256:
                    raise ValueError('snapshot object hash collision/corruption')
            except OSError:
                # Filesystems without hardlinks retain independent copied trial bytes.
                with destination.open('xb') as target, temporary.open('rb') as origin:
                    shutil.copyfileobj(origin, target)
                return
        finally:
            temporary.unlink(missing_ok=True)
    try:
        os.link(object_path, destination)
    except FileExistsError:
        raise
    except OSError:
        with destination.open('xb') as target, object_path.open('rb') as origin:
            shutil.copyfileobj(origin, target)
    if sha256(destination) != expected_sha256:
        raise ValueError('snapshot destination hash mismatch')


def resolve_source_motion(proposal):
    """Only hash-pinned direct vendor motion files may reach trusted torch.load."""
    value = proposal.get('source_motion')
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('source_motion requires exactly path and sha256')
    if not isinstance(value['path'], str) or not isinstance(value['sha256'], str):
        raise ValueError('source_motion path and sha256 must be strings')
    digest = value['sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('source_motion sha256 must be lowercase hexadecimal')
    relative = Path(value['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('source_motion must use a repository-relative allowlisted path')
    source = (ROOT / relative).resolve()
    allowed = (UPSTREAM / 'data/g1-kimodo-generated/proto').resolve()
    if source.parent != allowed or source.suffix != '.motion' or not source.is_file():
        raise ValueError('source_motion is outside trusted vendor motion allowlist')
    if sha256(source) != digest:
        raise ValueError('source_motion SHA256 mismatch before deserialization')
    return source, digest


def proposal_producer(proposal):
    """Classify the fixed protocol methods without mislabeling LLM proposals."""
    if proposal.get('control_tool') not in ('global_clock_v1', 'local_clock_1p4_2p8_v1'):
        raise ValueError('control_tool must be global_clock_v1 or local_clock_1p4_2p8_v1')
    method = proposal.get('method')
    if method not in ('calibration', 'surrogate_search', 'llm_feedback', 'llm_feedback_off'):
        raise ValueError('unsupported generalization method')
    if method in ('calibration', 'surrogate_search'):
        return {'kind': 'baseline', 'id': 'tool-selection-' + method, 'version': '1'}
    return {'kind': 'agentic', 'id': 'tool-selection-' + method, 'version': '1',
            'model': 'gpt-6.1-sol'}



def validate_parameters(parameters):
    """One identical numeric search domain for both tools; reject JSON booleans."""
    names = {'left_gain_a', 'right_gain_a', 'left_gain_b', 'right_gain_b', 'timing_shift_s'}
    if not isinstance(parameters, dict) or set(parameters) != names:
        raise ValueError('exactly five multifeature parameters required')
    for name, value in parameters.items():
        lower, upper = (-.3, .3) if name == 'timing_shift_s' else (.35, 1.6)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not lower <= value <= upper):
            raise ValueError(name + ' outside finite numeric tool-selection domain')
    return parameters


def build_generator(proposal, raw, **kwargs):
    """Dispatch locally without replacing either reference class or old runtime."""
    from pulsemorph.agentic.multifeature_reference import MultiFeatureReferencePlayer
    from pulsemorph.agentic.local_multifeature_reference import LocalMultiFeatureReferencePlayer
    proposal_producer(proposal)
    parameters = validate_parameters(proposal.get('parameters'))
    player_class = (MultiFeatureReferencePlayer if proposal['control_tool'] == 'global_clock_v1'
                    else LocalMultiFeatureReferencePlayer)
    return player_class(raw, **kwargs, **parameters)


def execute_trial(proposal_path, output, video=False):
    """Run once. Preserve partial evidence on error and return its failed manifest.

    Run creation errors (outside root, existing directory) raise before execution.
    CLI exits nonzero for any failed manifest. Not safe for concurrent threads:
    instrumentation temporarily replaces process-global upstream functions.
    """
    output = Path(output).resolve()
    allowed = (ROOT / 'artifacts/runs/agentic').resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError('output must be a new directory strictly under artifacts/runs/agentic')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'schema': 'pulsemorph.agentic_generalization_trial.v1', 'status': 'failed',
                'error': None, 'output': str(output), 'common_contract_validated': False,
                'experiment_kind': 'tool_selection_v1',
                'video_requested': bool(video), 'video_created': False}
    states, velocities, state_times, targets, subrows, rows, calls = [], [], [], [], [], [], []
    input_calls, output_calls, physics_ms = [], [], []
    hashes, restorations = {}, []
    counts = {'initialization': 0, 'substeps': 0, 'control_boundary_forward': 0}
    warnings_initial = None
    renderer, writer, video_frames = None, None, []
    expected_controls = 0
    control_dt, physics_dt, decimation = .02, .001, 20
    try:
        proposal_bytes = Path(proposal_path).read_bytes()
        (output / 'proposal.json').write_bytes(proposal_bytes)
        proposal = json.loads(proposal_bytes)
        if not isinstance(proposal, dict) or not isinstance(proposal.get('candidate_id'), str):
            raise ValueError('proposal must be an object with a string candidate_id')
        producer = proposal_producer(proposal)
        validate_parameters(proposal.get('parameters'))
        manifest['control_tool'] = proposal['control_tool']
        source, source_digest = resolve_source_motion(proposal)
        manifest['source_motion'] = {'path': str(source.relative_to(ROOT)), 'sha256': source_digest}
        manifest['candidate_id'] = proposal['candidate_id']
        manifest['proposal_sha256'] = sha256(output / 'proposal.json')
        sys.path.insert(0, str(ROOT / 'src'))
        sys.path.insert(0, str(UPSTREAM))
        import mujoco
        import onnxruntime as ort
        import torch
        import yaml
        from deployment import test_tracker_mujoco as upstream
        from pulsemorph.agentic.common_player import CommonReferencePlayer, materialize_reference
        from pulsemorph.motion.contracts import save_reference, load_reference
        from pulsemorph.motion.g1 import profile_from_yaml

        def capture(model, data):
            nonlocal renderer, writer
            if not video:
                return
            import imageio.v2 as imageio
            if renderer is None:
                renderer = mujoco.Renderer(model, height=480, width=640)
                writer = imageio.get_writer(str(output / 'simulation.mp4'), fps=25, codec='libx264', quality=7)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 3, 140, -15
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
            video_frames.append({'frame_index': len(video_frames), 'video_pts_seconds': len(video_frames)/25,
                                 'actual_sim_time_seconds': float(data.time), 'physics_substep_count': counts['substeps']})

        model_path = UPSTREAM / 'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'
        metadata = yaml.safe_load(model_path.with_suffix('.yaml').read_text())
        control_dt = float(metadata['timing']['control_dt'])
        physics_dt = float(metadata['timing']['physics_dt'])
        decimation = int(metadata['timing']['decimation'])
        if (control_dt, physics_dt, decimation) != (.02, .001, 20):
            raise ValueError('generalization requires policy50Hz/physics1000Hz/decimation20')
        duration = float(proposal.get('duration_s', 4))
        if duration != 4. or float(proposal.get('source_start_s', 0)) != 0.:
            raise ValueError('multifeature experiment requires duration4s/source start0')
        expected_controls = round(duration / control_dt)
        if not np.isclose(expected_controls * control_dt, duration, rtol=0, atol=1e-9):
            raise ValueError('duration must be an integral number of policy steps')
        if not np.isclose(physics_dt * decimation, control_dt):
            raise ValueError('inconsistent checkpoint timing')
        shutil.copyfile(source, output / 'source.motion')
        # Recheck copied bytes before trusted deserialization (including source copy races).
        if sha256(output / 'source.motion') != source_digest:
            raise ValueError('copied source_motion SHA256 mismatch before deserialization')
        raw = torch.load(output / 'source.motion', map_location='cpu', weights_only=False)
        generator = build_generator(proposal, raw, joint_names=metadata['robot']['joint_names'],
            body_names=metadata['robot']['body_names'], source_start_s=0.,
            control_steps=expected_controls, section_windows_s=[[.4, 2.], [2., 3.6]],
            source_provenance={'path': str(source.relative_to(ROOT)), 'sha256': sha256(source),
                'kind': 'upstream generated motion; not new agentic learning'})
        profile = profile_from_yaml(model_path.with_suffix('.yaml'))
        audio = proposal['audio']
        if audio['duration_s'] != 4 or audio['observed_through_sample'] != 88200:
            raise ValueError('expected explicit four-second/88200-sample offline audio access')
        common = materialize_reference(generator, profile=profile, candidate_id=proposal['candidate_id'],
            observation_id=proposal['observation_id'], task_id=proposal['task_id'],
            runtime_session_id=proposal.get('runtime_session_id', proposal['candidate_id']+':offline-trial'),
            source_path=source.relative_to(ROOT), source_sha256=sha256(source),
            audio_observed_through_sample=audio['observed_through_sample'], audio_duration_s=audio['duration_s'],
            protocol_sha256=proposal.get('protocol_sha256'),
            producer_kind=producer['kind'])
        producer['control_tool'] = proposal['control_tool']
        manifest['control_tool'] = proposal['control_tool']
        manifest['producer'] = producer
        common = replace(common, metadata={**common.metadata, 'producer': manifest['producer'],
            'control_tool': proposal['control_tool'], 'experiment_kind': 'tool_selection_v1'})
        save_reference(common, output / 'common_reference')
        loaded = load_reference(output / 'common_reference')
        player = CommonReferencePlayer(generator, loaded)
        manifest.update(common_contract_validated=True,
                        common_reference_manifest='common_reference/manifest.json',
                        common_reference_knot_count=len(loaded.time_offsets_s),
                        reference_initial_posture_sha256=hashlib.sha256(generator.get_state_at_frame(0)['dof_pos'].tobytes()).hexdigest(),
                        reference_initial_posture_is_physics_state=False,
                        body_profile_sha256=profile['body_sha256'], controller_profile_sha256=profile['yaml_sha256'])
        if not np.isclose(player.control_dt, control_dt) or player.total_frames != expected_controls:
            raise ValueError('reference timing does not match checkpoint')
        np.savez_compressed(output / 'reference_cache.npz', **player.archive())
        np.save(output / 'unwarped_source_root_positions.npy', np.asarray([
            generator.source.state(i * control_dt, 1.)['body_pos'][0]
            for i in range(expected_controls + 1)]))
        draft_arrays, draft_metadata = player.draft_motion_reference(0)
        draft_metadata['control_tool'] = proposal['control_tool']
        np.savez_compressed(output / 'motion_reference_draft.npz', **draft_arrays)
        write_json(output / 'adapter_motion_reference_draft.json', draft_metadata)
        write_json(output / 'motion_reference_draft.json', {
            'schema': 'pulsemorph.agentic_motion_reference_draft.v1', 'common_contract_validated': True,
            'arrays': 'reference_cache.npz', 'joint_names': metadata['robot']['joint_names'],
            'body_names': metadata['robot']['body_names'], 'control_dt': control_dt,
            'quaternion_order': 'xyzw', 'candidate_id': proposal['candidate_id'],
            'source_motion_sha256': sha256(source), 'proposal_sha256': manifest['proposal_sha256'],
            'control_tool': proposal['control_tool'], 'producer': producer})
        snapshot = output / 'source_snapshot'
        snapshot.mkdir()
        sources = [Path(__file__), ROOT / 'src/pulsemorph/agentic/local_multifeature_reference.py',
            ROOT / 'experiments/agentic/execute_generalization.py',
            ROOT / 'experiments/agentic/execute_local_capability.py',
            ROOT / 'experiments/agentic/evaluate_generalization.py',
            ROOT / 'experiments/agentic/evaluate_multifeature.py', ROOT / 'experiments/agentic/evaluate_trial.py',
            ROOT / 'experiments/agentic/execute_trial.py',
            ROOT / 'src/pulsemorph/agentic/reference.py',
            ROOT / 'src/pulsemorph/agentic/multifeature_reference.py',
            ROOT / 'src/pulsemorph/agentic/common_player.py',
            ROOT / 'experiments/causal_music/reference.py',
            model_path.with_suffix('.yaml'), source,
            UPSTREAM / 'protomotions/data/assets/mjcf/g1_holo_compat.xml']
        sources += sorted((ROOT / 'src/pulsemorph/motion').glob('*.py'))
        sources += [UPSTREAM / 'deployment' / name for name in
                    ['test_tracker_mujoco.py', 'motion_utils.py', 'state_utils.py', 'tracker_inputs.py']]
        sources += sorted((UPSTREAM / 'protomotions/data/assets/mesh/G1').glob('*.stl'))
        checkpoint = UPSTREAM / metadata['metadata']['checkpoint']
        for path in sources + [model_path] + ([checkpoint] if checkpoint.exists() else []):
            hashes[str(path.relative_to(ROOT))] = sha256(path)
        for path in sources:
            destination = snapshot / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            snapshot_source(path, destination, output.parent.parent / 'source_snapshot_objects',
                            hashes[str(path.relative_to(ROOT))])
        write_json(output / 'preexecution_inputs.json', hashes)
        manifest.update({'preexecution_input_hashes': hashes, 'onnx_sha256': sha256(model_path),
            'motion_sha256': sha256(source), 'checkpoint_file_present': checkpoint.exists(),
            'checkpoint_sha256': sha256(checkpoint) if checkpoint.exists() else None,
            'reference_cache_sha256': sha256(output / 'reference_cache.npz'),
            'python': sys.version, 'platform': platform.platform(),
            'versions': {'numpy': np.__version__, 'mujoco': mujoco.__version__, 'onnxruntime': ort.__version__, 'torch': torch.__version__},
            'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()})
        real_session, real_step, real_initial = ort.InferenceSession, mujoco.mj_step, upstream.set_initial_pose

        class Session:
            def __init__(self, *args, **kwargs):
                options = ort.SessionOptions()
                options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
                kwargs['sess_options'] = options
                self.session = real_session(*args, **kwargs)

            def __getattr__(self, key):
                return getattr(self.session, key)

            def run(self, names, feed, *args, **kwargs):
                index = len(calls)
                row = {'call_index': index, 'role': 'warmup' if index == 0 else 'control',
                       'control_index': None if index == 0 else index - 1, 'output_names': list(names)}
                calls.append(row)
                input_calls.append({key: value.copy() for key, value in feed.items()})
                tick = time.perf_counter()
                try:
                    result = self.session.run(names, feed, *args, **kwargs)
                    output_calls.append(dict(zip(names, [v.copy() for v in result])))
                    row['all_finite'] = bool(all(np.isfinite(v).all() for v in list(feed.values()) + result))
                    if not row['all_finite']:
                        raise RuntimeError('nonfinite ONNX input/output')
                    return result
                except Exception as error:
                    row['error'] = repr(error)
                    raise
                finally:
                    row['elapsed_ms'] = (time.perf_counter() - tick) * 1000

        def initial(model, data, motion_player):
            nonlocal warnings_initial
            counts['initialization'] += 1
            if counts['initialization'] != 1 or counts['substeps']:
                raise RuntimeError('unexpected simulation reset')
            real_initial(model, data, motion_player)
            warnings_initial = [int(w.number) for w in data.warning]
            states.append(data.qpos.copy()); velocities.append(data.qvel.copy()); state_times.append(float(data.time))
            manifest['actual_initial_qpos_sha256'] = sha256_array(data.qpos)
            manifest['actual_initial_qvel_sha256'] = sha256_array(data.qvel)
            manifest['physics_contract'] = {'integrator': int(model.opt.integrator), 'solver': int(model.opt.solver),
                'joint_actuatorfrclimited': model.jnt_actfrclimited.tolist(),
                'joint_actuatorfrcrange': model.jnt_actfrcrange.tolist(),
                'actuator_forcelimited': model.actuator_forcelimited.tolist(),
                'actuator_forcerange': model.actuator_forcerange.tolist()}
            capture(model, data)

        def step(model, data):
            before = float(data.time)
            if counts['substeps'] % decimation == 0:
                targets.append(data.ctrl.copy())
            tick = time.perf_counter()
            real_step(model, data)
            physics_ms.append((time.perf_counter() - tick) * 1000)
            counts['substeps'] += 1
            if counts['substeps'] % decimation == 0:
                # mj_step can leave derived FK at the preceding integration
                # point. Refresh before the next policy observes torso pose.
                mujoco.mj_forward(model, data)
                counts['control_boundary_forward'] += 1
            now = float(data.time)
            finite = bool(all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc, data.ctrl, data.actuator_force)))
            warnings = [int(w.number) for w in data.warning]
            contiguous = bool(np.isclose(now - before, physics_dt, rtol=0, atol=1e-10))
            subrows.append([counts['substeps'], now, float(data.qpos[2]), int(data.ncon),
                float(np.max(np.abs(data.actuator_force))), float(np.linalg.norm(data.qvel)),
                float(np.linalg.norm(data.qacc)), float(np.max(np.abs(data.ctrl))), finite, contiguous])
            manifest['mujoco_warning_counts_after'] = warnings
            if not finite or not contiguous or warnings != warnings_initial:
                # Preserve the failed boundary too, even when it is not a control boundary.
                states.append(data.qpos.copy()); velocities.append(data.qvel.copy()); state_times.append(now)
                raise RuntimeError('nonfinite physics state, time discontinuity, or MuJoCo warning')
            if counts['substeps'] % 40 == 0 and counts['substeps'] < expected_controls * decimation:
                capture(model, data)
            if counts['substeps'] % decimation == 0:
                states.append(data.qpos.copy()); velocities.append(data.qvel.copy()); state_times.append(now)
                index = counts['substeps'] // decimation
                reference = player.get_state_at_frame(index)
                rows.append({'control_index': index - 1, 'state_time_s': now,
                    'timestamp_aligned_joint_rmse_rad': float(np.sqrt(np.mean((data.qpos[7:] - reference['dof_pos']) ** 2))) if reference is not None else None,
                    'timestamp_aligned_root_position_error_m': float(np.linalg.norm(data.qpos[:3] - reference['body_pos'][0])) if reference is not None else None})

        for owner, name, replacement in [(ort, 'InferenceSession', Session),
                (mujoco, 'mj_step', step), (upstream, 'set_initial_pose', initial),
                (upstream, 'MotionPlayer', lambda *args, **kwargs: player)]:
            restorations.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)
        upstream.run(str(model_path), str(output / 'source.motion'), realtime=False)
        if (counts['initialization'] != 1 or counts['substeps'] != expected_controls * decimation
                or counts['control_boundary_forward'] != expected_controls or len(calls) != expected_controls + 1):
            raise RuntimeError('execution counts violate checkpoint contract')
        manifest['status'] = 'success'
    except Exception as error:
        manifest['error'] = repr(error)
        write_json(output / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
    finally:
        for owner, name, original in reversed(restorations):
            setattr(owner, name, original)
        for resource in [writer, renderer]:
            if resource is not None:
                try:
                    resource.close()
                except Exception as close_error:
                    manifest['status'], manifest['error'] = 'failed', repr(close_error)
        manifest['video_created'] = bool(video_frames)
        manifest['video_frame_count'] = len(video_frames)
        write_json(output / 'video_timestamps.json', video_frames)
        np.save(output / 'qpos.npy', np.asarray(states))
        np.save(output / 'qvel.npy', np.asarray(velocities))
        np.save(output / 'state_times.npy', np.asarray(state_times))
        np.save(output / 'applied_joint_targets.npy', np.asarray(targets))
        for name, records in [('policy_inputs', input_calls), ('policy_outputs', output_calls)]:
            # Per-call keys preserve exact feeds even if a failed call is missing outputs.
            np.savez_compressed(output / (name + '.npz'), **{
                f'call_{index:04d}__{key}': value for index, record in enumerate(records) for key, value in record.items()})
        np.savez_compressed(output / 'substep_telemetry.npz', values=np.asarray(subrows), columns=np.asarray([
            'substep', 'time_s', 'root_height_m', 'contacts', 'max_abs_actuator_force', 'qvel_l2',
            'qacc_l2', 'max_abs_ctrl', 'all_finite', 'time_contiguous']))
        write_json(output / 'policy_calls.json', calls)
        write_json(output / 'telemetry.json', rows)
        manifest.update({'wall_seconds': time.perf_counter() - started,
            'control_boundary_forward_count': counts['control_boundary_forward'],
            'observation_alignment': 'mj_forward at each control boundary aligns qpos/qvel/torso FK at the same timestamp',
            'previous_agentic_observation_limitation': 'Prior preserved runner may expose torso FK stale by one physics step; no replay parity with prior physics is claimed',
            'expected_control_steps': expected_controls, 'actual_control_steps': len(rows),
            'actual_physics_steps': counts['substeps'], 'expected_physics_steps': expected_controls * decimation,
            'instrumented_set_initial_pose_call_count': counts['initialization'],
            'control_dt': control_dt, 'physics_dt': physics_dt, 'decimation': decimation,
            'initial_state_in_state_arrays': bool(states), 'mujoco_warning_counts_before': warnings_initial,
            'all_substeps_finite': all(bool(row[8]) for row in subrows) if subrows else None,
            'all_substeps_time_contiguous': all(bool(row[9]) for row in subrows) if subrows else None,
            'all_substeps_min_root_height_m': min((row[2] for row in subrows), default=None),
            'online_onnx_call_count': max(0, len(calls) - 1), 'warmup_onnx_call_count': min(1, len(calls)),
            'physics_substep_ms_p50_p95_p99': np.percentile(physics_ms, [50, 95, 99]).tolist() if physics_ms else [],
            'preexecution_hashes_unchanged': all((ROOT / path).exists() and sha256(ROOT / path) == digest for path, digest in hashes.items()),
            'limitations': ['No hardware safety or recovery claim', 'No arbitrary qpos write interception',
                'Shared target validation is structural; physical acceptance is evaluated independently',
                'Common reference has exact 20ms knots; between-knot common curve is authoritative',
                'Video uses 25 fps with exact physics timestamps recorded separately',
                'No learned weights changed; edits condition an existing pretrained tracker']})
        # JSON summary remains valid even when failed numerical arrays contain NaN/Inf.
        for key in ['all_substeps_min_root_height_m']:
            if manifest[key] is not None and not np.isfinite(manifest[key]):
                manifest[key] = None
        manifest['artifact_hashes'] = {str(path.relative_to(output)): sha256(path)
            for path in output.rglob('*') if path.is_file() and path != output / 'manifest.json'}
        write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    manifest = execute_trial(args.proposal, args.output, args.video)
    print(json.dumps({key: manifest.get(key) for key in ['status', 'candidate_id', 'output', 'error', 'actual_control_steps', 'actual_physics_steps']}))
    return 0 if manifest['status'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())

