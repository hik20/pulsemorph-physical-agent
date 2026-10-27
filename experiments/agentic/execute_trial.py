"""Immutable local trials: edited reference -> pretrained ONNX -> PD -> MuJoCo.

Only this wrapper is instrumented; the pinned upstream controller is unmodified.
The exported reference is an agentic draft, not a validated common contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
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
    manifest = {'schema': 'pulsemorph.agentic_trial.v1', 'status': 'failed',
                'error': None, 'output': str(output), 'common_contract_validated': False,
                'video_requested': bool(video), 'video_created': False}
    states, velocities, state_times, targets, subrows, rows, calls = [], [], [], [], [], [], []
    input_calls, output_calls, physics_ms = [], [], []
    hashes, restorations = {}, []
    counts = {'initialization': 0, 'substeps': 0}
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
        manifest['candidate_id'] = proposal['candidate_id']
        manifest['proposal_sha256'] = sha256(output / 'proposal.json')
        sys.path.insert(0, str(ROOT / 'src'))
        sys.path.insert(0, str(UPSTREAM))
        import mujoco
        import onnxruntime as ort
        import torch
        import yaml
        from deployment import test_tracker_mujoco as upstream
        from pulsemorph.agentic.reference import EditedReferencePlayer

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
        source = UPSTREAM / 'data/g1-kimodo-generated/proto/output_dance.motion'
        metadata = yaml.safe_load(model_path.with_suffix('.yaml').read_text())
        control_dt = float(metadata['timing']['control_dt'])
        physics_dt = float(metadata['timing']['physics_dt'])
        decimation = int(metadata['timing']['decimation'])
        duration = float(proposal.get('duration_s', 4))
        if not np.isfinite(duration) or duration <= 0 or duration > 30:
            raise ValueError('duration_s must be finite in (0,30]')
        expected_controls = round(duration / control_dt)
        if not np.isclose(expected_controls * control_dt, duration, rtol=0, atol=1e-9):
            raise ValueError('duration must be an integral number of policy steps')
        if not np.isclose(physics_dt * decimation, control_dt):
            raise ValueError('inconsistent checkpoint timing')
        shutil.copyfile(source, output / 'source.motion')
        # torch.load is restricted to this fixed, locally pinned trusted source.
        raw = torch.load(output / 'source.motion', map_location='cpu', weights_only=False)
        player = EditedReferencePlayer(raw, joint_names=metadata['robot']['joint_names'],
            body_names=metadata['robot']['body_names'], source_start_s=float(proposal.get('source_start_s', 0)),
            control_steps=expected_controls, source_rate=float(proposal.get('source_rate', 1)),
            arm_amplitude=float(proposal.get('arm_amplitude', 1)), edit_window_s=proposal.get('edit_window_s', [1, 3]),
            source_provenance={'path': str(source.relative_to(ROOT)), 'sha256': sha256(source),
                'kind': 'upstream generated motion; not new agentic learning'})
        if not np.isclose(player.control_dt, control_dt) or player.total_frames != expected_controls:
            raise ValueError('reference timing does not match checkpoint')
        np.savez_compressed(output / 'reference_cache.npz', **player.archive())
        draft_arrays, draft_metadata = player.draft_motion_reference(0)
        np.savez_compressed(output / 'motion_reference_draft.npz', **draft_arrays)
        write_json(output / 'adapter_motion_reference_draft.json', draft_metadata)
        write_json(output / 'motion_reference_draft.json', {
            'schema': 'pulsemorph.agentic_motion_reference_draft.v1', 'common_contract_validated': False,
            'arrays': 'reference_cache.npz', 'joint_names': metadata['robot']['joint_names'],
            'body_names': metadata['robot']['body_names'], 'control_dt': control_dt,
            'quaternion_order': 'xyzw', 'candidate_id': proposal['candidate_id'],
            'source_motion_sha256': sha256(source), 'proposal_sha256': manifest['proposal_sha256']})
        snapshot = output / 'source_snapshot'
        snapshot.mkdir()
        sources = [Path(__file__), ROOT / 'src/pulsemorph/agentic/reference.py',
            ROOT / 'experiments/causal_music/reference.py',
            model_path.with_suffix('.yaml'), source,
            UPSTREAM / 'protomotions/data/assets/mjcf/g1_holo_compat.xml']
        sources += [UPSTREAM / 'deployment' / name for name in
                    ['test_tracker_mujoco.py', 'motion_utils.py', 'state_utils.py', 'tracker_inputs.py']]
        sources += sorted((UPSTREAM / 'protomotions/data/assets/mesh/G1').glob('*.stl'))
        checkpoint = UPSTREAM / metadata['metadata']['checkpoint']
        for path in sources + [model_path] + ([checkpoint] if checkpoint.exists() else []):
            hashes[str(path.relative_to(ROOT))] = sha256(path)
        for path in sources:
            destination = snapshot / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
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
        if counts['initialization'] != 1 or counts['substeps'] != expected_controls * decimation or len(calls) != expected_controls + 1:
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
                'Draft export has not passed the shared MotionReference validator',
                'Video uses 25 fps with exact physics timestamps recorded separately',
                'No learned weights changed; edits condition an existing pretrained tracker']})
        # JSON summary remains valid even when failed numerical arrays contain NaN/Inf.
        for key in ['all_substeps_min_root_height_m']:
            if manifest[key] is not None and not np.isfinite(manifest[key]):
                manifest[key] = None
        manifest['artifact_hashes'] = {str(path.relative_to(output)): sha256(path)
            for path in output.rglob('*') if path.is_file() and path.name != 'manifest.json'}
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
