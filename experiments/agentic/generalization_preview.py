"""Render recorded physics qpos with mj_forward; no replay or resimulation."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pulsemorph.evidence import sha256, write_new_json

METHODS = ['surrogate_search', 'llm_feedback', 'llm_feedback_off']
LABELS = ['UNCHANGED BASELINE', 'SURROGATE SEARCH', 'LLM + PHYSICS FEEDBACK', 'LLM / FEEDBACK OFF']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_video(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_streams', '-show_frames',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    timestamps = [Fraction(f['pts']) * Fraction(stream['time_base'])
                  for f in info['frames'] if f['media_type'] == 'video']
    if timestamps != [Fraction(i, 25) for i in range(100)]:
        raise ValueError('Preview must contain exactly 100 frames at 25 Hz, 0..3.96s')
    if (stream['width'], stream['height']) != (960, 720):
        raise ValueError('Unexpected montage size')
    audio = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if len(audio) != 1 or abs(float(audio[0]['duration']) - 4.) > .04:
        raise ValueError('Expected one four-second audio stream')
    return dict(verified=True, width=960, height=720, frames=len(timestamps),
                fps=25, duration_s=4, audio_duration_s=float(audio[0]['duration']),
                video_codec=stream['codec_name'], audio_codec=audio[0]['codec_name'])


def render_case(pilot, output, task):
    # Refuse to render partially completed experiments; nothing is created first.
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    if (ROOT/'artifacts/media/agentic').resolve() not in output.parents:
        raise ValueError('preview output must be under artifacts/media/agentic')
    comparison_path = pilot / 'comparison.json'
    if not comparison_path.is_file():
        raise ValueError('comparison.json must exist before rendering')
    comparison = read(comparison_path)
    if not comparison.get('comparison_valid'):
        raise ValueError('Refusing preview of an invalid comparison')
    protocol_path = pilot / 'protocol.json'
    protocol = read(protocol_path)
    if sha256(protocol_path) != comparison['protocol_sha256']:
        raise ValueError('Protocol changed after comparison')
    selected = {row['method']: row['selected'] for row in task['methods']}
    if any(not selected.get(method) for method in METHODS):
        missing = [method for method in METHODS if not selected.get(method)]
        raise ValueError('No physically eligible selected trial for task ' + task['task_id'] + ': ' + ', '.join(missing))
    calibration_id = 'calibration_' + task['source_id']
    calibrations = [row for row in comparison['calibrations'] if row['candidate_id'] == calibration_id]
    if len(calibrations) != 1:
        raise ValueError('Expected one source-matched calibration: ' + calibration_id)
    rows = calibrations + [selected[method] for method in METHODS]
    candidates = [calibration_id] + [selected[method]['candidate_id'] for method in METHODS]
    inputs = {str(p): sha256(p) for p in [comparison_path, protocol_path, Path(__file__).resolve()]}
    traces, panel_info = [], []
    expected_times = np.arange(201) * .02
    for candidate, row in zip(candidates, rows):
        if not isinstance(candidate, str) or Path(candidate).name != candidate:
            raise ValueError('Invalid candidate identity')
        trial = pilot / 'trials' / candidate
        manifest_path = trial / 'manifest.json'
        manifest = read(manifest_path)
        if sha256(manifest_path)!=row['execution_manifest_sha256']:
            raise ValueError('Comparison execution manifest binding changed')
        evaluation_path=pilot/'evaluations'/f'{candidate}.json'
        evaluation=read(evaluation_path)
        if evaluation!=row:raise ValueError('Comparison row differs from saved evaluation')
        inputs[str(evaluation_path)]=sha256(evaluation_path)
        actual = {str(p.relative_to(trial)) for p in trial.rglob('*') if p.is_file() and p != manifest_path}
        if actual != set(manifest['artifact_hashes']):
            raise ValueError('Trial artifact inventory mismatch: ' + candidate)
        for name,digest in manifest['artifact_hashes'].items():
            path=(trial/name).resolve()
            if trial.resolve() not in path.parents or sha256(path)!=digest:
                raise ValueError('Trial artifact inventory changed: '+name)
            inputs[str(path)]=digest
        if manifest['status'] != 'success':
            raise ValueError('Cannot render a failed physical trial')
        for name in ['qpos.npy', 'state_times.npy']:
            path = trial / name
            if sha256(path) != manifest['artifact_hashes'][name]:
                raise ValueError('Recorded trace changed: ' + candidate + '/' + name)
            inputs[str(path)] = sha256(path)
        inputs[str(manifest_path)] = sha256(manifest_path)
        qpos = np.load(trial / 'qpos.npy', allow_pickle=False)
        times = np.load(trial / 'state_times.npy', allow_pickle=False)
        if qpos.shape != (201, 36) or not np.isfinite(qpos).all():
            raise ValueError('Expected 201 finite physical qpos samples')
        if times.shape != (201,) or not np.allclose(times, expected_times, rtol=0, atol=1e-10):
            raise ValueError('Recorded clock does not match 50 Hz physics states')
        traces.append(qpos)
        panel_info.append(dict(candidate_id=candidate, features=row['features'], target=task['goal']['feature_targets'],
                               per_dimension_pass=[v is not None and abs(v-g)<=tol for v,g,tol in zip(row['features'],task['goal']['feature_targets'],protocol['feature_tolerances'])],
                               source_trial=str(trial), physical_gate_passed=row['physical_gate_passed'],
                               evaluation=row))

    observation=next(t for t in protocol['tasks'] if t['task_id']==task['task_id'])['music_observation']
    audio=Path(observation['waveform_path'])
    if sha256(audio)!=observation['waveform_sha256']:
        raise ValueError('Case waveform changed')
    inputs[str(audio)]=sha256(audio)

    # Patch only a new output-local XML. No source/vendor/trial file is changed.
    xml = ROOT / 'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    tree = ET.parse(xml)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = (xml.parent / compiler.get('meshdir', '')).resolve()
    compiler.set('meshdir', str(meshdir))
    for mesh in root.findall('./asset/mesh'):
        asset = meshdir / mesh.attrib['file']
        inputs[str(asset)] = sha256(asset)
    inputs[str(xml)] = sha256(xml)
    # Require the exact XML that generated every displayed physical trial.
    xml_key = str(xml.relative_to(ROOT))
    for candidate in candidates:
        pinned = read(pilot / 'trials' / candidate / 'preexecution_inputs.json')
        if pinned[xml_key] != inputs[str(xml)]:
            raise ValueError('Render XML differs from recorded physics model')
    for sensor in root.findall('sensor'):
        root.remove(sensor)
    world = root.find('worldbody')
    if not any(g.get('type') == 'plane' for g in world.findall('geom')):
        ET.SubElement(world, 'geom', name='floor', type='plane',
                      size='0 0 0.05', rgba='.7 .7 .7 1')
    if not world.findall('light'):
        ET.SubElement(world, 'light', pos='2 0 5', dir='0 0 -1',
                      diffuse='.4 .4 .4', specular='.1 .1 .1', directional='true')

    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output/'preview_source.py')
    tree.write(output / 'render_model.xml', encoding='unicode')
    write_new_json(output / 'source_manifest.json', {
        'execution_kind': 'recorded_physics_state_render_mj_forward',
        'not_replay_or_resimulation': True, 'inputs_sha256': inputs,
        'render_model_sha256': sha256(output / 'render_model.xml')})

    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    model = mujoco.MjModel.from_xml_path(str(output / 'render_model.xml'))
    if model.nq != 36:
        raise ValueError('Unexpected render body joint layout')
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 3.6, 140, -15
    indices = list(range(0, 200, 2))
    silent = output / 'recorded_states_silent.mp4'
    renderer = mujoco.Renderer(model, height=360, width=480)
    try:
        with imageio.get_writer(str(silent), fps=25, codec='libx264', quality=8,
                                macro_block_size=1) as writer:
            for frame_index, index in enumerate(indices):
                montage = Image.new('RGB', (960, 720))
                for panel, (trace, label, row) in enumerate(zip(traces, LABELS, panel_info)):
                    data.qpos[:] = trace[index]
                    data.qvel[:] = 0
                    data.time = float(expected_times[index])
                    mujoco.mj_forward(model, data)  # Geometry only; never mj_step.
                    renderer.update_scene(data, camera=camera)
                    image = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 480, 29), fill=(18, 23, 31))
                    draw.text((12, 4), label, font=font, fill='white')
                    draw.rectangle((0, 263, 480, 360), fill=(18, 23, 31))
                    names=['L/A','R/A','L/B','R/B']
                    actual=row['features']; goals=row['target']
                    for line in range(2):
                        text='   '.join(f"{names[j]} {actual[j]:.2f}/{goals[j]:.2f} {'PASS' if row['per_dimension_pass'][j] else 'FAIL'}" for j in range(line*2,line*2+2))
                        draw.text((8,266+line*20),text,font=small,fill='white')
                    peak='missing' if actual[4] is None else f'{actual[4]:.2f}s'
                    draw.text((8,307),f"Accent {peak}/{goals[4]:.2f}s {'PASS' if row['per_dimension_pass'][4] else 'FAIL'} | Physics {'PASS' if row['physical_gate_passed'] else 'FAIL'}",font=small,fill='white')
                    draw.text((8,332),f'Actual / target | recorded qpos | t={index*.02:.2f}s',font=small,fill=(195,207,222))
                    montage.paste(image, ((panel % 2)*480, (panel // 2)*360))
                writer.append_data(np.asarray(montage))
                if frame_index == 50:
                    montage.save(output / 'middle.png')
    finally:
        renderer.close()
    target = output / 'comparison.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
               '-i', str(silent), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-t', '4',
               '-movflags', '+faststart', str(target)]
    subprocess.run(command, check=True, capture_output=True)
    write_new_json(output / 'ffprobecheck.json', verify_video(target))
    if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError('An input changed while rendering')
    write_new_json(output / 'manifest.json', {
        'schema': 'pulsemorph.agentic_recorded_state_preview.v1',
        'execution_kind': 'recorded_physics_state_render_mj_forward',
        'scope': 'Visualization of recorded realized qpos; no ONNX calls, mj_step, closed-loop replay or resimulation. No scientific or perceptual quality claim.',
        'task_id':task['task_id'], 'source_id':task['source_id'], 'goal':task['goal'], 'panels': panel_info, 'camera': dict(kind='tracking', trackbodyid=1,
            distance=3.6, azimuth=140, elevation=-15, same_configuration_all_panels=True),
        'state_indices': indices, 'state_sampling_hz': 50, 'video_fps': 25,
        'output_frames': 100, 'output_duration_s': 4,
        'last_displayed_state_s': 3.96, 'terminal_state_s': 4.,
        'audio': dict(path=str(audio), sha256=sha256(audio), mode='shared authored offline audio',
                      crop_applied=False, encoding='AAC lossy preview'),
        'inputs_manifest_sha256': sha256(output / 'source_manifest.json'),
        'outputs_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    return target


def render(pilot, output):
    pilot,output=Path(pilot).resolve(),Path(output).resolve()
    if not (pilot/'comparison.json').is_file():raise ValueError('comparison must exist before rendering')
    comparison=read(pilot/'comparison.json')
    if not comparison.get('comparison_valid'):raise ValueError('invalid comparison')
    if sha256(pilot/'protocol.json')!=comparison['protocol_sha256']:raise ValueError('protocol changed')
    if {t['task_id'] for t in comparison['tasks']} != {'wave_a', 'wave_b', 'high5_a', 'high5_b'} or len(comparison['tasks']) != 4:
        raise ValueError('Expected wave_a, wave_b, high5_a, high5_b cases')
    missing = [task['task_id'] + ':' + method for task in comparison['tasks']
               for method in METHODS if not any(row['method'] == method and row.get('selected') for row in task['methods'])]
    if missing:
        raise ValueError('Missing physically eligible selected trials: ' + ', '.join(missing))
    if (ROOT/'artifacts/media/agentic').resolve() not in output.parents:raise ValueError('output scope')
    # Reserve the immutable parent before individual case directories are created.
    output.mkdir(parents=True,exist_ok=False)
    targets=[]
    for task in comparison['tasks']:
        name=task['task_id']
        if Path(name).name!=name:raise ValueError('invalid task identity')
        targets.append(render_case(pilot,output/name,task))
    write_new_json(output/'manifest.json',dict(schema='pulsemorph.agentic_generalization_preview.v1',
        comparison_sha256=sha256(pilot/'comparison.json'),renderer_sha256=sha256(Path(__file__)),
        execution_kind='recorded_physics_state_render_mj_forward',cases=[str(p.relative_to(output)) for p in targets],
        outputs_sha256={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}))
    return targets


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(render(args.pilot,args.output))
