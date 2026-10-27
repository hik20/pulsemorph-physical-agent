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

METHODS = ['simple_search', 'llm_feedback', 'llm_feedback_off']
LABELS = ['UNCHANGED BASELINE', 'SIMPLE SECANT SEARCH', 'LLM + PHYSICS FEEDBACK', 'LLM / FEEDBACK OFF']


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


def render(pilot, output):
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
    selected = {row['method']: row['selected'] for row in comparison['methods']}
    if any(not selected.get(method) for method in METHODS):
        raise ValueError('Each method needs a physically eligible selected trial')
    rows = [comparison['calibration']] + [selected[method] for method in METHODS]
    candidates = ['calibration_0'] + [selected[method]['candidate_id'] for method in METHODS]
    inputs = {str(p): sha256(p) for p in [comparison_path, protocol_path, Path(__file__).resolve()]}
    traces, panel_info = [], []
    expected_times = np.arange(201) * .02
    for candidate, row in zip(candidates, rows):
        if not isinstance(candidate, str) or Path(candidate).name != candidate:
            raise ValueError('Invalid candidate identity')
        trial = pilot / 'trials' / candidate
        manifest_path = trial / 'manifest.json'
        manifest = read(manifest_path)
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
        panel_info.append(dict(candidate_id=candidate, intent_ratio=row['intent_ratio'],
                               source_trial=str(trial), physical_gate_passed=row['physical_gate_passed']))

    audio = Path(protocol['audio']['audio_path'])
    if not audio.is_absolute():
        audio = ROOT / audio
    if sha256(audio) != protocol['audio']['audio_sha256']:
        raise ValueError('Authored preview audio changed')
    inputs[str(audio)] = sha256(audio)

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
    camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 3, 140, -15
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
                    draw.rectangle((0, 299, 480, 360), fill=(18, 23, 31))
                    draw.text((12, 303), f"Actual arm RMS ratio: {row['intent_ratio']:.3f}  |  target 0.700", font=small, fill='white')
                    draw.text((12, 328), f'Recorded physics / mj_forward render | t={index*.02:.2f}s', font=small, fill=(195, 207, 222))
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
    verify_video(target)
    if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError('An input changed while rendering')
    write_new_json(output / 'manifest.json', {
        'schema': 'pulsemorph.agentic_recorded_state_preview.v1',
        'execution_kind': 'recorded_physics_state_render_mj_forward',
        'scope': 'Visualization of recorded realized qpos; no ONNX calls, mj_step, closed-loop replay or resimulation. No scientific or perceptual quality claim.',
        'panels': panel_info, 'camera': dict(kind='tracking', trackbodyid=1,
            distance=3, azimuth=140, elevation=-15, same_configuration_all_panels=True),
        'state_indices': indices, 'state_sampling_hz': 50, 'video_fps': 25,
        'output_frames': 100, 'output_duration_s': 4,
        'last_displayed_state_s': 3.96, 'terminal_state_s': 4.,
        'audio': dict(path=str(audio), sha256=sha256(audio), mode='shared authored offline audio',
                      crop_applied=False, encoding='AAC lossy preview'),
        'inputs_manifest_sha256': sha256(output / 'source_manifest.json'),
        'outputs_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(render(args.pilot, args.output))
