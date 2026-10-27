"""Prepare a pinned, silent three-panel recorded-expression replay; never run policy/steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
BASE = Path.home() / '.cache/pulsemorph-agentic'
DEFAULT_INPUT = BASE / 'analysis/expression-motor-v1/prepared_v2/completed.json'
OUTPUT_ROOT = BASE / 'renders/expression-motor-v1'
LABELS = ('unit', 'pose_low', 'motion_low')
TITLES = ('Değiştirilmemiş koşul', 'High5 sol kol · ortalama eklem farkı', 'High5 sol kol · zamana bağlı eklem değişimi')
BUDGET = {'frames': 200, 'fps': 25, 'duration_s': 8, 'mj_forward': 600, 'ONNX': 0, 'mj_step': 0, 'model_calls': 0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode('utf8')


def write(path, value):
    with Path(path).open('xb') as stream: stream.write(encoded(value))


def safe_output(path):
    path = Path(path).absolute()
    if path == OUTPUT_ROOT or not path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Separate private expression render directory required')
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 1024):
            raise ValueError('Linked output forbidden')
    return path


def inspect(completed):
    # All helpers below only read/hash saved proof; no runtime constructor/session/FK.
    from pulsemorph.agentic.app.phrase.assets import PhraseAssets, PUBLIC_PYTHON
    from pulsemorph.agentic.expression.runtime_v1 import ExpressionRuntime, expected_implementation_pins
    from pulsemorph.agentic.app.phrase.renderer import _load_trace, _verify_trace, _audio_binding
    from pulsemorph.agentic.phrase_expression_v1 import GAIN_FIELDS
    data = json.loads(completed.read_bytes())
    plan_path = completed.parent / 'prepared.json'
    plan = json.loads(plan_path.read_bytes())
    if data.get('schema') != 'pulsemorph.expression_motor_completed.v1' or data.get('new_model_calls') != 0:
        raise ValueError('Completed original expression experiment required')
    if plan.get('schema') != 'pulsemorph.expression_motor_prepared.v1' or plan.get('job_id') != 'expression_motor_v1':
        raise ValueError('Original prepared experiment identity required')
    trials = data.get('trials')
    if not isinstance(trials, list) or [t.get('label') for t in trials] != list(LABELS):
        raise ValueError('Exactly unit/pose_low/motion_low in fixed order required')
    assets = PhraseAssets(ROOT)
    implementation = expected_implementation_pins(ROOT, assets, PUBLIC_PYTHON, plan['environment'])
    context = types.SimpleNamespace(root=ROOT, artifact_root=completed.parent.resolve(),
                                    _implementation=implementation, environment=plan['environment'],
                                    _music=ExpressionRuntime._music)
    context._directory = types.MethodType(ExpressionRuntime._directory, context)
    rows = []
    for trial, label, index in zip(trials, LABELS, (-1, 0, 1)):
        expected = {field: 1. for field in GAIN_FIELDS}
        if label != 'unit': expected['second_left_' + ('pose' if label == 'pose_low' else 'motion') + '_gain'] = .75
        p = trial['expression']
        if trial['index'] != index or p['order'] != ['wave', 'high5'] or p['heading_policy'] != 'align_join_heading' or any(p[k] != v for k, v in expected.items()):
            raise ValueError('Original three expression identities differ')
        directory = completed.parent / 'trials' / label
        if Path(trial['paths']['directory']) != directory.resolve(): raise ValueError('Trial directory/label differs')
        ExpressionRuntime._verify_trial_files(context, trial)
        row = _load_trace(directory)
        _verify_trace(row)
        rows.append(row)
    audio = _audio_binding(rows, plan['music_observation'])
    pins = dict(plan['source_pins'])
    for path in (completed, plan_path, Path(__file__), ROOT / 'src/pulsemorph/agentic/app/phrase/renderer.py'):
        pins[str(path)] = sha(path)
    for row in rows:
        pins.update(row['result']['source_pins'])
    for path, digest in pins.items():
        if sha(path) != digest: raise ValueError('Source pin changed')
    return rows, {'schema': 'pulsemorph.expression_recorded_render_prepared.v1', 'input': str(completed),
                  'source_pins': pins, 'inventories': [r['inventory'] for r in rows], 'audio_binding_not_muxed': audio,
                  'labels': list(LABELS), 'budget': BUDGET, 'qualification': 'Recorded physical qpos only; no new controller or physics. Reference component names do not assert hand height, style, goals or causal quality.'}


def render(rows, output):
    # Same XML, mesh checks, scene, camera and lighting as frozen phrase renderer.
    from pulsemorph.agentic.app.phrase.renderer import _verify_trace
    xml = ROOT / 'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    tree = ET.parse(xml); doc = tree.getroot(); compiler = doc.find('compiler')
    meshdir = (xml.parent / compiler.get('meshdir', '')).resolve()
    for asset in [xml, *[(meshdir / m.attrib['file']).resolve() for m in doc.findall('./asset/mesh')]]:
        if any(row['result']['source_pins'].get(str(asset)) != sha(asset) for row in rows):
            raise ValueError('Body/mesh pin differs')
    compiler.set('meshdir', str(meshdir))
    for sensor in doc.findall('sensor'): doc.remove(sensor)
    world = doc.find('worldbody')
    if not any(g.get('type') == 'plane' for g in world.findall('geom')):
        ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05', rgba='.12 .16 .22 1')
    if not world.findall('light'): ET.SubElement(world, 'light', pos='2 0 5', dir='0 0 -1', diffuse='.6 .6 .6', directional='true')
    model_file = output / 'render_model.xml'; tree.write(model_file, encoding='unicode')
    import mujoco
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    model = mujoco.MjModel.from_xml_path(str(model_file)); state = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [.5] * 3; model.vis.headlight.diffuse[:] = [.8] * 3
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 2.8, 140, -10
    calls = 0
    deadline = time.monotonic() + 180
    with imageio.get_writer(str(output / 'comparison.mp4'), fps=25, codec='libx264', quality=8, macro_block_size=1, output_params=['-movflags', '+faststart']) as writer:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for index in range(0, 400, 2):
                if time.monotonic() > deadline: raise RuntimeError('Render wall budget exhausted; partial evidence retained')
                combined = Image.new('RGB', (1920, 600))
                for column, (row, title) in enumerate(zip(rows, TITLES)):
                    state.qpos[:] = row['qpos'][index]; state.qvel[:] = row['qvel'][index]; state.time = float(row['time'][index])
                    mujoco.mj_forward(model, state); calls += 1
                    renderer.update_scene(state, camera=camera)
                    panel = Image.new('RGB', (640, 600), (13, 20, 29)); panel.paste(Image.fromarray(renderer.render()), (0, 84))
                    draw = ImageDraw.Draw(panel); draw.text((14, 10), 'KAYITLI FİZİK', font=font, fill='white')
                    draw.text((14, 39), title, font=font, fill='#b9ddc2')
                    gain_text = ('Ortalama: 1 · Değişim: 1', 'Ortalama: 0,75 · Değişim: 1', 'Ortalama: 1 · Değişim: 0,75')[column]
                    draw.text((14, 63), gain_text, font=font, fill='#b7c8d9')
                    draw.text((14, 571), f'Kaydedilmiş qpos · {index*.02:.2f} sn · yeni fizik yok', font=font, fill='#b7c8d9')
                    combined.paste(panel, (column * 640, 0))
                writer.append_data(np.asarray(combined))
    if calls != 600: raise RuntimeError('Render budget differs')
    for row in rows: _verify_trace(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--authority-sha256')
    args = parser.parse_args(); output = safe_output(args.output); completed = args.input.resolve()
    rows, plan = inspect(completed); plan['output'] = str(output)
    digest = hashlib.sha256(encoded(plan)).hexdigest()
    if not args.execute:
        output.mkdir(parents=True, exist_ok=False); write(output / 'prepared.json', plan)
        print(json.dumps({'status': 'prepared_only', 'prepared_sha256': digest, 'new_FK_NN_steps': 0})); return
    if not args.authority or sha(args.authority) != args.authority_sha256: raise ValueError('Exact separate root authority required')
    authority = json.loads(args.authority.read_bytes())
    required = {'schema': 'pulsemorph.expression_render_authority.v1', 'approved': True, 'prepared_sha256': digest, 'budget': BUDGET}
    if any(type(authority.get(k)) is not type(v) or authority.get(k) != v for k, v in required.items()): raise ValueError('Render authority/budget differs')
    if json.loads((output / 'prepared.json').read_bytes()) != plan: raise ValueError('Preparation changed')
    write(output / 'execution_attempt.json', {'authority_sha256': args.authority_sha256, 'retry_allowed': False})
    render(rows, output)
    _, after = inspect(completed); after['output'] = str(output)
    if after != plan: raise ValueError('Raw proof/source changed during rendering')
    write(output / 'manifest.json', {'schema': 'pulsemorph.expression_recorded_render.v1', 'prepared_sha256': digest,
                                    'budget': BUDGET, 'silent': True, 'video_sha256': sha(output / 'comparison.mp4')})


if __name__ == '__main__': main()
