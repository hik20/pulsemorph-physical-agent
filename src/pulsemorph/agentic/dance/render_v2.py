"""Show recorded full-body dance physics with the exact planning PCM.

Rendering evaluates saved configurations only: no tracker calls or physics steps.
The selected character is the default view; comparison is a separate artifact.
"""
from pathlib import Path
import json
import shutil
import time
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from .assets_v1 import ROOT_CACHE, BODY, ordinary, sha, write_new
from .runtime_v1 import music_observation
from ..app.renderer import _run_mux


def load_trace(trial):
    directory = ordinary(trial['paths']['directory']).resolve()
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    qpos = np.load(directory / 'qpos.npy', allow_pickle=False)
    qvel = np.load(directory / 'qvel.npy', allow_pickle=False)
    times = np.load(directory / 'state_times.npy', allow_pickle=False)
    if (manifest.get('status') != 'success' or qpos.shape != (201, 36)
            or qvel.shape != (201, 35) or times.shape != (201,)
            or not all(np.isfinite(a).all() for a in (qpos, qvel, times))
            or not np.allclose(times, np.arange(201) * .02, rtol=0, atol=1e-9)):
        raise ValueError('Expected complete finite four-second physical dance trace')
    return {'directory': directory, 'manifest': manifest,
            'qpos': qpos, 'qvel': qvel, 'times': times}


def bind_audio(rows, music):
    observed = music_observation({'available': True, 'path': music['path'],
        'sha256': music['sha256'], 'sample_rate': 48000, 'duration_s': 4,
        'observed_through_sample': 192000})
    for row in rows:
        if row['manifest'].get('audio_observation') != observed:
            raise ValueError('Saved dance and rendered music lineage differ')
    return observed


def _render_dance(root, runtime, baseline, selected, music, output, *, cancel_event=None):
    root = Path(root).resolve()
    output = ordinary(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT_CACHE.resolve()):
        raise ValueError('Exclusive dedicated dance render output required')
    directories = [ordinary(t['paths']['directory']).resolve() for t in (baseline, selected)]
    if any(output == p or output.is_relative_to(p) for p in directories):
        raise ValueError('Separate render directory required')
    if shutil.disk_usage(output.anchor).free < 256 * 1024**2:
        raise RuntimeError('Rendering requires 256 MiB free on C')
    deadline = time.monotonic() + 180
    source_sha = sha(__file__)
    def check():
        if cancel_event and cancel_event.is_set():
            raise RuntimeError('Render cancelled; partial evidence retained')
        if time.monotonic() > deadline:
            raise RuntimeError('Render wall budget exhausted')
    for trial in (baseline, selected): runtime._verify_trial(trial)
    rows = [load_trace(t) for t in (baseline, selected)]
    audio = bind_audio(rows, music)
    def verify():
        check()
        for trial in (baseline, selected): runtime._verify_trial(trial)
        if bind_audio(rows, music) != audio or sha(__file__) != source_sha:
            raise ValueError('Render input/source changed')
    verify()
    xml = root / BODY
    tree = ET.parse(xml)
    document = tree.getroot()
    compiler = document.find('compiler')
    meshdir = (xml.parent / compiler.get('meshdir', '')).resolve()
    assets = [xml, *[(meshdir / item.attrib['file']).resolve()
                    for item in document.findall('./asset/mesh')]]
    for asset in assets:
        if any(row['manifest']['implementation_pins'].get(str(asset)) != sha(asset) for row in rows):
            raise ValueError('Physical body/mesh differs from render body')
    compiler.set('meshdir', str(meshdir))
    for sensor in document.findall('sensor'): document.remove(sensor)
    world = document.find('worldbody')
    if not any(g.get('type') == 'plane' for g in world.findall('geom')):
        ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05', rgba='.12 .16 .22 1')
    if not world.findall('light'):
        ET.SubElement(world, 'light', pos='2 0 5', dir='0 0 -1', diffuse='.6 .6 .6', directional='true')
    output.mkdir(parents=True, exist_ok=False)
    model_file = output / 'render_model.xml'
    tree.write(model_file, encoding='unicode')
    attempt = {'schema': 'pulsemorph.dance_render_attempt.v2', 'status': 'started', 'phase':'render',
               'frames_completed': 0, 'new_fk': 0, 'new_ONNX': 0, 'new_steps': 0}
    def save_attempt():
        temporary = output / 'attempt.tmp'
        temporary.write_text(json.dumps(attempt), encoding='utf8')
        temporary.replace(output / 'attempt.json')
    save_attempt()
    import mujoco
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 20)
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [.5] * 3
    model.vis.headlight.diffuse[:] = [.8] * 3
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 2.3, 140, -10
    writers = {}
    renderer = None
    try:
        for kind in ('selected', 'comparison'):
            writers[kind] = imageio.get_writer(str(output / (kind + '_silent.mp4')),
                fps=25, codec='libx264', quality=8, macro_block_size=1,
                ffmpeg_params=['-threads', '1'])
        renderer = mujoco.Renderer(model, height=480, width=640)
        for index in range(0, 200, 2):
            check()
            panels = []
            labels = ['BAŞLANGIÇ DANSI', 'DÜZENLENEN DANS' if selected['id'] != baseline['id'] else 'BAŞLANGIÇ KORUNDU']
            for row, label in zip(rows, labels):
                data.qpos[:] = row['qpos'][index]
                data.qvel[:] = row['qvel'][index]
                data.time = float(row['times'][index])
                mujoco.mj_forward(model, data)
                attempt['new_fk'] += 1
                save_attempt()
                renderer.update_scene(data, camera=camera)
                panel = Image.new('RGB', (640, 560), (13, 20, 29))
                panel.paste(Image.fromarray(renderer.render()), (0, 48))
                draw = ImageDraw.Draw(panel)
                draw.text((16, 12), label, font=font, fill='white')
                draw.text((16, 532), f'Ölçülmüş simülasyon kaydı · {index * .02:.2f} sn', font=font, fill='#b7c8d9')
                panels.append(panel)
            combined = Image.new('RGB', (1280, 560))
            combined.paste(panels[0], (0, 0))
            combined.paste(panels[1], (640, 0))
            writers['selected'].append_data(np.asarray(panels[1]))
            writers['comparison'].append_data(np.asarray(combined))
            attempt['frames_completed'] += 1
            save_attempt()
    except Exception:
        attempt['status'] = 'stopped'
        save_attempt()
        _save_failure(output, 'render')
        raise
    finally:
        if renderer is not None: renderer.close()
        for writer in writers.values(): writer.close()
    verify()
    for kind in ('selected', 'comparison'):
        attempt['phase']='mux_'+kind
        save_attempt()
        _run_mux([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
            '-nostdin', '-n', '-i', str(output / (kind + '_silent.mp4')),
            '-i', audio['path'], '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
            '-c:a', 'aac', '-t', '4', '-movflags', '+faststart', str(output / (kind + '.mp4'))],
            cancel_event, timeout=max(.1, deadline - time.monotonic()))
    verify()
    if attempt['frames_completed'] != 100 or attempt['new_fk'] != 200:
        raise RuntimeError('Recorded render budget differs')
    attempt['status'] = 'completed'
    save_attempt()
    manifest = {'schema': 'pulsemorph.dance_render.v2',
        'execution': 'recorded_qpos_mj_forward_only', 'frames': 100, 'fps': 25,
        'duration_s': 4, 'new_fk': 200, 'new_ONNX': 0, 'new_steps': 0,
        'renderer_sha256': source_sha, 'audio': audio,
        'audio_scope': 'Exact observed planning music; soundtrack is not beat-alignment evidence.',
        'baseline_id': baseline['id'], 'selected_id': selected['id'],
        'baseline_parameters': baseline['parameters'], 'selected_parameters': selected['parameters'],
        'physical_manifest_sha256': [sha(p / 'manifest.json') for p in directories],
        'comparison_sha256': sha(output / 'comparison.mp4'),
        'selected_sha256': sha(output / 'selected.mp4')}
    write_new(output / 'manifest.json', manifest)
    return {'selected': str(output / 'selected.mp4'), 'comparison': str(output / 'comparison.mp4'),
            'manifest': str(output / 'manifest.json'), 'new_fk': 200}


def _save_failure(output, phase):
    """Exclusive diagnostic; retain the first exception and partial counters."""
    path=Path(output)/'failure.json'
    if path.exists():return
    error=__import__('sys').exc_info()[1]
    attempt_path=Path(output)/'attempt.json'
    counters=json.loads(attempt_path.read_bytes()) if attempt_path.is_file() else {
        'frames_completed':0,'new_fk':0,'new_ONNX':0,'new_steps':0}
    write_new(path,{'schema':'pulsemorph.dance_render_failure.v2','phase':phase,
        'error_type':type(error).__name__,'message':str(error)[:4000],
        'traceback':traceback.format_exc()[-24000:],'attempt':counters,
        'qualification':'Render/mux failure; no new tracker inference or physics steps.'})


def render_dance(root,runtime,baseline,selected,music,output,*,cancel_event=None):
    output=ordinary(output).resolve()
    if output.exists() or output==ROOT_CACHE.resolve() or not output.is_relative_to(ROOT_CACHE.resolve()):
        raise ValueError('Exclusive dedicated dance render output required')
    if any(output==ordinary(t['paths']['directory']).resolve() or output.is_relative_to(ordinary(t['paths']['directory']).resolve()) for t in (baseline,selected)):
        raise ValueError('Separate render directory required')
    try:return _render_dance(root,runtime,baseline,selected,music,output,cancel_event=cancel_event)
    except Exception:
        output.mkdir(parents=True,exist_ok=True)
        attempt=output/'attempt.json'
        phase=json.loads(attempt.read_bytes()).get('phase','setup') if attempt.is_file() else 'setup'
        _save_failure(output,phase)
        raise
