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
LABELS = ['FRESH UNCHANGED BASELINE', 'OLD BEST: 8-TRIAL LLM PATH', 'NEW POST-COMPARISON SEARCH', 'EXACT FRESH CONFIRMATION']


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


def linked(ref, base, inputs):
    path = Path(ref['path'])
    if not path.is_absolute(): path = base / path
    path = path.resolve()
    if sha256(path) != ref['sha256']: raise ValueError('Linked artifact changed: '+str(path))
    inputs[str(path)] = ref['sha256']
    return path


def render(pilot, output, audit_path):
    pilot, output, audit_path = map(lambda p: Path(p).resolve(), (pilot, output, audit_path))
    if (ROOT/'artifacts/media/agentic').resolve() not in output.parents:
        raise ValueError('preview output must be under artifacts/media/agentic')
    summary_path, protocol_path = pilot/'summary.json', pilot/'protocol.json'
    summary, protocol, audit = read(summary_path), read(protocol_path), read(audit_path)
    if summary['status'] not in ('confirmed_witness', 'no_witness_within_budget'):
        raise ValueError('Capability search must finish without execution error')
    if audit.get('audit_passed') is not True or audit.get('original_evidence_unchanged') is not True:
        raise ValueError('A passed independent capability audit is required')
    if audit.get('status') != summary['status'] or audit.get('reported_status') != summary['status']:
        raise ValueError('Capability audit/summary status mismatch')
    for key in ('search_candidates','calibrations','confirmations','total_new_trials'):
        if audit['actual_counts'][key] != summary[key]:
            raise ValueError('Capability audit/summary counts mismatch: '+key)
    local = str(protocol.get('fixed_control_tool', '')).startswith('local_clock_')
    imported_prior_count = len(protocol.get('prior_trial_records', []))
    labels = LABELS.copy()
    if local:
        labels[2] = 'NEW LOCAL CLOCK SEARCH'
        if audit.get('audit_sources_unchanged') is not True:
            raise ValueError('Local audit source immutability check required')
    task = protocol['task']
    inputs = {str(p): sha256(p) for p in (summary_path, protocol_path, audit_path, Path(__file__).resolve())}
    old_protocol_path = linked(protocol['old_protocol'], ROOT, inputs)
    old_pilot = old_protocol_path.parent
    comparison_path = linked(protocol['old_comparison'], ROOT, inputs)
    comparison = read(comparison_path)
    inputs[str(comparison_path)] = sha256(comparison_path)
    if not comparison['comparison_valid'] or comparison['protocol_sha256'] != sha256(old_protocol_path):
        raise ValueError('Old comparison/protocol mismatch')
    old_task = next(t for t in comparison['tasks'] if t['task_id'] == task['task_id'])
    old_best = next(r['selected'] for r in old_task['methods'] if r['method'] == 'llm_feedback')
    if not old_best: raise ValueError('Old LLM selected candidate is missing')
    calibration_ref = protocol['calibration_proposal']
    cp = linked(calibration_ref,pilot,inputs) if isinstance(calibration_ref,dict) else pilot/calibration_ref
    inputs[str(cp)] = sha256(cp)
    baseline_id = read(cp)['candidate_id']
    new_id = summary.get('witness',{}).get('candidate_id') or summary.get('best_new')
    if not new_id:
        measured = [read(p) for p in (pilot/'evaluations').glob('high5_b__*capability__*.json')]
        measured = [r for r in measured if r.get('integrity_passed') is True and r.get('utility') is not None]
        if not measured: raise ValueError('No new observed candidate available')
        new_id = min(measured,key=lambda r:(r['utility'],r.get('utility_l2',float('inf')),r['candidate_id']))['candidate_id']
    confirm_id = summary.get('confirmation',{}).get('candidate_id')
    candidates = [baseline_id, old_best['candidate_id'], new_id, confirm_id]
    bases = [pilot, old_pilot, pilot, pilot]
    rows = [read(pilot/'evaluations'/f'{baseline_id}.json'), old_best,
            read(pilot/'evaluations'/f'{new_id}.json'),
            read(pilot/'evaluations'/f'{confirm_id}.json') if confirm_id else None]
    for identity, base in zip(candidates, bases):
        if identity is None: continue
        if base == pilot:
            record_path = pilot/'trial_records'/f'{identity}.json'
            record = read(record_path); inputs[str(record_path)] = sha256(record_path)
            for key in ('proposal','evaluation','manifest'): linked(record[key],ROOT,inputs)
        else:
            record = next((r for r in protocol['prior_trial_records'] if r['candidate_id'] == identity), None)
            if record is not None:
                for key in ('proposal','evaluation','execution_manifest'): linked(record[key],ROOT,inputs)
            # Local runs import no prior labels; old panel remains bound by the
            # hash-linked old comparison and its exact saved evaluation/manifest.
    traces, panel_info = [], []
    expected_times = np.arange(201) * .02
    for candidate, row, base in zip(candidates, rows, bases):
        if candidate is None:
            traces.append(None); panel_info.append(dict(available=False, reason='No confirmation: no witness within budget'))
            continue
        if not isinstance(candidate, str) or Path(candidate).name != candidate:
            raise ValueError('Invalid candidate identity')
        trial = base / 'trials' / candidate
        manifest_path = trial / 'manifest.json'
        manifest = read(manifest_path)
        if sha256(manifest_path)!=row['execution_manifest_sha256']:
            raise ValueError('Comparison execution manifest binding changed')
        evaluation_path=base/'evaluations'/f'{candidate}.json'
        evaluation=read(evaluation_path)
        if evaluation!=row:raise ValueError('Comparison row differs from saved evaluation')
        if base == pilot:
            checked = audit['trials'][candidate]
            for key in ('features','per_dimension_pass','physical_gate_passed','intent_passed','integrity_passed','utility'):
                if checked.get(key) != evaluation.get(key):
                    raise ValueError('Displayed evaluation differs from independent audit: '+candidate+':'+key)
            if checked['provenance'] != read(pilot/'trial_records'/f'{candidate}.json'):
                raise ValueError('Displayed trial provenance differs from independent audit')
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
        panel_info.append(dict(available=True, candidate_id=candidate, features=row['features'], target=task['goal']['feature_targets'],
                               per_dimension_pass=[v is not None and abs(v-g)<=tol for v,g,tol in zip(row['features'],task['goal']['feature_targets'],protocol['feature_tolerances'])],
                               source_trial=str(trial), physical_gate_passed=row['physical_gate_passed'],
                               evaluation=row))

    observation=task['music_observation']
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
    for candidate, base in zip(candidates, bases):
        if candidate is None: continue
        pinned = read(base / 'trials' / candidate / 'preexecution_inputs.json')
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
                for panel, (trace, label, row) in enumerate(zip(traces, labels, panel_info)):
                    if trace is None:
                        image = Image.new('RGB', (480,360), (18,23,31))
                        draw = ImageDraw.Draw(image)
                        draw.text((12,16), label, font=font, fill='white')
                        draw.text((12,120), 'NO FRESH CONFIRMATION', font=font, fill='white')
                        draw.text((12,150), 'No witness within the search budget.', font=small, fill='white')
                        draw.text((12,180), 'No state fabricated for this panel.', font=small, fill='white')
                        montage.paste(image, ((panel%2)*480,(panel//2)*360))
                        continue
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
                    draw.text((8,330),f'Actual / target | qpos | t={index*.02:.2f}s',font=small,fill=(195,207,222))
                    if panel == 2:
                        draw.rectangle((0,328,480,360),fill=(18,23,31))
                        draw.text((8,332),(f"0 imported +1 calibration +{summary['search_candidates']} new" if local else f"{imported_prior_count} prior + {summary['search_candidates']} new; unequal budgets"),font=small,fill=(255,210,100))
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
        'task_id':task['task_id'], 'source_id':task['source_id'], 'goal':task['goal'],
        'capability_status':summary['status'], 'cost_accounting': dict(old_path='8 LLM proposals from original comparison',
        new_path=('post-comparison local clock search with zero imported old feature labels; one fresh calibration is sole measurement prior; target-informed seed and probe schedule; zero new LLM proposal calls' if local else f'post-comparison numerical search seeded by{imported_prior_count} prior measurements; zero new LLM proposal calls'),
        imported_prior_measurements=imported_prior_count, fresh_calibration_as_search_prior=int(local),
        seed_probe_schedule=protocol.get('search',{}).get('initial_stage'),
        tool=protocol.get('fixed_control_tool','global_clock'),
        comparison_limitation='Different budgets, imported data and seed/probe schedules; not a matched causal tool comparison',
        new_search_candidates=summary['search_candidates'], calibrations=summary['calibrations'],
        confirmations=summary['confirmations'], fair_matched_benchmark=False), 'panels': panel_info, 'camera': dict(kind='tracking', trackbodyid=1,
            distance=3.6, azimuth=140, elevation=-15, same_configuration_all_panels=True),
        'state_indices': indices, 'state_sampling_hz': 50, 'video_fps': 25,
        'output_frames': 100, 'output_duration_s': 4,
        'last_displayed_state_s': 3.96, 'terminal_state_s': 4.,
        'audio': dict(path=str(audio), sha256=sha256(audio), mode='shared authored offline audio',
                      crop_applied=False, encoding='AAC lossy preview'),
        'inputs_manifest_sha256': sha256(output / 'source_manifest.json'),
        'outputs_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    return target



if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    args=parser.parse_args()
    print(render(args.pilot,args.output,args.audit))
