"""Audit-gated view of saved repair qpos. No physics rollout or model call."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from generalization_preview import read, sha256, write_new_json, verify_video

CONDITIONS = {'wave_a': 'combined_minus', 'high5_b': 'joint_position_minus'}
METHODS = ('surrogate_search', 'llm_feedback', 'llm_feedback_off')
TITLES = ('OLD INCUMBENT', 'SELECTED NUMERIC', 'LLM + FEEDBACK', 'LLM NEW FEEDBACK OFF')
LEGACY_REVIEW = ROOT/'artifacts/runs/agentic/repair_legacy_fields_review_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def inventory(path):
    return {str(p.relative_to(path)): sha256(p) for p in path.rglob('*') if p.is_file()}


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def linked(record, inputs):
    require(set(record) == {'path', 'sha256'}, 'exact artifact link required')
    path = resolve(record['path'])
    require(sha256(path) == record['sha256'], 'changed linked input: ' + str(path))
    inputs[str(path)] = record['sha256']
    return path


def gate(pilot, audit_path):
    audit, protocol, summary, selection = [read(p) for p in
        (audit_path, pilot/'protocol.json', pilot/'summary.json', pilot/'selection.json')]
    digest = sha256(pilot/'protocol.json')
    require(audit.get('schema') == 'pulsemorph.agentic_repair_audit.v2'
        and audit.get('audit_passed') is True and audit.get('errors') == []
        and audit.get('panel_evidence_unchanged') is True
        and audit.get('protected_evidence_unchanged') is True
        and audit.get('audit_sources_unchanged') is True
        and audit.get('protected_inventories_before_after_equal') is True
        and audit.get('bound_prior_reports_unchanged') is True
        and audit.get('legacy_bounded_impact_checks_passed') is True
        and audit.get('raw_metadata_semantically_correct') is False
        and audit.get('input_protocol_sha256') == digest
        and resolve(audit['input_panel']) == pilot, 'passed exact-protocol repair audit required')
    require(protocol.get('schema') == 'pulsemorph.agentic_robust_repair_protocol.v1'
        and summary.get('schema') == 'pulsemorph.agentic_repair_summary.v1'
        and summary.get('completed') is True and summary.get('errors') == []
        and summary.get('physical_trials') == 250 and summary.get('model_proposal_turns') == 32
        and summary.get('control_steps') == 50000 and summary.get('physics_steps') == 1000000
        and summary.get('protocol_sha256') == selection.get('protocol_sha256') == digest
        and selection.get('confirmation_used_for_selection') is False
        and len(selection['winners']) == len(summary['winners']) == 6,
        'complete frozen 250-trial / 32-proposal panel required')
    require(not (pilot/'abort.json').exists(), 'aborted panel')
    for name, pin in protocol['execution_code_sha256'].items():
        require(sha256(ROOT/name) == pin, 'frozen execution code changed: ' + name)
    return protocol, summary, selection


def render(pilot, task_id, output, audit_path):
    pilot, output, audit_path = map(resolve, (pilot, output, audit_path))
    require(task_id in CONDITIONS, 'only the two fixed diagnostic tasks are supported')
    require(not output.exists(), 'preview output already exists')
    require((ROOT/'artifacts/media/agentic').resolve() in output.parents, 'output scope')
    protocol, summary, selection = gate(pilot, audit_path)
    before = inventory(pilot)
    inputs = {str(p): sha256(p) for p in
        (audit_path, pilot/'protocol.json', pilot/'summary.json', pilot/'selection.json',
         Path(__file__).resolve(), Path(__file__).with_name('generalization_preview.py'))}
    disclosure_path, annotations_path = LEGACY_REVIEW/'report.json', LEGACY_REVIEW/'annotations.json'
    disclosure, annotations = read(disclosure_path), read(annotations_path)
    require(disclosure.get('schema') == 'pulsemorph.repair_legacy_fields_disclosure.v1'
        and disclosure.get('bounded_impact_checks_passed') is True
        and disclosure.get('raw_metadata_semantically_correct') is False
        and disclosure.get('inputs_unchanged_during_review') is True
        and disclosure.get('errors') == [] and len(annotations) == 250,
        'bounded legacy metadata disclosure and all trial annotations required')
    require(linked(disclosure['protocol'], inputs) == pilot/'protocol.json'
        and linked(disclosure['selection'], inputs) == pilot/'selection.json',
        'legacy disclosure protocol/selection mismatch')
    for path in (disclosure_path, annotations_path):
        inputs[str(path)] = sha256(path)
    task = next(t for t in protocol['tasks'] if t['task_id'] == task_id)
    condition, base = CONDITIONS[task_id], resolve(protocol['baseline_panel'])
    prior = read(linked(protocol['shared_priors'][task_id], inputs))
    incumbent = next(r for r in prior['observations'] if r['candidate_id'] == 'incumbent')
    winners = [next(w for w in selection['winners'] if w['task_id'] == task_id and w['method'] == m) for m in METHODS]
    panels = [(incumbent, base/'evaluations'/f'{condition}__selected_{task_id}.json',
               base/'trials'/f'{condition}__selected_{task_id}', 'incumbent')]
    for winner in winners:
        record = winner['condition_records'][condition]
        evaluation = linked(record['evaluation'], inputs)
        source = base if winner['candidate_id'] == 'incumbent' else pilot
        trial = source/'trials'/record['trial_id']
        if source == pilot:
            annotation = [a for a in annotations if a['trial_id'] == record['trial_id']]
            require(len(annotation) == 1, 'exact selected trial annotation required')
            annotation = annotation[0]
            require(annotation['repair_candidate_id'] == winner['candidate_id']
                and annotation['condition_id'] == condition
                and all(v is True for v in annotation['checks'].values()), 'selected trial annotation checks failed')
            high_path = linked(annotation['current_high_level_proposal'], inputs)
            require(high_path == pilot/'proposals'/(winner['candidate_id']+'.json'), 'current proposal lineage mismatch')
            high = read(high_path)
            require(all(high[k] == winner[k] for k in ('candidate_id', 'task_id', 'method', 'control_tool', 'parameters')),
                'selected high-level proposal identity/parameters differ')
            for name in ('response', 'observation'):
                linked(high[name], inputs)
            require(linked(annotation['copied_proposal'], inputs) == trial/'proposal.json', 'annotation trial mismatch')
        panels.append((winner, evaluation, trial, winner['method']))
    traces, descriptions = [], []
    expected_times = np.arange(201)*.02
    xml = ROOT/'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    inputs[str(xml)] = sha256(xml)
    for row, evaluation_path, trial, method in panels:
        evaluation, manifest = read(evaluation_path), read(trial/'manifest.json')
        require(evaluation.get('integrity_passed') is True and manifest['status'] == 'success', 'invalid saved trial')
        require(manifest['control_tool'] == row['control_tool'], 'saved trial tool differs')
        pins = manifest['preexecution_input_hashes']
        xml_pin = next(v for k, v in pins.items() if Path(k).name == xml.name)
        require(xml_pin == sha256(xml), 'saved trial model XML differs')
        for name, pin in manifest['artifact_hashes'].items():
            path = (trial/name).resolve()
            require(trial.resolve() in path.parents and sha256(path) == pin, 'saved trial artifact changed')
            inputs[str(path)] = pin
        for p in (evaluation_path, trial/'manifest.json'):
            inputs[str(p)] = sha256(p)
        qpos, times = [np.load(trial/n, allow_pickle=False) for n in ('qpos.npy', 'state_times.npy')]
        require(qpos.shape == (201, 36) and np.isfinite(qpos).all(), '201 finite saved qpos samples required')
        require(times.shape == (201,) and np.allclose(times, expected_times, rtol=0, atol=1e-10), '50 Hz saved clock required')
        traces.append(qpos)
        descriptions.append(dict(method=method, candidate_id=row['candidate_id'], control_tool=row['control_tool'],
            task_id=task_id, condition_id=condition, source_trial=str(trial), source_evaluation=str(evaluation_path),
            source_evaluation_sha256=sha256(evaluation_path), source_manifest_sha256=sha256(trial/'manifest.json'),
            primary=evaluation['primary'], physical_gate_passed=evaluation['physical_gate_passed']))
    audio = resolve(task['music_observation']['waveform_path'])
    require(sha256(audio) == task['music_observation']['waveform_sha256'], 'fixed task audio changed')
    inputs[str(audio)] = sha256(audio)
    tree = ET.parse(xml)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = (xml.parent/compiler.get('meshdir', '')).resolve()
    compiler.set('meshdir', str(meshdir))
    for mesh in root.findall('./asset/mesh'):
        asset = meshdir/mesh.attrib['file']
        inputs[str(asset)] = sha256(asset)
        for _, _, trial, _ in panels:
            pins = read(trial/'manifest.json')['preexecution_input_hashes']
            require(any(Path(k).name == asset.name and v == inputs[str(asset)] for k, v in pins.items()), 'saved trial mesh differs')
    for sensor in root.findall('sensor'):
        root.remove(sensor)
    world = root.find('worldbody')
    if not any(g.get('type') == 'plane' for g in world.findall('geom')):
        ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 0.05', rgba='.7 .7 .7 1')
    if not world.findall('light'):
        ET.SubElement(world, 'light', pos='2 0 5', dir='0 0 -1', diffuse='.4 .4 .4', specular='.1 .1 .1', directional='true')
    output.mkdir(parents=True, exist_ok=False)
    tree.write(output/'render_model.xml', encoding='unicode')
    shutil.copyfile(__file__, output/'preview_source.py')
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    font_path = 'C:/Windows/Fonts/arial.ttf'
    font, small = [ImageFont.truetype(font_path, n) for n in (16, 14)]
    model = mujoco.MjModel.from_xml_path(str(output/'render_model.xml'))
    require(model.nq == 36, 'unexpected render body')
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 3.6, 140, -15
    renderer = mujoco.Renderer(model, height=360, width=480)
    indices = list(range(0, 200, 2))
    accent_frame = min(range(100), key=lambda i: abs(i/25-task['goal']['feature_targets'][4]))
    snapshots = {50: 'middle.png', accent_frame: 'near_accent.png', 99: 'last.png'}
    silent = output/'recorded_states_silent.mp4'
    try:
        with imageio.get_writer(str(silent), fps=25, codec='libx264', quality=8, macro_block_size=1) as writer:
            for frame, index in enumerate(indices):
                montage = Image.new('RGB', (960, 720))
                for panel, (trace, desc, title) in enumerate(zip(traces, descriptions, TITLES)):
                    data.qpos[:] = trace[index]
                    data.qvel[:] = 0
                    data.time = float(expected_times[index])
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    image = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 480, 48), fill=(18, 23, 31))
                    draw.text((8, 3), f'{title} | {desc["control_tool"]}', font=font, fill='white')
                    draw.text((8, 25), f'{task_id} | {condition}', font=small, fill='white')
                    draw.rectangle((0, 272, 480, 360), fill=(18, 23, 31))
                    primary, goals = desc['primary'], task['goal']['feature_targets']
                    for line in range(2):
                        text = '  '.join(f'{["L/A","R/A","L/B","R/B"][j]} {primary["features"][j]:.2f}/{goals[j]:.2f} {"P" if primary["per_dimension_pass"][j] else "F"}' for j in range(line*2,line*2+2))
                        draw.text((8, 274+line*18), text, font=small, fill='white')
                    peak = primary['features'][4]
                    draw.text((8, 310), f'Accent {peak:.2f}/{goals[4]:.2f}s | Primary {"PASS" if primary["joint_success"] else "FAIL"} | Physics {"PASS" if desc["physical_gate_passed"] else "FAIL"}', font=small, fill='white')
                    draw.text((8, 331), f'Saved qpos t={index*.02:.2f}s | experiment 250P / 32M', font=small, fill=(195,207,222))
                    montage.paste(image, ((panel%2)*480, (panel//2)*360))
                writer.append_data(np.asarray(montage))
                if frame in snapshots:
                    montage.save(output/snapshots[frame])
    finally:
        renderer.close()
    target = output/'comparison.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', str(silent),
        '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
        '-t', '4', '-movflags', '+faststart', str(target)], check=True, capture_output=True)
    check = verify_video(target)
    write_new_json(output/'ffprobecheck.json', check)
    require(inventory(pilot) == before, 'panel evidence changed during rendering')
    require(all(sha256(Path(p)) == pin for p, pin in inputs.items()), 'preview input changed during rendering')
    write_new_json(output/'manifest.json', dict(schema='pulsemorph.agentic_repair_preview.v1',
        execution_kind='recorded_physics_state_render_mj_forward', development_only=True,
        acceptance_scope='numerical_active_lineage_with_legacy_metadata_exception',
        raw_execution_metadata_complete=False,
        legacy_metadata_exception=dict(report_path=str(disclosure_path), report_sha256=sha256(disclosure_path),
            annotations_path=str(annotations_path), annotations_sha256=sha256(annotations_path),
            bounded_impact_checks_passed=True, raw_metadata_semantically_correct=False,
            panel_annotations_authority='Current frozen selection and linked high-level proposals/responses/evaluations; legacy raw execution descriptive fields are not authoritative.'),
        scope='View only saved qpos; no mj_step, ONNX, model calls, replay, resimulation, aesthetics or quality acceptance.',
        task_id=task_id, condition_id=condition, panels=descriptions, inputs_sha256=inputs,
        state_indices=indices, state_sampling_hz=50, output_frames=100, video_fps=25, duration_s=4,
        terminal_saved_state_s=4, last_displayed_state_s=3.96, planned_panel_budget=protocol['planned_budget'],
        camera=dict(kind='tracking', trackbodyid=1, distance=3.6, azimuth=140, elevation=-15, same_all_panels=True),
        audio=dict(path=str(audio), sha256=sha256(audio), same_for_all_panels=True, encoding='AAC lossy preview'),
        snapshots=dict(middle=2.0, near_accent=accent_frame/25, last=3.96), ffprobe=check,
        outputs_sha256=inventory(output)))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'output', 'audit'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--task', choices=CONDITIONS, required=True)
    args = parser.parse_args()
    print(render(args.pilot, args.task, args.output, args.audit))
