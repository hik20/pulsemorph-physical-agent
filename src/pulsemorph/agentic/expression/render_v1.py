"""Replay two originally audited expression traces; no policy or physics stepping."""
from pathlib import Path
import json
import shutil
import time
import xml.etree.ElementTree as ET
import numpy as np
from ..app.phrase.renderer import _sha, _load_trace, _verify_trace, _audio_binding
from ..app.renderer import _run_mux


def _output(path, directories):
    raw = Path(path).absolute()
    for part in (raw, *raw.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 1024):
            raise ValueError('Linked render output forbidden')
    output = raw.resolve()
    if output.exists() or any(output == p or output.is_relative_to(p) for p in directories):
        raise ValueError('New separate render output required')
    if output.drive.upper() != 'C:': raise ValueError('Private C-cache render output required')
    allowed = (Path.home() / '.cache/pulsemorph-agentic').resolve()
    if not output.is_relative_to(allowed): raise ValueError('Private artifact cache required')
    return output


def _gain_text(trial):
    p = trial['expression']
    return f"High5 sol · ortalama {p['second_left_pose_gain']:g} · değişim {p['second_left_motion_gain']:g}"


def render_comparison(root, runtime, baseline, selected, music, output, *, cancel_event=None):
    root = Path(root).resolve()
    directories = [Path(t['paths']['directory']).resolve() for t in (baseline, selected)]
    output = _output(output, directories)
    deadline = time.monotonic() + 180
    def check():
        if cancel_event and cancel_event.is_set(): raise RuntimeError('Render cancelled; partial evidence retained')
        if time.monotonic() >= deadline: raise RuntimeError('Render wall budget exhausted; partial evidence retained')
    def verify():
        check()
        for trial in (baseline, selected): runtime._verify_trial(trial)
        for row in rows: _verify_trace(row)
        if _audio_binding(rows, music) != audio: raise ValueError('Planning audio changed')
        if _sha(__file__) != source_sha: raise ValueError('Renderer source changed')
    check()
    for trial in (baseline, selected): runtime._verify_trial(trial)
    rows = [_load_trace(p) for p in directories]
    audio = _audio_binding(rows, music); source_sha = _sha(__file__)
    verify()
    if shutil.disk_usage(output.anchor).free < 256 * 1024**2: raise RuntimeError('Render requires256MiB free')
    xml = root / 'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    tree = ET.parse(xml); doc = tree.getroot(); compiler = doc.find('compiler')
    meshdir = (xml.parent / compiler.get('meshdir', '')).resolve()
    for asset in [xml, *[(meshdir / m.attrib['file']).resolve() for m in doc.findall('./asset/mesh')]]:
        if any(r['result']['source_pins'].get(str(asset)) != _sha(asset) for r in rows): raise ValueError('Body/mesh pin differs')
    compiler.set('meshdir', str(meshdir))
    for sensor in doc.findall('sensor'): doc.remove(sensor)
    world = doc.find('worldbody')
    if not any(g.get('type') == 'plane' for g in world.findall('geom')):
        ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05', rgba='.12 .16 .22 1')
    if not world.findall('light'): ET.SubElement(world, 'light', pos='2 0 5', dir='0 0 -1', diffuse='.6 .6 .6', directional='true')
    output.mkdir(parents=True, exist_ok=False)
    attempt = {'schema':'pulsemorph.expression_render_attempt.v1','frames_completed':0,'new_fk':0,'new_ONNX':0,'new_steps':0,'status':'started','retry_allowed':False}
    def save_attempt():
        temp = output / 'attempt.tmp'; temp.write_text(json.dumps(attempt), encoding='utf8'); temp.replace(output / 'attempt.json')
    save_attempt()
    model_file = output / 'render_model.xml'; tree.write(model_file, encoding='unicode')
    import mujoco
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    model = mujoco.MjModel.from_xml_path(str(model_file)); data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [.5]*3; model.vis.headlight.diffuse[:] = [.8]*3
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 2.8, 140, -10
    renderer = None; writer = None
    try:
        writer = imageio.get_writer(str(output / 'render_silent.mp4'), fps=25, codec='libx264', quality=8, macro_block_size=1)
        renderer = mujoco.Renderer(model, height=480, width=640)
        labels = ['BİRİM BAŞLANGIÇ · KAYITLI FİZİK', 'BİRİM BAŞLANGIÇ SEÇİLDİ · KAYITLI FİZİK' if selected['id']==baseline['id'] else 'NİHAİ SEÇİM · KAYITLI FİZİK']
        for index in range(0,400,2):
            check(); combined = Image.new('RGB',(1280,600))
            for column,(row,trial,label) in enumerate(zip(rows,(baseline,selected),labels)):
                check(); data.qpos[:] = row['qpos'][index]; data.qvel[:] = row['qvel'][index]; data.time = float(row['time'][index])
                mujoco.mj_forward(model,data); attempt['new_fk'] += 1; save_attempt()
                renderer.update_scene(data,camera=camera)
                panel = Image.new('RGB',(640,600),(13,20,29)); panel.paste(Image.fromarray(renderer.render()),(0,84))
                draw=ImageDraw.Draw(panel); draw.text((14,10),label,font=font,fill='white'); draw.text((14,39),_gain_text(trial),font=font,fill='#b9ddc2')
                draw.text((14,63),'Planlama müziği · kayıtlı fizik',font=font,fill='#b7c8d9')
                draw.text((14,571),f'Gerçekleşen qpos · {index*.02:.2f} sn · yeni fizik yok',font=font,fill='#b7c8d9'); combined.paste(panel,(column*640,0))
            writer.append_data(np.asarray(combined)); attempt['frames_completed'] += 1; save_attempt()
    except Exception:
        attempt['status']='stopped'; save_attempt(); raise
    finally:
        if renderer is not None: renderer.close()
        if writer is not None: writer.close()
    verify()
    _run_mux([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(output/'render_silent.mp4'),'-i',audio['path'],'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','8','-movflags','+faststart',str(output/'comparison.mp4')],cancel_event,timeout=max(.1,deadline-time.monotonic()))
    verify()
    if attempt['new_fk']!=400 or attempt['frames_completed']!=200: raise RuntimeError('Render budget differs')
    attempt['status']='completed'; save_attempt()
    manifest={'schema':'pulsemorph.expression_render.v1','execution':'recorded_qpos_mj_forward_only','frames':200,'fps':25,'duration_s':8,'new_fk':400,'new_ONNX':0,'new_steps':0,'renderer_sha256':source_sha,'audio':audio,'audio_scope':'Verified planning music replay; not beat alignment evidence.','baseline_id':baseline['id'],'selected_id':selected['id'],'input_results_sha256':[_sha(p/'result.json') for p in directories],'inventories':[r['inventory'] for r in rows],'source_pins':[r['result']['source_pins'] for r in rows],'comparison_sha256':_sha(output/'comparison.mp4')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return {'comparison':str(output/'comparison.mp4'),'manifest':str(output/'manifest.json'),'new_fk':400}
