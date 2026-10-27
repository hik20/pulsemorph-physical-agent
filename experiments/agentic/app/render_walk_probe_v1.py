"""Render saved physical state traces with mj_forward, without new physics."""
import json
from pathlib import Path
import subprocess
import shutil
import os
import tempfile
import time
import wave
import xml.etree.ElementTree as ET
import numpy as np
from pulsemorph.agentic.app.runtime import _module

def _audio_binding(audited,audio_path):
    """Require the same actually observed PCM in both physical trial manifests."""
    from pulsemorph.evidence import sha256
    observations=[row['manifest'].get('audio_observation') for row in audited]
    if any(not isinstance(obs,dict) or type(obs.get('available')) is not bool for obs in observations):
        raise ValueError('explicit recorded audio observation required for rendering')
    if all(not obs['available'] for obs in observations):
        if audio_path: raise ValueError('no-audio trials cannot receive a posthoc soundtrack')
        return None
    if not all(obs['available'] for obs in observations) or not audio_path:
        raise ValueError('both trials must bind the supplied observed audio')
    path=Path(audio_path).resolve()
    digest=sha256(path)
    with wave.open(str(path),'rb') as pcm:
        rate,count=pcm.getframerate(),pcm.getnframes()
    for obs in observations:
        if Path(obs['path']).resolve()!=path or obs['sha256']!=digest:
            raise ValueError('render audio path/hash differs from observed PCM')
        if (obs['sample_rate']!=rate or obs['observed_through_sample']!=count
                or obs['duration_s']!=count/rate or count/rate!=4.):
            raise ValueError('render PCM timing differs from recorded observation')
    return {'path':str(path),'sha256':digest,'sample_rate':rate,
            'observed_through_sample':count,'duration_s':count/rate}

def _run_mux(command,cancel_event=None,timeout=60):
    """Poll ffmpeg so cancellation interrupts the process tree promptly."""
    if cancel_event and cancel_event.is_set(): raise RuntimeError('render cancelled')
    with tempfile.TemporaryFile() as log:
        process=subprocess.Popen(command,stdout=log,stderr=log,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        started=time.monotonic()
        try:
            while process.poll() is None:
                if cancel_event and cancel_event.is_set(): raise RuntimeError('render cancelled; partial files retained')
                if time.monotonic()-started>timeout: raise RuntimeError('render mux timed out; partial files retained')
                time.sleep(.05)
            if process.returncode: raise RuntimeError('render mux failed; partial files retained')
        finally:
            if process.poll() is None:
                if os.name=='nt':
                    try:
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=5)
                    except subprocess.SubprocessError:
                        process.kill()
                else: process.kill()
                if process.poll() is None: process.kill()
                process.wait(timeout=5)

def render_comparison(baseline_dir,selected_dir,output_dir,audio_path,cancel_event=None):
    root=Path(__file__).resolve().parents[3]
    directories=[Path(baseline_dir).resolve(),Path(selected_dir).resolve()]
    audited=[_module(root,'evaluate_multifeature')._audit(p) for p in directories]
    audio_binding=_audio_binding(audited,audio_path)
    output=Path(output_dir).resolve()
    allowed=(root/'artifacts/media/agentic/app_walk_admission_v1').resolve()
    if allowed not in output.parents: raise ValueError('render output must be inside this Walk pilot media directory')
    if shutil.disk_usage(root).free < 256*1024*1024: raise RuntimeError('at least 256MB free disk required for render')
    if cancel_event and cancel_event.is_set(): raise RuntimeError('render cancelled')
    output.mkdir(parents=True,exist_ok=False)
    xml=root/'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    from pulsemorph.evidence import sha256
    for row in audited:
        if row['manifest']['preexecution_input_hashes'][str(xml.relative_to(root))]!=sha256(xml):
            raise ValueError('render XML differs from physics XML')
    tree=ET.parse(xml);doc=tree.getroot();compiler=doc.find('compiler')
    asset=doc.find('asset')
    if asset is None: asset=ET.SubElement(doc,'asset')
    if not any(t.get('type')=='skybox' for t in asset.findall('texture')):
        ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.10 .14 .22',rgb2='.03 .05 .10',width='512',height='512')
    meshdir=(xml.parent/compiler.get('meshdir','')).resolve()
    for mesh in doc.findall('./asset/mesh'):
        path=meshdir/mesh.attrib['file']
        for row in audited:
            if row['manifest']['preexecution_input_hashes'].get(str(path.relative_to(root)))!=sha256(path):
                raise ValueError('render mesh differs from recorded physics mesh')
    compiler.set('meshdir',str((xml.parent/compiler.get('meshdir','')).resolve()))
    for sensor in doc.findall('sensor'): doc.remove(sensor)
    world=doc.find('worldbody')
    if not any(g.get('type')=='plane' for g in world.findall('geom')):
        ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .05',rgba='.12 .16 .22 1')
    if not world.findall('light'):
        ET.SubElement(world,'light',pos='2 0 5',dir='0 0 -1',diffuse='.6 .6 .6',directional='true')
    model_file=output/'render_model.xml';tree.write(model_file,encoding='unicode')
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    model=mujoco.MjModel.from_xml_path(str(model_file));data=mujoco.MjData(model)
    model.vis.headlight.ambient[:]=[.5,.5,.5]
    model.vis.headlight.diffuse[:]=[.8,.8,.8]
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid,camera.distance,camera.azimuth,camera.elevation=1,2.8,140,-10
    paths={name:output/(name+'.mp4') for name in ('before','after','comparison')}
    writers={name:imageio.get_writer(str(output/(name+'_silent.mp4')),fps=25,codec='libx264',quality=8,macro_block_size=1) for name in paths}
    renderer=mujoco.Renderer(model,height=480,width=640)
    try:
        for index in range(0,200,2):
            if cancel_event and cancel_event.is_set(): raise RuntimeError('render cancelled; partial files retained')
            images=[]
            for row,label in zip(audited,('WALK IDENTITY / RECORDED PHYSICS','WALK EDIT / RECORDED PHYSICS')):
                data.qpos[:]=row['arrays']['qpos'][index];data.qvel[:]=0;data.time=index*.02
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,640,34),fill=(12,18,28));draw.text((14,10),label,fill='white')
                draw.text((14,450),f'Recorded realized qpos | t={index*.02:.2f}s',fill='white')
                images.append(image)
            canvas=Image.new('RGB',(1280,480));canvas.paste(images[0],(0,0));canvas.paste(images[1],(640,0))
            for name,image in zip(('before','after','comparison'),[*images,canvas]):writers[name].append_data(np.asarray(image))
            if index==100:canvas.save(output/'middle.png')
    finally:
        renderer.close()
        for writer in writers.values():writer.close()
    for name,path in paths.items():
        if cancel_event and cancel_event.is_set(): raise RuntimeError('render cancelled; partial files retained')
        if _audio_binding(audited,audio_path)!=audio_binding:
            raise ValueError('observed audio changed before mux')
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',str(output/(name+'_silent.mp4'))]
        if audio_path:
            command+=['-i',str(audio_path),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','4']
        else:command+=['-c:v','copy']
        command+=['-movflags','+faststart',str(path)]
        _run_mux(command,cancel_event)
        if _audio_binding(audited,audio_path)!=audio_binding:
            raise ValueError('observed audio changed during mux; outputs are not accepted')
    manifest={'schema':'pulsemorph.walk_probe_render.v1','execution_kind':'recorded_qpos_mj_forward',
        'renderer_sha256':sha256(Path(__file__)),
        'physics_steps':0,'onnx_calls':0,'fps':25,'frames':100,'duration_s':4,
        'input_manifest_sha256':[sha256(p/'manifest.json') for p in directories],
        'audio_sha256':audio_binding['sha256'] if audio_binding else None,
        'audio_observation':audio_binding or {'available':False},
        'outputs_sha256':{p.name:sha256(p) for p in paths.values()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {**{k:str(v) for k,v in paths.items()},'middle':str(output/'middle.png'),'manifest':str(output/'manifest.json')}
