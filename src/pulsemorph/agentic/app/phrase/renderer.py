"""Replay audited eight-second phrase states; never step physics or run a policy."""
from pathlib import Path
import hashlib
import json
import shutil
import wave
import xml.etree.ElementTree as ET
import numpy as np

def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def _inventory(directory):
    return {p.relative_to(directory).as_posix():_sha(p) for p in directory.rglob('*') if p.is_file()}

def _directory(trial):
    if not isinstance(trial,dict) or not isinstance(trial.get('paths'),dict):raise ValueError('Recorded trial paths required')
    directory=Path(trial['paths']['directory']).resolve()
    if not directory.is_dir():raise ValueError('Recorded trial directory missing')
    return directory

def _load_trace(directory):
    directory=Path(directory).resolve()
    result=json.loads((directory/'result.json').read_text(encoding='utf8'))
    if result.get('status')!='completed':raise ValueError('Only completed recorded trials can render')
    if result.get('counts')!={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1}:raise ValueError('Recorded phrase execution counts differ')
    pins=result.get('source_pins')
    if not isinstance(pins,dict) or not pins:raise ValueError('Recorded source pins required')
    q=np.load(directory/'qpos.npy',allow_pickle=False);v=np.load(directory/'qvel.npy',allow_pickle=False);t=np.load(directory/'state_times.npy',allow_pickle=False)
    if q.shape!=(401,36) or v.shape!=(401,35) or t.shape!=(401,) or not all(np.isfinite(a).all() for a in (q,v,t)) or not np.allclose(t,np.arange(401)*.02,rtol=0,atol=1e-9):raise ValueError('Expected finite401-state eight-second recorded trace')
    return {'directory':directory,'result':result,'qpos':q,'qvel':v,'time':t,'inventory':_inventory(directory)}

def _verify_trace(row):
    if _inventory(row['directory'])!=row['inventory']:raise ValueError('Recorded trace changed during rendering')
    for name,expected in row['result']['source_pins'].items():
        if not Path(name).is_file() or _sha(name)!=expected:raise ValueError('Pinned source/asset changed; render not accepted')

def _audio_binding(rows,music):
    if not isinstance(music,dict) or not music.get('path'):raise ValueError('Measured eight-second planning PCM required')
    path=Path(music['path']).resolve();digest=_sha(path)
    if digest!=music.get('sha256'):raise ValueError('Planning PCM hash differs')
    with wave.open(str(path),'rb') as pcm:
        rate,frames,channels,width=pcm.getframerate(),pcm.getnframes(),pcm.getnchannels(),pcm.getsampwidth()
    if width!=2 or channels!=1 or frames!=8*rate:raise ValueError('Derived PCM16 mono audio must cover exactly8seconds')
    for row in rows:
        obs=row['result'].get('music_observation')
        if not isinstance(obs,dict) or obs.get('available') is not True or Path(obs.get('path','')).resolve()!=path or obs.get('sha256')!=digest or obs.get('sample_rate')!=rate or obs.get('observed_through_sample')!=frames or obs.get('duration_s')!=8:raise ValueError('Recorded planning-audio lineage differs')
    return {'path':str(path),'sha256':digest,'sample_rate':rate,'samples':frames,'duration_s':8}

def _frame_indices():return range(0,400,2)

def _program_text(trial):
    program=trial.get('program') or {}
    return 'L/R '+str(program.get('first_left_gain',1))+'/'+str(program.get('first_right_gain',1))+'  →  '+str(program.get('second_left_gain',1))+'/'+str(program.get('second_right_gain',1))

def render_phrase_comparison(root,baseline,selected,music,output_dir,*,cancel_event=None):
    """Return before/after/comparison paths. All frames depict recorded qpos.

    Before is silent. After/comparison may replay the verified planning music;
    that soundtrack does not assert that the mechanical baseline used proposals.
    """
    root=Path(root).resolve();directories=[_directory(baseline),_directory(selected)];output=Path(output_dir).resolve()
    if output.exists() or any(output==p or output.is_relative_to(p) for p in directories):raise ValueError('New separate media directory required')
    if shutil.disk_usage(output.anchor).free<256*1024*1024:raise RuntimeError('At least256MiB free disk required')
    rows=[_load_trace(p) for p in directories]
    for row in rows:_verify_trace(row)
    audio=_audio_binding(rows,music)
    from pulsemorph.motion.contracts import load_reference
    references=[load_reference(p/'common_reference') for p in directories]
    xml=root/'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    for row,ref in zip(rows,references):
        if ref.metadata['body']['profile_sha256']!=_sha(xml) or ref.time_offsets_s[-1]!=8.14:raise ValueError('Recorded common/body profile differs')
        if row['result']['source_pins'].get(str(xml))!=_sha(xml):raise ValueError('Recorded body pin missing or changed')
    tree=ET.parse(xml);doc=tree.getroot();compiler=doc.find('compiler');meshdir=(xml.parent/compiler.get('meshdir','')).resolve()
    for item in doc.findall('./asset/mesh'):
        mesh=(meshdir/item.attrib['file']).resolve()
        for row in rows:
            if row['result']['source_pins'].get(str(mesh))!=_sha(mesh):raise ValueError('Recorded mesh pin missing or changed')
    compiler.set('meshdir',str(meshdir))
    for item in doc.findall('sensor'):doc.remove(item)
    world=doc.find('worldbody')
    if not any(g.get('type')=='plane' for g in world.findall('geom')):ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .05',rgba='.12 .16 .22 1')
    if not world.findall('light'):ET.SubElement(world,'light',pos='2 0 5',dir='0 0 -1',diffuse='.6 .6 .6',directional='true')
    output.mkdir(parents=True,exist_ok=False);model_file=output/'render_model.xml';tree.write(model_file,encoding='unicode')
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from ..renderer import _run_mux
    font=None
    for name in ('C:/Windows/Fonts/segoeui.ttf','DejaVuSans.ttf'):
        try:font=ImageFont.truetype(name,18);break
        except OSError:continue
    if font is None:font=ImageFont.load_default()
    model=mujoco.MjModel.from_xml_path(str(model_file));data=mujoco.MjData(model);model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.8,.8,.8]
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING;camera.trackbodyid,camera.distance,camera.azimuth,camera.elevation=1,2.8,140,-10
    paths={name:output/(name+'.mp4') for name in ('before','after','comparison')};writers={};renderer=None
    try:
        for name in paths:writers[name]=imageio.get_writer(str(output/(name+'_silent.mp4')),fps=25,codec='libx264',quality=8,macro_block_size=1)
        renderer=mujoco.Renderer(model,height=480,width=640)
        for index in _frame_indices():
            if cancel_event and cancel_event.is_set():raise RuntimeError('Render cancelled; partial files retained')
            panels=[]
            for row,trial,label in zip(rows,(baseline,selected),('BAŞLANGIÇ · KAYITLI FİZİK','NİHAİ SEÇİM · KAYITLI FİZİK')):
                data.qpos[:]=row['qpos'][index];data.qvel[:]=row['qvel'][index];data.time=float(row['time'][index]);mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
                panel=Image.new('RGB',(640,600),(13,20,29));panel.paste(Image.fromarray(renderer.render()),(0,84));draw=ImageDraw.Draw(panel);draw.text((14,10),label,font=font,fill='white');draw.text((14,39),'Referans kazançları: '+_program_text(trial),font=font,fill='#b9ddc2');draw.text((14,571),f'Gerçekleşen qpos · t={index*.02:.2f} sn · yeni fizik yok',font=font,fill='#b7c8d9');panels.append(panel)
            both=Image.new('RGB',(1280,600));both.paste(panels[0],(0,0));both.paste(panels[1],(640,0))
            for name,image in zip(paths,[*panels,both]):writers[name].append_data(np.asarray(image))
    finally:
        if renderer is not None:renderer.close()
        for writer in writers.values():writer.close()
    for row in rows:_verify_trace(row)
    if _audio_binding(rows,music)!=audio:raise ValueError('Planning audio changed before mux')
    for name,path in paths.items():
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',str(output/(name+'_silent.mp4'))]
        if name!='before':command+=['-i',audio['path'],'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','8']
        else:command+=['-c:v','copy']
        command+=['-movflags','+faststart',str(path)];_run_mux(command,cancel_event)
    for row in rows:_verify_trace(row)
    if _audio_binding(rows,music)!=audio:raise ValueError('Planning audio changed during mux')
    manifest={'schema':'pulsemorph.phrase_app_render.v1','execution':'recorded_qpos_mj_forward_only','frames':200,'fps':25,'duration_s':8,'mj_forward_calls':400,'physics_steps':0,'NN_calls':0,'renderer_sha256':_sha(__file__),'input_result_sha256':[_sha(p/'result.json') for p in directories],'audio':audio,'soundtrack_scope':'After/comparison share planning-music replay; baseline is mechanical identity, not a model proposal or beat-alignment claim.','outputs_sha256':{p.name:_sha(p) for p in paths.values()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return {name:str(path) for name,path in paths.items()}
