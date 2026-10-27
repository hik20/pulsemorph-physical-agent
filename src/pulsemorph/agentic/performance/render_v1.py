"""Postprocess real saved performance states, never run actor or physics.

The caller owns execution authority and supplies an explicit display-FK cap.
Audio is the exact existing measured score; no audio is synthesized here.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import wave
import xml.etree.ElementTree as ET
import numpy as np
from ..app.renderer import _run_mux

ROOT=Path(__file__).resolve().parents[4]
CACHE=Path.home()/'.cache/pulsemorph-agentic'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def configure_display_buffer(model):
    """Display framebuffer only; no native state evaluation or physics change."""
    model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),1280)
    model.vis.global_.offheight=max(int(model.vis.global_.offheight),600)
def inventory(directory):
    result={}
    for p in directory.rglob('*'):
        require(not p.is_symlink() and not(getattr(p.lstat(),'st_file_attributes',0)&1024),'Linked input forbidden')
        if p.is_file():result[str(p.resolve())]=sha(p)
    return result
def sample_indices(times,start_s=0,duration_s=None):
    times=np.asarray(times)
    require(times.ndim==1 and len(times)>=2 and np.isfinite(times).all() and np.allclose(times,np.arange(len(times))*.02,rtol=0,atol=1e-9),'Exact saved20ms clock required')
    require(type(start_s) in (int,float) and math.isfinite(start_s) and 0<=start_s<times[-1],'Finite clip start required')
    duration=float(times[-1]-start_s) if duration_s is None else duration_s
    require(type(duration) in (int,float) and math.isfinite(duration) and duration>0 and start_s+duration<=times[-1]+1e-9,'Clip must lie inside saved trace')
    require(abs(start_s/.02-round(start_s/.02))<1e-8 and abs(duration/.02-round(duration/.02))<1e-8,'Clip edges must be saved control boundaries')
    first,end=round(start_s/.02),round((start_s+duration)/.02);indices=list(range(first,end,2));indices[-1]=end
    return indices,float(duration)
def music_binding(music):
    if music is None:return None
    path=Path(music['path']).absolute();require(not path.is_symlink(),'Linked music forbidden');digest=sha(path)
    require(digest==music['sha256'],'Planning score bytes changed')
    with wave.open(str(path),'rb') as w:
        require(w.getcomptype()=='NONE' and w.getsampwidth()==2 and w.getnchannels()==1 and w.getframerate()==48000,'PCM16mono48k score required')
        frames=w.getnframes();duration=frames/w.getframerate()
    require(abs(duration-music['duration_s'])<1e-9 and math.isfinite(duration),'Measured music duration differs')
    start=music.get('start_sim_time_s',2.)
    require(type(start) in (int,float) and math.isfinite(start) and start>=0,'Music simulation offset required')
    beats=music['beats_s'];require(type(beats) is list and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<duration for v in beats) and all(b>a for a,b in zip(beats,beats[1:])),'Construction beat timestamps required')
    return {'path':str(path),'sha256':digest,'duration_s':duration,'start_sim_time_s':float(start),'beats_s':list(beats),'grid_scope':'Construction timestamps, not motion/beat alignment proof.'}
def render_performance(trace_dir,output,*,music=None,start_s=0,duration_s=None,max_fk,label,cancel_event=None):
    trace=Path(trace_dir).absolute();output=Path(output).absolute()
    require(not output.exists() and output.resolve().is_relative_to(CACHE.resolve()) and not output.resolve().is_relative_to(trace.resolve()),'New separate private output required')
    for p in (trace,output,*output.parents):require(not p.is_symlink() and not(p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024),'Linked path forbidden')
    require(type(label) is str and 1<=len(label)<=100 and '\n' not in label,'Explicit short provenance label required')
    before=inventory(trace);backend=json.loads((trace/'backend.json').read_bytes())
    require(backend['schema']=='pulsemorph.performance_backend.v1' and backend['initialization_count']==1,'Actual performance trace required')
    q=np.load(trace/'qpos.npy',allow_pickle=False);v=np.load(trace/'qvel.npy',allow_pickle=False);times=np.load(trace/'state_times.npy',allow_pickle=False)
    require(q.shape==(len(times),36) and v.shape==(len(times),35) and all(np.isfinite(a).all() for a in (q,v)),'Actual finite G1 saved states required')
    indices,duration=sample_indices(times,start_s,duration_s)
    require(type(max_fk) is int and 0<len(indices)<=max_fk<=600,'Explicit finite render-FK authority required')
    require(type(backend['source_pins']) is dict and bool(backend['source_pins']),'Source pins required')
    audio=music_binding(music);own_sha=sha(__file__);deadline=time.monotonic()+240
    def check():
        require(not(cancel_event and cancel_event.is_set()),'Render cancelled; partial evidence retained')
        require(time.monotonic()<deadline,'Render wall budget exhausted')
    def verify():
        check();require(inventory(trace)==before and sha(__file__)==own_sha,'Trace/renderer changed')
        for path,digest in backend['source_pins'].items():require(sha(path)==digest,'Performance input source changed')
        require(music_binding(music)==audio,'Planning score changed')
    verify()
    # Rebuild the original pinned source XML before display-only material/light edits.
    sys.path.insert(0,str(ROOT/'research/vendor/ProtoMotions'))
    from deployment import test_tracker_mujoco as upstream
    from pulsemorph.motion.g1 import profile_from_yaml
    from ..app.phrase.assets import MODEL
    profile=profile_from_yaml(ROOT/(MODEL[:-5]+'.yaml'));doc=ET.fromstring(upstream._patch_mjcf_xml(Path(profile['body_path'])));compiler=doc.find('compiler')
    for key in ('meshdir','texturedir'):
        if compiler is not None and compiler.get(key):compiler.set(key,str((Path(profile['body_path']).parent/compiler.get(key)).resolve()))
    require(ET.tostring(doc,encoding='unicode')==(trace/'render_model.xml').read_text(encoding='utf8'),'Actual source body patch differs')
    asset=doc.find('asset');world=doc.find('worldbody')
    ET.SubElement(asset,'texture',name='display_checker',type='2d',builtin='checker',rgb1='.10 .14 .18',rgb2='.07 .095 .12',width='512',height='512')
    ET.SubElement(asset,'material',name='display_floor',texture='display_checker',texrepeat='4 4',reflectance='.05')
    floors=[g for g in world.findall('geom') if g.get('type')=='plane']
    if not floors:floors=[ET.SubElement(world,'geom',name='display_floor_geom',type='plane',size='0 0 .05',contype='0',conaffinity='0')]
    for floor in floors:floor.set('material','display_floor');floor.set('rgba','1 1 1 1')
    for light in world.findall('light'):world.remove(light)
    ET.SubElement(world,'light',name='display_key',pos='2 -2 4',dir='-.3 .2 -1',diffuse='.8 .8 .75',directional='true')
    ET.SubElement(world,'light',name='display_fill',pos='-2 1 3',dir='.3 -.1 -1',diffuse='.35 .45 .55',directional='true')
    output.mkdir(parents=True,exist_ok=False);(output/'display_model.xml').write_text(ET.tostring(doc,encoding='unicode'),encoding='utf8')
    import mujoco
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    model=mujoco.MjModel.from_xml_path(str(output/'display_model.xml'));configure_display_buffer(model);state=mujoco.MjData(model);model.vis.headlight.ambient[:]=[.3]*3;model.vis.headlight.diffuse[:]=[.4]*3
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING;camera.trackbodyid=1;camera.distance=2.7;camera.azimuth=140;camera.elevation=-9
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    attempt={'status':'started','frames':0,'render_FK':0,'new_NN':0,'new_physics_steps':0}
    def checkpoint():
        temporary=output/'attempt.tmp';temporary.write_text(json.dumps(attempt));temporary.replace(output/'attempt.json')
    checkpoint();renderer=None;writer=None
    try:
        renderer=mujoco.Renderer(model,height=600,width=1280);writer=imageio.get_writer(str(output/'silent.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-threads','1','-movflags','+faststart'])
        for frame_index,index in enumerate(indices):
            check();state.qpos[:]=q[index];state.qvel[:]=v[index];state.time=float(times[index]);mujoco.mj_forward(model,state);attempt['render_FK']+=1;checkpoint()
            renderer.update_scene(state,camera=camera);frame=Image.new('RGB',(1280,720),(10,17,23));frame.paste(Image.fromarray(renderer.render()),(0,64));draw=ImageDraw.Draw(frame)
            draw.text((24,14),label,font=font,fill='#edf3f5');draw.text((24,42),'Kayıtlı fizik · modelin bekleme süreleri oynatımdan çıkarıldı',font=small,fill='#aec1cb')
            beat_label='Müzik yok'
            if audio:
                local_time=float(times[index])-audio['start_sim_time_s'];beat=int(np.searchsorted(audio['beats_s'],local_time,side='right'))-1
                if beat>=0 and local_time<audio['duration_s']:
                    beat_label=f'Müzik vuruşu {beat+1}/{len(audio["beats_s"])} · yapım zaman çizelgesi';last_beat=audio['beats_s'][beat];pulse=max(0,1-(local_time-last_beat)/.18);draw.ellipse((1150,15,1174,39),fill=tuple(int(a+b*pulse) for a,b in [(80,164),(90,87),(95,30)]))
                else:beat_label='Müzik kesiti dışında'
            draw.text((24,674),f'Simülasyon {times[index]:.2f} sn · {beat_label}',font=small,fill='#bce0d0')
            footer='Kayıtlı fizik performansı'+(' · 100 BPM' if audio else '')
            draw.text((24,698),footer,font=small,fill='#9faeb9')
            progress=min(1.,max(0.,(float(times[index])-start_s)/duration));draw.line((900,706,1240,706),fill='#34454e',width=3);draw.line((900,706,900+340*progress,706),fill='#f4b17d',width=3)
            if frame_index==0:frame.save(output/'first_frame.png')
            if frame_index==len(indices)-1:frame.save(output/'last_frame.png')
            writer.append_data(np.asarray(frame));attempt['frames']+=1;checkpoint()
    except Exception:
        attempt['status']='stopped';checkpoint();raise
    finally:
        if renderer is not None:renderer.close()
        if writer is not None:writer.close()
    verify();command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(output/'silent.mp4')]
    if audio:
        trim=max(0.,start_s-audio['start_sim_time_s']);delay=max(0.,audio['start_sim_time_s']-start_s)
        command+=['-itsoffset',str(delay),'-ss',str(trim),'-i',audio['path'],'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-threads','1','-t',str(len(indices)/25)]
    else:command+=['-map','0:v:0','-c:v','copy','-an']
    command+=['-movflags','+faststart',str(output/'performance.mp4')];_run_mux(command,cancel_event,timeout=max(.1,deadline-time.monotonic()));verify()
    require(attempt['render_FK']==len(indices),'Render FK count differs');attempt['status']='completed';checkpoint()
    manifest={'schema':'pulsemorph.performance_render.v1','actual':attempt,'trace_dir':str(trace),'input_inventory':before,'source_pins':backend['source_pins'],'renderer_sha256':own_sha,'label_caller_supplied_not_attested':label,'start_sim_time_s':start_s,'sampled_duration_s':duration,'video_duration_s':len(indices)/25,'sample_indices':indices,'endpoint_replaced_last_sample':True,'audio':audio,'audio_mux':None if not audio else {'trim_source_s':max(0.,start_s-audio['start_sim_time_s']),'delay_video_s':max(0.,audio['start_sim_time_s']-start_s)},'video_sha256':sha(output/'performance.mp4'),'first_frame_sha256':sha(output/'first_frame.png'),'last_frame_sha256':sha(output/'last_frame.png'),'qualification':'Actual qpos replay only. Display material/lights do not change saved physics. Construction beat indicator is not measured dance synchronization.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf8')
    return {'video':str(output/'performance.mp4'),'manifest':str(output/'manifest.json'),'render_FK':len(indices),'last_frame':str(output/'last_frame.png')}
