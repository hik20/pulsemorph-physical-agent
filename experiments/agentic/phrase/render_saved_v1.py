"""Private saved-physical-state render; mj_forward only, never stepping/inference."""
from pathlib import Path
import json
import hashlib
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path('C:/Users/halke/.cache/pulsemorph-agentic/experiments/phrase_execution_v1')
OUT=ROOT/'artifacts/media/agentic/phrase_execution_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    result=json.loads((SOURCE/'result.json').read_text())
    if result['status']!='completed' or not result['prefix_gate']['passed']:raise ValueError('requires complete finite gated trace')
    for p,h in result['source_pins'].items():
        if sha(p)!=h:raise ValueError('recorded source pin changed')
    q=np.load(SOURCE/'qpos.npy',allow_pickle=False);v=np.load(SOURCE/'qvel.npy',allow_pickle=False);t=np.load(SOURCE/'state_times.npy',allow_pickle=False)
    if q.shape!=(401,36) or v.shape!=(401,35) or t.shape!=(401,) or not all(np.isfinite(x).all() for x in (q,v,t)) or not np.all(np.diff(t)>0):raise ValueError('invalid saved trace')
    OUT.mkdir(parents=True,exist_ok=False)
    xml=ROOT/'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    tree=ET.parse(xml);doc=tree.getroot();compiler=doc.find('compiler');compiler.set('meshdir',str((xml.parent/compiler.get('meshdir')).resolve()))
    asset=doc.find('asset');ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.10 .14 .22',rgb2='.03 .05 .10',width='512',height='512')
    for sensor in doc.findall('sensor'):doc.remove(sensor)
    world=doc.find('worldbody')
    if not any(g.get('type')=='plane' for g in world.findall('geom')):ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .05',rgba='.12 .16 .22 1')
    ET.SubElement(world,'light',pos='2 0 5',dir='0 0 -1',diffuse='.6 .6 .6',directional='true')
    tree.write(OUT/'render_model.xml',encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(OUT/'render_model.xml'));data=mujoco.MjData(model)
    model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.8,.8,.8]
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid,camera.distance,camera.azimuth,camera.elevation=1,2.8,140,-10
    renderer=mujoco.Renderer(model,height=480,width=640);writer=imageio.get_writer(str(OUT/'phrase_recorded_physics.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1)
    samples=[]
    try:
        for i in range(0,400,2):
            data.qpos[:]=q[i];data.qvel[:]=v[i];data.time=float(t[i]);mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera);image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
            label='WAVE REFERENCE' if i*.02<=3.5 else ('HERMITE / SLERP TRANSITION' if i*.02<4.5 else 'HIGH5 REFERENCE')
            draw.rectangle((0,0,640,55),fill=(12,18,28));draw.text((12,8),'PHRASE TRIAL / RECORDED PHYSICAL QPOS',fill='white');draw.text((12,29),label+' | no audio',fill=(180,215,240))
            draw.text((12,448),f't={t[i]:.2f}s | saved state + mj_forward only',fill='white')
            writer.append_data(np.asarray(image))
            if i in (0,150,175,200,225,300,398):
                path=OUT/f'frame_{i:03d}.png';image.save(path);samples.append(path.name)
    finally:renderer.close();writer.close()
    manifest={'schema':'pulsemorph.phrase_saved_state_render.v1','private_review_only':True,'input_result_sha256':sha(SOURCE/'result.json'),'input_arrays_sha256':{n:sha(SOURCE/n) for n in ('qpos.npy','qvel.npy','state_times.npy')},'renderer_sha256':sha(__file__),'frames':200,'fps':25,'duration_s':8,'mj_forward_calls':200,'new_physics_steps':0,'new_onnx_calls':0,'audio':False,'sample_images':samples,'video_sha256':sha(OUT/'phrase_recorded_physics.mp4'),'qualification':'Actual saved physical qpos replay only; visualization does not independently audit all controls/current FK or authorize app admission.'}
    with (OUT/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')


if __name__=='__main__':main()
