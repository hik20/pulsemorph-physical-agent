"""Prepare/authorize ACTUAL-only preview/commit body-task replay. No policy calls or physics steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.tasks_v1 import validate_task

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
def inventory(path):
    result={}
    for p in path.rglob('*'):
        require(not p.is_symlink() and not(getattr(p.lstat(),'st_file_attributes',0)&1024),'Linked input forbidden')
        if p.is_file():result[str(p.resolve())]=sha(p)
    return result
def require(value,message):
    if not value:raise ValueError(message)
def write(path,value):
    with Path(path).open('xb') as stream:stream.write(encoded(value))
def frames_for(times):
    times=np.asarray(times);n=len(times)-1
    require(n>0 and n%2==0 and np.isfinite(times).all() and np.allclose(times,np.arange(n+1)*.02,rtol=0,atol=1e-9),'Exact even 20ms trace required')
    return list(range(0,n,2))
def active_action(actions,clock):
    # Saved qpos at a boundary is the completed prior action, not an unexecuted
    # next action. The first frame is explicitly initialization/prelude.
    if abs(clock)<1e-10:
        found=[a for a in actions if abs(a['start_sim_time_s'])<1e-10]
    else:
        found=[a for a in actions if a['start_sim_time_s']<clock-1e-10 and clock<=a['start_sim_time_s']+a['planned_control_steps']*.02+1e-10]
    require(len(found)==1,'Recorded state action assignment missing/overlapping')
    return found[0]

def actual_commits(journal,episode):
    mapping={}
    for returned in sorted(journal.glob('commit_*/returned.json')):
        value=json.loads(returned.read_bytes());pending=json.loads(returned.with_name('pending.json').read_bytes())
        require(value['hypothetical_only'] is False and value['replay_no_new_work'] is False and value['preview_actual_byte_equal'] is True,'Only committed actual execution may render')
        require(value['commit_id']==pending['commit_id'] and value['preview_id']==pending['preview_id'],'Commit identity mismatch')
        matched=[a for a in episode['actions'] if a['request']['action_id']==value['action_id']]
        require(len(matched)==1 and matched[0]['status']=='completed' and matched[0]['before_revision']==value['before_revision'] and matched[0]['planned_control_steps']==value['control_steps'],'Actual runtime action/commit assignment differs')
        previews=[]
        for path in journal.glob('preview_*/pending.json'):
            item=json.loads(path.read_bytes())
            if item['action']['action_id']==value['action_id']:previews.append(item['action'])
        require(len(previews)==1 and previews[0]==matched[0]['request'],'Actual committed command differs from preview command')
        calls=[]
        for path in sorted((journal/'model').glob('call_*/validated.json')):
            item=json.loads(path.read_bytes());proposal=item['proposal'];receipt=item['receipt']
            if proposal['decision']=='commit' and proposal['preview_id']==value['preview_id'] and item['observation_sha256']==pending['observation_sha256']:
                require(receipt['live'] is True and receipt.get('completion_unconfirmed') is False,'Completed actual commit decision receipt required')
                calls.append(int(path.parent.name.split('_')[-1])+1)
        require(len(calls)==1,'Actual model commit decision missing/ambiguous')
        mapping[value['action_id']]={'commit_index':int(returned.parent.name.split('_')[-1])+1,'model_call':calls[0],'preview_id':value['preview_id']}
    return mapping

def prepare(trace,task_path,journal,output):
    trace,task_path,journal,output=map(lambda p:Path(p).absolute(),(trace,task_path,journal,output))
    cache=(Path.home()/'.cache/pulsemorph-agentic').resolve()
    require(output.resolve().is_relative_to(cache) and not any(output.resolve()==p.resolve() or output.resolve().is_relative_to(p.resolve()) for p in (trace,journal)),'Separate private C-cache output required')
    for path in (trace,task_path,journal,output,*output.parents):
        require(not path.is_symlink() and not(path.exists() and getattr(path.lstat(),'st_file_attributes',0)&1024),'Linked render path forbidden')
    episode=json.loads((trace/'episode.json').read_bytes());backend=json.loads((trace/'backend.json').read_bytes())
    task=validate_task(json.loads(task_path.read_bytes()));fixed=json.loads((journal/'fixed.json').read_bytes());result=json.loads((journal/'result.json').read_bytes())
    require(fixed['task']==task and result.get('fixture_run') is False,'Actual frozen task/journal required')
    q=np.load(trace/'qpos.npy',allow_pickle=False);v=np.load(trace/'qvel.npy',allow_pickle=False);times=np.load(trace/'state_times.npy',allow_pickle=False);indices=frames_for(times)
    require(q.shape==(len(times),36) and v.shape==(len(times),35) and all(np.isfinite(x).all() for x in (q,v)),'Finite actual states required')
    require(backend['initialization_count']==episode['initialization_count']==backend['mjdata_instances']==backend['onnx_sessions']==1 and backend['online_calls']==len(times)-1,'Persistent instance/count mismatch')
    require(type(backend['source_pins']) is dict and bool(backend['source_pins']),'Source pins required')
    require(backend['schema']=='pulsemorph.embodied_hybrid_backend.v1','Actual hybrid backend required')
    mapping=actual_commits(journal,episode)
    require(len(mapping)<=result['commit_attempts'] and sum(a['planned_control_steps']*20 for a in episode['actions'] if a['request']['action_id'] in mapping)==result['confirmed_actual_steps'],'Committed actual count mismatch')
    for index in indices:active_action(episode['actions'],float(times[index]))
    pins={**inventory(trace),**inventory(journal),str(task_path.resolve()):sha(task_path),str(Path(__file__).resolve()):sha(__file__),**backend['source_pins']}
    for path,digest in pins.items():require(sha(path)==digest,'Pinned input changed')
    initial=json.loads((journal/'actual_observation_00.json').read_bytes())
    plan={'schema':'pulsemorph.embodied_recorded_render_plan.v2','trace':str(trace),'task_path':str(task_path),'journal':str(journal),'output':str(output),'input_pins':pins,'frames':len(indices),'fps':25,'duration_s':float(times[-1]),'max_FK':len(indices),'new_ONNX':0,'new_steps':0,'audio':'none','task_start_s':initial['sim_time_s'],'model_action_ids':mapping,'endpoint_task_verified':result['task_verified_actual'],'qualification':'Recorded physical simulation-time replay. Model waiting time omitted. Target marker is visual only; no contact target or new task success.'}
    return plan,(q,v,times,indices,episode,task)

def render(plan,data):
    q,v,times,indices,episode,task=data;output=Path(plan['output']);deadline=time.monotonic()+180
    def verify():
        for path,digest in plan['input_pins'].items():require(sha(path)==digest,'Pinned input changed during replay')
    verify()
    # Reconstruct the original pinned body patch before adding visual-only sites.
    sys.path.insert(0,str(ROOT/'research/vendor/ProtoMotions'))
    from deployment import test_tracker_mujoco as upstream
    from pulsemorph.motion.g1 import profile_from_yaml
    from pulsemorph.agentic.app.phrase.assets import MODEL
    profile=profile_from_yaml(ROOT/(MODEL[:-5]+'.yaml'))
    doc=ET.fromstring(upstream._patch_mjcf_xml(Path(profile['body_path'])))
    compiler=doc.find('compiler')
    for name in ('meshdir','texturedir'):
        if compiler is not None and compiler.get(name):compiler.set(name,str((Path(profile['body_path']).parent/compiler.get(name)).resolve()))
    require(ET.tostring(doc,encoding='unicode')==(Path(plan['trace'])/'render_model.xml').read_text(encoding='utf8'),'Original patched source/body XML differs')
    world=doc.find('worldbody')
    ET.SubElement(world,'geom',name='task_visual_target',type='sphere',pos=' '.join(map(str,task['target_world_m'])),size=str(task['tolerance_m']),rgba='1 .55 .18 .42',contype='0',conaffinity='0',mass='0',group='0')
    if not any(g.get('type')=='plane' for g in world.findall('geom')):ET.SubElement(world,'geom',name='display_floor',type='plane',size='0 0 .05',rgba='.12 .16 .22 1',contype='0',conaffinity='0')
    if not world.findall('light'):ET.SubElement(world,'light',pos='2 0 5',dir='0 0 -1',diffuse='.6 .6 .6',directional='true')
    (output/'display_model.xml').write_text(ET.tostring(doc,encoding='unicode'),encoding='utf8')
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    model=mujoco.MjModel.from_xml_path(str(output/'display_model.xml'));state=mujoco.MjData(model)
    wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,task['body']);require(wrist>=0,'Named wrist missing')
    marker=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'task_visual_target')
    model.vis.headlight.ambient[:]=[.5]*3;model.vis.headlight.diffuse[:]=[.8]*3
    bodycam=mujoco.MjvCamera();bodycam.type=mujoco.mjtCamera.mjCAMERA_TRACKING;bodycam.trackbodyid=1;bodycam.distance=2.5;bodycam.azimuth=140;bodycam.elevation=-10
    close=mujoco.MjvCamera();close.type=mujoco.mjtCamera.mjCAMERA_FREE;close.lookat[:]=task['target_world_m'];close.distance=.43;close.azimuth=140;close.elevation=-12
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    attempt={'frames':0,'FK':0,'NN':0,'physics_steps':0,'status':'started'}
    def checkpoint():
        temporary=output/'attempt.tmp';temporary.write_bytes(encoded(attempt));temporary.replace(output/'attempt.json')
    checkpoint();errors=[];renderer=None;writer=None
    try:
        renderer=mujoco.Renderer(model,height=480,width=640)
        writer=imageio.get_writer(str(output/'episode.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-threads','1','-movflags','+faststart'])
        for index in indices:
            require(time.monotonic()<deadline,'Render wall budget exceeded')
            state.qpos[:]=q[index];state.qvel[:]=v[index];state.time=float(times[index]);mujoco.mj_forward(model,state);attempt['FK']+=1;checkpoint()
            task_active=times[index]>=plan['task_start_s']-1e-10;model.geom_rgba[marker,3]=.42 if task_active else 0
            error=float(np.linalg.norm(state.xpos[wrist]-np.asarray(task['target_world_m'])));errors.append({'state_index':index,'sim_time_s':float(times[index]),'distance_m':error})
            action=active_action(episode['actions'],float(times[index]));info=plan['model_action_ids'].get(action['request']['action_id']);finished=abs(float(times[index])-(action['start_sim_time_s']+action['planned_control_steps']*.02))<1e-9
            if info is not None:action_label=f'Gerçek uygulama {info["commit_index"]} · model çağrısı {info["model_call"]}'
            elif times[index]<plan['task_start_s']+1e-10:action_label='Hazırlık / motor tutuşu'
            else:action_label='Kısmi gerçek yürütme · tamamlanma doğrulanmadı'
            action_label+=(' tamamlandı' if finished else ' sürüyor')
            frame=Image.new('RGB',(1280,640),(13,20,29));draw=ImageDraw.Draw(frame)
            draw.text((18,12),'KAYITLI FİZİK · aynı beden, kesintisiz simülasyon',font=font,fill='white')
            draw.text((18,42),'Model bekleme süreleri çıkarıldı · simülasyon zamanı oynatılıyor',font=small,fill='#b7c8d9')
            for column,camera in enumerate((bodycam,close)):
                renderer.update_scene(state,camera=camera);frame.paste(Image.fromarray(renderer.render()),(column*640,80))
            draw.text((18,570),f't={times[index]:.2f} sn · {action_label}',font=font,fill='#9bd8c9')
            measure=f'Hedefe uzaklık {error*1000:.1f} mm · tolerans {task["tolerance_m"]*1000:g} mm' if task_active else f'Hazırlık · hedef t={plan["task_start_s"]:.2f} sn anında verildi'
            draw.text((658,570),measure,font=font,fill='#ffb076')
            footer='Turuncu küre görsel hedef. Varsayımsal önizleme hareketleri bu videoya dahil değildir.'
            if index==indices[-1]:footer='Gerçek uç nokta sonucu: '+('görev doğrulandı' if plan['endpoint_task_verified'] else 'görev doğrulanmadı')+' · bu kayıt yeni yürütme değildir.'
            draw.text((18,608),footer,font=small,fill='#b7c8d9')
            writer.append_data(np.asarray(frame));attempt['frames']+=1;checkpoint()
        require(attempt['FK']==plan['max_FK'],'FK budget mismatch');verify();attempt['status']='completed';checkpoint()
    except Exception:
        attempt['status']='stopped';checkpoint();raise
    finally:
        if renderer is not None:renderer.close()
        if writer is not None:writer.close()
    verify();write(output/'frame_errors.json',errors);write(output/'manifest.json',{'plan_sha256':hashlib.sha256(encoded(plan)).hexdigest(),'actual':attempt,'video_sha256':sha(output/'episode.mp4'),'audio':'none','endpoint_task_verified_unchanged':plan['endpoint_task_verified']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('task',type=Path);p.add_argument('journal',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--authority',type=Path);p.add_argument('--authority-sha256');a=p.parse_args()
    plan,data=prepare(a.trace,a.task,a.journal,a.output);digest=hashlib.sha256(encoded(plan)).hexdigest()
    if not a.execute:a.output.mkdir(parents=True,exist_ok=False);write(a.output/'prepared.json',plan);print(json.dumps({'prepared_sha256':digest,'frames':plan['frames'],'new_FK':0}))
    else:
        require(a.authority is not None and sha(a.authority)==a.authority_sha256,'Exact separate root render authority required');authority=json.loads(a.authority.read_bytes())
        require(authority=={'schema':'pulsemorph.embodied_render_authority.v2','approved':True,'prepared_sha256':digest,'max_FK':plan['max_FK'],'new_ONNX':0,'new_steps':0},'Authority/budget mismatch')
        require(json.loads((a.output/'prepared.json').read_bytes())==plan,'Prepared input changed');write(a.output/'execution_attempt.json',{'authority_sha256':a.authority_sha256,'retry_allowed':False});render(plan,data)
