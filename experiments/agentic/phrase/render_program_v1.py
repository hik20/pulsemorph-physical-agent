"""Prepare or render a private comparison of TWO saved physical eight-second traces.

Default prepares only. Explicit --execute-render authorizes400mj_forward calls,
never mj_step, policy inference or a physical rerun. Optional audio is local
shared playback only: it informed candidate planning, not baseline recording.
"""
from pathlib import Path
import argparse,hashlib,json,sys,shutil,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
BASELINE=Path('C:/Users/halke/.cache/pulsemorph-agentic/experiments/phrase_execution_v1')
CANDIDATE=Path('C:/Users/halke/.cache/pulsemorph-agentic/experiments/phrase_program_execution_v1')
OUT=Path('C:/Users/halke/.cache/pulsemorph-agentic/renders/phrase_program_comparison_v1')
MUSIC=ROOT/'artifacts/runs/agentic/phrase_program_music_v1/two_section_original.wav'
MUSIC_SHA='0e0d161cbf1f109e65af17d5483c6f5172715494391859d24108438e781a6893'
SAMPLE_INDICES=(60,150,200,300,398)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def capture(source):
    import numpy as np
    result=json.loads((source/'result.json').read_text())
    expected={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1}
    if result.get('status')!='completed' or result.get('counts')!=expected or not result.get('prefix_gate',{}).get('passed'):raise ValueError('requires completed8s physical trace with recorded prefix gate')
    files={source/'result.json':sha(source/'result.json')}
    for name,digest in result['source_pins'].items():
        path=Path(name)
        if sha(path)!=digest:raise ValueError('recorded source/binary pin changed')
        files[path]=digest
    for name in ('qpos.npy','qvel.npy','state_times.npy','common_reference/manifest.json','common_reference/arrays.npz'):files[source/name]=sha(source/name)
    q=np.load(source/'qpos.npy',allow_pickle=False);v=np.load(source/'qvel.npy',allow_pickle=False);t=np.load(source/'state_times.npy',allow_pickle=False)
    if q.shape!=(401,36) or v.shape!=(401,35) or t.shape!=(401,) or not all(np.isfinite(a).all() for a in (q,v,t)) or not np.allclose(t,np.arange(401)*.02,rtol=0,atol=1e-9):raise ValueError('invalid saved physical8s clocks/layout')
    return result,files,(q,v,t)
def prepare():
    return {'schema':'pulsemorph.phrase_program_comparison_preparation.v1','status':'prepared_not_rendered','baseline':str(BASELINE),'candidate':str(CANDIDATE),'candidate_result_available':(CANDIDATE/'result.json').is_file(),'output':str(OUT),'frames':200,'fps':25,'duration_s':8,'dimensions':[1280,640],'explicit_execution_flag':'--execute-render','planned_mj_forward_calls':400,'mj_step_ONNX_calls':0,'sample_times_s':[1.2,3.,4.,6.,7.96],'audio_mode':'mutedMP4 plus optional original-audio shared replay in localHTML; baseline recorded without audio','music_sha256':MUSIC_SHA,'renderer_sha256':sha(__file__)}
def execute():
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    sys.path.insert(0,str(ROOT/'src'))
    from pulsemorph.motion import load_reference
    from pulsemorph.agentic.phrase_program import parse_program
    from pulsemorph.agentic.phrase_reference import plain
    if OUT.exists():raise FileExistsError('new comparison output required')
    if shutil.disk_usage('C:/').free<512*1024*1024:raise ValueError('render requires512MiB free')
    left,leftpins,leftdata=capture(BASELINE);right,rightpins,rightdata=capture(CANDIDATE)
    leftref=load_reference(BASELINE/'common_reference');rightref=load_reference(CANDIDATE/'common_reference')
    if leftref.metadata['producer']['id']!='fixed-wave-high5-phrase-reference' or rightref.metadata['producer']['id']!='phrase-program-reference':raise ValueError('wrong fixed/program reference producers')
    program=parse_program(plain(rightref.metadata['provenance']['phrase_program']))
    canonical=json.dumps(program,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
    if hashlib.sha256(canonical).hexdigest()!=rightref.metadata['provenance']['phrase_program_sha256']:raise ValueError('typed reference program hash mismatch')
    if leftref.metadata['body']!=rightref.metadata['body'] or leftref.metadata['controller']!=rightref.metadata['controller']:raise ValueError('comparison requires equal body/controller profiles')
    if not np.array_equal(leftdata[2],rightdata[2]):raise ValueError('saved clocks must be exactly equal')
    if sha(MUSIC)!=MUSIC_SHA:raise ValueError('original planning music hash mismatch')
    # Rendering geometry is the exact pinned body/meshes, with only view/floor/light additions.
    xml=ROOT/'research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    if sha(xml)!=leftref.metadata['body']['profile_sha256']:raise ValueError('body binary profile mismatch')
    OUT.mkdir(parents=True);tree=ET.parse(xml);doc=tree.getroot();compiler=doc.find('compiler');compiler.set('meshdir',str((xml.parent/compiler.get('meshdir')).resolve()))
    ET.SubElement(doc.find('asset'),'texture',type='skybox',builtin='gradient',rgb1='.10 .14 .22',rgb2='.03 .05 .10',width='512',height='512')
    for sensor in doc.findall('sensor'):doc.remove(sensor)
    world=doc.find('worldbody')
    if not any(g.get('type')=='plane' for g in world.findall('geom')):ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .05',rgba='.12 .16 .22 1')
    ET.SubElement(world,'light',pos='2 0 5',dir='0 0 -1',diffuse='.6 .6 .6',directional='true');tree.write(OUT/'render_model.xml',encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(OUT/'render_model.xml'));states=[mujoco.MjData(model),mujoco.MjData(model)]
    model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.8,.8,.8]
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING;camera.trackbodyid,camera.distance,camera.azimuth,camera.elevation=1,2.8,140,-10
    renderer=mujoco.Renderer(model,height=480,width=640)
    fontpath=Path('C:/Windows/Fonts/segoeui.ttf');font=ImageFont.truetype(str(fontpath),16) if fontpath.is_file() else ImageFont.load_default();small=ImageFont.truetype(str(fontpath),14) if fontpath.is_file() else font
    video=OUT/'fixed_vs_music_program_muted.mp4';writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=8,macro_block_size=1);samples=[];forward_calls=0
    try:
        for index in range(0,400,2):
            frame=Image.new('RGB',(1280,640),(12,18,28));draw=ImageDraw.Draw(frame)
            for side,(data,arrays) in enumerate(zip(states,(leftdata,rightdata))):
                q,v,t=arrays;data.qpos[:]=q[index];data.qvel[:]=v[index];data.time=float(t[index]);mujoco.mj_forward(model,data);forward_calls+=1;renderer.update_scene(data,camera=camera);frame.paste(Image.fromarray(renderer.render()),(side*640,104))
            draw.text((12,8),'FIXED COMPOSITION /',font=font,fill='white');draw.text((12,31),'recorded earlier without music planning',font=font,fill=(185,215,240));draw.text((12,61),'Original fixed Wave -> High5 reference',font=small,fill=(185,215,240))
            draw.text((652,8),'MUSIC + INSTRUCTION PROGRAM /',font=font,fill='white');draw.text((652,31),'recorded physical states',font=font,fill=(185,215,240));draw.text((652,58),' -> '.join(program['order'])+' | heading: '+program['heading_policy'],font=small,fill=(185,215,240));draw.text((652,80),f"first L{program['first_left_gain']:g} R{program['first_right_gain']:g} | second L{program['second_left_gain']:g} R{program['second_right_gain']:g}",font=small,fill=(255,205,150))
            draw.text((12,590),f'Recorded saved qpos | t={index*.02:.2f}s | mj_forward only',font=small,fill='white');draw.text((12,614),'Torso orientation C0 only | no new skill learning | no beat-alignment guarantee | muted video',font=small,fill=(185,215,240));writer.append_data(np.asarray(frame))
            if index in SAMPLE_INDICES:
                name=f'frame_{index*.02:04.2f}s.png';frame.save(OUT/name);samples.append(name)
    finally:renderer.close();writer.close()
    if forward_calls!=400:raise ValueError('unexpected render-only FK count')
    for path,digest in {**leftpins,**rightpins}.items():
        if sha(path)!=digest:raise ValueError('source/binary/trace changed during render')
    if sha(MUSIC)!=MUSIC_SHA:raise ValueError('music changed during render')
    shutil.copyfile(MUSIC,OUT/'planning_music.wav')
    if sha(OUT/'planning_music.wav')!=MUSIC_SHA:raise ValueError('planning audio copy mismatch')
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Private recorded phrase comparison</title><style>body{background:#0c121c;color:#eee;font:16px system-ui;margin:24px}video{width:min(100%,1280px)}button{padding:10px}</style><h1>Recorded physical traces</h1><p>Baseline was recorded without music planning. Measured summaries of this original audio informed only the candidate plan; no raw audio was heard by the model. Optional shared replay soundtrack is not evidence baseline heard audio or either trace follows beats.</p><video id="v" controls src="fixed_vs_music_program_muted.mp4"></video><p><label><input id="audio-enabled" type="checkbox"> Enable optional shared replay audio (off by default)</label></p><audio id="a" src="planning_music.wav"></audio><button id="play">Enable audio and play both from start</button><p id="play-status" role="status">Optional audio is off.</p><p>Four arm gains/order/heading are reference-program parameters, not guaranteed realized expression. No new skill/style learning, quality improvement, safety or admission claim.</p><script>
const v=document.querySelector('#v'),a=document.querySelector('#a'),enabled=document.querySelector('#audio-enabled'),status=document.querySelector('#play-status');
function syncAudio(force=false){if(!enabled.checked){a.pause();return;}if(force||Math.abs(a.currentTime-v.currentTime)>.15)a.currentTime=v.currentTime;}
function playAudio(){if(!enabled.checked)return;syncAudio(true);a.play().then(()=>{status.textContent='Optional audio follows video playback; no beat-alignment claim.';}).catch(()=>{a.pause();status.textContent='Audio playback was blocked. Use the play-both button or disable optional audio.';});}
enabled.onchange=()=>{if(!enabled.checked){a.pause();status.textContent='Optional audio is off.';}else{syncAudio(true);if(!v.paused)playAudio();else status.textContent='Optional audio enabled; resume the video to hear it.';}};
v.onplay=()=>{if(enabled.checked)playAudio();};v.onpause=()=>a.pause();v.onended=()=>a.pause();v.onseeking=()=>{a.pause();syncAudio(true);};v.onseeked=()=>{syncAudio(true);if(!v.paused&&enabled.checked)playAudio();};v.ontimeupdate=()=>{if(enabled.checked&&!v.paused)syncAudio();};
document.querySelector('#play').onclick=()=>{enabled.checked=true;a.pause();v.currentTime=0;a.currentTime=0;v.play().then(()=>playAudio()).catch(()=>{a.pause();status.textContent='Video playback was blocked. Press the native video play control to resume.';});};
</script>''',encoding='utf-8')
    manifest={'schema':'pulsemorph.phrase_program_saved_comparison.v1','baseline_result_sha256':sha(BASELINE/'result.json'),'candidate_result_sha256':sha(CANDIDATE/'result.json'),'baseline_array_hashes':{name:sha(BASELINE/name) for name in ('qpos.npy','qvel.npy','state_times.npy')},'candidate_array_hashes':{name:sha(CANDIDATE/name) for name in ('qpos.npy','qvel.npy','state_times.npy')},'typed_program':program,'typed_reference_manifest_sha256':sha(CANDIDATE/'common_reference/manifest.json'),'source_binary_pins_before_after_unchanged':True,'source_pin_count':{'baseline':len(leftpins),'candidate':len(rightpins)},'renderer_sha256':sha(__file__),'font_sha256':sha(fontpath) if fontpath.is_file() else None,'frames':200,'fps':25,'duration_s':8,'dimensions':[1280,640],'mj_forward_calls':forward_calls,'new_mj_step_ONNX_calls':0,'video_audio':False,'optional_shared_replay_audio_sha256':MUSIC_SHA,'audio_qualification':'Original audio features informed candidate planning only; baseline recorded without music planning. Shared optional playback is posthoc, not beat-alignment proof.','samples':samples,'video_sha256':sha(video),'qualification':'Saved physical qpos visualization only; not a new trial, independent controls audit, improvement judgment, learning or app admission.'};dump(OUT/'manifest.json',manifest);print(json.dumps({'status':'rendered_saved_states','output':str(OUT),'frames':200,'mj_forward_calls':400,'new_mj_step_ONNX_calls':0}))
def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute-render',action='store_true');args=parser.parse_args()
    if args.execute_render:execute()
    else:print(json.dumps(prepare(),indent=2))
if __name__=='__main__':main()
