"""Build a captioned local demo from saved UI, real traces and recorded job evidence.

No models, simulation, remote services or invented UI frames are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[3]
JOBS=ROOT/'artifacts/runs/agentic/app/jobs'
WAVE='20261027T023422_8efe4bc4'
HIGH5='20261027T022322_76db8965'
UNSUPPORTED='20261027T022051_538b6498'
REUSE='20261027T023826_67ff3517'
BG='#101513';PANEL='#1c2420';INK='#f5f3e9';MUTED='#afbeb1';ORANGE='#ffae73';MINT='#b6dec5'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def font(size,bold=False):
    path=Path('C:/Windows/Fonts/seguisb.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf')
    return ImageFont.truetype(str(path),size)
def wrap(draw,text,size,width,bold=False):
    lines=[];current=''
    for word in text.split():
        candidate=(current+' '+word).strip()
        if draw.textlength(candidate,font=font(size,bold))>width and current:
            lines.append(current);current=word
        else:current=candidate
    if current:lines.append(current)
    return lines

def paragraph(draw,text,x,y,width,size=30,color=INK,gap=12,bold=False):
    for line in wrap(draw,text,size,width,bold):
        draw.text((x,y),line,font=font(size,bold),fill=color);y+=size+gap
    return y

def base(number,title,kicker='RECORDED LOCAL WORKFLOW'):
    image=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(image)
    d.rounded_rectangle((88,58,127,97),8,fill=MINT)
    d.text((99,60),'p',font=font(29,True),fill=BG)
    d.text((146,58),'PulseMorph',font=font(32,True),fill=INK)
    d.text((146,101),'PROVISIONAL NAME  /  RESEARCH PROTOTYPE',font=font(16),fill=MUTED)
    d.text((1170,75),kicker,font=font(20),fill=ORANGE)
    paragraph(d,title,96,170,1720,58,INK,bold=True)
    d.line((96,1012,1824,1012),fill='#334237',width=2)
    d.text((96,1034),'Condensed recorded demonstration • saved UI + measured physical trials • no wall-clock live claim',font=font(21),fill=MUTED)
    d.text((1730,1032),f'{number:02d} / 09',font=font(21),fill=MUTED)
    return image

def card(d,box,label,value,note=None,color=MINT):
    x,y,w,h=box;d.rounded_rectangle((x,y,x+w,y+h),18,fill=PANEL)
    d.text((x+28,y+25),label.upper(),font=font(22),fill=MUTED)
    paragraph(d,value,x+28,y+74,w-56,48,color,bold=True)
    if note:paragraph(d,note,x+28,y+h-92,w-56,23,MUTED)

def screenshot(image,path,box,region=None):
    shot=Image.open(path).convert('RGB')
    if region:shot=shot.crop(region)
    x,y,w,h=box;shot.thumbnail((w,h),Image.Resampling.LANCZOS)
    image.paste(shot,(x+(w-shot.width)//2,y))

def build(output,wave_screenshot=None,memory_screenshot=None):
    if output.exists():raise FileExistsError('Demo output directory already exists; use a new version')
    jobs={key:read(JOBS/key/'state.json') for key in (WAVE,HIGH5,UNSUPPORTED,REUSE)}
    if any(j['status']!='complete' for j in jobs.values()):raise ValueError('All narrated jobs must be complete before rendering')
    wave,high5,reuse=jobs[WAVE],jobs[HIGH5],jobs[REUSE]
    assert wave['outcome']=='target_met' and len(wave['trials'])==3 and len(wave['model_calls'])==5
    assert high5['outcome']=='partial' and sum(high5['selected']['metrics']['per_dimension_pass'])==3
    assert len(reuse['memory_used'])>=1
    assert jobs[UNSUPPORTED]['outcome']=='unsupported' and not jobs[UNSUPPORTED]['trials']
    ui=ROOT/'artifacts/media/agentic/app_ui_v1'
    high5_ui=ui/'high5-final-desktop.png';unsupported_ui=ui/'desktop-unsupported-record.png'
    wave_ui=Path(wave_screenshot).resolve() if wave_screenshot else high5_ui
    memory_ui=Path(memory_screenshot).resolve() if memory_screenshot else wave_ui
    wave_video=JOBS/WAVE/'media/comparison.mp4';high5_video=JOBS/HIGH5/'media/comparison.mp4'
    music_wave=ROOT/'artifacts/media/agentic/app_music_v1'/f"{wave['request']['music_id']}.wav"
    music_high5=ROOT/'artifacts/media/agentic/app_music_v1'/f"{high5['request']['music_id']}.wav"
    inputs={Path(__file__).resolve(),high5_ui,unsupported_ui,wave_ui,memory_ui,wave_video,high5_video,music_wave,music_high5}
    inputs.update(JOBS/key/'state.json' for key in jobs)
    for key in (WAVE,HIGH5):
        inputs.add(JOBS/key/'media/manifest.json');inputs.add(JOBS/key/'targets.json')
        inputs.update((JOBS/key/'model_calls').glob('*/receipt.json'))
    for key in (WAVE,HIGH5):
        manifest=read(JOBS/key/'media/manifest.json')
        assert sha(JOBS/key/'media/comparison.mp4')==manifest['outputs_sha256']['comparison.mp4']
        inputs.add(JOBS/key/'trials/baseline/manifest.json')
        inputs.add(JOBS/key/'trials'/f"trial_{jobs[key]['selected']['index']:02d}"/'manifest.json')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    assert sha(music_wave)==wave['music']['sha256'] and sha(music_high5)==high5['music']['sha256']
    output.mkdir(parents=True);(output/'build_script_snapshot.py').write_bytes(Path(__file__).read_bytes());scenes=[]
    def add(image,duration,label,video=None,audio=None):
        index=len(scenes);plate=output/f'{index:02d}_{label}.png';image.save(plate)
        scenes.append({'index':index,'label':label,'duration_s':duration,'plate':plate,'video':video,'audio':audio or music_wave})
    image=base(1,'A precise request. A measured movement.');d=ImageDraw.Draw(image)
    paragraph(d,'An agentic motion editor you can inspect.',96,290,1650,43,MINT)
    paragraph(d,'Natural language + original music → bounded motion tools → a pretrained controller → real simulated physics.',96,390,1550,37,INK)
    card(d,(96,600,530,290),'Requested model','gpt-6.1-sol','Real receipts. Resolved model ID is not reported.')
    card(d,(660,600,560,290),'Real execution','ONNX → PD → MuJoCo','50 Hz policy / 1,000 Hz numerical physics.')
    card(d,(1254,600,570,290),'Evidence boundary','4 seconds / 2 motions','Edits existing motions. No new learned weights.',ORANGE)
    add(image,8,'opening')
    image=base(2,'Start with an explicit, testable intent.');d=ImageDraw.Draw(image)
    shot=Image.open(wave_ui);screenshot(image,wave_ui,(96,285,910,700),(0,0,shot.width,min(shot.height,850)))
    paragraph(d,'SAVED APPLICATION SCREENSHOT',1055,294,700,23,ORANGE)
    paragraph(d,'Wave: left / right arm targets',1055,350,700,38,MINT,bold=True)
    paragraph(d,'Section A: 1.10 / 0.90',1055,420,700,35)
    paragraph(d,'Section B: 0.90 / 1.10',1055,470,700,35)
    paragraph(d,'Align the dominant motion accent with Late Rise: measured audio target 3.005 seconds.',1055,550,700,33)
    paragraph(d,'Ratios are physical joint-motion proxies relative to this job’s unedited baseline. They are not literal hand width or aesthetic scores.',1055,735,700,27,MUTED)
    add(image,12,'request')
    image=base(3,'The model proposes. The local runtime decides.');d=ImageDraw.Draw(image)
    steps=[('01','Interpret','Intent → frozen targets'),('02','Propose','4 gains + 1 timing shift'),('03','Admit','Strict common motion contract'),('04','Execute','Pretrained ONNX / PD / MuJoCo'),('05','Measure','Physical gate + five features')]
    for i,(n,title,note) in enumerate(steps):
        y=300+i*126;d.rounded_rectangle((96,y,1824,y+106),14,fill=PANEL)
        d.text((125,y+25),n,font=font(33),fill=ORANGE);d.text((245,y+24),title,font=font(36,True),fill=MINT)
        d.text((720,y+29),note,font=font(31),fill=INK)
    add(image,12,'bounded_loop')
    image=base(4,'Three proposals. Three fresh physical measurements.');d=ImageDraw.Draw(image)
    values=[t['metrics']['score'] for t in wave['trials']]
    for i,value in enumerate(values):
        card(d,(96+i*588,325,550,280),f'Proposal {i+1}',f'{value:.3f}','Worst normalized feature error. Lower is better.',MINT if value<=1 else ORANGE)
    d.line((120,706,1795,706),fill=ORANGE,width=3)
    for i,value in enumerate(values):
        x=360+i*588;d.ellipse((x-13,693,x+13,719),fill=MINT if value<=1 else ORANGE)
    paragraph(d,'The first two candidates missed timing by 0.105 s. The third reached 3.000 s against a fixed 3.005 s target.',96,760,1710,36)
    paragraph(d,'4 fresh physical trials including baseline • 5 real model calls including interpretation and explanation • fixed tolerances throughout',96,912,1710,27,MUTED)
    add(image,14,'feedback')
    image=base(5,'Wave: all five measured tolerances passed.');d=ImageDraw.Draw(image)
    paragraph(d,'Physical gate passed independently. Accent: 3.000 s / 3.005 s target. Observed joints, not aesthetic judgment.',96,265,1720,24,MINT)
    d.text((96,310),'RECORDED REPLAY  /  original speed  /  four-second clip looped  /  not live capture',font=font(20),fill=ORANGE)
    add(image,16,'wave_result',wave_video)
    image=base(6,'High five: a useful partial result, with a real gap.');d=ImageDraw.Draw(image)
    paragraph(d,'Selected: 3 / 5 tolerances • physics passed • accent 2.220 s / 2.405 s target. Timing remains 0.185 s early.',96,265,1720,24,ORANGE)
    d.text((96,310),'RECORDED REPLAY  /  original speed  /  four-second clip looped  /  not live capture',font=font(20),fill=ORANGE)
    d.rectangle((96,307,1824,340),fill=BG)
    d.text((96,310),'RECORDED REPLAY  /  worst error 5.447 → selected 1.850  /  historical earliest-tie selection  /  not live capture',font=font(20),fill=ORANGE)
    add(image,14,'partial_result',high5_video,music_high5)
    image=base(7,'Carry forward observations, not claims of learning.');d=ImageDraw.Draw(image)
    shot=Image.open(memory_ui);screenshot(image,memory_ui,(96,280,910,700),(270,265,min(shot.width,1400),min(shot.height,900)))
    paragraph(d,'ACTUAL PERSISTED CONTEXT',1055,300,700,23,ORANGE)
    paragraph(d,f"{len(reuse['memory_used'])} prior measured record retrieved",1055,365,700,38,MINT,bold=True)
    paragraph(d,'The subsequent completed wave job received real Meterless memory context from an earlier app trial.',1055,510,700,32)
    paragraph(d,'This is persistent experience retrieval. It does not update the model or controller weights, and it does not prove that memory improves quality.',1055,715,700,28,MUTED)
    add(image,12,'memory')
    image=base(8,'Useful boundaries are part of the demonstration.');d=ImageDraw.Draw(image)
    shot=Image.open(unsupported_ui);screenshot(image,unsupported_ui,(96,280,910,700),(710,335,min(shot.width,1400),min(shot.height,1130)))
    paragraph(d,'OUTSIDE THE TOOL DOMAIN',1055,300,700,23,ORANGE)
    paragraph(d,'Unsupported request: honestly declined.',1055,355,700,39,MINT,bold=True)
    paragraph(d,'1 real model call. No physical trial. No fabricated movement.',1055,500,700,32)
    paragraph(d,'Zetaris access is unavailable / not verified. The competition integration is not ready.',1055,635,700,32,ORANGE)
    paragraph(d,'No new dance skill, general body transfer, hardware acceptance or live recovery is claimed.',1055,820,700,27,MUTED)
    add(image,14,'boundaries')
    image=base(9,'An inspectable prototype. An honest next step.');d=ImageDraw.Draw(image)
    paragraph(d,'Model call receipts. Hash-pinned physical inputs. Full realized traces. Frozen targets. Explicit misses.',96,305,1720,43,MINT)
    paragraph(d,'The comparison clips above replay saved realized qpos from actual ONNX → PD → MuJoCo trials. They are not live capture or new simulation.',96,510,1640,34)
    paragraph(d,'Next: stronger timing controllability, broader validated skills, and a verified external data integration.',96,690,1650,35)
    d.rounded_rectangle((96,858,1824,950),15,fill=PANEL)
    d.text((126,883),'Local research preview • provisional PulseMorph name • zero public publishing in this build',font=font(28),fill=ORANGE)
    add(image,10,'closing')
    for scene in scenes:
        plate=scene['plate'];clip=output/f"{scene['index']:02d}.mp4"
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-loop','1','-framerate','25','-i',str(plate)]
        if scene['video']:
            # Real recorded clip, looped at original speed; no invented intermediate states.
            command+=['-stream_loop','-1','-i',str(scene['video']),'-stream_loop','-1','-i',str(scene['audio'])]
            command+=['-filter_complex','[1:v]scale=1728:648[recorded];[0:v][recorded]overlay=96:344:shortest=1[v]',
                      '-map','[v]','-map','2:a:0']
        else:command+=['-stream_loop','-1','-i',str(scene['audio']),'-map','0:v:0','-map','1:a:0']
        command+=['-t',str(scene['duration_s']),'-vf' if not scene['video'] else '-metadata',
                  'format=yuv420p' if not scene['video'] else 'comment=Looped recorded physics comparison; not live capture',
                  '-af','volume=0.15','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p',
                  '-r','25','-c:a','aac','-b:a','128k','-movflags','+faststart',str(clip)]
        subprocess.run(command,check=True,capture_output=True,timeout=180)
        scene['clip']=clip
    concat=output/'concat.txt';concat.write_text(''.join("file '"+str(s['clip']).replace('\\','/')+"'\n" for s in scenes))
    final=output/'pulsemorph_recorded_workflow.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0','-i',str(concat),
                    '-c','copy','-movflags','+faststart',str(final)],check=True,capture_output=True,timeout=90)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(final)]))
    duration=float(probe['format']['duration']);assert 100<=duration<=130
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    (output/'ffprobe.json').write_text(json.dumps(probe,indent=2))
    for name,timestamp in [('first',2),('middle',56),('last',108)]:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-ss',str(timestamp),'-i',str(final),'-frames:v','1',str(output/(name+'.png'))],check=True,capture_output=True,timeout=30)
    assert all(sha(ROOT/name)==digest for name,digest in hashes.items()),'A source changed during build; do not accept this output'
    manifest={'schema':'pulsemorph.agentic_recorded_workflow_video.v1','duration_s':duration,'width':1920,'height':1080,
              'source_hashes':hashes,'video_sha256':sha(final),'narrated_jobs':list(jobs),
              'model_calls_during_build':0,'physics_steps_during_build':0,'public_upload':False,
              'scope':'Authored English caption cards, actual saved UI screenshots/crops, looped recorded physical comparison video. Time compressed; not wall-clock live.',
              'music':'Original app composition/synthesis at 0.15 gain; no external media or voice service.',
              'scenes':[{'index':s['index'],'label':s['label'],'duration_s':s['duration_s'],'source_video':str(s['video']) if s['video'] else None} for s in scenes]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'video':str(final),'duration_s':duration}))
    return final

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/media/agentic/app_demo_video_v1')
    parser.add_argument('--wave-screenshot',type=Path);parser.add_argument('--memory-screenshot',type=Path)
    args=parser.parse_args();build(args.output.resolve(),args.wave_screenshot,args.memory_screenshot)
if __name__=='__main__':main()
