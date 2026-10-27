"""Owned procedural100BPM score and measured PCM descriptor, no model/physics."""
import hashlib
import io
import json
from pathlib import Path
import wave
import numpy as np

RATE=48000
DURATION=16.8
BPM=100
BEAT=.6
SEED=20261028

def synthesize_music():
    frames=806400;mix=np.zeros(frames,dtype=np.float64);rng=np.random.default_rng(SEED)
    def add(start,sound,gain):
        index=round(start*RATE);count=min(len(sound),frames-index)
        if count>0:mix[index:index+count]+=gain*sound[:count]
    def tone(freq,duration,decay=4):
        t=np.arange(round(duration*RATE))/RATE
        return (np.sin(2*np.pi*freq*t)+.22*np.sin(4*np.pi*freq*t))*np.exp(-decay*t/max(duration,.01))*np.minimum(1,t/.006)
    bass=[55.,65.40639132514966,73.41619197935188,61.7354126570155,55.,73.41619197935188,65.40639132514966]
    melody=[220.,261.6255653005986,293.6647679174076,329.6275569128699,293.6647679174076,261.6255653005986,246.94165062806206,220.]
    for beat in range(28):
        onset=beat*BEAT;bar=beat//4;energy= .78 if beat<8 else .92 if beat<16 else 1.
        # Pitch-swept synthesized kick, brushed synthetic snare, no sample assets.
        t=np.arange(round(.25*RATE))/RATE;kick=np.sin(2*np.pi*(45*t+70*.025*(1-np.exp(-t/.025))))*np.exp(-t/ .065)
        add(onset,kick,.46*energy)
        if beat%4 in (1,3):
            t=np.arange(round(.18*RATE))/RATE;noise=rng.standard_normal(len(t));noise=np.concatenate(([0.],np.diff(noise)))*np.exp(-t/.035);add(onset,noise,.07*energy)
        add(onset,tone(bass[bar],.49,3),.27*energy)
        add(onset+.3,tone(bass[bar]*2,.19,4),.085*energy)
        if beat%2==0 or beat>=16:add(onset+.15,tone(melody[(beat+bar)%8],.34,3),.10*energy)
        for offset in (0.,.3):
            t=np.arange(round(.07*RATE))/RATE;noise=rng.standard_normal(len(t));hat=np.concatenate(([0.],np.diff(noise)))*np.exp(-t/.012);add(onset+offset,hat,.022*energy)
        if beat in (0,8,16,24):
            for ratio in (2.,2.5,3.):add(onset,tone(bass[bar]*ratio,1.15,3),.055)
    # Short edge fades avoid clicks without a silent introduction or padding.
    attack=round(.005*RATE);release=round(.06*RATE);mix[:attack]*=np.linspace(0,1,attack);mix[-release:]*=np.linspace(1,0,release)
    mix*=.92/max(float(np.max(np.abs(mix))),1e-12)
    pcm=np.rint(mix*32767).astype('<i2');stream=io.BytesIO()
    with wave.open(stream,'wb') as score:score.setnchannels(1);score.setsampwidth(2);score.setframerate(RATE);score.writeframes(pcm.tobytes())
    raw=stream.getvalue();normalized=pcm.astype(float)/32768
    sections=[]
    for start,end,label in [(0,4.8,'İlk motif'),(4.8,9.6,'Yanıt'),(9.6,16.8,'Genişleyen son bölüm')]:
        samples=normalized[round(start*RATE):round(end*RATE)];sections.append({'start_s':start,'end_s':end,'label':label,'rms':float(np.sqrt(np.mean(samples*samples)))})
    waveform=[float(np.max(np.abs(chunk))) for chunk in np.array_split(normalized,128)]
    descriptor={'schema':'pulsemorph.performance_music.v1','id':'owned_groove_100_v1','label':'Pulse groove · 100 BPM','sha256':hashlib.sha256(raw).hexdigest(),'sample_rate':RATE,'sample_frames':frames,'channels':1,'pcm_bits':16,'duration_s':DURATION,'bpm':BPM,'bpm_source':'deterministic score construction; not an audio tempo estimate','beat_period_s':BEAT,'beats_s':[round(i*BEAT,10) for i in range(28)],'phrase_boundaries_s':[0.,4.8,9.6,16.8],'phrases':sections,'rms':float(np.sqrt(np.mean(normalized*normalized))),'peak_amplitude':float(np.max(np.abs(normalized))),'waveform128':waveform,'source':{'kind':'procedural oscillators and seeded noise','seed':SEED,'external_audio_assets':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'qualification':'Construction beat grid and measured audio energy, not learned musicality or motion synchronization evidence. Source/score distribution license remains a separate user decision.'}
    return raw,descriptor

def create_demo_music(directory):
    directory=Path(directory).absolute()
    for p in (directory,*directory.parents):
        if p.is_symlink() or (p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024):raise ValueError('Linked score output forbidden')
    raw,descriptor=synthesize_music();directory.mkdir(parents=True,exist_ok=False);path=directory/'groove_100bpm.wav'
    with path.open('xb') as stream:stream.write(raw)
    descriptor={**descriptor,'path':str(path)}
    with (directory/'descriptor.json').open('x',encoding='utf8') as stream:json.dump(descriptor,stream,indent=2,ensure_ascii=False,allow_nan=False)
    return descriptor
