"""Owned eight-second PCM music, independent of frozen experiment artifacts."""
import copy,hashlib,io,json,math,re,threading,wave,stat
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from ..uploads import _decode,_label
from ..music import RATE,_synthesize,_analyze

TRANSFORM='pcm16_downmix_resample_poly_48k_8s_v1'
LOCK=threading.RLock()
MAX_META=128*1024
MAX_WAV=800*1024
def linked(path):return path.is_symlink() or bool(getattr(path.lstat(),'st_file_attributes',0)&1024) if path.exists() or path.is_symlink() else False
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def sha(b):return hashlib.sha256(b).hexdigest()
def encode(values):
    output=io.BytesIO()
    with wave.open(output,'wb') as w:w.setparams((1,2,RATE,len(values),'NONE','not compressed'));w.writeframes(np.asarray(values,dtype='<i2').tobytes())
    return output.getvalue()

def measure(path):
    result=_analyze(path)
    with wave.open(str(path),'rb') as w:values=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
    if len(values)!=8*RATE:raise ValueError('exact8second stored PCM required')
    sections=[]
    for start,end in [(0,4),(4,8)]:
        x=values[start*RATE:end*RATE];energy=np.sqrt(np.mean(x.reshape(-1,480)**2,axis=1));i=int(np.argmax(energy));sections.append({'start_s':start,'end_s':end,'rms':float(np.sqrt(np.mean(x*x))),'peak_amplitude':float(np.max(np.abs(x))),'peak_10ms_rms':float(energy[i]),'peak_10ms_center_s':start+(i+.5)/100})
    return {**result,'schema':'pulsemorph.phrase_music_summary.v1','sample_frames':8*RATE,'sections':sections,'bpm':None,'bpm_source':'not inferred','limitations':['Measured energy/transients are not beat or style labels.','Audio summary does not guarantee motion synchronization.']}

class PhraseMusic:
    def __init__(self,root,artifact_root=None):
        self.root=Path(root).resolve();base=Path(artifact_root or Path.home()/'.cache/pulsemorph-agentic/app_phrase_v1').absolute();self.directory=base/'music';self._warnings=[]
        for path in [self.directory,*self.directory.parents]:
            if linked(path):raise ValueError('linked phrase music directory forbidden')
        self.directory.mkdir(parents=True,exist_ok=True)
        with LOCK:
            identity='phrase_original_pulse_v1'
            if not (self.directory/(identity+'.json')).exists() and not (self.directory/(identity+'.wav')).exists():
                samples,rate,channels,frames=_decode(_synthesize('pulse'));original=np.r_[np.rint(samples[:,0].astype(float)*.45),np.rint(samples[:,0].astype(float)*.90)].astype('<i2')
                self._store(identity,encode(original),'Two-section original pulse',{'kind':'procedural_owned','recipe':'app.music._synthesize(pulse), concatenate roundedPCM*.45 then*.90','generator_sha256':sha(Path(__file__).parent.parent.joinpath('music.py').read_bytes()),'source_bytes_retained':False},'Original procedural oscillators/noise, no external recording; project distribution license undecided.')
    def _store(self,identity,data,label,provenance,rights):
        path,meta=self._paths(identity)
        if path.exists() or meta.exists():return self.get(identity)
        with path.open('xb') as f:f.write(data)
        record={'id':identity,'label':label,'sha256':sha(data),**measure(path),'provenance':provenance,'rights':rights,'url':'/media/phrase/music/'+identity+'.wav'};record['descriptor_sha256']=sha(canonical(record))
        with meta.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
        return self.get(identity)
    def _paths(self,identity):
        if not isinstance(identity,str) or not re.fullmatch(r'phrase_(?:original_pulse_v1|upload_[a-f0-9]{24})',identity):raise ValueError('unknown phrase music')
        for parent in [self.directory,*self.directory.parents]:
            if linked(parent):raise ValueError('linked phrase music directory forbidden')
        path=self.directory/(identity+'.wav');meta=self.directory/(identity+'.json')
        if linked(path) or linked(meta) or path.resolve().parent!=self.directory.resolve() or meta.resolve().parent!=self.directory.resolve():raise ValueError('linked or escaping phrase music forbidden')
        return path,meta
    def get(self,identity):
        with LOCK:
            path,meta=self._paths(identity)
            if not meta.is_file() or not path.is_file():raise ValueError('phrase music record is incomplete')
            if meta.stat().st_size>MAX_META or path.stat().st_size>MAX_WAV:raise ValueError('phrase music record exceeds bounded read size')
            with meta.open('rb') as stream:metadata_bytes=stream.read(MAX_META+1)
            if len(metadata_bytes)>MAX_META:raise ValueError('phrase metadata exceeds128KiB')
            record=json.loads(metadata_bytes.decode('utf8'))
            if not isinstance(record,dict):raise ValueError('phrase music metadata object required')
            expected=record.pop('descriptor_sha256',None)
            if sha(canonical(record))!=expected or sha(path.read_bytes())!=record.get('sha256') or record.get('id')!=identity:raise ValueError('phrase music integrity mismatch')
            with wave.open(str(path),'rb') as w:
                if (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())!=(1,2,RATE,8*RATE):raise ValueError('phrase music PCM mismatch')
            measured=measure(path)
            if measured['rms']<=1e-6:raise ValueError('silent phrase audio is unsupported')
            if any(record.get(k)!=v for k,v in measured.items()):raise ValueError('phrase music descriptor differs from measured storedPCM')
            provenance=record.get('provenance')
            if not isinstance(provenance,dict) or provenance.get('source_bytes_retained') is not False:raise ValueError('phrase music provenance missing')
            if identity.startswith('phrase_upload_'):
                rate=provenance.get('original_sample_rate');frames=provenance.get('original_frame_count');begin=provenance.get('start_frame');channels=provenance.get('original_channels');source_sha=provenance.get('source_sha256');source_bytes=provenance.get('source_bytes')
                if type(rate) is not int or not 8000<=rate<=96000 or type(frames) is not int or not 8*rate<=frames<=120*rate or type(begin) is not int or not 0<=begin<=frames-8*rate or type(channels) is not int or channels not in (1,2) or not isinstance(source_sha,str) or not re.fullmatch('[a-f0-9]{64}',source_sha) or type(source_bytes) is not int or not 44<=source_bytes<=32*1024*1024:raise ValueError('invalid phrase upload source provenance')
                expected_id='phrase_upload_'+sha(canonical({'source_sha256':source_sha,'start_frame':begin,'duration_s':8,'transform':TRANSFORM}))[:24]
                if any(type(provenance.get(k)) not in (int,float) or not math.isfinite(provenance[k]) for k in ('start_s','end_s')) or identity!=expected_id or provenance.get('transform')!=TRANSFORM or provenance.get('selected_frame_count')!=8*rate or provenance.get('start_s')!=begin/rate or provenance.get('end_s')!=(begin+8*rate)/rate or provenance.get('derived_wav_sha256')!=record['sha256'] or provenance.get('derived_frame_count')!=8*RATE:raise ValueError('phrase upload identity/crop lineage mismatch')
            elif provenance.get('kind')!='procedural_owned' or provenance.get('generator_sha256')!=sha(Path(__file__).parent.parent.joinpath('music.py').read_bytes()):raise ValueError('owned phrase recipe provenance mismatch')
            record['descriptor_sha256']=expected;return {**record,'path':str(path)}
    def catalog(self):
        with LOCK:
            valid=[];warnings=[];identities=sorted({p.stem for extension in ('*.json','*.wav') for p in self.directory.glob('phrase_'+extension)})
            for identity in identities:
                try:valid.append({k:v for k,v in self.get(identity).items() if k!='path'})
                except (ValueError,OSError,KeyError,TypeError,wave.Error):warnings.append({'id':identity,'category':'invalid_or_partial_phrase_music_preserved'})
            self._warnings=warnings
            if not valid:raise ValueError('No valid eight-second phrase music is available; invalid records preserved')
            return valid
    def status(self):
        try:count=len(self.catalog());available=True
        except ValueError:count=0;available=False
        return {'local_only':True,'available':available,'duration_s':8,'catalog_count':count,'warnings':copy.deepcopy(self._warnings),'format':'uncompressed PCM16 source8..120s mono/stereo8..96kHz; derived48kmono8s','audio_analysis':'decoded storedPCM, no inferred BPM/beats','integrity_qualification':'descriptor selfhash is integrity, not authenticated provenance'}
    def import_wav(self,data,label,start_s=0):
        label=_label(label)
        if type(start_s) not in (int,float) or not math.isfinite(start_s) or start_s<0:raise ValueError('finite nonnegative crop start required')
        samples,rate,channels,frames=_decode(data)
        if frames<8*rate or start_s+8>frames/rate:raise ValueError('source must contain the complete8second crop')
        begin=round(start_s*rate)
        if begin+8*rate>frames:raise ValueError('crop exceeds actual source frames')
        selected=samples[begin:begin+8*rate].astype(float).mean(axis=1);g=math.gcd(rate,RATE);resampled=resample_poly(selected,RATE//g,rate//g)
        if len(resampled)!=8*RATE:raise ValueError('resampling did not produce exact8seconds; no padding/clamp')
        pcm=np.clip(np.rint(resampled),-32768,32767).astype('<i2')
        if float(np.sqrt(np.mean((pcm.astype(float)/32768)**2)))<=1e-6:raise ValueError('silent phrase upload unsupported; no fake accent')
        clip=encode(pcm);source_sha=sha(data);identity='phrase_upload_'+sha(canonical({'source_sha256':source_sha,'start_frame':begin,'duration_s':8,'transform':TRANSFORM}))[:24]
        provenance={'kind':'local_uploaded_PCM16','source_sha256':source_sha,'source_bytes':len(data),'source_bytes_retained':False,'original_frame_count':frames,'original_sample_rate':rate,'original_channels':channels,'selected_frame_count':8*rate,'start_frame':begin,'start_s':begin/rate,'end_s':(begin+8*rate)/rate,'transform':TRANSFORM,'derived_wav_sha256':sha(clip),'derived_frame_count':8*RATE}
        with LOCK:return self._store(identity,clip,label,provenance,'User-provided audio; ownership/redistribution unknown. Local processing only.')
