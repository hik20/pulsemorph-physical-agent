"""Immutable first-four-second PCM derived from a verified eight-second record."""
import copy,hashlib,io,json,threading,wave
from pathlib import Path
import numpy as np
from ..app.music import _analyze
from .assets_v1 import ROOT_CACHE,ordinary,sha

LOCK=threading.RLock()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
class DanceMusic:
    def __init__(self,music8,artifact_root=ROOT_CACHE/'music4'):
        self.music8=music8;self.directory=ordinary(artifact_root);self.directory.mkdir(parents=True,exist_ok=True)
    def __call__(self,identity):
        with LOCK:
            source=self.music8.get(identity);p=ordinary(source['path'])
            if sha(p)!=source['sha256']:raise ValueError('Eight-second source changed')
            with wave.open(str(p),'rb') as w:
                if (w.getcomptype(),w.getsampwidth(),w.getnchannels(),w.getframerate(),w.getnframes())!=('NONE',2,1,48000,384000):raise ValueError('Verified8s48kmono PCM16 required')
                pcm=w.readframes(192000)
            if len(pcm)!=384000:raise ValueError('Incomplete four-second source; no padding')
            out=io.BytesIO()
            with wave.open(out,'wb') as w:w.setparams((1,2,48000,192000,'NONE','not compressed'));w.writeframes(pcm)
            data=out.getvalue();digest=hashlib.sha256(data).hexdigest();path=ordinary(self.directory/(digest+'.wav'));meta=ordinary(self.directory/(digest+'.json'))
            lineage={'id':identity,'sha256':source['sha256'],'selected_sample_start':0,'selected_samples':192000}
            if path.exists():
                if path.read_bytes()!=data:raise ValueError('Derived music collision/tamper preserved')
            else:
                with path.open('xb') as f:f.write(data)
            values=np.frombuffer(pcm,dtype='<i2').astype(float)/32768
            sections=[{'start_s':a,'end_s':b,'rms':float(np.sqrt(np.mean(values[a*48000:b*48000]**2))),'peak_amplitude':float(np.max(np.abs(values[a*48000:b*48000])))} for a,b in [(0,2),(2,4)]]
            record={**_analyze(path),'schema':'pulsemorph.dance_music_summary.v1','id':identity,'label':source['label'],'sha256':digest,'duration_s':4,'sample_rate':48000,'sample_frames':192000,'channels':1,'pcm_bits':16,'sections':sections,'source8s':lineage,'bpm':None,'bpm_source':'not inferred','limitations':['First4s of selected verified8s PCM. Energy/transients are descriptive, not beat/style/synchronization proof.']}
            # Same PCM can legitimately be selected under another source ID; the
            # immutable shared descriptor stores PCM measurement, not source alias.
            measurement={k:v for k,v in record.items() if k not in ('id','label','source8s')}
            if meta.exists():
                if meta.stat().st_size>128*1024 or json.loads(meta.read_bytes())!=measurement:raise ValueError('Derived music metadata changed')
            else:
                with meta.open('xb') as f:f.write(canonical(measurement))
            if sha(p)!=source['sha256'] or sha(path)!=digest:raise ValueError('Music changed during derivation')
            return {**copy.deepcopy(record),'path':str(path.resolve())}
