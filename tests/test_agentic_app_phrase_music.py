"""Real PCM conversion fixtures, no model/controller/simulation."""
import tempfile,unittest,io,wave,hashlib,json
from pathlib import Path
import numpy as np
from pulsemorph.agentic.app.phrase.music import PhraseMusic

def wav(seconds=10,rate=44100,channels=2):
    x=(np.sin(np.arange(seconds*rate)*.03)*10000).astype('<i2');samples=np.tile(x[:,None],(1,channels));b=io.BytesIO()
    with wave.open(b,'wb') as w:w.setparams((channels,2,rate,len(samples),'NONE','not compressed'));w.writeframes(samples.tobytes())
    return b.getvalue()
class PhraseMusicTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.music=PhraseMusic(Path.cwd(),Path(self.temp.name))
    def test_generated_original_is_owned8s_measured_not_frozen_dependency(self):
        m=self.music.get('phrase_original_pulse_v1');self.assertEqual(m['sha256'],'0e0d161cbf1f109e65af17d5483c6f5172715494391859d24108438e781a6893');self.assertEqual(m['sample_frames'],384000);self.assertAlmostEqual(m['sections'][1]['rms']/m['sections'][0]['rms'],2,places=3);self.assertIsNone(m['bpm'])
    def test_stereo44100_crop_real_frames_hash_and_reopen(self):
        data=wav();m=self.music.import_wav(data,'fixture',1.234);self.assertTrue(m['id'].startswith('phrase_upload_'));self.assertEqual(m['provenance']['start_frame'],round(1.234*44100));self.assertEqual(m['provenance']['source_sha256'],hashlib.sha256(data).hexdigest());self.assertFalse(m['provenance']['source_bytes_retained']);self.assertEqual(m['sample_frames'],384000)
        with wave.open(m['path'],'rb') as w:self.assertEqual((w.getnframes(),w.getframerate(),w.getnchannels()),(384000,48000,1))
        reopened=PhraseMusic(Path.cwd(),Path(self.temp.name));self.assertEqual(reopened.get(m['id'])['sha256'],m['sha256']);self.assertEqual(len(reopened.catalog()),2)
    def test_short_crop_invalid_start_and_no_padding_rejected(self):
        for data,start in [(wav(4),0),(wav(8),.1),(wav(),-1),(wav(),True),(wav(),float('nan'))]:
            with self.assertRaises(ValueError):self.music.import_wav(data,'fixture',start)
    def test_corrupt_partial_metadata_preserved_valid_original_still_visible(self):
        root=self.music.directory;bad=root/('phrase_upload_'+'f'*24+'.json');bad.write_text('{bad');catalog=self.music.catalog();self.assertEqual(len(catalog),1);self.assertTrue(bad.exists());self.assertEqual(len(self.music.status()['warnings']),1)
    def test_silent_upload_rejected_without_fake_accent(self):
        output=io.BytesIO()
        with wave.open(output,'wb') as w:w.setparams((1,2,8000,64000,'NONE','not compressed'));w.writeframes(bytes(128000))
        with self.assertRaisesRegex(ValueError,'silent'):self.music.import_wav(output.getvalue(),'silent',0)
        self.assertEqual(len(self.music.catalog()),1)
    def test_descriptor_selfhash_is_not_remeasurement_authority(self):
        from pulsemorph.agentic.app.phrase.music import canonical,sha
        path=self.music.directory/'phrase_original_pulse_v1.json';record=json.loads(path.read_text());record.pop('descriptor_sha256');record['rms']=.9;record['descriptor_sha256']=sha(canonical(record));path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'measured storedPCM'):self.music.get('phrase_original_pulse_v1')
        self.assertFalse(self.music.status()['available'])
    def test_tampered_stored_clip_fails_closed(self):
        m=self.music.get('phrase_original_pulse_v1');Path(m['path']).write_bytes(b'tamper')
        with self.assertRaises(ValueError):self.music.get(m['id'])
if __name__=='__main__':unittest.main()
