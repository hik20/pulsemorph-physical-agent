"""Saved-proof export boundaries; synthetic buffers only, no physical execution."""
from pathlib import Path
import copy
import io
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase import exports_v1 as e

class PhraseExportTests(unittest.TestCase):
    def test_prose_preserves_instruction_ratio_url_relative(self):
        text='sol/sağ 110/90; /solsağ110/90 https://example.org/ref relative/motion.npy'
        self.assertEqual(e._prose(text,set()),text)
        out=e._prose(text+' C:/private/audio.wav /home/private/audio.wav',set())
        self.assertTrue(out.startswith(text));self.assertNotIn('private',out)

    def test_manifest_only_absolute_source_paths_transform(self):
        original={'metadata':{'provenance':{'sources':[{'path':'C:/private/music.wav','sha256':'a'*64},{'path':'vendor/wave.motion','sha256':'b'*64}]}}}
        portable,changes=e._portable_manifest(original,set())
        self.assertEqual(original['metadata']['provenance']['sources'][0]['path'],'C:/private/music.wav')
        self.assertEqual(portable['metadata']['provenance']['sources'][0]['path'],'external_inputs/'+'a'*64+'/input.wav')
        self.assertEqual(len(changes),1);self.assertNotIn('private',str(portable)+str(changes))

    def test_manifest_rejects_other_absolute_metadata(self):
        data={'metadata':{'provenance':{'sources':[]},'hidden':'C:/private/key'}}
        with self.assertRaises(e.PhraseResultExportError):e._portable_manifest(data,set())

    def test_final_result_rejects_nested_private_paths(self):
        for text in ('C:/private/file','/home/private/file','http://127.0.0.1:8765/local'):
            with self.subTest(text=text),self.assertRaises(e.PhraseResultExportError):e._result_paths_safe({'planning_music':{'extra':text}},set())
        e._result_paths_safe({'field':'/metadata/provenance/sources/2/path','text':'sol/sağ 110/90 https://example.org/ref'},set())

    def test_music_provenance_is_finite_whitelist(self):
        music={'id':'phrase_original_pulse_v1','provenance':{'kind':'procedural_owned','source_bytes_retained':False,'generator_sha256':'a'*64,'extra':'C:/private/file'}}
        clean=e._music_provenance(music)
        self.assertEqual(set(clean),{'kind','source_bytes_retained','generator_sha256'})
        self.assertNotIn('private',str(clean))

    def test_uploaded_music_identity_crop_bound(self):
        p={'kind':'local_uploaded_PCM16','source_sha256':'a'*64,'source_bytes':800000,'source_bytes_retained':False,'original_frame_count':480000,'original_sample_rate':48000,'original_channels':2,'selected_frame_count':384000,'start_frame':48000,'start_s':1.,'end_s':9.,'transform':'pcm16_downmix_resample_poly_48k_8s_v1','derived_wav_sha256':'b'*64,'derived_frame_count':384000}
        import json
        identity={'source_sha256':p['source_sha256'],'start_frame':48000,'duration_s':8,'transform':p['transform']}
        m={'id':'phrase_upload_'+e.sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())[:24],'sha256':'b'*64,'provenance':p}
        self.assertEqual(e._music_provenance(m),p)
        p['start_frame']+=1
        with self.assertRaises(e.PhraseResultExportError):e._music_provenance(m)

    def test_reader_hash_then_race_and_byte_change(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'data';p.write_bytes(b'original');r=e._Reader([directory])
            self.assertEqual(r.data(p,e.sha(b'original')),b'original')
            p.write_bytes(b'tampered')
            with self.assertRaises(e.PhraseResultExportError):r.hash(p)

    def test_reader_bound_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'data';p.write_bytes(b'1234')
            with patch.object(e,'MAX_AUDIT_BYTES',3),self.assertRaises(e.PhraseResultExportError):e._Reader([directory]).data(p)
            for value in ('../escape','C:/private','/absolute'):
                with self.subTest(value=value),self.assertRaises(e.PhraseResultExportError):e.relative(Path(directory),value)

    def test_arrays_exact_shape_finite_and_no_pickle(self):
        def data(a):
            stream=io.BytesIO();np.save(stream,a,allow_pickle=False);return stream.getvalue()
        e._npy(data(np.zeros((401,36))), (401,36))
        for a in (np.zeros((400,36)),np.zeros((401,36),dtype=np.float32),np.full((401,36),np.nan)):
            with self.assertRaises(e.PhraseResultExportError):e._npy(data(a),(401,36))

    def test_unsupported_failed_unmeasured_and_list_fail_closed(self):
        class Manager:
            def get(self,_):return self.value
        manager=Manager()
        for value in (None,[],{'id':'job','source_id':e.SOURCE_ID,'status':'failed'}, {'id':'job','source_id':e.SOURCE_ID,'status':'complete','outcome':'unsupported'}, {'id':'job','source_id':e.SOURCE_ID,'status':'complete','outcome':'partial','selected':None}):
            manager.value=value
            with self.subTest(value=value),self.assertRaises(e.PhraseResultExportError):e.build_phrase_result_bundle(manager,'job')

if __name__=='__main__':unittest.main()
