import json,tempfile,unittest,hashlib
import http.client,threading
from pathlib import Path
from unittest.mock import patch
from pulsemorph.agentic.embodied.app import config_v5 as c
from pulsemorph.agentic.embodied.app.profile_v5 import verify_vendor

class PortableTests(unittest.TestCase):
    def test_identity_changed_after_config_load_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            p,v,i=self.setup_config(d)
            with patch.object(c,'IDENTITY_SHA256',c.digest(i)):
                cfg=c.load_config(p)
                i.write_text(json.dumps({'schema':'pulsemorph.body_asset_identity.v1','files':[]}))
                with self.assertRaisesRegex(ValueError,'identity changed'):verify_vendor(cfg)
    def setup_config(self,d):
        d=Path(d);identity=d/'identity.json';identity.write_text('{}')
        value={'schema':c.SCHEMA,'runtime_python':'local-python','vendor_root':'vendor','profile_dir':'profile','artifact_root':'jobs','asset_identity':'identity.json','port':8777}
        p=d/'config.json';p.write_text(json.dumps(value));return p,value,identity
    def test_relative_paths_and_alternate_python_accepted(self):
        with tempfile.TemporaryDirectory()as d:
            p,v,i=self.setup_config(d)
            with patch.object(c,'IDENTITY_SHA256',c.digest(i)):
                actual=c.load_config(p)
            self.assertEqual(actual['runtime_python'],Path(d)/'local-python')
            self.assertEqual(actual['vendor_root'],Path(d)/'vendor')
    def test_unknown_bool_and_manifest_tamper_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            p,v,i=self.setup_config(d)
            for delta in ({'port':True},{'api_key':'forbidden'},{'schema':'other'}):
                p.write_text(json.dumps({**v,**delta}))
                with patch.object(c,'IDENTITY_SHA256',c.digest(i)),self.assertRaises(ValueError):c.load_config(p)
            p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):c.load_config(p)
    def test_missing_assets_clear_unavailable_no_native(self):
        with tempfile.TemporaryDirectory()as d:
            p,v,i=self.setup_config(d)
            with patch.object(c,'IDENTITY_SHA256',c.digest(i)):cfg=c.load_config(p)
            result=c.metadata_status(cfg)
            self.assertFalse(result['available']);self.assertFalse(result['physical_admission'])
    def test_asset_escape_and_changed_bytes(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);vendor=root/'vendor';vendor.mkdir();asset=vendor/'own';asset.write_bytes(b'actual')
            identity=root/'identity.json';row={'path':'own','bytes':6,'sha256':hashlib.sha256(b'actual').hexdigest()}
            identity.write_text(json.dumps({'schema':'pulsemorph.body_asset_identity.v1','files':[row]}));cfg={'asset_identity':identity,'vendor_root':vendor};self.enterContext(patch.object(c,'IDENTITY_SHA256',c.digest(identity)))
            verify_vendor(cfg);asset.write_bytes(b'tamper')
            with self.assertRaises(ValueError):verify_vendor(cfg)
            row['path']='../outside';identity.write_text(json.dumps({'schema':'pulsemorph.body_asset_identity.v1','files':[row]}))
            with self.assertRaises(ValueError):verify_vendor(cfg)

    def test_missing_profile_ui_and_write_denial(self):
        from pulsemorph.agentic.embodied.app.server_v5 import Server
        with tempfile.TemporaryDirectory()as d:
            p,v,i=self.setup_config(d)
            with patch.object(c,'IDENTITY_SHA256',c.digest(i)):cfg=c.load_config(p)
            cfg['port']=0;server=Server(cfg);thread=threading.Thread(target=server.serve_forever);thread.start()
            try:
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
                conn.request('GET','/api/status');response=conn.getresponse();data=json.loads(response.read())
                self.assertEqual(response.status,200);self.assertFalse(data['ready']);self.assertIsNone(data['example'])
                conn.request('POST','/api/jobs',body='{}',headers={'Content-Type':'application/json','X-PulseMorph-Token':data['csrf_token']})
                response=conn.getresponse();self.assertEqual(response.status,503);response.read()
                conn.request('GET','/api/status',headers={'Host':'foreign.invalid'})
                response=conn.getresponse();self.assertEqual(response.status,403);response.read();conn.close()
            finally:server.shutdown();server.server_close();thread.join()

    def test_owned_profile_has_no_clip_or_golden_and_detects_tamper(self):
        from pulsemorph.agentic.embodied.app.profile_v5 import build_profile,verify_profile
        import yaml
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);vendor=root/'vendor';names=['left_hip_pitch_joint','left_knee_joint','left_ankle_pitch_joint','left_elbow_joint']+['joint_'+str(i)for i in range(25)]
            controller=vendor/'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml';controller.parent.mkdir(parents=True);controller.write_text(yaml.safe_dump({'joint_names':names}))
            xml=vendor/'protomotions/data/assets/mjcf/g1_holo_compat.xml';xml.parent.mkdir(parents=True);xml.write_text('<mujoco>'+''.join('<joint name="'+n+'" range="-2 2"/>'for n in names)+'</mujoco>')
            identity=root/'identity.json';identity.write_text(json.dumps({'schema':'pulsemorph.body_asset_identity.v1','files':[]}));cfg={'asset_identity':identity,'vendor_root':vendor,'profile_dir':root/'profile'};self.enterContext(patch.object(c,'IDENTITY_SHA256',c.digest(identity)))
            result=build_profile(cfg);self.assertFalse(result['physical_admission']);self.assertEqual(set(p.name for p in cfg['profile_dir'].iterdir()),{'standing.npz','manifest.json'})
            verify_profile(cfg);(cfg['profile_dir']/'standing.npz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):verify_profile(cfg)

    def test_rehashed_wrong_recipe_rejected(self):
        # Reuse a synthetic public profile, then alter both q arrays and its hash.
        from pulsemorph.agentic.embodied.app.profile_v5 import build_profile,verify_profile
        import yaml,numpy as np
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);vendor=root/'vendor';names=['joint_'+str(i)for i in range(29)]
            controller=vendor/'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml';controller.parent.mkdir(parents=True);controller.write_text(yaml.safe_dump({'joint_names':names}))
            xml=vendor/'protomotions/data/assets/mjcf/g1_holo_compat.xml';xml.parent.mkdir(parents=True);xml.write_text('<mujoco>'+''.join('<joint name="'+n+'" range="-2 2"/>'for n in names)+'</mujoco>')
            identity=root/'identity.json';identity.write_text(json.dumps({'schema':'pulsemorph.body_asset_identity.v1','files':[]}));cfg={'asset_identity':identity,'vendor_root':vendor,'profile_dir':root/'profile'};self.enterContext(patch.object(c,'IDENTITY_SHA256',c.digest(identity)));build_profile(cfg)
            archive=cfg['profile_dir']/'standing.npz'
            with np.load(archive,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
            arrays['reference_q'][0]=.1;arrays['qpos'][7]=.1;np.savez_compressed(archive,**arrays)
            p=cfg['profile_dir']/'manifest.json';v=json.loads(p.read_bytes());v['array_sha256']=c.digest(archive);p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'assignment'):verify_profile(cfg)

    def test_offline_materialization_exclusive_and_source_preserved(self):
        from pulsemorph.agentic.embodied.app.acquisition_v5 import materialize,canonical_asset
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source=root/'source';source.mkdir();raw=b'a\nb\n';canonical=raw.replace(b'\n',b'\r\n');rows=[]
            for index in range(70):
                name='assets/'+str(index)+'.txt';p=source/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
                rows.append({'path':name,'bytes':len(canonical),'sha256':hashlib.sha256(canonical).hexdigest(),'tracked_blob_sha256':hashlib.sha256(raw).hexdigest(),'materialization':'lf_to_crlf'})
            identity=root/'identity.json';identity.write_text(json.dumps({'schema':'pulsemorph.body_asset_identity.v1','upstream':{'commit':'fixed'},'files':rows}))
            def run(args,**kwargs):return SimpleNamespace(returncode=0,stdout=b'fixed\n'if args[-2:]==['rev-parse','HEAD']else raw)
            with patch.object(c,'IDENTITY_SHA256',c.digest(identity)),patch('pulsemorph.agentic.embodied.app.acquisition_v5.subprocess.run',side_effect=run):
                result=materialize(source,root/'dest',identity);self.assertEqual(len(result['files']),70)
                self.assertEqual((source/rows[0]['path']).read_bytes(),raw);self.assertEqual((root/'dest'/rows[0]['path']).read_bytes(),canonical)
                with self.assertRaises(ValueError):materialize(source,root/'dest',identity)
            with self.assertRaises(ValueError):canonical_asset(b'unknown',rows[0])
