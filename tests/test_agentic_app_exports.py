"""Export boundary fixtures use synthetic arrays, never pretend to execute physics."""
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from pulsemorph.agentic.app import exports as e

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.jobs=Path(self.temp.name)/'jobs';self.id='fixture_job';self.d=self.jobs/self.id/'trials/trial_00';self.d.mkdir(parents=True)
        self.params=dict(zip(e.PARAMETERS,[1,1,1,1,0]));self.source=e.PINS['wave'];self.audio='C:\\private-user\\original-audio.wav'
        metadata={'schema_version':'pulsemorph.motion_reference.v1','reference_id':self.id+'__trial_0:common','producer':{'kind':'agentic','id':'fixture','version':'1'},'body':{'profile_id':'g1','profile_sha256':e.BODY_HASH,'joint_names':list(e.JOINT_NAMES),'anchor_body_name':'torso_link'},'controller':{'profile_id':'g1-bones-deploy','profile_sha256':e.CONTROLLER_HASH,'control_dt_s':.02},'frame':{'id':'reference_world','up_axis':'z','handedness':'right','quaternion_order':'xyzw','alignment':'recorded_reference_world'},'context':{'decision_id':'fixture','observation_id':'fixture','task_id':self.id,'runtime_session_id':'fixture','clock_epoch':'source-zero','decision_time_s':0},'validity':{'start_state_id':'fixture','start_time_s':0,'end_time_s':4.14,'start_mode':'reset_from_source','end_behavior':'offline_trial_end'},'observation':{'mode':'offline_file','observed_through_time_s':4,'audio_observed_through_sample':192000,'video_observed_through_time_s':None},'provenance':{'sources':[{'path':'vendor/output_wave.motion','sha256':self.source},{'path':self.audio,'sha256':'a'*64}],'notes':['Synthetic export fixture, not physics evidence.']}}
        common=self.d/'common_reference';common.mkdir();buf=BytesIO();np.savez_compressed(buf,time_offsets_s=np.arange(208)*.02,joint_position_rad=np.zeros((208,29)),joint_velocity_rad_s=np.zeros((208,29)),anchor_orientation_xyzw=np.tile([0,0,0,1],(208,1)));(common/'arrays.npz').write_bytes(buf.getvalue());(common/'manifest.json').write_text(json.dumps({'metadata':metadata,'arrays':{'path':'arrays.npz','sha256':e._sha(buf.getvalue())}}))
        for name,value in {'qpos.npy':np.zeros((201,36)),'qvel.npy':np.zeros((201,35)),'state_times.npy':np.arange(201)*.02,'applied_joint_targets.npy':np.ones((200,29))}.items():np.save(self.d/name,value)
        proposal={'job_id':self.id,'trial_index':0,'source_id':'wave','source_motion':{'sha256':self.source},'parameters':self.params,'control_tool':'global_clock_v1'};(self.d/'proposal.json').write_text(json.dumps(proposal));(self.d/'extra_recorded.txt').write_text('fixture')
        self.manifest={'status':'success','error':None,'job_id':self.id,'trial_index':0,'common_contract_validated':True,'strict_g1_admission_passed':True,'preexecution_hashes_unchanged':True,'actual_control_steps':200,'actual_physics_steps':4000,'online_onnx_call_count':200,'control_dt':.02,'physics_dt':.001,'source_id':'wave','motion_sha256':self.source,'source_motion':{'sha256':self.source},'proposal_sha256':e._sha((self.d/'proposal.json').read_bytes()),'candidate_id':self.id+'__trial_0','body_profile_sha256':e.BODY_HASH,'controller_profile_sha256':e.CONTROLLER_HASH,'onnx_sha256':'b'*64}
        self.refresh()
        selected={'index':0,'paths':{'directory':str(self.d)},'parameters':self.params,'tool_id':'global_clock_v1','metrics':{'integrity_passed':True,'integrity_errors':[],'features':[1,1,1,1,3],'physical_gate_passed':True,'intent_passed':True}}
        self.job={'status':'complete','outcome':'target_met','request':{'source_id':'wave','instruction':'fixture instruction'},'selected':selected,'plan':{'supported':True},'targets':[1,1,1,1,3],'tolerances':[.08]*5,'music':{'sha256':'a'*64,'provenance':{'source_sha256':'c'*64,'start_s':2,'end_s':6}}}
        self.manager=SimpleNamespace(jobs_dir=self.jobs,get=lambda _:copy.deepcopy(self.job))
    def refresh(self):
        self.manifest['artifact_hashes']={str(p.relative_to(self.d)):e._sha(p.read_bytes()) for p in self.d.rglob('*') if p.is_file() and p.name!='manifest.json'}
        # The common manifest is part of the recorded artifact inventory.
        self.manifest['artifact_hashes']['common_reference/manifest.json']=e._sha((self.d/'common_reference/manifest.json').read_bytes())
        (self.d/'manifest.json').write_text(json.dumps(self.manifest))
    def test_portable_identity_determinism_and_array_bytes(self):
        original=(self.d/'common_reference/manifest.json').read_bytes();data=e.build_result_bundle(self.manager,self.id);self.assertEqual(data,e.build_result_bundle(self.manager,self.id))
        with zipfile.ZipFile(BytesIO(data)) as z:
            self.assertEqual(set(z.namelist()),{'README.md','result.json','reference/manifest.json','reference/arrays.npz','realized/qpos.npy','realized/qvel.npy','realized/state_times.npy','realized/applied_joint_targets.npy'})
            for name in ['qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy']:self.assertEqual(z.read('realized/'+name),(self.d/name).read_bytes())
            self.assertEqual(z.read('reference/arrays.npz'),(self.d/'common_reference/arrays.npz').read_bytes())
            self.assertNotIn(b'private-user',b''.join(z.read(n) for n in z.namelist()));result=json.loads(z.read('result.json'));self.assertEqual(result['reference_manifest_transformation']['original_sha256'],e._sha(original));self.assertEqual(len(result['reference_manifest_transformation']['absolute_path_redactions']),1)
        self.assertEqual(original,(self.d/'common_reference/manifest.json').read_bytes())
    def test_tamper_any_recorded_file(self):
        (self.d/'extra_recorded.txt').write_text('tampered')
        with self.assertRaisesRegex(e.ResultExportError,'hash mismatch'):e.build_result_bundle(self.manager,self.id)
    def test_reject_unsupported_unfinished_and_integrity_failure(self):
        for field,value in [('status','running'),('outcome','unsupported')]:
            old=self.job[field];self.job[field]=value
            with self.assertRaises(e.ResultExportError):e.build_result_bundle(self.manager,self.id)
            self.job[field]=old
        self.job['selected']['metrics']['integrity_passed']=False
        with self.assertRaises(e.ResultExportError):e.build_result_bundle(self.manager,self.id)
    def test_path_escape_and_symlink(self):
        self.manifest['artifact_hashes']['../outside']='0'*64;(self.d/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(e.ResultExportError):e.build_result_bundle(self.manager,self.id)
    def test_selected_directory_must_match_index(self):
        self.job['selected']['paths']['directory']=str(self.d.parent/'baseline')
        with self.assertRaisesRegex(e.ResultExportError,'directory mismatch'):e.build_result_bundle(self.manager,self.id)
    def test_baseline_selected_supported(self):
        target=self.d.parent/'baseline';self.d.rename(target);self.d=target;self.job['selected']['index']=-1;self.job['selected']['paths']['directory']=str(target)
        self.manifest['trial_index']=-1;self.manifest['candidate_id']=self.id+'__trial_-1'
        p=json.loads((target/'proposal.json').read_text());p['trial_index']=-1;(target/'proposal.json').write_text(json.dumps(p));self.manifest['proposal_sha256']=e._sha((target/'proposal.json').read_bytes())
        p=json.loads((target/'common_reference/manifest.json').read_text());p['metadata']['reference_id']=self.id+'__trial_-1:common';(target/'common_reference/manifest.json').write_text(json.dumps(p));self.refresh();self.assertTrue(e.build_result_bundle(self.manager,self.id))
    def test_mutation_between_validation_and_copy_rejected(self):
        original=e._read;counts={}
        def changing(path,*args):
            counts[path]=counts.get(path,0)+1
            data=original(path,*args)
            if path.name=='qpos.npy' and counts[path]==2:return data+b'race'
            return data
        with patch.object(e,'_read',changing),self.assertRaisesRegex(e.ResultExportError,'changed during export'):e.build_result_bundle(self.manager,self.id)
    def test_wrong_layout_and_common_binding(self):
        np.save(self.d/'qvel.npy',np.zeros((200,35)));self.refresh()
        with self.assertRaisesRegex(e.ResultExportError,'array layout'):e.build_result_bundle(self.manager,self.id)
    def test_common_array_binding_rejects_rehashed_wrong_digest(self):
        p=self.d/'common_reference/manifest.json';m=json.loads(p.read_text());m['arrays']['sha256']='0'*64;p.write_text(json.dumps(m));self.refresh()
        with self.assertRaisesRegex(e.ResultExportError,'archive binding'):e.build_result_bundle(self.manager,self.id)
    def test_proposal_binding_and_parameter_bounds(self):
        self.job['selected']['parameters']=dict(self.params,left_gain_a=1.4)
        with self.assertRaisesRegex(e.ResultExportError,'proposal binding'):e.build_result_bundle(self.manager,self.id)
    def test_link_boundary_rejected(self):
        original=e._linked
        with patch.object(e,'_linked',lambda p: p.name=='qpos.npy' or original(p)),self.assertRaisesRegex(e.ResultExportError,'Linked'):e.build_result_bundle(self.manager,self.id)
    def test_absolute_path_outside_redaction_scope_rejected(self):
        p=self.d/'common_reference/manifest.json';m=json.loads(p.read_text());m['metadata']['provenance']['notes'].append('C:\\private\\file');p.write_text(json.dumps(m));self.refresh()
        with self.assertRaisesRegex(e.ResultExportError,'absolute path'):e.build_result_bundle(self.manager,self.id)
    def test_instruction_and_plan_slashes_preserved_only_absolute_paths_redacted(self):
        ordinary='sol/sağ 110/90; https://example.org/ref relative/path ./local/file'
        private=ordinary+' C:\\private\\file.wav /home/person/audio.wav \\\\server\\share\\audio.wav'
        self.job['request']['instruction']=private
        self.job['plan']={'interpretation':ordinary,'detail':private}
        data=e.build_result_bundle(self.manager,self.id)
        with zipfile.ZipFile(BytesIO(data)) as z:
            result=json.loads(z.read('result.json'))
            self.assertEqual(result['plan']['interpretation'],ordinary)
            expected=ordinary+' [local-path omitted] [local-path omitted] [local-path omitted]'
            self.assertEqual(result['instruction'],expected)
            self.assertEqual(result['plan']['detail'],expected)
            self.assertNotIn(b'private',z.read('result.json'))
            self.assertNotIn(b'/home/person',z.read('result.json'))

    def test_budget_rejected(self):
        with patch.object(e,'MAX_BUNDLE_BYTES',100),self.assertRaises(e.ResultExportError):e.build_result_bundle(self.manager,self.id)

if __name__=='__main__':unittest.main()
