"""Injected pure executor-state tests, no MuJoCo/ORT construction or FK/steps."""
import copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_agentic_embodied_fork as fixtures
from pulsemorph.agentic.performance.runtime_v1 import PerformanceEpisode,PerformancePreviews,full_snapshot,make_reference,curve_hash
from pulsemorph.agentic.performance.choreography_v1 import CompiledPerformance
from pulsemorph.agentic.embodied.runtime_v1 import fingerprint,ObservationMismatch,ActionRejected
from pulsemorph.agentic.embodied.fork_v1 import PreviewIntegrityError

class Backend(fixtures.FixtureBackend):
    def __init__(self):
        super().__init__();self.authority={'schema':'pulsemorph.performance_execution_authority.v1','max_control_steps':100,'session_id':'fixture'};self.profile={'body_sha256':'a'*64,'yaml_sha256':'b'*64};self.anchor_target=np.array([0.,0.,0.,1.]);self.source_pins={'fixture':'c'*64};self.performance_contract={'mode':'fixture only'};self.mj=SimpleNamespace(mj_name2id=lambda *args:0,mjtObj=SimpleNamespace(mjOBJ_BODY=0));self.data.ncon=0;self.data.xpos=np.array([[0.,0.,.8]]);self.data.subtree_com=np.array([[0.,0.,.8]])
    def _verify(self):pass
def curve(q0):
    times=np.arange(11)*.02;q=np.tile(q0,(11,1));q[1,0]+=.005;q[2:,0]+=.01;dq=np.zeros_like(q);dq[1,0]=.5
    return CompiledPerformance(.04,times,q,dq,{'admitted':True,'fixture_only':True},'d'*64)
def request(p,c,name='move'):
    return {'action_id':name,'expected_revision':p.revision,'observation_sha256':fingerprint(p.observe()),'program_sha256':c.program_sha256}

class Tests(unittest.TestCase):
    def parent(self):return PerformanceEpisode(Backend(),'fixture')
    def test_full_future_reference_and_all_aux_indices_preserved(self):
        p=self.parent();c=curve(p.reference_q);r=p.execute_curve(request(p,c),c);self.assertEqual(r['steps'],40);self.assertEqual(p.control_index,2);self.assertEqual(len(p.kinematics),3);self.assertEqual(len(p.reference_samples),3);self.assertEqual([x['control_index'] for x in p.references],[0,1]);self.assertEqual([x['sim_time_s'] for x in p.reference_samples],[0.,.02,.04]);self.assertEqual(p.common_references[0].time_offsets_s[-1],.2)
        self.assertEqual(p.references[0]['future_q'][3][0],.01);self.assertEqual(p.references[1]['future_q'][3][0],.01);self.assertEqual(p.backend.initialization_count,1)
    def test_stale_and_initial_reference_reject_before_control(self):
        p=self.parent();c=curve(p.reference_q);bad=request(p,c);bad['expected_revision']=True
        with self.assertRaises(ObservationMismatch):p.execute_curve(bad,c)
        changed=curve(p.reference_q+.1)
        with self.assertRaises(ActionRejected):p.execute_curve(request(p,changed),changed)
        self.assertEqual(p.backend.control_index,0)
    def test_preview_parent_aux_unchanged_and_commit_byte_parity(self):
        p=self.parent();c=curve(p.reference_q);pool=PerformancePreviews(p);before=full_snapshot(p);preview=pool.preview(request(p,c),c);self.assertEqual(full_snapshot(p),before);self.assertEqual(len(p.kinematics),1);result=pool.commit({'preview_id':preview['preview_id'],'expected_revision':0,'observation_sha256':fingerprint(p.observe()),'commit_id':'commit'});self.assertTrue(result['preview_actual_byte_equal']);self.assertEqual(len(p.kinematics),3);self.assertEqual(p.backend.initialization_count,1)
    def test_changed_preview_curve_rejected_before_actual_work(self):
        p=self.parent();c=curve(p.reference_q);pool=PerformancePreviews(p);preview=pool.preview(request(p,c),c);pool.entries[preview['preview_id']]['curve'].q[-1,0]+=.01
        with self.assertRaises(PreviewIntegrityError):pool.commit({'preview_id':preview['preview_id'],'expected_revision':0,'observation_sha256':fingerprint(p.observe()),'commit_id':'commit'})
        self.assertEqual(p.backend.control_index,0)
    def test_snapshot_binds_common_reference_inventory(self):
        p=self.parent();c=curve(p.reference_q);before=full_snapshot(p);p.common_references.append(make_reference(p,c,'owned_reference',0.,fingerprint(p.observe())));self.assertNotEqual(full_snapshot(p),before)
    def test_bool_commit_revision_rejected_before_actual_work(self):
        p=self.parent();first=curve(p.reference_q);p.execute_curve(request(p,first,'prelude'),first);c=curve(p.reference_q);pool=PerformancePreviews(p);preview=pool.preview(request(p,c,'next'),c);count=p.backend.control_index
        with self.assertRaises(PreviewIntegrityError):pool.commit({'preview_id':preview['preview_id'],'expected_revision':True,'observation_sha256':fingerprint(p.observe()),'commit_id':'commit'})
        self.assertEqual(p.backend.control_index,count)
    def test_changed_hypothetical_trace_rejected_before_actual_work(self):
        p=self.parent();c=curve(p.reference_q);pool=PerformancePreviews(p);preview=pool.preview(request(p,c),c);pool.entries[preview['preview_id']]['branch'].backend.data.qacc_warmstart[0]=4.
        with self.assertRaises(PreviewIntegrityError):pool.commit({'preview_id':preview['preview_id'],'expected_revision':0,'observation_sha256':fingerprint(p.observe()),'commit_id':'commit'})
        self.assertEqual(p.backend.control_index,0)

if __name__=='__main__':unittest.main()
