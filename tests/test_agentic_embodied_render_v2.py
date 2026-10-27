"""Actual commit mapping only. Synthetic journals, zero FK/render/model calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
path=Path(__file__).resolve().parents[1]/'experiments/agentic/embodied/render_episode_v2.py'
spec=importlib.util.spec_from_file_location('actual_preview_render_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ActualRenderTests(unittest.TestCase):
    def test_boundary_is_completed_previous_command(self):
        a=[{'start_sim_time_s':0,'planned_control_steps':30},{'start_sim_time_s':.6,'planned_control_steps':30}]
        self.assertIs(m.active_action(a,0),a[0]);self.assertIs(m.active_action(a,.6),a[0]);self.assertIs(m.active_action(a,.62),a[1]);self.assertIs(m.active_action(a,1.2),a[1])
    def fixture(self,root):
        action={'action_id':'actual-candidate','joint':.1}
        runtime={'request':action,'status':'completed','before_revision':2,'planned_control_steps':30,'start_sim_time_s':2}
        rows={
          'commit_00/pending.json':{'commit_id':'commit0','preview_id':'p0','observation_sha256':'obs0'},
          'commit_00/returned.json':{'commit_id':'commit0','preview_id':'p0','action_id':'actual-candidate','hypothetical_only':False,'replay_no_new_work':False,'preview_actual_byte_equal':True,'before_revision':2,'control_steps':30},
          'preview_00/pending.json':{'action':action},
          'model/call_01/validated.json':{'proposal':{'decision':'commit','preview_id':'p0'},'observation_sha256':'obs0','receipt':{'live':True,'completion_unconfirmed':False}}}
        for name,data in rows.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
        return {'actions':[runtime]},rows
    def test_maps_actual_commit_and_real_decision_call(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);episode,rows=self.fixture(root)
            self.assertEqual(m.actual_commits(root,episode),{'actual-candidate':{'commit_index':1,'model_call':2,'preview_id':'p0'}})
            (root/'preview_99').mkdir();(root/'preview_99/returned.json').write_text(json.dumps({'hypothetical_only':True,'qpos':'never rendered'}))
            self.assertEqual(len(m.actual_commits(root,episode)),1)
    def test_hypothetical_mismatch_and_unconfirmed_model_rejected(self):
        for kind in ('hypothetical','parity','command','receipt'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);episode,rows=self.fixture(root)
                if kind=='hypothetical':rows['commit_00/returned.json']['hypothetical_only']=True
                elif kind=='parity':rows['commit_00/returned.json']['preview_actual_byte_equal']=False
                elif kind=='command':episode['actions'][0]['request']={'action_id':'actual-candidate','joint':.2}
                else:rows['model/call_01/validated.json']['receipt']['completion_unconfirmed']=True
                for name,data in rows.items():(root/name).write_text(json.dumps(data))
                with self.assertRaises(ValueError):m.actual_commits(root,episode)
    def test_exact_frame_budget_and_no_extra_fk(self):
        import ast
        self.assertEqual(len(m.frames_for(np.arange(131)*.02)),65)
        tree=ast.parse(path.read_text());attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(attrs.count('mj_forward'),1);self.assertNotIn('mj_step',attrs);self.assertNotIn('InferenceSession',attrs)
if __name__=='__main__':unittest.main()
