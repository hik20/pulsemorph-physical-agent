"""Regression for the configured-profile path outside the original checkout."""
import ast,copy,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_agentic_embodied_fork import FixtureBackend
from pulsemorph.agentic.embodied.runtime_v2 import PersistentEpisode as OldEpisode,fingerprint
from pulsemorph.agentic.embodied.runtime_portable_v5 import PersistentEpisode
from pulsemorph.agentic.embodied.g1_portable_v5 import assemble_reference_inputs
from pulsemorph.agentic.embodied.reference_portable_v5 import make_reference
from pulsemorph.agentic.embodied.runtime_v2 import JointSegment

def backend():
    b=FixtureBackend();b.anchor_target=np.array([0.,0.,0.,1.]);b.source_pins={}
    b.profile={'body_sha256':'a'*64,'yaml_sha256':'b'*64,'joint_names':tuple(b.joint_names),'anchor_body_name':'torso_link','control_dt_s':.02}
    return b
def action(episode):
    return {'action_id':'configured','expected_revision':0,'observation_sha256':fingerprint(episode.observe()),'tool':'move_joints','arguments':{'duration_s':.4,'joint_targets_rad':{'joint_0':.1}}}
class Tests(unittest.TestCase):
    def test_original_default_missing_but_configured_profile_executes_without_fallback(self):
        old=OldEpisode(backend(),'old',max_control_steps=20)
        new=PersistentEpisode(backend(),'new',max_control_steps=20)
        with patch('pulsemorph.motion.g1.profile_from_yaml',side_effect=FileNotFoundError('Missing default checkout')) as default:
            with self.assertRaises(FileNotFoundError):old.execute(action(old))
            self.assertEqual(old.backend.control_index,0)
            result=new.execute(action(new))
            self.assertEqual(default.call_count,1)
        self.assertEqual(result['control_steps'],20);self.assertEqual(result['sampled_observations_count'],20)
        self.assertEqual(new.reference_q[0],.1);self.assertIsInstance(new,OldEpisode)
    def test_explicit_profile_identity_mismatch_rejects(self):
        b=backend();q=np.zeros(29);segment=JointSegment(q,q,q,.4,b.joint_ranges)
        ref=make_reference(b,segment,'fixture','ref',0,0.,'c'*64,q,q)
        good=assemble_reference_inputs(ref,0.,profile=b.profile)
        self.assertEqual(good['mimic_future_dof_pos'].shape,(1,4,29))
        with self.assertRaises(ValueError):assemble_reference_inputs(ref,0.,profile={**b.profile,'body_sha256':'d'*64})
        with self.assertRaises(TypeError):assemble_reference_inputs(ref,0.)
    def test_control_state_machine_diff_is_only_reference_import(self):
        import pulsemorph.agentic.embodied.runtime_v2 as old
        import pulsemorph.agentic.embodied.runtime_portable_v5 as new
        def method(module):
            tree=ast.parse(Path(module.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='PersistentEpisode')
            return next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
        a,b=method(old),method(new)
        for n in ast.walk(a):
            if isinstance(n,ast.ImportFrom)and n.module=='reference_v1':n.module='reference_portable_v5'
        self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
