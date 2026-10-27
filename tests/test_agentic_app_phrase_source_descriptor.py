"""Source descriptors are numerical data, never physical execution claims."""
from pathlib import Path
import sys,copy,unittest
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.app.phrase import source_descriptor_v1 as d

def reference():
    t=np.arange(208,dtype=float)*.02;q=np.tile(t[:,None],(1,29))
    metadata={'body':{'joint_names':list(d.JOINT_NAMES),'profile_id':'g1','profile_sha256':d.BODY_SHA},'controller':{'profile_sha256':d.CONTROLLER_SHA},'provenance':{'sources':[{'sha256':'a'*64}]}}
    return SimpleNamespace(time_offsets_s=t,joint_position_rad=q,metadata=metadata,validate=lambda:None)
class Assets:
    def __init__(self):
        self.manifest={'schema':'pulsemorph.private_phrase_assets.v1','files':{source+'/'+name:digest for source,files in d.REFERENCE_HASHES.items() for name,digest in files.items()}};self.refs={source:reference() for source in ('wave','high5')};self.calls=0
    def verify(self):self.calls+=1
    def references(self):return self.refs
class DescriptorTests(unittest.TestCase):
    def test_componentwise_means_rms_window_mapping(self):
        a=Assets();result=d.describe_sources(a);self.assertEqual(a.calls,2)
        self.assertEqual(result['phases'][1]['global_window_s'],[5.2,7.2]);self.assertEqual(result['phases'][1]['source_window_s'],[1.2,3.2])
        left=result['phases'][0]['arms']['left'];t=np.arange(60,161)*.02
        self.assertAlmostEqual(left['q0_displacement_rms_rad'],np.sqrt(np.mean(t*t)))
        self.assertAlmostEqual(left['centered_temporal_rms_rad'],np.sqrt(np.mean((t-t.mean())**2)))
        self.assertAlmostEqual(left['mean_pose_offset_rms_rad'],t.mean());self.assertEqual(left['per_joint_peak_to_peak_rad'],[2.]*7)
        self.assertNotIn('inactive',left)
    def test_return_is_owned_and_inputs_unchanged(self):
        a=Assets();before=a.refs['wave'].joint_position_rad.copy();result=d.describe_sources(a);result['phases'][0]['arms']['left']['joint_names'][0]='mutated';np.testing.assert_array_equal(before,a.refs['wave'].joint_position_rad);self.assertEqual(d.describe_sources(a)['phases'][0]['arms']['left']['joint_names'][0],d.JOINT_NAMES[15])
    def test_nonfinite_and_wrong_clock_fail(self):
        for mode in ('nan','clock','shape'):
            a=Assets()
            if mode=='nan':a.refs['wave'].joint_position_rad[100,0]=np.nan
            elif mode=='clock':a.refs['wave'].time_offsets_s[100]+=.001
            else:a.refs['wave'].joint_position_rad=np.zeros((208,28))
            with self.subTest(mode=mode),self.assertRaises(ValueError):d.describe_sources(a)
    def test_malformed_profile_lineage_or_source_map_fails(self):
        for mode in ('profile','lineage','map','pin'):
            a=Assets()
            if mode=='profile':a.refs['wave'].metadata['body']['profile_id']='other'
            elif mode=='lineage':a.refs['wave'].metadata['provenance']['sources'][0]['sha256']='not a hash'
            elif mode=='map':a.refs.pop('high5')
            else:a.manifest['files']['wave/arrays.npz']='b'*64
            with self.subTest(mode=mode),self.assertRaises(ValueError):d.describe_sources(a)
    def test_mid_read_source_change_rejected(self):
        a=Assets();old=a.verify
        def change():
            old()
            if a.calls==2:a.manifest['extra']='changed'
        a.verify=change
        with self.assertRaisesRegex(ValueError,'changed'):d.describe_sources(a)
    def test_integrity_failure_not_reported_as_valid_descriptor(self):
        a=Assets();a.verify=lambda:(_ for _ in ()).throw(RuntimeError('fixture integrity'))
        with self.assertRaisesRegex(RuntimeError,'fixture integrity'):d.describe_sources(a)

if __name__=='__main__':unittest.main()
