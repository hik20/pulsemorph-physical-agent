"""Independent synthetic servo/history/cap tests. Zero actual backend calls."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
path=Path(__file__).resolve().parents[1]/'experiments/agentic/embodied/audit_hybrid_v1.py'
spec=importlib.util.spec_from_file_location('hybrid_saved_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class HybridAuditTests(unittest.TestCase):
    def fixture(self):
        n=2;arms=list(range(15,29));cfg={'arm_indices':arms,'nonarm_indices':list(range(15))}
        rows={key:np.full((n,29),value) for key,value in [('q_ref',.1),('dq_ref',.2),('qfrc_bias_joint',3.),('kp',10.),('kd',2.)]}
        raw={f'call_{i+1:04d}__joint_pos_targets':np.full((1,29),i+.5,np.float32) for i in range(n)}
        applied=np.stack([raw[f'call_{i+1:04d}__joint_pos_targets'][0].copy() for i in range(n)])
        applied[:,arms]=(.1+(2./10)*.2+3./10)
        rows['raw_graph_PD']=np.stack([raw[f'call_{i+1:04d}__joint_pos_targets'][0] for i in range(n)]).astype(float)
        rows['applied_PD']=applied.astype(float);rows['qpos_at_control']=np.zeros((n,36));rows['qvel_at_control']=np.zeros((n,35))
        feeds={'call_0001__historical_processed_actions':np.zeros((1,1,29),np.float32),'call_0002__historical_processed_actions':applied[0][None,None,:].copy()}
        return rows,raw,applied,feeds,cfg
    def test_formula_applied_history_and_raw_outputs_separate(self):
        args=self.fixture();before=copy.deepcopy(args[1]);self.assertTrue(module.check_hybrid_controls(*args))
        for key in before:self.assertTrue(np.array_equal(before[key],args[1][key]))
        self.assertFalse(np.array_equal(args[2][0],args[1]['call_0001__joint_pos_targets'][0]))
    def test_old_graph_pd_arm_identity_is_not_required(self):
        args=self.fixture();self.assertTrue(module.check_hybrid_controls(*args))
    def test_wrong_formula_nonarm_raw_or_applied_history_rejected(self):
        for kind in ('formula','nonarm','raw','history','kp','map','nan'):
            rows,raw,applied,feeds,cfg=self.fixture()
            if kind=='formula':rows['qfrc_bias_joint'][0,15]+=.1
            elif kind=='nonarm':applied[0,0]+=.2;rows['applied_PD'][0,0]=applied[0,0]
            elif kind=='raw':raw['call_0001__joint_pos_targets'][0,15]=9
            elif kind=='history':feeds['call_0002__historical_processed_actions'][:]=0
            elif kind=='kp':rows['kp'][0,15]=0
            elif kind=='map':cfg['arm_indices'][0]=14
            else:rows['dq_ref'][0,15]=np.nan
            with self.subTest(kind=kind),self.assertRaises(ValueError):module.check_hybrid_controls(rows,raw,applied,feeds,cfg)
    def force_fixture(self):
        cfg={'compiled_model':{'actuator_trnid':[[i+1,-1] for i in range(29)],'jnt_actfrcrange':[[-1,1]]*30,'jnt_actfrclimited':[0]+[1]*29,'actuator_forcerange':[[0,0]]*29,'actuator_forcelimited':[0]*29}}
        row={'control_index':0,'steps_completed_in_chunk_control':20,'joint_caps_observed_compliant':True,'peak_abs_joint_qfrc_actuator':[.5]*29,'peak_abs_individual_actuator_force':[2.]*29}
        return [row],cfg,1
    def test_joint_aggregate_caps_distinct_from_individual_force(self):self.assertTrue(module.check_force_peaks(*self.force_fixture()))
    def test_force_violation_missing_rows_or_asymmetric_uncertainty_rejected(self):
        for kind in ('peak','scope','asymmetric','negative'):
            rows,cfg,n=self.force_fixture()
            if kind=='peak':rows[0]['peak_abs_joint_qfrc_actuator'][0]=1.1
            elif kind=='scope':rows[0]['steps_completed_in_chunk_control']=19
            elif kind=='asymmetric':cfg['compiled_model']['jnt_actfrcrange'][1]=[-.5,1]
            else:rows[0]['peak_abs_joint_qfrc_actuator'][0]=-.1
            with self.subTest(kind=kind),self.assertRaises(ValueError):module.check_force_peaks(rows,cfg,n)
if __name__=='__main__':unittest.main()
