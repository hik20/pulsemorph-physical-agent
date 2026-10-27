"""Synthetic export arrays, no FK/model/physics execution."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
P=Path(__file__).resolve().parents[1]/'experiments/agentic/embodied/audit_episode_v1.py'
spec=importlib.util.spec_from_file_location('episode_audit_fixture',P);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class EmbodiedAuditTests(unittest.TestCase):
    def arrays(self):
        n=3;q=np.zeros((n+1,36));v=np.zeros((n+1,35));times=np.arange(n+1)*.02
        targets=np.array([[i+1]*29 for i in range(n)],dtype=np.float32)
        feeds={};outputs={};calls=[]
        for i in range(n+1):
            prefix=f'call_{i:04d}__'
            for k in audit.INPUTS:feeds[prefix+k]=np.zeros((1,29),np.float32)
            feeds[prefix+'current_root_local_ang_vel']=np.zeros((1,3),np.float32)
            feeds[prefix+'historical_processed_actions']=np.zeros((1,1,29),np.float32) if i<2 else targets[i-2][None,None,:].copy()
            for k in audit.OUTPUTS:outputs[prefix+k]=np.zeros((1,29),np.float32) if i==0 else targets[i-1][None].copy()
            calls.append({'index':i,'role':'warmup' if i==0 else 'control','status':'completed','sim_time_s':max(0,i-1)*.02})
        sub=np.zeros((n*20,6));sub[:,0]=np.arange(1,n*20+1);sub[:,1]=np.arange(1,n*20+1)*.001
        return [q,v,times,targets,feeds,outputs,calls,sub]
    def test_accepts_continuous_saved_arrays(self):self.assertEqual(audit.audit_arrays(*self.arrays()),3)
    def test_history_reset_at_chunk_boundary_rejected(self):
        x=self.arrays();x[4]['call_0003__historical_processed_actions'][:]=0
        with self.assertRaisesRegex(ValueError,'history'):audit.audit_arrays(*x)
    def test_pd_targets_must_be_actor_output(self):
        x=self.arrays();x[3][1,0]=99
        with self.assertRaises(ValueError):audit.audit_arrays(*x)
    def test_reset_clock_and_multiple_warmups_rejected(self):
        for kind in ('clock','warmup','substep'):
            x=self.arrays()
            if kind=='clock':x[2][2]=0
            elif kind=='warmup':x[6][2]['role']='warmup'
            else:x[7][22,0]=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.audit_arrays(*x)
    def test_actual_state_and_output_contract(self):
        for kind in ('q','angular','dtype','extra','nan'):
            x=self.arrays()
            if kind=='q':x[0][1,7]=1
            elif kind=='angular':x[1][1,3]=1
            elif kind=='dtype':x[5]['call_0001__actions']=np.zeros((1,29),np.float64)
            elif kind=='extra':x[4]['invented']=np.zeros(1)
            else:x[0][0,0]=np.nan
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.audit_arrays(*x)
if __name__=='__main__':unittest.main()
