"""Pure injected orchestration: no provider/compiler/native simulation calls."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.performance import orchestrator_v1 as o

def program(value=.1,rationale='fixture',bad_start=False):
    return {'music_sha256':'a'*64,'duration_s':16.8,'tempo_bpm':100,'rationale':rationale,'keyframes':[{'beat_index':0,'q_rad':[.01 if bad_start else 0.]*29},{'beat_index':28,'q_rad':[value]*29}]}
def decision(kind,value=None,pid=''):
    return {'decision':kind,'program':value,'preview_id':pid,'summary':'Sentetik fixture kararı','reason':'Gerçek model çağrısı değil'}
class Parent:
    def __init__(self):self.backend=SimpleNamespace(sim_time_s=2.,joint_names=['joint_'+str(i) for i in range(29)],joint_ranges=[(-2.,2.)]*29);self.reference_q=np.zeros(29);self.reference_dq=np.zeros(29);self.revision=0;self.marker=0.
    def observe(self):return {'schema':'fixture','revision':self.revision,'marker':self.marker}
    def capabilities(self):return {'mode':'pure fixture'}
class Pool:
    max_previews=3;max_total_preview_controls=2520
    def __init__(self):self.parent=Parent();self.entries={};self.commits={};self.preview_calls=[]
    def preview(self,request,curve):
        pid='preview_'+str(len(self.entries));branch=SimpleNamespace(marker=curve.marker);self.entries[pid]={'branch':branch};self.preview_calls.append(copy.deepcopy(request));return {'preview_id':pid,'hypothetical_only':True}
    def commit(self,request):
        self.commits[request['commit_id']]=copy.deepcopy(request);self.parent.marker=self.entries[request['preview_id']]['branch'].marker;self.parent.revision+=1;return {'preview_id':request['preview_id'],'preview_actual_byte_equal':True}
class Provider:
    def __init__(self,decisions):self.provider=SimpleNamespace(model='gpt-6.1-sol');self.decisions=iter(decisions);self.contexts=[]
    def propose(self,context,**kwargs):
        self.contexts.append(copy.deepcopy(context));value=next(self.decisions)
        if isinstance(value,Exception):raise value
        # This receipt exists only in a temporary mock run, not actual evidence.
        return SimpleNamespace(data=copy.deepcopy(value),receipt={'provider':'synthetic_fixture','live':True,'tool_events':0,'model_requested':'gpt-6.1-sol','model_returned':None}),{'fixture_only':True}
def compile_fixture(mapped,names,ranges,q0,dq0):
    q=np.asarray([[f['joint_targets_rad'][n] for n in names] for f in mapped['keyframes']])
    if not np.array_equal(q[0],q0):raise o.ChoreographyError('Initial reference must match exactly',{'joint_ranges':[],'activation':{}})
    return SimpleNamespace(report={'admitted':True},program_sha256=o.fingerprint(mapped),marker=float(q[-1,0]),duration_s=16.8,times=np.array([0.,16.8]),q=q,dq=np.zeros_like(q))
def evaluate_fixture(episode,music,start):return {'performance_screen_passed':episode.marker>0,'feedback':{'observed_marker':episode.marker,'not_aesthetic':True},'music_alignment':{'fixture_only':True}}

class Tests(unittest.TestCase):
    def run_fixture(self,decisions):
        provider=Provider(decisions);pool=Pool()
        with tempfile.TemporaryDirectory() as d,patch.object(o,'ChoreographyProvider',lambda p:p),patch.object(o,'compile_program',compile_fixture),patch.object(o,'evaluate_episode',evaluate_fixture):
            result=o.PerformanceOrchestrator(provider).run(pool,{'sha256':'a'*64,'duration_s':16.8},'Müzik ve kapsam sabit',Path(d)/'run');fixed=json.loads((Path(d)/'run/fixed.json').read_bytes())
        return result,pool,provider,fixed
    def test_actual_preview_feedback_next_model_and_explicit_non_autopick_commit(self):
        r,pool,p,fixed=self.run_fixture([decision('preview',program(.1)),decision('preview',program(.2)),decision('commit',None,'preview_0')]);self.assertEqual(r['status'],'completed');self.assertEqual(p.contexts[1]['candidate_feedback'][0]['feedback']['observed_marker'],.1);self.assertEqual(next(iter(pool.commits.values()))['preview_id'],'preview_0');self.assertEqual(p.contexts[0]['fixed_requirements'],p.contexts[2]['fixed_requirements']);self.assertEqual(fixed['instruction'],'Müzik ve kapsam sabit')
    def test_commit_before_two_admitted_previews_fails_without_actual(self):
        r,pool,p,_=self.run_fixture([decision('preview',program(.1)),decision('commit',None,'preview_0')]);self.assertEqual(r['status'],'failed');self.assertEqual(pool.commits,{});self.assertEqual(p.contexts[1]['eligible_preview_ids'],[])
    def test_bad_q0_is_zero_work_rejection_delivered_for_revision(self):
        r,pool,p,_=self.run_fixture([decision('preview',program(.1,bad_start=True)),decision('preview',program(.1)),decision('preview',program(.2)),decision('commit',None,'preview_1')]);self.assertEqual(r['status'],'completed');self.assertEqual(len(pool.preview_calls),2);self.assertEqual(p.contexts[1]['candidate_feedback'][0]['status'],'compile_rejected');self.assertIn('Initial reference',p.contexts[1]['candidate_feedback'][0]['compile_rejection']['message']);self.assertEqual(p.contexts[1]['initial_reference_q'],[0.]*29)
    def test_unknown_provider_failure_no_retry_or_physics(self):
        r,pool,p,_=self.run_fixture([RuntimeError('fixture unconfirmed provider completion')]);self.assertEqual(r['status'],'failed');self.assertEqual(len(p.contexts),1);self.assertEqual(pool.preview_calls,[]);self.assertEqual(r['model_completion_unconfirmed'],1)
    def test_rationale_only_change_does_not_unlock_numeric_revision(self):
        r,pool,p,_=self.run_fixture([decision('preview',program(.1,'first')),decision('preview',program(.1,'different words')),decision('commit',None,'preview_0')]);self.assertEqual(r['actual_commits'],0);self.assertEqual(len(pool.preview_calls),1)

if __name__=='__main__':unittest.main()
