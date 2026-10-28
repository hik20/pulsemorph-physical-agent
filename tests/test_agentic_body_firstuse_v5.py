import ast,json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from pulsemorph.agentic.embodied.app.firstuse_v5 import parse_request,execute,close_evidence,digest
from pulsemorph.agentic.embodied.preview_provider_portable_v5 import plan_preview_decision
from pulsemorph.agentic.embodied.preview_orchestrator_v2 import preview_schema

class FirstUseTests(unittest.TestCase):
    def test_parent_export_failure_preserves_all_sandboxes_and_detects_changed_source(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source=root/'pinned.py';source.write_text('before');pins={str(source):digest(source)}
            source.write_text('after')
            class Episode:
                def export(self,path):raise OSError('Injected disk failure')
            class Fork:
                _previews={'preview_000':{},'preview_001':{}}
                def export_preview(self,identity,path):path.mkdir();(path/'trace.json').write_text(identity)
            result={'status':'completed'}
            close_evidence(Episode(),object(),Fork(),root,result,pins)
            self.assertEqual(result['status'],'failed');self.assertTrue(result['artifact_export_incomplete']);self.assertTrue(result['source_integrity_failed'])
            self.assertEqual((root/'sandbox_preview_000/trace.json').read_text(),'preview_000')
            self.assertEqual((root/'sandbox_preview_001/trace.json').read_text(),'preview_001')
    def test_hold_has_no_provider_or_task_override(self):
        v={'schema':'pulsemorph.portable_firstuse_request.v5','mode':'hold','instruction':'Finite hold only','task':None,'provider':None}
        self.assertEqual(parse_request(v),v)
        for field,value in [('provider','openrouter_free'),('task',{}),('extra',True)]:
            with self.assertRaises(ValueError):parse_request({**v,field:value})
    def test_execute_without_exact_digest_starts_nothing(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'plan.json';p.write_text('{}')
            with self.assertRaises(ValueError):execute(p,approved_plan_sha256='0'*64)
    def test_provider_identity_is_not_disguised_as_codex(self):
        class Provider:
            model='nvidia/nemotron-3.5-super-vl:free'
            def status(self):return {'enabled':True}
            def generate_json(self,*a,**k):
                return SimpleNamespace(data={'decision':'stop','action':None,'preview_id':None,'summary':'Dur','reason':'Kanıt yetersiz'},receipt={'provider':'openrouter','model_requested':self.model,'model_returned':self.model,'live':False,'tool_events':0,'elapsed_seconds':0.,'trust_boundary':'Fixture only'})
        capabilities={'tools':{'move_joints':{'arguments_schema':{'type':'object','properties':{'duration_s':{'type':'number'},'joint_targets_rad':{'type':'object','properties':{'joint':{'type':'number'}},'required':['joint'],'additionalProperties':False}},'required':['duration_s','joint_targets_rad'],'additionalProperties':False}}}}
        schema=preview_schema(capabilities,[])
        with tempfile.TemporaryDirectory()as d:
            r=plan_preview_decision(Provider(),{'observation_sha256':'a'*64},schema,Path(d)/'model')
            self.assertEqual(r['receipt']['model_requested'],Provider.model);self.assertFalse(r['receipt']['live'])
    def test_portable_orchestrator_inner_state_machine_ast_unchanged(self):
        import pulsemorph.agentic.embodied.preview_orchestrator_v2 as old
        import pulsemorph.agentic.embodied.preview_orchestrator_portable_v5 as new
        def methods(module):
            tree=ast.parse(Path(module.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));return {n.name:ast.dump(n,include_attributes=False)for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(methods(old)['_run'],methods(new)['_run'])
