"""New expression provenance/adapter gates only; no worker/env/controller launched."""
import ast,copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_agentic_phrase_expression import reference,expression
from pulsemorph.motion.contracts import MotionReference
from pulsemorph.agentic.phrase_reference import plain
from pulsemorph.agentic.expression.player_v1 import ExpressionPlayer,expression_hash,validate_expression_reference
from pulsemorph.agentic.expression import worker_v1 as worker
from pulsemorph.agentic.expression.runtime_v1 import ExpressionRuntime,expected_implementation_pins,validate_trial_identity,validate_trial_paths
from pulsemorph.agentic.app.phrase import worker as frozen

def common(value=None):
    value=value or expression();source=reference();metadata=plain(source.metadata);metadata['reference_id']='fixture_job__baseline';metadata['validity']['end_time_s']=8.14;metadata['producer']={'kind':'agentic','id':'phrase-expression-reference','version':'1'};metadata['context'].update(task_id='fixture_job',runtime_session_id='fixture_job__baseline',expression_sha256=expression_hash(value));metadata['provenance'].update(phrase_expression=value,phrase_expression_sha256=expression_hash(value));t=np.arange(408)*.02;q=np.tile((.01*t)[:,None],(1,29));dq=np.full_like(q,.01)
    return MotionReference(metadata,t,q,dq,np.tile([0.,0.,0.,1.],(408,1)))
def carrier(ref):return {'body_pos':np.zeros((33,3),np.float32),'body_rot':np.tile([0.,0.,0.,1.],(33,1)).astype('float32'),'dof_pos':ref.joint_position_rad[0].astype('float32'),'dof_vel':ref.joint_velocity_rad_s[0].astype('float32')}
class AdapterTests(unittest.TestCase):
    def test_original_expression_producer_preserved_and_old_sampling_inherited(self):
        ref=common();p=ExpressionPlayer(ref,carrier(ref),16,expression());self.assertEqual(p.reference.metadata['producer']['id'],'phrase-expression-reference')
        self.assertIs(ExpressionPlayer.get_state_at_frame,frozen.PhrasePlayer.get_state_at_frame);self.assertIs(ExpressionPlayer.get_future_references,frozen.PhrasePlayer.get_future_references)
        self.assertEqual(p.get_future_references(0,[1,2,4,8])['dof_pos'].shape,(4,29));self.assertEqual(p.get_state_at_frame(400)['dof_pos'].shape,(29,))
        with self.assertRaises(ValueError):frozen.PhrasePlayer(ref,carrier(ref),16)
    def test_changed_expression_hash_old_producer_or_carrier_rejected(self):
        ref=common();wrong=expression(first_left_pose_gain=.75)
        with self.assertRaises(ValueError):validate_expression_reference(ref,wrong)
        meta=plain(ref.metadata);meta['producer']['id']='phrase-program-reference';bad=MotionReference(meta,ref.time_offsets_s,ref.joint_position_rad,ref.joint_velocity_rad_s,ref.anchor_orientation_xyzw)
        with self.assertRaises(ValueError):validate_expression_reference(bad,expression())
        initial=carrier(ref);initial['dof_pos'][0]+=.1
        with self.assertRaises(ValueError):ExpressionPlayer(ref,initial,16,expression())
    def test_invalid_metadata_stops_before_environment_or_started_sentinel(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'trial';directory.mkdir();path=directory/'request.json';request={'schema':'pulsemorph.expression_physical_request.v1','artifact_root':temp,'id':'fixture_job__baseline','index':-1,'job_id':'fixture_job','expression':expression(),'source_pins':{},'environment':{},'music_observation':{}}
            path.write_text(json.dumps(request));bad=reference()
            with patch.object(worker,'load_reference',return_value=bad),patch.object(worker,'environment') as environment:
                with self.assertRaises(ValueError):worker.execute(path)
                environment.assert_not_called();self.assertFalse((directory/'worker_started.json').exists())
    def test_numeric_closures_and_upstream_call_ast_identical(self):
        def execute_tree(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        old=execute_tree(Path(frozen.__file__));new=execute_tree(Path(worker.__file__))
        for name in ('budget','Session','initial','step'):
            a=next(n for n in ast.walk(old) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);b=next(n for n in ast.walk(new) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
        def upstream(node):return next(n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='upstream' and n.func.attr=='run')
        self.assertEqual(ast.dump(upstream(old)),ast.dump(upstream(new)))
        self.assertIs(worker.environment,frozen.environment);self.assertIs(worker.PrefixGate,frozen.PrefixGate);self.assertIs(worker.call_budget,frozen.call_budget)
    def test_runtime_reverse_or_nonunit_baseline_never_spawns(self):
        runtime=ExpressionRuntime.__new__(ExpressionRuntime)
        with patch('subprocess.Popen') as process,patch('subprocess.run') as run:
            for value in (expression(order=['high5','wave']),expression(second_left_pose_gain=.75)):
                with self.assertRaises(ValueError):runtime.run_trial(value,Path('unused'),'fixture',-1,{})
            process.assert_not_called();run.assert_not_called()
    def test_request_size_bound_before_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'request.json';path.write_bytes(b' '* (128*1024+1))
            with self.assertRaises(ValueError):worker.load_request(path)
    def test_new_audit_schema_preserves_old_metrics_and_separate_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);q=np.zeros((401,36));t=np.arange(401)*.02;np.save(directory/'qpos.npy',q);np.save(directory/'state_times.npy',t)
            old={'schema':'pulsemorph.app_phrase_audit.v1','inventory':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('qpos.npy','state_times.npy')},'metrics':{'arm_rms_rad':[1.,2.,3.,4.]},'qualification':'Synthetic audit fixture, not a physical record.'}
            assets=SimpleNamespace(references=lambda:{'wave':reference(),'high5':reference()})
            with patch.object(frozen,'audit_saved',return_value=old) as audit,patch.object(worker,'profile_from_yaml',return_value={'joint_names':tuple(reference().metadata['body']['joint_names'])}):result=worker.audit_saved(directory,assets,{})
            audit.assert_called_once();self.assertEqual(result['schema'],'pulsemorph.expression_saved_audit.v1');self.assertEqual(result['metrics'],old['metrics']);self.assertEqual(len(result['expression_diagnostics']),4);self.assertEqual(result['diagnostic_new_FK_calls'],0);self.assertEqual(old['schema'],'pulsemorph.app_phrase_audit.v1')
    def test_pin_map_includes_new_and_frozen_numeric_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);(directory/'manifest.json').write_text('{}');python=directory/'python';python.write_bytes(b'synthetic executable fixture');assets=SimpleNamespace(directory=directory,manifest={'live_numeric_pins':{},'files':{}})
            pins=expected_implementation_pins(ROOT,assets,python,{'module_pins':{}})
            for relative in ('src/pulsemorph/agentic/phrase_expression_v1.py','src/pulsemorph/agentic/expression/worker_v1.py','src/pulsemorph/agentic/expression/player_v1.py','src/pulsemorph/agentic/app/phrase/worker.py'):self.assertIn(str(ROOT/relative),pins)
    def test_wrapper_identity_and_bool_spoofs_rejected(self):
        trial={'id':'fixture__trial_0','index':0};request={'id':trial['id'],'index':0,'job_id':'fixture'};result=copy.deepcopy(request);validate_trial_identity(trial,request,result)
        for field,value in (('id','spoofed'),('index',1),('index',False),('id',1)):
            bad=dict(trial,**{field:value})
            with self.subTest(field=field,value=value),self.assertRaises(RuntimeError):validate_trial_identity(bad,request,result)
        for field,value in (('index',False),('job_id',True),('id',True)):
            bad=dict(result,**{field:value})
            with self.subTest(result_field=field),self.assertRaises(RuntimeError):validate_trial_identity(trial,request,bad)
    def test_direct_worker_identity_types_reject_before_reference_or_env(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'request.json';path.write_text('{}')
            valid={'schema':'pulsemorph.expression_physical_request.v1','artifact_root':temp,'id':'fixture__trial_0','index':0,'job_id':'fixture','expression':expression(),'source_pins':{},'environment':{},'music_observation':{}}
            for field,value in (('index',False),('id',True),('job_id',True)):
                with patch.object(worker,'load_reference') as load,patch.object(worker,'environment') as env:
                    with self.assertRaises(ValueError):worker.validate_request(path,dict(valid,**{field:value}))
                    load.assert_not_called();env.assert_not_called()
    def test_wrapper_paths_cannot_reassign_downstream_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp).resolve();paths={'directory':str(directory),'result':str(directory/'result.json'),'audit':str(directory/'audit.json'),'common_reference':str(directory/'common_reference')};validate_trial_paths({'paths':paths},directory)
            for name in paths:
                bad=dict(paths,**{name:str(directory/'other')})
                with self.subTest(name=name),self.assertRaises(RuntimeError):validate_trial_paths({'paths':bad},directory)
            with self.assertRaises(RuntimeError):validate_trial_paths({'paths':{**paths,'extra':'unexpected'}},directory)
            with self.assertRaises(RuntimeError):validate_trial_paths({'paths':dict(paths,result=True)},directory)
if __name__=='__main__':unittest.main()
