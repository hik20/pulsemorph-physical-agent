"""Dynamic transport-goal pinning; pure provider mocks, no real calls."""
import copy,json,sys,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from pulsemorph.agentic.expression import planner_v2 as p
from test_agentic_expression_planner import Provider,request,context,response,GOALS

class PlannerV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name);patcher=patch.object(p,'OUTPUT_ROOT',self.base);patcher.start();self.addCleanup(patcher.stop)
    def test_actual_initial_transport_and_saved_schema_bind_1p2(self):
        provider=Provider();result=p.plan_expression(provider,request(),context(),self.base/'initial')
        received=provider.calls[0][1];stored=json.loads((self.base/'initial/schema.json').read_text());prepared=json.loads((self.base/'initial/request.json').read_text())
        self.assertEqual(received,stored);self.assertEqual(stored,prepared['provider_schema']);self.assertNotIn('response_schema',prepared)
        for field in ('mean_pose_ratio','temporal_ratio'):
            node=received['properties']['goals']['properties'][field];self.assertEqual(node['type'],'number');self.assertEqual(node['enum'],[1.2])
        self.assertEqual(result['goals'],GOALS);self.assertFalse(result['receipt']['live']);self.assertIsNone(result['receipt']['model_returned'])
        self.assertEqual(result['receipt']['provider_schema_sha256'],hashlib.sha256((self.base/'initial/schema.json').read_bytes()).hexdigest())
    def test_each_request_has_its_own_goal_enums_without_global_mutation(self):
        req=request();req['goals']['mean_pose_ratio']=.9;req['goals']['temporal_ratio']=1.1;data=response();data['goals']=copy.deepcopy(req['goals']);provider=Provider(data)
        p.plan_expression(provider,req,context(),self.base/'different');schema=provider.calls[0][1]
        self.assertEqual(schema['properties']['goals']['properties']['mean_pose_ratio']['enum'],[.9]);self.assertEqual(schema['properties']['goals']['properties']['temporal_ratio']['enum'],[1.1])
        self.assertNotIn('enum',p.RESPONSE_SCHEMA['properties']['goals']['properties']['mean_pose_ratio'])
        prepared=p.prepare_expression_request(request(),context());self.assertEqual(prepared['provider_schema']['properties']['goals']['properties']['mean_pose_ratio']['enum'],[1.2])
    def test_revision_has_same_pinned_goals_and_actual_transport_schema(self):
        feedback={'baseline':{'kind':'fresh_verified_unit'},'trials':[{'index':0,'status':'partial','actual_ratios':[1.1,1.05]},{'index':1,'status':'rejected','metrics':None,'report':{'violations':['left_elbow_joint']}}]}
        initial=p.prepare_expression_request(request(),context());provider=Provider();p.plan_expression(provider,request(),context(),self.base/'revision',feedback=feedback)
        revision=json.loads((self.base/'revision/request.json').read_text());self.assertEqual(revision['purpose'],'revision');self.assertEqual(revision['provider_schema'],initial['provider_schema']);self.assertEqual(revision['feedback'],feedback);self.assertEqual(provider.calls[0][1],revision['provider_schema'])
    def test_ignored_transport_enum_still_rejects_echo_without_overwrite(self):
        data=response();data['goals']['mean_pose_ratio']=1.2203;data['goals']['temporal_ratio']=1.2203;provider=Provider(data)
        with self.assertRaises(p.PlannerResponseRejected) as caught:p.plan_expression(provider,request(),context(),self.base/'ignored')
        self.assertEqual(caught.exception.category,'EXPLICIT_GOALS_ECHO_MISMATCH');self.assertEqual(len(provider.calls),1)
        raw=json.loads((self.base/'ignored/response.json').read_text());self.assertEqual(raw['goals'],data['goals']);self.assertEqual(json.loads((self.base/'ignored/request.json').read_text())['request']['goals'],GOALS)
        receipt=json.loads((self.base/'ignored/receipt.json').read_text());self.assertFalse(receipt['completion_unconfirmed']);self.assertFalse(receipt['live']);self.assertIsNone(receipt['model_returned']);self.assertFalse((self.base/'ignored/validated_result.json').exists())
        with self.assertRaises(FileExistsError):p.plan_expression(provider,request(),context(),self.base/'ignored')
        self.assertEqual(len(provider.calls),1)
    def test_freeze6_and_unsupported_gate_still_preserved(self):
        data=response();data['expression']['first_left_pose_gain']=.9
        with self.assertRaises(p.PlannerResponseRejected) as caught:p.plan_expression(Provider(data),request(),context(),self.base/'frozen')
        self.assertEqual(caught.exception.category,'FROZEN_GAIN_CHANGED')
        data=response();data.update(supported=False,unsupported_requests=['new skill']);result=p.plan_expression(Provider(data),request(),context(),self.base/'unsupported');self.assertFalse(result['locally_supported'])
    def test_malformed_or_changed_goal_dto_stops_before_provider(self):
        for index,value in enumerate((True,float('nan'),10**1000,'1.2')):
            req=request();req['goals']['mean_pose_ratio']=value;provider=Provider()
            with self.subTest(value=str(value)[:20]),self.assertRaises(ValueError):p.plan_expression(provider,req,context(),self.base/str(index))
            self.assertEqual(provider.calls,[])

if __name__=='__main__':unittest.main()
