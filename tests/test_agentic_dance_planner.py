"""Finite mock planner boundaries; no provider/network/controller calls."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.dance import planner_v1 as p

def context():return {'music_summary':{'sha256':'a'*64,'duration_s':4,'sample_rate':48000,'sample_frames':192000},'source':{'id':'output_dance','baseline_parameters':p.baseline_parameters(),'baseline_qualification':p.BASELINE_QUALIFICATION},'constraints':{'physical_screen_only':True}}
def response():return {'supported':True,'intent_summary':'Edit existing arms','unsupported_requests':[],'parameters':{k:0. if k=='timing_shift_s' else 1. for k in p.FIELDS},'expected_changes':['Bounded reference arm edit, not measured intent.']}
class Provider:
    model='gpt-6.1-sol'
    def __init__(self,data=None,live=False):self.data=data or response();self.live=live;self.calls=0
    def status(self):return {'enabled':True}
    def generate_json(self,prompt,schema,**kwargs):
        self.calls+=1;return SimpleNamespace(data=copy.deepcopy(self.data),receipt={'provider':'fixture','model_requested':self.model,'model_returned':None,'live':self.live,'tool_events':0,'elapsed_seconds':.1,'trust_boundary':'fixture only'})

class Tests(unittest.TestCase):
    def test_strict_parameters(self):
        good=response()['parameters']
        for bad in ({**good,'left_gain_a':True},{**good,'left_gain_a':float('nan')},{**good,'timing_shift_s':.151},{**good,'left_gain_b':10**999},{**good,'unknown':1}):
            with self.assertRaises(ValueError):p.parse_parameters(bad)
    def test_request_and_actual4s_context(self):
        ready=p.prepare_dance_request({'instruction':'Make the existing arms quieter'},context())
        self.assertNotIn('goals',ready['request']);self.assertFalse(ready['fixed_scope']['intent_success_measured'])
        bad=context();bad['music_summary']['duration_s']=8
        with self.assertRaises(ValueError):p.prepare_dance_request({'instruction':'edit'},bad)
    def test_fixture_receipt_never_claims_live_and_bindings(self):
        with tempfile.TemporaryDirectory(dir=p.OUTPUT_ROOT if hasattr(p,'OUTPUT_ROOT') else Path.home()/'.cache/pulsemorph-agentic') as d:
            out=Path(d)/'call';provider=Provider();result=p.plan_dance(provider,{'instruction':'edit existing arms'},context(),out)
            self.assertFalse(result['receipt']['live']);self.assertEqual(provider.calls,1)
            receipt=json.loads((out/'receipt.json').read_bytes());self.assertEqual(receipt['request_file_sha256'],p._hash((out/'request.json').read_bytes()))
            self.assertEqual(receipt['response_sha256'],p._hash((out/'response.json').read_bytes()))
    def test_invalid_response_preserved_no_retry(self):
        data=response();data['parameters']['left_gain_a']=1.3;provider=Provider(data)
        with tempfile.TemporaryDirectory(dir=Path.home()/'.cache/pulsemorph-agentic') as d:
            out=Path(d)/'call'
            with self.assertRaises(p.PlannerResponseRejected):p.plan_dance(provider,{'instruction':'edit arms'},context(),out)
            self.assertTrue((out/'response.json').exists());self.assertTrue((out/'receipt.json').exists());self.assertEqual(provider.calls,1)
    def test_unsupported_is_not_overridden(self):
        data=response();data.update(supported=False,unsupported_requests=['New leg choreography unavailable'])
        with tempfile.TemporaryDirectory(dir=Path.home()/'.cache/pulsemorph-agentic') as d:
            self.assertFalse(p.plan_dance(Provider(data),{'instruction':'new steps'},context(),Path(d)/'call')['locally_supported'])

if __name__=='__main__':unittest.main()
