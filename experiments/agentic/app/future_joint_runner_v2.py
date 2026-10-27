"""One prospective symbolic-metadata setup repair; original v1 consumed zero calls."""
from pathlib import Path
import argparse
import importlib.util
import importlib.metadata
import json
import site
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / 'artifacts/runs/agentic/app_future_joint_probe_plan_v1'
OLD_OUT = ROOT / 'artifacts/runs/agentic/app_future_joint_probe_v1'
OUT = ROOT / 'artifacts/runs/agentic/app_future_joint_probe_v2'
SUPPLEMENT = PLAN / 'runner_supplement_v2.json'
DIAGNOSIS_SHA = 'c10b0b8c8dc24257f3782de0f64cd20c2dc523d62d038a99fe68bed8d602b309'
spec = importlib.util.spec_from_file_location('future_probe', Path(__file__).with_name('future_joint_probe_v1.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
np = probe.np


def declared_metadata():
    diagnosis = OLD_OUT / 'metadata_failure_review_v1.json'
    if probe.digest(diagnosis) != DIAGNOSIS_SHA:
        raise ValueError('diagnosis pin changed')
    failed = json.loads((OLD_OUT / 'result.json').read_text())
    if failed.get('status') != 'failed' or any(failed.get(k) != 0 for k in ('attempted_calls','completed_calls','durable_attempt_records')):
        raise ValueError('v1 must have consumed zero inference calls')
    return json.loads(diagnosis.read_text())['actual']


def validate_metadata(nodes, concrete, declared):
    """Exact symbolic metadata and independently exact batch-one array contract."""
    actual = [{'name':n.name,'shape':n.shape,'type':n.type} for n in nodes]
    if actual != declared:
        raise ValueError('actual metadata differs from pinned symbolic declaration')
    if len(concrete) != len(declared):
        raise ValueError('metadata arity differs')
    for c,d in zip(concrete,declared):
        if d['shape'] != ['batch_size', *c['shape'][1:]] or c['shape'][0] != 1 or c['name'] != d['name'] or c['dtype'] != 'float32' or d['type'] != 'tensor(float)':
            raise ValueError('exact symbolic axis/concrete batch-one layout required')


def environment():
    import onnxruntime as ort
    import numpy.core._multiarray_umath as numpy_binary
    import onnxruntime.capi.onnxruntime_pybind11_state as ort_binary
    import yaml
    modules = (np, ort, yaml, numpy_binary, ort_binary)
    prefix = probe.PUBLIC_PYTHON.resolve().parents[1]
    origins = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):
            raise ValueError('module outside public environment: ' + module.__name__)
        origins[module.__name__] = {'path': str(path), 'sha256': probe.digest(path)}
    if Path(sys.executable).resolve() != probe.PUBLIC_PYTHON.resolve() or not sys.flags.isolated or site.ENABLE_USER_SITE is not False:
        raise ValueError('exact isolated public environment required')
    versions = {n: importlib.metadata.version(n) for n in ('numpy','onnxruntime','PyYAML')}
    if versions != {'numpy':'1.26.4','onnxruntime':'1.31.0','PyYAML':'6.0.3'}:
        raise ValueError('package versions changed')
    return {'python': str(Path(sys.executable).resolve()), 'python_sha256': probe.digest(sys.executable),
            'isolated': True, 'user_site': False, 'versions': versions, 'origins': origins}


def pins_for(protocol):
    pins = {str(ROOT / p): h for p,h in protocol['source_pins'].items()}
    for row in protocol['pairs']:
        for entry in row['feeds'].values():
            pins[str(PLAN / entry['file'])] = entry['sha256']
    pins[str(PLAN / 'protocol.json')] = '9123e79ac5d1935d85743c32ef0479b7a0ab6f8f28a3a62ba9c5e2e6c4cab98e'
    pins[str(Path(__file__).resolve())] = probe.digest(__file__)
    return pins


def verify(pins):
    for path, expected in pins.items():
        if probe.digest(path) != expected:
            raise ValueError('frozen pin changed: ' + path)


def freeze():
    protocol = json.loads((PLAN / 'protocol.json').read_text())
    env = environment()
    pins = pins_for(protocol)
    verify(pins)
    metadata = declared_metadata()
    for name in ('result.json','metadata_failure_review_v1.json'):
        path = OLD_OUT / name
        pins[str(path)] = probe.digest(path)
    for record in env['origins'].values():
        pins[record['path']] = record['sha256']
    pins[env['python']] = env['python_sha256']
    probe.write_new(SUPPLEMENT, {'schema':'pulsemorph.future_joint_runner_supplement.v2',
        'protocol_sha256':pins[str(PLAN/'protocol.json')], 'runner_sha256':probe.digest(__file__),
        'environment':env,'pins':pins,'output':str(OUT),'maximum_calls':6,
        'exact_symbolic_metadata':metadata,'previous_revision_calls':0,'total_revision_call_budget':6,
        'setup_repair_only':'Exact symbolic batch_size metadata, concrete feeds batch1 unchanged. No further revision/retry after any inference.',
        'required_approval_fields':['approved=true','protocol_sha256','runner_sha256','supplement_sha256'],
        'status':'frozen_not_executed','new_onnx_runs':0,'new_physics_steps':0,'new_llm_calls':0})


def execute(approval_path):
    supplement = json.loads(SUPPLEMENT.read_text())
    approval = json.loads(Path(approval_path).read_text())
    if approval.get('approved') is not True or any(approval.get(k) != v for k,v in {
        'protocol_sha256':supplement['protocol_sha256'],'runner_sha256':supplement['runner_sha256'],
        'supplement_sha256':probe.digest(SUPPLEMENT)}.items()):
        raise ValueError('exact root execution approval required')
    # Exclusive top-level directory is the no-resume/no-retry boundary.
    OUT.mkdir(parents=True, exist_ok=False)
    result={'status':'failed','attempted_calls':0,'completed_calls':0,'physics_steps':0,'llm_calls':0,
            'warmup_calls':0,'approval_sha256':probe.digest(approval_path),'supplement_sha256':probe.digest(SUPPLEMENT)}
    session = None
    try:
        pins = dict(supplement['pins'])
        pins[str(SUPPLEMENT)] = probe.digest(SUPPLEMENT)
        pins[str(Path(approval_path).resolve())] = probe.digest(approval_path)
        verify(pins)
        if environment() != supplement['environment']:
            raise ValueError('environment differs from frozen supplement')
        if declared_metadata() != supplement['exact_symbolic_metadata']:
            raise ValueError('declared metadata differs')
        protocol=json.loads((PLAN/'protocol.json').read_text())
        pairs=[]
        for row in protocol['pairs']:
            arrays=[]
            for kind in ('baseline','edited','expected'):
                with np.load(PLAN/row['feeds'][kind]['file'],allow_pickle=False) as z:
                    arrays.append({n:z[n].copy() for n in z.files})
            pairs.append((row['decision'],*arrays))
        import onnxruntime as ort
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        session=ort.InferenceSession(str(probe.MODEL),sess_options=options,providers=['CPUExecutionProvider'])
        if session.get_providers()!=['CPUExecutionProvider']:
            raise ValueError('CPU-only provider required')
        for kind,nodes in (('inputs',session.get_inputs()),('outputs',session.get_outputs())):
            validate_metadata(nodes,protocol[kind],supplement['exact_symbolic_metadata'][kind])
        actual_options=session.get_session_options()
        if (actual_options.intra_op_num_threads,actual_options.inter_op_num_threads)!=(2,1):
            raise ValueError('session threads differ')
        class CheckedSession:
            def run(self,names,feed):
                verify(pins)
                index=result['attempted_calls']
                if index>=6: raise ValueError('maximum six calls')
                # Primitive already fsyncs its attempt before entering this wrapper.
                result['attempted_calls']+=1
                tick=time.perf_counter()
                values=session.run(names,feed)
                result['completed_calls']+=1
                np.savez(OUT/f'raw_actual_{index:02d}.npz',**dict(zip(names,values)))
                result.setdefault('call_elapsed_ms',[]).append((time.perf_counter()-tick)*1000)
                verify(pins)
                return values
        rows=probe.finite_calls(CheckedSession(),pairs,OUT/'calls')
        measurements=[]
        for i,(decision,baseline,edited,expected) in enumerate(pairs):
            b,e=rows[2*i][2],rows[2*i+1][2]
            outputs={}
            for j,name in enumerate(probe.OUTPUTS):
                delta=(e[j]-b[j])[0]
                outputs[name]={'baseline_replay_max_abs_error':float(np.max(np.abs(b[j]-expected[name]))),
                    'delta_by_joint':dict(zip(protocol['joint_names'],map(float,delta))),
                    'max_abs_delta':float(np.max(np.abs(delta)))}
                if name in ('actions','joint_pos_targets'):
                    outputs[name]['side_arm_l2_delta']={side:float(np.linalg.norm([delta[k] for k,n in enumerate(protocol['joint_names']) if n.startswith(side+'_') and any(t in n for t in ('shoulder','elbow','wrist'))])) for side in ('left','right')}
            measurements.append({'decision':decision,'outputs':outputs,'input_max_abs_delta':{n:float(np.max(np.abs(edited[n]-baseline[n]))) for n in probe.INPUTS}})
        probe.write_new(OUT/'measurements.json',measurements)
        result['status']='completed'
        result['interpretation']='Three same-state open-loop output differences only; no closed-loop motion/features or causal training explanation.'
    except Exception as error:
        result['error']=str(error)
    finally:
        result['durable_attempt_records']=len(list((OUT/'calls').glob('attempt_*.json'))) if (OUT/'calls').exists() else 0
        probe.write_new(OUT/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true')
    g.add_argument('--execute',type=Path,metavar='ROOT_APPROVAL')
    a=p.parse_args()
    if a.freeze: freeze()
    else:
        result=execute(a.execute)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['status']=='completed' else 1)
