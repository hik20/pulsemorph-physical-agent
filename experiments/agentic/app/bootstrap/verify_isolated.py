"""Run with the dedicated interpreter -I; use explicit project source paths."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import site
import sys
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests',action='store_true')
    parser.add_argument('--test-name',action='append',help='Optional explicit fixture test names; default discovers all app fixtures')
    args=parser.parse_args();root=args.root.resolve();prefix=Path(sys.prefix).resolve()
    if args.output.exists():raise FileExistsError(args.output)
    assert sys.flags.isolated==1 and not site.ENABLE_USER_SITE
    assert not any('site-packages' in p.lower() and not Path(p).resolve().is_relative_to(prefix) for p in sys.path)
    sys.path.insert(0,str(root/'src'))
    before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src/pulsemorph/agentic/app').rglob('*.py')}
    modules={}
    for name in ['torch','numpy','scipy','onnxruntime','mujoco','yaml','jsonschema','PIL','imageio','imageio_ffmpeg','etils.epath','OpenGL','filelock','networkx','jinja2']:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        assert path.is_relative_to(prefix),(name,path)
        modules[name]=str(path.relative_to(prefix))
    for name in ['providers','orchestrator','server','integrations','runtime','renderer','music','uploads','runtime_config']:
        importlib.import_module('pulsemorph.agentic.app.'+name)
    import torch,onnxruntime
    result={'schema':'pulsemorph.isolated_app_import_fixture_check.v1','python':sys.version,'executable':sys.executable,
            'isolated':bool(sys.flags.isolated),'user_site_enabled':site.ENABLE_USER_SITE,
            'sys_path':sys.path,'modules_inside_dedicated_env':modules,'source_hashes_before':before,
            'torch_build':torch.__version__,'torch_cuda_build':torch.version.cuda,
            'torch_cuda_available':torch.cuda.is_available(),'onnx_available_providers':onnxruntime.get_available_providers(),
            'model_calls':0,'physics_steps':0,'tests':None}
    if args.tests:
        sys.path.insert(0,str(root/'tests'));suite=unittest.TestSuite()
        if args.test_name:
            for name in args.test_name:suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
        else:suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_agentic_app_*.py'))
        stream=io.StringIO();tested=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        result['tests']={'run':tested.testsRun,'failures':len(tested.failures),'errors':len(tested.errors),'passed':tested.wasSuccessful(),'log':stream.getvalue()}
    after={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src/pulsemorph/agentic/app').rglob('*.py')}
    result['source_hashes_unchanged']=before==after
    result['passed']=result['source_hashes_unchanged'] and (result['tests'] is None or result['tests']['passed'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'passed':result['passed'],'isolated':result['isolated'],'tests':result['tests']['run'] if result['tests'] else 0,'output':str(args.output)}))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
