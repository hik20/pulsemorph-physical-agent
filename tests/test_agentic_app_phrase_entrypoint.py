"""Parent-process import regression from a fresh isolated launch, no controller."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe')


class EntrypointTests(unittest.TestCase):
    def test_spoofed_already_loaded_helper_is_rejected_before_admission(self):
        code = '''import sys,types
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src'),str(root),str(root/'experiments/agentic')]
fake=types.ModuleType('deployment.state_utils');fake.__file__=str(root/'research/vendor/ProtoMotions/deployment/state_utils.py')
sys.modules['deployment.state_utils']=fake
from pulsemorph.agentic.app.phrase.entrypoint_v2 import prepare_parent_imports
try:prepare_parent_imports(root)
except RuntimeError as error:
 assert str(error)=='Fresh parent deployment module cache required'
 print('cached-code-rejected')
else:raise AssertionError('spoofed module cache was accepted')
'''
        result = subprocess.run([str(PYTHON), '-I', '-B', '-c', code, str(ROOT)],
                                capture_output=True, text=True, timeout=15,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'cached-code-rejected')

    def test_fresh_parent_composes_aligned_baseline_without_hidden_test_paths(self):
        code = '''import sys,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src'),str(root),str(root/'experiments/agentic')]
assert 'deployment.state_utils' not in sys.modules
from pulsemorph.agentic.app.phrase.entrypoint_v2 import prepare_parent_imports
proof=prepare_parent_imports(root)
from pulsemorph.agentic.app.phrase.assets import PhraseAssets
from pulsemorph.agentic.phrase_program import compile_program
assets=PhraseAssets(root)
program={'schema':'pulsemorph.phrase_program.v1','order':['wave','high5'],'heading_policy':'align_join_heading','first_left_gain':1.,'first_right_gain':1.,'second_left_gain':1.,'second_right_gain':1.,'rationale':'Fresh isolated parent compile-only regression'}
reference,report=compile_program(assets.references(),program,reference_id='entrypoint_fixture_no_execution',sources=assets.source_records())
assert report['strict_future_queries']==400 and report['execution_authority'] is False
print(json.dumps({'isolated':bool(sys.flags.isolated),'helper_sha256':proof['helper_sha256'],'queries':400,'physics':0,'ONNX':0}))
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([str(PYTHON), '-I', '-B', '-c', code, str(ROOT)], cwd=directory,
                                    capture_output=True, text=True, timeout=40,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0, result.stderr)
        proof = json.loads(result.stdout)
        self.assertTrue(proof['isolated'])
        self.assertEqual(proof['queries'], 400)
        self.assertEqual(proof['physics'], 0)
        self.assertEqual(proof['ONNX'], 0)


if __name__ == '__main__':
    unittest.main()
