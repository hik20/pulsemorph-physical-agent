"""V2 composition only; no app construction, subprocesses or native runtime."""
import ast,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.app import server_v1,server_v2
class Tests(unittest.TestCase):
    def test_frozen_http_reused(self):
        self.assertIs(server_v2.Handler,server_v1.Handler);self.assertIs(server_v2.Server,server_v1.Server)
    def test_only_versioned_worker_selected_without_implicit_start(self):
        tree=ast.parse(Path(server_v2.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='BodyAgentManager'];self.assertEqual(len(calls),1);worker=next(k.value for k in calls[0].keywords if k.arg=='worker');self.assertIn('worker_v2.py',ast.unparse(worker));self.assertNotIn('worker_v1.py',ast.unparse(worker))
if __name__=='__main__':unittest.main()
