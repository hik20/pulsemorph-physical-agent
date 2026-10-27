"""V3 composition only; no native controller or application initialization."""
import ast,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.app import server_v1,server_v3
class Tests(unittest.TestCase):
    def test_frozen_http_reused(self):
        self.assertIs(server_v3.Handler,server_v1.Handler);self.assertIs(server_v3.Server,server_v1.Server)
    def test_versioned_worker3_only(self):
        tree=ast.parse(Path(server_v3.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='BodyAgentManager'];self.assertEqual(len(calls),1);worker=next(k.value for k in calls[0].keywords if k.arg=='worker');self.assertIn('worker_v3.py',ast.unparse(worker));self.assertNotIn('worker_v2.py',ast.unparse(worker))
if __name__=='__main__':unittest.main()
