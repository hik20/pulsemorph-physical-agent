"""Renderer setup/diagnostic fixtures only; no GL/FK/NN/physics or codecs."""
import ast,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.dance import render_v2 as r

class Tests(unittest.TestCase):
    def test_render_failure_preserves_actual_partial_counts(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT_CACHE.parent) as td:
            out=Path(td)/'render';trial={'paths':{'directory':str(Path(td)/'trial')}}
            def fail(*args,**kwargs):
                out.mkdir();(out/'attempt.json').write_text(json.dumps({'phase':'render','frames_completed':86,'new_fk':174,'new_ONNX':0,'new_steps':0}));raise BrokenPipeError('fixture encoder ended')
            with patch.object(r,'ROOT_CACHE',Path(td)),patch.object(r,'_render_dance',fail):
                with self.assertRaises(BrokenPipeError):r.render_dance(ROOT,None,trial,trial,{},out)
            failure=json.loads((out/'failure.json').read_bytes());self.assertEqual(failure['error_type'],'BrokenPipeError');self.assertEqual(failure['attempt']['new_fk'],174);self.assertIn('fixture encoder ended',failure['traceback']);self.assertEqual(failure['phase'],'render')
    def test_mux_failure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT_CACHE.parent) as td:
            out=Path(td)/'render';trial={'paths':{'directory':str(Path(td)/'trial')}}
            def fail(*args,**kwargs):
                out.mkdir();(out/'attempt.json').write_text(json.dumps({'phase':'mux_selected','frames_completed':100,'new_fk':200,'new_ONNX':0,'new_steps':0}));raise RuntimeError('fixture mux failed')
            with patch.object(r,'ROOT_CACHE',Path(td)),patch.object(r,'_render_dance',fail):
                with self.assertRaises(RuntimeError):r.render_dance(ROOT,None,trial,trial,{},out)
                before=(out/'failure.json').read_bytes()
                with self.assertRaises(ValueError):r.render_dance(ROOT,None,trial,trial,{},out)
            self.assertEqual(before,(out/'failure.json').read_bytes());self.assertEqual(json.loads(before)['phase'],'mux_selected')
    def test_encoder_resource_bound_and_numeric_budget_unchanged(self):
        tree=ast.parse(Path(r.__file__).read_text(encoding='utf-8-sig'));writers=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='get_writer'];self.assertEqual(len(writers),1);kw={x.arg:ast.literal_eval(x.value) for x in writers[0].keywords};self.assertEqual(kw['ffmpeg_params'],['-threads','1']);self.assertEqual(kw['fps'],25)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)];self.assertNotIn('mj_step',calls);self.assertNotIn('run',calls)
    def test_frozen_http_composition(self):
        from pulsemorph.agentic.dance import server_v2,server_v1
        self.assertIs(server_v2.Facade,server_v1.Facade);self.assertIs(server_v2.Handler,server_v1.Handler);self.assertIs(server_v2.Server,server_v1.Server)

if __name__=='__main__':unittest.main()
