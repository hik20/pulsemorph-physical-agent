"""Pure portable-hybrid code identity and preparation checks; no native construction."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
class Tests(unittest.TestCase):
 def test_portable_hybrid_control_methods_match_frozen_original_ast(self):
  folder=ROOT/'src/pulsemorph/agentic/embodied';old=ast.parse((folder/'hybrid_arm_backend_v1.py').read_text());new=ast.parse((folder/'backend_portable_hybrid_v5.py').read_text());cls=lambda tree:next(v for v in tree.body if isinstance(v,ast.ClassDef));methods=lambda tree:{v.name:ast.dump(v,include_attributes=False) for v in cls(tree).body if isinstance(v,ast.FunctionDef)}
  a,b=methods(old),methods(new)
  for name in set(a)-{'__init__'}:self.assertEqual(a[name],b[name],name)
  funcs=lambda tree:{v.name:ast.dump(v,include_attributes=False) for v in tree.body if isinstance(v,ast.FunctionDef)}
  self.assertEqual(funcs(old),funcs(new))
 def test_constructor_uses_explicit_owned_profile_not_private_motion_loader(self):
  folder=ROOT/'src/pulsemorph/agentic/embodied';hybrid=(folder/'backend_portable_hybrid_v5.py').read_text();base=(folder/'backend_portable_v5.py').read_text();tree=ast.parse(hybrid);cls=next(v for v in tree.body if isinstance(v,ast.ClassDef));constructor=next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='__init__')
  self.assertEqual(cls.bases[0].id,'PortableTrackerBackend');self.assertIn('config',[v.arg for v in constructor.args.kwonlyargs]);self.assertIn('config=config',hybrid)
  for text in (hybrid,base):
   self.assertNotIn('PhraseAssets',text);self.assertNotIn('initial_carrier',text)
  self.assertIn("standing.npz",base);self.assertIn('owned_symmetric_stand_v1',base)
 def test_portable_formula_unchanged_and_import_constructs_nothing(self):
  before=set(sys.modules)
  from pulsemorph.agentic.embodied.backend_portable_hybrid_v5 import applied_targets
  from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import applied_targets as old
  self.assertFalse(any(n not in before and (n=='mujoco' or n.startswith('mujoco.') or n=='onnxruntime' or n.startswith('onnxruntime.')) for n in sys.modules))
  args=(np.linspace(-.2,.2,29,dtype=np.float32),np.linspace(-.1,.1,29),np.linspace(-.3,.3,29),np.linspace(-1,1,29),np.full(29,10.),np.full(29,2.),list(range(15,29)))
  np.testing.assert_array_equal(applied_targets(*args),old(*args))
 def test_portable_helper_imports_no_native_and_rejects_unapproved_authority(self):
  path=ROOT/'experiments/agentic/embodied/pilot_core_v2_portable_v1.py';spec=importlib.util.spec_from_file_location('portable_pilot_fixture',path);module=importlib.util.module_from_spec(spec);before=set(sys.modules);spec.loader.exec_module(module)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan.json';a=Path(d)/'authority.json';p.write_text('{}');a.write_text(json.dumps({'schema':'pulsemorph.portable_core_native_pilot_authority.v1','approved':False,'prepared_sha256':module.sha(p),'session_id':'fixture','budgets':module.BUDGETS}))
   with self.assertRaises(ValueError):module.execute(p,a,module.sha(a))
  self.assertFalse(any(n not in before and (n=='mujoco' or n.startswith('mujoco.') or n=='onnxruntime' or n.startswith('onnxruntime.')) for n in sys.modules))
if __name__=='__main__':unittest.main()
