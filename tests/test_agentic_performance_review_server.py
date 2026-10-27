"""Pure range/role admission. No HTTP listener, model, FK or physics."""
import ast
from pathlib import Path
import unittest
from pulsemorph.agentic.performance import review_server_v1 as server

class PerformanceReviewTests(unittest.TestCase):
    def test_full_prefix_suffix_and_clamped_end_ranges(self):
        self.assertEqual(server.byte_range(None,100),(0,99,200))
        self.assertEqual(server.byte_range('bytes=0-9',100),(0,9,206))
        self.assertEqual(server.byte_range('bytes=50-',100),(50,99,206))
        self.assertEqual(server.byte_range('bytes=-12',100),(88,99,206))
        self.assertEqual(server.byte_range('bytes=90-999',100),(90,99,206))
    def test_invalid_ranges_fail_not_full_file_fallback(self):
        for header in ('bytes=-','bytes=-0','bytes=100-','bytes=10-1','bytes=0-1,2-3','wrong'):
            with self.subTest(header=header),self.assertRaises(ValueError):server.byte_range(header,100)
    def test_server_imports_only_standard_library_no_worker(self):
        tree=ast.parse(Path(server.__file__).read_text());modules=[]
        for node in tree.body:
            if isinstance(node,ast.Import):modules.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):modules.append(node.module)
        self.assertFalse(any(any(token in m for token in ('mujoco','onnx','provider','backend','worker','runtime')) for m in modules))
        self.assertIn('do_POST',server.Handler.__dict__)
if __name__=='__main__':unittest.main()
