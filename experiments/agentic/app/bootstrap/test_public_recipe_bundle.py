"""Read-only package-closure check; no downloads or environment mutation."""
import hashlib
import json
from pathlib import Path
import re
import unittest
from urllib.parse import urlsplit

class PublicRecipeBundleTests(unittest.TestCase):
    def test_exact_self_contained_public_recipe_and_lock(self):
        directory=Path(__file__).resolve().parent
        path=directory/'public_wheel_recipe.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),"7fb9f48647dfae52d9f0335820b3cc424423282d1bcc376271f5fd76d1ddaf64")
        recipe=json.loads(path.read_text());self.assertEqual(len(recipe['wheels']),30)
        self.assertFalse(recipe['unresolved']);self.assertEqual(sum(r['bytes'] for r in recipe['wheels']),344153316)
        lock=(directory/'requirements-win-cp311-public.lock').read_text()
        for row in recipe['wheels']:
            parsed=urlsplit(row['url']);self.assertEqual((parsed.scheme,parsed.hostname),('https','files.pythonhosted.org'))
            self.assertFalse(parsed.username or parsed.password or parsed.query)
            self.assertTrue(re.fullmatch('[0-9a-f]{64}',row['sha256']))
            self.assertIn(row['url']+' --hash=sha256:'+row['sha256'],lock)

if __name__=='__main__':unittest.main()
