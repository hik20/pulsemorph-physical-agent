"""Interpreter selection boundary fixtures; no model or physical execution."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pulsemorph.agentic.app.runtime_config import make_runtime, RuntimeConfigurationError


class RuntimeConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.python = self.root / 'env/Scripts/python.exe'
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b'fixture executable; never executed')
        self.proof = {'python_executable': str(self.python.resolve()), 'version': '3.11.4',
            'version_info': [3, 11, 4], 'prefix': str(self.root / 'env'),
            'base_prefix': str(self.root / 'base'), 'user_site_enabled': False,
            'probe_isolated_flag': True, 'venv_config_exists': True,
            'venv_include_system_site_packages': 'false'}

    def tearDown(self):
        self.temp.cleanup()

    def response(self, proof=None):
        def invoke(command, **kwargs):
            value = copy.deepcopy(self.proof if proof is None else proof)
            value['probe_isolated_flag'] = '-I' in command
            return SimpleNamespace(returncode=0, stdout=json.dumps(value))
        return invoke

    def test_default_preserves_workspace_interpreter_without_isolation_claim(self):
        with patch('subprocess.run') as run:
            runtime = make_runtime(self.root)
        run.assert_not_called()
        self.assertEqual(runtime.python, self.root / '.venv-motor/Scripts/python.exe')
        self.assertEqual(runtime.runtime_configuration['selection'], 'default_workspace')
        self.assertIsNone(runtime.runtime_configuration['user_site_enabled'])

    def test_missing_or_remote_path_rejected_without_execution(self):
        for value in [self.root / 'missing.exe', 'https://example.com/python.exe', r'\\remote\share\python.exe']:
            with self.subTest(value=value), patch('subprocess.run') as run:
                with self.assertRaises(RuntimeConfigurationError): make_runtime(self.root, value)
                run.assert_not_called()

    def test_valid_explicit_records_both_probe_modes_and_scrubs_environment(self):
        with patch('subprocess.run', side_effect=self.response()) as run:
            runtime = make_runtime(self.root, self.python)
        self.assertEqual(runtime.python, self.python.resolve())
        self.assertEqual(run.call_count, 2)
        proof = runtime.runtime_configuration
        self.assertTrue(proof['probe_isolated_flag'])
        self.assertFalse(proof['normal_mode_probe_isolated_flag'])
        self.assertFalse(proof['normal_mode_user_site_enabled'])
        for call in run.call_args_list:
            self.assertNotIn('PYTHONHOME', call.kwargs['env'])
            self.assertNotIn('PYTHONPATH', call.kwargs['env'])
            self.assertNotIn('shell', call.kwargs)
            self.assertEqual(call.kwargs['timeout'], 10)

    def test_wrong_version_user_site_global_or_system_site_rejected(self):
        variants = [{'version': '3.12.1', 'version_info': [3, 12, 1]},
                    {'user_site_enabled': True}, {'prefix': self.proof['base_prefix']},
                    {'venv_include_system_site_packages': 'true'}]
        for change in variants:
            proof = dict(self.proof, **change)
            with self.subTest(change=change), patch('subprocess.run', side_effect=self.response(proof)):
                with self.assertRaises(RuntimeConfigurationError): make_runtime(self.root, self.python)

    def test_normal_mode_user_site_enabled_rejected_even_if_I_probe_passed(self):
        def invoke(command, **kwargs):
            proof = dict(self.proof, probe_isolated_flag='-I' in command,
                         user_site_enabled='-I' not in command)
            return SimpleNamespace(returncode=0, stdout=json.dumps(proof))
        with patch('subprocess.run', side_effect=invoke), self.assertRaises(RuntimeConfigurationError):
            make_runtime(self.root, self.python)

    def test_malformed_metadata_and_executable_mismatch_rejected(self):
        for proof in [{}, dict(self.proof, python_executable=str(self.root/'other.exe'))]:
            with self.subTest(proof=proof), patch('subprocess.run', side_effect=self.response(proof)):
                with self.assertRaises(RuntimeConfigurationError): make_runtime(self.root, self.python)
        with patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='not JSON')):
            with self.assertRaises(RuntimeConfigurationError): make_runtime(self.root, self.python)

    def test_probe_timeout_and_failure_are_sanitized(self):
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('fixture', 10)):
            with self.assertRaisesRegex(RuntimeConfigurationError, 'could not be validated'):
                make_runtime(self.root, self.python)
        with patch('subprocess.run', return_value=SimpleNamespace(returncode=1, stdout='fixture private diagnostic')):
            with self.assertRaisesRegex(RuntimeConfigurationError, '^Selected Python probe failed$'):
                make_runtime(self.root, self.python)
