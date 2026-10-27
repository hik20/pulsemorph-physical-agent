"""Explicit local interpreter selection without changing physical runtime mechanics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

from .runtime import AppRuntime

_PROBE = """import json,sys,site,pathlib
cfg=pathlib.Path(sys.prefix)/'pyvenv.cfg'
values={}
if cfg.is_file():
    for line in cfg.read_text(encoding='utf-8').splitlines():
        if '=' in line:
            key,value=line.split('=',1);values[key.strip().lower()]=value.strip().lower()
print(json.dumps({'python_executable':sys.executable,'version':'.'.join(map(str,sys.version_info[:3])),
    'version_info':list(sys.version_info[:3]),'prefix':sys.prefix,'base_prefix':sys.base_prefix,
    'user_site_enabled':site.ENABLE_USER_SITE,'probe_isolated_flag':bool(sys.flags.isolated),
    'venv_config_exists':cfg.is_file(),'venv_include_system_site_packages':values.get('include-system-site-packages')}))
"""


class RuntimeConfigurationError(ValueError):
    pass


def _probe(python: Path, *, isolated: bool) -> dict:
    environment = dict(os.environ)
    environment.pop('PYTHONHOME', None)
    environment.pop('PYTHONPATH', None)
    try:
        with tempfile.TemporaryDirectory(prefix='pulsemorph-python-probe-') as directory:
            result = subprocess.run([str(python), *(['-I'] if isolated else []), '-c', _PROBE],
                cwd=directory, env=environment, capture_output=True, text=True, encoding='utf-8',
                timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeConfigurationError('Selected Python probe failed')
        proof = json.loads(result.stdout)
        if (not isinstance(proof, dict) or not isinstance(proof.get('version_info'), list)
                or len(proof['version_info']) != 3
                or any(type(value) is not int for value in proof['version_info'])
                or proof['version_info'][:2] != [3, 11]
                or type(proof.get('user_site_enabled')) is not bool
                or type(proof.get('probe_isolated_flag')) is not bool
                or proof.get('version') != '.'.join(map(str, proof['version_info']))):
            raise RuntimeConfigurationError('Selected interpreter must report Python 3.11 and valid isolation metadata')
        if Path(proof['python_executable']).resolve() != python:
            raise RuntimeConfigurationError('Selected interpreter reported a different executable')
        if proof['probe_isolated_flag'] is not isolated:
            raise RuntimeConfigurationError('Selected interpreter isolation flag mismatch')
        if (proof['user_site_enabled'] is not False or proof.get('venv_config_exists') is not True
                or proof.get('venv_include_system_site_packages') != 'false'
                or Path(proof['prefix']).resolve() == Path(proof['base_prefix']).resolve()):
            raise RuntimeConfigurationError('Explicit runtime requires a venv with user/system site packages disabled')
        return proof
    except RuntimeConfigurationError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeConfigurationError('Selected Python probe could not be validated') from exc


def make_runtime(root, python_executable=None) -> AppRuntime:
    runtime = AppRuntime(root)
    if python_executable is None:
        runtime.runtime_configuration = {
            'selection': 'default_workspace', 'python_executable': str(runtime.python.resolve()),
            'version': None, 'user_site_enabled': None,
            'verification': 'legacy_workspace_default_not_isolation_validated',
        }
        return runtime
    raw_path = str(python_executable)
    if '://' in raw_path or raw_path.startswith(('\\\\', '//')):
        raise RuntimeConfigurationError('Explicit runtime must use a local executable path')
    python = Path(python_executable).expanduser().resolve()
    if not python.is_file() or python.suffix.lower() != '.exe':
        raise RuntimeConfigurationError('Explicit runtime Python must be an existing local .exe file')
    isolated = _probe(python, isolated=True)
    normal = _probe(python, isolated=False)
    if any(isolated[key] != normal[key] for key in ('python_executable', 'version', 'prefix', 'base_prefix')):
        raise RuntimeConfigurationError('Interpreter identity changed between isolation probes')
    runtime.python = python
    runtime.runtime_configuration = {
        **isolated, 'selection': 'explicit_isolated',
        'venv_include_system_site_packages': False,
        'venv_config_value': isolated['venv_include_system_site_packages'],
        'normal_mode_user_site_enabled': normal['user_site_enabled'],
        'normal_mode_probe_isolated_flag': normal['probe_isolated_flag'],
        'normal_probe_environment': 'PYTHONHOME/PYTHONPATH scrubbed; empty temporary cwd',
        'verification': 'isolated_and_normal_mode_venv_probes_passed',
    }
    return runtime
