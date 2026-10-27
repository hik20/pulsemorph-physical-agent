"""Versioned launch fix: pin and expose deployment helpers before composition.

The first enhanced launch's code and failed job are retained. The reference
compiler performs a lazy deployment.state_utils import in the parent process;
the worker's own import path cannot satisfy that parent dependency.
"""
from pathlib import Path
import importlib
import sys


def prepare_parent_imports(root):
    if any(name == 'deployment' or name.startswith('deployment.') for name in sys.modules):
        raise RuntimeError('Fresh parent deployment module cache required')
    from .assets import PhraseAssets, sha
    root = Path(root).resolve()
    assets = PhraseAssets(root)
    vendor = root / 'research/vendor/ProtoMotions'
    relative = 'research/vendor/ProtoMotions/deployment/state_utils.py'
    expected = vendor / 'deployment/state_utils.py'
    if assets.manifest['live_numeric_pins'].get(relative) != sha(expected):
        raise RuntimeError('Parent deployment helper is not the admitted source')
    sys.path.insert(0, str(vendor))
    helper = importlib.import_module('deployment.state_utils')
    if Path(helper.__file__).resolve() != expected.resolve() or sha(helper.__file__) != assets.manifest['live_numeric_pins'][relative]:
        raise RuntimeError('Parent deployment helper resolved outside the admitted source')
    return {'helper_path': str(expected), 'helper_sha256': sha(expected),
            'profile_manifest_sha256': sha(assets.directory / 'manifest.json'),
            'model_or_physics_execution': False}


def main():
    root = Path(__file__).resolve().parents[5]
    prepare_parent_imports(root)
    from .server import main as server_main
    server_main()


if __name__ == '__main__':
    main()
