"""Runtime-native admission for the explicitly configured isolated interpreter."""
import sys,site,importlib.metadata
from pathlib import Path
from .config_v5 import digest,VERSIONS
def environment(config):
    import numpy as np,mujoco,onnxruntime,numpy.core._multiarray_umath as nb
    import onnxruntime.capi.onnxruntime_pybind11_state as ob
    import mujoco._functions as mf,mujoco._structs as ms
    python=Path(sys.executable).resolve()
    if python!=config['runtime_python'] or not sys.flags.isolated or site.ENABLE_USER_SITE is not False:raise RuntimeError('Configured isolated runtime required')
    versions={n:importlib.metadata.version(n) for n in VERSIONS}
    if versions!=VERSIONS:raise RuntimeError('Admitted versions differ')
    prefixes={Path(sys.prefix).resolve()};pins={}
    for module in (np,mujoco,onnxruntime,nb,ob,mf,ms):
        path=Path(module.__file__).resolve()
        if not any(path.is_relative_to(p) for p in prefixes):raise RuntimeError('Module outside configured environment')
        pins[str(path)]=digest(path)
    for path in Path(mujoco.__file__).parent.glob('mujoco*.dll'):pins[str(path.resolve())]=digest(path)
    if sys.version_info[:2]!=(3,11):raise RuntimeError('CPython3.11 required')
    return {'versions':versions,'python':sys.version,'python_sha256':digest(python),'isolated':True,'user_site':False,'module_pins':pins}
