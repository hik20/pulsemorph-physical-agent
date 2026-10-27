"""Read-only local prerequisites including hash-gated trusted motion deserialization.

No login, network, inference, simulation or install. Only pinned bundled motion
pickles are loaded in fresh isolated children; uploads are never accepted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import os
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
METERLESS_COMMIT = "1742dec56a66e58eef9dda2d2b865823758ca8cd"
PROTO_COMMIT = "f69fdcaabff31846b4d2247207417b57dc788828"
ONNX_SHA256 = "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def command(args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=15)
        return result.returncode == 0, result.stdout.strip().splitlines()[:1]
    except (OSError, subprocess.TimeoutExpired):
        return False, []


TRUSTED_MOTION_PINS = {'wave': 'cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93', 'high5': 'ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44'}
MOTION_CLASS_SOURCE_PINS = {'protomotions/__init__.py': '215ddbf6868e9126b1d3d3f352b5905d6117fde3e6c6de741e15d68677282d57', 'protomotions/simulator/__init__.py': '9d5c459ee656dcf65855629b281b995fbe9e594817e961969629c032fcfe8139', 'protomotions/simulator/base_simulator/__init__.py': '5cb3b4f448bc834954ee1625cca0af86f4212c505ee42070138bf63ea2bdc566', 'protomotions/simulator/base_simulator/simulator_state.py': '31cf14fab319151602f43fcf6ad2751b71088cda3a3f4feefadfa61262102731', 'protomotions/utils/__init__.py': '5cb3b4f448bc834954ee1625cca0af86f4212c505ee42070138bf63ea2bdc566', 'protomotions/utils/rotations.py': 'ef1475d33d6f941ccdea4a612477483db4fad8236ba7786fbe8bf91a65c1cef8', 'protomotions/utils/motion_interpolation_utils.py': '5de11616661e0dd70e858b3bb009ba1c5fd10435cd36ae6938d728070b147f9e'}

def validate_trusted_motion(root, identity, *, pins=None, source_pins=None, loader=None):
    """Deserialize ONLY exact pinned bundled examples, after hash/source checks.

    Internal dependency injection is for unit fixtures; CLI never accepts pins,
    arbitrary filenames or uploads. This loads trusted pickle data, not inference.
    """
    import numpy as np
    root=Path(root).resolve(); vendor=root/'research/vendor/ProtoMotions'
    pins=TRUSTED_MOTION_PINS if pins is None else pins
    source_pins=MOTION_CLASS_SOURCE_PINS if source_pins is None else source_pins
    if identity not in pins:raise ValueError('unsupported trusted source identity')
    motion=vendor/f'data/g1-kimodo-generated/proto/output_{identity}.motion'
    if motion.is_symlink() or not motion.is_file() or motion.resolve().parent != (vendor/'data/g1-kimodo-generated/proto').resolve():
        raise ValueError('trusted motion missing or linked outside pinned directory')
    from io import BytesIO
    if motion.stat().st_size>1024*1024:raise ValueError('trusted motion exceeds bounded payload size')
    payload=motion.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=pins[identity]:raise ValueError('trusted motion hash mismatch before deserialization')
    source_hashes={}
    for relative,expected in source_pins.items():
        path=vendor/relative
        if not path.is_file():raise ModuleNotFoundError('Missing pinned serialization/interpolation source: '+relative)
        source_bytes=path.read_bytes()
        raw_sha=hashlib.sha256(source_bytes).hexdigest()
        canonical_sha=hashlib.sha256(source_bytes.replace(b'\r\n',b'\n')).hexdigest()
        if path.is_symlink() or canonical_sha!=expected:raise ValueError('pinned serialization/interpolation source hash mismatch: '+relative)
        source_hashes[relative]={'observed_raw_sha256':raw_sha,'canonical_LF_sha256':canonical_sha,'pinned_git_blob_sha256':expected,'normalization':'Python source CRLF-to-LF only; binary motion unchanged'}
    # An isolated child has no preloaded project packages; never add original D root.
    sys.path.insert(0,str(vendor))
    if loader is None:
        import torch
        import protomotions.utils.motion_interpolation_utils
        loader=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    raw=loader(BytesIO(payload))
    if not isinstance(raw,dict):raise ValueError('trusted motion payload must be a dictionary')
    conversion=raw.get('state_conversion')
    if (type(conversion).__name__!='StateConversion' or type(conversion).__module__!='protomotions.simulator.base_simulator.simulator_state' or getattr(conversion,'name',None)!='COMMON'):
        raise ValueError('trusted motion requires the pinned COMMON StateConversion enum')
    fps=raw.get('fps')
    if isinstance(fps,bool) or not isinstance(fps,(int,float)) or fps!=30:raise ValueError('trusted motion expected30fps')
    shapes={'dof_pos':(29,), 'dof_vel':(29,), 'rigid_body_pos':(33,3), 'rigid_body_rot':(33,4), 'rigid_body_vel':(33,3), 'rigid_body_ang_vel':(33,3), 'rigid_body_contacts':(33,)}
    frames=None;fields={}
    for key,tail in shapes.items():
        value=raw.get(key)
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        value=np.asarray(value)
        if value.ndim!=len(tail)+1 or value.shape[1:]!=tail or value.dtype.kind not in 'biuf' or not np.isfinite(value).all():raise ValueError('invalid trusted motion field: '+key)
        if frames is not None and len(value)!=frames:raise ValueError('trusted motion frame count mismatch')
        frames=len(value);fields[key]={'shape':list(value.shape),'dtype':str(value.dtype)}
    if (frames-1)/fps<4.14:raise ValueError('trusted motion lacks4.14s future coverage')
    origins={}
    for name,module in list(sys.modules.items()):
        if name=='protomotions' or name.startswith('protomotions.'):
            file=getattr(module,'__file__',None)
            if file:
                resolved=Path(file).resolve()
                if not resolved.is_relative_to(vendor.resolve()):raise ValueError('trusted motion vendor import outside selected root')
                origins[name]=str(resolved)
    return {'source_id':identity,'sha256':pins[identity],'frames':frames,'fps':fps,'coverage_s':(frames-1)/fps,'fields':fields,'vendor_import_origins':origins,'source_code_hashes':source_hashes,'scope':'trusted bundled pickle + finite source arrays and serialization/interpolation import closure only; no ONNX/inference/physics/render','inference_calls':0,'physics_steps':0}


def trusted_motion_probe(root, identity):
    """Fresh isolated bounded process, using this script and its own selected root."""
    script=Path(root)/'experiments/agentic/app/preflight.py'
    env=dict(os.environ)
    for key in list(env):
        if key in ('PYTHONPATH','PYTHONHOME') or key.startswith(('ZETARIS_','OPENAI_')):env.pop(key)
    env.update(PYTHONDONTWRITEBYTECODE='1',GIT_NO_LAZY_FETCH='1',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull)
    try:
        result=subprocess.run([sys.executable,'-I','-B',str(script),'--trusted-motion-child',identity],cwd=root,env=env,capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        detail=json.loads(result.stdout)
        return result.returncode==0 and detail.get('ok') is True,detail
    except (OSError,subprocess.TimeoutExpired,ValueError):
        return False,{'ok':False,'source_id':identity,'error':'Trusted source dependency probe failed or timed out; supply pinned serialization/interpolation sources','scope':'no inference/physics'}


def raw_profile_asset_checks(root):
    """Use export's own literal raw pins; never normalize executable profiles."""
    root=Path(root)
    source=root/'src/pulsemorph/agentic/app/exports.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('BODY_HASH','CONTROLLER_HASH'):
                    constants[target.id]=ast.literal_eval(node.value)
    paths={'BODY_HASH':'protomotions/data/assets/mjcf/g1_holo_compat.xml','CONTROLLER_HASH':'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml'}
    if set(constants)!=set(paths):raise ValueError('Export raw profile pins unavailable')
    rows=[]
    for key,relative in paths.items():
        expected=constants[key]
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('Export raw profile pin invalid')
        path=root/'research/vendor/ProtoMotions'/relative
        observed=digest(path) if path.is_file() and not path.is_symlink() else None
        rows.append({'name':'raw_profile_asset:'+key,'ok':observed==expected,'detail':{'relative_path':'research/vendor/ProtoMotions/'+relative,'expected_raw_sha256':expected,'observed_raw_sha256':observed,'pin_source':'src/pulsemorph/agentic/app/exports.py:'+key,'normalization':'NONE; raw exact bytes required','action':'Use a fresh dedicated checkout of the pinned commit with Git for Windows core.autocrlf=true for these tracked XML/YAML files, then verify these raw hashes. Do not rewrite existing evidence or normalize these profiles; Python-only LF source-pin normalization does not apply.'}})
    return rows


def probe(runtime_python=None):
    checks = []
    def add(name, ok, detail=None, critical=True):
        checks.append({"name": name, "ok": bool(ok), "critical": critical, "detail": detail})
    add("python", sys.version_info >= (3, 11), {"version": platform.python_version(), "executable": sys.executable})
    add("windows_runtime_layout", os.name == "nt", "Windows local Python executable; workspace default or explicitly validated isolated venv")
    selected_python = Path(runtime_python).resolve() if runtime_python is not None else ROOT / ".venv-motor/Scripts/python.exe"
    add("runtime_python", selected_python.is_file(), str(selected_python))
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from pulsemorph.agentic.app.runtime_config import make_runtime
        configured = make_runtime(ROOT, runtime_python)
        runtime_configuration = configured.runtime_configuration
        add("runtime_configuration", True, runtime_configuration)
    except Exception as exc:
        runtime_configuration = {"selection": "explicit_isolated" if runtime_python is not None else "default_workspace", "verified": False}
        add("runtime_configuration", False, {"error_type": type(exc).__name__, "message": "Selected runtime configuration could not be validated"})
    add("preflight_uses_runtime_python", Path(sys.executable).resolve() == selected_python.resolve(),
        {"current": sys.executable, "expected": str(selected_python),
         "instruction": "Run preflight using the selected interpreter itself; --runtime-python does not move these dependency probes to another process"})
    modules = [("numpy", "numpy"), ("scipy", "scipy"), ("torch", "torch"), ("onnxruntime", "onnxruntime"), ("mujoco", "mujoco"),
               ("jsonschema", "jsonschema"), ("imageio", "imageio"), ("PIL", "Pillow"), ("imageio_ffmpeg", "imageio-ffmpeg"), ("yaml", "PyYAML")]
    for module, package in modules:
        try:
            imported = importlib.import_module(module)
            version = importlib.metadata.version(package)
            add("python_module:" + module, True, {"version": version, "interpreter": "current preflight interpreter"})
        except Exception as exc:
            add("python_module:" + module, False, {"error_type": type(exc).__name__})
    for executable, args in [("node", ["--version"]), ("ffmpeg", ["-version"]), ("codex", ["--version"]), ("git", ["--version"])]:
        found = shutil.which("codex.cmd") or shutil.which("codex") if executable == "codex" else shutil.which(executable)
        ok, version = command([found, *args]) if found else (False, [])
        add("executable:" + executable, ok, {"found": bool(found), "version_line": version})
    for directory, commit, guarded in [("ProtoMotions", PROTO_COMMIT, "deployment"), ("Meterless", METERLESS_COMMIT, "engines/world-model/reference/src")]:
        vendor = ROOT / "research/vendor" / directory
        ok, actual = command(["git", "-C", str(vendor), "rev-parse", "HEAD"])
        add("vendor_commit:" + directory, ok and actual == [commit], {"expected": commit, "actual": actual[0] if actual else None})
        clean, _ = command(["git", "-C", str(vendor), "diff", "--exit-code", "HEAD", "--", guarded])
        add("vendor_source_clean:" + directory, clean, {"tracked_scope": guarded, "not_a_full_asset_license_or_checkout_audit": True})
    paths = ["src/pulsemorph/agentic/app/server.py", "experiments/agentic/app/execute_app.py", "experiments/agentic/app/meterless_bridge.ts",
             "experiments/agentic/evaluate_multifeature.py", "experiments/agentic/evaluate_trial.py",
             "research/vendor/Meterless/engines/world-model/reference/node_modules/tsx/dist/cli.mjs"]
    for relative in paths:
        add("file:" + relative, (ROOT / relative).is_file())
    proto = ROOT / "research/vendor/ProtoMotions"
    model = proto / "data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx"
    add("controller_onnx", model.is_file() and digest(model) == ONNX_SHA256, {"expected_sha256": ONNX_SHA256})
    for relative in ["data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml", "protomotions/data/assets/mjcf/g1_holo_compat.xml"]:
        add("proto_file:" + relative, (proto / relative).is_file())
    try:
        tree = ast.parse((ROOT / "src/pulsemorph/agentic/app/runtime.py").read_text(encoding="utf-8"))
        pins = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PINS" for target in node.targets))
        for identity, expected in pins.items():
            motion = proto / f"data/g1-kimodo-generated/proto/output_{identity}.motion"
            add("source_motion:" + identity, motion.is_file() and digest(motion) == expected, {"expected_sha256": expected, "pin_source": "src/pulsemorph/agentic/app/runtime.py:PINS"})
    except Exception as exc:
        add("runtime_source_pins", False, {"error_type": type(exc).__name__})
    try:
        xml = proto / "protomotions/data/assets/mjcf/g1_holo_compat.xml"
        doc = ET.parse(xml).getroot()
        mesh_root = xml.parent / doc.find("compiler").get("meshdir", "")
        meshes = [mesh_root / item.attrib["file"] for item in doc.findall("./asset/mesh")]
        add("g1_meshes", bool(meshes) and all(mesh.is_file() and mesh.stat().st_size > 200 for mesh in meshes), {"mesh_count": len(meshes), "scope": "presence and non-placeholder size; execution records exact hashes"})
    except Exception as exc:
        add("g1_meshes", False, {"error_type": type(exc).__name__})
    try:
        for check in raw_profile_asset_checks(ROOT):
            add(check['name'],check['ok'],check['detail'])
    except Exception as exc:
        add('raw_profile_asset_pins',False,{'error_type':type(exc).__name__,'action':'Restore current own exports.py literal raw profile pins; no profile normalization permitted'})
    for identity in TRUSTED_MOTION_PINS:
        ok,detail=trusted_motion_probe(ROOT,identity)
        add('trusted_motion_payload:'+identity,ok,detail)
    free = shutil.disk_usage(ROOT).free
    add("free_disk", free >= 256 * 1024 * 1024, {"free_bytes": free, "runtime_minimum_bytes": 256 * 1024 * 1024, "note": "minimum per operation; accumulated job artifacts require more space"})
    sys.path.insert(0, str(ROOT / "src"))
    from pulsemorph.agentic.app.providers import CodexLocalProvider
    from pulsemorph.agentic.app.integrations import AppIntegrations
    provider = CodexLocalProvider(trusted_local=False).status()
    integration = AppIntegrations(ROOT).status()
    add("codex_authentication", False, "Not probed. CLI presence/version is configuration only; user login and actual model call are required.", critical=False)
    add("zetaris_optional_access", not integration["zetaris"]["missing"] and os.environ.get("ZETARIS_APP_TRANSPORT") == "ui_proxy_experimental", integration["zetaris"], critical=False)
    return {"schema": "pulsemorph.agentic_app_preflight.v1", "root": str(ROOT), "read_only": True,
            "no_network_or_authentication_probe": True, "checks": checks, "provider": provider,
            "runtime_configuration": runtime_configuration,
            "ok": all(check["ok"] for check in checks if check["critical"]),
            "not_verified": ["Codex authentication/model access", "Zetaris query access", "offscreen render/display availability", "fresh external installation", "public redistribution rights"],
            "competition_note": "Optional runtime Zetaris access remains an open competition integration requirement until a real authorized context query is evidenced."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-motion-child",choices=("wave","high5"),help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--runtime-python", type=Path, help="Expected explicitly isolated Windows Python 3.11 interpreter; execute preflight with the same interpreter")
    args = parser.parse_args()
    if args.trusted_motion_child:
        try:
            detail=validate_trusted_motion(ROOT,args.trusted_motion_child)
            print(json.dumps({'ok':True,**detail}));return 0
        except Exception as exc:
            print(json.dumps({'ok':False,'source_id':args.trusted_motion_child,'error_type':type(exc).__name__,'error':str(exc),'action':'Supply exact pinned ProtoMotions serialization/interpolation sources; never substitute source payloads','inference_calls':0,'physics_steps':0}));return 1
    report = probe(args.runtime_python)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        failed = [check["name"] for check in report["checks"] if check["critical"] and not check["ok"]]
        print("App prerequisites: " + ("PASS" if report["ok"] else "MISSING"))
        print(f"Python {platform.python_version()} | {len(report['checks'])} checks | critical missing: {', '.join(failed) or 'none'}")
        print("Codex: CLI configuration only; authentication/model access not verified. Launch requires private local trust.")
        print("Zetaris: " + next(check["detail"]["state"] for check in report["checks"] if check["name"] == "zetaris_optional_access"))
        print("No network, jobs, installs, authentication stores, or credentials were accessed.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
