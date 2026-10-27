"""Read-only exact identity bridge gate. Never launches physics or a model."""
import hashlib
import json
from pathlib import Path
import numpy as np

RAW_REQUIRED = ("qpos.npy", "qvel.npy", "state_times.npy", "applied_joint_targets.npy")
CONTAINERS = ("policy_inputs.npz", "policy_outputs.npz", "reference_cache.npz", "common_reference/arrays.npz")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_current_assets(root, old_manifest):
    root = Path(root).resolve()
    assets = {name: digest for name, digest in old_manifest["preexecution_input_hashes"].items()
              if name.replace("\\", "/").startswith("research/vendor/")}
    if not assets: raise ValueError("Baseline lacks source/controller/body asset pins")
    for name, digest in assets.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root / "research/vendor") or sha(path) != digest:
            raise ValueError("Current controlled source/controller/body asset differs: " + name)
    return assets


def compare_numeric_arrays(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    results = {"raw_files": {}, "container_keys": {}}
    for name in RAW_REQUIRED + ("actions.npy",):
        left, right = baseline / name, candidate / name
        if name == "actions.npy" and not left.exists() and not right.exists(): continue
        if not left.is_file() or not right.is_file() or sha(left) != sha(right):
            raise ValueError("Identity bridge raw bytes differ or missing: " + name)
        results["raw_files"][name] = sha(left)
    for name in CONTAINERS:
        with np.load(baseline / name, allow_pickle=False) as before, np.load(candidate / name, allow_pickle=False) as after:
            if set(before.files) != set(after.files): raise ValueError("Identity bridge container keys differ: " + name)
            keys = {}
            for key in sorted(before.files):
                a, b = before[key], after[key]
                if a.shape != b.shape or a.dtype != b.dtype or a.tobytes(order="C") != b.tobytes(order="C"):
                    raise ValueError("Identity bridge numeric array differs: " + name + "/" + key)
                keys[key] = {"shape": list(a.shape), "dtype": a.dtype.str, "bytes_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}
            results["container_keys"][name] = keys
    return results


def _inventory(directory, manifest):
    directory = Path(directory).resolve()
    expected = manifest["artifact_hashes"]
    # Nested common_reference/manifest.json is a tracked artifact too.
    actual = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*") if path.is_file() and path != directory / "manifest.json"}
    normalized = {name.replace("\\", "/"): digest for name, digest in expected.items()}
    if set(actual) != set(normalized): raise ValueError("Identity bridge artifact inventory differs")
    for name, path in actual.items():
        if path.is_symlink() or directory not in path.resolve().parents or sha(path) != normalized[name]:
            raise ValueError("Identity bridge artifact hash differs: " + name)


def gate_identity(root, baseline, candidate, *, binding):
    """Protocol-bound identity admission, including fresh candidate integrity audit.

    binding contains the frozen baseline record, exact candidate identity/source
    and effective protocol SHA. It may not be inferred from candidate claims.
    """
    root, baseline, candidate = Path(root), Path(baseline), Path(candidate)
    old, new = [json.loads((directory / "manifest.json").read_text()) for directory in (baseline, candidate)]
    frozen = binding["baseline"]
    if baseline.resolve() != (root / frozen["directory"]).resolve() or sha(baseline / "manifest.json") != frozen["manifest_sha256"]:
        raise ValueError("Identity baseline differs from frozen protocol binding")
    state_path = baseline.parents[1] / "state.json"
    report_path = root / frozen["independent_audit_receipt"]
    if sha(state_path) != frozen["state_sha256"] or sha(report_path) != frozen["independent_audit_sha256"]:
        raise ValueError("Identity baseline state/independent-audit binding changed")
    state = json.loads(state_path.read_text()); report = json.loads(report_path.read_text())
    audited = next((row for row in report["jobs"] if row["id"] == state["id"]), None)
    if not audited or not audited["passed"] or audited["state_sha256"] != frozen["state_sha256"] or not any(row["trial"] == "baseline" and row["state"] == "independently_verified" for row in audited["physics"]):
        raise ValueError("Identity baseline independent audit not established")
    for directory, manifest in ((baseline, old), (candidate, new)): _inventory(directory, manifest)
    proposal_path = candidate / "proposal.json"
    proposal = json.loads(proposal_path.read_text())
    expected_parameters = {name: 1. for name in ("left_gain_a", "right_gain_a", "left_gain_b", "right_gain_b")}
    expected_parameters["timing_shift_s"] = 0.
    if (new["schema"] != "pulsemorph.wide_clock_probe_trial.v1" or proposal["schema"] != "pulsemorph.wide_clock_probe_proposal.v1"
            or new["candidate_id"] != binding["candidate_id"] or proposal["candidate_id"] != binding["candidate_id"]
            or new["source_id"] != binding["source_id"] or proposal["source_id"] != binding["source_id"]
            or proposal["control_tool"] != "wide_global_clock_0p6_v1" or new["control_tool"] != "wide_global_clock_0p6_v1"
            or proposal["parameters"] != expected_parameters or proposal["protocol_sha256"] != binding["protocol_sha256"]
            or new["proposal_sha256"] != sha(proposal_path) or new["producer"]["id"] != "pulsemorph-wide-clock-probe-v1"):
        raise ValueError("Candidate is not the exact frozen wide-clock identity proposal")
    if old["status"] != "success" or new["status"] != "success": raise ValueError("Identity bridge requires independently audited successful traces")
    if old["versions"] != new["versions"]: raise ValueError("Identity bridge critical versions differ")
    for field in ("onnx_sha256", "body_profile_sha256", "controller_profile_sha256"):
        if old[field] != new[field]: raise ValueError("Identity bridge controller/body pin differs: " + field)
    if old["source_motion"]["sha256"] != new["source_motion"]["sha256"]: raise ValueError("Identity bridge source differs")
    old_assets, new_assets = verify_current_assets(root, old), verify_current_assets(root, new)
    if old_assets != new_assets: raise ValueError("Identity bridge controlled asset inventory differs")
    if new["audio_observation"] != {"available": False, "duration_s": 0., "observed_through_sample": None}:
        raise ValueError("Prospective identity must declare no audio")
    from pulsemorph.agentic.app import runtime
    module = runtime._module(root, "evaluate_multifeature")
    audited_new = runtime._audit_app(module, module._audit, candidate)
    if audited_new["strict_common_control_rows"] != 200: raise ValueError("Fresh identity integrity/common audit incomplete")
    result = compare_numeric_arrays(baseline, candidate)
    result.update(passed=True, absolute_tolerance=0, relative_tolerance=0,
                  fresh_candidate_independent_audit={"strict_common_rows": audited_new["strict_common_control_rows"], "fresh_anchor_audit": audited_new["fresh_anchor_audit"]},
                  old_audio_observation=old["audio_observation"], new_audio_observation=new["audio_observation"],
                  qualification="Cross-environment/wrapper/class comparison. Matching version strings were not assumed sufficient; actual numerical bytes matched. Audio/provenance metadata differ explicitly; no numeric feed exemption.")
    return result
