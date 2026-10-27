"""Pinned six-trial runner. Freeze/review first; execution requires root receipt."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import site
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT / "research/vendor/ProtoMotions")]
PLAN = ROOT / "artifacts/runs/agentic/app_wide_clock_probe_plan_v1"
PROTOCOL_PATH = PLAN / "protocol_review_v2.json"
PROTOCOL_SHA = "fef437557a23985e797731f4421d2d15566d4ce3cf72e3ae40146c5cc69d29a2"
ROOT_REVIEW_SHA = "a4ed616645c3362c0244570865d164700d96f9fa24c051763e2568a1b92904c6"
SUPPLEMENT = PLAN / "runner_supplement_v1.json"
OUTPUT = ROOT / "artifacts/runs/agentic/app_wide_clock_probe_v1"
PYTHON = Path("C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe")
EXPECTED_SCHEDULE = [("wave", 0.), ("high5", 0.), ("wave", -.6), ("wave", .6), ("high5", -.6), ("high5", .6)]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream: json.dump(value, stream, indent=2, allow_nan=False)


def validate_schedule(protocol):
    rows = protocol["prospective_schedule"]
    if protocol["maximum_future_fresh_trials"] != 6 or len(rows) != 6: raise ValueError("Exactly the reviewed six-call budget is required")
    for index, (row, expected) in enumerate(zip(rows, EXPECTED_SCHEDULE)):
        if (row["index"] != index or (row["source"], row["shift_s"]) != expected or row["gains"] != [1, 1, 1, 1]
                or row["tool"] != "wide_global_clock_0p6_v1" or row["candidate_id"] != f"app_wide_clock_probe_v1__{index:02d}_{expected[0]}"):
            raise ValueError("Reviewed candidate order/source/identity/gain/shift changed")
    if protocol["audio"] != {"available": False, "duration_s": 0., "observed_through_sample": None}: raise ValueError("No-audio scope changed")


def verify_environment(protocol):
    import numpy, scipy, torch, mujoco, onnxruntime
    if Path(sys.executable).resolve() != PYTHON.resolve() or not sys.flags.isolated or site.ENABLE_USER_SITE is not False:
        raise ValueError("Exact public production Python under-I with user site disabled required")
    if protocol["selected_python"] != str(PYTHON): raise ValueError("Reviewed interpreter path changed")
    actual = {"numpy": numpy.__version__, "torch": torch.__version__, "mujoco": mujoco.__version__, "onnxruntime": onnxruntime.__version__}
    if actual != protocol["versions"] or scipy.__version__ != protocol["scipy"]: raise ValueError("Critical numerical versions changed")
    for module in (numpy, scipy, torch, mujoco, onnxruntime):
        if not Path(module.__file__).resolve().is_relative_to(PYTHON.parent.parent / "Lib/site-packages"):
            raise ValueError("Numerical module is outside selected public environment")
    return {"executable": str(PYTHON), "isolated": True, "user_site_enabled": False, "versions": actual, "scipy": scipy.__version__}


def load_context():
    if sha(PROTOCOL_PATH) != PROTOCOL_SHA or sha(PLAN / "root_review.json") != ROOT_REVIEW_SHA: raise ValueError("Frozen design/root review changed")
    protocol = read(PROTOCOL_PATH); validate_schedule(protocol); verify_environment(protocol)
    for name, digest in protocol["source_code_hashes"].items():
        if sha(ROOT / name) != digest: raise ValueError("Frozen code changed: " + name)
    import numpy as np
    import torch
    from pulsemorph.agentic.app import runtime
    from pulsemorph.motion.g1 import profile_from_yaml
    import bridge_checks
    executor_path = HERE / "execute_wide_clock_v1.py"
    spec = importlib.util.spec_from_file_location("pinned_wide_executor", executor_path)
    executor = importlib.util.module_from_spec(spec); spec.loader.exec_module(executor)
    raw = {}
    for source in ("wave", "high5"):
        baseline = protocol["baseline_bridges"][source]
        if sha(ROOT / baseline["directory"] / "manifest.json") != baseline["manifest_sha256"]: raise ValueError("Archived baseline manifest changed")
        bridge_checks.verify_current_assets(ROOT, read(ROOT / baseline["directory"] / "manifest.json"))
        path = f"research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_{source}.motion"
        source_sha = protocol["baseline_bridges"][source]["current_source_controller_body_mesh_assets_verified"][path.replace("/", "\\")] if path.replace("/", "\\") in baseline["current_source_controller_body_mesh_assets_verified"] else baseline["current_source_controller_body_mesh_assets_verified"][path]
        source_path, _ = executor.resolve_source_motion({"source_motion": {"path": path, "sha256": source_sha}})
        raw[source] = torch.load(source_path, map_location="cpu", weights_only=False)
    # Import the actual numerical entry point and independent audit definitions
    # now so freeze records their loaded transitive source/native module bytes.
    importlib.import_module("deployment.test_tracker_mujoco")
    audit_module = runtime._module(ROOT, "evaluate_multifeature")
    return protocol, executor, raw, profile_from_yaml(), runtime, audit_module, bridge_checks


def collect_pins(protocol):
    paths = {ROOT / name for name in protocol["source_code_hashes"]}
    paths.update((PROTOCOL_PATH, PLAN / "root_review.json", Path(__file__), HERE / "runner_tests.py"))
    for baseline in protocol["baseline_bridges"].values():
        paths.update(ROOT / name for name in baseline["old_preexecution_source_pins"])
        paths.update((ROOT / baseline["directory"] / "manifest.json", ROOT / baseline["independent_audit_receipt"]))
        paths.add((ROOT / baseline["directory"]).parents[1] / "state.json")
    paths.update((ROOT / "src/pulsemorph/motion").glob("*.py"))
    # Evaluator dynamic imports and the executor's provenance-only snapshots.
    for name in ("experiments/music_control/accents.py", "experiments/music_control/evaluate.py"):
        paths.add(ROOT / name)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename: continue
        path = Path(filename)
        if path.suffix == ".pyc":
            try: path = Path(importlib.util.source_from_cache(str(path)))
            except ValueError: continue
        path = path.resolve()
        if path.is_file() and (path.is_relative_to(ROOT) or path.is_relative_to(PYTHON.parent.parent)):
            paths.add(path)
    return {str(path.resolve()): sha(path) for path in sorted(paths)}


def verify_pins(protocol, supplement):
    verify_environment(protocol)
    if sha(PROTOCOL_PATH) != PROTOCOL_SHA or sha(PLAN / "root_review.json") != ROOT_REVIEW_SHA: raise ValueError("Frozen review/protocol changed")
    if sha(Path(__file__)) != supplement["runner_sha256"]: raise ValueError("Runner changed after review")
    for name, digest in supplement["file_sha256"].items():
        if sha(name) != digest: raise ValueError("Pinned transitive/source/controller input changed: " + name)
    if shutil.disk_usage(ROOT).free < 256 * 1024 * 1024: raise ValueError("At least256MiB free before each call required")


def preflight(entry, executor, raw, profile):
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    from pulsemorph.agentic.common_player import materialize_reference
    from pulsemorph.motion.g1 import assemble_reference_inputs
    proposal = make_proposal(entry)
    generator = executor.build_generator(proposal, raw[entry["source"]], joint_names=profile["joint_names"], body_names=profile["body_names"], source_start_s=0., control_steps=200,
        section_windows_s=[[.4, 2.], [2., 3.6]], source_provenance={"path": proposal["source_motion"]["path"], "sha256": proposal["source_motion"]["sha256"], "kind": "reviewed bounded wide-clock probe"})
    reference = materialize_reference(generator, profile=profile, candidate_id=entry["candidate_id"], observation_id="wide:noaudio", task_id="app_wide_clock_probe_v1", runtime_session_id=entry["candidate_id"],
        source_path=proposal["source_motion"]["path"], source_sha256=proposal["source_motion"]["sha256"], audio_observed_through_sample=None, audio_duration_s=0., producer_kind="agentic")
    reference = executor.compatible_metadata(reference, executor.proposal_producer(proposal))
    for index in range(200): assemble_reference_inputs(reference, index * .02)
    curve = CubicHermiteSpline(reference.time_offsets_s, reference.joint_position_rad, reference.joint_velocity_rad_s)
    roots = curve.derivative().roots(extrapolate=False)
    xml = ET.parse(profile["body_path"]).getroot()
    limits = {joint.attrib["name"]: [float(v) for v in joint.attrib["range"].split()] for joint in xml.findall(".//joint") if "name" in joint.attrib and "range" in joint.attrib}
    extrema = []
    for index, name in enumerate(profile["joint_names"]):
        if name not in limits: raise ValueError("Missing unchanged joint range")
        critical = roots[index][np.isfinite(roots[index]) & (roots[index] >= 0) & (roots[index] <= 4.14)]
        values = curve(np.unique(np.r_[reference.time_offsets_s, critical]))[:, index]
        if values.min() < limits[name][0] - 1e-8 or values.max() > limits[name][1] + 1e-8: raise ValueError("Common joint range failed: " + name)
        extrema.append({"joint": name, "minimum_rad": float(values.min()), "maximum_rad": float(values.max())})
    return {"passed": True, "strict_future_queries": 200, "continuous_joint_ranges": extrema, "new_physics_steps": 0}


def make_proposal(entry):
    pins = {"wave": "cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93", "high5": "ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44"}
    return {"schema": "pulsemorph.wide_clock_probe_proposal.v1", "candidate_id": entry["candidate_id"], "job_id": "app_wide_clock_probe_v1", "trial_index": entry["index"],
        "source_id": entry["source"], "control_tool": entry["tool"], "parameters": dict(left_gain_a=1., right_gain_a=1., left_gain_b=1., right_gain_b=1., timing_shift_s=entry["shift_s"]),
        "reason": "Reviewed fixed wide-clock identity/timing probe; no audio, model, goal or catalog admission", "duration_s": 4, "source_start_s": 0,
        "observation_id": "wide:noaudio", "task_id": "app_wide_clock_probe_v1", "runtime_session_id": entry["candidate_id"],
        "audio": {"available": False, "duration_s": 0., "observed_through_sample": None},
        "source_motion": {"path": f"research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_{entry['source']}.motion", "sha256": pins[entry["source"]]}, "protocol_sha256": PROTOCOL_SHA}


def run_schedule(protocol, output, *, verify, common, execute, audit, bridge):
    """Injectable state machine; default-free test callbacks never start physics."""
    validate_schedule(protocol)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)  # atomic no-retry/no-parallel lock
    attempts, results, passed_bridges = 0, [], set()
    entry = None
    try:
        for entry in protocol["prospective_schedule"]:
            if attempts >= 6: raise ValueError("Physical-call budget exhausted")
            if entry["index"] >= 2 and passed_bridges != {"wave", "high5"}: raise ValueError("Both identities must bridge before extremes")
            verify()
            save(output / "preflights" / f"{entry['index']:02d}.json", common(entry))
            verify()  # pin/space checks immediately before the actual subprocess
            save(output / "attempts" / f"{entry['index']:02d}.json", {"status": "started", "at": datetime.now(timezone.utc).isoformat(), "candidate": entry, "protocol_sha256": PROTOCOL_SHA})
            attempts += 1
            manifest = execute(entry, output)
            verify()  # also fail closed if any controlled byte changed during execution
            if manifest["status"] != "success": raise ValueError("Physical execution failed; no replacement")
            if manifest["actual_control_steps"] != 200 or manifest["actual_physics_steps"] != 4000 or manifest["online_onnx_call_count"] != 200:
                raise ValueError("Unexpected physical/policy counts")
            summary = audit(entry, output)
            if entry["index"] < 2:
                bridge_result = bridge(entry, output)
                if bridge_result.get("passed") is not True: raise ValueError("Identity bridge failed")
                save(output / "bridges" / (entry["source"] + ".json"), bridge_result)
                passed_bridges.add(entry["source"])
            save(output / "evaluations" / f"{entry['index']:02d}.json", summary)
            results.append(summary)
        result = {"status": "complete", "attempted_physical_calls": attempts, "completed_audited_trials": len(results), "known_successful_physics_steps": len(results) * 4000,
                  "identity_bridges": sorted(passed_bridges), "trials": results, "new_model_calls": 0, "audio_available": False}
    except Exception as exc:
        result = {"status": "stopped_no_replacement", "failed_candidate": entry, "error_type": type(exc).__name__, "error": str(exc), "attempted_physical_calls": attempts,
                  "completed_audited_trials": len(results), "known_successful_physics_steps": len(results) * 4000, "failed_or_unconfirmed_candidate_may_have_steps": attempts > len(results),
                  "identity_bridges": sorted(passed_bridges), "trials": results, "new_model_calls": 0, "audio_available": False}
    save(output / "result.json", result)
    return result


def freeze():
    if SUPPLEMENT.exists(): raise FileExistsError("Runner supplement already frozen; create a reviewed revision, never overwrite")
    protocol, executor, raw, profile, runtime, module, bridge = load_context()
    for entry in protocol["prospective_schedule"]: preflight(entry, executor, raw, profile)
    supplement = {"schema": "pulsemorph.wide_clock_runner_supplement.v1", "status": "FROZEN_AWAITING_ROOT_RUNNER_REVIEW_NO_PHYSICS_AUTHORIZED", "protocol_sha256": PROTOCOL_SHA,
        "root_design_review_sha256": ROOT_REVIEW_SHA, "runner_sha256": sha(Path(__file__)), "runner_tests_sha256": sha(HERE / "runner_tests.py"),
        "environment": verify_environment(protocol), "file_sha256": collect_pins(protocol), "maximum_physical_calls": 6, "new_physics_steps": 0, "new_model_calls": 0,
        "execution_approval_contract": {"decision": "APPROVE_PINNED_WIDE_CLOCK_RUNNER_EXECUTION", "required_fields": ["runner_sha256", "supplement_sha256", "protocol_sha256"], "issuer": "root technical review; design-only review is insufficient"}}
    save(SUPPLEMENT, supplement)
    print(json.dumps({"runner_sha256": supplement["runner_sha256"], "supplement_sha256": sha(SUPPLEMENT), "pinned_files": len(supplement["file_sha256"]), "new_physics": 0}))


def execute_reviewed(approval_path):
    if OUTPUT.exists(): raise FileExistsError("Existing trial output forbids retry/resume/parallel runs")
    supplement = read(SUPPLEMENT); approval = read(approval_path)
    if (approval.get("decision") != "APPROVE_PINNED_WIDE_CLOCK_RUNNER_EXECUTION" or approval.get("runner_sha256") != sha(Path(__file__))
            or approval.get("supplement_sha256") != sha(SUPPLEMENT) or approval.get("protocol_sha256") != PROTOCOL_SHA):
        raise ValueError("Exact root runner execution review is required; design approval alone is insufficient")
    protocol, executor, raw, profile, runtime, module, bridge_checks = load_context()
    verify_pins(protocol, supplement)
    import numpy as np
    def execute(entry, output):
        proposal_path = output / "proposals" / f"{entry['index']:02d}.json"; save(proposal_path, make_proposal(entry))
        directory = output / "trials" / f"{entry['index']:02d}_{entry['source']}"
        with (output / "attempts" / f"{entry['index']:02d}.runtime.log").open("xb") as log:
            proc = subprocess.run([str(PYTHON), "-I", str(HERE / "execute_wide_clock_v1.py"), "--proposal", str(proposal_path), "--output", str(directory)], cwd=ROOT, stdout=log, stderr=log, timeout=180)
        manifest = read(directory / "manifest.json")
        if proc.returncode: raise ValueError("Executor exited nonzero; manifest/log preserved")
        return manifest
    def audit(entry, output):
        directory = output / "trials" / f"{entry['index']:02d}_{entry['source']}"
        row = runtime._audit_app(module, module._audit, directory)
        if row["strict_common_control_rows"] != 200: raise ValueError("Full fresh common audit incomplete")
        arrays, names = row["arrays"], row["names"]; rms = []
        for start, end in ((.8, 1.6), (2.4, 3.2)):
            mask = (arrays["state_times"] >= start - 1e-10) & (arrays["state_times"] <= end + 1e-10)
            for side in ("left", "right"):
                group = [i for i, name in enumerate(names) if name.startswith(side + "_") and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))]
                rms.append(float(np.sqrt(np.mean((arrays["qpos"][mask, 7:][:, group] - arrays["qpos"][0, 7:][group]) ** 2))))
        base_index = 0 if entry["source"] == "wave" else 1
        base = rms if entry["index"] < 2 else measurements[base_index]["arm_rms_rad"]
        accent = module.accent_metrics(arrays["qvel"][:, 6:], arrays["state_times"], [1.8, 3.2])
        manifest = row["manifest"]
        summary = {"candidate": entry, "integrity_passed": True, "strict_common_rows": 200, "fresh_anchor_audit": row["fresh_anchor_audit"],
            "arm_rms_rad": rms, "arm_ratios_to_fresh_identity": [value / denominator if denominator > 0 else None for value, denominator in zip(rms, base)],
            "accent": {key: value for key, value in accent.items() if key not in ("rms_speed_rad_s", "smoothed_rms_speed_rad_s")},
            "min_root_height_m": manifest["all_substeps_min_root_height_m"], "mean_joint_rmse_rad": row["mean_joint_rmse_rad"],
            "physical_screen_passed": manifest["all_substeps_min_root_height_m"] >= .6 and row["mean_joint_rmse_rad"] <= .35,
            "physical_steps": 4000, "online_onnx_calls": 200, "no_intent_goal_pass_claim": True}
        measurements[entry["index"]] = summary
        return summary
    def bridge(entry, output):
        return bridge_checks.gate_identity(ROOT, ROOT / protocol["baseline_bridges"][entry["source"]]["directory"], output / "trials" / f"{entry['index']:02d}_{entry['source']}",
            binding={"baseline": protocol["baseline_bridges"][entry["source"]], "candidate_id": entry["candidate_id"], "source_id": entry["source"], "protocol_sha256": PROTOCOL_SHA})
    measurements = {}
    result = run_schedule(protocol, OUTPUT, verify=lambda: verify_pins(protocol, supplement), common=lambda entry: preflight(entry, executor, raw, profile), execute=execute, audit=audit, bridge=bridge)
    print(json.dumps({key: result[key] for key in ("status", "attempted_physical_calls", "completed_audited_trials")}))
    return 0 if result["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true"); modes.add_argument("--execute", action="store_true")
    parser.add_argument("--approval", type=Path); args = parser.parse_args()
    if args.freeze: freeze(); return 0
    if args.approval is None: raise ValueError("--approval root runner-review receipt required")
    return execute_reviewed(args.approval)


if __name__ == "__main__": raise SystemExit(main())
