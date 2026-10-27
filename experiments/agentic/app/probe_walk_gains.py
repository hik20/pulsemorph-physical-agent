"""Four preregistered Walk gain probes; reference/realized response diagnostic."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent), str(ROOT / "research/vendor/ProtoMotions")]
BASE = ROOT / "artifacts/runs/agentic/app_walk_gain_probe_v1"
PRIOR = ROOT / "artifacts/runs/agentic/app_walk_admission_v1/trials/identity_global"
NAMES = ["left_gain_a", "right_gain_a", "left_gain_b", "right_gain_b"]
SOURCE = "research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_walk.motion"
SOURCE_HASH = "09061b8fa0df9e99daaf857293b32cfa8d2faa68631ff20ed0fdde0683e80884"
PYTHON = Path("C:/Users/halke/.cache/pulsemorph-agentic/envs/app-cpu-py311-v1/Scripts/python.exe")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream: json.dump(value, stream, indent=2, allow_nan=False)


def common_preflight(executor, raw, proposal, profile):
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    from pulsemorph.agentic.common_player import materialize_reference
    from pulsemorph.motion.g1 import assemble_reference_inputs
    generator = executor.build_generator(proposal, raw, joint_names=profile["joint_names"], body_names=profile["body_names"],
        source_start_s=0., control_steps=200, section_windows_s=[[.4, 2.], [2., 3.6]],
        source_provenance={"path": SOURCE, "sha256": SOURCE_HASH, "kind": "same pinned Walk; bounded gain diagnostic, not learning"})
    reference = materialize_reference(generator, profile=profile, candidate_id=proposal["candidate_id"], observation_id=proposal["observation_id"],
        task_id=proposal["task_id"], runtime_session_id=proposal["runtime_session_id"], source_path=SOURCE, source_sha256=SOURCE_HASH,
        audio_observed_through_sample=None, audio_duration_s=0., producer_kind="agentic")
    reference = executor.compatible_metadata(reference, executor.proposal_producer(proposal))
    executor.strict_g1_preflight(reference)
    for index in range(200): assemble_reference_inputs(reference, index * .02)
    curve = CubicHermiteSpline(reference.time_offsets_s, reference.joint_position_rad, reference.joint_velocity_rad_s)
    roots = curve.derivative().roots(extrapolate=False)
    xml = ET.parse(profile["body_path"]).getroot()
    limits = {j.attrib["name"]: [float(v) for v in j.attrib["range"].split()] for j in xml.findall(".//joint") if "name" in j.attrib and "range" in j.attrib}
    extrema = []
    for joint, name in enumerate(profile["joint_names"]):
        if name not in limits: raise ValueError("Explicit unchanged joint range missing: " + name)
        interior = roots[joint][np.isfinite(roots[joint]) & (roots[joint] >= 0) & (roots[joint] <= 4.14)]
        points = np.unique(np.r_[0., 4.14, curve.x[(curve.x >= 0) & (curve.x <= 4.14)], interior])
        values = curve(points)[:, joint]
        inside = values.min() >= limits[name][0] - 1e-8 and values.max() <= limits[name][1] + 1e-8
        extrema.append({"joint": name, "min_rad": float(values.min()), "max_rad": float(values.max()), "limits_rad": limits[name], "within_limits": bool(inside)})
        if not inside: raise ValueError("Committed reference exceeds unchanged joint range: " + name)
    return {"strict_common_control_queries": 200, "analytic_common_joint_extrema": extrema, "passed": True,
            "scope": "common admission and continuous commanded-curve range check; not a physical safety certificate"}


def measure(audit, center):
    import numpy as np
    q = audit["arrays"]["qpos"][:, 7:]
    times = audit["arrays"]["state_times"]
    reference = np.asarray(audit["cache"]["committed__dof_pos"]) if "cache" in audit else None
    if reference is None: raise ValueError("Committed reference cache absent")
    if reference.shape != q.shape or not np.allclose(times, np.arange(len(times)) * .02, rtol=0, atol=1e-8):
        raise ValueError("Realized/reference grids do not align")
    groups = [[i for i, name in enumerate(audit["names"]) if name.startswith(side + "_") and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))] for side in ("left", "right")]
    values = {"actual_rms_rad": [], "committed_reference_rms_rad": [], "same_time_actual_reference_rmse_rad": []}
    for start, end in ((.8, 1.6), (2.4, 3.2)):
        mask = (times >= start - 1e-10) & (times <= end + 1e-10)
        for group in groups:
            for key, array in [("actual_rms_rad", q - center), ("committed_reference_rms_rad", reference - center), ("same_time_actual_reference_rmse_rad", q - reference)]:
                values[key].append(float(np.sqrt(np.mean(array[mask][:, group] ** 2))))
    return values


def diagnostics(matrix):
    import numpy as np
    matrix = np.asarray(matrix)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(matrix.dtype).eps * singular[0]
    return {"matrix_rows_outputs_columns_perturbed_gains": matrix.tolist(), "singular_values": singular.tolist(),
            "numerical_rank": int(np.sum(singular > tolerance)), "rank_absolute_tolerance": float(tolerance),
            "frobenius_norm": float(np.linalg.norm(matrix)), "column_l2_norms": np.linalg.norm(matrix, axis=0).tolist(),
            "max_absolute_entry": float(np.max(np.abs(matrix))),
            "condition_number": None if singular[-1] == 0 else float(singular[0] / singular[-1]),
            "scope": "one-sided finite difference at step0.6; numerical rank is not useful controllability, local infinitesimal Jacobian or global feasibility proof"}


def main():
    import numpy as np
    import torch
    from pulsemorph.agentic.app import runtime
    from pulsemorph.motion.g1 import profile_from_yaml
    setup_resume = "--resume-setup" in sys.argv[1:]
    if BASE.exists():
        if not setup_resume or not (BASE / "setup_failure_v1.json").is_file() or (BASE / "trials").exists() or list(BASE.glob("*.proposal.json")):
            raise FileExistsError("Gain probe already initialized; never repeat actual physical candidates")
    else:
        if setup_resume: raise ValueError("No preserved setup failure to resume")
        BASE.mkdir(parents=True)
    manifest = json.loads((PRIOR / "manifest.json").read_text())
    for relative, expected in manifest["preexecution_input_hashes"].items():
        if sha(ROOT / relative) != expected: raise ValueError("Borrowed baseline source/controller/runtime pin differs: " + relative)
    prior_diff = json.loads((PRIOR.parent.parent / "wrapper_diff.json").read_text())
    executor_path = ROOT / "experiments/agentic/app/execute_walk_probe_v1.py"
    if sha(executor_path) != prior_diff["executor_probe_sha256"] or sha(ROOT / SOURCE) != SOURCE_HASH:
        raise ValueError("Borrowed baseline Walk source/wrapper differs")
    module = runtime._module(ROOT, "evaluate_multifeature"); original_audit = module._audit
    borrowed = runtime._audit_app(module, original_audit, PRIOR)
    borrowed["cache"] = dict(np.load(PRIOR / "reference_cache.npz", allow_pickle=False))
    center = borrowed["arrays"]["qpos"][0, 7:].copy()
    baseline = measure(borrowed, center)
    if any(v < .001 for key in ("actual_rms_rad", "committed_reference_rms_rad") for v in baseline[key]):
        raise ValueError("Borrowed baseline actual/reference denominator too small")
    baseline_proof = {"path": PRIOR.relative_to(ROOT).as_posix(), "physical_trial_is_borrowed_not_new": True,
                      "manifest_sha256": sha(PRIOR / "manifest.json"), "common_manifest_sha256": sha(PRIOR / "common_reference/manifest.json"),
                      "exact_current_preexecution_pins_verified": manifest["preexecution_input_hashes"], "full_independent_audit_passed": True,
                      "strict_common_rows": borrowed["strict_common_control_rows"], "fresh_anchor_audit": borrowed["fresh_anchor_audit"],
                      "center": "same prior actual initial qpos29 joints for reference and realized measurements", "measurements": baseline}
    baseline_receipt = BASE / ("borrowed_baseline_reaudit_setup_v2.json" if setup_resume else "borrowed_baseline_audit.json")
    save(baseline_receipt, baseline_proof)
    identity = dict(zip(NAMES, [1.] * 4), timing_shift_s=0.)
    parameters = [dict(identity, **{name: 1.6}) for name in NAMES]
    protocol = {"schema": "pulsemorph.walk_gain_protocol.v1", "at": datetime.now(timezone.utc).isoformat(),
                "hypothesis_question": "Does weak observed gain response coincide with weak committed-reference edit or reference/realized tracking attenuation? No motor-only causality inferred from correlations.",
                "maximum_new_trials": 4, "new_model_calls": 0, "audio": {"available": False, "duration_s": 0., "observed_through_sample": None},
                "source": {"path": SOURCE, "sha256": SOURCE_HASH}, "executor_sha256": sha(executor_path), "script_sha256": sha(Path(__file__)),
                "numeric_runtime_sha256": sha(ROOT / "src/pulsemorph/agentic/app/runtime.py"), "borrowed_baseline_receipt_sha256": sha(baseline_receipt),
                "parameters": parameters, "tool": "global_clock_v1", "duration_s": 4, "source_start_s": 0,
                "measurement_windows_s": [[.8, 1.6], [2.4, 3.2]], "outputs_order": NAMES, "inputs_order": NAMES,
                "response_formula": "(RMSratio_to_matching_borrowed_baseline-1)/0.6; columns=one changed gain", "gain_step": .6,
                "rank_tolerance": "max(matrix.shape)*float64epsilon*largestSingularValue; report scales and do not equate numerical rank with useful control",
                "denominator_floor_rad": .001, "unchanged_physical_screen": {"min_root_height_m": .6, "max_mean_joint_rmse_rad": .35},
                "stop_rule": "Stop after any common/range/integrity/execution failure; preserve failed evidence; no replacements or adaptive calls.",
                "no_optimizer_or_intent_target_pass_claim": True, "public_app_catalog_unchanged": True}
    protocol_path = BASE / ("protocol_setup_v2.json" if setup_resume else "protocol.json")
    if setup_resume:
        prior_protocol = json.loads((BASE / "protocol.json").read_text())
        if prior_protocol["parameters"] != parameters or prior_protocol["executor_sha256"] != sha(executor_path):
            raise ValueError("Setup amendment may not change source/wrapper or declared probe parameters")
        protocol["setup_amendment"] = {"prior_protocol_sha256": sha(BASE / "protocol.json"), "failure_receipt": "setup_failure_v1.json",
                                        "reason": "Pinned vendor source import path added before trusted torch.load; no physical candidate attempted or replaced"}
    save(protocol_path, protocol)
    spec = importlib.util.spec_from_file_location("walk_gain_executor", executor_path)
    executor = importlib.util.module_from_spec(spec); spec.loader.exec_module(executor)
    raw = torch.load(ROOT / SOURCE, map_location="cpu", weights_only=False)
    profile = profile_from_yaml()
    results = []
    try:
        for index, (name, values) in enumerate(zip(NAMES, parameters)):
            if shutil.disk_usage(ROOT).free < 256 * 1024 * 1024: raise RuntimeError("Insufficient disk before next fixed gain probe")
            proposal = {"schema": "pulsemorph.source_capability_probe_proposal.v1", "candidate_id": "app_walk_gain_probe_v1__" + name,
                        "job_id": "app_walk_gain_probe_v1", "trial_index": index, "source_id": "walk", "control_tool": "global_clock_v1",
                        "parameters": values, "reason": "Preregistered one-at-a-time gain1.6 diagnostic: " + name + "; no goal/learning claim",
                        "duration_s": 4, "source_start_s": 0, "observation_id": "walk_gain_probe:no_audio", "task_id": "app_walk_gain_probe_v1",
                        "runtime_session_id": "walk_gain_probe:" + name, "audio": protocol["audio"], "source_motion": protocol["source"], "protocol_sha256": sha(protocol_path)}
            save(BASE / (name + ".proposal.json"), proposal)
            save(BASE / (name + ".common_preflight.json"), common_preflight(executor, raw, proposal, profile))
            output = BASE / "trials" / name
            with (BASE / (name + ".runtime.log")).open("xb") as log:
                process = subprocess.run([str(PYTHON), "-I", str(executor_path), "--proposal", str(BASE / (name + ".proposal.json")), "--output", str(output)], cwd=ROOT, stdout=log, stderr=log, timeout=180)
            actual = json.loads((output / "manifest.json").read_text())
            if process.returncode or actual["status"] != "success": raise RuntimeError("Declared gain trial failed; no replacement: " + name)
            audit = runtime._audit_app(module, original_audit, output)
            audit["cache"] = dict(np.load(output / "reference_cache.npz", allow_pickle=False))
            if not np.array_equal(audit["arrays"]["qpos"][0, 7:], center): raise ValueError("Initial joint reset differs from borrowed baseline")
            measured = measure(audit, center)
            row = {"name": name, "index": index, "parameters": values, "measurements": measured,
                   "actual_ratios": (np.asarray(measured["actual_rms_rad"]) / baseline["actual_rms_rad"]).tolist(),
                   "reference_ratios": (np.asarray(measured["committed_reference_rms_rad"]) / baseline["committed_reference_rms_rad"]).tolist(),
                   "per_window_tracking_rmse_change_rad": (np.asarray(measured["same_time_actual_reference_rmse_rad"]) - baseline["same_time_actual_reference_rmse_rad"]).tolist(),
                   "actual_physics_steps": actual["actual_physics_steps"], "online_onnx_calls": actual["online_onnx_call_count"],
                   "integrity_passed": True, "strict_common_rows": audit["strict_common_control_rows"], "fresh_anchor_audit": audit["fresh_anchor_audit"],
                   "min_root_height_m": actual["all_substeps_min_root_height_m"], "mean_joint_rmse_rad": audit["mean_joint_rmse_rad"], "manifest_sha256": sha(output / "manifest.json")}
            row["physical_screen_passed"] = row["min_root_height_m"] >= .6 and row["mean_joint_rmse_rad"] <= .35
            save(BASE / (name + ".evaluation.json"), row); results.append(row)
    except Exception as exc:
        save(BASE / "failure.json", {"status": "stopped_no_replacement", "phase_trial_index": len(results), "error_type": type(exc).__name__, "error": str(exc), "completed_new_trials": len(results), "new_model_calls": 0})
        raise
    actual_matrix = np.column_stack([(np.asarray(row["actual_ratios"]) - 1) / .6 for row in results])
    reference_matrix = np.column_stack([(np.asarray(row["reference_ratios"]) - 1) / .6 for row in results])
    final = {"schema": "pulsemorph.walk_gain_response.v1", "new_physical_trials": 4, "new_physics_steps": sum(row["actual_physics_steps"] for row in results),
             "new_online_onnx_calls": sum(row["online_onnx_calls"] for row in results), "new_LLM_calls": 0, "audio_available": False,
             "borrowed_baseline_count_not_new": 1, "baseline_measurements": baseline, "probes": results,
             "actual_response": diagnostics(actual_matrix), "committed_reference_response": diagnostics(reference_matrix),
             "scope": "Four fixed one-sided probes in one source/reset/controller. No optimizer, intent success, source learning, motor-only causal attribution or global infeasibility claim."}
    save(BASE / "result.json", final)
    print(json.dumps({key: final[key] for key in ("new_physical_trials", "new_physics_steps", "new_online_onnx_calls", "new_LLM_calls")}))


if __name__ == "__main__": main()
