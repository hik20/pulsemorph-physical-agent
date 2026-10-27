"""Bounded Walk source precheck. Does not admit a source or bypass app guards."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research/vendor/ProtoMotions")]
SOURCE = "research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_walk.motion"
EXPECTED = "09061b8fa0df9e99daaf857293b32cfa8d2faa68631ff20ed0fdde0683e80884"
COMMIT = "f69fdcaabff31846b4d2247207417b57dc788828"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def precheck():
    import numpy as np
    import torch
    from pulsemorph.motion.g1 import profile_from_yaml, assemble_reference_inputs
    from pulsemorph.agentic.common_player import materialize_reference
    vendor = ROOT / "research/vendor/ProtoMotions"
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT or sha(ROOT / SOURCE) != EXPECTED:
        raise ValueError("Walk vendor commit or trusted source hash mismatch before deserialization")
    executor_path = ROOT / "experiments/agentic/app/execute_app.py"
    spec = importlib.util.spec_from_file_location("walk_precheck_unmodified_app", executor_path)
    executor = importlib.util.module_from_spec(spec); spec.loader.exec_module(executor)
    profile_path = vendor / "data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml"
    profile = profile_from_yaml(profile_path)
    if profile["yaml_sha256"] != "57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139":
        raise ValueError("Controller profile differs from existing pinned source inventory")
    raw = torch.load(ROOT / SOURCE, map_location="cpu", weights_only=False)
    frames, fps = len(raw["dof_pos"]), float(raw["fps"])
    if (frames - 1) / fps < 4.14:
        raise ValueError("Walk source does not cover full0..4.14second reference horizon")
    proposal = {"schema": "pulsemorph.agentic_app_proposal.v1", "control_tool": "global_clock_v1",
                "parameters": dict(left_gain_a=1., right_gain_a=1., left_gain_b=1., right_gain_b=1., timing_shift_s=0.),
                "source_motion": {"path": SOURCE, "sha256": EXPECTED}}
    generator = executor.build_generator(proposal, raw, joint_names=profile["joint_names"], body_names=profile["body_names"],
        source_start_s=0., control_steps=200, section_windows_s=[[.4, 2.], [2., 3.6]],
        source_provenance={"path": SOURCE, "sha256": EXPECTED, "kind": "existing trusted upstream generated source; not learned by this pilot"})
    reference = materialize_reference(generator, profile=profile, candidate_id="app_walk_admission_v1_precheck",
        observation_id="walk:no_audio", task_id="app_walk_admission_v1", runtime_session_id="walk:precheck",
        source_path=SOURCE, source_sha256=EXPECTED, audio_observed_through_sample=None, audio_duration_s=0., producer_kind="agentic")
    producer = executor.proposal_producer(proposal)
    reference = executor.compatible_metadata(reference, producer)
    count = executor.strict_g1_preflight(reference)
    # Check every control query, including its four future offsets, without physics.
    for index in range(200): assemble_reference_inputs(reference, index * .02)
    q = generator.source.splines["dof_pos"]
    xml = ET.parse(profile["body_path"]).getroot()
    ranges = {j.attrib["name"]: [float(v) for v in j.attrib["range"].split()] for j in xml.findall(".//joint") if "name" in j.attrib and "range" in j.attrib}
    roots = q.derivative().roots(extrapolate=False)
    diagnostics = []
    for joint, name in enumerate(profile["joint_names"]):
        points = np.unique(np.r_[0., 4.14, q.x[(q.x >= 0) & (q.x <= 4.14)], roots[joint][(roots[joint] >= 0) & (roots[joint] <= 4.14)]])
        values = q(points)[:, joint]
        limits = ranges.get(name)
        diagnostics.append({"joint": name, "minimum_rad": float(values.min()), "maximum_rad": float(values.max()), "mjcf_range_rad": limits,
                            "within_explicit_range": None if limits is None else bool(values.min() >= limits[0] - 1e-8 and values.max() <= limits[1] + 1e-8)})
    try:
        executor.resolve_source_motion(proposal)
        admitted, rejection = True, None
    except ValueError as exc:
        admitted, rejection = False, str(exc)
    return {"schema": "pulsemorph.walk_source_precheck.v1", "at": datetime.now(timezone.utc).isoformat(),
            "read_only_source_precheck": True, "source": {"path": SOURCE, "sha256": EXPECTED, "vendor_commit": commit, "frames": frames, "fps": fps, "last_source_time_s": (frames - 1) / fps},
            "profile": {key: profile[key] for key in ("yaml_sha256", "body_sha256", "joint_names", "body_names")},
            "full_requested_support_s": [0, 4.14], "padding_retime_or_clamp": False, "strict_common_preflight_checks": count,
            "all200_strict_common_control_queries_passed": True, "raw_source_continuous_joint_range_diagnostics": diagnostics,
            "joint_range_scope": "Analytic extrema of source Hermite joint curve within0..4.14s against explicit MJCF ranges; diagnostic only, not physical admission or a new safety certificate.",
            "unmodified_executor_admits_walk": admitted, "executor_rejection": rejection, "executor_sha256": sha(executor_path),
            "public_app_source_PINS_changed": False, "actual_physical_trials": 0, "actual_physics_steps": 0, "new_model_calls": 0,
            "decision": "source/common precheck passed; physics blocked by exact existing two-source executor allowlist" if not admitted else "precheck passed; bounded physics pilot may proceed",
            "rights": "Existing upstream generated Kimodo example; motion/character/media terms remain separately unresolved."}


def run_pilot():
    """Exactly three declared no-audio/no-model trials via the separate wrapper."""
    import numpy as np
    from pulsemorph.agentic.app import runtime
    base = ROOT / "artifacts/runs/agentic/app_walk_admission_v1"
    protocol_path = base / "protocol.json"
    if protocol_path.exists(): raise FileExistsError("Walk pilot already initialized; never repeat its physical trial budget")
    check = precheck()
    if not check["all200_strict_common_control_queries_passed"]: raise ValueError("Walk source/common precheck failed")
    diff = json.loads((base / "wrapper_diff.json").read_text())
    executor = ROOT / "experiments/agentic/app/execute_walk_probe_v1.py"
    renderer = ROOT / "experiments/agentic/app/render_walk_probe_v1.py"
    original_executor = ROOT / "experiments/agentic/app/execute_app.py"
    original_renderer = ROOT / "src/pulsemorph/agentic/app/renderer.py"
    if any(sha(path) != diff[key] for path, key in [(executor, "executor_probe_sha256"), (renderer, "renderer_probe_sha256"), (original_executor, "executor_original_sha256"), (original_renderer, "renderer_original_sha256")]):
        raise ValueError("Versioned pilot/original source differs from recorded narrow diff")
    python = Path("C:/Users/halke/.cache/pulsemorph-agentic/envs/app-cpu-py311-v1/Scripts/python.exe")
    identity = dict(left_gain_a=1., right_gain_a=1., left_gain_b=1., right_gain_b=1., timing_shift_s=0.)
    entries = [("identity_global", "global_clock_v1", identity), ("identity_local", "local_clock_1p4_2p8_v1", identity),
               ("asymmetric_edit", "global_clock_v1", dict(left_gain_a=.8, right_gain_a=1.2, left_gain_b=1.2, right_gain_b=.8, timing_shift_s=0.))]
    protocol = {"schema": "pulsemorph.walk_capability_protocol.v1", "at": datetime.now(timezone.utc).isoformat(),
                "source_precheck": check, "source_wrapper_diff_sha256": sha(base / "wrapper_diff.json"), "wrapper_pins": diff,
                "python": str(python), "maximum_actual_trials": 3, "new_model_calls": 0,
                "audio": {"available": False, "duration_s": 0., "observed_through_sample": None},
                "trials": [{"index": index, "name": name, "tool": tool, "parameters": parameters} for index, (name, tool, parameters) in enumerate(entries)],
                "observed_feature_windows_s": [[.8, 1.6], [2.4, 3.2]], "baseline_minimum_arm_denominator_rms_rad": .001,
                "physical_screen": {"min_root_height_m": .6, "max_mean_joint_rmse_rad": .35},
                "no_goal_targets_or_intent_pass_claim": True, "public_app_catalog_admission": False}
    with protocol_path.open("x", encoding="utf-8") as stream: json.dump(protocol, stream, indent=2)
    results, audited = [], []
    module = runtime._module(ROOT, "evaluate_multifeature")
    original_audit = module._audit
    for index, (name, tool, parameters) in enumerate(entries):
        if shutil.disk_usage(ROOT).free < 256 * 1024 * 1024: raise RuntimeError("Insufficient disk before next bounded Walk trial")
        output = base / "trials" / name
        proposal = {"schema": "pulsemorph.source_capability_probe_proposal.v1", "candidate_id": "app_walk_admission_v1__trial_" + str(index),
                    "job_id": "app_walk_admission_v1", "trial_index": index, "source_id": "walk", "control_tool": tool,
                    "parameters": parameters, "reason": "Bounded research source variation: " + name + "; no public app admission or skill learning",
                    "duration_s": 4, "source_start_s": 0, "observation_id": "app_walk_admission_v1:no_audio", "task_id": "app_walk_admission_v1",
                    "runtime_session_id": "app_walk_admission_v1:trial_" + str(index), "audio": protocol["audio"],
                    "source_motion": {"path": SOURCE, "sha256": EXPECTED}, "protocol_sha256": sha(protocol_path)}
        proposal_path = base / (name + ".proposal.json")
        with proposal_path.open("x", encoding="utf-8") as stream: json.dump(proposal, stream, indent=2)
        with (base / (name + ".runtime.log")).open("xb") as log:
            process = subprocess.run([str(python), "-I", str(executor), "--proposal", str(proposal_path), "--output", str(output)], cwd=ROOT, stdout=log, stderr=log, timeout=180)
        manifest = json.loads((output / "manifest.json").read_text())
        if process.returncode or manifest["status"] != "success":
            failure = {"schema": "pulsemorph.walk_capability_failure.v1", "trial": name, "returncode": process.returncode,
                       "manifest": manifest, "completed_prior_trial_count": len(results), "new_model_calls": 0,
                       "decision": "pilot stopped; failed trial evidence preserved; no admission claim"}
            with (base / "failure.json").open("x", encoding="utf-8") as stream: json.dump(failure, stream, indent=2)
            return failure
        row = runtime._audit_app(module, original_audit, output)
        audited.append(row)
        names, arrays = row["names"], row["arrays"]
        groups = [[j for j, joint in enumerate(names) if joint.startswith(side + "_") and any(part in joint for part in ("shoulder_", "elbow_", "wrist_"))] for side in ("left", "right")]
        values = []
        for start, end in ((.8, 1.6), (2.4, 3.2)):
            mask = (arrays["state_times"] >= start - 1e-10) & (arrays["state_times"] <= end + 1e-10)
            for group in groups:
                values.append(float(np.sqrt(np.mean((arrays["qpos"][mask, 7:][:, group] - arrays["qpos"][0, 7:][group]) ** 2))))
        result = {"name": name, "index": index, "parameters": parameters, "tool": tool, "integrity_passed": True,
                  "strict_common_rows": row["strict_common_control_rows"], "fresh_anchor_audit": row["fresh_anchor_audit"],
                  "arm_displacement_rms_rad": values, "min_root_height_m": manifest["all_substeps_min_root_height_m"],
                  "mean_joint_rmse_rad": row["mean_joint_rmse_rad"], "actual_physics_steps": manifest["actual_physics_steps"],
                  "online_onnx_calls": manifest["online_onnx_call_count"], "manifest_sha256": sha(output / "manifest.json")}
        result["physical_screen_passed"] = result["min_root_height_m"] >= .6 and result["mean_joint_rmse_rad"] <= .35
        if index == 0 and any(value < .001 for value in values):
            raise ValueError("Walk baseline has near-zero arm feature denominator; no meaningful ratio claim")
        result["observed_ratios_to_own_baseline"] = [value / denominator for value, denominator in zip(values, values if index == 0 else results[0]["arm_displacement_rms_rad"])]
        with (base / (name + ".evaluation.json")).open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2)
        results.append(result)
    parity = {key: bool(np.array_equal(audited[0]["arrays"][key], audited[1]["arrays"][key])) for key in ("qpos", "qvel", "state_times")}
    media = ROOT / "artifacts/media/agentic/app_walk_admission_v1"
    spec = importlib.util.spec_from_file_location("walk_pilot_renderer", renderer)
    rendering = importlib.util.module_from_spec(spec); spec.loader.exec_module(rendering)
    rendered = rendering.render_comparison(base / "trials/identity_global", base / "trials/asymmetric_edit", media / "comparison", None)
    final = {"schema": "pulsemorph.walk_capability_result.v1", "actual_trials": 3, "actual_physics_steps": sum(item["actual_physics_steps"] for item in results),
             "online_onnx_calls": sum(item["online_onnx_calls"] for item in results), "new_model_calls": 0, "audio_observed": False,
             "trials": results, "identity_global_local_array_parity": parity, "rendered": rendered,
             "scope": "Separate bounded source-variation pilot. No fixed intent targets or pass claim, no new skill learning, no public app admission, no aesthetics/recovery/safety claim.",
             "public_app_source_catalog_still_two": True}
    with (base / "result.json").open("x", encoding="utf-8") as stream: json.dump(final, stream, indent=2)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-pilot", action="store_true")
    args = parser.parse_args()
    if args.run_pilot:
        result = run_pilot()
        print(json.dumps({key: result.get(key) for key in ("schema", "actual_trials", "actual_physics_steps", "new_model_calls")}))
        return
    if args.output is None: parser.error("--output is required for a source precheck")
    if args.output.exists(): raise FileExistsError("Choose a new exclusive Walk precheck receipt")
    result = precheck()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"source_common_precheck": "passed", "executor_admits_walk": result["unmodified_executor_admits_walk"], "physical_trials": 0, "output": str(args.output)}))


if __name__ == "__main__": main()
