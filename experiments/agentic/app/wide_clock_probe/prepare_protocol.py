"""Prepare prospective wide-clock protocol and reference checks; no physics CLI."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research/vendor/ProtoMotions")]
OUT = ROOT / "artifacts/runs/agentic/app_wide_clock_probe_plan_v1"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import scipy
    from scipy.interpolate import CubicHermiteSpline
    import torch
    import mujoco
    import onnxruntime
    import site
    from pulsemorph.motion.g1 import profile_from_yaml, assemble_reference_inputs
    from pulsemorph.agentic.common_player import materialize_reference
    expected_python = Path("C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe")
    if Path(sys.executable).resolve() != expected_python.resolve() or not sys.flags.isolated or site.ENABLE_USER_SITE is not False:
        raise ValueError("Prospective preparation requires the selected public-wheel production environment under-I")
    if OUT.exists(): raise FileExistsError("Plan already frozen; preserve it and create a new revision for changes")
    OUT.mkdir(parents=True)
    versions = {"numpy": np.__version__, "mujoco": mujoco.__version__, "onnxruntime": onnxruntime.__version__, "torch": torch.__version__}
    executor_path = HERE / "execute_wide_clock_v1.py"
    spec = importlib.util.spec_from_file_location("wide_clock_prospective_executor", executor_path)
    executor = importlib.util.module_from_spec(spec); spec.loader.exec_module(executor)
    profile = profile_from_yaml()
    model_path = ROOT / "research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx"
    sources = {"wave": "cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93", "high5": "ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44"}
    jobs = {"wave": "20261027T023422_8efe4bc4", "high5": "20261027T022322_76db8965"}
    baselines = {}
    prior_report_path = ROOT / "artifacts/runs/agentic/app_validation_v1/jobs_audit_v1.json"
    prior_report = json.loads(prior_report_path.read_text())
    for source, job_id in jobs.items():
        directory = ROOT / "artifacts/runs/agentic/app/jobs" / job_id / "trials/baseline"
        manifest = json.loads((directory / "manifest.json").read_text())
        state = directory.parents[1] / "state.json"
        audited_job = next(row for row in prior_report["jobs"] if row["id"] == job_id)
        if not audited_job["passed"] or sha(state) != audited_job["state_sha256"]: raise ValueError("Old baseline audit/state binding changed")
        if not any(row["trial"] == "baseline" and row["state"] == "independently_verified" for row in audited_job["physics"]): raise ValueError("Old identity baseline lacks independent physical audit")
        if manifest["versions"] != versions: raise ValueError("Critical recorded runtime versions differ from public-wheel environment")
        if manifest["source_motion"]["sha256"] != sources[source] or manifest["onnx_sha256"] != sha(model_path): raise ValueError("Old/new source or controller identity differs")
        for relative, digest in manifest["artifact_hashes"].items():
            if sha(directory / relative) != digest: raise ValueError("Old baseline artifact changed: " + relative)
        compare = [name for name in ("qpos.npy", "qvel.npy", "state_times.npy", "applied_joint_targets.npy", "actions.npy") if (directory / name).is_file()]
        containers = [name for name in ("policy_inputs.npz", "policy_outputs.npz", "reference_cache.npz", "common_reference/arrays.npz") if (directory / name).is_file()]
        baselines[source] = {"directory": directory.relative_to(ROOT).as_posix(), "manifest_sha256": sha(directory / "manifest.json"),
            "state_sha256": sha(state), "independent_audit_receipt": prior_report_path.relative_to(ROOT).as_posix(), "independent_audit_sha256": sha(prior_report_path),
            "versions": manifest["versions"], "audio_observation": manifest["audio_observation"], "raw_array_sha256": {name: sha(directory / name) for name in compare},
            "containers_to_compare_every_key_shape_dtype_bytes": containers, "old_preexecution_source_pins": manifest["preexecution_input_hashes"],
            "new_trial_count": 0, "environment_qualification": "Legacy baseline environment/wrapper differs from public-wheel prospective environment/newclockclass. Matching critical version strings do not establish equality; strict raw-array and per-key numerical-container parity must actually pass."}
    schedule = [("wave", 0.), ("high5", 0.), ("wave", -.6), ("wave", .6), ("high5", -.6), ("high5", .6)]
    protocol = {"schema": "pulsemorph.prospective_wide_clock_protocol.v1", "status": "TEST_ONLY_AWAITING_ROOT_REVIEW_NO_PHYSICS_AUTHORIZED",
        "at": datetime.now(timezone.utc).isoformat(), "physical_trials_executed": 0, "maximum_future_fresh_trials": 6, "maximum_future_physics_steps": 24000,
        "new_LLM_calls": 0, "audio": {"available": False, "duration_s": 0., "observed_through_sample": None},
        "selected_python": str(expected_python), "isolated": bool(sys.flags.isolated), "user_site_enabled": site.ENABLE_USER_SITE, "versions": versions, "scipy": scipy.__version__,
        "math": {"T_s": 4, "shift_bound_s": .6, "rate_min": 1 - .6 * np.pi / 4, "rate_max": 1 + .6 * np.pi / 4,
                 "continuous_clock": "tau=t+s*sin(pi*t/4)^2; monotone0..4, C1 at0/4, identity after4through4.14; not C2",
                 "common_materialization": "analytic generator exact at20ms knots; validated common Hermite/SLERP is authoritative betweenknots, not exact analytic wideclock there"},
        "baseline_bridges": baselines, "prospective_schedule": [{"index": i, "source": source, "shift_s": shift, "gains": [1, 1, 1, 1], "tool": "wide_global_clock_0p6_v1", "phase": "identity_bridge" if i < 2 else "extreme_timing_probe"} for i, (source, shift) in enumerate(schedule)],
        "mandatory_identity_gate_before_extremes": {"raw_arrays": "qpos,qvel,state_times,applied_joint_targets,actions raw.npy SHA256 exact", "containers": "policy_inputs/output,reference_cache and common_reference arrays: everykey,shape,dtype and contiguous array bytes exact; ZIP compression bytes need not match", "absolute_tolerance": 0, "relative_tolerance": 0,
                "audio_bridge": "Old baselines observed audio; new probe has none. Audio/context/provenance metadata may differ explicitly, but no actual policy input/output/common numeric array is exempted.",
                "reset": "initial qpos/qvel and source/controller/body/mesh identities must match; policy arrays gate covers actual observations/actions",
                "failure": "Any version/source/controller/bodymesh/reset/array/identity/integrity mismatch stops before remaining candidates. No environment switching, replacement identity or tolerance relaxation."},
        "per_candidate_before_physics": "all200strict common futurequeries and analytic continuous common joint extrema within unchanged explicit G1 ranges over0..4.14s",
        "physical_screen_unchanged": {"min_root_height_m": .6, "max_mean_joint_rmse_rad": .35},
        "measurements_only": {"arm_ratio_windows_s": [[.8, 1.6], [2.4, 3.2]], "accent_window_s": [1.8, 3.2], "accent_prominence_rad_s": .15,
                              "ratio_baseline": "matching fresh wideidentity for source; compare bridge to archivedbaseline first", "timing_reference_for_readout_only_s": {"high5_music_example": 2.405},
                              "intent_targets": "none; no all-five-goal success claim, no posthoc tolerancechange or all13 robustness claim"},
        "stop_rule": "Any common/range/integrity/execution or identitybridge failure: preserve failed artifacts and stop remaining calls, no replacement. Physical-screen failure is a recorded candidate, not a replacement permission.",
        "source_code_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in (ROOT / "src/pulsemorph/agentic/wide_clock_reference.py", executor_path, Path(__file__), ROOT / "experiments/agentic/app/execute_app.py", ROOT / "src/pulsemorph/agentic/multifeature_reference.py", ROOT / "src/pulsemorph/agentic/app/runtime.py")},
        "public_app_unchanged": "existing ±.3domain/tools/schemas/PINS/UI remain unchanged; this tool is not admitted", "source_and_character_rights": "separate unresolved review; no new asset/model download or learning"}
    xml = ET.parse(profile["body_path"]).getroot()
    limits = {j.attrib["name"]: [float(v) for v in j.attrib["range"].split()] for j in xml.findall(".//joint") if "name" in j.attrib and "range" in j.attrib}
    raw_sources = {}
    checks = []
    for index, (source, shift) in enumerate(schedule):
        source_path = f"research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_{source}.motion"
        proposal = {"schema": "pulsemorph.wide_clock_probe_proposal.v1", "control_tool": "wide_global_clock_0p6_v1", "source_motion": {"path": source_path, "sha256": sources[source]},
                    "parameters": dict(left_gain_a=1., right_gain_a=1., left_gain_b=1., right_gain_b=1., timing_shift_s=shift)}
        trusted, digest = executor.resolve_source_motion(proposal)
        if source not in raw_sources: raw_sources[source] = torch.load(trusted, map_location="cpu", weights_only=False)
        generator = executor.build_generator(proposal, raw_sources[source], joint_names=profile["joint_names"], body_names=profile["body_names"], source_start_s=0., control_steps=200,
            section_windows_s=[[.4, 2.], [2., 3.6]], source_provenance={"path": source_path, "sha256": digest, "kind": "prospective test only"})
        reference = materialize_reference(generator, profile=profile, candidate_id=f"wide_plan_{index}", observation_id="wide:noaudio", task_id="wide_clock_probe", runtime_session_id=f"wide_plan_{index}",
            source_path=source_path, source_sha256=digest, audio_observed_through_sample=None, audio_duration_s=0., producer_kind="agentic")
        reference = executor.compatible_metadata(reference, executor.proposal_producer(proposal))
        for control in range(200): assemble_reference_inputs(reference, control * .02)
        curve = CubicHermiteSpline(reference.time_offsets_s, reference.joint_position_rad, reference.joint_velocity_rad_s)
        extrema = curve.derivative().roots(extrapolate=False); violations = []
        for joint, name in enumerate(profile["joint_names"]):
            if name not in limits: raise ValueError("Missing unchanged joint range")
            roots = extrema[joint][np.isfinite(extrema[joint]) & (extrema[joint] >= 0) & (extrema[joint] <= 4.14)]
            values = curve(np.unique(np.r_[reference.time_offsets_s, roots]))[:, joint]
            if values.min() < limits[name][0] - 1e-8 or values.max() > limits[name][1] + 1e-8: violations.append(name)
        checks.append({"index": index, "source": source, "shift_s": shift, "strict_common_queries": 200, "common_joint_range_violations": violations,
                       "minimum_rate": generator.minimum_source_rate, "maximum_rate": generator.maximum_source_rate, "physical_steps": 0})
        if violations: break
    protocol["reference_only_preparation_checks"] = checks
    protocol["reference_checks_passed"] = len(checks) == 6 and not any(row["common_joint_range_violations"] for row in checks)
    with (OUT / "protocol_v1.json").open("x", encoding="utf-8") as stream: json.dump(protocol, stream, indent=2)
    print(json.dumps({"prospective_candidates": 6, "reference_checks_passed": protocol["reference_checks_passed"], "new_physics_steps": 0, "protocol": str(OUT / "protocol_v1.json")}))


if __name__ == "__main__": main()
