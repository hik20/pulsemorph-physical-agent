"""Prospective wide clock proof fixtures. No model or physics execution."""
import ast
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import hashlib
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from pulsemorph.agentic.wide_clock_reference import WideClockReferencePlayer
from pulsemorph.agentic.multifeature_reference import MultiFeatureReferencePlayer

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wide_probe_fixture", ROOT / "experiments/agentic/app/wide_clock_probe/execute_wide_clock_v1.py")
PROBE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(PROBE)


def fixture(**changes):
    times = np.arange(601) / 100
    joints = [f"leg_{i}" for i in range(15)] + [f"{side}_{part}_joint" for side in ("left", "right") for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    raw = dict(fps=100, dof_pos=np.sin(times[:, None] * np.linspace(.4, 1.4, 29)) * .2,
               rigid_body_pos=np.stack([times * .1, times * 0, times * 0], axis=-1)[:, None],
               rigid_body_rot=Rotation.from_euler("z", times * .1).as_quat()[:, None])
    args = dict(joint_names=joints, body_names=["torso_link"], source_start_s=.2, control_steps=200, source_provenance={"kind": "fixture"})
    args.update(changes)
    return raw, args


class WideClockTests(unittest.TestCase):
    def test_identity_gate_rejects_success_status_with_wrong_provenance(self):
        spec = importlib.util.spec_from_file_location("wide_bridge_metadata_fixture", ROOT / "experiments/agentic/app/wide_clock_probe/bridge_checks.py")
        bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); baseline = root / "jobs/old/trials/baseline"; candidate = root / "jobs/new/trials/identity"
            baseline.mkdir(parents=True); candidate.mkdir(parents=True)
            (baseline / "manifest.json").write_text(json.dumps({"status": "success", "artifact_hashes": {}}))
            state = baseline.parents[1] / "state.json"; state.write_text('{"id":"old"}')
            report = root / "audit.json"; report.write_text(json.dumps({"jobs": [{"id": "old", "passed": True, "state_sha256": bridge.sha(state), "physics": [{"trial": "baseline", "state": "independently_verified"}]}]}))
            proposal = candidate / "proposal.json"; proposal.write_text('{"schema":"pulsemorph.agentic_app_proposal.v1"}')
            (candidate / "manifest.json").write_text(json.dumps({"schema": "pulsemorph.agentic_app_trial.v1", "status": "success", "artifact_hashes": {"proposal.json": bridge.sha(proposal)}}))
            binding = {"baseline": {"directory": baseline.relative_to(root).as_posix(), "manifest_sha256": bridge.sha(baseline / "manifest.json"), "state_sha256": bridge.sha(state), "independent_audit_receipt": "audit.json", "independent_audit_sha256": bridge.sha(report)}, "candidate_id": "new", "source_id": "wave", "protocol_sha256": "a" * 64}
            with patch.object(bridge, "compare_numeric_arrays") as numeric:
                with self.assertRaisesRegex(ValueError, "exact frozen wide-clock identity"): bridge.gate_identity(root, baseline, candidate, binding=binding)
                numeric.assert_not_called()
            binding["baseline"]["manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "frozen protocol binding"): bridge.gate_identity(root, baseline, candidate, binding=binding)

    def test_bridge_checks_each_numeric_key_and_raw_byte_without_tolerance(self):
        spec = importlib.util.spec_from_file_location("wide_bridge_fixture", ROOT / "experiments/agentic/app/wide_clock_probe/bridge_checks.py")
        bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
        with tempfile.TemporaryDirectory() as temporary:
            before, after = [Path(temporary) / name for name in ("old", "new")]
            for folder in (before, after):
                (folder / "common_reference").mkdir(parents=True)
                for name in bridge.RAW_REQUIRED: np.save(folder / name, np.arange(12, dtype=np.float32).reshape(3, 4))
                for name in bridge.CONTAINERS: np.savez(folder / name, first=np.arange(4, dtype=np.float32), second=np.ones((2, 3), dtype=np.float64))
            self.assertEqual(len(bridge.compare_numeric_arrays(before, after)["container_keys"]), 4)
            np.savez(after / "policy_inputs.npz", first=np.array([0., 1., 2., 3.000001], dtype=np.float32), second=np.ones((2, 3), dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "numeric array differs"): bridge.compare_numeric_arrays(before, after)
            np.savez(after / "policy_inputs.npz", first=np.arange(4, dtype=np.float32), second=np.ones((2, 3), dtype=np.float64))
            np.save(after / "qvel.npy", np.arange(12, dtype=np.float64).reshape(3, 4))
            with self.assertRaisesRegex(ValueError, "raw bytes differ"): bridge.compare_numeric_arrays(before, after)

    def test_zero_shift_exact_old_reference_arrays(self):
        raw, args = fixture()
        for gains in ({}, dict(left_gain_a=.35, right_gain_a=1.6, left_gain_b=.7, right_gain_b=1.2)):
            old, new = MultiFeatureReferencePlayer(raw, **args, **gains), WideClockReferencePlayer(raw, **args, **gains)
            before, after = old.archive(), new.archive()
            for key in before: np.testing.assert_array_equal(before[key], after[key])
        with self.assertRaises(ValueError): MultiFeatureReferencePlayer(raw, **args, timing_shift_s=.6)
        self.assertEqual(WideClockReferencePlayer(raw, **args, timing_shift_s=.6).timing_shift_s, .6)

    def test_clock_derivative_rate_bounds_monotone_and_c1(self):
        raw, args = fixture()
        for shift in (-.6, .6):
            player = WideClockReferencePlayer(raw, **args, timing_shift_s=shift)
            samples = np.asarray([player.source_clock(float(t)) for t in np.linspace(0, 4.14, 3001)])
            self.assertTrue(np.all(np.diff(samples[:, 0]) > 0))
            self.assertGreaterEqual(samples[:, 1].min(), 1 - .6 * np.pi / 4 - 1e-12)
            self.assertLessEqual(samples[:, 1].max(), 1 + .6 * np.pi / 4 + 1e-12)
            self.assertGreater(player.minimum_source_rate, .5); self.assertLess(player.maximum_source_rate, 1.5)
            for t in (.1, .7, 1., 1.9, 2.6, 3., 3.8):
                step = 1e-6
                finite = (player.source_clock(t + step)[0] - player.source_clock(t - step)[0]) / (2 * step)
                self.assertAlmostEqual(finite, player.source_clock(t)[1], places=8)
            for t in (0., 4., 4.02, 4.14): self.assertEqual(player.source_clock(t), (.2 + t, 1.))
            for t in (1e-8, 4 - 1e-8, 4 + 1e-8):
                self.assertAlmostEqual(player.source_clock(t)[0], .2 + t, places=7)
                self.assertAlmostEqual(player.source_clock(t)[1], 1., places=7)

    def test_state_derivative_and_all_future_queries_finite(self):
        raw, args = fixture()
        for shift in (-.6, .6):
            player = WideClockReferencePlayer(raw, **args, timing_shift_s=shift, left_gain_a=.35, right_gain_b=1.6)
            for time in (.6, 1.1, 1.7, 2.3, 2.9, 3.4):
                step = .001
                finite = (player.state_at_time(time + step)["dof_pos"] - player.state_at_time(time - step)["dof_pos"]) / (2 * step)
                np.testing.assert_allclose(finite, player.state_at_time(time)["dof_vel"], atol=.0005, rtol=.001)
            for index in range(200):
                for array in player.get_future_references(index, [1, 2, 4, 8]).values(): self.assertTrue(np.isfinite(array).all())
            _, metadata = player.draft_motion_reference(199)
            self.assertEqual(metadata["control_tool"], "wide_global_clock_0p6_v1")
            self.assertFalse(metadata["public_app_admitted"])
            self.assertEqual(metadata["physical_acceptance"], "not_evaluated")

    def test_invalid_parameters_time_and_short_horizon(self):
        raw, args = fixture()
        for changes in [dict(timing_shift_s=.60001), dict(timing_shift_s=True), dict(timing_shift_s=float("nan")), dict(timing_shift_s=".6"), dict(left_gain_a=.349), dict(right_gain_b=1.601), dict(control_steps=199), dict(section_windows_s=[[.3, 2], [2, 3.6]])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): WideClockReferencePlayer(raw, **dict(args, **changes))
        player = WideClockReferencePlayer(raw, **args)
        for time in (True, "1", -.001, 4.141, float("nan"), float("inf")):
            for method in (player.source_clock, player.state_at_time):
                with self.subTest(time=time, method=method), self.assertRaises(ValueError): method(time)
        short = {key: value[:400] if isinstance(value, np.ndarray) else value for key, value in raw.items()}
        with self.assertRaises(ValueError): WideClockReferencePlayer(short, **args)

    def test_executor_tool_schema_and_exact_two_source_guard(self):
        proposal = {"schema": "pulsemorph.wide_clock_probe_proposal.v1", "control_tool": "wide_global_clock_0p6_v1"}
        self.assertEqual(PROBE.proposal_producer(proposal)["id"], "pulsemorph-wide-clock-probe-v1")
        for tool in ("global_clock_v1", "local_clock_1p4_2p8_v1", None):
            with self.assertRaises(ValueError): PROBE.proposal_producer(dict(proposal, control_tool=tool))
        with self.assertRaises(ValueError): PROBE.proposal_producer(dict(proposal, schema="pulsemorph.agentic_app_proposal.v1"))
        base = "research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_"
        for name, digest in [("wave", "cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93"), ("high5", "ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44")]:
            self.assertEqual(PROBE.resolve_source_motion({"source_motion": {"path": base + name + ".motion", "sha256": digest}})[1], digest)
        for path, digest in [(base + "walk.motion", "09061b8fa0df9e99daaf857293b32cfa8d2faa68631ff20ed0fdde0683e80884"), (base + "wave.motion", "0" * 64), ("../wave.motion", "0" * 64)]:
            with self.assertRaises(ValueError): PROBE.resolve_source_motion({"source_motion": {"path": path, "sha256": digest}})
        valid = dict(left_gain_a=1., right_gain_a=1., left_gain_b=1., right_gain_b=1., timing_shift_s=.6)
        self.assertEqual(PROBE.validate_parameters(valid), valid)
        with self.assertRaises(ValueError): PROBE.validate_parameters(dict(valid, timing_shift_s=.601))

    def test_numeric_execution_ast_only_provenance_and_snapshot_addition(self):
        old = ast.parse((ROOT / "experiments/agentic/app/execute_app.py").read_text())
        new = ast.parse((ROOT / "experiments/agentic/app/wide_clock_probe/execute_wide_clock_v1.py").read_text())
        original = {node.name: node for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        class Normalize(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value in ("pulsemorph.agentic_app_trial.v1", "pulsemorph.wide_clock_probe_trial.v1"): node.value = "TRIAL_SCHEMA"
                elif node.value in ("app_physical_trial_v1", "app_physical_trial", "wide_clock_capability_probe"): node.value = "EXPERIMENT_PROVENANCE"
                return node
            def visit_List(self, node):
                self.generic_visit(node)
                node.elts = [item for item in node.elts if not (isinstance(item, ast.BinOp) and isinstance(item.right, ast.Constant) and item.right.value == "src/pulsemorph/agentic/wide_clock_reference.py")]
                return node
        for node in new.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)): continue
            if node.name in ("proposal_producer", "validate_parameters", "build_generator"): continue
            self.assertEqual(ast.dump(Normalize().visit(node), include_attributes=False), ast.dump(Normalize().visit(original[node.name]), include_attributes=False), node.name)


if __name__ == "__main__": unittest.main()
