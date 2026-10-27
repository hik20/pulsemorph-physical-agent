"""No-physics fixtures for exact order, pin/bridge failure, attempts and budget."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("wide_runner_fixture", Path(__file__).with_name("runner_v1.py"))
RUNNER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUNNER)


def protocol():
    return {"maximum_future_fresh_trials": 6, "audio": {"available": False, "duration_s": 0., "observed_through_sample": None},
        "prospective_schedule": [{"index": index, "source": source, "shift_s": shift, "gains": [1, 1, 1, 1], "tool": "wide_global_clock_0p6_v1",
            "candidate_id": f"app_wide_clock_probe_v1__{index:02d}_{source}"} for index, (source, shift) in enumerate(RUNNER.EXPECTED_SCHEDULE)]}


def manifest(): return {"status": "success", "actual_control_steps": 200, "actual_physics_steps": 4000, "online_onnx_call_count": 200}


class RunnerTests(unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory(); self.output = Path(self.temp.name) / "newrun"
    def tearDown(self): self.temp.cleanup()

    def run_fixture(self, **overrides):
        self.calls, self.bridges, self.common = [], [], []
        def execute(entry, output):
            attempt = output / "attempts" / f"{entry['index']:02d}.json"
            self.assertTrue(attempt.is_file(), "attempt must be durable before physical callback")
            self.assertEqual(json.loads(attempt.read_text())["candidate"], entry)
            if entry["index"] >= 2: self.assertEqual(self.bridges, [0, 1])
            self.calls.append(entry["index"]); return manifest()
        def common(entry): self.common.append(entry["index"]); return {"passed": True}
        def bridge(entry, output): self.bridges.append(entry["index"]); return {"passed": True}
        callbacks = dict(verify=lambda: None, common=common, execute=execute,
            audit=lambda entry, output: {"index": entry["index"], "physical_screen_passed": entry["index"] != 3}, bridge=bridge)
        callbacks.update(overrides)
        return RUNNER.run_schedule(protocol(), self.output, **callbacks)

    def test_exact_six_order_attempt_before_call_and_both_bridges_first(self):
        result = self.run_fixture()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.calls, [0, 1, 2, 3, 4, 5]); self.assertEqual(self.bridges, [0, 1]); self.assertEqual(self.common, self.calls)
        self.assertEqual(result["attempted_physical_calls"], 6)
        self.assertFalse(result["trials"][3]["physical_screen_passed"], "physical gate failure is recorded, never a replacement")
        self.assertEqual(len(list((self.output / "attempts").glob("*.json"))), 6)

    def test_second_identity_bridge_failure_stops_before_all_extremes(self):
        def bridge(entry, output): self.bridges.append(entry["index"]); return {"passed": entry["index"] == 0}
        result = self.run_fixture(bridge=bridge)
        self.assertEqual(self.calls, [0, 1]); self.assertEqual(result["status"], "stopped_no_replacement")
        self.assertEqual(result["identity_bridges"], ["wave"])
        self.assertEqual(result["attempted_physical_calls"], 2)

    def test_common_range_failure_is_not_a_physical_attempt_or_replaced(self):
        def common(entry):
            if entry["index"] == 2: raise ValueError("fixture continuous range failed")
            return {"passed": True}
        result = self.run_fixture(common=common)
        self.assertEqual(self.calls, [0, 1]); self.assertFalse((self.output / "attempts/02.json").exists())
        self.assertEqual(result["failed_candidate"]["index"], 2)

    def test_pin_change_during_call_stops_immediately_and_preserves_attempt(self):
        changed = False
        def verify():
            if changed: raise ValueError("fixture changed controlled source")
        def execute(entry, output):
            nonlocal changed
            self.calls.append(entry["index"]); changed = True; return manifest()
        result = self.run_fixture(verify=verify, execute=execute)
        self.assertEqual(self.calls, [0]); self.assertEqual(result["status"], "stopped_no_replacement")
        self.assertTrue((self.output / "attempts/00.json").exists()); self.assertEqual(result["completed_audited_trials"], 0)

    def test_execution_and_integrity_failures_stop_without_retry(self):
        for phase in ("execute", "audit"):
            with self.subTest(phase=phase):
                if self.output.exists(): self.output = Path(self.temp.name) / ("newrun_" + phase)
                def failed(entry, output): raise RuntimeError("fixture " + phase + " failed")
                result = self.run_fixture(**{phase: failed})
                self.assertEqual(result["status"], "stopped_no_replacement"); self.assertEqual(result["attempted_physical_calls"], 1)
                self.assertEqual(len(list((self.output / "attempts").glob("*.json"))), 1)

    def test_budget_order_and_existing_output_fail_before_callbacks(self):
        bad = protocol(); bad["prospective_schedule"].append(copy.deepcopy(bad["prospective_schedule"][-1]))
        with self.assertRaises(ValueError): RUNNER.run_schedule(bad, self.output, verify=None, common=None, execute=None, audit=None, bridge=None)
        bad = protocol(); bad["prospective_schedule"][0]["shift_s"] = .6
        with self.assertRaises(ValueError): RUNNER.run_schedule(bad, self.output, verify=None, common=None, execute=None, audit=None, bridge=None)
        self.output.mkdir()
        with self.assertRaises(FileExistsError): RUNNER.run_schedule(protocol(), self.output, verify=None, common=None, execute=None, audit=None, bridge=None)

    def test_design_approval_is_insufficient_for_any_execution(self):
        supplement = Path(self.temp.name) / "supplement.json"; supplement.write_text("{}")
        approval = Path(self.temp.name) / "approval.json"; approval.write_text('{"decision":"APPROVE_FROZEN_SIX_TRIAL_DESIGN_PENDING_PINNED_RUNNER_REVIEW"}')
        with patch.object(RUNNER, "SUPPLEMENT", supplement), patch.object(RUNNER, "OUTPUT", self.output), patch.object(RUNNER, "load_context") as context:
            with self.assertRaises(ValueError): RUNNER.execute_reviewed(approval)
            context.assert_not_called()

    def test_changed_pinned_file_fails_closed(self):
        controlled = Path(self.temp.name) / "controlled.py"; controlled.write_text("fixture_original")
        supplement = {"runner_sha256": RUNNER.sha(Path(RUNNER.__file__)), "file_sha256": {str(controlled): RUNNER.sha(controlled)}}
        with patch.object(RUNNER, "verify_environment"), patch.object(RUNNER.shutil, "disk_usage", return_value=type("Disk", (), {"free": 1024**3})()):
            RUNNER.verify_pins({}, supplement)
            controlled.write_text("fixture_changed")
            with self.assertRaisesRegex(ValueError, "Pinned transitive/source/controller input changed"): RUNNER.verify_pins({}, supplement)


if __name__ == "__main__": unittest.main()
