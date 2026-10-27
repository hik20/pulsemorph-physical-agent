"""Admission/schema fixtures for a separate Walk-only research wrapper."""
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("walk_probe_fixture", ROOT / "experiments/agentic/app/execute_walk_probe_v1.py")
PROBE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(PROBE)
WALK = "research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_walk.motion"
HASH = "09061b8fa0df9e99daaf857293b32cfa8d2faa68631ff20ed0fdde0683e80884"


class WalkProbeTests(unittest.TestCase):
    def test_only_exact_walk_source_hash_admitted_without_deserialization(self):
        path, digest = PROBE.resolve_source_motion({"source_motion": {"path": WALK, "sha256": HASH}})
        self.assertEqual(digest, HASH)
        self.assertEqual(path, ROOT / WALK)
        for entry in [{"path": WALK, "sha256": "0" * 64}, {"path": WALK.replace("walk", "wave"), "sha256": "cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93"}, {"path": "../output_walk.motion", "sha256": HASH}]:
            with self.assertRaises(ValueError): PROBE.resolve_source_motion({"source_motion": entry})
        with patch.object(PROBE, "sha256", return_value="0" * 64), self.assertRaises(ValueError):
            PROBE.resolve_source_motion({"source_motion": {"path": WALK, "sha256": HASH}})

    def test_probe_schema_and_producer_are_distinct_from_app(self):
        proposal = {"schema": "pulsemorph.source_capability_probe_proposal.v1", "control_tool": "global_clock_v1"}
        self.assertEqual(PROBE.proposal_producer(proposal)["id"], "pulsemorph-walk-capability-probe-v1")
        with self.assertRaises(ValueError): PROBE.proposal_producer(dict(proposal, schema="pulsemorph.agentic_app_proposal.v1"))

    def test_execution_mechanics_ast_only_declared_provenance_strings_change(self):
        original = ast.parse((ROOT / "experiments/agentic/app/execute_app.py").read_text())
        probe = ast.parse((ROOT / "experiments/agentic/app/execute_walk_probe_v1.py").read_text())
        names = {node.name: node for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        mapped = {"pulsemorph.source_capability_probe_trial.v1": "pulsemorph.agentic_app_trial.v1", "source_capability_probe": "app_physical_trial_v1"}
        class RestoreStrings(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    if node.value == "source_capability_probe":
                        # execute_trial has two historical provenance strings.
                        node.value = "app_physical_trial_v1"
                return node
        for candidate in [node for node in probe.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]:
            if candidate.name in {"resolve_source_motion", "proposal_producer", "execute_trial"}: continue
            self.assertEqual(ast.dump(candidate, include_attributes=False), ast.dump(names[candidate.name], include_attributes=False))
        original_trial = names["execute_trial"]
        probe_trial = next(node for node in probe.body if isinstance(node, ast.FunctionDef) and node.name == "execute_trial")
        # Compare assignment-by-assignment after dropping only the known manifest provenance constants.
        class NormalizeProvenance(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value in ("pulsemorph.source_capability_probe_trial.v1", "pulsemorph.agentic_app_trial.v1"): node.value = "TRIAL_SCHEMA"
                elif node.value in ("source_capability_probe", "app_physical_trial_v1", "app_physical_trial"): node.value = "EXPERIMENT_PROVENANCE"
                return node
        self.assertEqual(ast.dump(NormalizeProvenance().visit(probe_trial), include_attributes=False), ast.dump(NormalizeProvenance().visit(original_trial), include_attributes=False))


if __name__ == "__main__": unittest.main()
