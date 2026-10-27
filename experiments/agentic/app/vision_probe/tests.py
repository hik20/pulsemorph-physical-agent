"""Offline security/receipt fixtures. Never starts the real CLI or a model."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from provider_inline import CodexInlineVisionProvider, ProviderError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


class Process:
    def __init__(self, args, **kwargs):
        self.stdin = io.BytesIO(); self.returncode = 0
        kwargs["stdout"].write((json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}}) + "\n" + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})).encode())
    def poll(self): return self.returncode


class InlineVisionTests(unittest.TestCase):
    def test_controlled_image_boundary_before_any_process(self):
        with tempfile.TemporaryDirectory() as directory, patch("provider_inline.subprocess.Popen") as process:
            root = Path(directory); outside = root / "outside.png"; outside.write_bytes(b"fixture")
            owned = root / "owned"; owned.mkdir()
            provider = CodexInlineVisionProvider(trusted_local=True, image_root=owned)
            provider.executable = "fixture-codex"
            with self.assertRaises(ValueError): provider.generate_json("fixture", SCHEMA, image=outside)
            process.assert_not_called()

    def test_inline_flag_normal_rules_and_same_disabled_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = root / "fixture.png"; image.write_bytes(b"controlled fixture bytes")
            provider = CodexInlineVisionProvider(trusted_local=True, image_root=root); provider.executable = "fixture-codex"
            with patch("provider_inline.subprocess.Popen", side_effect=Process) as process:
                result = provider.generate_json("fixture", SCHEMA, image=image)
            args = process.call_args.args[0]
            self.assertIn("--image", args); self.assertIn("--output-schema", args)
            self.assertNotIn("--ignore-rules", args); self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", args)
            self.assertEqual(args[args.index("-m") + 1], "gpt-6.1-sol")
            self.assertEqual(args[args.index("-s") + 1], "read-only")
            for feature in ("shell_tool", "apps", "plugins", "browser_use", "computer_use", "image_generation", "hooks", "multi_agent", "view_image"):
                self.assertIn(feature, args)
            self.assertEqual(result.data, {"ok": True})
            self.assertIsNone(result.receipt["model_returned"])
            self.assertEqual(result.receipt["inline_images"], 1)
            self.assertFalse(result.receipt["ignore_rules_used"])

    def test_tool_activity_is_rejected_not_advisory_success(self):
        class ToolProcess(Process):
            def __init__(self, args, **kwargs):
                self.stdin = io.BytesIO(); self.returncode = 0
                kwargs["stdout"].write(json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}).encode())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = root / "fixture.png"; image.write_bytes(b"fixture")
            provider = CodexInlineVisionProvider(trusted_local=True, image_root=root); provider.executable = "fixture-codex"
            with patch("provider_inline.subprocess.Popen", side_effect=ToolProcess), self.assertRaises(ProviderError):
                provider.generate_json("fixture", SCHEMA, image=image)


if __name__ == "__main__": unittest.main()
