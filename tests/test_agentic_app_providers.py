"""Fixtures verify bridge behavior; they are not live model evidence."""
import io
import json
import unittest
from unittest.mock import patch
from threading import Event
from pulsemorph.agentic.app.providers import CodexLocalProvider, ProviderError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


class FakeProcess:
    def __init__(self, args, **kwargs):
        self.stdin = io.BytesIO()
        self.returncode = 0
        kwargs["stdout"].write((json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": '{"ok":true}'}}) + "\n" +
            json.dumps({"type": "turn.completed", "usage": {"output_tokens": 5}})).encode())
    def poll(self):
        return 0


class ProviderTests(unittest.TestCase):
    def test_requires_private_trust(self):
        with self.assertRaises(ProviderError):
            CodexLocalProvider().generate_json("test", SCHEMA)

    def test_fixture_response_and_security_arguments(self):
        with patch("shutil.which", return_value="codex.cmd"), patch(
                "subprocess.Popen", side_effect=FakeProcess) as popen:
            result = CodexLocalProvider(trusted_local=True).generate_json("test", SCHEMA)
        self.assertEqual(result.data, {"ok": True})
        self.assertIsNone(result.receipt["model_returned"])
        args = popen.call_args.args[0]
        self.assertIn("--ignore-user-config", args)
        self.assertNotIn("--ignore-rules", args)
        self.assertIn("shell_tool", args)
        self.assertIn("gpt-6.1-sol", args)

    def test_cancel_before_launch(self):
        event = Event()
        event.set()
        with patch("shutil.which", return_value="codex.cmd"), self.assertRaisesRegex(
                ProviderError, "cancelled"):
            CodexLocalProvider(trusted_local=True).generate_json("test", SCHEMA, cancel_event=event)

    def test_rejects_tool_event(self):
        class ToolProcess(FakeProcess):
            def __init__(self, args, **kwargs):
                super().__init__(args, **kwargs)
                kwargs["stdout"].write(b'\n{"type":"item.started","item":{"type":"command_execution"}}')
        with patch("shutil.which", return_value="codex.cmd"), patch(
                "subprocess.Popen", side_effect=ToolProcess), self.assertRaisesRegex(
                    ProviderError, "tool activity"):
            CodexLocalProvider(trusted_local=True).generate_json("test", SCHEMA)

    def test_rejects_invalid_schema_response(self):
        bad_schema = {"type": "object", "required": ["missing"]}
        with patch("shutil.which", return_value="codex.cmd"), patch(
                "subprocess.Popen", side_effect=FakeProcess), self.assertRaisesRegex(
                    ProviderError, "schema validation"):
            CodexLocalProvider(trusted_local=True).generate_json("test", bad_schema)
