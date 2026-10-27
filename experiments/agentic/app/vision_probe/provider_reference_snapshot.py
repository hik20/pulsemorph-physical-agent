"""Codex bridge for private local trusted requests, never a public service."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from threading import Event
from typing import Any


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResult:
    data: dict[str, Any]
    receipt: dict[str, Any]


class CodexLocalProvider:
    name = "codex_local"
    model = "gpt-6.1-sol"

    def __init__(self, *, trusted_local: bool = False):
        self.trusted_local = trusted_local
        self.executable = shutil.which("codex.cmd") or shutil.which("codex")

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": bool(self.executable),
                "enabled": self.trusted_local and bool(self.executable),
                "model_requested": self.model, "trusted_local_only": True,
                "authentication_verified": False}

    def generate_json(self, prompt: str, schema: dict[str, Any], *, timeout: float = 120,
                      cancel_event: Event | None = None) -> ProviderResult:
        if not self.trusted_local:
            raise ProviderError("Explicit private local trust is required")
        if not self.executable:
            raise ProviderError("Codex CLI is unavailable")
        if not 0 < timeout <= 600:
            raise ValueError("timeout must be between 0 and 600 seconds")
        if cancel_event and cancel_event.is_set():
            raise ProviderError("Generation cancelled")
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="pulsemorph-provider-") as directory:
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            args = [self.executable, "exec", "--ignore-user-config",
                    "--ephemeral", "--skip-git-repo-check", "-C", directory,
                    "-m", self.model, "-s", "read-only", "-c", 'approval_policy="never"',
                    "-c", 'web_search="disabled"', "-c", "agents.enabled=false"]
            for feature in ("shell_tool", "shell_snapshot", "apps", "plugins", "browser_use",
                            "computer_use", "image_generation", "hooks", "multi_agent",
                            "view_image", "workspace_dependencies"):
                args.extend(["--disable", feature])
            args.extend(["--output-schema", str(schema_path), "--json", "-"])
            # File-backed streams prevent output pipe deadlock. Never retain raw diagnostics.
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    process.stdin.write(("Return JSON matching schema. Do not use tools.\n" + prompt).encode())
                    process.stdin.close()
                    while process.poll() is None:
                        if cancel_event and cancel_event.is_set():
                            raise ProviderError("Generation cancelled")
                        if time.monotonic() - started > timeout:
                            raise ProviderError("Generation timed out")
                        time.sleep(0.05)
                    if process.returncode:
                        raise ProviderError("Codex failed; check local authentication/model access")
                    out.seek(0)
                    raw = out.read().decode("utf-8")
                finally:
                    if process.poll() is None:
                        import os
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            process.kill()
                        process.wait()
        try:
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
            messages, usage = [], {}
            for event in events:
                item = event.get("item", {})
                if item and item.get("type") != "agent_message":
                    raise ProviderError("Unexpected tool activity; response rejected")
                if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                    messages.append(item["text"])
                if event.get("type") == "turn.completed":
                    usage = {k: v for k, v in event.get("usage", {}).items()
                             if k.endswith("tokens") and isinstance(v, int)}
            data = json.loads(messages[-1])
            import jsonschema
            jsonschema.validate(data, schema)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Provider response failed JSON/schema validation") from exc
        return ProviderResult(data, {"provider": self.name, "model_requested": self.model,
            "model_returned": None, "usage": usage, "live": True, "tool_events": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "trust_boundary": "private_local_trusted_requests_only"})
