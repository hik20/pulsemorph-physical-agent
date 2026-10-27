"""Exactly three preregistered advisory inline-image calls, with no tools."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from provider_inline import CodexInlineVisionProvider

OUT = ROOT / "artifacts/runs/agentic/app_vision_probe_v1"
MEDIA = ROOT / "artifacts/media/agentic/app_vision_probe_v1"
SCHEMA = {"type": "object", "properties": {
    "can_tell_difference": {"type": "boolean"},
    "observations": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 4},
    "uncertainty": {"type": "string", "maxLength": 500},
    "frame_indices": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 4}, "maxItems": 4}},
    "required": ["can_tell_difference", "observations", "uncertainty", "frame_indices"], "additionalProperties": False}
PROMPT = """Inspect the attached image only. It contains two rows A and B, with four matched columns indexed1..4 at times0.8,1.6,2.4,3.2seconds.
Compare visible body pose/motion differences between rows at matched columns. If no meaningful visible difference is discernible, say so and set can_tell_difference=false. Do not invent a difference.
Report at most four short observations and the frame indices that support them. State uncertainty honestly. Four still images do not establish continuous motion.
Do not grade beauty, preference, safety, balance, physical execution, intent compliance, model quality or which row is better. Do not infer hidden gains or source conditions.
This is advisory visual observation only and grants no execution authority. Do not use tools or read any files. Return only the requested JSON, without hidden reasoning."""


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    if (OUT / "protocol.json").exists(): raise FileExistsError("Probe already preregistered; no automatic repeats or additional calls")
    render = json.loads((OUT / "render_manifest.json").read_text())
    inspection = json.loads((OUT / "capability_inspection.json").read_text())
    if sha(ROOT / "src/pulsemorph/agentic/app/providers.py") != inspection["original_provider_sha256"] or sha(HERE / "provider_inline.py") != inspection["copied_provider_sha256"]:
        raise ValueError("Provider/security source pins changed")
    if len(render["cases"]) != 3 or not all(render["cases"][0]["body_panel_A_B_pixel_equal"]):
        raise ValueError("Preregistered controls incomplete")
    for case in render["cases"]:
        if sha(ROOT / case["image_path"]) != case["image_sha256"]: raise ValueError("Blind input image changed")
    protocol = {"schema": "pulsemorph.advisory_inline_vision_protocol.v1", "at": datetime.now(timezone.utc).isoformat(),
                "maximum_logical_model_calls": 3, "requested_model": "gpt-6.1-sol", "normal_private_ChatGPT_auth": True,
                "ignore_rules_used": False, "provider_security_profile": "same disabled-tool flags/temp cwd/read-only sandbox as original provider; one controlled inline PNG extension",
                "cases": render["cases"], "response_schema": SCHEMA, "prompt": PROMPT,
                "expected_control_behavior": {"v01": "false: exact body pixels duplicated", "v02": "true: clearly changed kinematic-only arm poses; not a physical trial", "v03": "no forced expected answer; actual archived physical edit may be ambiguous"},
                "decision_rule": "Null false positive or positive-control miss is a failure. Real-case uncertainty is unresolved, never forced to pass. Three cases are not a benchmark or human validation.",
                "stop_rule": "Any CLI/image/model/schema/tool-activity failure stops all remaining calls; no retries, model/provider/account changes or extra smoke calls.",
                "execution_authority": "none; advisory only", "new_physics_steps": 0, "new_ONNX_calls": 0,
                "image_manifest_sha256": sha(OUT / "render_manifest.json"), "inspection_sha256": sha(OUT / "capability_inspection.json"),
                "source_hashes": {p.name: sha(p) for p in (HERE / "render_cases.py", HERE / "provider_inline.py", Path(__file__))}}
    save(OUT / "protocol.json", protocol)
    provider = CodexInlineVisionProvider(trusted_local=True, image_root=MEDIA)
    results = []
    for index, case in enumerate(render["cases"]):
        folder = OUT / "calls" / case["case_id"]; folder.mkdir(parents=True, exist_ok=False)
        request = {"prompt": PROMPT, "schema": SCHEMA, "image_sha256": case["image_sha256"], "case_id": case["case_id"], "logical_call_index": index}
        save(folder / "request.json", request)
        save(folder / "attempt.json", {"status": "started", "at": datetime.now(timezone.utc).isoformat(), "logical_call_index": index, "protocol_sha256": sha(OUT / "protocol.json")})
        try:
            if sha(ROOT / case["image_path"]) != case["image_sha256"]: raise ValueError("Image changed before call")
            result = provider.generate_json(PROMPT, SCHEMA, image=ROOT / case["image_path"], timeout=180)
            save(folder / "response.json", result.data)
            receipt = dict(result.receipt, request_sha256=sha(folder / "request.json"), response_sha256=sha(folder / "response.json"), case_id=case["case_id"], logical_call_index=index)
            save(folder / "receipt.json", receipt)
            expected = False if index == 0 else True if index == 1 else None
            results.append({"case_id": case["case_id"], "condition_hidden_from_model": case["kind_hidden_from_prompt"], "response": result.data,
                            "control_result": "unresolved_real_case" if expected is None else "passed" if result.data["can_tell_difference"] is expected else "failed",
                            "receipt": receipt})
        except Exception as exc:
            save(folder / "failure.json", {"status": "completion_unconfirmed", "error_type": type(exc).__name__, "safe_error": "Inline image/model/schema/tool validation failed; no retry or provider change", "raw_diagnostics_retained": False})
            save(OUT / "stopped.json", {"logical_calls_attempted": index + 1, "completed_calls": len(results), "results": results, "reason": "Unsupported or failed inline vision cannot establish feasibility; remaining calls stopped", "new_physics_steps": 0})
            raise
    save(OUT / "result.json", {"schema": "pulsemorph.advisory_inline_vision_result.v1", "logical_calls_attempted": 3, "completed_calls": 3,
                              "results": results, "new_physics_steps": 0, "new_ONNX_calls": 0, "execution_authority": "none",
                              "scope": "Three tiny advisory still-image cases, not a benchmark/human validation or completed VLM teacher capability."})
    print(json.dumps({"logical_calls": 3, "completed": 3, "control_results": [r["control_result"] for r in results], "new_physics": 0}))


if __name__ == "__main__": main()
