"""Prepare a local source/evidence draft. Never publishes, installs or submits."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import re
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "artifacts/runs/agentic/app_demo_package_v1"
JOBS = ["20261027T022051_538b6498", "20261027T022322_76db8965", "20261027T023422_8efe4bc4"]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "artifacts/runs/agentic") or out.exists():
        raise ValueError("Choose a new dedicated agentic output directory")
    kit = out / "sourcekit"
    kit.mkdir(parents=True)
    include = set()
    for folder in ["src/pulsemorph/agentic/app"]:
        include.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".js", ".css", ".html"})
    explicit = ["pyproject.toml", "src/pulsemorph/__init__.py", "src/pulsemorph/evidence.py", "src/pulsemorph/agentic/__init__.py",
                "src/pulsemorph/agentic/reference.py", "src/pulsemorph/agentic/common_player.py", "src/pulsemorph/agentic/multifeature_reference.py", "src/pulsemorph/agentic/local_multifeature_reference.py",
                "src/pulsemorph/motion/__init__.py", "src/pulsemorph/motion/contracts.py", "src/pulsemorph/motion/g1.py",
                "experiments/agentic/app/execute_app.py", "experiments/agentic/app/meterless_bridge.ts", "experiments/agentic/app/start.ps1", "experiments/agentic/app/preflight.py", "experiments/agentic/app/requirements-app.txt", "experiments/agentic/app/README.md",
                "experiments/agentic/app/audit_jobs.py", "experiments/agentic/app/audit_uploaded_job.py", "experiments/agentic/app/make_upload_smoke.py", "experiments/agentic/execute_tool_selection_compatible.py", "experiments/agentic/execute_generalization.py", "experiments/agentic/execute_local_capability.py", "experiments/agentic/evaluate_generalization.py", "experiments/agentic/execute_trial.py",
                "experiments/agentic/evaluate_trial.py", "experiments/agentic/evaluate_multifeature.py", "experiments/causal_music/reference.py", "experiments/music_control/accents.py", "experiments/music_control/evaluate.py",
                "design/agentic/APP_ARCHITECTURE.md", "design/coordination/MOTION_REFERENCE_CONTRACT.md"]
    include.update(ROOT / relative for relative in explicit)
    include.update((ROOT / "tests").glob("test_agentic_app_*.py"))
    include.update(path for path in (ROOT / "experiments/agentic/app/bootstrap").glob("*")
                   if path.is_file() and path.suffix in {".py", ".md", ".lock"})
    # The current executor snapshots historical source bytes as provenance.
    # These dependencies matter even when the module is never imported/executed.
    required_source_paths = set()
    pending = list(include)
    inspected = set()
    while pending:
        path = pending.pop()
        if path in inspected or path.suffix != ".py": continue
        inspected.add(path)
        content = path.read_text(encoding="utf-8-sig")
        dependencies = [ROOT / name for name in re.findall(r"ROOT\s*/\s*['\"]([^'\"]+\.py)['\"]", content)]
        dependencies += [path.parent / name for name in re.findall(r"with_name\(['\"]([^'\"]+\.py)['\"]\)", content)]
        for dependency in dependencies:
            if "research/vendor" in dependency.as_posix(): continue
            if not dependency.is_file(): raise ValueError("Missing required source dependency: " + str(dependency))
            required_source_paths.add(dependency.relative_to(ROOT).as_posix())
            if dependency not in include:
                include.add(dependency); pending.append(dependency)
    files, compiled = [], 0
    for path in sorted(include):
        relative = path.relative_to(ROOT)
        if not path.is_file() or "__pycache__" in relative.parts:
            raise ValueError("Missing explicit source or invalid package path")
        if path.suffix == ".py":
            compile(path.read_bytes(), relative.as_posix(), "exec")
            compiled += 1
        destination = kit / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        files.append({"path": relative.as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    pins = {"ProtoMotions": {"commit": "f69fdcaabff31846b4d2247207417b57dc788828", "source": "https://github.com/NVlabs/ProtoMotions", "onnx_sha256": "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c"},
            "Meterless": {"commit": "1742dec56a66e58eef9dda2d2b865823758ca8cd", "source": "https://github.com/Meterless/Meterless"},
            "motions": {"wave": "cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93", "high5": "ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44"}}
    (kit / "DEPENDENCY_PINS.json").write_text(json.dumps(pins, indent=2), encoding="utf-8")
    readme = """# PulseMorph local source kit draft

This kit contains current first-party app sources and the shared helper sources they load. It is for local review in the existing Windows workspace. It is not a tested clean-machine distribution and it is not an exact source snapshot of every earlier job.

Read experiments/agentic/app/README.md, then run its read-only preflight with the existing .venv-motor interpreter. Existing selective ProtoMotions and Meterless checkouts, controller ONNX/YAML, G1 meshes and source motions, Node/tsx, FFmpeg and the motor environment remain prerequisites. No vendor source, weights, motion data, credentials, authentication stores, environment binaries or large frozen artifacts are included. DEPENDENCY_PINS.json identifies the required external versions/bytes. Authentication belongs to the user; no login is bundled.

Current runtime layout targets Windows. The workspace motor interpreter remains the fallback; the launcher/preflight also accept an explicit validated isolated runtime. Read experiments/agentic/app/README.md and bootstrap/README.md for the selected-interpreter command, executed clone pilot and its receipts. The full public-wheel fresh-machine recipe remains unexecuted. Additional app requirements supplement an already compatible motor environment. Check code, checkpoint, motion, mesh and media terms independently before any public redistribution. This source kit grants no blanket redistribution license.

The local product demo works with bounded four-second edits, not weight learning or arbitrary new choreography. StudioMusic also imports a four-second segment from PCM16 WAV (mono/stereo8–96kHz,4–120s,max32MiB) and retains only its derived48kHzmonoPCM16 clip plus full-source-hash/crop provenance. Uploaded BPM and redistribution rights remain unknown. No uploaded media is included here. A procedural-input browser smoke generator is included but not automatically run. Meterless local persistence is real. Zetaris has no live query evidence and remains an open competition requirement. This draft is not submission-ready. Nothing was published or submitted by packaging it.
"""
    (kit / "README.md").write_text(readme, encoding="utf-8")
    manifest = {"schema": "pulsemorph.app_sourcekit.v1", "created_at": datetime.now(timezone.utc).isoformat(), "scope": "current-source review kit; existing-workspace prerequisites", "clean_external_install_tested": False,
                "public_publish_or_submission": False, "python_sources_compile_checked": compiled, "sources": files, "external_dependencies": pins,
                "excluded": ["vendor source", "large weights and datasets", "source motion files", "auth/environment files", "job telemetry arrays", "private CLI stores"],
                "required_source_paths_verified": sorted(required_source_paths),
                "provenance_snapshot_dependency_note": "Historical executor/helper source bytes are included read-only because execute_app snapshots them; their experiments are not replayed by packaging.",
                "rights": "Review draft only. No blanket redistribution permission asserted."}
    (kit / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = out / "pulsemorph-sourcekit-draft.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(kit.rglob("*")):
            if path.is_file(): zipped.write(path, "pulsemorph-sourcekit/" + path.relative_to(kit).as_posix())
    summaries = []
    for identity in JOBS:
        folder = ROOT / "artifacts/runs/agentic/app/jobs" / identity
        job = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        selected = job.get("selected")
        summaries.append({"id": identity, "state_sha256": sha(folder / "state.json"), "status": job["status"], "request": job["request"], "live": job["live"], "outcome": job["outcome"], "targets": job.get("targets"), "baseline": job.get("baseline"),
                          "selected": selected, "model_call_count": len(job["model_calls"]), "physical_trial_count": len(list((folder / "trials").glob("*/manifest.json"))), "proposal_pass_counts": [sum(t["metrics"]["per_dimension_pass"]) for t in job["trials"]],
                          "scope": "archived completed job; original selected result preserved"})
    (out / "JOB_EVIDENCE_SUMMARY.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "PACKAGE_MANIFEST.json").write_text(json.dumps({"schema": "pulsemorph.app_demo_draft.v1", "submission_ready": False, "open_requirements": ["live authorized Zetaris context query", "accessible authorized source repository", "clean external setup and redistribution review", "final 1–4 minute delivery video and access verification"], "source_archive": {"path": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size}, "source_files": len(files), "python_sources_compile_checked": compiled}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "source_files": len(files), "python_compile_checked": compiled, "archive_bytes": archive.stat().st_size}))


if __name__ == "__main__": main()
