"""Read-only uploaded-audio job audit; no model, physics stepping or memory IO."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import wave

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from audit_jobs import (read, sha, require, audit_calls, audit_targets, audit_selection,
                        audit_implementation, audit_media, audit_physics, PhysicsAuditor)


def audit_music(job, root=ROOT):
    from pulsemorph.agentic.app.uploads import StudioMusic
    observed = job["music"]
    require(observed["id"].startswith("upload_"), "job did not use uploaded music")
    descriptor = StudioMusic(root).get(observed["id"])
    require({key: value for key, value in descriptor.items() if key != "path"} == observed,
            "actual uploaded descriptor differs from job music")
    path = Path(descriptor["path"])
    require(sha(path) == observed["sha256"], "canonical WAV hash differs")
    with wave.open(str(path), "rb") as stream:
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) == (1, 2, 48000, 192000), "canonical WAV format differs")
        require(len(stream.readframes(192000)) == 384000, "canonical PCM truncated")
    provenance = descriptor["provenance"]
    require(provenance["source_bytes_retained"] is False, "service claims full source retention")
    require(provenance["derived_wav_sha256"] == descriptor["sha256"], "derived hash differs")
    require(provenance["start_s"] == provenance["start_frame"] / provenance["original_sample_rate"] and
            provenance["end_s"] - provenance["start_s"] == 4, "source crop clock differs")
    require(descriptor["bpm"] is None and descriptor["beat_times_s"] == [], "uploaded tempo incorrectly asserted")
    return {"state": "verified", "id": descriptor["id"], "sha256": descriptor["sha256"],
            "measured_accent_s": descriptor["accent_time_s"], "provenance": provenance}, descriptor


def audit_source_witness(descriptor, root=ROOT):
    """Optional separately retained procedural smoke source, confined to its dir."""
    import numpy as np
    from scipy.signal import resample_poly
    root = Path(root).resolve()
    smoke = root / "artifacts/runs/agentic/app_user_music_v1"
    require(smoke.resolve() == smoke, "smoke directory resolves outside its expected path")
    witness_path = smoke / "source_input.json"
    if not witness_path.is_file():
        return {"state": "not_available", "service_source_retention_claim": False}
    require(witness_path.resolve().is_relative_to(smoke.resolve()) and not witness_path.is_symlink(), "source witness path escapes smoke directory")
    witness = read(witness_path)
    relative = Path(witness["destination"].replace("\\", "/"))
    require(not relative.is_absolute(), "source witness must use project-relative path")
    source = (root / relative).resolve()
    require(smoke.resolve() in source.parents and not (root / relative).is_symlink(), "source witness path escapes smoke directory")
    require(source.is_file() and source.stat().st_size <= 32 * 1024 * 1024, "source witness unavailable or oversized")
    require(sha(source) == witness["sha256"] == descriptor["provenance"]["source_sha256"], "source witness hash differs")
    with wave.open(str(source), "rb") as stream:
        rate, channels, frames = stream.getframerate(), stream.getnchannels(), stream.getnframes()
        require(stream.getsampwidth() == 2 and stream.getcomptype() == "NONE", "source witness must be PCM16")
        raw = stream.readframes(frames)
    provenance = descriptor["provenance"]
    require((rate, channels, frames) == (44100, 2, 352800), "expected eight-second stereo44100Hz smoke witness")
    require((rate, channels, frames) == (provenance["original_sample_rate"], provenance["original_channels"], provenance["original_frame_count"]), "source witness format differs from import provenance")
    require(len(raw) == frames * channels * 2 and provenance["read_frame_count"] == frames, "source witness sample count differs")
    require((provenance["start_s"], provenance["end_s"], provenance["start_frame"], provenance["selected_frame_count"]) == (2., 6., 88200, 176400), "expected source-clock crop2..6seconds")
    require(witness["clip_start_s"] == 2 and witness["clip_duration_s"] == 4 and witness["channels"] == channels and witness["sample_rate"] == rate and witness["duration_s"] == frames / rate, "source witness receipt differs")
    samples = np.frombuffer(raw, "<i2").reshape(frames, channels)
    mono = samples[88200:264600].astype(np.float64).mean(axis=1)
    common = math.gcd(rate, 48000)
    clip = np.clip(np.rint(resample_poly(mono, 48000 // common, rate // common)), -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, 48000, 192000, "NONE", "not compressed")); stream.writeframes(clip.tobytes())
    require(hashlib.sha256(output.getvalue()).hexdigest() == descriptor["sha256"], "independently reconstructed crop hash differs")
    return {"state": "verified", "source_path": source.relative_to(root).as_posix(), "sha256": sha(source),
            "source_channels": channels, "source_sample_rate": rate, "source_frames": frames,
            "source_crop_s": [2, 6], "reconstructed_derived_sha256": descriptor["sha256"],
            "scope": "separately retained procedural smoke witness; service does not retain uploaded full source"}


def run_audit(root=ROOT, job_id="20261027T031331_adc16d6a"):
    root = Path(root).resolve()
    require(job_id.replace("_", "").isalnum(), "invalid job identity")
    directory = root / "artifacts/runs/agentic/app/jobs" / job_id
    job = read(directory / "state.json")
    require(job["id"] == job_id and job["status"] == "complete", "uploaded job is not complete")
    result = {"schema": "pulsemorph.uploaded_job_audit.v1", "at": datetime.now(timezone.utc).isoformat(),
              "read_only": True, "job_id": job_id, "new_model_calls": 0, "new_physics_steps": 0,
              "memory_operations": 0, "state_sha256": sha(directory / "state.json")}
    result["model"] = audit_calls(directory, job)
    result["targets"] = audit_targets(directory, job)
    result["selection"] = audit_selection(job)
    result["implementation"] = audit_implementation(directory, job)
    result["media"] = audit_media(directory, job)
    result["music"], descriptor = audit_music(job, root)
    result["source_witness"] = audit_source_witness(descriptor, root)
    auditor = PhysicsAuditor(root)
    result["physics"] = audit_physics(directory, job, auditor)
    require(len(result["physics"]) == 2 and all(row["state"] == "independently_verified" and row["evaluation"] == "matches_current_evaluator" for row in result["physics"]), "both new physical trials must match current runtime/evaluation")
    require(job["outcome"] == "target_met" and job["selected"]["metrics"]["per_dimension_pass"] == [True] * 5, "selected result does not meet all five fixed targets")
    require(result["model"]["completed"] == result["model"]["live_completed"] == 3, "expected three completed live calls")
    result.update(passed=True, physical_trials_audited=2, selected_targets_passed=5,
                  current_runtime_sha256=sha(root / "src/pulsemorph/agentic/app/runtime.py"),
                  script_sha256=sha(Path(__file__)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default="20261027T031331_adc16d6a")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Choose a new audit receipt")
    result = run_audit(job_id=args.job_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed": True, "job_id": args.job_id, "physical_trials_audited": 2, "output": str(args.output)}))


if __name__ == "__main__": main()
