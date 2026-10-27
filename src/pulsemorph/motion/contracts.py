"""Validated motion targets, not physical-control or aesthetic guarantees.

The SHA-256 binds numeric payload bytes to the manifest; it is not a signature
and does not authenticate metadata or the producer.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

SCHEMA_VERSION = "pulsemorph.motion_reference.v1"
_ARRAYS = ("time_offsets_s", "joint_position_rad", "joint_velocity_rad_s", "anchor_orientation_xyzw")


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _has_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_boolean(item) for item in value)
    return False


@dataclass(frozen=True, eq=False)
class MotionReference:
    metadata: dict
    time_offsets_s: np.ndarray
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    anchor_orientation_xyzw: np.ndarray

    def __post_init__(self):
        try:
            # JSON roundtrip enforces a portable metadata vocabulary and ownership.
            metadata = json.loads(json.dumps(_plain(self.metadata), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON values") from exc
        object.__setattr__(self, "metadata", _freeze(metadata))
        for name in _ARRAYS:
            supplied = getattr(self, name)
            if _has_boolean(supplied):
                raise ValueError(f"{name} cannot contain boolean numbers")
            raw = np.asarray(supplied)
            if raw.dtype.kind not in "iuf":
                raise ValueError(f"{name} must have a real numeric dtype")
            # An immutable bytes backing store prevents setflags(write=True).
            owned = np.frombuffer(raw.astype(np.float64).tobytes(), dtype=np.float64).reshape(raw.shape)
            object.__setattr__(self, name, owned)
        self.validate()

    def validate(self) -> None:
        m = self.metadata
        if not isinstance(m, Mapping) or m.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported motion reference schema_version")
        _text(m.get("reference_id"), "reference_id")
        fields = {
            "producer": ("kind", "id", "version"),
            "body": ("profile_id", "profile_sha256", "anchor_body_name"),
            "controller": ("profile_id", "profile_sha256"),
            "frame": ("id", "up_axis", "handedness", "quaternion_order", "alignment"),
            "context": ("decision_id", "observation_id", "task_id", "runtime_session_id", "clock_epoch"),
            "validity": ("start_state_id", "start_mode", "end_behavior"),
            "observation": ("mode",),
            "provenance": (),
        }
        for group, names in fields.items():
            if not isinstance(m.get(group), Mapping):
                raise ValueError(f"missing metadata group {group}")
            for name in names:
                _text(m[group].get(name), f"{group}.{name}")
        if m["producer"]["kind"] not in ("agentic", "learned", "baseline"):
            raise ValueError("unsupported producer kind")
        if "seed" in m["producer"] and (type(m["producer"]["seed"]) is not int or m["producer"]["seed"] < 0):
            raise ValueError("producer.seed must be a nonnegative integer")
        for group in ("body", "controller"):
            _hash(m[group]["profile_sha256"], f"{group}.profile_sha256")
        if _number(m["controller"].get("control_dt_s"), "control_dt_s") <= 0:
            raise ValueError("control_dt_s must be positive")
        for key, expected in {"up_axis": "z", "handedness": "right", "quaternion_order": "xyzw", "alignment": "recorded_reference_world"}.items():
            if m["frame"][key] != expected:
                raise ValueError(f"unsupported frame {key}")
        names = m["body"].get("joint_names")
        if not isinstance(names, tuple) or not names:
            raise ValueError("body.joint_names must be a nonempty list")
        for name in names:
            _text(name, "joint_names entry")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        t = self.time_offsets_s
        if t.ndim != 1 or len(t) < 2 or t[0] != 0 or not np.all(np.diff(t) > 0):
            raise ValueError("time_offsets_s must start at zero and strictly increase, H >= 2")
        for name, shape in (("joint_position_rad", (len(t), len(names))), ("joint_velocity_rad_s", (len(t), len(names))), ("anchor_orientation_xyzw", (len(t), 4))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for name in _ARRAYS:
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains nonfinite values")
        if not np.allclose(np.linalg.norm(self.anchor_orientation_xyzw, axis=1), 1, rtol=0, atol=1e-5):
            raise ValueError("anchor orientations must be unit quaternions")
        decision = _number(m["context"].get("decision_time_s"), "decision_time_s")
        start = _number(m["validity"].get("start_time_s"), "start_time_s")
        end = _number(m["validity"].get("end_time_s"), "end_time_s")
        if not math.isclose(end, start + float(t[-1]), rel_tol=0, abs_tol=1e-9):
            raise ValueError("end_time_s must equal start_time_s + final offset")
        obs = m["observation"]
        if obs["mode"] not in ("offline_file", "live_causal"):
            raise ValueError("unsupported observation mode")
        observed = _number(obs.get("observed_through_time_s"), "observed_through_time_s")
        for key in ("audio_observed_through_sample", "video_observed_through_time_s"):
            if key not in obs:
                raise ValueError(f"missing observation.{key}")
        sample = obs["audio_observed_through_sample"]
        if sample is not None and (type(sample) is not int or sample < 0):
            raise ValueError("audio sample must be null or a nonnegative integer")
        video = obs["video_observed_through_time_s"]
        if video is not None:
            _number(video, "video_observed_through_time_s")
        if obs["mode"] == "live_causal" and (observed > decision or (video is not None and video > decision)):
            raise ValueError("live observations cannot extend beyond decision time")
        provenance = m["provenance"]
        if not isinstance(provenance.get("sources"), tuple) or not isinstance(provenance.get("notes"), tuple):
            raise ValueError("provenance sources and notes must be lists")
        for source in provenance["sources"]:
            if not isinstance(source, Mapping):
                raise ValueError("provenance source must be an object")
            _text(source.get("path"), "source.path")
            _hash(source.get("sha256"), "source.sha256")
        for note in provenance["notes"]:
            _text(note, "provenance note")


def save_reference(reference: MotionReference, directory: Path) -> Path:
    """Create a new immutable package directory; never overwrite an artifact."""
    reference.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = BytesIO()
    np.savez_compressed(payload, **{name: getattr(reference, name) for name in _ARRAYS})
    data = payload.getvalue()
    with (directory / "arrays.npz").open("xb") as stream:
        stream.write(data)
    manifest = {"metadata": _plain(reference.metadata), "arrays": {"path": "arrays.npz", "sha256": sha256(data).hexdigest()}}
    path = directory / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2)
    return path


def load_reference(directory_or_manifest: Path) -> MotionReference:
    """Verify the payload before NumPy parses it (metadata is not signed)."""
    path = Path(directory_or_manifest)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    spec = manifest.get("arrays", {})
    if not isinstance(spec, dict):
        raise ValueError("manifest arrays must be an object")
    # Fixed basename also rejects platform-dependent absolute and traversal paths.
    if spec.get("path") != "arrays.npz":
        raise ValueError("array payload path must be arrays.npz")
    payload_path = path.parent / "arrays.npz"
    if payload_path.is_symlink() or payload_path.resolve().parent != path.parent.resolve():
        raise ValueError("array payload must remain inside package")
    _hash(spec.get("sha256"), "arrays.sha256")
    data = payload_path.read_bytes()
    if sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("array payload SHA-256 mismatch")
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAYS) or len(archive.files) != len(_ARRAYS):
            raise ValueError("array payload keys do not match contract")
        arrays = {name: archive[name] for name in _ARRAYS}
    if "metadata" not in manifest:
        raise ValueError("manifest metadata is required")
    return MotionReference(metadata=manifest["metadata"], **arrays)
