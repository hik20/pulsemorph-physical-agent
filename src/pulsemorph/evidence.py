"""Content-addressed evidence inventory; validates actual local artifact bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(root: Path, path: Path, *, role: str) -> dict:
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence must be inside the project root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size, "role": role}


def verify(root: Path, entries: list[dict]) -> list[dict]:
    results = []
    for entry in entries:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Evidence path escapes project root")
        exists = path.is_file()
        matches = exists and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]
        results.append({"path": entry["path"], "exists": exists, "matches": matches})
    return results


def write_new_json(path: Path, payload: dict) -> None:
    """Write a new record without silently replacing prior evidence."""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("files", type=Path, nargs="*")
    args = parser.parse_args()
    if args.verify:
        payload = json.loads(args.verify.read_text(encoding="utf-8"))
        checks = verify(args.root, payload["artifacts"])
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        if not all(check["matches"] for check in checks):
            raise SystemExit(1)
    else:
        if not args.files or not args.output:
            parser.error("inventory requires --output and one or more files")
        payload = {"schema": "pulsemorph.evidence_inventory.v1",
                   "scope": "Actual byte inventory; no inference of physical or learning success",
                   "artifacts": [artifact(args.root, path, role="execution_evidence") for path in args.files]}
        write_new_json(args.output, payload)
        print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
