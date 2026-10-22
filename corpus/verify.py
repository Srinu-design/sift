"""Verify that every corpus file listed in a manifest exists and matches its SHA-256."""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify_manifest(manifest_path: Path) -> list[str]:
    problems: list[str] = []
    if not manifest_path.is_file():
        return [f"{manifest_path}: manifest file not found"]

    base_dir = manifest_path.parent
    for line_no, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        rel_path = str(row["path"])
        expected_sha = str(row["sha256"])
        target = base_dir / rel_path
        if not target.is_file():
            problems.append(f"{manifest_path}:{line_no}: missing file {rel_path}")
            continue
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            problems.append(
                f"{manifest_path}:{line_no}: sha256 mismatch for {rel_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
    return problems


def verify_manifests(paths: Sequence[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        problems.extend(verify_manifest(path))
    return problems


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python corpus/verify.py",
        description=__doc__,
    )
    p.add_argument("manifests", nargs="*", type=Path)
    args = p.parse_args()

    manifests: list[Path] = (
        list(args.manifests) if args.manifests else sorted(HERE.glob("*manifest.jsonl"))
    )
    if not manifests:
        print("no manifest files found", file=sys.stderr)
        return 1

    problems = verify_manifests(manifests)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
