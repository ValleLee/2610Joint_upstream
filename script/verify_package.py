"""Audit this local release package before a future GitHub upload."""

from __future__ import annotations

import hashlib
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_FILE = 100 * 1024 * 1024
WEIGHT_SUFFIXES = {".ckpt", ".pth", ".pt", ".safetensors"}
REQUIRED = (
    "config/train_rgbdepth_teacher_perimg.yaml",
    "docs/evidence/run_config.yaml",
    "docs/evidence/run_logging.log",
    "docs/evidence/da_comparison_100.csv",
    "docs/figures/garden_depth_vs_da.png",
    "docs/figures/sculpture_depth_vs_da.png",
    "data/pairs/k24_ratio100_val2_ignore/train_pairs.txt",
    "data/pairs/k24_ratio100_val2_ignore/val_pairs.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"Missing: {name}")
    for name in ("output", "logs", "checkpoints"):
        if (ROOT / name).exists():
            errors.append(f"Unexpected local directory: {name}")
    files = [path for path in ROOT.rglob("*") if path.is_file()
             and ".git" not in path.parts
             and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts
             and path.suffix != ".pyc"]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix in WEIGHT_SUFFIXES:
            errors.append(f"Weight file included: {relative}")
        if path.stat().st_size >= MAX_GITHUB_FILE:
            errors.append(f"File >= 100 MiB: {relative}")
        if path.suffix != ".md":
            continue
        for target in re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", path.read_text()):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http:", "https:", "/", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"Broken Markdown link: {relative}: {target}")
    evidence = ROOT / "docs/evidence/provenance.json"
    if evidence.is_file():
        provenance = json.loads(evidence.read_text())
        for tag, sample in provenance["samples"].items():
            for name, metadata in sample["originals"].items():
                copied = ROOT / "docs/figures/source" / tag / name
                if not copied.is_file() or sha256(copied) != metadata["sha256"]:
                    errors.append(f"Original sample/hash mismatch: {tag}/{name}")
    for filename, expected in (("train_pairs.txt", 36430), ("val_pairs.txt", 715)):
        pair_file = ROOT / "data/pairs/k24_ratio100_val2_ignore" / filename
        if pair_file.is_file():
            count = 0
            for line in pair_file.open():
                if not line.strip():
                    continue
                count += 1
                if len(line.split()) != 5:
                    errors.append(f"Invalid five-column list: {filename}:{count}")
                    break
            if count != expected:
                errors.append(f"Unexpected pair count: {filename}: {count} != {expected}")
    comparison = ROOT / "docs/evidence/da_comparison_100.csv"
    summary_file = ROOT / "docs/evidence/da_comparison_100_summary.json"
    if comparison.is_file() and summary_file.is_file():
        rows = list(csv.DictReader(comparison.open()))
        summary = json.loads(summary_file.read_text())
        wins = sum(int(row["joint_teacher_wins"]) for row in rows)
        if len(rows) != summary["sample_count"] or wins != summary["joint_teacher_wins"]:
            errors.append("DA comparison CSV and summary disagree")
    manifest_file = ROOT / "docs/evidence/source_copy_manifest.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text())
        listed = {entry["path"]: entry["package_sha256"] for entry in manifest["files"]}
        current = {str(path.relative_to(ROOT)) for path in files if path != manifest_file}
        if set(listed) != current:
            errors.append("Source copy manifest file list differs from the package")
        for relative, expected_hash in listed.items():
            path = ROOT / relative
            if path.is_file() and sha256(path) != expected_hash:
                errors.append(f"Source copy manifest hash mismatch: {relative}")
    else:
        errors.append("Missing source copy manifest")
    if errors:
        raise SystemExit("Package verification failed:\n" + "\n".join(errors))
    print(f"OK: {len(files)} files; no weights or oversized files; links, pair lists, and sample hashes verified")


if __name__ == "__main__":
    main()
