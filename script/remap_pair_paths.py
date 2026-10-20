"""Rewrite historical absolute path prefixes into machine-local pair lists.

Example:
  python script/remap_pair_paths.py --replace /old/data=/new/data \
      --replace /old/da=/new/da --output-dir data/pairs/local --check-files
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "data/pairs/k24_ratio100_val2_ignore"
DEFAULT_OUTPUT = PROJECT / "data/pairs/local"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="append", default=[], metavar="OLD=NEW",
                        help="Replace an absolute path prefix; can be supplied multiple times")
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    replacements = []
    for spec in args.replace:
        old, sep, new = spec.partition("=")
        if not sep or not old.startswith("/") or not new.startswith("/"):
            parser.error(f"Expected absolute OLD=NEW prefix: {spec}")
        replacements.append((old.rstrip("/"), new.rstrip("/")))
    if not replacements:
        parser.error("Supply at least one --replace OLD=NEW")
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("train_pairs.txt", "val_pairs.txt"):
        src = args.input_dir / filename
        output = []
        missing = []
        for line_no, line in enumerate(src.read_text().splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{src}:{line_no}: expected five columns")
            mapped = []
            for field in fields:
                for old, new in replacements:
                    if field == old or field.startswith(old + "/"):
                        field = new + field[len(old):]
                        break
                if args.check_files and not Path(field).is_file():
                    missing.append((line_no, field))
                mapped.append(field)
            output.append(" ".join(mapped))
        if missing:
            raise FileNotFoundError(f"{filename}: {len(missing)} missing files; first: {missing[:3]}")
        dst = args.output_dir / filename
        dst.write_text("\n".join(output) + "\n")
        print(f"{dst}: {len(output)} pairs")


if __name__ == "__main__":
    main()
