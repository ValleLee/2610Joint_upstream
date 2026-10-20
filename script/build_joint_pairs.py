"""Copy four-column ratio100 pairs and append the matching images_4 GT RGB path."""

import argparse
from pathlib import Path


DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data/pairs/k24_ratio100_val2_ignore"


def build(name: str, source: Path, dest: Path, gt_root: Path) -> None:
    output = []
    missing = []
    for number, line in enumerate((source / name).read_text().splitlines(), 1):
        cols = line.split()
        if len(cols) != 4:
            raise ValueError(f"{name}:{number}: expected 4 columns, got {len(cols)}")
        rendered_rgb = Path(cols[0])
        scene = rendered_rgb.parents[3].name
        frame = rendered_rgb.name.replace("_ref.png", ".png")
        gt_rgb = gt_root / scene / "gaussian_splat/images_4" / frame
        for path in (*cols, str(gt_rgb)):
            if not Path(path).is_file():
                missing.append((number, path))
        output.append(" ".join([*cols, str(gt_rgb)]))
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} missing files; first: {missing[:5]}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / name).write_text("\n".join(output) + "\n")
    print(f"{name}: wrote {len(output)} joint pairs to {dest / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory with four-column train_pairs.txt and val_pairs.txt")
    parser.add_argument("--gt-root", type=Path, required=True,
                        help="Directory containing <scene>/gaussian_splat/images_4")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    for filename in ("train_pairs.txt", "val_pairs.txt"):
        build(filename, args.source_dir, args.output_dir, args.gt_root)
