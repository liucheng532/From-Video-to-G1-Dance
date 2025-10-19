import argparse
import os
import pathlib
import subprocess
import sys
from natsort import natsorted


def derive_output_name(csv_path: pathlib.Path, root_dir: pathlib.Path) -> str:
    """Derive a stable artifact name from a CSV path relative to root.

    Example: root_dir/foo/bar/walk_01.csv -> foo_bar_walk_01
    """
    rel = csv_path.relative_to(root_dir).with_suffix("")
    parts = list(rel.parts)
    # sanitize parts: replace spaces
    parts = [p.replace(" ", "_") for p in parts]
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Batch-convert CSV motions to npz via csv_to_npz.py and upload to WandB registry.")
    parser.add_argument("--input_dir", required=True, help="Directory containing CSV files (recursively).")
    parser.add_argument("--input_fps", type=int, default=30, help="FPS of CSV motions (default: 30).")
    parser.add_argument("--output_fps", type=int, default=50, help="Output FPS for interpolation (default: 50).")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")

    args, unknown = parser.parse_known_args()

    input_dir = pathlib.Path(args.input_dir)
    assert input_dir.is_dir(), f"Input directory not found: {input_dir}"

    csv_files = []
    for root, _, files in os.walk(input_dir):
        for fn in natsorted(files):
            if fn.endswith(".csv"):
                csv_files.append(pathlib.Path(root) / fn)

    if not csv_files:
        print(f"[WARN] No CSV files under: {input_dir}")
        return

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tool = repo_root / "scripts" / "csv_to_npz.py"
    assert tool.is_file(), f"csv_to_npz.py not found at: {tool}"

    for csv_path in csv_files:
        out_name = derive_output_name(csv_path, input_dir)
        cmd = [
            sys.executable,
            str(tool),
            "--input_file", str(csv_path),
            "--input_fps", str(args.input_fps),
            "--output_name", out_name,
            "--output_fps", str(args.output_fps),
        ]
        if args.headless:
            cmd.append("--headless")
        # pass through extra args to AppLauncher if provided
        cmd.extend(unknown)

        print("[RUN]", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

