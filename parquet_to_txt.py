#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError as exc:
    print("Missing dependency: pandas is required. Install with 'pip install pandas pyarrow'", file=sys.stderr)
    raise SystemExit(1) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all Parquet files in a folder to a single text file using the 'text' column."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing Parquet files (default: current directory).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.txt",
        help="Output text file path (default: output.txt).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for .parquet files in subdirectories.",
    )
    return parser.parse_args()


def find_parquet_files(directory: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(directory.rglob("*.parquet"))
    return sorted(directory.glob("*.parquet"))


def read_texts_from_parquet(path: Path) -> list[str]:
    df = pd.read_parquet(path, columns=["text"])
    if "text" not in df.columns:
        raise ValueError(f"Parquet file {path} does not contain a 'text' column.")
    return df["text"].astype(str).tolist()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    parquet_files = find_parquet_files(input_dir, args.recursive)
    if not parquet_files:
        raise SystemExit(f"No .parquet files found in {input_dir} {'recursively' if args.recursive else ''}.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for parquet_file in parquet_files:
            try:
                texts = read_texts_from_parquet(parquet_file)
            except Exception as exc:
                raise SystemExit(f"Failed to read {parquet_file}: {exc}") from exc
            for text in texts:
                output_file.write(f"{text}\n")
            total_rows += len(texts)

    print(f"Converted {len(parquet_files)} parquet file(s) to '{output_path}' with {total_rows} text row(s).")


if __name__ == "__main__":
    main()
