#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyu_rag.constants import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_DIR, DEFAULT_SAMPLE_PER_CLASS, DEFAULT_SEED
from hyu_rag.data import sample_us_patents, validate_prepared_sample, write_prepared_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the 200-row HYU US patent RAG demo dataset.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Raw CSV path. Defaults to test_data.csv.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for processed outputs.")
    parser.add_argument("--sample-per-class", type=int, default=DEFAULT_SAMPLE_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_df = sample_us_patents(
        args.input_csv,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    summary = validate_prepared_sample(sample_df, sample_per_class=args.sample_per_class)
    paths = write_prepared_outputs(sample_df, args.output_dir, seed=args.seed)
    print(f"Prepared {summary['rows']} rows")
    print(f"Category counts: {summary['category_counts']}")
    print(f"Sample CSV: {paths.sample_csv}")
    print(f"Knowledge TSV: {paths.knowledge_tsv}")
    print(f"Demo tasks: {paths.tasks_jsonl}")


if __name__ == "__main__":
    main()
