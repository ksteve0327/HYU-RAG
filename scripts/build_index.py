#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyu_rag.constants import DPR_CONTEXT_ENCODER
from hyu_rag.indexing import build_custom_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DPR/FAISS custom RAG index from knowledge.tsv.")
    parser.add_argument("--knowledge-tsv", default="data/processed/knowledge.tsv")
    parser.add_argument("--output-dir", default="artifacts/index")
    parser.add_argument("--ctx-encoder-name", default=DPR_CONTEXT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-words", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hnsw-m", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_custom_index(
        args.knowledge_tsv,
        args.output_dir,
        ctx_encoder_name=args.ctx_encoder_name,
        batch_size=args.batch_size,
        chunk_words=args.chunk_words,
        max_length=args.max_length,
        hnsw_m=args.hnsw_m,
        device=args.device,
    )
    print(f"Passages dataset: {paths.passages_path}")
    print(f"FAISS index: {paths.index_path}")
    print(f"Metadata: {paths.metadata_path}")


if __name__ == "__main__":
    main()
