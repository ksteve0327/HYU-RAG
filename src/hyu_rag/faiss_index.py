"""Build the FAISS sidecar index in a fresh process."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_faiss_index(passages_path: str | Path, index_path: str | Path, *, hnsw_m: int = 128) -> None:
    import faiss
    from datasets import load_from_disk

    dataset = load_from_disk(str(passages_path))
    index = faiss.IndexHNSWFlat(768, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)
    dataset.get_index("embeddings").save(str(index_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS HNSW index for a saved RAG passages dataset.")
    parser.add_argument("--passages-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--hnsw-m", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_faiss_index(args.passages_path, args.index_path, hnsw_m=args.hnsw_m)


if __name__ == "__main__":
    main()
