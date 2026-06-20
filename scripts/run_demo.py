#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyu_rag.constants import RAG_SEQUENCE_MODEL, RAG_TOKEN_MODEL
from hyu_rag.tracing import DemoConfig, run_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG-Sequence/RAG-Token tracing over the HYU patent index.")
    parser.add_argument("--passages-path", default="artifacts/index/patent_knowledge_dataset")
    parser.add_argument("--index-path", default="artifacts/index/patent_knowledge_hnsw_index.faiss")
    parser.add_argument("--tasks-path", default="data/processed/demo_tasks.jsonl")
    parser.add_argument("--output-dir", default="artifacts/runs")
    parser.add_argument("--n-docs", type=int, default=5)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--trace-max-steps", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--retrieval-backend",
        choices=["brute_force", "hf_faiss"],
        default="brute_force",
        help="brute_force avoids FAISS/torch OpenMP conflicts on macOS; hf_faiss uses Hugging Face RagRetriever.",
    )
    parser.add_argument("--rag-sequence-model", default=RAG_SEQUENCE_MODEL)
    parser.add_argument("--rag-token-model", default=RAG_TOKEN_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_demo(
        DemoConfig(
            passages_path=Path(args.passages_path),
            index_path=Path(args.index_path),
            tasks_path=Path(args.tasks_path),
            output_dir=Path(args.output_dir),
            n_docs=args.n_docs,
            num_beams=args.num_beams,
            max_length=args.max_length,
            trace_max_steps=args.trace_max_steps,
            examples_per_task=args.examples_per_task,
            device=args.device,
            retrieval_backend=args.retrieval_backend,
            rag_sequence_model=args.rag_sequence_model,
            rag_token_model=args.rag_token_model,
        )
    )
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
