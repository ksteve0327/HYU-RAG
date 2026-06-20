"""Shared constants for the HYU patent RAG demo."""

from __future__ import annotations

DEFAULT_CLASS_CODES = ("AA", "AB", "AC", "AD")
DEFAULT_INPUT_CSV = "test_data.csv"
DEFAULT_OUTPUT_DIR = "data/processed"
DEFAULT_SAMPLE_PER_CLASS = 50
DEFAULT_SEED = 42

RAG_SEQUENCE_MODEL = "facebook/rag-sequence-nq"
RAG_TOKEN_MODEL = "facebook/rag-token-nq"
DPR_CONTEXT_ENCODER = "facebook/dpr-ctx_encoder-multiset-base"
