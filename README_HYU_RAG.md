# HYU US Patent RAG Demo

This repo keeps the Hugging Face `transformers` clones untouched and adds local demo code under `src/hyu_rag/` and `scripts/`.

Raw patent CSVs, generated processed data, FAISS indexes, run artifacts, vendor clones, and source PDFs are intentionally not committed. Place `test_data.csv` in the repo root before running the full demo. The tests use a small sanitized fixture under `tests/fixtures/`.

## 1. Prepare data

```bash
python scripts/prepare_data.py \
  --input-csv test_data.csv \
  --output-dir data/processed \
  --sample-per-class 50 \
  --seed 42
```

Outputs:

- `data/processed/test_us_200_seed42.csv`
- `data/processed/knowledge.tsv`
- `data/processed/demo_tasks.jsonl`

The preparation step filters `국가코드 == "US"`, samples `AA/AB/AC/AD` 50 rows each, and builds English-centered knowledge text from title, abstract, claims, IPC/CPC, applicant, assignee, legal status, and category code.

## 2. Install RAG dependencies

Use Python 3.10 or 3.11 when possible:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-rag.txt
```

If Python 3.11 is not installed, let `uv` download it or use a compatible Python 3.10-3.12 environment.

## 3. Build custom DPR/FAISS index

```bash
python scripts/build_index.py \
  --knowledge-tsv data/processed/knowledge.tsv \
  --output-dir artifacts/index \
  --batch-size 8 \
  --max-length 256
```

Outputs:

- `artifacts/index/patent_knowledge_dataset`
- `artifacts/index/patent_knowledge_hnsw_index.faiss`
- `artifacts/index/index_metadata.json`

## 4. Run RAG-Sequence / RAG-Token trace demo

```bash
python scripts/run_demo.py \
  --passages-path artifacts/index/patent_knowledge_dataset \
  --index-path artifacts/index/patent_knowledge_hnsw_index.faiss \
  --tasks-path data/processed/demo_tasks.jsonl \
  --output-dir artifacts/runs \
  --n-docs 5 \
  --num-beams 2 \
  --max-length 64 \
  --examples-per-task 4
```

The default `--retrieval-backend brute_force` uses the saved DPR embeddings with exact dot-product retrieval. This avoids a macOS OpenMP conflict when FAISS and torch are loaded in the same process. The FAISS index is still built and smoke-tested; pass `--retrieval-backend hf_faiss` if your environment can load Hugging Face `RagRetriever` with FAISS safely.

Each run writes:

- `rag_trace.jsonl`
- `report.md`

The report compares retrieved documents, document scores, RAG-Sequence candidate scoring, and RAG-Token token-step document contributions.
