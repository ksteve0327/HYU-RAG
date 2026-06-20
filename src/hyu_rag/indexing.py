"""Build a custom DPR/FAISS knowledge index for RAG."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .constants import DPR_CONTEXT_ENCODER


@dataclass(frozen=True)
class IndexPaths:
    passages_path: Path
    index_path: Path
    metadata_path: Path


def _require_rag_dependencies():
    try:
        import importlib.util

        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast  # noqa: F401

        if importlib.util.find_spec("faiss") is None:
            raise ImportError("No module named 'faiss'")
    except ImportError as exc:
        raise RuntimeError(
            "RAG indexing dependencies are missing. Install them with: "
            "uv pip install -r requirements-rag.txt"
        ) from exc


def split_words(text: str, chunk_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_words]).strip() for i in range(0, len(words), chunk_words) if words[i : i + chunk_words]]


def load_knowledge_tsv(path: str | Path, *, chunk_words: int) -> dict[str, list[str]]:
    titles: list[str] = []
    texts: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["title", "text"]:
            raise ValueError(f"Expected knowledge TSV columns ['title', 'text'], got {reader.fieldnames}")
        for row in reader:
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            if not title or not text:
                raise ValueError("knowledge.tsv contains an empty title or text field")
            chunks = split_words(text, chunk_words)
            for chunk_index, chunk in enumerate(chunks, start=1):
                titles.append(f"{title} | passage {chunk_index}")
                texts.append(chunk)
    return {"title": titles, "text": texts}


def build_custom_index(
    knowledge_tsv: str | Path,
    output_dir: str | Path,
    *,
    ctx_encoder_name: str = DPR_CONTEXT_ENCODER,
    batch_size: int = 8,
    chunk_words: int = 100,
    max_length: int = 256,
    hnsw_m: int = 128,
    device: str = "auto",
) -> IndexPaths:
    _require_rag_dependencies()
    import numpy as np
    import torch
    from datasets import Dataset, Features, Sequence, Value
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    passages_path = output_dir / "patent_knowledge_dataset"
    index_path = output_dir / "patent_knowledge_hnsw_index.faiss"
    metadata_path = output_dir / "index_metadata.json"
    if passages_path.exists():
        shutil.rmtree(passages_path)
    if index_path.exists():
        index_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    records = load_knowledge_tsv(knowledge_tsv, chunk_words=chunk_words)
    print(f"Loaded {len(records['title'])} passages from {knowledge_tsv}")
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(ctx_encoder_name)
    encoder = DPRContextEncoder.from_pretrained(ctx_encoder_name).to(device_name)
    encoder.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records["title"]), batch_size):
            end = start + batch_size
            if start == 0 or start % (batch_size * 25) == 0:
                print(f"Embedding passages {start + 1}-{min(end, len(records['title']))} / {len(records['title'])}")
            inputs = tokenizer(
                records["title"][start:end],
                records["text"][start:end],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device_name) for key, value in inputs.items()}
            batch_embeddings = encoder(**inputs, return_dict=True).pooler_output.detach().cpu().numpy()
            embeddings.extend(batch_embeddings.astype("float32"))

    print("Saving passages dataset")
    features = Features(
        {
            "title": Value("string"),
            "text": Value("string"),
            "embeddings": Sequence(Value("float32")),
        }
    )
    dataset = Dataset.from_dict(
        {
            "title": records["title"],
            "text": records["text"],
            "embeddings": [embedding.tolist() for embedding in embeddings],
        },
        features=features,
    )
    dataset.save_to_disk(passages_path)
    print("Building FAISS HNSW index in a fresh process")
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hyu_rag.faiss_index",
            "--passages-path",
            str(passages_path),
            "--index-path",
            str(index_path),
            "--hnsw-m",
            str(hnsw_m),
        ],
        check=True,
        env=env,
    )

    metadata = {
        "knowledge_tsv": str(Path(knowledge_tsv).resolve()),
        "passages_path": str(passages_path.resolve()),
        "index_path": str(index_path.resolve()),
        "ctx_encoder_name": ctx_encoder_name,
        "batch_size": batch_size,
        "chunk_words": chunk_words,
        "passage_count": len(records["title"]),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return IndexPaths(passages_path=passages_path, index_path=index_path, metadata_path=metadata_path)
