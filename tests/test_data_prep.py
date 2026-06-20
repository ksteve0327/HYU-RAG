from __future__ import annotations

import json
from pathlib import Path

from hyu_rag.data import (
    build_knowledge_text,
    clean_claim,
    sample_us_patents,
    validate_prepared_sample,
    write_prepared_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "tests" / "fixtures" / "patents_minimal.csv"


def test_sample_us_patents_balanced_counts():
    sample_df = sample_us_patents(RAW_CSV, sample_per_class=1, seed=42)
    summary = validate_prepared_sample(sample_df, sample_per_class=1)
    assert summary["rows"] == 4
    assert summary["category_counts"] == {"AA": 1, "AB": 1, "AC": 1, "AD": 1}
    assert set(sample_df["country_code"]) == {"US"}
    assert not sample_df["application_id"].duplicated().any()


def test_knowledge_text_uses_english_source_fields():
    sample_df = sample_us_patents(RAW_CSV, sample_per_class=1, seed=42)
    text = build_knowledge_text(sample_df.iloc[0])
    assert "Title:" in text
    assert "Abstract:" in text
    assert "Main Claim:" in text
    assert "AI요약" not in text
    assert "요약(원문)" not in text


def test_clean_claim_removes_korean_claim_prefix():
    assert clean_claim("[청구항1]\n 1. A method comprising: receiving data.") == "1. A method comprising: receiving data."


def test_write_prepared_outputs(tmp_path):
    sample_df = sample_us_patents(RAW_CSV, sample_per_class=1, seed=42)
    paths = write_prepared_outputs(sample_df, tmp_path, seed=42)
    assert paths.sample_csv.exists()
    assert paths.knowledge_tsv.exists()
    assert paths.tasks_jsonl.exists()
    tasks = [json.loads(line) for line in paths.tasks_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 12
    assert {task["task_type"] for task in tasks} == {"classification", "qa", "summary"}
