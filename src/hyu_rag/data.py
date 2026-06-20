"""Data preparation for the HYU US patent RAG demo."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import DEFAULT_CLASS_CODES, DEFAULT_SAMPLE_PER_CLASS, DEFAULT_SEED


REQUIRED_COLUMNS = {
    "출원번호",
    "발명의 명칭",
    "요약",
    "중분류",
    "소분류",
    "국가코드",
    "대표청구항",
    "독립항",
    "출원인정리",
    "현재권리자정리",
    "법적상태",
    "메인 IPC",
    "전체 IPC",
}

MISSING_VALUES = {"", "-", "nan", "none", "null", "NaN", "None", "NULL"}
CLAIM_PREFIX_RE = re.compile(r"\[\s*청구항\s*\d+\s*\]\s*", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PreparedPaths:
    sample_csv: Path
    knowledge_tsv: Path
    tasks_jsonl: Path


def clean_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in MISSING_VALUES:
        return ""
    return WHITESPACE_RE.sub(" ", text)


def clean_multiline(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if text in MISSING_VALUES:
        return ""
    return re.sub(r"[ \t]+", " ", text)


def clean_claim(value: object) -> str:
    text = clean_multiline(value)
    text = CLAIM_PREFIX_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _main_cpc_columns(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col == "메인 CPC" or re.fullmatch(r"메인 CPC\.\d+", col)]


def _first_non_empty(values: Iterable[object]) -> str:
    for value in values:
        cleaned = clean_value(value)
        if cleaned:
            return cleaned
    return ""


def read_patent_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    cpc_columns = _main_cpc_columns(df.columns)
    if not cpc_columns:
        raise ValueError("Missing required duplicate-aware column: 메인 CPC")

    normalized = df.copy()
    normalized["application_id"] = normalized["출원번호"].map(clean_value)
    normalized["country_code"] = normalized["국가코드"].map(lambda value: clean_value(value).upper())
    normalized["category_code"] = normalized["중분류"].map(clean_value)
    normalized["subcategory_code"] = normalized["소분류"].map(clean_value)
    normalized["main_cpc"] = normalized[cpc_columns].apply(_first_non_empty, axis=1)
    return normalized


def build_knowledge_title(row: pd.Series) -> str:
    title = clean_value(row.get("발명의 명칭"))
    return f"{row['application_id']} | {title}".strip()


def build_knowledge_text(row: pd.Series) -> str:
    fields = [
        ("Application ID", row.get("application_id")),
        ("Title", row.get("발명의 명칭")),
        ("Abstract", row.get("요약")),
        ("Main Claim", clean_claim(row.get("대표청구항"))),
        ("Independent Claim", clean_claim(row.get("독립항"))),
        ("Main IPC", row.get("메인 IPC")),
        ("All IPC", row.get("전체 IPC")),
        ("Main CPC", row.get("main_cpc")),
        ("Applicant", row.get("출원인정리")),
        ("Assignee", row.get("현재권리자정리")),
        ("Legal Status", row.get("법적상태")),
        ("Category Code", row.get("category_code")),
        ("Subcategory Code", row.get("subcategory_code")),
        ("Country Code", row.get("country_code")),
    ]
    lines = []
    for label, value in fields:
        cleaned = clean_claim(value) if "Claim" in label else clean_value(value)
        if cleaned:
            lines.append(f"{label}: {cleaned}")
    return "\n".join(lines)


def build_classification_query(row: pd.Series) -> str:
    claim = clean_claim(row.get("대표청구항"))
    return "\n".join(
        [
            "Classify this US patent into one of these category codes: AA, AB, AC, AD.",
            "Return only the category code.",
            f"Title: {clean_value(row.get('발명의 명칭'))}",
            f"Abstract: {clean_value(row.get('요약'))}",
            f"Main Claim: {claim}",
        ]
    )


QA_FIELDS = (
    ("main_ipc", "메인 IPC", "What is the main IPC code for this patent?"),
    ("main_cpc", "main_cpc", "What is the main CPC code for this patent?"),
    ("applicant", "출원인정리", "Who is the applicant for this patent?"),
    ("assignee", "현재권리자정리", "Who is the current assignee for this patent?"),
    ("legal_status", "법적상태", "What is the legal status of this patent?"),
    ("category_code", "category_code", "What is the category code of this patent?"),
)


def _qa_field_for_row(row: pd.Series, row_number: int) -> tuple[str, str, str]:
    start = row_number % len(QA_FIELDS)
    for offset in range(len(QA_FIELDS)):
        key, column, question = QA_FIELDS[(start + offset) % len(QA_FIELDS)]
        answer = clean_value(row.get(column))
        if answer:
            return key, question, answer
    return "category_code", "What is the category code of this patent?", clean_value(row.get("category_code"))


def build_tasks(sample_df: pd.DataFrame) -> list[dict]:
    tasks: list[dict] = []
    for row_number, (_, row) in enumerate(sample_df.iterrows()):
        app_id = row["application_id"]
        common = {
            "application_id": app_id,
            "title": clean_value(row.get("발명의 명칭")),
            "category_code": clean_value(row.get("category_code")),
            "country_code": clean_value(row.get("country_code")),
        }
        tasks.append(
            {
                **common,
                "task_id": f"{app_id}:classification",
                "task_type": "classification",
                "query": build_classification_query(row),
                "target": clean_value(row.get("category_code")),
            }
        )

        qa_key, qa_question, qa_answer = _qa_field_for_row(row, row_number)
        tasks.append(
            {
                **common,
                "task_id": f"{app_id}:qa:{qa_key}",
                "task_type": "qa",
                "query": "\n".join(
                    [
                        qa_question,
                        f"Application ID: {app_id}",
                        f"Title: {clean_value(row.get('발명의 명칭'))}",
                    ]
                ),
                "target": qa_answer,
            }
        )

        tasks.append(
            {
                **common,
                "task_id": f"{app_id}:summary",
                "task_type": "summary",
                "query": "\n".join(
                    [
                        "Write a short technical summary of this patent.",
                        f"Title: {clean_value(row.get('발명의 명칭'))}",
                        f"Abstract: {clean_value(row.get('요약'))}",
                        f"Main Claim: {clean_claim(row.get('대표청구항'))}",
                    ]
                ),
                "target": clean_value(row.get("요약")),
            }
        )
    return tasks


def sample_us_patents(
    csv_path: str | Path,
    *,
    sample_per_class: int = DEFAULT_SAMPLE_PER_CLASS,
    seed: int = DEFAULT_SEED,
    class_codes: Iterable[str] = DEFAULT_CLASS_CODES,
) -> pd.DataFrame:
    class_codes = tuple(class_codes)
    df = read_patent_csv(csv_path)
    us_df = df[df["country_code"] == "US"].copy()
    us_df = us_df[us_df["application_id"] != ""].drop_duplicates(subset=["application_id"], keep="first")

    counts = us_df["category_code"].value_counts().to_dict()
    too_small = {code: counts.get(code, 0) for code in class_codes if counts.get(code, 0) < sample_per_class}
    if too_small:
        raise ValueError(f"Not enough US rows per category for sampling: {too_small}")

    sampled_frames = []
    for code in class_codes:
        group = us_df[us_df["category_code"] == code].sort_values("application_id")
        sampled = group.sample(n=sample_per_class, random_state=seed).sort_values("application_id")
        sampled_frames.append(sampled)

    sample_df = pd.concat(sampled_frames, ignore_index=True)
    sample_df.insert(0, "sample_index", range(1, len(sample_df) + 1))
    sample_df["knowledge_title"] = sample_df.apply(build_knowledge_title, axis=1)
    sample_df["knowledge_text"] = sample_df.apply(build_knowledge_text, axis=1)
    return sample_df


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_prepared_outputs(sample_df: pd.DataFrame, output_dir: str | Path, *, seed: int) -> PreparedPaths:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = output_dir / f"test_us_200_seed{seed}.csv"
    knowledge_tsv = output_dir / "knowledge.tsv"
    tasks_jsonl = output_dir / "demo_tasks.jsonl"

    sample_df.to_csv(sample_csv, index=False, encoding="utf-8")
    sample_df[["knowledge_title", "knowledge_text"]].rename(
        columns={"knowledge_title": "title", "knowledge_text": "text"}
    ).to_csv(knowledge_tsv, sep="\t", index=False, encoding="utf-8")
    write_jsonl(tasks_jsonl, build_tasks(sample_df))
    return PreparedPaths(sample_csv=sample_csv, knowledge_tsv=knowledge_tsv, tasks_jsonl=tasks_jsonl)


def validate_prepared_sample(sample_df: pd.DataFrame, *, sample_per_class: int) -> dict:
    if len(sample_df) != sample_per_class * len(DEFAULT_CLASS_CODES):
        raise AssertionError(f"Unexpected sample row count: {len(sample_df)}")
    if not (sample_df["country_code"] == "US").all():
        raise AssertionError("Prepared sample contains non-US rows")
    if sample_df["application_id"].duplicated().any():
        raise AssertionError("Prepared sample contains duplicate application_id values")

    counts = sample_df["category_code"].value_counts().to_dict()
    expected = {code: sample_per_class for code in DEFAULT_CLASS_CODES}
    if counts != expected:
        raise AssertionError(f"Unexpected category counts: {counts}; expected {expected}")
    if sample_df["knowledge_title"].map(clean_value).eq("").any():
        raise AssertionError("Prepared sample contains empty knowledge titles")
    if sample_df["knowledge_text"].map(clean_value).eq("").any():
        raise AssertionError("Prepared sample contains empty knowledge text")
    return {"rows": len(sample_df), "category_counts": counts}
