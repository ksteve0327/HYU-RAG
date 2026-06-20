#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from collections import Counter
from pathlib import Path


MODEL_LABELS = {
    "rag_sequence": "RAG-Sequence",
    "rag_token": "RAG-Token",
}

TASK_LABELS = {
    "classification": "중분류 예측",
    "qa": "질의응답",
    "summary": "요약/분석",
}

CATEGORY_LABELS = {
    "AA": "AI 코어 및 가속기",
    "AB": "제조 및 패키징",
    "AC": "시스템 통합 및 데이터 관리",
    "AD": "시스템 연동 및 플랫폼 통합",
}

QA_QUESTIONS_KO = {
    "main_ipc": "이 특허의 메인 IPC 코드는 무엇인가요?",
    "main_cpc": "이 특허의 메인 CPC 코드는 무엇인가요?",
    "applicant": "이 특허의 출원인은 누구인가요?",
    "assignee": "이 특허의 현재권리자는 누구인가요?",
    "legal_status": "이 특허의 법적상태는 무엇인가요?",
    "category_code": "이 특허의 중분류 코드는 무엇인가요?",
}

CLAIM_PREFIX_RE = re.compile(r"\[\s*청구항\s*\d+\s*\]\s*", re.IGNORECASE)


def clean_value(value: object) -> str:
    text = str(value or "").strip()
    if text in {"", "-", "nan", "None", "NULL", "null"}:
        return ""
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_optional_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_first_tsv(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            return row
    return {}


def read_dataset_preview(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        from datasets import load_from_disk
    except ImportError:
        return {"load_error": "datasets 패키지가 없어 저장 dataset을 직접 열지 못했습니다."}

    dataset = load_from_disk(str(path))
    if len(dataset) == 0:
        return {"num_rows": 0}
    row = dataset[0]
    embedding = row.get("embeddings") or []
    embedding_preview = [f"{float(value):.4f}" for value in embedding[:8]]
    embedding_norm = math.sqrt(sum(float(value) * float(value) for value in embedding)) if embedding else None
    return {
        "num_rows": len(dataset),
        "features": list(dataset.features.keys()),
        "title": row.get("title"),
        "text": row.get("text"),
        "embedding_dim": len(embedding),
        "embedding_preview": embedding_preview,
        "embedding_norm": embedding_norm,
    }


def read_faiss_preview(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    preview = {"file_size_bytes": path.stat().st_size}
    try:
        import faiss
    except ImportError:
        preview["load_error"] = "faiss 패키지가 없어 index 내부 메타데이터를 직접 열지 못했습니다."
        return preview

    index = faiss.read_index(str(path))
    metric_names = {
        getattr(faiss, "METRIC_INNER_PRODUCT", 0): "inner product",
        getattr(faiss, "METRIC_L2", 1): "L2",
    }
    preview.update(
        {
            "index_type": type(index).__name__,
            "ntotal": getattr(index, "ntotal", ""),
            "dimension": getattr(index, "d", ""),
            "metric": metric_names.get(getattr(index, "metric_type", None), getattr(index, "metric_type", "")),
        }
    )
    return preview


def load_task_examples(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    examples: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            task_type = clean_value(task.get("task_type"))
            if task_type and task_type not in examples:
                examples[task_type] = task
            if {"classification", "qa", "summary"} <= set(examples):
                break
    return examples


def load_sample_rows(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            app_id = clean_value(row.get("application_id") or row.get("출원번호"))
            if app_id:
                rows[app_id] = row
    return rows


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def compact(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def compact_multiline(value: object, limit: int = 900) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def fmt_float(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def fmt_bytes(value: object) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return ""
    units = ("B", "KB", "MB", "GB")
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"


def category_ko(code: object) -> str:
    code = clean_value(code)
    label = CATEGORY_LABELS.get(code)
    return f"{code} - {label}" if label else code


def clean_claim_for_report(value: object) -> str:
    text = compact_multiline(value, limit=1200)
    text = CLAIM_PREFIX_RE.sub("", text)
    return " ".join(text.split())


def first_non_empty(row: dict | None, *columns: str) -> str:
    row = row or {}
    for column in columns:
        value = clean_value(row.get(column))
        if value:
            return value
    return ""


def qa_key(task: dict) -> str:
    parts = str(task.get("task_id", "")).split(":qa:")
    return parts[1] if len(parts) == 2 else ""


def korean_query(task: dict, source_row: dict | None) -> str:
    title = first_non_empty(source_row, "발명의 명칭") or clean_value(task.get("title"))
    korean_summary = first_non_empty(source_row, "AI요약(목적+솔루션)", "AI요약(목적)", "요약")
    if task["task_type"] == "classification":
        parts = [
            "이 미국 특허를 AA, AB, AC, AD 중 하나의 중분류 코드로 분류하세요.",
            "중분류 코드만 반환하세요.",
            f"발명의 명칭: {title}",
        ]
        if korean_summary:
            parts.append(f"한글 요약: {korean_summary}")
        return "\n".join(parts)
    if task["task_type"] == "qa":
        question = QA_QUESTIONS_KO.get(qa_key(task), "이 특허에 대한 질문에 답하세요.")
        return "\n".join([question, f"출원번호: {task.get('application_id')}", f"발명의 명칭: {title}"])
    if task["task_type"] == "summary":
        parts = ["이 특허의 기술 내용을 짧게 요약하세요.", f"발명의 명칭: {title}"]
        if korean_summary:
            parts.append(f"한글 요약: {korean_summary}")
        return "\n".join(parts)
    return clean_value(task.get("query"))


def korean_target(task: dict, source_row: dict | None) -> str:
    if task["task_type"] == "classification":
        return category_ko(task.get("target"))
    if task["task_type"] == "qa":
        key = qa_key(task)
        if key == "main_ipc":
            return first_non_empty(source_row, "메인 IPC") or clean_value(task.get("target"))
        if key == "main_cpc":
            return first_non_empty(source_row, "main_cpc", "메인 CPC", "메인 CPC.1") or clean_value(task.get("target"))
        if key == "applicant":
            return first_non_empty(source_row, "출원인", "출원인정리") or clean_value(task.get("target"))
        if key == "assignee":
            return first_non_empty(source_row, "현재권리자", "현재권리자정리") or clean_value(task.get("target"))
        if key == "legal_status":
            return first_non_empty(source_row, "법적상태") or clean_value(task.get("target"))
        if key == "category_code":
            return category_ko(task.get("target"))
        return clean_value(task.get("target"))
    if task["task_type"] == "summary":
        return first_non_empty(source_row, "AI요약(목적+솔루션)", "AI요약(목적)", "AI요약(솔루션)") or clean_value(
            task.get("target")
        )
    return clean_value(task.get("target"))


def bilingual_block(label: str, english_text: object, korean_text: object) -> str:
    return (
        f'<div class="bilingual"><h4>{esc(label)}</h4>'
        f'<div class="lang-block"><span>영문</span><p>{esc(english_text)}</p></div>'
        f'<div class="lang-block"><span>한글</span><p>{esc(korean_text)}</p></div>'
        "</div>"
    )


def extract_function(path: Path, function_name: str, *, max_lines: int | None = None) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith(f"def {function_name}(")), None)
    if start is None:
        return ""
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")) and (line.startswith("def ") or line.startswith("class ")):
            end = index
            break
    snippet_lines = lines[start:end]
    if max_lines is not None and len(snippet_lines) > max_lines:
        snippet_lines = snippet_lines[:max_lines] + ["    ..."]
    return "\n".join(snippet_lines)


def extract_window(
    path: Path,
    start_pattern: str,
    *,
    title: str,
    source_label: str,
    before: int = 0,
    after: int = 40,
) -> tuple[str, str, str]:
    if not path.exists():
        return title, source_label, ""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((index for index, line in enumerate(lines) if start_pattern in line), None)
    if start is None:
        return title, source_label, ""
    window_start = max(0, start - before)
    window_end = min(len(lines), start + after)
    return title, source_label, "\n".join(lines[window_start:window_end])


def load_code_examples(source_root: Path) -> list[tuple[str, str, str]]:
    data_path = source_root / "src" / "hyu_rag" / "data.py"
    indexing_path = source_root / "src" / "hyu_rag" / "indexing.py"
    tracing_path = source_root / "src" / "hyu_rag" / "tracing.py"
    examples = [
        (
            "US 필터링과 중분류별 샘플링",
            "src/hyu_rag/data.py::sample_us_patents",
            extract_function(data_path, "sample_us_patents"),
        ),
        (
            "Knowledge text 생성",
            "src/hyu_rag/data.py::build_knowledge_text",
            extract_function(data_path, "build_knowledge_text"),
        ),
        (
            "Knowledge TSV를 passage로 분할",
            "src/hyu_rag/indexing.py::load_knowledge_tsv",
            extract_function(indexing_path, "load_knowledge_tsv"),
        ),
        extract_window(
            tracing_path,
            "question_hidden_states = model.question_encoder",
            title="Question encoder, retrieval, doc_scores",
            source_label="src/hyu_rag/tracing.py::retrieve_context",
            before=8,
            after=44,
        ),
        extract_window(
            tracing_path,
            "outputs = model(",
            title="RAG-Sequence 후보 문장 점수화",
            source_label="src/hyu_rag/tracing.py::trace_rag_sequence",
            before=16,
            after=34,
        ),
        extract_window(
            tracing_path,
            "seq_logprobs = torch.nn.functional.log_softmax",
            title="RAG-Token 토큰 단위 주변화",
            source_label="src/hyu_rag/tracing.py::trace_rag_token",
            before=12,
            after=46,
        ),
    ]
    return [(title, source, code) for title, source, code in examples if code]


def category_counts(sample_rows: dict[str, dict], records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in sample_rows.values():
        code = clean_value(row.get("category_code") or row.get("중분류"))
        if code:
            counts[code] += 1
    if counts:
        return counts
    for record in records:
        task = record.get("task", {})
        code = clean_value(task.get("category_code"))
        if code and task.get("task_type") == "classification":
            counts[code] += 1
    return counts


def render_summary(records: list[dict]) -> str:
    model_counts = Counter(MODEL_LABELS.get(record["model_type"], record["model_type"]) for record in records)
    task_counts = Counter(TASK_LABELS.get(record["task"]["task_type"], record["task"]["task_type"]) for record in records)
    doc_counts = sorted(set(len(record.get("retrieved_docs", [])) for record in records))
    cards = [
        ("추적 레코드", len(records)),
        ("모델", ", ".join(f"{key}: {value}" for key, value in sorted(model_counts.items()))),
        ("태스크", ", ".join(f"{key}: {value}" for key, value in sorted(task_counts.items()))),
        ("추적당 검색 문서 수", ", ".join(map(str, doc_counts))),
    ]
    items = "\n".join(
        f'<div class="metric"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>' for label, value in cards
    )
    return f'<section class="metrics">{items}</section>'


def render_build_section(records: list[dict], sample_rows: dict[str, dict], index_metadata: dict) -> str:
    counts = category_counts(sample_rows, records)
    count_text = ", ".join(f"{code}: {counts[code]}" for code in sorted(counts)) or "trace 기준 샘플"
    total_samples = len(sample_rows) if sample_rows else "trace subset"
    n_docs = sorted(set(len(record.get("retrieved_docs", [])) for record in records))
    n_docs_text = ", ".join(map(str, n_docs)) if n_docs else "CLI 설정값"
    question_shapes = sorted({tuple(record.get("question_hidden_shape", [])) for record in records})
    shape_text = ", ".join(str(list(shape)) for shape in question_shapes if shape) or "trace에 기록"
    passage_count = index_metadata.get("passage_count", "index metadata 기준")
    chunk_words = index_metadata.get("chunk_words", 100)
    ctx_encoder = index_metadata.get("ctx_encoder_name", "facebook/dpr-ctx_encoder-multiset-base")
    knowledge_tsv = index_metadata.get("knowledge_tsv", "data/processed/knowledge.tsv")
    passages_path = index_metadata.get("passages_path", "artifacts/index/patent_knowledge_dataset")
    index_path = index_metadata.get("index_path", "artifacts/index/patent_knowledge_hnsw_index.faiss")
    trace_count = len(records)
    model_counts = Counter(MODEL_LABELS.get(record["model_type"], record["model_type"]) for record in records)
    model_text = ", ".join(f"{model}: {count}" for model, count in sorted(model_counts.items()))
    token_steps = max((len(record.get("token_steps", [])) for record in records if record["model_type"] == "rag_token"), default=0)

    return f"""
    <section class="panel build-panel">
      <h2>RAG 구축 및 재현 절차</h2>
      <p>
        이 보고서의 핵심은 fine-tuning 성능 측정이 아니라, 논문의 RAG 구조를 HYU 미국 특허 데이터 위에서
        재현하고 <code>RAG-Sequence</code>와 <code>RAG-Token</code>의 retrieval 및 marginalization 흐름을 관찰하는 것입니다.
      </p>
      <div class="build-grid">
        <div>
          <h3>1. 데이터 준비</h3>
          <ul class="step-list">
            <li>원천 파일은 <code>test_data.csv</code>입니다.</li>
            <li><code>국가코드</code>를 공백 제거 및 대문자 정규화한 뒤 <code>US</code> 행만 유지했습니다.</li>
            <li><code>중분류</code>/<code>category_code</code> 기준 <code>AA</code>, <code>AB</code>, <code>AC</code>, <code>AD</code>를 각각 50개씩 <code>seed=42</code>로 샘플링했습니다.</li>
            <li>최종 샘플은 <strong>{esc(total_samples)}</strong>개이며 분포는 <strong>{esc(count_text)}</strong>입니다.</li>
            <li><code>출원번호</code>는 RAG 지식 문서와 태스크를 연결하는 ID로 사용했습니다.</li>
          </ul>
        </div>
        <div>
          <h3>2. Knowledge Source 구성</h3>
          <ul class="step-list">
            <li><code>title</code>은 <code>출원번호 | 발명의 명칭</code> 형식으로 만들었습니다.</li>
            <li><code>text</code>에는 영어 원문성이 높은 <code>Title</code>, <code>Abstract</code>, <code>Main Claim</code>, <code>Independent Claim</code>, <code>Main IPC</code>, <code>Main CPC</code>, <code>Applicant</code>, <code>Assignee</code>, <code>Legal Status</code>, <code>Category Code</code>를 넣었습니다.</li>
            <li><code>요약(원문)</code>은 US 행에서 값이 없어서 제외했고, 한국어 AI요약 컬럼은 retrieval text에 섞지 않았습니다.</li>
            <li>생성된 지식 파일은 <code>{esc(knowledge_tsv)}</code>입니다.</li>
          </ul>
        </div>
        <div>
          <h3>3. DPR/FAISS 인덱스 구축</h3>
          <ul class="step-list">
            <li>특허 문서를 약 <code>{esc(chunk_words)}</code> 단어 단위 passage로 나누어 총 <strong>{esc(passage_count)}</strong>개 passage를 만들었습니다.</li>
            <li>passage embedding은 <code>{esc(ctx_encoder)}</code>로 계산했습니다.</li>
            <li>검색용 dataset은 <code>{esc(passages_path)}</code>에 저장했습니다.</li>
            <li>FAISS HNSW index는 <code>{esc(index_path)}</code>에 저장했습니다.</li>
            <li>macOS에서 <code>torch</code>와 <code>faiss</code>를 같은 프로세스에 로드할 때 OpenMP 충돌이 발생할 수 있어, 현재 demo trace는 저장된 DPR embedding에 대한 exact dot-product 검색을 기본 backend로 사용합니다. FAISS index는 별도 구축 및 smoke test 대상입니다.</li>
          </ul>
        </div>
        <div>
          <h3>4. RAG Trace 실행</h3>
          <ul class="step-list">
            <li>모델은 <code>facebook/rag-sequence-nq</code>와 <code>facebook/rag-token-nq</code> 흐름을 각각 실행했습니다.</li>
            <li>이번 trace에는 <strong>{esc(trace_count)}</strong>개 레코드가 있으며 모델별 분포는 <strong>{esc(model_text)}</strong>입니다.</li>
            <li>각 query는 question encoder를 거쳐 hidden state <code>{esc(shape_text)}</code>를 만들고, 검색된 <code>n_docs={esc(n_docs_text)}</code>개의 문서에 대해 <code>doc_scores</code>와 softmax 확률을 기록했습니다.</li>
            <li>생성기 입력에는 query와 검색 passage를 결합한 context가 들어가며, 보고서에는 context preview와 retrieved doc snippets를 남겼습니다.</li>
          </ul>
        </div>
      </div>
      <div class="flow">
        <h3>RAG 흐름 관찰 포인트</h3>
        <ol class="flow-list">
          <li><strong>공통 retrieval:</strong> <code>x</code>를 question encoder로 임베딩하고, DPR passage embedding과의 유사도로 상위 문서를 찾은 뒤 <code>doc_scores</code>를 계산합니다.</li>
          <li><strong>RAG-Sequence:</strong> 전체 후보 문장 <code>y</code>에 대해 <code>logsumexp_z(log p_eta(z|x) + log p_theta(y|x,z))</code>를 비교합니다. 즉, 문서 선택을 sequence 단위로 주변화합니다.</li>
          <li><strong>RAG-Token:</strong> 매 decoder step마다 <code>logsumexp_z(log p_eta(z|x) + log p_theta(y_i|x,z,y_&lt;i))</code>를 계산합니다. 즉, 문서 기여도를 토큰 단위로 다시 주변화합니다.</li>
          <li><strong>현재 보고서의 토큰 추적:</strong> RAG-Token은 앞 <strong>{esc(token_steps)}</strong>개 step에 대해 top token과 문서별 기여도를 저장했습니다.</li>
        </ol>
      </div>
      <details class="commands">
        <summary>재현 명령</summary>
        <pre><code>python scripts/prepare_data.py --input-csv test_data.csv --output-dir data/processed --sample-per-class 50 --seed 42
PYTHONPATH=src uv run python scripts/build_index.py --knowledge-tsv data/processed/knowledge.tsv --output-dir artifacts/index --batch-size 8 --chunk-words 100 --max-length 256 --device auto
PYTHONPATH=src uv run python scripts/run_demo.py --passages-path artifacts/index/patent_knowledge_dataset --index-path artifacts/index/patent_knowledge_hnsw_index.faiss --tasks-path data/processed/demo_tasks.jsonl --output-dir artifacts/runs --retrieval-backend brute_force</code></pre>
      </details>
    </section>
    """


def render_kv_table(items: list[tuple[str, object]], *, value_limit: int = 420) -> str:
    rows = []
    for key, value in items:
        rows.append(f"<tr><th>{esc(key)}</th><td>{esc(compact_multiline(value, value_limit))}</td></tr>")
    return f'<div class="table-wrap kv-wrap"><table class="kv-table"><tbody>{"".join(rows)}</tbody></table></div>'


def render_sample_row_example(sample_rows: dict[str, dict]) -> str:
    row = next(iter(sample_rows.values()), {})
    if not row:
        return '<p class="muted">샘플 CSV를 찾지 못해 데이터 예시를 표시하지 못했습니다.</p>'
    items = [
        ("출원번호 / application_id", first_non_empty(row, "application_id", "출원번호")),
        ("국가코드 / country_code", first_non_empty(row, "country_code", "국가코드")),
        ("중분류 / category_code", first_non_empty(row, "category_code", "중분류")),
        ("소분류 / subcategory_code", first_non_empty(row, "subcategory_code", "소분류")),
        ("발명의 명칭", first_non_empty(row, "발명의 명칭")),
        ("요약", first_non_empty(row, "요약")),
        ("대표청구항(raw)", first_non_empty(row, "대표청구항")),
        ("대표청구항(cleaned)", clean_claim_for_report(row.get("대표청구항"))),
        ("메인 IPC", first_non_empty(row, "메인 IPC")),
        ("메인 CPC", first_non_empty(row, "main_cpc", "메인 CPC", "메인 CPC.1")),
        ("출원인", first_non_empty(row, "출원인정리", "출원인")),
        ("현재권리자", first_non_empty(row, "현재권리자정리", "현재권리자")),
        ("법적상태", first_non_empty(row, "법적상태")),
    ]
    return render_kv_table(items)


def render_knowledge_example(knowledge_example: dict, sample_rows: dict[str, dict]) -> str:
    row = next(iter(sample_rows.values()), {})
    title = clean_value(knowledge_example.get("title")) or first_non_empty(row, "knowledge_title")
    text = compact_multiline(knowledge_example.get("text") or row.get("knowledge_text"), 1500)
    if not title and not text:
        return '<p class="muted">knowledge.tsv 예시를 찾지 못했습니다.</p>'
    return (
        '<div class="code-example">'
        "<h4>knowledge.tsv 첫 행</h4>"
        f"<pre><code>title: {esc(title)}\n\ntext:\n{esc(text)}</code></pre>"
        "</div>"
    )


def render_task_examples(task_examples: dict[str, dict]) -> str:
    rows = []
    for task_type in ("classification", "qa", "summary"):
        task = task_examples.get(task_type)
        if not task:
            continue
        rows.append(
            "<tr>"
            f"<td><span class=\"badge badge-task\">{esc(TASK_LABELS.get(task_type, task_type))}</span></td>"
            f"<td>{esc(task.get('task_id'))}</td>"
            f"<td>{esc(compact_multiline(task.get('query'), 520))}</td>"
            f"<td>{esc(compact_multiline(task.get('target'), 260))}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="muted">demo_tasks.jsonl 예시를 찾지 못했습니다.</p>'
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>태스크</th><th>task_id</th><th>query 예시</th><th>target 예시</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_trace_data_example(records: list[dict]) -> str:
    sequence_record = next((record for record in records if record["model_type"] == "rag_sequence"), None)
    token_record = next((record for record in records if record["model_type"] == "rag_token"), None)
    if sequence_record is None:
        return '<p class="muted">trace 예시를 찾지 못했습니다.</p>'

    doc_rows = []
    for doc in sequence_record.get("retrieved_docs", [])[:3]:
        doc_rows.append(
            "<tr>"
            f"<td>{esc(doc.get('rank'))}</td>"
            f"<td>{fmt_float(doc.get('score'))}</td>"
            f"<td>{fmt_float(doc.get('probability'))}</td>"
            f"<td>{esc(compact(doc.get('title'), 180))}</td>"
            "</tr>"
        )
    context_preview = sequence_record.get("context_previews", [""])[0] if sequence_record.get("context_previews") else ""
    task = sequence_record["task"]
    trace_intro = render_kv_table(
        [
            ("trace 모델", MODEL_LABELS.get(sequence_record["model_type"], sequence_record["model_type"])),
            ("task_id", task.get("task_id")),
            ("question_hidden_shape", sequence_record.get("question_hidden_shape")),
            ("doc_scores", sequence_record.get("doc_scores")),
            ("doc_probabilities", sequence_record.get("doc_probabilities")),
            ("generated_text", sequence_record.get("generated_text")),
        ],
        value_limit=520,
    )

    token_html = ""
    if token_record and token_record.get("token_steps"):
        step = token_record["token_steps"][0]
        contribution_rows = []
        for contribution in step.get("document_contributions", []):
            contribution_rows.append(
                "<tr>"
                f"<td>{esc(contribution.get('rank'))}</td>"
                f"<td>{fmt_float(contribution.get('probability'))}</td>"
                f"<td>{esc(compact(contribution.get('title'), 220))}</td>"
                "</tr>"
            )
        token_html = (
            '<div class="trace-subexample">'
            "<h4>RAG-Token 첫 decoder step 문서 기여도</h4>"
            f'<p class="muted">생성 토큰: <code>{esc(step.get("generated_token"))}</code>, token_id={esc(step.get("generated_token_id"))}</p>'
            '<div class="table-wrap compact"><table>'
            "<thead><tr><th>문서 rank</th><th>토큰 기여 확률</th><th>문서 제목</th></tr></thead>"
            f"<tbody>{''.join(contribution_rows)}</tbody></table></div>"
            "</div>"
        )

    return (
        '<div class="trace-example">'
        "<h4>RAG-Sequence trace JSONL 예시</h4>"
        f"{trace_intro}"
        '<div class="table-wrap compact"><table>'
        "<thead><tr><th>rank</th><th>doc_score</th><th>softmax 확률</th><th>검색 문서</th></tr></thead>"
        f"<tbody>{''.join(doc_rows)}</tbody></table></div>"
        '<div class="code-example">'
        "<h4>context preview</h4>"
        f"<pre><code>{esc(compact_multiline(context_preview, 1200))}</code></pre>"
        "</div>"
        f"{token_html}"
        "</div>"
    )


def render_result_box(items: list[tuple[str, object]], *, value_limit: int = 520, extra_html: str = "") -> str:
    return (
        '<div class="code-result">'
        "<h4>이 코드로 생성/확인된 결과</h4>"
        f"{render_kv_table(items, value_limit=value_limit)}"
        f"{extra_html}"
        "</div>"
    )


def render_candidates_result(record: dict | None) -> str:
    if not record:
        return ""
    rows = []
    for candidate in record.get("candidates", [])[:3]:
        rows.append(
            "<tr>"
            f"<td>{esc(candidate.get('candidate_index'))}</td>"
            f"<td>{fmt_float(candidate.get('sequence_nll'))}</td>"
            f"<td>{fmt_float(candidate.get('sequence_score'))}</td>"
            f"<td>{esc(compact(candidate.get('text'), 180))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="table-wrap compact result-table"><table>'
        "<thead><tr><th>#</th><th>sequence NLL</th><th>score</th><th>후보 텍스트</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_token_step_result(record: dict | None) -> str:
    if not record or not record.get("token_steps"):
        return ""
    step = record["token_steps"][0]
    contribution_rows = []
    for contribution in step.get("document_contributions", []):
        contribution_rows.append(
            "<tr>"
            f"<td>{esc(contribution.get('rank'))}</td>"
            f"<td>{fmt_float(contribution.get('probability'))}</td>"
            f"<td>{esc(compact(contribution.get('title'), 220))}</td>"
            "</tr>"
        )
    token_rows = []
    for token in step.get("top_marginal_tokens", [])[:5]:
        token_rows.append(
            "<tr>"
            f"<td><code>{esc(token.get('token'))}</code></td>"
            f"<td>{esc(token.get('token_id'))}</td>"
            f"<td>{fmt_float(token.get('logprob'))}</td>"
            "</tr>"
        )
    contribution_table = (
        '<div class="table-wrap compact result-table"><table>'
        "<thead><tr><th>문서 rank</th><th>기여 확률</th><th>문서 제목</th></tr></thead>"
        f"<tbody>{''.join(contribution_rows)}</tbody></table></div>"
        if contribution_rows
        else ""
    )
    token_table = (
        '<div class="table-wrap compact result-table"><table>'
        "<thead><tr><th>상위 토큰</th><th>token_id</th><th>marginal logprob</th></tr></thead>"
        f"<tbody>{''.join(token_rows)}</tbody></table></div>"
        if token_rows
        else ""
    )
    return contribution_table + token_table


def render_prepared_outputs_result(first_row: dict, knowledge_example: dict, task_examples: dict[str, dict], sample_count: int) -> str:
    output_counts = render_kv_table(
        [
            ("처리 샘플 CSV", f"{sample_count} rows, 첫 행은 US / AA 샘플"),
            ("Knowledge TSV", f"{sample_count} rows, 각 행은 retriever가 검색할 title/text"),
            ("Demo task JSONL", f"{sample_count * 3} rows, 샘플당 classification/QA/summary 3개"),
        ],
        value_limit=620,
    )
    sample_preview = render_kv_table(
        [
            ("sample_index", first_non_empty(first_row, "sample_index")),
            ("application_id", first_non_empty(first_row, "application_id", "출원번호")),
            ("country_code", first_non_empty(first_row, "country_code", "국가코드")),
            ("category_code", first_non_empty(first_row, "category_code", "중분류")),
            ("subcategory_code", first_non_empty(first_row, "subcategory_code", "소분류")),
            ("title", first_non_empty(first_row, "발명의 명칭")),
        ],
        value_limit=620,
    )
    knowledge_preview = render_kv_table(
        [
            ("title", knowledge_example.get("title") or first_row.get("knowledge_title")),
            ("text preview", knowledge_example.get("text") or first_row.get("knowledge_text")),
        ],
        value_limit=720,
    )

    task_rows = []
    for task_type in ("classification", "qa", "summary"):
        task = task_examples.get(task_type)
        if not task:
            continue
        task_rows.append(
            "<tr>"
            f"<td>{esc(TASK_LABELS.get(task_type, task_type))}</td>"
            f"<td>{esc(task.get('task_id'))}</td>"
            f"<td>{esc(compact_multiline(task.get('query'), 360))}</td>"
            f"<td>{esc(compact_multiline(task.get('target'), 180))}</td>"
            "</tr>"
        )
    task_preview = (
        '<div class="table-wrap compact result-table"><table>'
        "<thead><tr><th>태스크</th><th>task_id</th><th>query preview</th><th>target preview</th></tr></thead>"
        f"<tbody>{''.join(task_rows)}</tbody></table></div>"
        if task_rows
        else ""
    )

    return (
        '<div class="result-subblock">'
        "<h4>산출물 row 수</h4>"
        f"{output_counts}"
        "<h4>처리 샘플 CSV 첫 행</h4>"
        f"{sample_preview}"
        "<h4>Knowledge TSV 첫 행</h4>"
        f"{knowledge_preview}"
        "<h4>Demo task JSONL 예시</h4>"
        f"{task_preview}"
        "</div>"
    )


def build_code_results(
    records: list[dict],
    sample_rows: dict[str, dict],
    index_metadata: dict,
    knowledge_example: dict,
    task_examples: dict[str, dict],
    dataset_preview: dict,
    faiss_preview: dict,
) -> dict[str, str]:
    sequence_record = next((record for record in records if record["model_type"] == "rag_sequence"), None)
    token_record = next((record for record in records if record["model_type"] == "rag_token"), None)
    first_row = next(iter(sample_rows.values()), {})
    counts = category_counts(sample_rows, records)
    country_counts = Counter(
        clean_value(row.get("country_code") or row.get("국가코드")) for row in sample_rows.values()
    )
    country_counts.pop("", None)
    unique_app_ids = {
        clean_value(row.get("application_id") or row.get("출원번호")) for row in sample_rows.values()
    }
    unique_app_ids.discard("")
    task_counts = Counter(task.get("task_type") for task in task_examples.values())

    knowledge_text = knowledge_example.get("text") or first_row.get("knowledge_text", "")
    knowledge_labels = [
        line.split(":", 1)[0]
        for line in str(knowledge_text).splitlines()
        if ":" in line and line.split(":", 1)[0]
    ][:10]

    first_doc = sequence_record.get("retrieved_docs", [{}])[0] if sequence_record else {}
    second_doc = sequence_record.get("retrieved_docs", [{}, {}])[1] if sequence_record and len(sequence_record.get("retrieved_docs", [])) > 1 else {}
    token_steps = token_record.get("token_steps", []) if token_record else []

    results = {
        "US 필터링과 중분류별 샘플링": render_result_box(
            [
                ("최종 샘플 수", len(sample_rows)),
                ("국가코드 분포", ", ".join(f"{key}: {value}" for key, value in sorted(country_counts.items()))),
                ("중분류 분포", ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))),
                ("출원번호 중복 여부", "없음" if len(unique_app_ids) == len(sample_rows) else "중복 있음"),
                ("첫 샘플", first_non_empty(first_row, "application_id", "출원번호")),
            ],
            value_limit=620,
            extra_html=render_prepared_outputs_result(first_row, knowledge_example, task_examples, len(sample_rows)),
        ),
        "Knowledge text 생성": render_result_box(
            [
                ("knowledge title", knowledge_example.get("title") or first_row.get("knowledge_title")),
                ("text에 포함된 필드", ", ".join(knowledge_labels)),
                ("청구항 prefix 제거", "예" if "[청구항" not in str(knowledge_text) else "아니오"),
                ("한국어 AI요약 제외", "예" if "AI요약" not in str(knowledge_text) else "아니오"),
                ("첫 knowledge text 길이", f"{len(str(knowledge_text).split())} words"),
                ("표시된 task 타입", ", ".join(sorted(task_counts))),
                ("전체 task 수", f"{len(sample_rows) * 3}개, 샘플당 classification/QA/summary 3개"),
            ],
            value_limit=680,
        ),
        "Knowledge TSV를 passage로 분할": render_result_box(
            [
                ("chunk_words", index_metadata.get("chunk_words", 100)),
                ("passage_count", dataset_preview.get("num_rows") or index_metadata.get("passage_count")),
                ("샘플당 평균 passage", f"{float(index_metadata.get('passage_count', 0)) / len(sample_rows):.2f}" if sample_rows and index_metadata.get("passage_count") else ""),
                ("dataset feature", ", ".join(dataset_preview.get("features", [])) or dataset_preview.get("load_error", "")),
                ("첫 passage title", dataset_preview.get("title")),
                ("첫 passage text", dataset_preview.get("text")),
                ("첫 passage embedding", f"{dataset_preview.get('embedding_dim')}차원, 앞 8개 값 [{', '.join(dataset_preview.get('embedding_preview', []))}]"),
                ("embedding L2 norm", fmt_float(dataset_preview.get("embedding_norm"))),
                ("FAISS index 요약", f"{faiss_preview.get('index_type', '')}, ntotal={faiss_preview.get('ntotal', '')}, d={faiss_preview.get('dimension', '')}, metric={faiss_preview.get('metric', '')}".strip(", ")),
                ("FAISS 파일 크기", fmt_bytes(faiss_preview.get("file_size_bytes"))),
                ("DPR context encoder 출력", f"{index_metadata.get('ctx_encoder_name')} -> passage별 {dataset_preview.get('embedding_dim') or faiss_preview.get('dimension')}차원 embedding"),
            ],
            value_limit=760,
        ),
        "Question encoder, retrieval, doc_scores": render_result_box(
            [
                ("예시 task", sequence_record["task"]["task_id"] if sequence_record else ""),
                ("question_hidden_shape", sequence_record.get("question_hidden_shape") if sequence_record else ""),
                ("n_docs", len(sequence_record.get("retrieved_docs", [])) if sequence_record else ""),
                ("1위 문서", first_doc.get("title")),
                ("1위 doc_score / 확률", f"{fmt_float(first_doc.get('score'))} / {fmt_float(first_doc.get('probability'))}"),
                ("2위 문서", second_doc.get("title")),
                ("2위 doc_score / 확률", f"{fmt_float(second_doc.get('score'))} / {fmt_float(second_doc.get('probability'))}"),
            ],
            value_limit=760,
        ),
        "RAG-Sequence 후보 문장 점수화": render_result_box(
            [
                ("예시 task", sequence_record["task"]["task_id"] if sequence_record else ""),
                ("최종 generated_text", sequence_record.get("generated_text") if sequence_record else ""),
                ("후보 수", len(sequence_record.get("candidates", [])) if sequence_record else ""),
                ("선택 기준", "sequence_nll이 가장 낮은 후보를 최종 후보로 표시"),
            ],
            extra_html=render_candidates_result(sequence_record),
        ),
        "RAG-Token 토큰 단위 주변화": render_result_box(
            [
                ("예시 task", token_record["task"]["task_id"] if token_record else ""),
                ("최종 generated_text", token_record.get("generated_text") if token_record else ""),
                ("저장된 token step 수", len(token_steps)),
                ("첫 생성 토큰", token_steps[0].get("generated_token") if token_steps else ""),
                ("첫 token_id", token_steps[0].get("generated_token_id") if token_steps else ""),
                ("계산 의미", "각 step에서 문서별 token logprob와 doc logprob를 더한 뒤 logsumexp로 주변화"),
            ],
            value_limit=760,
            extra_html=render_token_step_result(token_record),
        ),
    }
    return results


def render_code_examples(code_examples: list[tuple[str, str, str]], code_results: dict[str, str]) -> str:
    if not code_examples:
        return '<p class="muted">코드 발췌를 찾지 못했습니다.</p>'
    cards = []
    for title, source, code in code_examples:
        cards.append(
            '<details class="code-card" open>'
            f"<summary>{esc(title)}<span>{esc(source)}</span></summary>"
            f"<pre><code>{esc(code)}</code></pre>"
            f"{code_results.get(title, '')}"
            "</details>"
        )
    return "".join(cards)


def render_implementation_detail_section(
    records: list[dict],
    sample_rows: dict[str, dict],
    index_metadata: dict,
    knowledge_example: dict,
    task_examples: dict[str, dict],
    code_examples: list[tuple[str, str, str]],
    dataset_preview: dict,
    faiss_preview: dict,
) -> str:
    code_results = build_code_results(
        records,
        sample_rows,
        index_metadata,
        knowledge_example,
        task_examples,
        dataset_preview,
        faiss_preview,
    )
    return (
        '<section class="panel detail-panel">'
        "<h2>구현 상세와 데이터 예시</h2>"
        "<p>아래 예시는 실제 생성된 파일과 구현 코드에서 가져온 것입니다. 원천 CSV 한 행이 knowledge 문서와 세 가지 demo task로 바뀌고, 이후 trace JSONL에 retrieval 점수와 생성 흐름이 기록되는 구조를 보여줍니다.</p>"
        '<div class="detail-section">'
        "<h3>1. 샘플 CSV 행 예시</h3>"
        "<p class=\"muted\">처리 결과 파일 <code>data/processed/test_us_200_seed42.csv</code>의 첫 샘플입니다. raw claim의 한국어 청구항 prefix는 knowledge text 생성 시 제거됩니다.</p>"
        f"{render_sample_row_example(sample_rows)}"
        "</div>"
        '<div class="detail-section">'
        "<h3>2. Knowledge 문서 예시</h3>"
        "<p class=\"muted\"><code>knowledge.tsv</code>는 RAG retriever가 검색하는 지식 원천입니다. 각 행은 하나의 특허 문서를 나타내고, 인덱스 구축 단계에서 100단어 내외 passage로 다시 분할됩니다.</p>"
        f"{render_knowledge_example(knowledge_example, sample_rows)}"
        "</div>"
        '<div class="detail-section">'
        "<h3>3. Demo task JSONL 예시</h3>"
        "<p class=\"muted\">동일한 특허에서 중분류 예측, QA, 요약/분석 태스크를 생성합니다. v1은 fine-tuning이 아니라 RAG 흐름 관찰용이므로 target은 정성 비교 기준입니다.</p>"
        f"{render_task_examples(task_examples)}"
        "</div>"
        '<div class="detail-section">'
        "<h3>4. Trace 데이터 예시</h3>"
        "<p class=\"muted\">모델 실행 후 <code>rag_trace.jsonl</code>에 저장되는 값입니다. question encoder hidden state, 검색 문서, doc score, context preview, token-level contribution을 확인할 수 있습니다.</p>"
        f"{render_trace_data_example(records)}"
        "</div>"
        '<div class="detail-section">'
        "<h3>5. 핵심 코드 발췌</h3>"
        "<p class=\"muted\">아래 코드는 보고서를 만든 실제 구현에서 가져왔습니다. 모든 코드 블록은 기본 펼침 상태이며, 각 코드 바로 아래에는 현재 산출물 기준으로 확인된 실행 결과를 붙였습니다.</p>"
        f"{render_code_examples(code_examples, code_results)}"
        "</div>"
        "</section>"
    )


def render_overview_table(records: list[dict], sample_rows: dict[str, dict]) -> str:
    rows = []
    for record in records:
        task = record["task"]
        top_doc = record.get("retrieved_docs", [{}])[0].get("title", "")
        model_label = MODEL_LABELS.get(record["model_type"], record["model_type"])
        task_label = TASK_LABELS.get(task["task_type"], task["task_type"])
        target_ko = korean_target(task, sample_rows.get(task["application_id"]))
        rows.append(
            "<tr>"
            f"<td><span class=\"badge badge-model\">{esc(model_label)}</span></td>"
            f"<td><span class=\"badge badge-task\">{esc(task_label)}</span></td>"
            f"<td>{esc(task['application_id'])}</td>"
            f"<td><div class=\"cell-lang\"><span>영문</span>{esc(compact(task.get('target'), 120))}</div>"
            f"<div class=\"cell-lang\"><span>한글</span>{esc(compact(target_ko, 120))}</div></td>"
            f"<td>{esc(compact(record.get('generated_text'), 140))}</td>"
            f"<td>{esc(compact(top_doc, 140))}</td>"
            "</tr>"
        )
    return (
        '<section class="panel">'
        "<h2>개요</h2>"
        '<div class="table-wrap"><table>'
        "<thead><tr><th>모델</th><th>태스크</th><th>출원번호</th><th>정답/참조</th><th>생성 결과</th><th>상위 검색 문서</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div></section>"
    )


def render_docs(record: dict) -> str:
    items = []
    for doc in record.get("retrieved_docs", []):
        items.append(
            '<li class="doc">'
            f'<div><strong>#{esc(doc.get("rank"))}</strong> {esc(doc.get("title"))}</div>'
            f'<div class="muted">문서 점수={fmt_float(doc.get("score"))} · 확률={fmt_float(doc.get("probability"))}</div>'
            f'<p>{esc(compact(doc.get("text_preview"), 520))}</p>'
            "</li>"
        )
    return f'<ol class="docs">{"".join(items)}</ol>'


def render_sequence(record: dict) -> str:
    candidates = []
    for candidate in record.get("candidates", [])[:5]:
        candidates.append(
            "<tr>"
            f"<td>{esc(candidate.get('candidate_index'))}</td>"
            f"<td>{fmt_float(candidate.get('sequence_nll'))}</td>"
            f"<td>{fmt_float(candidate.get('sequence_score'))}</td>"
            f"<td>{esc(candidate.get('text'))}</td>"
            "</tr>"
        )
    return (
        "<h4>RAG-Sequence 후보</h4>"
        '<div class="table-wrap compact"><table>'
        "<thead><tr><th>#</th><th>NLL</th><th>점수</th><th>텍스트</th></tr></thead>"
        f"<tbody>{''.join(candidates)}</tbody></table></div>"
    )


def render_token(record: dict) -> str:
    steps = []
    for step in record.get("token_steps", [])[:8]:
        contributions = sorted(
            step.get("document_contributions", []), key=lambda item: item.get("probability", 0), reverse=True
        )
        top_doc = contributions[0] if contributions else {}
        top_tokens = ", ".join(
            f"{token.get('token')!r} ({fmt_float(token.get('logprob'))})"
            for token in step.get("top_marginal_tokens", [])[:3]
        )
        steps.append(
            "<tr>"
            f"<td>{esc(step.get('position'))}</td>"
            f"<td><code>{esc(step.get('generated_token'))}</code></td>"
            f"<td>{esc(compact(top_doc.get('title'), 160))}</td>"
            f"<td>{fmt_float(top_doc.get('probability'))}</td>"
            f"<td>{esc(top_tokens)}</td>"
            "</tr>"
        )
    return (
        "<h4>RAG-Token 토큰 단계</h4>"
        '<div class="table-wrap compact"><table>'
        "<thead><tr><th>위치</th><th>생성 토큰</th><th>가장 기여한 문서</th><th>기여도</th><th>상위 주변화 토큰</th></tr></thead>"
        f"<tbody>{''.join(steps)}</tbody></table></div>"
    )


def render_record(record: dict, sample_rows: dict[str, dict]) -> str:
    task = record["task"]
    detail = render_sequence(record) if record["model_type"] == "rag_sequence" else render_token(record)
    model_label = MODEL_LABELS.get(record["model_type"], record["model_type"])
    task_label = TASK_LABELS.get(task["task_type"], task["task_type"])
    source_row = sample_rows.get(task["application_id"])
    query_ko = korean_query(task, source_row)
    target_ko = korean_target(task, source_row)
    return (
        '<article class="trace">'
        f"<h3>{esc(model_label)} · {esc(task_label)} · {esc(task['application_id'])}</h3>"
        '<div class="grid">'
        f'<div>{bilingual_block("질문/입력", task.get("query"), query_ko)}</div>'
        f'<div>{bilingual_block("정답/참조", task.get("target"), target_ko)}<h4>생성 결과</h4><p>{esc(record.get("generated_text"))}</p></div>'
        "</div>"
        "<h4>검색 문서</h4>"
        f"{render_docs(record)}"
        f"{detail}"
        "</article>"
    )


def render_html(
    records: list[dict],
    title: str,
    sample_rows: dict[str, dict] | None = None,
    index_metadata: dict | None = None,
    knowledge_example: dict | None = None,
    task_examples: dict[str, dict] | None = None,
    code_examples: list[tuple[str, str, str]] | None = None,
    dataset_preview: dict | None = None,
    faiss_preview: dict | None = None,
) -> str:
    sample_rows = sample_rows or {}
    index_metadata = index_metadata or {}
    knowledge_example = knowledge_example or {}
    task_examples = task_examples or {}
    code_examples = code_examples or []
    dataset_preview = dataset_preview or {}
    faiss_preview = faiss_preview or {}
    trace_sections = "\n".join(render_record(record, sample_rows) for record in records)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #0f766e;
      --accent-soft: #e6f4f1;
      --task-soft: #eef2ff;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    header {{
      padding: 36px 40px 20px;
      background: #172033;
      color: #fff;
    }}
    header p {{ max-width: 920px; color: #d6deea; margin: 8px 0 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 56px; }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; line-height: 1.25; }}
    h2 {{ font-size: 22px; }}
    h3 {{ font-size: 18px; }}
    h4 {{ font-size: 14px; color: #344054; }}
    p {{ margin: 0 0 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 24px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: var(--shadow); }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .metric strong {{ font-size: 18px; }}
    .panel, .trace {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: var(--shadow); }}
    .trace {{ padding: 22px; }}
    .build-panel p {{ max-width: 980px; }}
    .build-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 16px; }}
    .build-grid > div {{ border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: #fbfcff; }}
    .detail-panel > p {{ max-width: 980px; }}
    .detail-section {{ margin-top: 22px; }}
    .detail-section h3 {{ margin-bottom: 6px; }}
    .step-list, .flow-list {{ margin: 0; padding-left: 20px; }}
    .step-list li, .flow-list li {{ margin-bottom: 8px; }}
    .flow {{ margin-top: 18px; border-top: 1px solid var(--line); padding-top: 16px; }}
    .commands {{ margin-top: 16px; border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; background: #fbfcff; }}
    .commands summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
    pre {{ overflow-x: auto; margin: 12px 0 0; background: #101828; color: #e4e7ec; border-radius: 8px; padding: 14px; font-size: 12px; line-height: 1.5; }}
    pre code {{ background: transparent; border: 0; padding: 0; color: inherit; }}
    .code-example {{ margin-top: 10px; }}
    .code-card {{ border: 1px solid var(--line); border-radius: 8px; margin-top: 10px; background: #fbfcff; overflow: hidden; }}
    .code-card summary {{ cursor: pointer; padding: 12px 14px; font-weight: 700; color: #344054; }}
    .code-card summary span {{ display: block; margin-top: 3px; color: var(--muted); font-size: 12px; font-weight: 500; }}
    .code-card pre {{ border-radius: 0; margin: 0; }}
    .code-result {{ border-top: 1px solid var(--line); padding: 14px; background: #ffffff; }}
    .code-result h4 {{ margin-bottom: 10px; }}
    .code-result .kv-wrap {{ margin-bottom: 10px; }}
    .result-subblock h4 {{ margin-top: 14px; }}
    .result-subblock h4:first-child {{ margin-top: 0; }}
    .result-table {{ margin-top: 10px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ background: #f1f4f8; text-align: left; color: #344054; font-weight: 650; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    tr:last-child td {{ border-bottom: 0; }}
    .kv-table th {{ width: 210px; white-space: nowrap; }}
    .kv-table td {{ min-width: 360px; }}
    .trace-example {{ display: grid; gap: 12px; }}
    .trace-subexample {{ margin-top: 4px; }}
    .compact table {{ font-size: 12px; }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 650; white-space: nowrap; }}
    .badge-model {{ background: var(--accent-soft); color: var(--accent); }}
    .badge-task {{ background: var(--task-soft); color: #3730a3; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr); gap: 18px; margin: 12px 0 16px; }}
    .docs {{ padding-left: 22px; margin: 0 0 18px; }}
    .doc {{ margin-bottom: 12px; }}
    .doc p {{ color: #344054; margin-top: 4px; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .bilingual h4 {{ margin-bottom: 8px; }}
    .lang-block {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: #fbfcff; }}
    .lang-block span, .cell-lang span {{ display: inline-flex; font-size: 11px; font-weight: 700; color: var(--accent); margin-right: 6px; }}
    .lang-block p {{ margin: 4px 0 0; }}
    .cell-lang {{ margin-bottom: 6px; }}
    .cell-lang:last-child {{ margin-bottom: 0; }}
    code {{ background: #f1f4f8; border: 1px solid var(--line); border-radius: 5px; padding: 1px 5px; }}
    @media (max-width: 860px) {{
      header {{ padding: 28px 22px 18px; }}
      main {{ padding: 20px 14px 40px; }}
      .metrics {{ grid-template-columns: 1fr 1fr; }}
      .grid {{ grid-template-columns: 1fr; }}
      .build-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <p>HYU 미국 특허 custom knowledge index를 대상으로 RAG-Sequence와 RAG-Token의 검색, 문서 점수, 생성 흐름을 비교한 정성 분석 보고서입니다. fine-tuning 기반 정확도 보고서가 아니라 모델 흐름 관찰용입니다.</p>
  </header>
  <main>
    {render_summary(records)}
    {render_build_section(records, sample_rows, index_metadata)}
    {render_implementation_detail_section(records, sample_rows, index_metadata, knowledge_example, task_examples, code_examples, dataset_preview, faiss_preview)}
    {render_overview_table(records, sample_rows)}
    <section class="panel"><h2>추적 상세</h2><p class="muted">각 trace에는 영문/한글 질문·정답, 검색 문서, 문서 점수, 생성 결과, 모델별 marginalization 관찰값이 포함됩니다.</p></section>
    {trace_sections}
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HYU RAG trace JSONL을 standalone HTML 보고서로 렌더링합니다.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default="HYU 미국 특허 RAG 데모 보고서")
    parser.add_argument("--sample-csv", type=Path, default=Path("data/processed/test_us_200_seed42.csv"))
    parser.add_argument("--index-metadata", type=Path, default=Path("artifacts/index/index_metadata.json"))
    parser.add_argument("--knowledge-tsv", type=Path, default=Path("data/processed/knowledge.tsv"))
    parser.add_argument("--tasks-path", type=Path, default=Path("data/processed/demo_tasks.jsonl"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.trace)
    if not records:
        raise SystemExit(f"No records found in {args.trace}")
    sample_rows = load_sample_rows(args.sample_csv)
    index_metadata = read_optional_json(args.index_metadata)
    knowledge_example = read_first_tsv(args.knowledge_tsv)
    task_examples = load_task_examples(args.tasks_path)
    code_examples = load_code_examples(args.source_root)
    passages_path = Path(index_metadata["passages_path"]) if index_metadata.get("passages_path") else None
    index_path = Path(index_metadata["index_path"]) if index_metadata.get("index_path") else None
    dataset_preview = read_dataset_preview(passages_path)
    faiss_preview = read_faiss_preview(index_path)
    output = args.output or args.trace.with_name("report.html")
    output.write_text(
        render_html(
            records,
            args.title,
            sample_rows,
            index_metadata,
            knowledge_example,
            task_examples,
            code_examples,
            dataset_preview,
            faiss_preview,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
