# HYU US Patent RAG Demo

HYU 미국 특허 데이터를 사용해 Lewis et al.의 Retrieval-Augmented Generation(RAG) 구조를 실행해보는 데모 프로젝트입니다. 목표는 fine-tuning이나 정확도 최적화가 아니라, custom knowledge source 위에서 retrieval, `doc_scores`, context 구성, `RAG-Sequence`와 `RAG-Token`의 marginalization 흐름을 관찰하는 것입니다.

## 주요 기능

- `test_data.csv`에서 `국가코드 == "US"` 행만 사용
- `중분류` 기준 `AA/AB/AC/AD` 각 50개, 총 200개 샘플링
- 특허 제목, 요약, 청구항, IPC/CPC, 출원인, 권리자, 법적상태를 이용한 `knowledge.tsv` 생성
- DPR context encoder 기반 passage embedding 생성
- FAISS HNSW index 생성
- Hugging Face RAG checkpoint 기반 `RAG-Sequence` / `RAG-Token` trace 실행
- trace JSONL, Markdown report, HTML report 생성

## 공개 저장소 정책

원본 특허 CSV, 생성 데이터, FAISS index, run artifact, vendor clone, 논문 PDF는 공개 저장소에 커밋하지 않습니다.

로컬에서 전체 데모를 실행하려면 repo root에 `test_data.csv`를 직접 넣어야 합니다. 테스트는 공개 가능한 작은 fixture인 `tests/fixtures/patents_minimal.csv`를 사용합니다.

## 프로젝트 구조

```text
.
├── scripts/
│   ├── prepare_data.py
│   ├── build_index.py
│   ├── run_demo.py
│   └── render_report_html.py
├── src/hyu_rag/
│   ├── data.py
│   ├── indexing.py
│   ├── retriever.py
│   └── tracing.py
├── tests/
│   ├── fixtures/patents_minimal.csv
│   └── test_data_prep.py
├── pyproject.toml
└── requirements-rag.txt
```

## 설치

Python 3.10-3.12 환경을 사용합니다. macOS에서는 `uv` 사용을 권장합니다.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-rag.txt
```

또는 매 명령을 `uv run`으로 실행할 수 있습니다.

## 1. 데이터 준비

repo root에 `test_data.csv`를 둔 뒤 실행합니다.

```bash
PYTHONPATH=src uv run python scripts/prepare_data.py \
  --input-csv test_data.csv \
  --output-dir data/processed \
  --sample-per-class 50 \
  --seed 42
```

생성 산출물:

- `data/processed/test_us_200_seed42.csv`
- `data/processed/knowledge.tsv`
- `data/processed/demo_tasks.jsonl`

처리 방식:

- `국가코드`를 공백 제거 및 대문자 정규화 후 `US`만 유지
- `출원번호` 중복 제거
- `중분류` 기준 `AA/AB/AC/AD` 각 50개 샘플링
- `메인 CPC` 중복 헤더는 pandas suffix 컬럼까지 읽고 비어 있지 않은 첫 값을 사용
- `[청구항1]` 같은 한국어 청구항 prefix 제거
- 한국어 AI요약 컬럼은 retrieval text에 섞지 않음

## 2. DPR/FAISS index 생성

```bash
PYTHONPATH=src uv run python scripts/build_index.py \
  --knowledge-tsv data/processed/knowledge.tsv \
  --output-dir artifacts/index \
  --batch-size 8 \
  --chunk-words 100 \
  --max-length 256 \
  --device auto
```

생성 산출물:

- `artifacts/index/patent_knowledge_dataset`
- `artifacts/index/patent_knowledge_hnsw_index.faiss`
- `artifacts/index/index_metadata.json`

`knowledge.tsv`의 각 특허 문서는 약 100단어 단위 passage로 나뉘며, `facebook/dpr-ctx_encoder-multiset-base`로 768차원 embedding을 생성합니다.

## 3. RAG trace 실행

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/run_demo.py \
  --passages-path artifacts/index/patent_knowledge_dataset \
  --index-path artifacts/index/patent_knowledge_hnsw_index.faiss \
  --tasks-path data/processed/demo_tasks.jsonl \
  --output-dir artifacts/runs \
  --n-docs 2 \
  --num-beams 1 \
  --max-length 16 \
  --trace-max-steps 4 \
  --examples-per-task 4 \
  --device auto \
  --retrieval-backend brute_force
```

기본 `brute_force` backend는 저장된 DPR embedding을 exact dot-product로 검색합니다. macOS에서 `torch`와 `faiss`를 같은 프로세스에 로드할 때 발생할 수 있는 OpenMP 충돌을 피하기 위한 설정입니다. FAISS index는 별도로 생성되고 smoke test 대상으로 유지됩니다.

생성 산출물:

- `artifacts/runs/<timestamp>/rag_trace.jsonl`
- `artifacts/runs/<timestamp>/report.md`

## 4. HTML report 생성

```bash
uv run python scripts/render_report_html.py \
  --trace artifacts/runs/<timestamp>/rag_trace.jsonl \
  --sample-csv data/processed/test_us_200_seed42.csv \
  --index-metadata artifacts/index/index_metadata.json \
  --knowledge-tsv data/processed/knowledge.tsv \
  --tasks-path data/processed/demo_tasks.jsonl \
  --source-root . \
  --output artifacts/runs/<timestamp>/report.html
```

HTML report에는 다음 정보가 포함됩니다.

- RAG 구축 및 재현 절차
- 데이터 샘플, `knowledge.tsv`, task JSONL 예시
- 핵심 코드 발췌와 해당 코드로 생성된 결과
- retrieved documents, `doc_scores`, softmax 확률
- `RAG-Sequence` 후보별 sequence NLL
- `RAG-Token` decoder step별 문서 기여도와 top marginal tokens
- 질문/입력 및 정답/참조의 영문/한글 표시

## RAG 흐름 요약

공통 retrieval 단계:

```text
query x
 -> question encoder
 -> question hidden state
 -> DPR passage embedding 검색
 -> top-k documents
 -> doc_scores = q · d
 -> context = query + retrieved passage
```

`RAG-Sequence`는 전체 후보 문장 단위로 문서를 주변화합니다.

```text
log p(y|x) = logsumexp_z(
  log p_eta(z|x) + log p_theta(y|x,z)
)
```

`RAG-Token`은 decoder step마다 문서를 주변화합니다.

```text
log p(y_i|x,y_<i) = logsumexp_z(
  log p_eta(z|x) + log p_theta(y_i|x,z,y_<i)
)
```

## 테스트

```bash
uv run --with pytest pytest -q
python3 -m compileall -q scripts src tests
```

현재 테스트는 raw CSV 없이 실행되도록 `tests/fixtures/patents_minimal.csv`를 사용합니다.

## 참고 모델

- `facebook/rag-sequence-nq`
- `facebook/rag-token-nq`
- `facebook/dpr-ctx_encoder-multiset-base`

## 라이선스와 데이터 주의

이 repository에는 구현 코드와 sanitized test fixture만 포함합니다. 원본 특허 데이터의 사용, 배포, 공개 여부는 데이터 제공처의 정책을 따르세요.
