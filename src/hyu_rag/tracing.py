"""Trace RAG-Sequence and RAG-Token retrieval/generation flows."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .constants import RAG_SEQUENCE_MODEL, RAG_TOKEN_MODEL


@dataclass(frozen=True)
class DemoConfig:
    passages_path: Path
    index_path: Path
    tasks_path: Path
    output_dir: Path
    n_docs: int = 5
    num_beams: int = 2
    max_length: int = 64
    trace_max_steps: int = 8
    examples_per_task: int = 4
    device: str = "auto"
    retrieval_backend: str = "brute_force"
    rag_sequence_model: str = RAG_SEQUENCE_MODEL
    rag_token_model: str = RAG_TOKEN_MODEL


def _require_rag_dependencies():
    try:
        import torch  # noqa: F401
        from transformers import RagRetriever, RagSequenceForGeneration, RagTokenForGeneration  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "RAG tracing dependencies are missing. Install them with: "
            "uv pip install -r requirements-rag.txt"
        ) from exc


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_demo_tasks(tasks: list[dict], *, examples_per_task: int) -> list[dict]:
    selected: list[dict] = []
    for task_type in ("classification", "qa", "summary"):
        selected.extend([task for task in tasks if task["task_type"] == task_type][:examples_per_task])
    return selected


def _device_name(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _load_model(model_type: str, model_name: str, config: DemoConfig):
    from transformers import RagSequenceForGeneration, RagTokenForGeneration

    if config.retrieval_backend == "brute_force":
        from .retriever import BruteForcePatentRetriever

        retriever = BruteForcePatentRetriever.from_pretrained(model_name, config.passages_path)
        model_class = RagSequenceForGeneration if model_type == "rag_sequence" else RagTokenForGeneration
        model = model_class.from_pretrained(model_name)
        model.rag.retriever = retriever
        return model
    elif config.retrieval_backend == "hf_faiss":
        from transformers import RagRetriever

        retriever = RagRetriever.from_pretrained(
            model_name,
            index_name="custom",
            passages_path=str(config.passages_path),
            index_path=str(config.index_path),
        )
    else:
        raise ValueError(f"Unsupported retrieval backend: {config.retrieval_backend}")
    model_class = RagSequenceForGeneration if model_type == "rag_sequence" else RagTokenForGeneration
    model = model_class.from_pretrained(model_name, retriever=retriever)
    return model


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _md_cell(value: object, *, limit: int | None = None) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|")
    text = " ".join(text.split())
    if limit is not None and len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def retrieve_context(model, query: str, *, n_docs: int, device: str) -> dict:
    import torch

    retriever = model.retriever
    tokenizer = retriever.question_encoder_tokenizer
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        question_hidden_states = model.question_encoder(input_ids, attention_mask=attention_mask)[0]
        retrieved = retriever(
            input_ids,
            question_hidden_states.detach().cpu().to(torch.float32).numpy(),
            prefix=model.generator.config.prefix,
            n_docs=n_docs,
            return_tensors="pt",
        )
        context_input_ids = retrieved["context_input_ids"].to(device)
        context_attention_mask = retrieved["context_attention_mask"].to(device)
        retrieved_doc_embeds = retrieved["retrieved_doc_embeds"].to(device)
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)

    doc_ids = retrieved["doc_ids"].detach().cpu().numpy()
    doc_dicts = retriever.index.get_doc_dicts(doc_ids)
    docs = []
    score_values = doc_scores[0].detach().cpu().tolist()
    probabilities = _softmax(score_values)
    for rank, (title, text, score, probability) in enumerate(
        zip(doc_dicts[0]["title"], doc_dicts[0]["text"], score_values, probabilities), start=1
    ):
        docs.append(
            {
                "rank": rank,
                "title": str(title),
                "score": score,
                "probability": probability,
                "text_preview": str(text).replace("\n", " ")[:500],
            }
        )

    context_previews = [
        _decode(retriever.generator_tokenizer, ids)
        for ids in context_input_ids[:n_docs].detach().cpu().tolist()
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_hidden_shape": list(question_hidden_states.shape),
        "context_input_ids": context_input_ids,
        "context_attention_mask": context_attention_mask,
        "doc_scores": doc_scores,
        "docs": docs,
        "context_previews": [preview[:700] for preview in context_previews],
    }


def trace_rag_sequence(model, task: dict, *, config: DemoConfig, device: str) -> dict:
    import torch

    retrieved = retrieve_context(model, task["query"], n_docs=config.n_docs, device=device)
    context_input_ids = retrieved["context_input_ids"]
    context_attention_mask = retrieved["context_attention_mask"]
    doc_scores = retrieved["doc_scores"]

    with torch.no_grad():
        generator_input_ids = context_input_ids[: config.n_docs]
        generator_attention_mask = context_attention_mask[: config.n_docs]
        output_sequences = model.generator.generate(
            generator_input_ids,
            attention_mask=generator_attention_mask,
            num_beams=config.num_beams,
            num_return_sequences=config.num_beams,
            min_length=1,
            max_length=config.max_length,
        )
        deduped = list({tuple(sequence.tolist()): sequence for sequence in output_sequences}.values())
        output_sequences = torch.stack(deduped)
        num_candidates = output_sequences.shape[0]
        individual_input_ids = generator_input_ids.repeat(num_candidates, 1)
        individual_attention_mask = generator_attention_mask.repeat(num_candidates, 1)
        individual_doc_scores = doc_scores[:1, :].repeat(num_candidates, 1)
        outputs = model(
            context_input_ids=individual_input_ids,
            context_attention_mask=individual_attention_mask,
            doc_scores=individual_doc_scores,
            labels=output_sequences,
            exclude_bos_score=True,
            n_docs=config.n_docs,
        )
        losses = outputs.loss.detach().cpu().tolist()

    tokenizer = model.retriever.generator_tokenizer
    candidates = []
    for index, (sequence, loss) in enumerate(zip(output_sequences.detach().cpu().tolist(), losses)):
        candidates.append(
            {
                "candidate_index": index,
                "sequence_nll": float(loss),
                "sequence_score": float(-loss),
                "text": _decode(tokenizer, sequence),
            }
        )
    candidates.sort(key=lambda item: item["sequence_nll"])
    return {
        "model_type": "rag_sequence",
        "task": task,
        "question_hidden_shape": retrieved["question_hidden_shape"],
        "retrieved_docs": retrieved["docs"],
        "context_previews": retrieved["context_previews"],
        "doc_scores": [doc["score"] for doc in retrieved["docs"]],
        "doc_probabilities": [doc["probability"] for doc in retrieved["docs"]],
        "candidates": candidates[: min(10, len(candidates))],
        "generated_text": candidates[0]["text"] if candidates else "",
    }


def trace_rag_token(model, task: dict, *, config: DemoConfig, device: str) -> dict:
    import torch

    retrieved = retrieve_context(model, task["query"], n_docs=config.n_docs, device=device)
    context_input_ids = retrieved["context_input_ids"]
    context_attention_mask = retrieved["context_attention_mask"]
    doc_scores = retrieved["doc_scores"]
    tokenizer = model.retriever.generator_tokenizer

    with torch.no_grad():
        generated = model.generate(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_beams=config.num_beams,
            min_length=1,
            max_length=config.max_length,
            n_docs=config.n_docs,
        )
        trace_len = min(config.trace_max_steps, generated.shape[1])
        decoder_input_ids = generated[:, :trace_len].to(device)
        raw_outputs = model(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            decoder_input_ids=decoder_input_ids,
            do_marginalize=False,
            n_docs=config.n_docs,
        )
        seq_logprobs = torch.nn.functional.log_softmax(raw_outputs.logits, dim=-1).view(
            1, config.n_docs, trace_len, raw_outputs.logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        token_steps = []
        for position in range(trace_len):
            token_id = int(generated[0, position].detach().cpu().item())
            contribution_logits = seq_logprobs[0, :, position, token_id] + doc_logprobs[0, :]
            contribution_probs = torch.softmax(contribution_logits, dim=0).detach().cpu().tolist()
            marginalized = torch.logsumexp(seq_logprobs[0, :, position, :] + doc_logprobs[0, :, None], dim=0)
            top_values, top_ids = torch.topk(marginalized, k=5)
            token_steps.append(
                {
                    "position": position,
                    "generated_token": tokenizer.decode([token_id], skip_special_tokens=False),
                    "generated_token_id": token_id,
                    "document_contributions": [
                        {
                            "rank": rank + 1,
                            "title": retrieved["docs"][rank]["title"],
                            "probability": float(probability),
                        }
                        for rank, probability in enumerate(contribution_probs)
                    ],
                    "top_marginal_tokens": [
                        {
                            "token": tokenizer.decode([int(tok_id)], skip_special_tokens=False),
                            "token_id": int(tok_id),
                            "logprob": float(value.detach().cpu().item()),
                        }
                        for value, tok_id in zip(top_values, top_ids)
                    ],
                }
            )

    return {
        "model_type": "rag_token",
        "task": task,
        "question_hidden_shape": retrieved["question_hidden_shape"],
        "retrieved_docs": retrieved["docs"],
        "context_previews": retrieved["context_previews"],
        "doc_scores": [doc["score"] for doc in retrieved["docs"]],
        "doc_probabilities": [doc["probability"] for doc in retrieved["docs"]],
        "token_steps": token_steps,
        "generated_text": _decode(tokenizer, generated[0].detach().cpu().tolist()),
    }


def render_report(results: list[dict]) -> str:
    lines = [
        "# HYU US Patent RAG Demo Report",
        "",
        "This report compares RAG-Sequence and RAG-Token traces over the same custom US patent knowledge index.",
        "",
        "| Model | Task | Application | Target | Generated | Top Retrieved Document |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        task = result["task"]
        top_doc = result["retrieved_docs"][0]["title"] if result["retrieved_docs"] else ""
        lines.append(
            f"| {_md_cell(result['model_type'])} | {_md_cell(task['task_type'])} | "
            f"{_md_cell(task['application_id'])} | {_md_cell(task.get('target', ''), limit=120)} | "
            f"{_md_cell(result['generated_text'], limit=140)} | {_md_cell(top_doc, limit=140)} |"
        )
    lines.extend(["", "## Trace Notes", ""])
    for result in results:
        task = result["task"]
        lines.append(f"### {result['model_type']} - {task['task_id']}")
        lines.append(f"- Query: {_md_cell(task['query'], limit=500)}")
        lines.append(f"- Target: {_md_cell(task.get('target', ''))}")
        lines.append(f"- Generated: {_md_cell(result['generated_text'])}")
        lines.append("- Retrieved documents:")
        for doc in result["retrieved_docs"]:
            lines.append(
                f"  - #{doc['rank']} score={doc['score']:.4f} prob={doc['probability']:.4f}: {_md_cell(doc['title'])}"
            )
        if result["model_type"] == "rag_sequence":
            lines.append("- RAG-Sequence candidates by sequence NLL:")
            for candidate in result["candidates"][:3]:
                lines.append(f"  - nll={candidate['sequence_nll']:.4f}: {_md_cell(candidate['text'])}")
        else:
            lines.append("- RAG-Token first token-step contributions:")
            for step in result["token_steps"][:3]:
                top_doc = max(step["document_contributions"], key=lambda item: item["probability"])
                lines.append(
                    f"  - pos={step['position']} token={step['generated_token']!r} "
                    f"top_doc_prob={top_doc['probability']:.4f} top_doc={_md_cell(top_doc['title'])}"
                )
        lines.append("")
    return "\n".join(lines)


def run_demo(config: DemoConfig) -> Path:
    _require_rag_dependencies()
    import torch

    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_name(torch, config.device)
    tasks = select_demo_tasks(read_jsonl(config.tasks_path), examples_per_task=config.examples_per_task)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = config.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    results: list[dict] = []
    sequence_model = _load_model("rag_sequence", config.rag_sequence_model, config).to(device)
    sequence_model.eval()
    for task in tasks:
        results.append(trace_rag_sequence(sequence_model, task, config=config, device=device))
    del sequence_model

    token_model = _load_model("rag_token", config.rag_token_model, config).to(device)
    token_model.eval()
    for task in tasks:
        results.append(trace_rag_token(token_model, task, config=config, device=device))

    trace_path = run_dir / "rag_trace.jsonl"
    report_path = run_dir / "report.md"
    write_jsonl(trace_path, results)
    report_path.write_text(render_report(results), encoding="utf-8")
    return run_dir
