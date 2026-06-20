"""Small exact retriever compatible with Hugging Face RAG models.

This avoids loading FAISS in the same process as torch on macOS, where OpenMP
runtime conflicts can abort the process. For this 1,791-passage demo index,
exact numpy dot-product retrieval is fast enough and still uses the DPR
embeddings built for the custom knowledge source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class BruteForceIndex:
    def __init__(self, dataset):
        self.dataset = dataset
        self.embeddings = np.asarray(dataset["embeddings"], dtype="float32")

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs: int = 5) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(question_hidden_states, dtype="float32")
        scores = query @ self.embeddings.T
        doc_ids = np.argsort(-scores, axis=1)[:, :n_docs].astype("int64")
        doc_embeds = np.stack([self.embeddings[ids] for ids in doc_ids]).astype("float32")
        return doc_ids, doc_embeds

    def get_doc_dicts(self, doc_ids: np.ndarray) -> list[dict]:
        doc_dicts = []
        for ids in doc_ids:
            rows = self.dataset[[int(i) for i in ids]]
            doc_dicts.append({"title": rows["title"], "text": rows["text"]})
        return doc_dicts

    def init_index(self) -> None:
        return None

    def is_initialized(self) -> bool:
        return True


class BruteForcePatentRetriever:
    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index: BruteForceIndex):
        self.config = config
        self.question_encoder_tokenizer = question_encoder_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.index = index
        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size

    @classmethod
    def from_pretrained(cls, model_name: str, passages_path: str | Path):
        from datasets import load_from_disk
        from transformers import RagConfig, RagTokenizer

        config = RagConfig.from_pretrained(model_name)
        rag_tokenizer = RagTokenizer.from_pretrained(model_name, config=config)
        dataset = load_from_disk(str(passages_path))
        index = BruteForceIndex(dataset)
        return cls(
            config=config,
            question_encoder_tokenizer=rag_tokenizer.question_encoder,
            generator_tokenizer=rag_tokenizer.generator,
            index=index,
        )

    def init_retrieval(self) -> None:
        return None

    def _postprocess_docs(self, docs: list[dict], input_strings: list[str], prefix: str | None, n_docs: int):
        prefix = prefix or ""
        rag_input_strings = []
        for batch_index in range(len(docs)):
            for doc_index in range(n_docs):
                title = str(docs[batch_index]["title"][doc_index]).removeprefix('"').removesuffix('"')
                text = str(docs[batch_index]["text"][doc_index])
                rag_input_strings.append(
                    (prefix + title + self.config.title_sep + text + self.config.doc_sep + input_strings[batch_index])
                    .replace("  ", " ")
                    .strip()
                )
        encoded = self.generator_tokenizer(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def __call__(
        self,
        question_input_ids,
        question_hidden_states,
        *,
        prefix: str | None,
        n_docs: int,
        return_tensors: str | None = "pt",
    ) -> dict:
        import torch

        if hasattr(question_input_ids, "detach"):
            question_ids_for_decode = question_input_ids.detach().cpu()
        else:
            question_ids_for_decode = question_input_ids

        question_hidden_states = np.asarray(question_hidden_states, dtype="float32")
        doc_ids, retrieved_doc_embeds = self.index.get_top_docs(question_hidden_states, n_docs=n_docs)
        docs = self.index.get_doc_dicts(doc_ids)
        input_strings = self.question_encoder_tokenizer.batch_decode(
            question_ids_for_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        context_input_ids, context_attention_mask = self._postprocess_docs(docs, input_strings, prefix, n_docs)
        if return_tensors != "pt":
            raise ValueError("BruteForcePatentRetriever currently supports return_tensors='pt' only")
        return {
            "context_input_ids": context_input_ids,
            "context_attention_mask": context_attention_mask,
            "retrieved_doc_embeds": torch.tensor(retrieved_doc_embeds, dtype=torch.float32),
            "doc_ids": torch.tensor(doc_ids, dtype=torch.long),
        }
