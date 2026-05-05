import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
        self.texts: List[str] = []

    def add(self, embeddings: np.ndarray, ids: Sequence[str], texts: Sequence[str]) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings with shape [n, {self.dim}], got {embeddings.shape}")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.ids.extend(ids)
        self.texts.extend(texts)

    def search(self, query_embeddings: np.ndarray, top_k: int):
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[None, :]
        faiss.normalize_L2(query_embeddings)
        scores, indices = self.index.search(query_embeddings, top_k)
        results = []
        for row_scores, row_indices in zip(scores, indices):
            row = []
            for score, idx in zip(row_scores.tolist(), row_indices.tolist()):
                if idx < 0:
                    continue
                row.append(
                    {
                        "score": float(score),
                        "id": self.ids[idx],
                        "text": self.texts[idx],
                    }
                )
            results.append(row)
        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        faiss.write_index(self.index, index_path)
        metadata = [{"id": pid, "text": text} for pid, text in zip(self.ids, self.texts)]
        Path(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FaissRetriever":
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        dim = faiss.read_index(index_path).d
        retriever = cls(dim=dim)
        retriever.index = faiss.read_index(index_path)
        retriever.ids = [item["id"] for item in metadata]
        retriever.texts = [item["text"] for item in metadata]
        return retriever


def load_passages(path: str) -> List[dict]:
    passages = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            passages.append(json.loads(line))
    return passages


def sample_passages(
    passages: Sequence[dict],
    sample_size: Optional[int] = None,
    seed: int = 13,
) -> List[dict]:
    if sample_size is None or sample_size >= len(passages):
        return list(passages)
    rng = random.Random(seed)
    indices = list(range(len(passages)))
    rng.shuffle(indices)
    indices = sorted(indices[:sample_size])
    return [passages[idx] for idx in indices]


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
