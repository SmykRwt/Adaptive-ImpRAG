import re
from typing import Iterable, Sequence


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def batch_exact_match(predictions: Sequence[str], golds: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    return sum(exact_match(p, g) for p, g in zip(predictions, golds)) / len(predictions)


def retrieval_recall_at_k(answer_strings: Iterable[str], passages: Sequence[str]) -> float:
    normalized_passages = [_normalize(p) for p in passages]
    for answer in answer_strings:
        normalized_answer = _normalize(answer)
        if normalized_answer and any(normalized_answer in passage for passage in normalized_passages):
            return 1.0
    return 0.0
