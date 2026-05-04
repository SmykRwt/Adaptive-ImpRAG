from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class ImpRAGExample:
    query: str
    answer: str
    candidate_passages: List[Dict[str, str]]
    positives: Optional[List[int]] = None
    hard_negatives: Optional[List[int]] = None
    task_type: str = "retrieval"


class JsonlImpRAGDataset(Dataset):
    def __init__(self, rows: Sequence[Dict]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        return self.rows[index]


class ImpRAGCollator:
    def __init__(self, tokenizer, max_query_length: int, max_passage_length: int):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise ValueError("This base implementation currently expects batch_size=1 for training with generation.")

        example = batch[0]
        query = example["query"]
        answer = example["answer"]
        candidate_passages = example["candidate_passages"]
        positives = example.get("positives", [])
        hard_negatives = example.get("hard_negatives", [])

        query_text = f"{query}\nAnswer:"
        answer_text = f" {answer}"
        full_text = query_text + answer_text

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        query_tokens = self.tokenizer(
            query_text,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        labels = full_tokens["input_ids"].clone()
        prefix_len = query_tokens["input_ids"].shape[1]
        labels[:, :prefix_len] = -100

        passage_texts = [item["text"] for item in candidate_passages]
        passage_tokens = self.tokenizer(
            passage_texts,
            truncation=True,
            max_length=self.max_passage_length,
            padding=True,
            return_tensors="pt",
        )

        positive_mask = torch.zeros(1, len(candidate_passages), dtype=torch.float32)
        negative_mask = torch.ones(1, len(candidate_passages), dtype=torch.float32)
        for idx in positives:
            positive_mask[0, idx] = 1.0
            negative_mask[0, idx] = 0.0

        hard_negative_mask = torch.zeros(1, len(candidate_passages), dtype=torch.float32)
        for idx in hard_negatives:
            hard_negative_mask[0, idx] = 1.0

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
            "candidate_input_ids": passage_tokens["input_ids"].unsqueeze(0),
            "candidate_attention_mask": passage_tokens["attention_mask"].unsqueeze(0),
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
            "hard_negative_mask": hard_negative_mask,
            "query": query,
            "answer": answer,
            "candidate_passages": candidate_passages,
        }
