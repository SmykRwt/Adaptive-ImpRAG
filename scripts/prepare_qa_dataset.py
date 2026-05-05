import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a QA dataset subset for Adaptive ImpRAG.")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--question-column", type=str, default=None)
    parser.add_argument("--answer-column", type=str, default=None)
    parser.add_argument("--dataset-format", type=str, default="auto", choices=["auto", "nq_open", "hotpotqa", "generic"])
    return parser.parse_args()


def choose_rows(rows: Iterable[dict], sample_size: Optional[int], seed: int) -> list[dict]:
    rows = list(rows)
    if sample_size is None or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    indices = sorted(indices[:sample_size])
    return [rows[idx] for idx in indices]


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        normalized = []
        for item in value:
            normalized.extend(normalize_answers(item))
        return [item for item in normalized if item]
    if isinstance(value, dict):
        if "text" in value:
            return normalize_answers(value["text"])
        if "answer" in value:
            return normalize_answers(value["answer"])
        if "value" in value:
            return normalize_answers(value["value"])
    return [str(value)]


def row_to_example(row: dict, args) -> Optional[dict]:
    fmt = args.dataset_format
    if fmt == "auto":
        if "question" in row and "answer" in row:
            fmt = "nq_open"
        elif "question" in row and "answer" in row and isinstance(row.get("answer"), str):
            fmt = "hotpotqa"
        else:
            fmt = "generic"

    if fmt == "nq_open":
        query = row["question"]
        answers = normalize_answers(row["answer"])
    elif fmt == "hotpotqa":
        query = row["question"]
        answers = normalize_answers(row["answer"])
    else:
        question_col = args.question_column or "question"
        answer_col = args.answer_column or "answer"
        query = row.get(question_col)
        answers = normalize_answers(row.get(answer_col))

    if not query or not answers:
        return None
    return {
        "query": str(query).strip(),
        "answer": answers[0],
        "answers": answers,
    }


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    rows = choose_rows(dataset, args.sample_size, args.seed)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            example = row_to_example(row, args)
            if example is None:
                continue
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({"output_path": str(output_path), "examples_written": written}, indent=2))


if __name__ == "__main__":
    main()
